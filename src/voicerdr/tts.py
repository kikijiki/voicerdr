import logging
import queue
import threading
from typing import Any, Self

log = logging.getLogger(__name__)


class _SpeechItem(str):
    """Queue item tagged with the preemption generation it belongs to."""

    generation: int

    def __new__(cls, text: str, generation: int) -> Self:
        item = super().__new__(cls, text)
        item.generation = generation
        return item


class Speaker:
    """Background Kokoro TTS → local speakers (PyAudio).

    Separate from the STT pipeline so announcements do not block the mic path.
    """

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        enabled: bool = True,
        speed: float = 1.0,
    ) -> None:
        self.voice = voice
        self.enabled = enabled
        self.speed = speed
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._kokoro: Any = None
        self._engine_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._sample_rate = 24000
        self._cancel = threading.Event()
        self._generation = 0
        self._active_generation: int | None = None
        self._stopping = False
        self.last_error: str | None = None
        self.last_spoken: str | None = None

    def start(self) -> None:
        with self._state_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping = False
        self._thread = threading.Thread(
            target=self._worker, name="voicerdr-tts", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._state_lock:
            self._stopping = True
            self._interrupt_locked(clear_queue=True)
            thread = self._thread
            if thread and thread.is_alive():
                self._q.put(None)
        if thread and thread.is_alive():
            thread.join(timeout=4)
        if thread and thread.is_alive():
            log.warning("TTS worker did not stop within timeout")
        else:
            with self._state_lock:
                self._thread = None
                self._stopping = False

    def interrupt(self, *, clear_queue: bool = True) -> int:
        """Preempt playback; optionally drop queued speech. Thread-safe."""
        with self._state_lock:
            return self._interrupt_locked(clear_queue=clear_queue)

    def _interrupt_locked(self, *, clear_queue: bool) -> int:
        self._generation += 1
        self._cancel.set()
        if not clear_queue:
            preserved: list[str | None] = []
            while True:
                try:
                    preserved.append(self._q.get_nowait())
                except queue.Empty:
                    break
            for item in preserved:
                if item is None:
                    self._q.put(None)
                else:
                    self._q.put(_SpeechItem(str(item), self._generation))
            return 0

        removed = 0
        stop_requested = False
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                stop_requested = True
            else:
                removed += 1
        if stop_requested:
            self._q.put(None)
        return removed

    def speak(self, text: str, *, replace: bool = False) -> None:
        text = " ".join((text or "").split()).strip()
        if not text or not self.enabled:
            return
        with self._state_lock:
            if self._stopping:
                return
            self._start_locked()
            if replace:
                self._interrupt_locked(clear_queue=True)
            self._q.put(_SpeechItem(text, self._generation))

    def _ensure_engine(self) -> None:
        if self._kokoro is not None:
            return
        with self._engine_lock:
            if self._kokoro is not None:
                return
            from pathlib import Path

            from kokoro_onnx import Kokoro
            from pipecat.services.kokoro.tts import (
                KOKORO_CACHE_DIR,
                _ensure_model_files,
            )

            model = Path(KOKORO_CACHE_DIR) / "kokoro-v1.0.onnx"
            voices = Path(KOKORO_CACHE_DIR) / "voices-v1.0.bin"
            _ensure_model_files(model, voices)
            self._kokoro = Kokoro(str(model), str(voices))
            log.info("kokoro TTS ready voice=%s", self.voice)

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            generation = getattr(item, "generation", self._generation)
            with self._state_lock:
                if generation != self._generation or self._stopping:
                    continue
                self._active_generation = generation
                self._cancel.clear()
            try:
                self._say(str(item))
                with self._state_lock:
                    if generation == self._generation and not self._cancel.is_set():
                        self.last_spoken = str(item)
                        self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                log.exception("TTS failed")
            finally:
                with self._state_lock:
                    if self._active_generation == generation:
                        self._active_generation = None

    def _say(self, text: str) -> None:
        self._ensure_engine()
        assert self._kokoro is not None
        samples, rate = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )
        self._sample_rate = int(rate)
        self._play(samples, int(rate))

    def _play(self, samples: Any, rate: int) -> None:
        import numpy as np
        import pyaudio

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        # Soft clip / convert to int16 for PortAudio.
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16).tobytes()

        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                output=True,
                frames_per_buffer=1024,
            )
            chunk = 1024 * 2  # bytes for int16 mono
            for i in range(0, len(pcm), chunk):
                with self._state_lock:
                    interrupted = self._cancel.is_set() or (
                        self._active_generation is not None
                        and self._active_generation != self._generation
                    )
                if interrupted:
                    break
                stream.write(pcm[i : i + chunk])
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    log.debug("failed to close TTS output stream", exc_info=True)
            pa.terminate()
