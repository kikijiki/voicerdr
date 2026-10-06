import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voicerdr.config import DEFAULT_VAD_STOP_SECS

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FinalTranscript:
    utterance_id: str
    text: str
    provenance_valid: bool = True
    provenance_error: str | None = None
    listener_generation: int | None = None
    origin: str = "voice"


OnTranscript = Callable[[FinalTranscript], None]
OnSpeechStart = Callable[[], None]
OnSpeechStop = Callable[[], None]
OnPartialTranscript = Callable[[str], None]
OnCaptureStart = Callable[[], None]


@dataclass(frozen=True, slots=True)
class STTObservation:
    """Timing data for one locally transcribed utterance.

    Event timestamps are Unix timestamps. Durations are calculated with a
    monotonic clock and therefore remain valid if the system clock changes.
    ``None`` means the associated VAD boundary was not observed.
    """

    sequence: int
    utterance_id: str
    transcript: str
    speech_started_at: float | None
    speech_stopped_at: float | None
    transcript_received_at: float
    utterance_seconds: float | None
    start_to_transcript_seconds: float | None
    stt_latency_seconds: float | None
    accepted: bool = True
    rejection_reason: str | None = None


@dataclass(slots=True)
class _UtteranceTiming:
    utterance_id: str
    started_at: float
    started_monotonic: float
    stopped_at: float | None = None
    stopped_monotonic: float | None = None


def _ensure_nltk_punkt() -> None:
    """Best-effort provision of tokenizer data used by Pipecat/Moonshine."""
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
            return
        except LookupError:
            pass
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        log.debug("nltk punkt ensure failed", exc_info=True)


def _resolve_moonshine_model(name: str) -> Any:
    from pipecat.services.moonshine.stt import Model as MoonshineModel

    key = name.strip().lower().replace("_", "-")
    for member in MoonshineModel:
        if member.value == key or member.name.lower().replace("_", "-") == key:
            return member
    return name


def _probe_input_device(device_index: int | None, sample_rate: int) -> int | None:
    """Return a usable input device index (or None for default). Raises on failure."""
    import pyaudio

    p = pyaudio.PyAudio()
    try:
        candidates: list[int | None] = [device_index]
        if device_index is not None:
            candidates.append(None)
        # Prefer pipewire/default when auto-selecting.
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if int(info.get("maxInputChannels") or 0) < 1:
                continue
            name = str(info.get("name") or "").lower()
            if name in ("pipewire", "default") or "pulse" in name:
                candidates.append(i)

        tried: set[int | None] = set()
        errors: list[str] = []
        for cand in candidates:
            if cand in tried:
                continue
            tried.add(cand)
            try:
                kwargs: dict[str, Any] = {
                    "format": pyaudio.paInt16,
                    "channels": 1,
                    "rate": sample_rate,
                    "input": True,
                    "frames_per_buffer": 1024,
                }
                if cand is not None:
                    kwargs["input_device_index"] = cand
                stream = p.open(**kwargs)
                stream.close()
                if cand != device_index:
                    log.info(
                        "audio probe: using device %s (requested %s)",
                        cand,
                        device_index,
                    )
                return cand
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{cand}: {exc}")
        raise RuntimeError(
            "no usable mic input device; tried "
            + "; ".join(errors[:6])
            + (" …" if len(errors) > 6 else "")
        )
    finally:
        p.terminate()


class VoiceListener:
    """Mic → VAD → Moonshine STT. TTS is handled by ``voicerdr.tts.Speaker``.

    Runs Pipecat on a dedicated thread/event loop so the daemon control
    socket stays responsive. Start/stop with listen/mute.
    """

    def __init__(
        self,
        *,
        on_transcript: OnTranscript,
        on_speech_start: OnSpeechStart | None = None,
        on_speech_stop: OnSpeechStop | None = None,
        on_partial_transcript: OnPartialTranscript | None = None,
        on_capture_start: OnCaptureStart | None = None,
        input_device_index: int | None = None,
        stt_model: str = "base",
        sample_rate: int = 16000,
        vad_stop_secs: float = DEFAULT_VAD_STOP_SECS,
        observation_limit: int = 100,
        listener_generation: int = 0,
    ) -> None:
        if observation_limit < 1:
            raise ValueError("observation_limit must be at least 1")
        if vad_stop_secs <= 0:
            raise ValueError("vad_stop_secs must be greater than zero")
        self.on_transcript = on_transcript
        self.on_speech_start = on_speech_start
        self.on_speech_stop = on_speech_stop
        self.on_partial_transcript = on_partial_transcript
        self.on_capture_start = on_capture_start
        self.input_device_index = input_device_index
        self.stt_model = stt_model
        self.sample_rate = sample_rate
        self.vad_stop_secs = vad_stop_secs
        self.listener_generation = listener_generation
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: Any = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._observation_lock = threading.Lock()
        self._active_utterance: _UtteranceTiming | None = None
        self._pending_utterances: deque[_UtteranceTiming] = deque()
        self._last_final_utterance_id: str | None = None
        self._final_source_ids: dict[str, str] = {}
        self._completed_utterance_ids: set[str] = set()
        self._session_id = uuid.uuid4().hex
        self._observation_sequence = 0
        self._vad_sequence = 0
        self._stt_observations: deque[STTObservation] = deque(maxlen=observation_limit)
        self._last_partial_transcript = ""
        self.running = False
        self.last_error: str | None = None

    @property
    def stt_observations(self) -> tuple[STTObservation, ...]:
        """Return a thread-safe snapshot of recent STT timing observations."""
        with self._observation_lock:
            return tuple(self._stt_observations)

    @property
    def last_stt_observation(self) -> STTObservation | None:
        """Return the most recent STT observation, if any."""
        with self._observation_lock:
            if not self._stt_observations:
                return None
            return self._stt_observations[-1]

    def clear_stt_observations(self) -> None:
        """Discard recorded observations without affecting the voice pipeline."""
        with self._observation_lock:
            self._stt_observations.clear()

    def _speech_started(
        self, *, timestamp: float | None = None, detection_delay: float = 0.0
    ) -> str:
        """Record a VAD speech-start and notify the consumer asynchronously."""
        delay = max(0.0, detection_delay)
        observed_at = time.time() if timestamp is None else timestamp
        with self._observation_lock:
            self._vad_sequence += 1
            timing = _UtteranceTiming(
                utterance_id=f"voice:{self._session_id}:{self._vad_sequence}",
                started_at=observed_at - delay,
                started_monotonic=time.monotonic() - delay,
            )
            self._active_utterance = timing
            self._last_partial_transcript = ""
            utterance_id = timing.utterance_id

        capture_callback = self.on_capture_start
        if capture_callback is not None:
            try:
                capture_callback()
            except Exception:
                log.exception("on_capture_start failed")

        callback = self.on_speech_start
        if callback is None:
            return utterance_id

        # The callback will commonly interrupt TTS. Keep arbitrary consumer
        # work and failures completely outside Pipecat's real-time event loop.
        threading.Thread(
            target=self._run_speech_start_callback,
            args=(callback,),
            name="voicerdr-speech-start",
            daemon=True,
        ).start()
        return utterance_id

    @staticmethod
    def _run_speech_start_callback(callback: OnSpeechStart) -> None:
        try:
            callback()
        except Exception:
            log.exception("on_speech_start failed")

    def _speech_stopped(
        self, *, timestamp: float | None = None, detection_delay: float = 0.0
    ) -> str | None:
        """Record the end boundary reported by VAD, if a start was observed."""
        delay = max(0.0, detection_delay)
        observed_at = time.time() if timestamp is None else timestamp
        with self._observation_lock:
            timing = self._active_utterance
            if timing is not None:
                timing.stopped_at = observed_at - delay
                timing.stopped_monotonic = time.monotonic() - delay
                self._pending_utterances.append(timing)
                self._active_utterance = None
            utterance_id = timing.utterance_id if timing is not None else None

        callback = self.on_speech_stop
        if callback is not None:
            try:
                callback()
            except Exception:
                log.exception("on_speech_stop failed")
        return utterance_id

    def _partial_received(self, text: str) -> None:
        """Forward a changed streaming hypothesis without treating it as final."""
        text = text.strip()
        if not text:
            return
        with self._observation_lock:
            if text == self._last_partial_transcript:
                return
            self._last_partial_transcript = text
        callback = self.on_partial_transcript
        if callback is not None:
            try:
                callback(text)
            except Exception:
                log.exception("on_partial_transcript failed")

    def _transcript_received(
        self,
        text: str,
        *,
        utterance_id: str | None,
        final_source_id: str | None = None,
    ) -> STTObservation:
        """Accept only a final bound to the exact VAD turn token."""
        received_at = time.time()
        received_monotonic = time.monotonic()
        with self._observation_lock:
            known_utterance_id = (
                self._final_source_ids.get(final_source_id)
                if final_source_id is not None
                else None
            )
            timing = None
            rejection_reason: str | None = None
            if not utterance_id:
                rejection_reason = "final frame has no VAD turn token"
            elif known_utterance_id is not None and known_utterance_id != utterance_id:
                rejection_reason = "final source identity conflicts with VAD turn token"
            elif utterance_id in self._completed_utterance_ids:
                pass
            else:
                pending_index = next(
                    (
                        index
                        for index, item in enumerate(self._pending_utterances)
                        if item.utterance_id == utterance_id
                    ),
                    None,
                )
                if pending_index is not None:
                    timing = self._pending_utterances[pending_index]
                    del self._pending_utterances[pending_index]
                elif (
                    self._active_utterance is not None
                    and self._active_utterance.utterance_id == utterance_id
                ):
                    timing = self._active_utterance
                    self._active_utterance = None
                else:
                    rejection_reason = (
                        "final frame VAD turn token is unknown or expired"
                    )
            accepted = rejection_reason is None
            if accepted and final_source_id is not None and known_utterance_id is None:
                self._final_source_ids[final_source_id] = str(utterance_id)
            if accepted:
                self._completed_utterance_ids.add(str(utterance_id))
                self._last_final_utterance_id = str(utterance_id)
            else:
                timing = None
            self._last_partial_transcript = ""
            self._observation_sequence += 1
            observed_utterance_id = (
                str(utterance_id)
                if utterance_id
                else f"voice:{self._session_id}:orphan:{self._observation_sequence}"
            )

            started_at = timing.started_at if timing else None
            stopped_at = timing.stopped_at if timing else None
            utterance_seconds = None
            start_to_transcript_seconds = None
            stt_latency_seconds = None
            if timing is not None:
                start_to_transcript_seconds = max(
                    0.0, received_monotonic - timing.started_monotonic
                )
                if timing.stopped_monotonic is not None:
                    utterance_seconds = max(
                        0.0, timing.stopped_monotonic - timing.started_monotonic
                    )
                    stt_latency_seconds = max(
                        0.0, received_monotonic - timing.stopped_monotonic
                    )

            observation = STTObservation(
                sequence=self._observation_sequence,
                utterance_id=observed_utterance_id,
                transcript=text,
                speech_started_at=started_at,
                speech_stopped_at=stopped_at,
                transcript_received_at=received_at,
                utterance_seconds=utterance_seconds,
                start_to_transcript_seconds=start_to_transcript_seconds,
                stt_latency_seconds=stt_latency_seconds,
                accepted=accepted,
                rejection_reason=rejection_reason,
            )
            self._stt_observations.append(observation)
            return observation

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self.last_error = None
        self._thread = threading.Thread(
            target=self._thread_main, name="voicerdr-voice", daemon=True
        )
        self._thread.start()

    def wait_until_ready(self, timeout: float = 45.0) -> bool:
        """Block until the pipeline is up, failed, or timed out."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._ready.is_set():
                return self.running and not self.last_error
            if self.last_error:
                return False
            thread = self._thread
            if thread is not None and not thread.is_alive() and not self.running:
                return False
            time.sleep(0.1)
        return self.running and not self.last_error

    @property
    def capture_eligible(self) -> bool:
        """Whether this listener thread can still open or retain audio input."""
        thread = self._thread
        return bool(thread and thread.is_alive())

    @property
    def in_callback_thread(self) -> bool:
        """Whether the caller is running on this listener's event-loop thread."""
        return self._thread is threading.current_thread()

    def stop(self) -> bool:
        self.request_stop()
        thread = self._thread

        # Transcripts are delivered on this event-loop thread. A spoken mute
        # command therefore cannot wait on the same loop or join itself.
        if self.in_callback_thread:
            return False

        if thread and thread.is_alive():
            thread.join(timeout=2)
        stopped = not self.capture_eligible
        if stopped:
            self.running = False
            self._ready.set()
            self._runner = None
            self._loop = None
            self._thread = None
        else:
            log.warning("voice listener did not stop within timeout")
        return stopped

    def request_stop(self) -> None:
        """Revoke listener work without waiting for model initialization."""
        self._stop.set()
        loop = self._loop
        runner = self._runner

        if loop and runner is not None and loop.is_running():

            async def _cancel() -> None:
                try:
                    await runner.cancel()
                except Exception:
                    log.exception("voice runner cancel failed")

            fut = asyncio.run_coroutine_threadsafe(_cancel(), loop)

            # Observe failures without making the control response wait for the
            # audio loop. Full stop/shutdown performs the bounded join.
            def _cancel_done(done: Any) -> None:
                try:
                    done.result()
                except Exception:
                    log.debug("voice cancel failed", exc_info=True)

            fut.add_done_callback(_cancel_done)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._amain())
        except Exception as exc:
            self.last_error = str(exc)
            log.exception("voice listener crashed")
        finally:
            self.running = False
            self._ready.set()
            self._runner = None
            self._loop = None

    async def _amain(self) -> None:
        # Import pipecat lazily so non-voice installs still import voicerdr.
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.frames.frames import (
            AudioRawFrame,
            CancelFrame,
            EndFrame,
            ErrorFrame,
            Frame,
            StartFrame,
            TranscriptionFrame,
            VADUserStartedSpeakingFrame,
            VADUserStoppedSpeakingFrame,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import (
            PipelineParams,
            PipelineWorker,
            ProcessorUnusablePolicy,
        )
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.services.moonshine.stt import MoonshineSTTService
        from pipecat.transports.local.audio import (
            LocalAudioTransport,
            LocalAudioTransportParams,
        )
        from pipecat.workers.runner import WorkerRunner

        if self._stop.is_set():
            return
        _ensure_nltk_punkt()
        if self._stop.is_set():
            return
        model = _resolve_moonshine_model(self.stt_model)
        if self._stop.is_set():
            return
        device = _probe_input_device(self.input_device_index, self.sample_rate)
        self.input_device_index = device
        if self._stop.is_set():
            return
        listener = self

        class TranscriptSink(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TranscriptionFrame):
                    text = frame.text or ""
                    if text.strip():
                        source_id = str(
                            frame.broadcast_sibling_id or frame.timestamp or frame.id
                        )
                        turn_token = frame.metadata.get("voicerdr_utterance_id")
                        observation = listener._transcript_received(
                            text,
                            utterance_id=(
                                str(turn_token) if isinstance(turn_token, str) else None
                            ),
                            final_source_id=source_id,
                        )
                        log.info("transcript: %s", text)
                        log.debug("STT timing: %s", observation)
                        try:
                            listener.on_transcript(
                                FinalTranscript(
                                    observation.utterance_id,
                                    text,
                                    provenance_valid=observation.accepted,
                                    provenance_error=observation.rejection_reason,
                                    listener_generation=listener.listener_generation,
                                )
                            )
                        except Exception:
                            log.exception("on_transcript failed")
                await self.push_frame(frame, direction)

        if self._stop.is_set():
            return
        transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=False,
                audio_in_sample_rate=self.sample_rate,
                audio_in_channels=1,
                input_device_index=device,
            )
        )
        vad = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(start_secs=0.2, stop_secs=self.vad_stop_secs)
            )
        )

        class TurnBoundMoonshineSTTService(MoonshineSTTService):
            """Carry the VAD turn token onto the final generated by Moonshine."""

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self._voicerdr_output_turn: str | None = None

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if isinstance(frame, VADUserStoppedSpeakingFrame):
                    token = frame.metadata.get("voicerdr_utterance_id")
                    self._voicerdr_output_turn = (
                        str(token) if isinstance(token, str) else None
                    )
                await super().process_frame(frame, direction)

            async def push_frame(
                self,
                frame: Frame,
                direction: FrameDirection = FrameDirection.DOWNSTREAM,
            ):
                if isinstance(frame, TranscriptionFrame):
                    token = self._voicerdr_output_turn
                    if token:
                        frame.metadata["voicerdr_utterance_id"] = token
                await super().push_frame(frame, direction)

        stt = TurnBoundMoonshineSTTService(
            settings=MoonshineSTTService.Settings(model=model),
        )
        if self._stop.is_set():
            return

        class LiveTranscriptTap(FrameProcessor):
            """Share Moonshine's loaded model to expose interim words during speech."""

            def __init__(self) -> None:
                super().__init__()
                self._stream: Any = None
                self._speaking = False
                self._lines: dict[int, str] = {}
                self._pre_roll = bytearray()

            def _on_event(self, event: Any) -> None:
                line = getattr(event, "line", None)
                text = str(getattr(line, "text", "") or "").strip()
                line_id = getattr(line, "line_id", None)
                if text and line_id is not None:
                    self._lines[int(line_id)] = text
                    combined = " ".join(self._lines[key] for key in sorted(self._lines))
                    listener._partial_received(combined)

            def _start_stream(self) -> None:
                self._close_stream()
                self._lines.clear()
                try:
                    self._stream = stt._transcriber.create_stream(update_interval=0.25)
                    self._stream.add_listener(self._on_event)
                    self._stream.start()
                    if self._pre_roll:
                        self._add_audio(bytes(self._pre_roll))
                except Exception:
                    log.exception("could not start live Moonshine transcript")
                    self._close_stream()

            def _add_audio(self, audio: bytes) -> None:
                if self._stream is None or not audio:
                    return
                import numpy as np

                samples = (
                    np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                ).tolist()
                try:
                    self._stream.add_audio(samples, listener.sample_rate)
                except Exception:
                    # The segmented STT remains the final-result fallback.
                    log.exception("live Moonshine update failed")
                    self._close_stream()

            def _close_stream(self, *, finish: bool = False) -> None:
                stream, self._stream = self._stream, None
                if stream is None:
                    return
                try:
                    if finish:
                        stream.stop()
                except Exception:
                    log.debug("live Moonshine stream stop failed", exc_info=True)
                try:
                    stream.close()
                except Exception:
                    log.debug("live Moonshine stream close failed", exc_info=True)

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, AudioRawFrame):
                    if self._speaking:
                        self._add_audio(frame.audio)
                    else:
                        self._pre_roll.extend(frame.audio)
                        max_bytes = listener.sample_rate * 2
                        if len(self._pre_roll) > max_bytes:
                            del self._pre_roll[:-max_bytes]
                elif isinstance(frame, VADUserStartedSpeakingFrame):
                    token = listener._speech_started(
                        timestamp=frame.timestamp,
                        detection_delay=frame.start_secs,
                    )
                    frame.metadata["voicerdr_utterance_id"] = token
                    self._speaking = True
                    self._start_stream()
                    self._pre_roll.clear()
                elif isinstance(frame, VADUserStoppedSpeakingFrame):
                    token = listener._speech_stopped(
                        timestamp=frame.timestamp,
                        detection_delay=frame.stop_secs,
                    )
                    if token:
                        frame.metadata["voicerdr_utterance_id"] = token
                    self._speaking = False
                    self._close_stream(finish=True)
                    self._pre_roll.clear()
                elif isinstance(frame, (EndFrame, CancelFrame)):
                    self._close_stream()
                await self.push_frame(frame, direction)

        live_transcript = LiveTranscriptTap()
        sink = TranscriptSink()
        if self._stop.is_set():
            return
        input_transport = transport.input()
        pipeline = Pipeline(
            [
                input_transport,
                vad,
                live_transcript,
                stt,
                sink,
            ]
        )
        worker = PipelineWorker(
            pipeline,
            name="voicerdr-listen",
            enable_rtvi=False,
            enable_turn_tracking=False,
            cancel_on_idle_timeout=False,
            idle_timeout_secs=None,
            processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
            params=PipelineParams(allow_interruptions=True),
        )

        def pipeline_failed(message: str) -> None:
            # Pipecat reports processor failures as frames, rather than raising
            # them from runner.run(). Publish failure before waking listen callers.
            if self.last_error is None:
                self.last_error = message
            self.running = False
            self._stop.set()
            self._ready.set()
            log.error("voice pipeline failed: %s", message)

        pipeline_started = False
        input_started = False

        def maybe_ready() -> None:
            if (
                not pipeline_started
                or not input_started
                or self._stop.is_set()
                or self.last_error
                or any(not processor.is_usable for processor in pipeline.processors)
            ):
                return
            self.running = True
            self._ready.set()
            log.info(
                "voice listener started device=%s model=%s",
                self.input_device_index,
                model,
            )

        @input_transport.event_handler("on_after_process_frame")
        async def on_input_frame(_transport: Any, frame: Frame) -> None:
            # The transport forwards StartFrame before starting its stream, so
            # the sink alone cannot establish that the microphone actually opened.
            nonlocal input_started
            if isinstance(frame, StartFrame):
                input_started = True
                maybe_ready()

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(_worker: Any, _frame: StartFrame) -> None:
            nonlocal pipeline_started
            pipeline_started = True
            maybe_ready()

        @worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(_worker: Any, frame: ErrorFrame) -> None:
            if (
                not self.running
                or frame.fatal
                or (frame.processor is not None and not frame.processor.is_usable)
            ):
                pipeline_failed(frame.error)

        @worker.event_handler("on_setup_timeout")
        async def on_setup_timeout(_worker: Any) -> None:
            pipeline_failed("voice pipeline setup timed out")

        @worker.event_handler("on_pipeline_timeout")
        async def on_pipeline_timeout(_worker: Any, frame: Frame) -> None:
            if isinstance(frame, StartFrame):
                pipeline_failed("voice pipeline startup timed out")

        runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
        self._runner = runner
        self._loop = asyncio.get_running_loop()
        if self._stop.is_set():
            return
        await runner.add_workers(worker)
        if self._stop.is_set():
            await runner.cancel()
            return
        run_task = asyncio.create_task(runner.run())
        while not self._stop.is_set() and not run_task.done():
            await asyncio.sleep(0.2)
        if not run_task.done():
            await runner.cancel()
            try:
                await asyncio.wait_for(run_task, timeout=5)
            except Exception:  # noqa: BLE001
                run_task.cancel()
        else:
            exc = run_task.exception()
            if exc:
                raise exc
            if not self._stop.is_set():
                raise RuntimeError("voice pipeline stopped unexpectedly")
