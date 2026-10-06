import tempfile
import unittest
from pathlib import Path

from voicerdr.config import AppConfig
from voicerdr.daemon import Daemon
from voicerdr.dictation import DictationBuffer
from voicerdr.paths import RuntimePaths
from voicerdr.voice import VoiceListener


class _Speaker:
    def __init__(self) -> None:
        self.interrupt_calls: list[bool] = []

    def interrupt(self, *, clear_queue: bool = True) -> int:
        self.interrupt_calls.append(clear_queue)
        return 2


class DaemonAudioIntegrationTests(unittest.TestCase):
    def test_speech_start_preempts_current_and_queued_tts(self) -> None:
        daemon = Daemon.__new__(Daemon)
        daemon.speaker = _Speaker()

        daemon._handle_speech_start()

        self.assertEqual(daemon.speaker.interrupt_calls, [True])

    def test_status_exposes_timing_without_duplicating_transcript(self) -> None:
        daemon = Daemon.__new__(Daemon)
        listener = VoiceListener(on_transcript=lambda _text: None)
        token = listener._speech_started(timestamp=10.0)
        listener._speech_stopped(timestamp=11.0)
        listener._transcript_received("private spoken text", utterance_id=token)
        daemon.voice = listener

        timing = daemon._stt_timing_status()

        self.assertIsNotNone(timing)
        assert timing is not None
        self.assertNotIn("transcript", timing)
        self.assertEqual(timing["sequence"], 1)

    def test_partial_transcript_exposes_mode_and_wait_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            daemon = Daemon.__new__(Daemon)
            daemon.paths = RuntimePaths(
                plugin_root=root,
                config_dir=root / "config",
                state_dir=root / "state",
                herdr_bin="herdr",
                herdr_socket=None,
            )
            daemon.config = AppConfig(wake_phrases=["secretary"])
            daemon.dictation = DictationBuffer()

            daemon._handle_partial_transcript(
                "Secretary listen, tell nine this is a longer message"
            )

            state = daemon._activity_status_snapshot()
            self.assertEqual(state["phase"], "listening")
            self.assertEqual(state["mode"], "awaiting_plan")
            self.assertTrue(state["capture_active"])
            self.assertIn("LLM interpretation", state["waiting_for"])
            self.assertIn("longer message", state["live_transcript"])


if __name__ == "__main__":
    unittest.main()
