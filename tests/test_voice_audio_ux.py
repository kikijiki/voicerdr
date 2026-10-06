import asyncio
import threading
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from unittest.mock import MagicMock, patch

from voicerdr.config import DEFAULT_VAD_STOP_SECS
from voicerdr.voice import VoiceListener


class VoiceAudioUXTests(unittest.TestCase):
    def test_confirmed_550ms_thinking_pause_stays_inside_one_shot_boundary(
        self,
    ) -> None:
        # Live evidence at 09:48:32.153. Keep the exact prematurely finalized
        # transcript here so this threshold regression remains tied to the incident.
        transcript = (
            "Jenny tell nine that it should check the latest logs because sometimes "
            "the the the"
        )
        observed_pause_secs = 0.55
        listener = VoiceListener(on_transcript=lambda _final: None)

        self.assertEqual(listener.vad_stop_secs, DEFAULT_VAD_STOP_SECS)
        self.assertGreater(listener.vad_stop_secs, observed_pause_secs)
        self.assertEqual(
            transcript,
            "Jenny tell nine that it should check the latest logs because sometimes "
            "the the the",
        )

    def test_listener_accepts_configured_vad_stop_threshold(self) -> None:
        listener = VoiceListener(
            on_transcript=lambda _final: None,
            vad_stop_secs=1.8,
        )

        self.assertEqual(listener.vad_stop_secs, 1.8)

    def test_listener_rejects_nonpositive_vad_stop_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "vad_stop_secs"):
            VoiceListener(on_transcript=lambda _final: None, vad_stop_secs=0)

    def test_speech_start_callback_does_not_block_vad_path(self) -> None:
        callback_started = threading.Event()
        callback_release = threading.Event()

        def blocking_callback() -> None:
            callback_started.set()
            callback_release.wait(timeout=2)

        listener = VoiceListener(
            on_transcript=lambda _text: None,
            on_speech_start=blocking_callback,
        )
        before = time.monotonic()
        listener._speech_started()
        elapsed = time.monotonic() - before

        self.assertLess(elapsed, 0.1)
        self.assertTrue(callback_started.wait(timeout=1))
        callback_release.set()

    def test_failing_speech_start_callback_is_isolated(self) -> None:
        called = threading.Event()

        def failing_callback() -> None:
            called.set()
            raise RuntimeError("consumer failed")

        listener = VoiceListener(
            on_transcript=lambda _text: None,
            on_speech_start=failing_callback,
        )
        with patch("voicerdr.voice.log.exception") as exception_log:
            listener._speech_started()
            self.assertTrue(called.wait(timeout=1))
            # Wait for the callback thread to reach the exception handler.
            for _ in range(100):
                if exception_log.called:
                    break
                time.sleep(0.001)
        exception_log.assert_called_once_with("on_speech_start failed")

    def test_stt_observation_records_vad_adjusted_latency(self) -> None:
        listener = VoiceListener(on_transcript=lambda _text: None)
        with (
            patch("voicerdr.voice.time.time", side_effect=[102.0, 105.5, 106.0]),
            patch(
                "voicerdr.voice.time.monotonic",
                side_effect=[12.0, 15.5, 16.0],
            ),
        ):
            token = listener._speech_started(timestamp=102.0, detection_delay=0.2)
            listener._speech_stopped(timestamp=105.5, detection_delay=0.5)
            observation = listener._transcript_received(
                "hello locally", utterance_id=token
            )

        self.assertEqual(observation.sequence, 1)
        self.assertEqual(observation.transcript, "hello locally")
        self.assertAlmostEqual(observation.speech_started_at, 101.8)
        self.assertAlmostEqual(observation.speech_stopped_at, 105.0)
        self.assertAlmostEqual(observation.utterance_seconds, 3.2)
        self.assertAlmostEqual(observation.start_to_transcript_seconds, 4.2)
        self.assertAlmostEqual(observation.stt_latency_seconds, 1.0)
        self.assertIs(listener.last_stt_observation, observation)
        self.assertEqual(listener.stt_observations, (observation,))

    def test_observations_are_bounded_and_clearable(self) -> None:
        listener = VoiceListener(
            on_transcript=lambda _text: None,
            observation_limit=2,
        )
        for text in ("one", "two", "three"):
            token = listener._speech_started()
            listener._speech_stopped()
            listener._transcript_received(text, utterance_id=token)

        self.assertEqual(
            [item.transcript for item in listener.stt_observations],
            ["two", "three"],
        )
        listener.clear_stt_observations()
        self.assertEqual(listener.stt_observations, ())
        self.assertIsNone(listener.last_stt_observation)

    def test_partial_transcripts_are_forwarded_only_when_changed(self) -> None:
        partials: list[str] = []
        listener = VoiceListener(
            on_transcript=lambda _text: None,
            on_partial_transcript=partials.append,
        )

        listener._partial_received("Secretary")
        listener._partial_received("Secretary")
        listener._partial_received("Secretary listen")

        self.assertEqual(partials, ["Secretary", "Secretary listen"])

    def test_delayed_finals_keep_ordered_vad_turn_identity(self) -> None:
        listener = VoiceListener(on_transcript=lambda _final: None)
        first_token = listener._speech_started(timestamp=1.0)
        listener._speech_stopped(timestamp=2.0)
        second_token = listener._speech_started(timestamp=3.0)
        listener._speech_stopped(timestamp=4.0)

        first = listener._transcript_received(
            "same words", utterance_id=first_token, final_source_id="final-a"
        )
        duplicate_first = listener._transcript_received(
            "same words", utterance_id=first_token, final_source_id="final-a"
        )
        second = listener._transcript_received(
            "same words", utterance_id=second_token, final_source_id="final-b"
        )
        duplicate_second = listener._transcript_received(
            "same words", utterance_id=second_token, final_source_id="final-b"
        )

        self.assertNotEqual(first.utterance_id, second.utterance_id)
        self.assertEqual(first.utterance_id, duplicate_first.utterance_id)
        self.assertEqual(second.utterance_id, duplicate_second.utterance_id)

    def test_late_different_source_duplicate_cannot_steal_next_turn(self) -> None:
        listener = VoiceListener(on_transcript=lambda _final: None)
        first_token = listener._speech_started()
        listener._speech_stopped()
        first = listener._transcript_received(
            "same words", utterance_id=first_token, final_source_id="first-source"
        )
        second_token = listener._speech_started()
        listener._speech_stopped()

        late = listener._transcript_received(
            "same words", utterance_id=first_token, final_source_id="late-source"
        )
        second = listener._transcript_received(
            "same words", utterance_id=second_token, final_source_id="second-source"
        )

        self.assertTrue(late.accepted)
        self.assertEqual(late.utterance_id, first.utterance_id)
        self.assertTrue(second.accepted)
        self.assertEqual(second.utterance_id, second_token)
        self.assertNotEqual(first.utterance_id, second.utterance_id)

    def test_unknown_final_token_is_rejected_without_consuming_pending_turn(
        self,
    ) -> None:
        listener = VoiceListener(on_transcript=lambda _final: None)
        genuine_token = listener._speech_started()
        listener._speech_stopped()

        orphan = listener._transcript_received(
            "late words",
            utterance_id="voice:different:unknown",
            final_source_id="orphan-source",
        )
        genuine = listener._transcript_received(
            "genuine words",
            utterance_id=genuine_token,
            final_source_id="genuine-source",
        )

        self.assertFalse(orphan.accepted)
        self.assertIn("unknown", orphan.rejection_reason or "")
        self.assertTrue(genuine.accepted)
        self.assertEqual(genuine.utterance_id, genuine_token)


class VoicePipelineLifecycleTests(unittest.TestCase):
    @contextmanager
    def mock_audio(self) -> Iterator[tuple[VoiceListener, MagicMock]]:
        # Keep Pipecat's transport, pipeline, and runner real. Only the model
        # loader, cache provisioning, device probe, and audio hardware are mocked.
        with (
            patch("voicerdr.voice._ensure_nltk_punkt"),
            patch("voicerdr.voice._probe_input_device", return_value=0),
            patch(
                "pipecat.services.moonshine.stt.MoonshineSTTService._load",
                return_value=MagicMock(),
            ),
            patch("pipecat.transports.local.audio.pyaudio.PyAudio") as audio,
        ):
            listener = VoiceListener(on_transcript=lambda _final: None)
            try:
                yield listener, audio.return_value
            finally:
                self.assertTrue(listener.stop())

    def test_readiness_waits_for_microphone_setup_and_pipeline_start(self) -> None:
        from pipecat.transports.local.audio import LocalAudioInputTransport

        entered = threading.Event()
        release = threading.Event()
        start_entered = threading.Event()
        start_release = threading.Event()
        original_setup = LocalAudioInputTransport.setup
        original_start = LocalAudioInputTransport.start

        async def delayed_setup(transport, setup):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            await original_setup(transport, setup)

        async def delayed_start(transport, frame):
            start_entered.set()
            while not start_release.is_set():
                await asyncio.sleep(0.01)
            await original_start(transport, frame)

        with (
            self.mock_audio() as (listener, audio),
            patch.object(LocalAudioInputTransport, "setup", delayed_setup),
            patch.object(LocalAudioInputTransport, "start", delayed_start),
        ):
            try:
                listener.start()
                self.assertTrue(entered.wait(timeout=5))
                self.assertFalse(listener.wait_until_ready(timeout=0.05))
                self.assertFalse(listener.running)
                audio.open.assert_not_called()
                release.set()
                self.assertTrue(start_entered.wait(timeout=5))
                self.assertFalse(listener.wait_until_ready(timeout=0.05))
                audio.open.return_value.start_stream.assert_not_called()
                start_release.set()
                self.assertTrue(listener.wait_until_ready(timeout=5))
                audio.open.return_value.start_stream.assert_called_once()
                self.assertIsNone(listener.last_error)
            finally:
                release.set()
                start_release.set()

    def test_microphone_setup_error_fails_readiness_and_stops_listener(self) -> None:
        with self.mock_audio() as (listener, audio):
            audio.open.side_effect = OSError("microphone unavailable after probe")
            readiness_states: list[bool] = []
            original_set = listener._ready.set

            def signal_ready() -> None:
                readiness_states.append(listener.running)
                original_set()

            with patch.object(listener._ready, "set", side_effect=signal_ready):
                listener.start()
                self.assertFalse(listener.wait_until_ready(timeout=5))
                self.assertIn("microphone unavailable", listener.last_error or "")
                self.assertFalse(listener.running)
                assert listener._thread is not None
                listener._thread.join(timeout=3)
                self.assertFalse(listener.capture_eligible)
                self.assertTrue(readiness_states)
                self.assertFalse(any(readiness_states))

    def test_unusable_processor_after_start_stops_listener_and_reports_error(
        self,
    ) -> None:
        from pipecat.transports.local.audio import LocalAudioInputTransport

        transports = []
        original_setup = LocalAudioInputTransport.setup

        async def capture_transport(transport, setup):
            transports.append(transport)
            await original_setup(transport, setup)

        with (
            self.mock_audio() as (listener, _audio),
            patch.object(LocalAudioInputTransport, "setup", capture_transport),
        ):
            listener.start()
            self.assertTrue(listener.wait_until_ready(timeout=5))
            assert listener._loop is not None
            future = asyncio.run_coroutine_threadsafe(
                transports[0].push_error(
                    "microphone lost", force_treat_as_permanent=True
                ),
                listener._loop,
            )
            future.result(timeout=3)
            assert listener._thread is not None
            listener._thread.join(timeout=3)
            self.assertFalse(listener.capture_eligible)
            self.assertFalse(listener.running)
            self.assertIn("microphone lost", listener.last_error or "")
            self.assertFalse(listener.wait_until_ready(timeout=0.1))

    def test_setup_timeout_is_reported_as_startup_failure(self) -> None:
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.transports.local.audio import LocalAudioInputTransport

        async def stuck_setup(_transport, _setup):
            await asyncio.Event().wait()

        with (
            self.mock_audio() as (listener, _audio),
            patch.object(LocalAudioInputTransport, "setup", stuck_setup),
            patch(
                "pipecat.pipeline.worker.PipelineWorker",
                partial(PipelineWorker, setup_timeout_secs=0.1),
            ),
        ):
            listener.start()
            self.assertFalse(listener.wait_until_ready(timeout=5))
            self.assertIn("setup timed out", listener.last_error or "")
            self.assertFalse(listener.running)

    def test_microphone_start_error_cannot_report_successful_readiness(self) -> None:
        with self.mock_audio() as (listener, audio):
            audio.open.return_value.start_stream.side_effect = OSError(
                "microphone could not start"
            )
            listener.start()
            self.assertFalse(listener.wait_until_ready(timeout=5))
            self.assertIn("microphone could not start", listener.last_error or "")
            self.assertFalse(listener.running)

    def test_pipeline_start_timeout_is_reported_as_startup_failure(self) -> None:
        from pipecat.frames.frames import StartFrame
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.processors.audio.vad_processor import VADProcessor

        original_process = VADProcessor.process_frame

        async def block_start(processor, frame, direction):
            if isinstance(frame, StartFrame):
                await asyncio.Event().wait()
            await original_process(processor, frame, direction)

        with (
            self.mock_audio() as (listener, _audio),
            patch.object(VADProcessor, "process_frame", block_start),
            patch(
                "pipecat.pipeline.worker.PipelineWorker",
                partial(PipelineWorker, start_timeout_secs=0.1),
            ),
        ):
            listener.start()
            self.assertFalse(listener.wait_until_ready(timeout=5))
            self.assertIn("startup timed out", listener.last_error or "")
            self.assertFalse(listener.running)


if __name__ == "__main__":
    unittest.main()
