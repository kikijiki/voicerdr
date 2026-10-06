import threading
import unittest
from unittest.mock import patch

from voicerdr.tts import Speaker


class SpeakerAudioUXTests(unittest.TestCase):
    def test_interrupt_cancels_playback_and_clears_pending_speech(self) -> None:
        speaker = Speaker()
        speaker._q.put("pending one")
        speaker._q.put("pending two")

        removed = speaker.interrupt()

        self.assertEqual(removed, 2)
        self.assertTrue(speaker._cancel.is_set())
        self.assertTrue(speaker._q.empty())

    def test_interrupt_can_preserve_queue(self) -> None:
        speaker = Speaker()
        speaker._q.put("pending")

        removed = speaker.interrupt(clear_queue=False)

        self.assertEqual(removed, 0)
        self.assertEqual(speaker._q.get_nowait(), "pending")

    def test_stop_before_start_does_not_poison_next_worker(self) -> None:
        speaker = Speaker()
        spoken = threading.Event()

        speaker.stop()
        with patch.object(speaker, "_say", side_effect=lambda _text: spoken.set()):
            speaker.speak("still works")
            self.assertTrue(spoken.wait(timeout=1))
            speaker.stop()

        self.assertEqual(speaker.last_spoken, "still works")

    def test_replace_preempts_item_taken_before_worker_switches(self) -> None:
        speaker = Speaker()
        old_item_taken = threading.Event()
        allow_old_item = threading.Event()
        spoken: list[str] = []

        original_get = speaker._q.get

        def controlled_get(*args: object, **kwargs: object) -> str | None:
            item = original_get(*args, **kwargs)
            if str(item) == "old":
                old_item_taken.set()
                allow_old_item.wait(timeout=2)
            return item

        def record_speech(text: str) -> None:
            spoken.append(text)

        with (
            patch.object(speaker._q, "get", side_effect=controlled_get),
            patch.object(speaker, "_say", side_effect=record_speech),
        ):
            speaker.speak("old")
            self.assertTrue(old_item_taken.wait(timeout=1))
            speaker.speak("new", replace=True)
            allow_old_item.set()
            for _ in range(100):
                if speaker.last_spoken == "new":
                    break
                threading.Event().wait(0.01)
            speaker.stop()

        self.assertEqual(spoken, ["new"])
        self.assertEqual(speaker.last_spoken, "new")


if __name__ == "__main__":
    unittest.main()
