import atexit
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from voicerdr.config import AppConfig
from voicerdr.daemon import Daemon
from voicerdr.dictation import DictationBuffer
from voicerdr.intent import BoundIntentPlan, IntentEvidence, IntentPlan, IntentTarget
from voicerdr.paths import RuntimePaths
from voicerdr.secretary import SecretaryError, VerificationResult
from voicerdr.talk_policy import TalkPolicy
from voicerdr.voice import FinalTranscript

_TEST_RUNTIME = tempfile.TemporaryDirectory()
atexit.register(_TEST_RUNTIME.cleanup)


def planned(
    action_kind: str,
    *,
    target: IntentTarget | None = None,
    message: str | None = None,
    clarification: str | None = None,
) -> IntentPlan:
    return IntentPlan(
        action_kind=action_kind,
        target=target,
        message=message,
        mode=None,
        clarification=clarification,
        confidence=0.99,
        reason="mocked semantic decision",
        evidence=(
            IntentEvidence("post_wake_content", message)
            if action_kind == "agent_prompt" and message
            else None
        ),
        workspace_evidence=(
            IntentEvidence("post_wake_content", "nine")
            if action_kind in {"agent_prompt", "status"}
            else None
        ),
    )


class FakeSecretary:
    def __init__(self, plans: list[IntentPlan | Exception]) -> None:
        self.plans = plans
        self.calls: list[dict[str, object]] = []

    def plan(self, utterance: str, **kwargs: object) -> BoundIntentPlan:
        self.calls.append({"utterance": utterance, **kwargs})
        planned = self.plans.pop(0)
        if isinstance(planned, Exception):
            raise planned
        return BoundIntentPlan(
            str(kwargs["utterance_digest"]),
            str(kwargs["catalog_digest"]),
            planned,
        )

    def verify_prompt(
        self, *, proposed: IntentPlan, **kwargs: object
    ) -> VerificationResult:
        assert proposed.target and proposed.target.pane_id and proposed.message
        return VerificationResult(
            approved=True,
            utterance_digest=str(kwargs["utterance_digest"]),
            catalog_digest=str(kwargs["catalog_digest"]),
            plan_digest=str(kwargs["plan_digest"]),
            action_kind="agent_prompt",
            target=proposed.target.as_dict(),
            message=proposed.message,
            reason="exactly supported",
        )


class FakeHerdr:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, str, bool]] = []

    @staticmethod
    def workspace_list() -> list[dict[str, object]]:
        return [{"workspace_id": "w9", "label": "#9 backend", "number": 9}]

    @staticmethod
    def agent_list() -> list[dict[str, object]]:
        return [
            {
                "workspace_id": "w9",
                "pane_id": "w9:p1",
                "terminal_title_stripped": "Backend work",
                "agent_status": "idle",
            }
        ]

    def agent_prompt(self, target: str, text: str, *, wait: bool) -> None:
        self.prompts.append((target, text, wait))


def daemon_for_dictation(plans: list[IntentPlan | Exception]) -> Daemon:
    daemon = Daemon.__new__(Daemon)
    # Preserve this suite's explicit legacy wake vocabulary independently of
    # the fresh-install assistant defaults.
    daemon.config = AppConfig(
        focus_on_prompt=False,
        wake_phrases=["secretary"],
    )
    daemon.policy = TalkPolicy(daemon.config)
    daemon.herdr = FakeHerdr()
    daemon.secretary = FakeSecretary(plans)
    daemon.verifier = daemon.secretary
    daemon.dictation = DictationBuffer()
    daemon.pending_clarification = None
    daemon.last_transcript = None
    daemon.last_voice_action = None
    daemon.last_summary = None
    daemon.focused_workspace_id = None
    daemon._last_final_frame = None
    daemon._reload_aliases = Mock()
    daemon._notify = Mock()
    daemon._record_activity = Mock()
    daemon._set_activity_status = Mock()
    root = Path(tempfile.mkdtemp(dir=_TEST_RUNTIME.name))
    daemon.paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
    daemon._replay_ledger_error = None
    daemon._consumed_utterance_ids = daemon._initialize_replay_ledger()
    return daemon


class DictationFlowTests(unittest.TestCase):
    def test_completed_dictation_clears_after_successful_clarification(self) -> None:
        prompt = replace(
            planned(
                "agent_prompt",
                target=IntentTarget("w9", "w9:p1"),
                message="run tests",
            ),
            evidence=IntentEvidence("clarification_request", "run tests"),
            workspace_evidence=IntentEvidence("clarification_answer", "nine"),
        )
        daemon = daemon_for_dictation(
            [
                planned("dictation_start", message="run tests"),
                planned("dictation_finish"),
                planned("clarification", clarification="Which workspace?"),
                prompt,
            ]
        )

        daemon._handle_transcript("Secretary listen, run tests")
        daemon._handle_transcript("finish")
        self.assertTrue(daemon.dictation.active)
        self.assertEqual(
            daemon.pending_clarification["resume_phase"], "dictation_ready"
        )

        daemon._handle_transcript("nine")

        self.assertEqual(daemon.herdr.prompts, [("w9:p1", "run tests", False)])
        self.assertIsNone(daemon.pending_clarification)
        self.assertFalse(daemon.dictation.active)
        self.assertEqual(daemon.dictation.parts, [])
        self.assertEqual(daemon.dictation.raw_transcripts, [])

        daemon._handle_transcript("some unrelated speech")

        self.assertEqual(daemon.last_voice_action["result"]["code"], "wake_not_matched")
        self.assertEqual(len(daemon.secretary.calls), 4)

    def test_failed_completed_dictation_clarification_retains_buffer(self) -> None:
        prompt = replace(
            planned(
                "agent_prompt",
                target=IntentTarget("w9", "w9:p1"),
                message="run tests",
            ),
            evidence=IntentEvidence("clarification_request", "run tests"),
            workspace_evidence=IntentEvidence("clarification_answer", "nine"),
        )
        daemon = daemon_for_dictation(
            [
                planned("dictation_start", message="run tests"),
                planned("dictation_finish"),
                planned("clarification", clarification="Which workspace?"),
                prompt,
            ]
        )
        daemon.secretary.verify_prompt = Mock(side_effect=SecretaryError("offline"))

        daemon._handle_transcript("Secretary listen, run tests")
        daemon._handle_transcript("finish")
        daemon._handle_transcript("nine")

        self.assertEqual(daemon.herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "verifier_error")
        self.assertTrue(daemon.dictation.active)
        self.assertEqual(daemon.dictation.joined(), "run tests")
        self.assertEqual(
            daemon.pending_clarification["resume_phase"], "dictation_ready"
        )

    def test_dictation_fragments_preserve_bytes_and_explicit_boundaries(self) -> None:
        first = "tell nine run\ttests"
        second = "with\u2003care\nnow"
        message = "run\ttests\nwith\u2003care\nnow"
        daemon = daemon_for_dictation(
            [
                planned("dictation_start", message=first),
                planned("dictation_append", message=second),
                planned("dictation_finish"),
                planned(
                    "agent_prompt",
                    target=IntentTarget("w9", "w9:p1"),
                    message=message,
                ),
            ]
        )

        daemon._handle_transcript(f"Secretary listen, {first}")
        daemon._handle_transcript(second)
        self.assertEqual(daemon.dictation.parts, [first, second])
        self.assertEqual(daemon.dictation.joined(), f"{first}\n{second}")
        daemon._handle_transcript("finish")

        self.assertEqual(daemon.herdr.prompts, [("w9:p1", message, False)])
        ready = daemon.secretary.calls[-1]["state"]
        self.assertEqual(ready["complete_fragments"], [first, second])

    def test_no_action_during_dictation_retains_buffer_and_retry_state(self) -> None:
        daemon = daemon_for_dictation(
            [planned("dictation_start", message="release notes"), planned("no_action")]
        )

        daemon._handle_transcript("Secretary listen, release notes")
        daemon._handle_transcript("unclear noise")

        self.assertTrue(daemon.dictation.active)
        self.assertEqual(daemon.dictation.joined(), "release notes")
        status = daemon._set_activity_status.call_args.kwargs
        self.assertTrue(status["capture_active"])
        self.assertIn("retry", status["waiting_for"])

    def test_planner_error_during_dictation_retains_buffer_and_retry_state(
        self,
    ) -> None:
        daemon = daemon_for_dictation(
            [
                planned("dictation_start", message="release notes"),
                SecretaryError("offline"),
            ]
        )

        daemon._handle_transcript("Secretary listen, release notes")
        daemon._handle_transcript("more detail")

        self.assertTrue(daemon.dictation.active)
        self.assertEqual(daemon.dictation.joined(), "release notes")
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")
        status = daemon._set_activity_status.call_args.kwargs
        self.assertTrue(status["capture_active"])
        self.assertIn("retained", status["waiting_for"])

    def test_wake_and_message_is_planned_as_one_shot(self) -> None:
        daemon = daemon_for_dictation(
            [
                planned(
                    "agent_prompt",
                    target=IntentTarget("w9", "w9:p1"),
                    message="hello",
                )
            ]
        )

        daemon._handle_transcript("Secretary tell nine hello")

        self.assertFalse(daemon.dictation.active)
        self.assertEqual(daemon.herdr.prompts, [("w9:p1", "hello", False)])
        self.assertEqual(daemon.secretary.calls[0]["utterance"], "tell nine hello")

    def test_llm_controls_buffer_start_append_finish_and_final_action(self) -> None:
        daemon = daemon_for_dictation(
            [
                planned("dictation_start", message="tell nine first part"),
                planned("dictation_append", message="and the second part"),
                planned("dictation_finish"),
                planned(
                    "agent_prompt",
                    target=IntentTarget("w9", "w9:p1"),
                    message="first part\nand the second part",
                ),
            ]
        )

        daemon._handle_transcript("Secretary listen, tell nine first part")
        daemon._handle_transcript("and the second part")

        self.assertTrue(daemon.dictation.active)
        self.assertEqual(
            daemon.dictation.joined(), "tell nine first part\nand the second part"
        )
        self.assertEqual(daemon.herdr.prompts, [])

        daemon._handle_transcript("send it.")

        self.assertFalse(daemon.dictation.active)
        self.assertEqual(
            daemon.herdr.prompts,
            [("w9:p1", "first part\nand the second part", False)],
        )
        self.assertEqual(
            daemon.secretary.calls[2]["state"]["phase"], "dictation_capture"
        )
        self.assertEqual(daemon.secretary.calls[3]["state"]["phase"], "dictation_ready")

    def test_cancel_words_are_preserved_when_llm_says_append(self) -> None:
        daemon = daemon_for_dictation(
            [
                planned("dictation_start", message="migration notes"),
                planned("dictation_append", message="cancel that migration workaround"),
            ]
        )

        daemon._handle_transcript("Secretary listen, migration notes")
        daemon._handle_transcript("cancel that migration workaround")

        self.assertTrue(daemon.dictation.active)
        self.assertEqual(
            daemon.dictation.joined(),
            "migration notes\ncancel that migration workaround",
        )

    def test_send_it_words_are_preserved_when_llm_says_append(self) -> None:
        daemon = daemon_for_dictation(
            [
                planned("dictation_start", message="deployment notes"),
                planned("dictation_append", message="send it after every test passes"),
            ]
        )

        daemon._handle_transcript("Secretary listen, deployment notes")
        daemon._handle_transcript("send it after every test passes")

        self.assertEqual(
            daemon.dictation.joined(),
            "deployment notes\nsend it after every test passes",
        )

    def test_model_cannot_rewrite_dictation_content(self) -> None:
        daemon = daemon_for_dictation(
            [
                planned("dictation_start", message="release notes"),
                planned("dictation_append", message="rewritten model words"),
            ]
        )

        daemon._handle_transcript("Secretary listen, release notes")
        daemon._handle_transcript("preserve these exact user words")

        self.assertTrue(daemon.dictation.active)
        self.assertEqual(daemon.dictation.joined(), "release notes")
        self.assertEqual(daemon.last_voice_action["result"]["code"], "plan_blocked")
        self.assertFalse(daemon.last_voice_action["result"]["sent"])

    def test_dictation_transition_clarification_returns_to_llm_and_resumes(
        self,
    ) -> None:
        daemon = daemon_for_dictation(
            [
                planned("dictation_start", message="deployment notes"),
                planned(
                    "clarification",
                    clarification="Should I append those words or finish dictation?",
                ),
                planned("dictation_finish"),
                planned("no_action"),
            ]
        )

        daemon._handle_transcript("Secretary listen, deployment notes")
        daemon._handle_transcript("go ahead")
        self.assertTrue(daemon.dictation.active)

        daemon._handle_transcript("finish the dictation")

        self.assertTrue(daemon.dictation.active)
        self.assertEqual(daemon.dictation.joined(), "deployment notes")
        clarification_state = daemon.secretary.calls[2]["state"]
        self.assertEqual(clarification_state["phase"], "clarification")
        self.assertEqual(clarification_state["resume_phase"], "dictation_capture")
        self.assertEqual(daemon.herdr.prompts, [])

    def test_duplicate_final_frame_is_idempotently_withheld(self) -> None:
        daemon = daemon_for_dictation(
            [
                planned(
                    "agent_prompt",
                    target=IntentTarget("w9", "w9:p1"),
                    message="run tests",
                )
            ]
        )

        final = FinalTranscript("voice:test:one", "Secretary tell nine run tests")
        daemon._handle_transcript(final)
        daemon._handle_transcript(final)

        self.assertEqual(daemon.herdr.prompts, [("w9:p1", "run tests", False)])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"], "utterance_already_consumed"
        )
        self.assertFalse(daemon.last_voice_action["result"]["sent"])


if __name__ == "__main__":
    unittest.main()
