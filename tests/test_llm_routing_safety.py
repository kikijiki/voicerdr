import atexit
import fcntl
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from voicerdr.config import AppConfig
from voicerdr.control_client import ControlClient, ControlClientError
from voicerdr.control_protocol import ControlRequest
from voicerdr.daemon import Daemon, ReplayLedgerError
from voicerdr.dictation import DictationBuffer
from voicerdr.herdr_client import HerdrError
from voicerdr.intent import BoundIntentPlan, IntentEvidence, IntentPlan, IntentTarget
from voicerdr.paths import RuntimePaths
from voicerdr.secretary import (
    VERIFICATION_REASON_ACTION,
    VERIFICATION_REASON_APPROVED,
    VERIFICATION_REASON_PAYLOAD,
    SecretaryError,
    SecretaryTransportError,
    VerificationResult,
)
from voicerdr.talk_policy import TalkPolicy
from voicerdr.voice import FinalTranscript, VoiceListener

WORKSPACES = [
    {"workspace_id": "w1", "label": "#1 frontend", "number": 1},
    {"workspace_id": "w2", "label": "#2 backend", "number": 2},
]
AGENTS = [
    {
        "workspace_id": "w1",
        "pane_id": "w1:p1",
        "terminal_title_stripped": "Frontend",
        "agent_status": "idle",
    },
    {
        "workspace_id": "w2",
        "pane_id": "w2:p1",
        "terminal_title_stripped": "Backend",
        "agent_status": "idle",
    },
]
_TEST_RUNTIME = tempfile.TemporaryDirectory()
atexit.register(_TEST_RUNTIME.cleanup)


def planned(
    action_kind: str,
    *,
    target: IntentTarget | None = None,
    message: str | None = None,
    mode: str | None = None,
    clarification: str | None = None,
    confidence: float = 0.99,
    actions: tuple[IntentPlan, ...] = (),
    evidence: IntentEvidence | None = None,
    agent_evidence: str | IntentEvidence | None = None,
    workspace_evidence: str | IntentEvidence | None = None,
    unresolved_slots: tuple[str, ...] = (),
) -> IntentPlan:
    return IntentPlan(
        action_kind=action_kind,
        target=target,
        message=message,
        mode=mode,
        clarification=clarification,
        confidence=confidence,
        reason="mocked LLM decision",
        actions=actions,
        evidence=(
            evidence or IntentEvidence("post_wake_content", message)
            if action_kind == "agent_prompt" and message
            else None
        ),
        agent_evidence=(
            IntentEvidence("post_wake_content", agent_evidence)
            if isinstance(agent_evidence, str)
            else agent_evidence
        ),
        workspace_evidence=(
            workspace_evidence
            if isinstance(workspace_evidence, IntentEvidence)
            else IntentEvidence(
                "post_wake_content",
                workspace_evidence
                or (
                    {"w1": "one", "w2": "two"}.get(target.workspace_id)
                    if target
                    else ""
                ),
            )
            if action_kind in {"agent_prompt", "status"}
            else None
        ),
        unresolved_slots=unresolved_slots,
    )


class FakeHerdr:
    def __init__(self) -> None:
        self.workspaces = [dict(row) for row in WORKSPACES]
        self.agents = [dict(row) for row in AGENTS]
        self.prompts: list[tuple[str, str, bool]] = []
        self.prompt_error_after_send: HerdrError | None = None
        self.focus_error: OSError | None = None
        self.notification_error: OSError | None = None

    def workspace_list(self) -> list[dict[str, object]]:
        return self.workspaces

    def agent_list(self) -> list[dict[str, object]]:
        return self.agents

    def agent_prompt(self, target: str, text: str, *, wait: bool) -> None:
        self.prompts.append((target, text, wait))
        if self.prompt_error_after_send:
            raise self.prompt_error_after_send

    def workspace_focus(self, _workspace_id: str) -> None:
        if self.focus_error:
            raise self.focus_error

    def notification_show(
        self, _title: str, _body: str, *, sound: str = "done"
    ) -> None:
        if self.notification_error:
            raise self.notification_error


class FakeSecretary:
    def __init__(self, plan: IntentPlan | Exception) -> None:
        self.plan_result = plan
        self.plan_calls: list[dict[str, object]] = []
        self.verification: VerificationResult | Exception | None = None
        self.on_verify = None

    def plan(self, utterance: str, **kwargs: object) -> BoundIntentPlan:
        self.plan_calls.append({"utterance": utterance, **kwargs})
        if isinstance(self.plan_result, Exception):
            raise self.plan_result
        return BoundIntentPlan(
            str(kwargs["utterance_digest"]),
            str(kwargs["catalog_digest"]),
            self.plan_result,
        )

    def verify_prompt(
        self, *, proposed: IntentPlan, **kwargs: object
    ) -> VerificationResult:
        if self.on_verify:
            self.on_verify()
        if isinstance(self.verification, Exception):
            raise self.verification
        if self.verification:
            verification = self.verification
            return replace(
                verification,
                utterance_digest=(
                    verification.utterance_digest or str(kwargs["utterance_digest"])
                ),
                catalog_digest=(
                    verification.catalog_digest or str(kwargs["catalog_digest"])
                ),
                plan_digest=verification.plan_digest or str(kwargs["plan_digest"]),
            )
        assert proposed.target and proposed.target.pane_id and proposed.message
        return VerificationResult(
            approved=True,
            utterance_digest=str(kwargs["utterance_digest"]),
            catalog_digest=str(kwargs["catalog_digest"]),
            plan_digest=str(kwargs["plan_digest"]),
            action_kind="agent_prompt",
            target={
                "workspace_id": proposed.target.workspace_id,
                "pane_id": proposed.target.pane_id,
            },
            message=proposed.message,
            reason="exact match",
        )


def make_daemon(
    plan: IntentPlan | Exception,
) -> tuple[Daemon, FakeHerdr, FakeSecretary]:
    daemon = Daemon.__new__(Daemon)
    # These routing-safety fixtures intentionally exercise a legacy customized
    # wake phrase; fresh-install Jenny defaults are covered in test_regressions.
    daemon.config = AppConfig(
        focus_on_prompt=False,
        wake_phrases=["secretary"],
    )
    daemon.policy = TalkPolicy(daemon.config)
    herdr = FakeHerdr()
    secretary = FakeSecretary(plan)
    daemon.herdr = herdr
    daemon.secretary = secretary
    daemon.verifier = secretary
    daemon.speaker = Mock(enabled=True, speak=Mock(), last_error=None, last_spoken=None)
    daemon.dictation = DictationBuffer()
    daemon.pending_clarification = None
    daemon._voice_lock = threading.Lock()
    daemon._voice_state_lock = threading.RLock()
    daemon._authority_lock = threading.RLock()
    daemon._listener_generation = 0
    daemon.voice = None
    daemon.last_transcript = None
    daemon.last_voice_action = None
    daemon.last_summary = None
    daemon.focused_workspace_id = "w1"
    daemon._last_final_frame = None
    daemon._reload_aliases = Mock()
    daemon._notify = Mock()
    daemon._set_activity_status = Mock()
    daemon.activity_events = []
    daemon._record_activity = lambda event, **fields: daemon.activity_events.append(
        {"event": event, **fields}
    )
    root = Path(tempfile.mkdtemp(dir=_TEST_RUNTIME.name))
    daemon.paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
    daemon._replay_ledger_error = None
    daemon._consumed_utterance_ids = daemon._initialize_replay_ledger()
    return daemon, herdr, secretary


class LLMRoutingSafetyTests(unittest.TestCase):
    def test_punctuated_configured_wake_preserves_planning_evidence(self) -> None:
        daemon, _, secretary = make_daemon(planned("no_action"))
        daemon.config.wake_phrases = ["hey jenny"]
        transcript = "Hey, Jenny, summarize number six."

        daemon._handle_transcript(transcript)

        self.assertEqual(len(secretary.plan_calls), 1)
        call = secretary.plan_calls[0]
        self.assertEqual(call["utterance"], "summarize number six.")
        self.assertEqual(call["raw_transcript"], transcript)
        self.assertEqual(call["activation_phrase"], "hey jenny")
        self.assertEqual(call["state"]["source_content"], "summarize number six.")

    def test_fresh_wake_request_supersedes_stale_clarification_without_concat(
        self,
    ) -> None:
        daemon, herdr, secretary = make_daemon(
            planned(
                "clarification",
                clarification="Did you mean status for workspace two?",
                unresolved_slots=("action", "workspace"),
            )
        )
        daemon._handle_transcript("Secretary summarized, too.")
        clarification_id = daemon.pending_clarification["clarification_id"]

        secretary.plan_result = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon._handle_transcript("Secretary tell one run tests")

        secretary.plan_result = planned("no_action")
        complaint = "Secretary why did you keep appending my old request?"
        daemon._handle_transcript(complaint)

        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])
        self.assertIsNone(daemon.pending_clarification)
        self.assertEqual(
            [call["raw_transcript"] for call in secretary.plan_calls],
            [
                "Secretary summarized, too.",
                "Secretary tell one run tests",
                complaint,
            ],
        )
        self.assertEqual(
            [call["utterance"] for call in secretary.plan_calls],
            [
                "summarized, too.",
                "tell one run tests",
                "why did you keep appending my old request?",
            ],
        )
        self.assertTrue(
            any(
                event.get("event") == "clarification_superseded"
                and event.get("clarification_id") == clarification_id
                for event in daemon.activity_events
            )
        )

    def test_wrong_workspace_index_evidence_cannot_authorize_prompt(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w2", "w2:p1"),
                message="run tests",
                workspace_evidence="one",
            )
        )

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "plan_blocked")
        self.assertIn(
            "different catalog workspace", daemon.last_voice_action["message"]
        )

    def test_one_shot_delivery_preserves_tabs_newlines_and_unicode_whitespace(
        self,
    ) -> None:
        message = "run\ttests\nwith\u2003care"
        daemon, herdr, secretary = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message=message,
            )
        )

        daemon._handle_transcript(f"Secretary, tell one {message}")

        self.assertEqual(secretary.plan_calls[0]["utterance"], f"tell one {message}")
        self.assertEqual(herdr.prompts, [("w1:p1", message, False)])

    def test_wrong_workspace_label_evidence_cannot_authorize_status(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "status",
                target=IntentTarget("w2", "w2:p1"),
                workspace_evidence="frontend",
            )
        )
        daemon.summarize_agent = Mock()

        daemon._handle_transcript("Secretary summarize frontend")

        daemon.summarize_agent.assert_not_called()
        self.assertEqual(herdr.prompts, [])
        self.assertIn(
            "different catalog workspace", daemon.last_voice_action["message"]
        )

    def test_ordinal_workspace_evidence_selects_only_its_exact_row(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "status",
                target=IntentTarget("w2", "w2:p1"),
                workspace_evidence="second",
            )
        )
        daemon.summarize_agent = Mock(return_value="Backend is working.")
        daemon._handle_transcript("Secretary summarize the second workspace")
        daemon.summarize_agent.assert_called_once_with("w2:p1")
        self.assertEqual(herdr.prompts, [])

        daemon, _, _ = make_daemon(
            planned(
                "status",
                target=IntentTarget("w2", "w2:p1"),
                workspace_evidence="first",
            )
        )
        daemon.summarize_agent = Mock()
        daemon._handle_transcript("Secretary summarize the first workspace")
        daemon.summarize_agent.assert_not_called()
        self.assertIn(
            "different catalog workspace", daemon.last_voice_action["message"]
        )

    def test_routing_evidence_cannot_overlap_prompt_payload(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="one run tests",
                evidence=IntentEvidence("post_wake_content", "one run tests"),
                workspace_evidence=IntentEvidence("raw_transcript", "one"),
            )
        )
        daemon._handle_transcript("Secretary tell one run tests")
        self.assertEqual(herdr.prompts, [])
        self.assertIn("overlaps prompt payload", daemon.last_voice_action["message"])

    def test_capability_mac_encoding_is_unambiguous_and_tamper_fails_delivery(
        self,
    ) -> None:
        self.assertNotEqual(
            Daemon._canonical_capability_bytes(("a", "b\0c")),
            Daemon._canonical_capability_bytes(("a\0b", "c")),
        )
        daemon, herdr, _ = make_daemon(planned("no_action"))
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        catalog = daemon._space_directory()
        source_bundle = {
            "post_wake_content": "tell one run tests",
            "raw_transcript": "Secretary tell one run tests",
        }
        capability = daemon._mint_prompt_capability(
            utterance_id="typed:sealed",
            utterance_digest="a" * 64,
            catalog_digest=daemon._catalog_fingerprint(catalog),
            plan_digest="b" * 64,
            plan=plan,
            source_bundle=source_bundle,
            listener_generation=0,
            origin="typed",
            global_generation=0,
        )
        tampered = replace(capability, pane_id="w2:p1")
        result = daemon._deliver_verified_prompt(
            tampered, catalog=catalog, source_bundle=source_bundle
        )
        self.assertEqual(result["code"], "invalid_verified_capability")
        self.assertEqual(herdr.prompts, [])

    def test_duplicate_agent_title_evidence_requires_clarification(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p2"),
                message="run tests",
                workspace_evidence="frontend",
                agent_evidence="Security review",
            )
        )
        herdr.agents.extend(
            [
                {
                    "workspace_id": "w1",
                    "pane_id": "w1:p2",
                    "terminal_title_stripped": "Security review",
                    "agent_status": "idle",
                },
                {
                    "workspace_id": "w1",
                    "pane_id": "w1:p3",
                    "terminal_title_stripped": "Security review",
                    "agent_status": "idle",
                },
            ]
        )

        daemon._handle_transcript("Secretary tell frontend Security review run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertIn(
            "uniquely identify one catalog pane", daemon.last_voice_action["message"]
        )

    def test_duplicate_agent_title_cannot_select_status_pane(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "status",
                target=IntentTarget("w1", "w1:p2"),
                workspace_evidence="frontend",
                agent_evidence="Security review",
            )
        )
        herdr.agents.extend(
            [
                {
                    "workspace_id": "w1",
                    "pane_id": "w1:p2",
                    "terminal_title_stripped": "Security review",
                    "agent_status": "idle",
                },
                {
                    "workspace_id": "w1",
                    "pane_id": "w1:p3",
                    "terminal_title_stripped": "Security review",
                    "agent_status": "idle",
                },
            ]
        )
        daemon.summarize_agent = Mock()

        daemon._handle_transcript("Secretary summarize frontend Security review")

        daemon.summarize_agent.assert_not_called()
        self.assertEqual(herdr.prompts, [])
        self.assertIn(
            "uniquely identify one catalog pane", daemon.last_voice_action["message"]
        )

    def test_planner_preflight_failure_gets_bounded_corrective_replan(self) -> None:
        invalid = planned(
            "agent_prompt",
            target=IntentTarget("w2", "w2:p1"),
            message="run tests",
            workspace_evidence="one",
        )
        valid = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
            workspace_evidence="one",
        )
        daemon, herdr, secretary = make_daemon(invalid)
        decisions = iter((invalid, valid))

        def replan(utterance: str, **kwargs: object) -> BoundIntentPlan:
            secretary.plan_calls.append({"utterance": utterance, **kwargs})
            return BoundIntentPlan(
                str(kwargs["utterance_digest"]),
                str(kwargs["catalog_digest"]),
                next(decisions),
            )

        secretary.plan = replan
        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])
        self.assertEqual(len(secretary.plan_calls), 2)
        self.assertIn("correction", secretary.plan_calls[1])

    def test_multi_agent_combined_evidence_corrects_to_exact_safe_prompt(self) -> None:
        invalid = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p2"),
            message="Run the exact audit.",
            evidence=IntentEvidence("post_wake_content", "Run the exact audit."),
            workspace_evidence="gamma Security review",
        )
        valid = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p2"),
            message="Run the exact audit.",
            evidence=IntentEvidence("post_wake_content", "Run the exact audit."),
            workspace_evidence="gamma",
            agent_evidence="Security review",
        )
        daemon, herdr, secretary = make_daemon(invalid)
        herdr.workspaces[0]["label"] = "#1 gamma"
        herdr.agents.append(
            {
                "workspace_id": "w1",
                "pane_id": "w1:p2",
                "terminal_title_stripped": "Security review",
                "agent_status": "idle",
            }
        )
        decisions = iter((invalid, valid))

        def replan(utterance: str, **kwargs: object) -> BoundIntentPlan:
            secretary.plan_calls.append({"utterance": utterance, **kwargs})
            return BoundIntentPlan(
                str(kwargs["utterance_digest"]),
                str(kwargs["catalog_digest"]),
                next(decisions),
            )

        secretary.plan = replan
        daemon._handle_transcript(
            "Secretary ask gamma Security review Run the exact audit."
        )

        self.assertEqual(herdr.prompts, [("w1:p2", "Run the exact audit.", False)])
        self.assertEqual(len(secretary.plan_calls), 2)
        self.assertNotIn("clarification_only", secretary.plan_calls[1])

    def test_multi_agent_bad_evidence_ends_in_llm_clarification_not_plan_blocked(
        self,
    ) -> None:
        invalid = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p2"),
            message="Run the exact audit.",
            evidence=IntentEvidence("post_wake_content", "Run the exact audit."),
            workspace_evidence="gamma Security review",
        )
        clarification = planned(
            "clarification",
            clarification="Please name gamma and Security review separately.",
            unresolved_slots=("workspace", "agent"),
        )
        daemon, herdr, secretary = make_daemon(invalid)
        herdr.workspaces[0]["label"] = "#1 gamma"
        herdr.agents.append(
            {
                "workspace_id": "w1",
                "pane_id": "w1:p2",
                "terminal_title_stripped": "Security review",
                "agent_status": "idle",
            }
        )
        decisions = iter((invalid, invalid, clarification))

        def replan(utterance: str, **kwargs: object) -> BoundIntentPlan:
            secretary.plan_calls.append({"utterance": utterance, **kwargs})
            return BoundIntentPlan(
                str(kwargs["utterance_digest"]),
                str(kwargs["catalog_digest"]),
                next(decisions),
            )

        secretary.plan = replan
        daemon._handle_transcript(
            "Secretary ask gamma Security review Run the exact audit."
        )

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"], "clarification_required"
        )
        self.assertEqual(len(secretary.plan_calls), 3)
        self.assertTrue(secretary.plan_calls[2]["clarification_only"])
        self.assertTrue(
            all(
                "previous_plan" not in call and "messages" not in call
                for call in secretary.plan_calls
            )
        )

    def test_failed_clarification_only_retry_is_visible_planner_error(self) -> None:
        invalid = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p2"),
            message="Run the exact audit.",
            evidence=IntentEvidence("post_wake_content", "Run the exact audit."),
            workspace_evidence="gamma Security review",
        )
        daemon, herdr, secretary = make_daemon(invalid)
        herdr.workspaces[0]["label"] = "#1 gamma"
        herdr.agents.append(
            {
                "workspace_id": "w1",
                "pane_id": "w1:p2",
                "terminal_title_stripped": "Security review",
                "agent_status": "idle",
            }
        )
        calls = 0

        def replan(utterance: str, **kwargs: object) -> BoundIntentPlan:
            nonlocal calls
            calls += 1
            secretary.plan_calls.append({"utterance": utterance, **kwargs})
            if calls == 3:
                raise SecretaryError("clarification contract unavailable")
            return BoundIntentPlan(
                str(kwargs["utterance_digest"]),
                str(kwargs["catalog_digest"]),
                invalid,
            )

        secretary.plan = replan
        daemon._handle_transcript(
            "Secretary ask gamma Security review Run the exact audit."
        )

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")
        self.assertIn(
            "clarification contract unavailable", daemon.last_voice_action["message"]
        )
        self.assertTrue(secretary.plan_calls[2]["clarification_only"])

    def test_self_contradictory_verifier_is_retried_independently(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        calls = 0

        def verify(*, proposed: IntentPlan, **kwargs: object) -> VerificationResult:
            nonlocal calls
            calls += 1
            assert proposed.target and proposed.message
            return VerificationResult(
                approved=calls > 1,
                action_kind="agent_prompt",
                target=proposed.target.as_dict(),
                message=proposed.message,
                reason="structured decision",
                utterance_digest=str(kwargs["utterance_digest"]),
                catalog_digest=str(kwargs["catalog_digest"]),
                plan_digest=str(kwargs["plan_digest"]),
            )

        secretary.verify_prompt = verify
        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(calls, 2)
        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])

    def test_logged_proper_substring_false_rejection_is_corrected(self) -> None:
        message = "I agree with his assessment and it should do it."
        plan = planned(
            "agent_prompt",
            target=IntentTarget("wAD", "wAD:p1"),
            message=message,
            workspace_evidence="six",
        )
        daemon, herdr, secretary = make_daemon(plan)
        herdr.workspaces = [
            {"workspace_id": "wAD", "label": "#6 combeanie", "number": 6}
        ]
        herdr.agents = [
            {
                "workspace_id": "wAD",
                "pane_id": "wAD:p1",
                "terminal_title_stripped": "enable-robot-motion-commands",
                "agent_status": "working",
            }
        ]
        calls: list[dict[str, object]] = []

        def verify(*, proposed: IntentPlan, **kwargs: object) -> VerificationResult:
            calls.append(kwargs)
            assert proposed.target and proposed.message
            rejected = len(calls) == 1
            return VerificationResult(
                approved=not rejected,
                action_kind="agent_prompt",
                target=proposed.target.as_dict(),
                message=proposed.message,
                reason=(
                    "expected routing word 'that'; payload is not contiguous"
                    if rejected
                    else "server-established payload facts and safety checks pass"
                ),
                reason_kind=(
                    VERIFICATION_REASON_PAYLOAD
                    if rejected
                    else VERIFICATION_REASON_APPROVED
                ),
                utterance_digest=str(kwargs["utterance_digest"]),
                catalog_digest=str(kwargs["catalog_digest"]),
                plan_digest=str(kwargs["plan_digest"]),
                source_exact=not rejected,
            )

        secretary.verify_prompt = verify
        daemon._handle_transcript(
            "Secretary tells six that I agree with his assessment and it should do it."
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0]["established_payload_facts"],
            {
                "evidence_source": "post_wake_content",
                "evidence_source_present": True,
                "message_equals_evidence_quote": True,
                "evidence_quote_occurrences": 1,
                "evidence_quote_is_unique_contiguous": True,
                "source_exact": True,
            },
        )
        self.assertIn("correction", calls[1])
        self.assertEqual(herdr.prompts, [("wAD:p1", message, False)])

    def test_factual_reason_cannot_hide_behind_semantic_check(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        calls = 0

        def verify(*, proposed: IntentPlan, **kwargs: object) -> VerificationResult:
            nonlocal calls
            calls += 1
            assert proposed.target and proposed.message
            return VerificationResult(
                approved=calls > 1,
                action_kind="agent_prompt",
                target=proposed.target.as_dict(),
                message=proposed.message,
                reason="payload is not contiguous" if calls == 1 else "safe",
                reason_kind=(
                    VERIFICATION_REASON_PAYLOAD
                    if calls == 1
                    else VERIFICATION_REASON_APPROVED
                ),
                utterance_digest=str(kwargs["utterance_digest"]),
                catalog_digest=str(kwargs["catalog_digest"]),
                plan_digest=str(kwargs["plan_digest"]),
                source_exact=True,
                action_exact=calls > 1,
            )

        secretary.verify_prompt = verify
        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(calls, 2)
        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])

    def test_genuine_semantic_verifier_veto_remains_terminal(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        secretary.verification = VerificationResult(
            approved=False,
            action_kind="agent_prompt",
            target={"workspace_id": "w1", "pane_id": "w1:p1"},
            message="run tests",
            reason="the utterance does not authorize an agent prompt",
            reason_kind=VERIFICATION_REASON_ACTION,
            action_exact=False,
        )

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(len(secretary.plan_calls), 1)
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"], "verifier_rejected"
        )

    def test_existing_workspace_without_agent_has_helpful_error(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w3", "w3:p1"),
            message="if Dms trashed.",
            workspace_evidence="three",
        )
        daemon, herdr, _ = make_daemon(plan)
        herdr.workspaces = [
            {"workspace_id": "w3", "label": "#3 nixos-config", "number": 3}
        ]
        herdr.agents = []

        daemon._handle_transcript("Secretary, ask three if Dms trashed.")

        self.assertEqual(herdr.prompts, [])
        self.assertIn(
            "Workspace '#3 nixos-config' exists", daemon.last_voice_action["message"]
        )
        self.assertIn("no live agent pane", daemon.last_voice_action["message"])

    def test_mute_generation_during_planner_revokes_delivery(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        original = secretary.plan

        def planning(utterance: str, **kwargs: object) -> BoundIntentPlan:
            result = original(utterance, **kwargs)
            daemon._mute_voice()
            return result

        secretary.plan = planning
        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "listener_authority_revoked",
        )

    def test_spoken_mute_completes_off_the_real_listener_callback_thread(
        self,
    ) -> None:
        daemon, _herdr, _secretary = make_daemon(planned("control", mode="mute"))
        callback_returned = threading.Event()
        release_transport = threading.Event()
        mute_completed = threading.Event()
        original_commit = daemon._commit_voice_action

        def observe_commit(action: dict[str, object]) -> None:
            original_commit(action)
            result = action.get("result")
            if isinstance(result, dict) and result.get("ok") is True:
                mute_completed.set()

        daemon._commit_voice_action = observe_commit
        listener = VoiceListener(
            on_transcript=daemon._handle_transcript,
            listener_generation=daemon._current_listener_generation(),
        )

        def deliver_from_listener() -> None:
            listener.running = True
            listener.on_transcript(
                FinalTranscript(
                    "voice:callback-mute",
                    "Secretary mute",
                    listener_generation=listener.listener_generation,
                )
            )
            callback_returned.set()
            if not release_transport.wait(2):
                raise AssertionError("listener transport was not released")
            listener.running = False

        listener._thread_main = deliver_from_listener
        daemon.voice = listener
        listener.start()

        self.assertTrue(callback_returned.wait(2))
        self.assertTrue(listener.capture_eligible)
        self.assertEqual(daemon.policy.mic_mode, "mute")
        self.assertEqual(
            daemon.last_voice_action["result"]["code"], "mic_closure_pending"
        )
        self.assertIsNone(daemon.last_voice_action["result"]["ok"])
        self.assertEqual(
            daemon._set_activity_status.call_args.kwargs["phase"], "muting"
        )
        self.assertTrue(daemon._set_activity_status.call_args.kwargs["capture_active"])
        self.assertFalse(mute_completed.is_set())

        release_transport.set()
        self.assertTrue(mute_completed.wait(2))
        self.assertFalse(listener.capture_eligible)
        self.assertIsNone(daemon.voice)
        self.assertEqual(
            daemon.last_voice_action["result"],
            {"ok": True, "sent": False, "mode": "mute"},
        )
        self.assertEqual(daemon._set_activity_status.call_args.kwargs["phase"], "muted")
        self.assertFalse(daemon._set_activity_status.call_args.kwargs["capture_active"])
        self.assertFalse(getattr(daemon, "_delivery_closed", False))

    def test_handoff_freezes_mode_during_real_voice_planning_race(self) -> None:
        daemon, _herdr, secretary = make_daemon(planned("control", mode="mute"))
        daemon.policy.set_mic_mode("listen")
        daemon._persist_mic_preference("listen")
        daemon._schedule_shutdown = Mock()
        planning = threading.Event()
        release_plan = threading.Event()
        original_plan = secretary.plan

        def blocked_plan(utterance: str, **kwargs: object) -> BoundIntentPlan:
            planning.set()
            if not release_plan.wait(2):
                raise AssertionError("planner barrier was not released")
            return original_plan(utterance, **kwargs)

        secretary.plan = blocked_plan
        voice_thread = threading.Thread(
            target=daemon._handle_transcript,
            args=("Secretary mute",),
        )
        voice_thread.start()
        self.assertTrue(planning.wait(2))

        handoff = json.loads(
            daemon._dispatch(ControlRequest("handoff", "handoff_quit", {}))
        )
        self.assertEqual(handoff["result"]["mode"], "listen")
        self.assertTrue(daemon._delivery_closed)
        release_plan.set()
        voice_thread.join(2)

        self.assertFalse(voice_thread.is_alive())
        self.assertEqual(daemon.policy.mic_mode, "listen")
        self.assertEqual(
            json.loads(daemon.paths.mic_preference.read_text()),
            {"mode": "mute", "version": 1},
        )
        self.assertEqual(daemon.last_voice_action["result"]["code"], "delivery_closed")

    def test_mute_generation_during_verifier_revokes_delivery(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        secretary.on_verify = daemon._mute_voice

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "listener_authority_revoked",
        )

    def test_mute_revokes_voice_but_not_typed_origin(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        original = secretary.plan

        def planning(utterance: str, **kwargs: object) -> BoundIntentPlan:
            result = original(utterance, **kwargs)
            daemon._mute_voice()
            return result

        secretary.plan = planning
        daemon._handle_transcript(
            FinalTranscript(
                "typed:mute-independent",
                "Secretary tell one run tests",
                origin="typed",
            )
        )

        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])

    def test_quit_globally_revokes_typed_delivery_before_ack(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        daemon._stop = threading.Event()
        original = secretary.plan

        def planning(utterance: str, **kwargs: object) -> BoundIntentPlan:
            result = original(utterance, **kwargs)
            with patch("voicerdr.daemon.threading.Thread.start"):
                response = daemon._dispatch(ControlRequest("quit-1", "quit", {}))
            self.assertTrue(json.loads(response)["result"]["quitting"])
            return result

        secretary.plan = planning
        daemon._handle_transcript(
            FinalTranscript(
                "typed:quit-revocation",
                "Secretary tell one run tests",
                origin="typed",
            )
        )

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "delivery_closed",
        )

    def test_quit_during_verifier_closes_delivery(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        daemon._stop = threading.Event()

        def quit_during_verify() -> None:
            with patch("voicerdr.daemon.threading.Thread.start"):
                daemon._dispatch(ControlRequest("quit-verify", "quit", {}))

        secretary.on_verify = quit_during_verify
        daemon._handle_transcript(
            FinalTranscript(
                "typed:quit-during-verifier",
                "Secretary tell one run tests",
                origin="typed",
            )
        )

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "delivery_closed")

    def test_post_quit_ack_ingest_is_rejected_before_planner(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))
        daemon._stop = threading.Event()
        with patch("voicerdr.daemon.threading.Thread.start"):
            quit_response = json.loads(
                daemon._dispatch(ControlRequest("quit", "quit", {}))
            )
        ingest_response = json.loads(
            daemon._dispatch(
                ControlRequest(
                    "late",
                    "ingest_transcript",
                    {
                        "utterance_id": "typed:after-quit",
                        "text": "Secretary tell one run tests",
                    },
                )
            )
        )

        self.assertTrue(quit_response["result"]["quitting"])
        self.assertEqual(ingest_response["error"]["code"], "delivery_closed")
        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])

    def test_already_connected_ingest_after_quit_ack_is_rejected(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))
        daemon._stop = threading.Event()
        server_side, client_side = socket.socketpair()
        try:
            with patch("voicerdr.daemon.threading.Thread.start"):
                daemon._dispatch(ControlRequest("quit", "quit", {}))
            request = {
                "id": "connected-before-quit",
                "method": "ingest_transcript",
                "params": {
                    "utterance_id": "typed:connected-after-quit",
                    "text": "Secretary tell one run tests",
                },
            }
            client_side.sendall((json.dumps(request) + "\n").encode())
            daemon._handle_conn(server_side)
            response = json.loads(client_side.recv(65536).decode())
        finally:
            client_side.close()
            server_side.close()

        self.assertEqual(response["error"]["code"], "delivery_closed")
        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])

    def test_stale_listener_final_is_withheld_before_planner(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))
        daemon._listener_generation = 7

        daemon._handle_transcript(
            FinalTranscript(
                "voice:old:1",
                "Secretary tell one run tests",
                listener_generation=6,
            )
        )

        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "listener_authority_revoked",
        )

    def test_stop_timeout_blocks_listener_replacement(self) -> None:
        daemon, _, _ = make_daemon(planned("no_action"))

        class StuckListener:
            running = True
            listener_generation = -1
            last_error = None

            @staticmethod
            def stop() -> bool:
                return False

        daemon.voice = StuckListener()
        before = daemon._listener_generation
        with patch("voicerdr.daemon.VoiceListener") as listener_type:
            self.assertFalse(daemon._start_voice(wait_secs=0))

        listener_type.assert_not_called()
        self.assertIsInstance(daemon.voice, StuckListener)
        self.assertGreater(daemon._listener_generation, before)

    def test_replay_transaction_reloads_shared_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
            first = Daemon.__new__(Daemon)
            first.paths = paths
            first._consumed_utterance_ids = first._initialize_replay_ledger()
            second = Daemon.__new__(Daemon)
            second.paths = paths
            second._consumed_utterance_ids = set(first._consumed_utterance_ids)

            self.assertTrue(first._consume_utterance_id("typed:first"))
            self.assertTrue(second._consume_utterance_id("typed:second"))

            self.assertEqual(
                second._load_consumed_utterances(),
                {"typed:first", "typed:second"},
            )

    def test_replay_unlock_error_latches_fail_closed_before_planner(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))
        real_flock = fcntl.flock
        calls = 0

        def fail_unlock(fd: int, operation: int) -> None:
            nonlocal calls
            calls += 1
            if operation == fcntl.LOCK_UN:
                raise OSError("unlock failed")
            real_flock(fd, operation)

        with patch("voicerdr.daemon.fcntl.flock", side_effect=fail_unlock):
            daemon._handle_transcript(
                FinalTranscript(
                    "typed:unlock-error",
                    "Secretary do nothing",
                    origin="typed",
                )
            )

        self.assertGreaterEqual(calls, 2)
        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])
        self.assertIn("unlock failed", daemon._replay_ledger_error)
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "replay_ledger_unavailable",
        )

    def test_replay_lock_close_error_is_normalized_and_latched(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))
        real_open = os.open
        real_close = os.close
        lock_fd: int | None = None

        def tracked_open(path: object, flags: int, *args: object) -> int:
            nonlocal lock_fd
            fd = real_open(path, flags, *args)
            if Path(path) == daemon.paths.utterance_ledger_lock:
                lock_fd = fd
            return fd

        def fail_lock_close(fd: int) -> None:
            real_close(fd)
            if fd == lock_fd:
                raise OSError("close failed")

        with (
            patch("voicerdr.daemon.os.open", side_effect=tracked_open),
            patch("voicerdr.daemon.os.close", side_effect=fail_lock_close),
        ):
            daemon._handle_transcript(
                FinalTranscript(
                    "typed:close-error",
                    "Secretary do nothing",
                    origin="typed",
                )
            )

        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])
        self.assertIn("close failed", daemon._replay_ledger_error)
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "replay_ledger_unavailable",
        )

    def test_new_state_directory_chain_fsyncs_each_parent_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(
                root,
                root / "config",
                root / "one" / "two" / "state",
                "herdr",
                None,
            )
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths
            real_open = os.open
            real_fsync = os.fsync
            opened: dict[int, Path] = {}
            fsynced: list[Path] = []

            def tracked_open(path: object, flags: int, *args: object) -> int:
                fd = real_open(path, flags, *args)
                opened[fd] = Path(path)
                return fd

            def tracked_fsync(fd: int) -> None:
                if fd in opened:
                    fsynced.append(opened[fd])
                real_fsync(fd)

            with (
                patch("voicerdr.daemon.os.open", side_effect=tracked_open),
                patch("voicerdr.daemon.os.fsync", side_effect=tracked_fsync),
            ):
                daemon._initialize_replay_ledger()

            self.assertIn(root, fsynced)
            self.assertIn(root / "one", fsynced)
            self.assertIn(root / "one" / "two", fsynced)
            self.assertIn(paths.utterance_ledger_lock, fsynced)

    def test_process_lifetime_daemon_lock_rejects_second_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
            first = Daemon.__new__(Daemon)
            first.paths = paths
            first._daemon_lock_fd = None
            second = Daemon.__new__(Daemon)
            second.paths = paths
            second._daemon_lock_fd = None
            first._acquire_daemon_lock()
            try:
                with self.assertRaisesRegex(RuntimeError, "another voicerdr daemon"):
                    second._acquire_daemon_lock()
            finally:
                first._release_daemon_lock()

    def test_daemon_authority_creation_fsyncs_each_new_parent_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(
                root,
                root / "config",
                root / "one" / "two" / "state",
                "herdr",
                None,
            )
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths
            daemon._daemon_lock_fd = None
            real_open = os.open
            real_fsync = os.fsync
            opened: dict[int, Path] = {}
            fsynced: list[Path] = []

            def tracked_open(path: object, flags: int, *args: object) -> int:
                fd = real_open(path, flags, *args)
                opened[fd] = Path(path)
                return fd

            def tracked_fsync(fd: int) -> None:
                if fd in opened:
                    fsynced.append(opened[fd])
                real_fsync(fd)

            with (
                patch("voicerdr.daemon.os.open", side_effect=tracked_open),
                patch("voicerdr.daemon.os.fsync", side_effect=tracked_fsync),
            ):
                daemon._acquire_daemon_lock()
            try:
                self.assertIn(root, fsynced)
                self.assertIn(root / "one", fsynced)
                self.assertIn(root / "one" / "two", fsynced)
                self.assertIn(paths.daemon_lock, fsynced)
            finally:
                daemon._release_daemon_lock()

    def test_losing_direct_daemon_contender_cannot_seed_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
            owner = Daemon.__new__(Daemon)
            owner.paths = paths
            owner._daemon_lock_fd = None
            owner._acquire_daemon_lock()
            try:
                config = AppConfig.load(paths, seed=False)
                contender = Daemon(paths, config)
                with (
                    patch("voicerdr.daemon.seed_config_files") as seed,
                    patch.object(
                        contender, "_ensure_state_dir_durable"
                    ) as prepare_state,
                    self.assertRaisesRegex(RuntimeError, "another voicerdr daemon"),
                ):
                    contender.run_forever()
                seed.assert_not_called()
                prepare_state.assert_not_called()
                self.assertFalse(paths.config_dir.exists())
            finally:
                owner._release_daemon_lock()

    def test_direct_daemon_cli_loads_configuration_read_only(self) -> None:
        from voicerdr.cli import main

        paths = Mock()
        config = Mock()
        with (
            patch("voicerdr.cli.resolve_paths", return_value=paths),
            patch("voicerdr.cli.AppConfig.load", return_value=config) as load,
            patch("voicerdr.cli.run_daemon", return_value=23) as run,
        ):
            result = main(["daemon", "--foreground"])

        self.assertEqual(result, 23)
        load.assert_called_once_with(paths, seed=False)
        run.assert_called_once_with(paths, config)

    def test_daemon_lock_rejects_a_concurrent_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
            owner = Daemon.__new__(Daemon)
            owner.paths = paths
            owner._daemon_lock_fd = None
            owner._acquire_daemon_lock()
            program = """
import sys
from pathlib import Path
from voicerdr.daemon import Daemon
from voicerdr.paths import RuntimePaths
d=Daemon.__new__(Daemon)
r=Path(sys.argv[1])
d.paths=RuntimePaths(r,r/'config',r/'state','herdr',None)
d._daemon_lock_fd=None
try:
    d._acquire_daemon_lock()
except RuntimeError:
    print('blocked')
else:
    print('unsafe-acquired')
    d._release_daemon_lock()
"""
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", program, str(root)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            finally:
                owner._release_daemon_lock()

            self.assertEqual(completed.stdout.strip(), "blocked")

    def test_bind_control_never_unlinks_a_live_socket(self) -> None:
        daemon, _, _ = make_daemon(planned("no_action"))
        live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        live.bind(str(daemon.paths.control_socket))
        live.listen(1)
        try:
            with self.assertRaisesRegex(RuntimeError, "already live"):
                daemon._bind_control()
            self.assertTrue(daemon.paths.control_socket.exists())
        finally:
            live.close()
            daemon.paths.control_socket.unlink(missing_ok=True)

    def test_bind_control_rejects_symlink_and_regular_file(self) -> None:
        for kind in ("symlink", "file"):
            with self.subTest(kind=kind):
                daemon, _, _ = make_daemon(planned("no_action"))
                path = daemon.paths.control_socket
                if kind == "symlink":
                    target = daemon.paths.state_dir / "target"
                    target.write_text("not a socket")
                    path.symlink_to(target)
                else:
                    path.write_text("not a socket")
                with self.assertRaisesRegex(RuntimeError, "not an owned socket"):
                    daemon._bind_control()
                self.assertTrue(
                    path.is_symlink() if kind == "symlink" else path.is_file()
                )

    def test_normal_stale_socket_is_quarantined_before_rebind(self) -> None:
        daemon, _, _ = make_daemon(planned("no_action"))
        path = daemon.paths.control_socket
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(path))
        stale.close()

        daemon._bind_control()
        try:
            self.assertTrue(path.is_socket())
            current = os.lstat(path)
            self.assertEqual(
                daemon._control_socket_identity,
                (current.st_dev, current.st_ino),
            )
            self.assertEqual(list(path.parent.glob("control.sock.quarantine.*")), [])
        finally:
            assert daemon._server is not None
            daemon._server.close()
            daemon._unlink_owned_control_socket()

    def test_shutdown_unlinks_only_the_socket_inode_it_bound(self) -> None:
        daemon, _, _ = make_daemon(planned("no_action"))
        daemon._bind_control()
        assert daemon._server is not None
        daemon._server.close()
        daemon.paths.control_socket.unlink()
        replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        replacement.bind(str(daemon.paths.control_socket))
        try:
            daemon._unlink_owned_control_socket()
            self.assertTrue(daemon.paths.control_socket.is_socket())
        finally:
            replacement.close()
            daemon.paths.control_socket.unlink(missing_ok=True)

    def test_stale_socket_inode_is_rechecked_before_unlink(self) -> None:
        daemon, _, _ = make_daemon(planned("no_action"))
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(daemon.paths.control_socket))
        stale.close()
        replacement: socket.socket | None = None

        def replace_during_probe(_path: object) -> bool:
            nonlocal replacement
            daemon.paths.control_socket.unlink()
            replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            replacement.bind(str(daemon.paths.control_socket))
            return False

        try:
            with (
                patch.object(
                    daemon, "_socket_is_live", side_effect=replace_during_probe
                ),
                self.assertRaisesRegex(RuntimeError, "changed during stale check"),
            ):
                daemon._bind_control()
            self.assertTrue(daemon.paths.control_socket.is_socket())
        finally:
            if replacement is not None:
                replacement.close()
            daemon.paths.control_socket.unlink(missing_ok=True)

    def test_stale_socket_replacement_during_quarantine_is_restored(self) -> None:
        daemon, _, _ = make_daemon(planned("no_action"))
        path = daemon.paths.control_socket
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(path))
        stale.close()
        replacement: socket.socket | None = None
        real_rename = os.rename

        def race_rename(source: object, destination: object) -> None:
            nonlocal replacement
            if os.fspath(source) == os.fspath(path):
                path.unlink()
                replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                replacement.bind(str(path))
            real_rename(source, destination)

        try:
            with (
                patch("voicerdr.daemon.os.rename", side_effect=race_rename),
                self.assertRaisesRegex(RuntimeError, "changed during quarantine"),
            ):
                daemon._bind_control()
            self.assertTrue(path.is_socket())
        finally:
            if replacement is not None:
                replacement.close()
            path.unlink(missing_ok=True)

    def test_shutdown_quarantine_restores_racing_replacement(self) -> None:
        daemon, _, _ = make_daemon(planned("no_action"))
        path = daemon.paths.control_socket
        daemon._bind_control()
        assert daemon._server is not None
        replacement: socket.socket | None = None
        real_rename = os.rename

        def race_rename(source: object, destination: object) -> None:
            nonlocal replacement
            if os.fspath(source) == os.fspath(path):
                path.unlink()
                replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                replacement.bind(str(path))
            real_rename(source, destination)

        try:
            with patch("voicerdr.daemon.os.rename", side_effect=race_rename):
                daemon._unlink_owned_control_socket()
            self.assertTrue(path.is_socket())
        finally:
            daemon._server.close()
            if replacement is not None:
                replacement.close()
            path.unlink(missing_ok=True)

    def test_normal_owned_socket_cleanup_removes_quarantine(self) -> None:
        daemon, _, _ = make_daemon(planned("no_action"))
        path = daemon.paths.control_socket
        daemon._bind_control()
        assert daemon._server is not None
        daemon._server.close()

        daemon._unlink_owned_control_socket()

        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.glob("control.sock.quarantine.*")), [])

    def test_expired_clarification_loses_wake_free_authority(self) -> None:
        daemon, herdr, secretary = make_daemon(
            planned("clarification", clarification="Which agent?")
        )
        daemon._handle_transcript("Secretary tell frontend run tests")
        assert daemon.pending_clarification
        daemon.pending_clarification["expires_at"] = time.monotonic() - 1
        calls_before = len(secretary.plan_calls)

        daemon._handle_transcript("reviewer")

        self.assertIsNone(daemon.pending_clarification)
        self.assertEqual(len(secretary.plan_calls), calls_before)
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "wake_not_matched")

    def test_status_surfaces_replay_ledger_not_ready(self) -> None:
        daemon, _, _ = make_daemon(planned("no_action"))
        daemon._replay_ledger_ready = False
        daemon._replay_ledger_error = "permission denied"
        daemon._started_at = time.time()
        daemon.activity_workspace = None
        daemon._activity_status = {}
        daemon._mic_transition_count = 1
        daemon.voice = None
        daemon.speaker = Mock(last_error=None, last_spoken=None)
        daemon.subscriber = None
        daemon.status_watcher = None

        status = daemon.status()

        self.assertFalse(status["replay_ledger"]["ready"])
        self.assertEqual(status["replay_ledger"]["error"], "permission denied")
        self.assertTrue(status["mic_transition_pending"])
        self.assertEqual(status["mic_preference"]["mode"], daemon.policy.mic_mode)
        self.assertFalse(status["mic_preference"]["explicit"])
        self.assertTrue(
            status["mic_preference"]["path"].endswith("mic_preference.json")
        )

    def test_direct_prompt_rpc_cannot_reach_agent_prompt(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))

        response = daemon._dispatch(
            ControlRequest(
                "rpc-1",
                "prompt",
                {"space": "frontend", "agent": "Frontend", "text": "run tests"},
            )
        )

        self.assertIn("verified_planner_required", response)
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(secretary.plan_calls, [])

    def test_ingest_rpc_delivers_only_after_planner_and_verifier(self) -> None:
        daemon, herdr, secretary = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="run tests",
            )
        )

        response = daemon._dispatch(
            ControlRequest(
                "rpc-2",
                "ingest_transcript",
                {"utterance_id": "typed:rpc-2", "text": "Secretary tell one run tests"},
            )
        )

        self.assertIn('"ok": true', response)
        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])
        self.assertEqual(len(secretary.plan_calls), 1)

    def test_ingest_rpc_requires_client_utterance_identity(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))

        response = daemon._dispatch(
            ControlRequest(
                "rpc-no-id",
                "ingest_transcript",
                {"text": "Secretary tell one run tests"},
            )
        )

        self.assertIn("utterance_id is required", response)
        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])

    def test_control_client_reuses_generated_identity_after_timeout(self) -> None:
        client = ControlClient(Path("/unused"))
        client.call = Mock(
            side_effect=[ControlClientError("timed out"), {"accepted": True}]
        )

        result = client.ingest_transcript("Secretary summarize two", retries=1)

        self.assertEqual(result, {"accepted": True})
        first = client.call.call_args_list[0].args[1]
        second = client.call.call_args_list[1].args[1]
        self.assertEqual(first["utterance_id"], second["utterance_id"])
        self.assertTrue(first["utterance_id"].startswith("typed:"))

    def test_same_text_on_distinct_vad_turns_is_not_collapsed(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        original_plan = secretary.plan
        counter_lock = threading.Lock()
        active = 0
        max_active = 0

        def observed_plan(utterance: str, **kwargs: object) -> BoundIntentPlan:
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            try:
                return original_plan(utterance, **kwargs)
            finally:
                with counter_lock:
                    active -= 1

        secretary.plan = observed_plan
        finals = [
            FinalTranscript("voice:session:1", "Secretary tell one run tests"),
            FinalTranscript("voice:session:2", "Secretary tell one run tests"),
        ]

        threads = [
            threading.Thread(target=daemon._handle_transcript, args=(item,))
            for item in finals
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(
            herdr.prompts,
            [("w1:p1", "run tests", False), ("w1:p1", "run tests", False)],
        )
        self.assertEqual(max_active, 1)

    def test_delayed_retry_identity_is_consumed_once(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        final = FinalTranscript("typed:stable-retry", "Secretary tell one run tests")

        daemon._handle_transcript(final)
        secretary.plan_result = planned("no_action")
        daemon._handle_transcript(
            FinalTranscript("voice:session:later", "Secretary Samurais, too.")
        )
        daemon._handle_transcript(final)

        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])
        self.assertEqual(len(secretary.plan_calls), 2)
        self.assertEqual(
            daemon.last_voice_action["result"]["code"], "utterance_already_consumed"
        )

    def test_consumed_typed_identity_survives_daemon_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
            first = Daemon.__new__(Daemon)
            first.paths = paths
            first._consumed_utterance_ids = first._initialize_replay_ledger()
            self.assertTrue(first._consume_utterance_id("typed:persisted"))

            restarted = Daemon.__new__(Daemon)
            restarted.paths = paths
            restarted._consumed_utterance_ids = restarted._load_consumed_utterances()
            self.assertFalse(restarted._consume_utterance_id("typed:persisted"))

    def test_absent_ledger_is_created_with_file_and_directory_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths
            real_open = os.open
            real_fsync = os.fsync
            opened: dict[int, Path] = {}
            fsynced: list[Path] = []

            def tracked_open(path: os.PathLike[str], flags: int, *args: int) -> int:
                fd = real_open(path, flags, *args)
                opened[fd] = Path(path)
                return fd

            def tracked_fsync(fd: int) -> None:
                if fd in opened:
                    fsynced.append(opened[fd])
                real_fsync(fd)

            with (
                patch("voicerdr.daemon.os.open", side_effect=tracked_open),
                patch("voicerdr.daemon.os.fsync", side_effect=tracked_fsync),
            ):
                consumed = daemon._initialize_replay_ledger()

            self.assertEqual(consumed, set())
            self.assertEqual(paths.utterance_ledger.read_text(), "[]\n")
            self.assertIn(paths.state_dir.parent, fsynced)
            self.assertIn(paths.state_dir, fsynced)
            self.assertTrue(any(path.name.endswith(".tmp") for path in fsynced))

    def test_corrupt_ledger_withholds_before_planner(self) -> None:
        daemon, herdr, secretary = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="run tests",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daemon.paths = RuntimePaths(
                root, root / "config", root / "state", "herdr", None
            )
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.paths.utterance_ledger.write_text("{corrupt")
            try:
                daemon._consumed_utterance_ids = daemon._initialize_replay_ledger()
            except ReplayLedgerError as exc:
                daemon._replay_ledger_error = str(exc)

            daemon._handle_transcript(
                FinalTranscript("typed:corrupt", "Secretary tell one run tests")
            )

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "replay_ledger_unavailable",
        )

    def test_failed_atomic_replace_preserves_old_ledger_and_blocks_planner(
        self,
    ) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daemon.paths = RuntimePaths(
                root, root / "config", root / "state", "herdr", None
            )
            daemon._replay_ledger_error = None
            daemon._consumed_utterance_ids = daemon._initialize_replay_ledger()
            with patch("voicerdr.daemon.os.replace", side_effect=OSError("ENOSPC")):
                daemon._handle_transcript(
                    FinalTranscript("typed:enospc", "Secretary do nothing")
                )

            self.assertEqual(daemon.paths.utterance_ledger.read_text(), "[]\n")
        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])
        self.assertNotIn("typed:enospc", daemon._consumed_utterance_ids)

    def test_failed_file_fsync_withholds_before_planner(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daemon.paths = RuntimePaths(
                root, root / "config", root / "state", "herdr", None
            )
            daemon._replay_ledger_error = None
            daemon._consumed_utterance_ids = daemon._initialize_replay_ledger()
            with patch(
                "voicerdr.daemon.os.fsync", side_effect=PermissionError("denied")
            ):
                daemon._handle_transcript(
                    FinalTranscript("typed:fsync", "Secretary summarize two")
                )

        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "replay_ledger_unavailable",
        )

    def test_failed_directory_fsync_latches_closed_even_after_replace(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            daemon.paths = RuntimePaths(
                root, root / "config", root / "state", "herdr", None
            )
            daemon._replay_ledger_error = None
            daemon._consumed_utterance_ids = daemon._initialize_replay_ledger()
            real_fsync = os.fsync
            call_count = 0

            def fail_directory_fsync(fd: int) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("directory fsync failed")
                real_fsync(fd)

            with patch("voicerdr.daemon.os.fsync", side_effect=fail_directory_fsync):
                daemon._handle_transcript(
                    FinalTranscript("typed:dir-fsync", "Secretary summarize two")
                )

            durable_file = daemon.paths.utterance_ledger.read_text()

        self.assertIn("typed:dir-fsync", durable_file)
        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])
        self.assertIsNotNone(daemon._replay_ledger_error)

    def test_unmatched_voice_final_is_visibly_withheld_before_planner(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))

        daemon._handle_transcript(
            FinalTranscript(
                "voice:session:orphan:1",
                "Secretary tell one run tests",
                provenance_valid=False,
                provenance_error="unknown VAD turn token",
            )
        )

        self.assertEqual(secretary.plan_calls, [])
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "unmatched_final_provenance",
        )

    def test_planner_and_verifier_cannot_authorize_unsupported_prompt_text(
        self,
    ) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="delete everything",
                evidence=IntentEvidence("post_wake_content", "run tests"),
            )
        )

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"],
            "plan_blocked",
        )
        self.assertFalse(
            any(event["event"] == "verification" for event in daemon.activity_events)
        )

    def test_post_send_focus_and_notification_oserrors_preserve_sent_truth(
        self,
    ) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="run tests",
            )
        )
        daemon.config.focus_on_prompt = True
        daemon.config.speak_acks = False
        daemon._notify = Daemon._notify.__get__(daemon)
        herdr.focus_error = OSError("focus failed")
        herdr.notification_error = OSError("notify failed")

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])
        self.assertTrue(daemon.last_voice_action["result"]["sent"])
        self.assertTrue(daemon.last_voice_action["result"]["ok"])

    def test_post_send_activity_exception_preserves_sent_truth(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="run tests",
            )
        )
        original_record = daemon._record_activity

        def fail_after_send(event: str, **fields: object) -> None:
            if event in {"prompt_sent", "action"}:
                raise OSError("activity storage failed")
            original_record(event, **fields)

        daemon._record_activity = fail_after_send

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])
        self.assertTrue(daemon.last_voice_action["result"]["sent"])

    def test_post_send_batch_notification_oserror_preserves_sent_truth(self) -> None:
        status = planned("status", target=IntentTarget("w2", "w2:p1"))
        prompt = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, _ = make_daemon(planned("batch", actions=(status, prompt)))
        daemon.summarize_agent = Mock(return_value="Backend is working.")
        daemon.config.focus_on_prompt = True
        daemon.config.speak_acks = False
        daemon._notify = Daemon._notify.__get__(daemon)
        herdr.focus_error = OSError("focus failed")
        herdr.notification_error = OSError("notify failed")

        daemon._handle_transcript("Secretary summarize two and tell one run tests")

        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])
        self.assertTrue(daemon.last_voice_action["result"]["sent"])
        self.assertNotIn("nothing was sent", str(daemon.last_voice_action).casefold())

    def test_workspace_only_prompt_cannot_choose_among_multiple_agents(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="run tests",
            )
        )
        herdr.agents.append(
            {
                "workspace_id": "w1",
                "pane_id": "w1:p2",
                "terminal_title_stripped": "Security review",
                "agent_status": "idle",
            }
        )

        daemon._handle_transcript("Secretary tell frontend run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")
        self.assertIn("Last rejected proposal", daemon.last_voice_action["message"])
        self.assertIn("Nothing was sent", daemon.last_voice_action["message"])
        self.assertFalse(
            any(event["event"] == "verifying" for event in daemon.activity_events)
        )

    def test_explicit_agent_evidence_selects_exact_pane_in_multi_agent_space(
        self,
    ) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p2"),
                message="run tests",
                agent_evidence="Security review",
                workspace_evidence="frontend",
            )
        )
        herdr.agents.append(
            {
                "workspace_id": "w1",
                "pane_id": "w1:p2",
                "terminal_title_stripped": "Security review",
                "agent_status": "idle",
            }
        )

        daemon._handle_transcript("Secretary tell frontend Security review run tests")

        self.assertEqual(herdr.prompts, [("w1:p2", "run tests", False)])

    def test_pending_clarification_survives_planner_outage_and_no_action(self) -> None:
        daemon, herdr, secretary = make_daemon(
            planned("clarification", clarification="Which agent?")
        )
        daemon._handle_transcript("Secretary tell frontend run tests")
        pending = daemon.pending_clarification
        assert pending is not None

        secretary.plan_result = SecretaryError("offline")
        daemon._handle_transcript("reviewer")
        self.assertEqual(
            daemon.pending_clarification["clarification_id"],
            pending["clarification_id"],
        )
        self.assertIn("run tests", daemon.pending_clarification["request_text"])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")

        secretary.plan_result = SecretaryError("malformed JSON")
        daemon._handle_transcript("reviewer again")
        self.assertEqual(
            daemon.pending_clarification["clarification_id"],
            pending["clarification_id"],
        )
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")

        secretary.plan_result = planned("no_action")
        daemon._handle_transcript("background noise")
        self.assertEqual(
            daemon.pending_clarification["clarification_id"],
            pending["clarification_id"],
        )
        self.assertEqual(daemon.last_voice_action["result"]["code"], "no_action")

        secretary.plan_result = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
            evidence=IntentEvidence("clarification_request", "run tests"),
        )
        secretary.verification = VerificationResult(
            approved=False,
            action_kind="agent_prompt",
            target={"workspace_id": "w1", "pane_id": "w1:p1"},
            message="run tests",
            reason="still ambiguous",
            reason_kind=VERIFICATION_REASON_ACTION,
            action_exact=False,
        )
        daemon._handle_transcript("one")
        self.assertEqual(
            daemon.pending_clarification["clarification_id"],
            pending["clarification_id"],
        )
        self.assertEqual(
            daemon.last_voice_action["result"]["code"], "verifier_rejected"
        )
        self.assertEqual(herdr.prompts, [])

    def test_prompt_preflight_rejects_every_non_allowlisted_agent_state(self) -> None:
        for state in (None, "", "unknown", "starting", "blocked", " working ", 7):
            with self.subTest(state=state):
                daemon, herdr, _ = make_daemon(
                    planned(
                        "agent_prompt",
                        target=IntentTarget("w1", "w1:p1"),
                        message="run tests",
                    )
                )
                if state is None:
                    herdr.agents[0].pop("agent_status", None)
                else:
                    herdr.agents[0]["agent_status"] = state
                daemon._handle_transcript("Secretary tell one run tests")
                self.assertEqual(herdr.prompts, [])
                self.assertEqual(
                    daemon.last_voice_action["result"]["code"], "plan_blocked"
                )

    def test_working_agent_state_remains_explicitly_safe(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="run tests",
            )
        )
        herdr.agents[0]["agent_status"] = "working"

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])

    def test_planner_digest_mismatch_fails_closed(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        secretary.plan = Mock(
            side_effect=lambda _utterance, **kwargs: BoundIntentPlan(
                "0" * 64, str(kwargs["catalog_digest"]), plan
            )
        )

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")

    def test_verifier_digest_mismatch_cannot_authorize_delivery(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        secretary.verification = VerificationResult(
            approved=True,
            action_kind="agent_prompt",
            target={"workspace_id": "w1", "pane_id": "w1:p1"},
            message="run tests",
            reason="claims approval",
            utterance_digest="0" * 64,
            catalog_digest="0" * 64,
            plan_digest="0" * 64,
        )

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "verifier_error")

    def test_status_timeout_before_sole_batch_prompt_sends_nothing(self) -> None:
        status = planned("status", target=IntentTarget("w2", "w2:p1"))
        prompt = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, _ = make_daemon(planned("batch", actions=(prompt, status)))
        daemon.summarize_agent = Mock(side_effect=SecretaryError("status timeout"))

        daemon._handle_transcript("Secretary send one run tests and summarize two")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "execution_error")

    def test_batch_transport_timeout_after_possible_send_reports_unknown(self) -> None:
        status = planned("status", target=IntentTarget("w2", "w2:p1"))
        prompt = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, _ = make_daemon(planned("batch", actions=(status, prompt)))
        daemon.summarize_agent = Mock(return_value="Backend is working.")
        herdr.prompt_error_after_send = HerdrError("transport timeout")

        daemon._handle_transcript("Secretary summarize two and tell one run tests")

        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])
        result = daemon.last_voice_action["result"]
        self.assertIsNone(result["sent"])
        self.assertIn("unknown", result["summary"])
        self.assertNotIn("nothing was sent", str(result).casefold())

    def test_summarized_too_is_not_delivered_as_prompt(self) -> None:
        daemon, herdr, secretary = make_daemon(
            planned(
                "clarification",
                clarification="Did you want the status of workspace two?",
            )
        )

        daemon._handle_transcript("Secretary summarized, too.")

        self.assertEqual(secretary.plan_calls[0]["utterance"], "summarized, too.")
        self.assertEqual(herdr.prompts, [])
        self.assertFalse(daemon.last_voice_action["result"]["sent"])

    def test_samurais_too_is_not_delivered_as_prompt(self) -> None:
        daemon, herdr, secretary = make_daemon(planned("no_action"))

        daemon._handle_transcript("Secretary Samurais, too.")

        self.assertEqual(secretary.plan_calls[0]["utterance"], "Samurais, too.")
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["action_kind"], "no_action")

    def test_ambiguous_planner_decision_asks_and_does_not_send(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned("clarification", clarification="Which workspace and pane?")
        )

        daemon._handle_transcript("Secretary send that over there")

        self.assertEqual(herdr.prompts, [])
        self.assertIsNotNone(daemon.pending_clarification)
        self.assertFalse(daemon.last_voice_action["result"]["sent"])

    def test_llm_unavailable_never_falls_back_to_focused_workspace(self) -> None:
        daemon, herdr, _ = make_daemon(SecretaryError("LLM unreachable"))

        daemon._handle_transcript("Secretary run the tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")
        self.assertFalse(daemon.last_voice_action["result"]["sent"])

    def test_malformed_planner_output_never_reaches_agent(self) -> None:
        daemon, herdr, _ = make_daemon(SecretaryError("malformed intent plan JSON"))

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")
        self.assertFalse(daemon.last_voice_action["result"]["sent"])

    def test_missing_or_fuzzy_target_id_is_withheld_even_with_one_agent(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("frontend", "Frontend"),
                message="run tests",
                workspace_evidence="frontend",
            )
        )

        daemon._handle_transcript("Secretary tell frontend run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "plan_blocked")

    def test_low_confidence_prompt_is_withheld_before_verification(self) -> None:
        daemon, herdr, secretary = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="run tests",
                confidence=0.5,
            )
        )

        daemon._handle_transcript("Secretary maybe tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertIsNone(secretary.verification)
        self.assertIn("confidence", daemon.last_voice_action["message"])

    def test_verifier_disagreement_is_visible_and_not_sent(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        secretary.verification = VerificationResult(
            approved=False,
            action_kind="agent_prompt",
            target={"workspace_id": "w1", "pane_id": "w1:p1"},
            message="run tests",
            reason="the destination was ambiguous",
            target_exact=False,
        )

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["result"]["code"], "verifier_rejected"
        )
        self.assertFalse(daemon.last_voice_action["result"]["sent"])
        self.assertIn("Nothing was sent", daemon.last_voice_action["message"])
        withheld = [
            event for event in daemon.activity_events if event["event"] == "withheld"
        ]
        self.assertEqual(withheld[-1]["sent"], False)

    def test_verifier_unavailable_is_visible_and_not_sent(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        secretary.verification = SecretaryError("verifier unavailable")

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "verifier_error")
        self.assertFalse(daemon.last_voice_action["result"]["sent"])

    def test_verifier_transport_failure_is_not_retried(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        calls = 0

        def unavailable(**_kwargs: object) -> VerificationResult:
            nonlocal calls
            calls += 1
            raise SecretaryTransportError("endpoint unavailable")

        secretary.verify_prompt = unavailable
        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(calls, 1)
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "verifier_error")

    def test_catalog_change_after_verification_blocks_send(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
        )
        daemon, herdr, secretary = make_daemon(plan)
        secretary.on_verify = lambda: herdr.workspaces[0].update(label="#1 renamed")

        daemon._handle_transcript("Secretary tell one run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "catalog_changed")

    def test_invalid_batch_child_prevents_all_execution(self) -> None:
        first = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run frontend tests",
        )
        invalid = planned(
            "agent_prompt",
            target=IntentTarget("missing", "missing:p1"),
            message="run backend tests",
        )
        daemon, herdr, _ = make_daemon(planned("batch", actions=(first, invalid)))

        daemon._handle_transcript("Secretary tell one tests and tell two tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")

    def test_later_batch_verifier_rejection_prevents_earlier_prompt(self) -> None:
        first = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run frontend tests",
        )
        second = planned(
            "agent_prompt",
            target=IntentTarget("w2", "w2:p1"),
            message="run backend tests",
        )
        daemon, herdr, secretary = make_daemon(
            planned("batch", actions=(first, second))
        )
        secretary.verify_prompt = Mock(
            side_effect=[
                VerificationResult(
                    True,
                    "agent_prompt",
                    {"workspace_id": "w1", "pane_id": "w1:p1"},
                    "run frontend tests",
                    "exact",
                ),
                VerificationResult(
                    False,
                    "agent_prompt",
                    {"workspace_id": "w2", "pane_id": "w2:p1"},
                    "run backend tests",
                    "second clause is ambiguous",
                ),
            ]
        )

        daemon._handle_transcript("Secretary send two messages")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "planner_error")
        secretary.verify_prompt.assert_not_called()

    def test_status_by_index_uses_llm_selected_exact_catalog_pane(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned("status", target=IntentTarget("w2", "w2:p1"))
        )
        daemon.summarize_agent = Mock(return_value="Backend tests are passing.")

        daemon._handle_transcript("Secretary summarize two")

        daemon.summarize_agent.assert_called_once_with("w2:p1")
        self.assertEqual(herdr.prompts, [])
        self.assertFalse(daemon.last_voice_action["result"]["sent"])

    def test_numeric_status_with_redundant_agent_evidence_binds_sole_pane(
        self,
    ) -> None:
        cases = (
            ("Secretary summarized one.", "one", "w1", "w1:p1"),
            ("Secretary summarized one", "one", "w1", "w1:p1"),
            ("Secretary summarize one.", "one", "w1", "w1:p1"),
            ("Secretary summarize one", "one", "w1", "w1:p1"),
            ("Secretary summarize 1.", "1", "w1", "w1:p1"),
            ("Secretary summarize #1.", "#1", "w1", "w1:p1"),
            ("Secretary summarize two.", "two", "w2", "w2:p1"),
            ("Secretary summarize 2.", "2", "w2", "w2:p1"),
            ("Secretary summarize #2.", "#2", "w2", "w2:p1"),
        )
        for transcript, quote, workspace_id, pane_id in cases:
            with self.subTest(transcript=transcript):
                plan = planned(
                    "status",
                    target=IntentTarget(workspace_id, pane_id),
                    workspace_evidence=quote,
                    agent_evidence=quote,
                )
                daemon, herdr, secretary = make_daemon(plan)
                herdr.workspaces[0]["label"] = "#1 whatshot"
                herdr.workspaces[1]["label"] = "#2 poot"
                herdr.agents[0]["agent_status"] = "working"
                herdr.agents[1]["agent_status"] = "working"
                daemon.summarize_agent = Mock(return_value=f"Summary for {pane_id}.")

                daemon._handle_transcript(transcript)

                daemon.summarize_agent.assert_called_once_with(pane_id)
                self.assertEqual(herdr.prompts, [])
                self.assertEqual(len(secretary.plan_calls), 1)
                self.assertEqual(
                    secretary.plan_calls[0]["utterance"],
                    transcript.removeprefix("Secretary "),
                )
                self.assertEqual(
                    daemon.last_voice_action["workspace_evidence"]["quote"], quote
                )
                self.assertEqual(
                    daemon.last_voice_action["agent_evidence"]["quote"], quote
                )
                self.assertEqual(
                    daemon.last_voice_action["catalog_digest"],
                    secretary.plan_calls[0]["catalog_digest"],
                )
                self.assertEqual(
                    daemon.last_voice_action["utterance_digest"],
                    secretary.plan_calls[0]["utterance_digest"],
                )

    def test_corrective_plan_can_retain_redundant_numeric_agent_evidence(self) -> None:
        combined = planned(
            "status",
            target=IntentTarget("w1", "w1:p1"),
            workspace_evidence="summarize one.",
            agent_evidence="one",
        )
        exact_number = planned(
            "status",
            target=IntentTarget("w1", "w1:p1"),
            workspace_evidence="one",
            agent_evidence="one",
        )
        daemon, herdr, secretary = make_daemon(combined)
        decisions = iter((combined, exact_number))

        def replan(utterance: str, **kwargs: object) -> BoundIntentPlan:
            secretary.plan_calls.append({"utterance": utterance, **kwargs})
            return BoundIntentPlan(
                str(kwargs["utterance_digest"]),
                str(kwargs["catalog_digest"]),
                next(decisions),
            )

        secretary.plan = replan
        daemon.summarize_agent = Mock(return_value="Frontend is working.")

        daemon._handle_transcript("Secretary summarize one.")

        daemon.summarize_agent.assert_called_once_with("w1:p1")
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(len(secretary.plan_calls), 2)
        rejection = next(
            event
            for event in daemon.activity_events
            if event["event"] == "planner_proposal_rejected"
        )
        self.assertIn("workspace", rejection["failure"])
        self.assertEqual(
            rejection["proposed_plan"]["workspace_evidence"]["quote"],
            "summarize one.",
        )

    def test_llm_declared_homophone_number_binds_frozen_catalog_row(self) -> None:
        plan = planned(
            "status",
            target=IntentTarget("w2", "w2:p1"),
            workspace_evidence=IntentEvidence(
                "post_wake_content", "too", catalog_number=2
            ),
            agent_evidence="too",
        )
        daemon, herdr, _ = make_daemon(plan)
        daemon.summarize_agent = Mock(return_value="Backend is working.")

        daemon._handle_transcript("Secretary summarized, too.")

        daemon.summarize_agent.assert_called_once_with("w2:p1")
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(
            daemon.last_voice_action["workspace_evidence"],
            {
                "source": "post_wake_content",
                "quote": "too",
                "catalog_number": 2,
            },
        )

    def test_llm_declared_number_is_sealed_through_prompt_verification(self) -> None:
        plan = planned(
            "agent_prompt",
            target=IntentTarget("w2", "w2:p1"),
            message="run tests",
            workspace_evidence=IntentEvidence(
                "post_wake_content", "too", catalog_number=2
            ),
            agent_evidence="too",
        )
        daemon, herdr, secretary = make_daemon(plan)

        daemon._handle_transcript("Secretary tell too run tests")

        self.assertEqual(herdr.prompts, [("w2:p1", "run tests", False)])
        self.assertEqual(
            secretary.plan_calls[0]["catalog_digest"],
            daemon.last_voice_action["catalog_digest"],
        )
        self.assertEqual(
            daemon.last_voice_action["workspace_evidence"]["catalog_number"], 2
        )

    def test_catalog_number_conflicting_with_exact_quote_is_rejected(self) -> None:
        daemon, herdr, secretary = make_daemon(
            planned(
                "status",
                target=IntentTarget("w2", "w2:p1"),
                workspace_evidence=IntentEvidence(
                    "post_wake_content", "one", catalog_number=2
                ),
                agent_evidence="one",
            )
        )
        daemon.summarize_agent = Mock()

        daemon._handle_transcript("Secretary summarize one.")

        daemon.summarize_agent.assert_not_called()
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(len(secretary.plan_calls), 3)
        self.assertIn("conflicts", daemon.last_voice_action["message"])

    def test_numeric_workspace_evidence_never_selects_among_multiple_panes(
        self,
    ) -> None:
        daemon, herdr, secretary = make_daemon(
            planned(
                "status",
                target=IntentTarget("w1", "w1:p1"),
                workspace_evidence="one",
                agent_evidence="one",
            )
        )
        herdr.agents.append(
            {
                "workspace_id": "w1",
                "pane_id": "w1:p2",
                "terminal_title_stripped": "Security review",
                "agent_status": "working",
            }
        )
        daemon.summarize_agent = Mock()

        daemon._handle_transcript("Secretary summarize one.")

        daemon.summarize_agent.assert_not_called()
        self.assertEqual(herdr.prompts, [])
        self.assertEqual(len(secretary.plan_calls), 3)
        self.assertIn(
            "uniquely identify one catalog pane",
            daemon.last_voice_action["message"],
        )
        rejected = [
            event
            for event in daemon.activity_events
            if event["event"] == "planner_proposal_rejected"
        ]
        self.assertEqual(len(rejected), 3)

    def test_numeric_evidence_tracks_frozen_task_worktree_numbers(self) -> None:
        daemon, herdr, secretary = make_daemon(
            planned(
                "status",
                target=IntentTarget("task-a", "task-a:p1"),
                workspace_evidence="one",
                agent_evidence="one",
            )
        )
        herdr.workspaces = [
            {
                "workspace_id": "task-a",
                "label": "#1 poot task-a",
                "number": 1,
                "worktree": {"repo_name": "poot", "branch": "task-a"},
            },
            {
                "workspace_id": "task-b",
                "label": "#2 poot task-b",
                "number": 2,
                "worktree": {"repo_name": "poot", "branch": "task-b"},
            },
        ]
        herdr.agents = [
            {
                "workspace_id": "task-a",
                "pane_id": "task-a:p1",
                "terminal_title_stripped": "Task A",
                "agent_status": "working",
            },
            {
                "workspace_id": "task-b",
                "pane_id": "task-b:p1",
                "terminal_title_stripped": "Task B",
                "agent_status": "working",
            },
        ]
        daemon.summarize_agent = Mock(return_value="Task summary.")

        daemon._handle_transcript("Secretary summarize one.")

        herdr.workspaces[0].update(label="#2 poot task-a", number=2)
        herdr.workspaces[1].update(label="#1 poot task-b", number=1)
        secretary.plan_result = planned(
            "status",
            target=IntentTarget("task-b", "task-b:p1"),
            workspace_evidence="one",
            agent_evidence="one",
        )
        daemon._handle_transcript("Secretary summarize one.")

        self.assertEqual(
            daemon.summarize_agent.call_args_list,
            [unittest.mock.call("task-a:p1"), unittest.mock.call("task-b:p1")],
        )
        first_catalog = secretary.plan_calls[0]["spaces"]
        second_catalog = secretary.plan_calls[1]["spaces"]
        self.assertEqual(
            [(row["workspace_id"], row["number"]) for row in first_catalog],
            [("task-a", 1), ("task-b", 2)],
        )
        self.assertEqual(
            [(row["workspace_id"], row["number"]) for row in second_catalog],
            [("task-b", 1), ("task-a", 2)],
        )
        self.assertNotEqual(
            secretary.plan_calls[0]["catalog_digest"],
            secretary.plan_calls[1]["catalog_digest"],
        )

    def test_activity_exposes_interpreting_verifying_and_exact_choice(self) -> None:
        daemon, herdr, _ = make_daemon(
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p1"),
                message="run tests",
            )
        )

        daemon._handle_transcript("Secretary tell one run tests")

        events = [event["event"] for event in daemon.activity_events]
        self.assertIn("interpreting", events)
        self.assertIn("intent_chosen", events)
        self.assertIn("verifying", events)
        self.assertIn("verification", events)
        self.assertIn("prompt_sent", events)
        sent = next(
            event for event in daemon.activity_events if event["event"] == "prompt_sent"
        )
        self.assertEqual(sent["target"], {"workspace_id": "w1", "pane_id": "w1:p1"})
        self.assertTrue(sent["sent"])
        self.assertEqual(herdr.prompts, [("w1:p1", "run tests", False)])


if __name__ == "__main__":
    unittest.main()
