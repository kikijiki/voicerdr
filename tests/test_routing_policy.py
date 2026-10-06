import atexit
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from voicerdr.config import AppConfig
from voicerdr.control_protocol import ControlRequest
from voicerdr.daemon import Daemon
from voicerdr.dictation import DictationBuffer
from voicerdr.herdr_client import HerdrClient, HerdrError
from voicerdr.intent import (
    BoundIntentPlan,
    IntentEvidence,
    IntentPlan,
    IntentTarget,
    IntentValidationError,
    plan_from_json,
)
from voicerdr.paths import RuntimePaths
from voicerdr.routing import resolve_route
from voicerdr.secretary import SecretaryClient, SecretaryError, VerificationResult
from voicerdr.talk_policy import TalkPolicy

WORKSPACES = [
    {"workspace_id": "w1", "label": "#1 frontend", "number": 1},
    {"workspace_id": "w2", "label": "#2 backend", "number": 2},
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
    reason: str = "clear mocked intent",
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
        reason=reason,
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
                    {"w1": "frontend", "w2": "backend"}.get(target.workspace_id)
                    if target
                    else ""
                ),
            )
            if action_kind in {"agent_prompt", "status"}
            else None
        ),
        unresolved_slots=unresolved_slots,
    )


def agent(
    pane_id: str,
    workspace_id: str,
    title: str,
    *,
    name: str | None = None,
    kind: str | None = None,
    status: str = "idle",
) -> dict[str, object]:
    result: dict[str, object] = {
        "pane_id": pane_id,
        "workspace_id": workspace_id,
        "terminal_title_stripped": title,
        "name": name,
        "agent_status": status,
    }
    if kind:
        result["agent"] = kind
    return result


class FakeHerdr:
    def __init__(
        self,
        agents: list[dict[str, object]],
        workspaces: list[dict[str, object]] | None = None,
    ) -> None:
        self.agents = agents
        self.workspaces = workspaces or WORKSPACES
        self.prompts: list[tuple[str, str, bool]] = []
        self.notifications: list[tuple[str, str, str]] = []
        self.focused: list[str] = []
        self.prompt_error: HerdrError | None = None
        self.read_output = "Inspecting the current task."
        self.read_error: HerdrError | None = None
        self.reads: list[tuple[str, str, int]] = []

    def workspace_list(self) -> list[dict[str, object]]:
        return self.workspaces

    def agent_list(self) -> list[dict[str, object]]:
        return self.agents

    def agent_prompt(self, target: str, text: str, *, wait: bool) -> None:
        if self.prompt_error:
            raise self.prompt_error
        self.prompts.append((target, text, wait))

    def workspace_focus(self, workspace_id: str) -> None:
        self.focused.append(workspace_id)

    def notification_show(self, title: str, body: str, sound: str) -> None:
        self.notifications.append((title, body, sound))

    def agent_get(self, target: str) -> dict[str, object]:
        for row in self.agents:
            if target in {row.get("pane_id"), row.get("name")}:
                return row
        raise HerdrError(f"unknown agent {target}")

    def agent_read(
        self,
        target: str,
        *,
        source: str = "recent-unwrapped",
        lines: int = 40,
    ) -> str:
        self.reads.append((target, source, lines))
        if self.read_error:
            raise self.read_error
        return self.read_output


class FakeSecretary:
    def __init__(self) -> None:
        self.result = "The frontend agent is running the authentication tests."
        self.error: SecretaryError | None = None
        self.calls: list[dict[str, str | None]] = []
        self.plan_results: list[IntentPlan] = []
        self.plan_calls: list[dict[str, object]] = []
        self.verification_approved = True

    def plan(self, utterance: str, **kwargs: object) -> BoundIntentPlan:
        self.plan_calls.append({"utterance": utterance, **kwargs})
        if self.error:
            raise self.error
        if not self.plan_results:
            raise SecretaryError("no mocked plan")
        return BoundIntentPlan(
            str(kwargs["utterance_digest"]),
            str(kwargs["catalog_digest"]),
            self.plan_results.pop(0),
        )

    def verify_prompt(
        self, *, proposed: IntentPlan, **kwargs: object
    ) -> VerificationResult:
        assert proposed.target and proposed.target.pane_id and proposed.message
        return VerificationResult(
            approved=self.verification_approved,
            utterance_digest=str(kwargs["utterance_digest"]),
            catalog_digest=str(kwargs["catalog_digest"]),
            plan_digest=str(kwargs["plan_digest"]),
            action_kind="agent_prompt",
            target={
                "workspace_id": proposed.target.workspace_id,
                "pane_id": proposed.target.pane_id,
            },
            message=proposed.message,
            reason="exact match" if self.verification_approved else "ambiguous speech",
        )

    def summarize_progress(self, **kwargs: str | None) -> str:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class FakeSpeaker:
    def __init__(self) -> None:
        self.enabled = True
        self.calls: list[tuple[str, bool]] = []

    def speak(self, text: str, *, replace: bool = False) -> None:
        self.calls.append((text, replace))


def daemon_with(herdr: FakeHerdr, config: AppConfig | None = None) -> Daemon:
    daemon = Daemon.__new__(Daemon)
    daemon.config = config or AppConfig(focus_on_prompt=False)
    daemon.policy = TalkPolicy(daemon.config)
    daemon.herdr = herdr
    daemon.secretary = FakeSecretary()
    daemon.verifier = daemon.secretary
    daemon.speaker = FakeSpeaker()
    daemon.focused_workspace_id = None
    daemon.last_summary = None
    daemon.last_voice_action = None
    daemon.last_transcript = None
    daemon.dictation = DictationBuffer()
    daemon.pending_clarification = None
    daemon._last_final_frame = None
    root = Path(tempfile.mkdtemp(dir=_TEST_RUNTIME.name))
    daemon.paths = RuntimePaths(root, root / "config", root / "state", "herdr", None)
    daemon._replay_ledger_error = None
    daemon._consumed_utterance_ids = daemon._initialize_replay_ledger()
    return daemon


class SpecificAgentRoutingTests(unittest.TestCase):
    def test_real_herdr_payload_shape_keeps_kind_separate_from_name(self) -> None:
        client = HerdrClient()
        row = {
            "agent": "codex",
            "agent_status": "working",
            "pane_id": "w1:p1",
            "terminal_title_stripped": "Implementing UI",
            "workspace_id": "w1",
        }
        with patch.object(
            client,
            "cli_json",
            side_effect=[
                {"id": "cli:agent:list", "result": {"agents": [row]}},
                {
                    "id": "cli:agent:get",
                    "result": {"agent": row, "type": "agent_info"},
                },
            ],
        ):
            agents = client.agent_list()
            fetched = client.agent_get("w1:p1")

        self.assertEqual(agents, [row])
        self.assertEqual(fetched, row)
        result = resolve_route(
            "frontend",
            workspaces=WORKSPACES,
            agents=agents,
            aliases={},
            agent_query="frontend",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.target, "w1:p1")

    def test_punctuation_only_workspace_does_not_match_everything(self) -> None:
        result = resolve_route(
            "...",
            workspaces=WORKSPACES,
            agents=[agent("w1:p1", "w1", "Ready")],
            aliases={},
        )
        self.assertFalse(result.ok)
        self.assertIn("#1 frontend", result.candidates or [])

    def test_selects_named_agent_in_multi_agent_workspace(self) -> None:
        agents = [
            agent("w1:p1", "w1", "Implementing UI", name="builder"),
            agent("w1:p2", "w1", "Reviewing UI", name="reviewer"),
        ]
        result = resolve_route(
            "frontend",
            workspaces=WORKSPACES,
            agents=agents,
            aliases={},
            agent_query="reviewer",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.target, "reviewer")
        self.assertEqual(result.pane_id, "w1:p2")

    def test_selects_terminal_title_phrase_in_multi_agent_workspace(self) -> None:
        agents = [
            agent("w1:p1", "w1", "Implementing UI"),
            agent("w1:p2", "w1", "Security review"),
        ]
        result = resolve_route(
            "frontend",
            workspaces=WORKSPACES,
            agents=agents,
            aliases={},
            agent_query="security",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.target, "w1:p2")

    def test_unmatched_title_returns_available_choices(self) -> None:
        agents = [
            agent("w1:p1", "w1", "Implementing UI"),
            agent("w1:p2", "w1", "Security review"),
        ]
        result = resolve_route(
            "frontend",
            workspaces=WORKSPACES,
            agents=agents,
            aliases={},
            agent_query="database",
        )
        self.assertFalse(result.ok)
        self.assertIn("Security review", result.candidates or [])

    def test_current_herdr_agent_kind_is_not_treated_as_a_unique_name(self) -> None:
        agents = [
            agent("w1:p1", "w1", "Implementing UI", kind="codex"),
            agent("w1:p2", "w1", "Security review", kind="codex"),
        ]

        result = resolve_route(
            "frontend",
            workspaces=WORKSPACES,
            agents=agents,
            aliases={},
            agent_query="codex",
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.candidates,
            ["codex — Implementing UI", "codex — Security review"],
        )

    def test_unique_current_herdr_pane_ignores_spurious_agent_selector(self) -> None:
        result = resolve_route(
            "frontend",
            workspaces=WORKSPACES,
            agents=[agent("w1:p1", "w1", "Implementing UI", kind="codex")],
            aliases={},
            agent_query="frontend",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.target, "w1:p1")

    def test_current_herdr_kind_is_omitted_from_secretary_agent_names(self) -> None:
        daemon = daemon_with(
            FakeHerdr([agent("w1:p1", "w1", "Implementing UI", kind="codex")])
        )
        daemon._reload_aliases = Mock()

        directory = daemon._space_directory()

        self.assertIsNone(directory[0]["agents"][0]["name"])

    def test_multiple_panes_do_not_implicitly_prefer_only_named_agent(self) -> None:
        agents = [
            agent("w1:p1", "w1", "Implementing UI", name="builder", kind="codex"),
            agent("w1:p2", "w1", "Security review", kind="codex"),
        ]

        result = resolve_route(
            "frontend", workspaces=WORKSPACES, agents=agents, aliases={}
        )

        self.assertFalse(result.ok)
        self.assertIn("builder", result.candidates or [])
        self.assertIn("codex — Security review", result.candidates or [])


class DeliverySafetyTests(unittest.TestCase):
    def test_public_route_and_prompt_is_a_fail_closed_tombstone(self) -> None:
        herdr = FakeHerdr(
            [
                agent("w1:p1", "w1", "Implementing UI", name="builder"),
                agent("w1:p2", "w1", "Security review", name="reviewer"),
            ]
        )
        daemon = daemon_with(herdr)

        result = daemon.route_and_prompt(
            "frontend", "check auth", agent="security review"
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "verified_planner_required")
        self.assertEqual(herdr.prompts, [])

    def test_missing_workspace_names_available_choices(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Ready")])
        daemon = daemon_with(herdr)

        result = daemon.route_and_prompt("mobile", "ship it")

        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "verified_planner_required")

    def test_ambiguous_voice_route_notifies_with_actionable_titles(self) -> None:
        herdr = FakeHerdr(
            [
                agent("w1:p1", "w1", "Implementing UI"),
                agent("w1:p2", "w1", "Security review"),
            ]
        )
        daemon = daemon_with(herdr)

        result = daemon.route_and_prompt("frontend", "ship it")

        self.assertFalse(result["ok"])
        self.assertIn("planner and verifier", result["message"])
        self.assertEqual(herdr.prompts, [])

    def test_blocked_target_is_refused_without_prompt(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Needs approval", status="blocked")])
        daemon = daemon_with(herdr)

        result = daemon.route_and_prompt("frontend", "continue")

        self.assertEqual(result["code"], "verified_planner_required")
        self.assertEqual(herdr.prompts, [])

    def test_working_target_accepts_followup_prompt(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Running tests", status="working")])
        daemon = daemon_with(herdr)

        result = daemon.route_and_prompt("frontend", "new work")

        self.assertFalse(result["ok"])
        self.assertEqual(herdr.prompts, [])

    def test_cli_not_ready_error_becomes_spoken_failure(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Ready")])
        herdr.prompt_error = HerdrError("agent_not_ready")
        daemon = daemon_with(herdr)

        result = daemon.route_and_prompt("frontend", "new work")

        self.assertEqual(result["code"], "verified_planner_required")
        self.assertEqual(herdr.prompts, [])

    def test_multi_pane_clarification_reply_returns_to_llm(self) -> None:
        herdr = FakeHerdr(
            [
                agent("w1:p1", "w1", "Implementing UI", name="builder"),
                agent("w1:p2", "w1", "Security review", name="reviewer"),
            ]
        )
        daemon = daemon_with(herdr)
        daemon._record_activity = Mock()
        daemon._set_activity_status = Mock()
        daemon._reload_aliases = Mock()
        daemon.secretary.plan_results = [
            planned(
                "clarification",
                clarification="Which frontend pane should receive it?",
                unresolved_slots=("agent",),
            ),
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p2"),
                message="check auth without changing the public API",
                evidence=IntentEvidence(
                    "clarification_request",
                    "check auth without changing the public API",
                ),
                agent_evidence=IntentEvidence("clarification_answer", "reviewer"),
                workspace_evidence=IntentEvidence("clarification_request", "frontend"),
            ),
        ]

        daemon._dispatch_utterance(
            "tell frontend check auth without changing the public API",
            transcript="Secretary tell frontend check auth without changing the public API",
        )

        self.assertEqual(daemon.last_voice_action["action_kind"], "clarification")
        self.assertEqual(herdr.prompts, [])

        daemon._handle_transcript("reviewer")

        self.assertIsNone(daemon.pending_clarification)
        self.assertEqual(
            herdr.prompts,
            [("w1:p2", "check auth without changing the public API", False)],
        )
        self.assertEqual(len(daemon.secretary.plan_calls), 2)
        clarification_state = daemon.secretary.plan_calls[1]["state"]
        self.assertEqual(clarification_state["phase"], "clarification")

    def test_clarification_chain_accumulates_context_and_clears_before_ready(
        self,
    ) -> None:
        herdr = FakeHerdr(
            [
                agent("w1:p1", "w1", "Implementing UI", name="builder"),
                agent("w1:p2", "w1", "Security review", name="reviewer"),
            ]
        )
        daemon = daemon_with(herdr)
        statuses: list[dict[str, object]] = []
        daemon._set_activity_status = lambda **fields: statuses.append(fields)
        daemon._reload_aliases = Mock()
        daemon.secretary.plan_results = [
            planned(
                "clarification",
                clarification="Which agent?",
                unresolved_slots=("agent",),
            ),
            planned(
                "clarification",
                clarification="Please say the exact pane name.",
                unresolved_slots=("agent",),
            ),
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p2"),
                message="run\ttests\nwith\u2003care",
                evidence=IntentEvidence(
                    "clarification_request", "run\ttests\nwith\u2003care"
                ),
                workspace_evidence=IntentEvidence("clarification_request", "frontend"),
                agent_evidence=IntentEvidence("clarification_answer", "reviewer"),
            ),
        ]

        daemon._dispatch_utterance(
            "tell frontend run\ttests\nwith\u2003care",
            transcript="Secretary tell frontend run\ttests\nwith\u2003care",
        )
        first_id = daemon.pending_clarification["clarification_id"]
        first_expiry = daemon.pending_clarification["expires_at"]
        daemon._handle_transcript("the review pane")
        second_id = daemon.pending_clarification["clarification_id"]
        self.assertEqual(first_id, second_id)
        self.assertEqual(first_expiry, daemon.pending_clarification["expires_at"])
        daemon._handle_transcript("reviewer")

        self.assertIsNone(daemon.pending_clarification)
        self.assertEqual(
            herdr.prompts, [("w1:p2", "run\ttests\nwith\u2003care", False)]
        )
        self.assertEqual(daemon.secretary.plan_calls[-1]["raw_transcript"], "reviewer")
        followups = daemon.secretary.plan_calls[-1]["state"]["pending_transaction"][
            "followups"
        ]
        self.assertEqual(
            [item["content"] for item in followups], ["the review pane", "reviewer"]
        )
        self.assertEqual(statuses[-1]["phase"], "ready")

    def test_clarification_resolves_different_slots_across_multiple_answers(
        self,
    ) -> None:
        herdr = FakeHerdr(
            [
                agent("w1:p1", "w1", "Implementing UI", name="builder"),
                agent("w1:p2", "w1", "Security review", name="reviewer"),
            ]
        )
        daemon = daemon_with(herdr)
        daemon._reload_aliases = Mock()
        message = "run\ttests\nwith\u2003care"
        daemon.secretary.plan_results = [
            planned(
                "clarification",
                clarification="Which workspace?",
                unresolved_slots=("workspace", "message", "agent"),
            ),
            planned(
                "clarification",
                clarification="What should I send?",
                unresolved_slots=("message", "agent"),
            ),
            planned(
                "clarification",
                clarification="Which agent?",
                unresolved_slots=("agent",),
            ),
            planned(
                "agent_prompt",
                target=IntentTarget("w1", "w1:p2"),
                message=message,
                evidence=IntentEvidence("clarification_followup_2", message),
                workspace_evidence=IntentEvidence(
                    "clarification_followup_1", "frontend"
                ),
                agent_evidence=IntentEvidence("clarification_answer", "reviewer"),
            ),
        ]
        daemon.secretary.verify_prompt = Mock(wraps=daemon.secretary.verify_prompt)

        daemon._handle_transcript("Jenny send a message")
        daemon._handle_transcript("frontend")
        daemon._handle_transcript(message)
        daemon._handle_transcript("reviewer")

        self.assertEqual(herdr.prompts, [("w1:p2", message, False)])
        self.assertIsNone(daemon.pending_clarification)
        sources = daemon.secretary.verify_prompt.call_args.kwargs["source_evidence"]
        self.assertEqual(sources["clarification_followup_1"], "frontend")
        self.assertEqual(sources["clarification_followup_2"], message)
        self.assertEqual(sources["clarification_answer"], "reviewer")
        state = daemon.secretary.plan_calls[-1]["state"]
        self.assertEqual(
            state["pending_transaction"]["followups"][1]["unresolved_slots"],
            ["message", "agent"],
        )


class FleetStatusTests(unittest.TestCase):
    def test_batch_status_and_fleet_summaries_are_notified_and_spoken(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Ready")])
        daemon = daemon_with(herdr)
        daemon._reload_aliases = Mock()
        daemon.secretary.plan_results = [
            planned(
                "batch",
                actions=(
                    planned("status", target=IntentTarget("w1", "w1:p1")),
                    planned("fleet_status"),
                ),
            )
        ]

        daemon._handle_transcript(
            "Jenny summarize frontend and tell me which agents are blocked"
        )

        result = daemon.last_voice_action["result"]
        self.assertTrue(result["ok"])
        self.assertFalse(result["sent"])
        self.assertEqual(
            daemon.speaker.calls[:2],
            [
                (daemon.secretary.result, False),
                ("No agents are blocked.", False),
            ],
        )
        self.assertIn(daemon.secretary.result, herdr.notifications[-1][1])
        self.assertIn("No agents are blocked.", herdr.notifications[-1][1])
        self.assertIn(daemon.secretary.result, result["summary"])
        self.assertIn("No agents are blocked.", result["summary"])

    def test_control_rpc_exposes_fleet_status(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Needs approval", status="blocked")])
        daemon = daemon_with(herdr)

        response = json.loads(daemon._dispatch(ControlRequest("1", "fleet_status", {})))

        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["blocked_count"], 1)

    def test_reports_blocked_agents_with_workspace_and_title(self) -> None:
        herdr = FakeHerdr(
            [
                agent("w1:p1", "w1", "Needs approval", status="blocked"),
                agent("w2:p1", "w2", "Running tests", status="working"),
            ]
        )
        result = daemon_with(herdr).fleet_status()

        self.assertEqual(result["blocked_count"], 1)
        self.assertIn("frontend: Needs approval", result["summary"])

    def test_reports_clear_fleet(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Ready")])
        self.assertEqual(
            daemon_with(herdr).fleet_status()["summary"],
            "No agents are blocked.",
        )

    def test_voice_fleet_status_is_notified_and_spoken(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Needs approval", status="blocked")])
        daemon = daemon_with(herdr)
        daemon.secretary.plan_results = [planned("fleet_status")]

        daemon._dispatch_utterance(
            "any blocked agents",
            transcript="any blocked agents",
        )

        self.assertEqual(daemon.last_voice_action["result"]["blocked_count"], 1)
        self.assertIn("blocked agent", daemon.speaker.calls[-1][0])


class ExplicitStatusSummaryTests(unittest.TestCase):
    def test_reads_bounded_output_from_resolved_pane_and_uses_secretary(self) -> None:
        herdr = FakeHerdr(
            [
                agent(
                    "w1:p2",
                    "w1",
                    "Authentication cleanup",
                    name="reviewer",
                    status="working",
                )
            ]
        )
        herdr.read_output = "\x1b[31mRunning auth tests\x1b[0m\n12 passed"
        daemon = daemon_with(herdr)

        summary = daemon.summarize_agent("reviewer")

        self.assertEqual(
            summary,
            "The frontend agent is running the authentication tests.",
        )
        self.assertEqual(herdr.reads, [("w1:p2", "visible", 40)])
        self.assertEqual(
            daemon.secretary.calls,
            [
                {
                    "workspace": "frontend",
                    "title": "Authentication cleanup",
                    "status": "working",
                    "excerpt": "Running auth tests\n12 passed",
                }
            ],
        )

    def test_settled_agent_uses_recent_unwrapped_output(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Test run", status="done")])
        daemon = daemon_with(herdr)

        daemon.summarize_agent("w1:p1")

        self.assertEqual(herdr.reads, [("w1:p1", "recent-unwrapped", 40)])

    def test_explicit_summary_does_not_fall_back_to_shallow_metadata(self) -> None:
        herdr = FakeHerdr(
            [agent("w1:p1", "w1", "Authentication cleanup", status="working")]
        )
        herdr.read_error = HerdrError("pane output unavailable")
        daemon = daemon_with(herdr)

        with self.assertRaisesRegex(SecretaryError, "could not inspect recent output"):
            daemon.summarize_agent("w1:p1")

        self.assertEqual(daemon.secretary.calls, [])

    def test_excerpt_sent_to_model_is_character_bounded(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Large build log")])
        herdr.read_output = "old context\n" + "x" * 7000
        daemon = daemon_with(herdr)

        daemon.summarize_agent("w1:p1")

        excerpt = str(daemon.secretary.calls[0]["excerpt"])
        self.assertLessEqual(len(excerpt), 6000)
        self.assertNotIn("old context", excerpt)

    def test_voice_status_resolves_and_reads_the_concrete_pane(self) -> None:
        herdr = FakeHerdr(
            [agent("w1:p2", "w1", "Authentication cleanup", name="reviewer")]
        )
        daemon = daemon_with(herdr)
        daemon.secretary.plan_results = [
            planned("status", target=IntentTarget("w1", "w1:p2"))
        ]

        daemon._dispatch_utterance(
            "review the frontend reviewer",
            transcript="review the frontend reviewer",
        )

        self.assertEqual(herdr.reads, [("w1:p2", "recent-unwrapped", 40)])
        self.assertEqual(
            daemon.last_voice_action["result"]["summary"],
            "The frontend agent is running the authentication tests.",
        )

    def test_workspace_only_status_summarizes_parent_then_linked_worktrees(
        self,
    ) -> None:
        herdr = FakeHerdr(
            [
                agent("w1:p1", "w1", "Parent orch", status="working"),
                agent("w1:p2", "w1", "Parent helper", status="done"),
                agent("w10:p1", "w10", "Worktree A", status="working"),
                agent("w11:p1", "w11", "Worktree B", status="idle"),
            ],
            workspaces=[
                {
                    "workspace_id": "w1",
                    "label": "#1 whatshot",
                    "number": 1,
                    "worktree": {
                        "repo_key": "/repo/whatshot/.git",
                        "is_linked_worktree": False,
                    },
                },
                {
                    "workspace_id": "w2",
                    "label": "#2 other",
                    "number": 2,
                    "worktree": {
                        "repo_key": "/repo/other/.git",
                        "is_linked_worktree": False,
                    },
                },
                {
                    "workspace_id": "w11",
                    "label": "#11 later worktree",
                    "number": 11,
                    "worktree": {
                        "repo_key": "/repo/whatshot/.git",
                        "is_linked_worktree": True,
                    },
                },
                {
                    "workspace_id": "w10",
                    "label": "#10 earlier worktree",
                    "number": 10,
                    "worktree": {
                        "repo_key": "/repo/whatshot/.git",
                        "is_linked_worktree": True,
                    },
                },
            ],
        )
        daemon = daemon_with(herdr)
        summaries = {
            "w1:p1": "Parent is reviewing the repo.",
            "w1:p2": "Helper finished spawning worktrees.",
            "w10:p1": "Worktree A is fixing the scene.",
            "w11:p1": "Worktree B is idle after review.",
        }
        daemon.summarize_agent = Mock(
            side_effect=lambda target, **_kwargs: summaries[target]
        )

        parts = daemon.summarize_workspace_group("w1")

        self.assertEqual(
            [call.args[0] for call in daemon.summarize_agent.call_args_list],
            ["w1:p1", "w1:p2", "w10:p1", "w11:p1"],
        )
        self.assertEqual(
            parts,
            [
                "whatshot (Parent orch): Parent is reviewing the repo.",
                "whatshot (Parent helper): Helper finished spawning worktrees.",
                "earlier worktree (Worktree A): Worktree A is fixing the scene.",
                "later worktree (Worktree B): Worktree B is idle after review.",
            ],
        )

    def test_workspace_only_multi_agent_status_does_not_require_agent_evidence(
        self,
    ) -> None:
        herdr = FakeHerdr(
            [
                agent("w1:p1", "w1", "Repository state review", status="working"),
                agent("w1:p7", "w1", "Claude Worktrees", name="whatshot_orch"),
            ]
        )
        daemon = daemon_with(herdr)
        daemon.secretary.plan_results = [
            planned(
                "status",
                target=IntentTarget("w1", "w1:p1"),
                workspace_evidence="one",
            )
        ]
        daemon.summarize_agent = Mock(
            side_effect=lambda target, **_kwargs: f"summary for {target}"
        )

        daemon._dispatch_utterance(
            "summarized one.",
            transcript="Jenny summarized one.",
        )

        self.assertEqual(
            [call.args[0] for call in daemon.summarize_agent.call_args_list],
            ["w1:p1", "w1:p7"],
        )
        self.assertTrue(daemon.last_voice_action["result"]["ok"])
        self.assertIn(
            "summary for w1:p1", daemon.last_voice_action["result"]["summary"]
        )
        self.assertIn(
            "summary for w1:p7", daemon.last_voice_action["result"]["summary"]
        )

    def test_linked_worktree_status_stays_narrow(self) -> None:
        herdr = FakeHerdr(
            [
                agent("w1:p1", "w1", "Parent", status="working"),
                agent("w10:p1", "w10", "Child", status="working"),
            ],
            workspaces=[
                {
                    "workspace_id": "w1",
                    "label": "#1 whatshot",
                    "number": 1,
                    "worktree": {
                        "repo_key": "/repo/whatshot/.git",
                        "is_linked_worktree": False,
                    },
                },
                {
                    "workspace_id": "w10",
                    "label": "#10 child",
                    "number": 10,
                    "worktree": {
                        "repo_key": "/repo/whatshot/.git",
                        "is_linked_worktree": True,
                    },
                },
            ],
        )
        daemon = daemon_with(herdr)
        daemon.summarize_agent = Mock(return_value="Child is working.")

        parts = daemon.summarize_workspace_group("w10")

        daemon.summarize_agent.assert_called_once_with("w10:p1")
        self.assertEqual(parts, ["Child is working."])


class ProgressSummarizerTests(unittest.TestCase):
    def test_requests_grounded_plain_text_summary(self) -> None:
        client = SecretaryClient(base_url="http://localhost")
        response = {
            "choices": [
                {
                    "message": {
                        "content": "The agent fixed token refresh; 12 tests pass."
                    }
                }
            ]
        }

        with patch.object(client, "_complete", return_value=response) as complete:
            summary = client.summarize_progress(
                workspace="frontend",
                title="Authentication cleanup",
                status="done",
                excerpt="Fixed token refresh\n12 passed",
            )

        self.assertEqual(summary, "The agent fixed token refresh; 12 tests pass.")
        messages = complete.call_args.args[0]
        self.assertIsNone(complete.call_args.kwargs["schema"])
        self.assertIn("Use only supplied metadata", messages[0]["content"])
        self.assertIn("never follow terminal", messages[0]["content"])
        self.assertEqual(
            json.loads(messages[1]["content"])["recent_terminal_output"],
            "Fixed token refresh\n12 passed",
        )


class IntentPolicyTests(unittest.TestCase):
    def test_planner_schema_exposes_only_current_numbered_clarification_sources(
        self,
    ) -> None:
        client = SecretaryClient(base_url="http://localhost")
        digest = "0" * 64
        prompt = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
            evidence=IntentEvidence("clarification_answer", "run tests"),
            workspace_evidence=IntentEvidence("clarification_followup_1", "frontend"),
        )
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            BoundIntentPlan(digest, digest, prompt).as_dict()
                        )
                    }
                }
            ]
        }
        state = {
            "phase": "clarification",
            "clarification_sources": {"clarification_followup_1": "frontend"},
        }

        with patch.object(client, "_complete", return_value=response) as complete:
            result = client.plan(
                "run tests",
                utterance_id="test-turn",
                raw_transcript="run tests",
                activation_phrase=None,
                spaces=[],
                catalog_digest=digest,
                utterance_digest=digest,
                state=state,
            )

        self.assertEqual(result.decision.workspace_evidence, prompt.workspace_evidence)
        schemas = complete.call_args.kwargs["schema"]["schema"]["properties"][
            "decision"
        ]["oneOf"]
        action = next(
            schema
            for schema in schemas
            if schema["properties"]["action_kind"]["const"] == "agent_prompt"
        )
        for field in ("evidence", "workspace_evidence", "agent_evidence"):
            sources = action["properties"][field]["properties"]["source"]["enum"]
            self.assertIn("clarification_followup_1", sources)
            self.assertNotIn("clarification_followup_2", sources)

    def test_numbered_clarification_sources_cannot_escape_their_transaction(
        self,
    ) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Ready")])
        daemon = daemon_with(herdr)
        daemon._reload_aliases = Mock()
        proposal = planned(
            "agent_prompt",
            target=IntentTarget("w1", "w1:p1"),
            message="run tests",
            workspace_evidence=IntentEvidence("clarification_followup_1", "frontend"),
        )
        daemon.secretary.plan_results = [proposal, proposal, proposal]

        daemon._handle_transcript("Jenny tell frontend run tests")

        self.assertEqual(herdr.prompts, [])
        self.assertEqual(daemon.last_voice_action["result"]["code"], "plan_blocked")

    def test_strict_plan_requires_every_schema_key(self) -> None:
        with self.assertRaises(IntentValidationError):
            plan_from_json({"action_kind": "fleet_status"})

    def test_strict_plan_keeps_exact_opaque_target(self) -> None:
        prompt = plan_from_json(
            {
                "action_kind": "agent_prompt",
                "target": {"workspace_id": "w1", "pane_id": "w1:p2"},
                "message": "check auth",
                "workspace_evidence": {
                    "source": "post_wake_content",
                    "quote": "frontend",
                },
                "evidence": {
                    "source": "post_wake_content",
                    "quote": "check auth",
                },
                "confidence": 0.99,
                "reason": "explicit target and content",
            }
        )
        self.assertEqual(prompt.target, IntentTarget("w1", "w1:p2"))

    def test_no_action_cannot_smuggle_actions(self) -> None:
        raw = planned("no_action").as_dict()
        raw["actions"] = [planned("fleet_status").as_dict()]
        with self.assertRaises(IntentValidationError):
            plan_from_json(raw)

    def test_voice_talk_policy_rejects_scoped_target(self) -> None:
        with self.assertRaisesRegex(IntentValidationError, "keys do not match"):
            plan_from_json(
                {
                    "action_kind": "talk_policy",
                    "target": {"workspace_id": "w1", "pane_id": None},
                    "mode": "silent",
                    "confidence": 0.99,
                    "reason": "scoped request",
                }
            )

    def test_voice_talk_policy_rejects_unsupported_mode_and_batch_split(self) -> None:
        with self.assertRaisesRegex(IntentValidationError, "supported mode"):
            plan_from_json(
                {
                    "action_kind": "talk_policy",
                    "mode": "workspace_silent",
                    "confidence": 0.99,
                    "reason": "unsupported",
                }
            )
        with self.assertRaisesRegex(IntentValidationError, "unsupported action"):
            plan_from_json(
                {
                    "action_kind": "batch",
                    "actions": [
                        {
                            "action_kind": "talk_policy",
                            "mode": "silent",
                            "confidence": 0.99,
                            "reason": "cannot split execution scope",
                        }
                    ],
                    "confidence": 0.99,
                    "reason": "unsupported split",
                }
            )


class TalkPolicyTests(unittest.TestCase):
    def test_all_modes_enforce_expected_lifecycle_scope(self) -> None:
        silent = TalkPolicy(AppConfig(talk_mode="silent"))
        self.assertFalse(silent.should_announce(pane_id="p1", status="blocked")[0])

        blocked = TalkPolicy(AppConfig(talk_mode="blocked_only"))
        self.assertFalse(blocked.should_announce(pane_id="p1", status="done")[0])
        self.assertTrue(blocked.should_announce(pane_id="p1", status="blocked")[0])

        milestones = TalkPolicy(AppConfig(talk_mode="milestones"))
        self.assertFalse(milestones.should_announce(pane_id="p1", status="done")[0])
        milestones.mark_prompted("p1")
        self.assertTrue(milestones.should_announce(pane_id="p1", status="done")[0])

        verbose = TalkPolicy(AppConfig(talk_mode="verbose"))
        self.assertTrue(verbose.should_announce(pane_id="p1", status="working")[0])

    def test_workspace_quiet_matches_id_or_label(self) -> None:
        policy = TalkPolicy(
            AppConfig(talk_mode="verbose", quiet_workspaces=["frontend"])
        )
        allowed, reason = policy.should_announce(
            pane_id="p1",
            status="working",
            workspace_id="w1",
            workspace_label="#1 frontend",
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "workspace_quiet")

    def test_new_blocked_transition_bypasses_normal_cooldown(self) -> None:
        policy = TalkPolicy(AppConfig(talk_mode="verbose", min_interval_secs=60))
        policy.mark_spoken("p1", "done")
        self.assertTrue(policy.should_announce(pane_id="p1", status="blocked")[0])

    def test_runtime_workspace_quiet_control_uses_resolved_workspace(self) -> None:
        daemon = daemon_with(FakeHerdr([agent("w1:p1", "w1", "Ready")]))
        result = daemon.set_talk_policy(space="frontend", quiet=True)

        self.assertTrue(result["ok"])
        self.assertIn("w1", result["quiet_workspaces"])
        self.assertFalse(
            daemon.policy.should_announce(
                pane_id="w1:p1", status="blocked", workspace_id="w1"
            )[0]
        )
        enabled = daemon.set_talk_policy(space="frontend", quiet=False)
        self.assertTrue(enabled["ok"])
        self.assertNotIn("w1", enabled["quiet_workspaces"])

    def test_voice_policy_command_changes_runtime_mode(self) -> None:
        daemon = daemon_with(FakeHerdr([agent("w1:p1", "w1", "Ready")]))
        daemon.secretary.plan_results = [planned("talk_policy", mode="silent")]

        daemon._dispatch_utterance(
            "be quiet",
            transcript="be quiet",
        )

        self.assertEqual(daemon.policy.talk_mode, "silent")
        self.assertEqual(daemon.last_voice_action["result"]["mode"], "silent")

    def test_urgent_blocked_announcement_replaces_queued_tts(self) -> None:
        herdr = FakeHerdr([agent("w1:p1", "w1", "Needs approval", status="blocked")])
        daemon = daemon_with(herdr, AppConfig(talk_mode="blocked_only"))

        daemon._maybe_announce(
            "w1:p1",
            "blocked",
            agent=herdr.agents[0],
        )

        self.assertEqual(daemon.speaker.calls[-1], ("Needs approval is blocked.", True))
        self.assertEqual(herdr.reads, [])
        self.assertEqual(daemon.secretary.calls, [])

    def test_lifecycle_announcement_uses_event_status_without_inspection(self) -> None:
        herdr = FakeHerdr(
            [agent("w1:p1", "w1", "Authentication cleanup", status="working")]
        )
        daemon = daemon_with(herdr, AppConfig(talk_mode="verbose"))

        daemon._maybe_announce("w1:p1", "done", agent=herdr.agents[0])

        self.assertEqual(
            daemon.speaker.calls[-1], ("Authentication cleanup is done.", False)
        )
        self.assertEqual(herdr.reads, [])
        self.assertEqual(daemon.secretary.calls, [])


class TalkConfigTests(unittest.TestCase):
    def test_loads_talk_mode_and_quiet_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plugin_root = root / "plugin"
            config_dir = root / "config"
            state_dir = root / "state"
            plugin_root.mkdir()
            config_dir.mkdir()
            (config_dir / "config.toml").write_text(
                '[talk]\nmode = "verbose"\nquiet_workspaces = ["frontend"]\n'
            )

            paths = RuntimePaths(
                plugin_root=plugin_root,
                config_dir=config_dir,
                state_dir=state_dir,
                herdr_bin="herdr",
                herdr_socket=None,
            )

            config = AppConfig.load(paths)

        self.assertEqual(config.talk_mode, "verbose")
        self.assertEqual(config.quiet_workspaces, ["frontend"])


if __name__ == "__main__":
    unittest.main()
