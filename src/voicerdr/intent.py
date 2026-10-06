import re
from dataclasses import dataclass, field
from typing import Any

ACTION_KINDS = frozenset(
    {
        "agent_prompt",
        "status",
        "fleet_status",
        "talk_policy",
        "control",
        "dictation_start",
        "dictation_append",
        "dictation_finish",
        "dictation_cancel",
        "clarification",
        "no_action",
        "batch",
    }
)
TALK_MODES = frozenset({"silent", "blocked_only", "milestones", "verbose"})
CONTROL_MODES = frozenset({"mute"})
CLARIFICATION_SLOTS = frozenset(
    {"action", "workspace", "agent", "message", "mode", "confirmation"}
)
EVIDENCE_SOURCES = (
    "post_wake_content",
    "raw_transcript",
    "dictation_buffer",
    "clarification_request",
    "clarification_answer",
)


class IntentValidationError(ValueError):
    """The model returned JSON that does not satisfy the intent contract."""


@dataclass(frozen=True)
class IntentTarget:
    """Exact opaque live-catalog identifiers selected by the planner."""

    workspace_id: str
    pane_id: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"workspace_id": self.workspace_id, "pane_id": self.pane_id}


@dataclass(frozen=True)
class IntentEvidence:
    """An exact, contiguous excerpt from a server-owned transcript source."""

    source: str
    quote: str
    catalog_number: int | None = None

    def as_dict(self) -> dict[str, str | int]:
        value: dict[str, str | int] = {"source": self.source, "quote": self.quote}
        if self.catalog_number is not None:
            value["catalog_number"] = self.catalog_number
        return value


@dataclass(frozen=True)
class IntentPlan:
    """One action-specific plan. Irrelevant nullable fields are omitted."""

    action_kind: str
    target: IntentTarget | None
    message: str | None
    mode: str | None
    clarification: str | None
    confidence: float
    reason: str
    actions: tuple["IntentPlan", ...] = field(default_factory=tuple)
    evidence: IntentEvidence | None = None
    agent_evidence: IntentEvidence | None = None
    workspace_evidence: IntentEvidence | None = None
    unresolved_slots: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "action_kind": self.action_kind,
            "confidence": self.confidence,
            "reason": self.reason,
        }
        if self.target is not None:
            value["target"] = self.target.as_dict()
        if self.message is not None:
            value["message"] = self.message
        if self.evidence is not None:
            value["evidence"] = self.evidence.as_dict()
        if self.agent_evidence is not None:
            value["agent_evidence"] = self.agent_evidence.as_dict()
        if self.workspace_evidence is not None:
            value["workspace_evidence"] = self.workspace_evidence.as_dict()
        if self.mode is not None:
            value["mode"] = self.mode
        if self.clarification is not None:
            value["clarification"] = self.clarification
        if self.action_kind == "clarification":
            value["unresolved_slots"] = list(self.unresolved_slots or ("action",))
        if self.actions:
            value["actions"] = [action.as_dict() for action in self.actions]
        return value


@dataclass(frozen=True)
class BoundIntentPlan:
    """A plan bound to the exact utterance and catalog supplied to the LLM."""

    utterance_digest: str
    catalog_digest: str
    decision: IntentPlan

    def as_dict(self) -> dict[str, Any]:
        return {
            "utterance_digest": self.utterance_digest,
            "catalog_digest": self.catalog_digest,
            "decision": self.decision.as_dict(),
        }


_COMMON_KEYS = frozenset({"action_kind", "confidence", "reason"})
_TARGET_KEYS = frozenset({"workspace_id", "pane_id"})
_EVIDENCE_KEYS = frozenset({"source", "quote"})
_NUMBERED_EVIDENCE_KEYS = _EVIDENCE_KEYS | {"catalog_number"}
_ACTION_FIELDS: dict[str, frozenset[str]] = {
    "agent_prompt": _COMMON_KEYS
    | {"target", "message", "evidence", "agent_evidence", "workspace_evidence"},
    "status": _COMMON_KEYS | {"target", "agent_evidence", "workspace_evidence"},
    "fleet_status": _COMMON_KEYS,
    "talk_policy": _COMMON_KEYS | {"mode"},
    "control": _COMMON_KEYS | {"mode"},
    "dictation_start": _COMMON_KEYS | {"message"},
    "dictation_append": _COMMON_KEYS | {"message"},
    "dictation_finish": _COMMON_KEYS,
    "dictation_cancel": _COMMON_KEYS,
    "clarification": _COMMON_KEYS | {"clarification", "unresolved_slots"},
    "no_action": _COMMON_KEYS,
    "batch": _COMMON_KEYS | {"actions"},
}


def bound_plan_from_json(raw: dict[str, Any]) -> BoundIntentPlan:
    if set(raw) != {"utterance_digest", "catalog_digest", "decision"}:
        raise IntentValidationError("bound intent keys do not match schema")
    utterance_digest = _digest(raw["utterance_digest"], "utterance_digest")
    catalog_digest = _digest(raw["catalog_digest"], "catalog_digest")
    decision = raw["decision"]
    if not isinstance(decision, dict):
        raise IntentValidationError("decision must be an object")
    return BoundIntentPlan(
        utterance_digest=utterance_digest,
        catalog_digest=catalog_digest,
        decision=plan_from_json(decision),
    )


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise IntentValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def plan_from_json(raw: dict[str, Any], *, nested: bool = False) -> IntentPlan:
    """Validate an action-specific object without repairing or coercing fields."""

    kind = raw.get("action_kind")
    if not isinstance(kind, str) or kind not in ACTION_KINDS:
        raise IntentValidationError(f"invalid action_kind: {kind!r}")
    if nested and kind == "batch":
        raise IntentValidationError("nested batches are not allowed")
    expected_keys = _ACTION_FIELDS[kind]
    variants = [expected_keys]
    if kind in {"agent_prompt", "status"}:
        variants.append(expected_keys ^ {"agent_evidence"})
    if kind in {"dictation_start", "dictation_finish"}:
        variants.append(expected_keys ^ {"message"})
    if set(raw) not in variants:
        missing = sorted(expected_keys - set(raw))
        extra = sorted(set(raw) - expected_keys)
        raise IntentValidationError(
            f"{kind} keys do not match schema (missing={missing}, extra={extra})"
        )

    confidence = raw["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise IntentValidationError("confidence must be a number")
    confidence_f = float(confidence)
    if not 0.0 <= confidence_f <= 1.0:
        raise IntentValidationError("confidence must be between zero and one")
    reason = _string(raw["reason"], "reason")

    target: IntentTarget | None = None
    if "target" in raw:
        target_raw = raw["target"]
        if not isinstance(target_raw, dict) or set(target_raw) != _TARGET_KEYS:
            raise IntentValidationError("target must contain workspace_id and pane_id")
        workspace_id = _opaque_id(target_raw["workspace_id"], "workspace_id")
        pane_id = target_raw["pane_id"]
        if kind in {"agent_prompt", "status"}:
            pane_id = _opaque_id(pane_id, "pane_id")
        elif pane_id is not None:
            raise IntentValidationError("workspace-scoped target pane_id must be null")
        target = IntentTarget(workspace_id, pane_id)

    message = _string(raw["message"], "message") if "message" in raw else None
    evidence: IntentEvidence | None = None
    if "evidence" in raw:
        evidence_raw = raw["evidence"]
        if not isinstance(evidence_raw, dict) or set(evidence_raw) != _EVIDENCE_KEYS:
            raise IntentValidationError("evidence must contain source and quote")
        source = _evidence_source(evidence_raw["source"], "evidence.source")
        evidence = IntentEvidence(
            source, _string(evidence_raw["quote"], "evidence.quote")
        )
    mode = _string(raw["mode"], "mode") if "mode" in raw else None
    clarification = (
        _string(raw["clarification"], "clarification")
        if "clarification" in raw
        else None
    )
    unresolved_slots: tuple[str, ...] = ()
    if "unresolved_slots" in raw:
        values = raw["unresolved_slots"]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) for value in values)
            or len(set(values)) != len(values)
            or any(value not in CLARIFICATION_SLOTS for value in values)
        ):
            raise IntentValidationError(
                "unresolved_slots must be unique supported slots"
            )
        unresolved_slots = tuple(values)
    agent_evidence = (
        _evidence(raw["agent_evidence"], "agent_evidence")
        if "agent_evidence" in raw
        else None
    )
    workspace_evidence = (
        _evidence(
            raw["workspace_evidence"],
            "workspace_evidence",
            allow_catalog_number=True,
        )
        if "workspace_evidence" in raw
        else None
    )
    actions: tuple[IntentPlan, ...] = ()
    if "actions" in raw:
        actions_raw = raw["actions"]
        if not isinstance(actions_raw, list) or not actions_raw:
            raise IntentValidationError("batch actions must be a non-empty array")
        if not all(isinstance(item, dict) for item in actions_raw):
            raise IntentValidationError("every batch action must be an object")
        actions = tuple(plan_from_json(item, nested=True) for item in actions_raw)

    plan = IntentPlan(
        kind,
        target,
        message,
        mode,
        clarification,
        confidence_f,
        reason,
        actions,
        evidence,
        agent_evidence,
        workspace_evidence,
        unresolved_slots,
    )
    _validate_action_shape(plan)
    return plan


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntentValidationError(f"{field_name} must be a non-empty string")
    return value


def _evidence_source(value: Any, field_name: str) -> str:
    source = _string(value, field_name)
    if source not in EVIDENCE_SOURCES and not re.fullmatch(
        r"clarification_followup_[1-9][0-9]*", source
    ):
        raise IntentValidationError(f"{field_name} is unsupported")
    return source


def _evidence(
    value: Any, field_name: str, *, allow_catalog_number: bool = False
) -> IntentEvidence:
    allowed_shapes = [_EVIDENCE_KEYS]
    if allow_catalog_number:
        allowed_shapes.append(_NUMBERED_EVIDENCE_KEYS)
    if not isinstance(value, dict) or set(value) not in allowed_shapes:
        suffix = " and optional catalog_number" if allow_catalog_number else ""
        raise IntentValidationError(
            f"{field_name} must contain source and quote{suffix}"
        )
    source = _evidence_source(value["source"], f"{field_name}.source")
    catalog_number = None
    if "catalog_number" in value:
        candidate_number = value["catalog_number"]
        if (
            isinstance(candidate_number, bool)
            or not isinstance(candidate_number, int)
            or candidate_number < 1
        ):
            raise IntentValidationError(
                f"{field_name}.catalog_number must be a positive integer"
            )
        catalog_number = candidate_number
    return IntentEvidence(
        source,
        _string(value["quote"], f"{field_name}.quote"),
        catalog_number,
    )


def _opaque_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise IntentValidationError(f"target.{field_name} must be an exact opaque ID")
    return value


def _validate_action_shape(plan: IntentPlan) -> None:
    if plan.action_kind == "talk_policy" and plan.mode not in TALK_MODES:
        raise IntentValidationError("talk_policy requires a supported mode")
    if plan.action_kind == "control" and plan.mode not in CONTROL_MODES:
        raise IntentValidationError("control requires mode=mute")
    if plan.action_kind == "batch":
        allowed = {"agent_prompt", "status", "fleet_status"}
        if any(action.action_kind not in allowed for action in plan.actions):
            raise IntentValidationError("batch contains an unsupported action")
        if sum(a.action_kind == "agent_prompt" for a in plan.actions) > 1:
            raise IntentValidationError(
                "batch contains multiple agent_prompt actions; atomic delivery is unsupported"
            )


_WAKE_PUNCTUATION = ",.:-\N{EN DASH}\N{EM DASH};!?\N{HORIZONTAL ELLIPSIS}"
_WAKE_TOKEN_SEPARATOR = rf"(?:\s|[{re.escape(_WAKE_PUNCTUATION)}])+"


@dataclass(frozen=True)
class WakeMatch:
    matched: bool
    phrase: str | None
    remainder: str
    remainder_start: int = 0


def match_wake_phrase(text: str, wake_phrases: list[str]) -> WakeMatch:
    """Match only an actual configured address at the utterance start."""

    if not wake_phrases:
        return WakeMatch(True, None, text, 0)
    for configured in wake_phrases:
        tokens = re.findall(r"\S+", configured)
        if not tokens:
            continue
        # ASR may punctuate an address between configured words. Keep every
        # token exact while normalizing only the separators used to compare it.
        pattern = r"^\s*" + _WAKE_TOKEN_SEPARATOR.join(
            re.escape(token) for token in tokens
        )
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        end = match.end()
        if end < len(text) and not (
            text[end].isspace() or text[end] in _WAKE_PUNCTUATION
        ):
            continue
        # Only the configured address and its delimiter are consumed. The
        # returned content is an untouched slice of the original transcript.
        while end < len(text) and (
            text[end].isspace() or text[end] in _WAKE_PUNCTUATION
        ):
            end += 1
        return WakeMatch(True, configured, text[end:], end)
    return WakeMatch(False, None, text, 0)


def strip_wake_phrase(text: str, wake_phrases: list[str]) -> tuple[bool, str]:
    match = match_wake_phrase(text, wake_phrases)
    return match.matched, match.remainder
