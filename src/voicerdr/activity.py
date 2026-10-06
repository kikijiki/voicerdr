"""Terminal-sized live view for the daemon's structured activity journal."""

import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time
import unicodedata
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any, ClassVar, TextIO

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Label, Static

from voicerdr.config import DEFAULT_ASSISTANT_AVATAR
from voicerdr.control_client import ControlClient

EVENT_LABELS = {
    "action": "RESULT",
    "announcement": "ANNOUNCED",
    "clarification": "CLARIFY",
    "clarification_expired": "CLARIFY EXPIRED",
    "clarification_superseded": "CLARIFY REPLACED",
    "control_request": "CONTROL",
    "viewer_control": "CONTROL RESULT",
    "daemon_started": "DAEMON STARTED",
    "delivery_closed": "DELIVERY CLOSED",
    "delivery_unknown": "DELIVERY UNKNOWN",
    "dictation_appended": "DICTATION ADDED",
    "dictation_finished": "DICTATION READY",
    "dictation_started": "DICTATION STARTED",
    "heard": "HEARD",
    "intent_chosen": "INTENT",
    "interpreting": "INTERPRETING",
    "no_action": "NOT SENT",
    "planner_retry": "PLANNER RETRY",
    "prompt_requested": "SENDING",
    "prompt_sent": "SENT",
    "replay_ledger_error": "JOURNAL ERROR",
    "verification": "VERIFIED",
    "verifier_retry": "VERIFY RETRY",
    "verifying": "VERIFYING",
    "withheld": "NOT SENT",
    "workspace_created": "WORKSPACE OPENED",
    "workspace_error": "WORKSPACE ERROR",
    "workspace_reused": "WORKSPACE REUSED",
}

MIN_CONTROL_WIDTH = 10


def _clock(value: Any) -> str:
    timestamp = str(value or "")
    if "T" in timestamp:
        return timestamp.split("T", 1)[1][:8]
    return timestamp[:8] or "--:--:--"


def _target(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value or "—")
    workspace = value.get("workspace_label") or value.get("workspace_id")
    pane = value.get("agent") or value.get("agent_name") or value.get("pane_id")
    if workspace and pane:
        return f"{workspace}/{pane}"
    return str(workspace or pane or "—")


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "—"
    return " ".join(str(value).splitlines()).strip() or "—"


def _result(value: Any) -> str:
    if not isinstance(value, dict):
        return _text(value)
    sent = value.get("sent")
    outcome = "SENT" if sent is True else "NOT SENT" if sent is False else ""
    detail = value.get("summary") or value.get("message") or value.get("code")
    if outcome and detail:
        return f"{outcome}: {_text(detail)}"
    return outcome or _text(detail)


def render_event(payload: dict[str, Any], *, debug: bool = False) -> str:
    """Turn one journal object into a concise, non-opaque activity line."""
    event = str(payload.get("event") or "activity")
    if debug:
        raw = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
        return f"{_clock(payload.get('time'))}  RAW {event.upper()}  {raw}"

    source = dict(payload)
    nested_action = source.get("action")
    if event == "action" and isinstance(nested_action, dict):
        source = {**nested_action, **source}
    label = EVENT_LABELS.get(event, event.replace("_", " ").upper())
    if event == "verification" and source.get("approved") is False:
        label = "NOT VERIFIED"

    fields: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(output_name: str, *keys: str, formatter: Any = _text) -> None:
        for key in keys:
            value = source.get(key)
            if value is not None and value != "" and key not in seen:
                fields.append((output_name, formatter(value)))
                seen.add(key)
                return

    add("transcript", "transcript", "utterance")
    add("message", "message", "question")
    add("target", "target", formatter=_target)
    add("mode", "mode", "input_mode", "chosen_mode")
    add("action", "chosen_action", "action_kind")
    if event != "action":
        add("action", "action")
    add("result", "result", formatter=_result)
    add("error", "error", "reason")
    add("code", "code")
    add("status", "agent_status")
    add("workspace", "workspace_label", "space")
    add("method", "method")
    if source.get("sent") is False and label not in {"NOT SENT", "DELIVERY UNKNOWN"}:
        fields.append(("result", "NOT SENT"))
    elif source.get("sent") is True and label != "SENT":
        fields.append(("result", "SENT"))

    detail = "  ".join(f"{name}={value}" for name, value in fields)
    return f"{_clock(payload.get('time'))}  {label:<16} {detail}".rstrip()


def render_line(line: str, *, debug: bool = False) -> str:
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        suffix = f": {line.rstrip()}" if debug else " (details hidden; use --debug)"
        return f"--:--:--  MALFORMED ENTRY{suffix}"
    if not isinstance(payload, dict):
        suffix = f": {line.rstrip()}" if debug else " (details hidden; use --debug)"
        return f"--:--:--  INVALID ENTRY{suffix}"
    return render_event(payload, debug=debug)


def _read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _cell_width(character: str) -> int:
    if unicodedata.combining(character) or character in {"\ufe0e", "\ufe0f"}:
        return 0
    if unicodedata.category(character).startswith("C"):
        return 0
    return 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1


def _display_width(value: str) -> int:
    return sum(_cell_width(character) for character in value)


def _fit(value: str, width: int, *, ellipsis: bool = False) -> str:
    if width <= 0:
        return ""
    if _display_width(value) <= width:
        return value
    marker = "…" if ellipsis and width > 1 else ""
    room = width - _display_width(marker)
    output: list[str] = []
    used = 0
    for character in value:
        cells = _cell_width(character)
        if used + cells > room:
            break
        output.append(character)
        used += cells
    return "".join(output) + marker


def _wrap(value: Any, *, width: int) -> list[str]:
    """Wrap text by terminal cells, including CJK and combining characters."""
    width = max(1, width)
    text = str(value if value is not None else "—") or "—"
    output: list[str] = []
    for paragraph in text.expandtabs(4).splitlines() or [""]:
        remaining = paragraph
        if not remaining:
            output.append("")
            continue
        while _display_width(remaining) > width:
            used = 0
            split_at = 0
            whitespace_at = -1
            for index, character in enumerate(remaining):
                cells = _cell_width(character)
                if used + cells > width:
                    break
                used += cells
                split_at = index + 1
                if character.isspace():
                    whitespace_at = split_at
            if split_at == 0:
                split_at = 1
            if whitespace_at > 0:
                split_at = whitespace_at
            output.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        output.append(remaining.rstrip())
    return output or ["—"]


def _field(label: str, value: Any, width: int) -> list[str]:
    prefix = f" {label}: "
    if width <= _display_width(prefix):
        return _wrap(f"{label}: {_text(value)}", width=width)
    wrapped = _wrap(
        value if value not in {None, ""} else "—", width=width - len(prefix)
    )
    continuation = " " * len(prefix)
    return [prefix + wrapped[0], *(continuation + line for line in wrapped[1:])]


def _primary_state(state: dict[str, Any]) -> str:
    if state.get("control_outcome_unknown"):
        return "MIC UNKNOWN"
    if state.get("delivery_closed"):
        return "SHUTTING DOWN"
    if state.get("mic_transition_pending"):
        return "MUTING"
    phase = str(state.get("phase") or "starting").lower()
    mode = str(state.get("mode") or "detecting").lower()
    if phase == "muted" or mode == "muted":
        return "MUTED"
    if phase == "verifying":
        return "VERIFYING"
    if phase == "interpreting":
        return "INTERPRETING"
    if mode == "dictation" or "dictation" in phase or phase == "awaiting_end_phrase":
        return "DICTATING"
    if phase == "listening" and bool(state.get("speech_active")):
        return "HEARING"
    if phase in {"ready", "listening", "transcribing", "starting"}:
        return "LISTENING"
    return phase.replace("_", " ").upper()


def _assistant_art(state: dict[str, Any], animation_tick: int) -> list[str]:
    """ASCII avatar frames keyed by activity primary state."""
    primary = _primary_state(state)
    eye = "-" if primary == "MUTED" else "o"
    if primary in {"INTERPRETING", "VERIFYING"}:
        eye = ">" if animation_tick % 2 else "<"
    mouths = {
        "HEARING": ("\\___/", " ___ "),
        "DICTATING": ("\\___/", " --- "),
        "INTERPRETING": (" ... ", " --- "),
        "VERIFYING": (" ... ", " === "),
        "MUTED": (" --- ", " --- "),
    }
    mouth = mouths.get(primary, ("\\___/", "\\___/"))[animation_tick % 2]
    radio = (" )))", "  ))")[animation_tick % 2] if primary == "HEARING" else ""
    return [
        "        .-~~~~~~~~~-.",
        '      .\'  .-"""""-.  \'.',
        "     /   /         \\   \\",
        f"    /   |  {eye}     {eye}  |   \\",
        f"   |    |     ^     |    |{radio}",
        f"   |    |   {mouth}   |    |",
        "   |     \\         /     |",
        "   |      '.___.'        |",
        "   |      /|   |\\        |",
        "   |     / |   | \\       |",
        "    \\___/  |   |  \\_____/",
        "       /___|___|___\\",
    ]


def _status_groups(state: dict[str, Any], width: int) -> list[list[str]]:
    phase = str(state.get("phase") or "starting").replace("_", " ").upper()
    mode = str(state.get("mode") or "detecting").replace("_", " ").upper()
    primary = _primary_state(state)
    delivery = str(state.get("delivery_status") or "").replace("_", " ").upper()
    live_transcript = (
        state.get("live_transcript")
        if str(state.get("phase") or "")
        in {
            "listening",
            "transcribing",
            "interpreting",
            "verifying",
            "awaiting_end_phrase",
        }
        else None
    )
    badge = f"[{primary}]"
    if delivery in {"SENT", "NOT SENT", "UNKNOWN"} and primary not in {
        "HEARING",
        "INTERPRETING",
        "VERIFYING",
        "DICTATING",
    }:
        badge += f" [{delivery}]"
    chosen = f"action={_text(state.get('chosen_action'))}"
    if state.get("chosen_mode"):
        chosen += f" · mode={_text(state['chosen_mode'])}"
    chosen += f" · target={_target(state.get('chosen_target'))}"
    return [
        _wrap(f"{badge}  MODE {mode} · PHASE {phase}", width=width),
        _field("WAITING", state.get("waiting_for") or "microphone startup", width),
        _field("TRANSCRIPT", live_transcript or "—", width),
        _field("BUFFER", state.get("dictation_buffer") or "—", width),
        _field("CHOSEN", chosen, width),
        _field("RESULT", state.get("last_result") or "—", width),
    ]


def _compact_groups(groups: list[list[str]], budget: int, width: int) -> list[str]:
    if budget <= 0:
        return []
    if sum(map(len, groups)) <= budget:
        return [line for group in groups for line in group]
    if budget < len(groups):
        priorities = (0, 2, 5, 1, 3, 4)
        return [
            _fit(groups[index][0], width, ellipsis=True)
            for index in priorities[:budget]
        ]

    allocation = [1] * len(groups)
    remaining = budget - len(groups)
    for index in (2, 3, 5, 1, 4, 0):
        extra = min(remaining, len(groups[index]) - 1)
        allocation[index] += extra
        remaining -= extra
    output: list[str] = []
    for group, count in zip(groups, allocation, strict=True):
        selected = group[:count]
        if count < len(group):
            selected[-1] = _fit(selected[-1] + " …", width, ellipsis=True)
        output.extend(selected)
    return output


def render_dashboard(
    state: dict[str, Any],
    history: list[str],
    *,
    columns: int,
    rows: int,
    animation_tick: int = 0,
    allow_art: bool = True,
) -> str:
    """Render exactly one screen, with recent history above bottom live status."""
    width = max(1, columns)
    height = max(1, rows)
    art = _assistant_art(state, animation_tick)
    art_width = max(map(_display_width, art))
    avatar = str(state.get("assistant_avatar") or DEFAULT_ASSISTANT_AVATAR).lower()
    show_art = (
        allow_art
        and avatar != "none"
        and width >= art_width + 42
        and height >= len(art) + 3
    )
    status_width = width - art_width - 2 if show_art else width
    groups = _status_groups(state, status_width)
    minimum_history = 3 if show_art else 2 if height >= 8 else 0
    status_budget = max(1, height - minimum_history)
    if show_art:
        panel_height = min(status_budget, max(len(art), sum(map(len, groups))))
        details = _compact_groups(groups, panel_height, status_width)
        details.extend([""] * (panel_height - len(details)))
        art = art[:panel_height]
        art.extend([""] * (panel_height - len(art)))
        status = [
            f"{_fit(picture, art_width):<{art_width}}  {_fit(detail, status_width)}"
            for picture, detail in zip(art, details, strict=True)
        ]
    else:
        status = _compact_groups(groups, status_budget, width)
    history_budget = height - len(status)

    upper: list[str] = []
    if history_budget:
        upper.append(_fit(" RECENT ACTIVITY", width, ellipsis=True))
        event_budget = history_budget - 1
        wrapped_events: list[str] = []
        for line in reversed(history):
            wrapped = _wrap(line, width=width)
            if len(wrapped) > event_budget - len(wrapped_events):
                remaining = event_budget - len(wrapped_events)
                if remaining > 0 and not wrapped_events:
                    wrapped_events = wrapped[:remaining]
                    if remaining < len(wrapped):
                        wrapped_events[-1] = _fit(
                            wrapped_events[-1] + " …", width, ellipsis=True
                        )
                break
            wrapped_events[0:0] = wrapped
        if event_budget and not wrapped_events:
            wrapped_events = [_fit(" (waiting for activity)", width, ellipsis=True)]
        upper.extend([""] * max(0, event_budget - len(wrapped_events)))
        upper.extend(wrapped_events)

    frame = upper + status
    if len(frame) < height:
        frame[0:0] = [""] * (height - len(frame))
    return "\n".join(_fit(line, width) for line in frame[-height:])


def _read_complete_lines(stream: TextIO, pending: str) -> tuple[list[str], str]:
    """Read appended journal data without treating a partial final line as an event."""
    data = pending + stream.read()
    if not data:
        return [], pending
    parts = data.split("\n")
    return parts[:-1], parts[-1]


def _inferred_daemon_mode(state: dict[str, Any]) -> str | None:
    """Infer only states that make a redundant mic control unambiguous."""
    phase = str(state.get("phase") or "").lower()
    if phase == "muted":
        return "mute"
    if phase in {
        "ready",
        "listening",
        "transcribing",
        "interpreting",
        "verifying",
        "awaiting_end_phrase",
    }:
        return "listen"
    return None


class ControlButton(Button):
    """Button requiring a same-cell left press/release before activation."""

    def __init__(self, label: str, *, id: str, variant: str = "default") -> None:
        super().__init__(label, id=id, variant=variant, compact=True)
        self._armed_at: tuple[int, int] | None = None
        self._release_authorized = False

    def disarm(self) -> None:
        self._armed_at = None

    def _on_mouse_down(self, event: events.MouseDown) -> None:
        disarm_peers = getattr(self.app, "_button_mouse_down", None)
        if disarm_peers is not None:
            disarm_peers(self)
        self._armed_at = (
            (int(event.screen_x), int(event.screen_y)) if event.button == 1 else None
        )
        # Do not delegate to Button's mouse broker: activation is release-only.
        event.stop()

    def _on_mouse_up(self, event: events.MouseUp) -> None:
        released_at = (int(event.screen_x), int(event.screen_y))
        deliberate = event.button == 1 and self._armed_at == released_at
        self._armed_at = None
        event.stop()
        if deliberate:
            self._release_authorized = True
            try:
                super().press()
            finally:
                self._release_authorized = False

    def press(self) -> Button:
        """Reject framework click synthesis not authorized by our MouseUp."""
        if self._release_authorized:
            return super().press()
        return self

    def _on_click(self, event: events.Click) -> None:
        # App-generated Click follows MouseUp. Activation already happened above,
        # where button identity, left-button use, and exact release cell are known.
        event.stop()


class ControlFinished(Message):
    def __init__(
        self,
        action: str,
        result: dict[str, Any] | None,
        error: str | None,
        reconciliation: dict[str, Any] | None = None,
        reconciliation_error: str | None = None,
    ) -> None:
        super().__init__()
        self.action = action
        self.result = result
        self.error = error
        self.reconciliation = reconciliation
        self.reconciliation_error = reconciliation_error


class ControlBar(Horizontal):
    """Notify the app immediately when the terminal changes control layout."""

    def on_resize(self, _event: events.Resize) -> None:
        callback = getattr(self.app, "_layout_changed", None)
        if callback is not None:
            self.app.call_after_refresh(callback)


class ActivityApp(App[None]):
    """Interactive activity viewer backed by the daemon control socket."""

    TITLE = "voicerdr activity"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("m", "toggle_mode", "Toggle microphone", show=False, priority=True),
        Binding("q", "quit_viewer", "Quit viewer", show=False, priority=True),
    ]
    CSS = """
    Screen {
        layout: vertical;
        overflow: hidden hidden;
    }

    #dashboard {
        width: 100%;
        height: 1fr;
        overflow: hidden hidden;
    }

    #controls {
        width: 100%;
        height: 1;
        align: center middle;
        overflow: hidden hidden;
    }

    ControlButton {
        width: auto;
        min-width: 3;
        height: 1;
        padding: 0 1;
        margin: 0;
        border: none;
    }

    ControlButton.narrow {
        width: auto;
        min-width: 6;
        padding: 0;
    }

    #quit-hint {
        width: auto;
        height: 1;
        padding: 0 1;
    }

    #quit-hint.narrow {
        width: 1;
        padding: 0;
    }
    """

    def __init__(
        self,
        path: Path,
        *,
        state_path: Path,
        history_lines: int,
        debug: bool,
        control_client: Any | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.state_path = state_path
        self.history: deque[str] = deque(maxlen=max(1, history_lines))
        self.debug_events = debug
        self.control_client = control_client or ControlClient(
            state_path.with_name("control.sock")
        )
        self.state: dict[str, Any] = {}
        self.daemon_mode: str | None = None
        self.control_outcome_unknown = False
        self.mic_transition_pending = False
        self.mic_controls_frozen = False
        self.feedback: str | None = None
        self.pending_action: str | None = None
        self.animation_tick = 0
        self._pending_text = ""
        self._last_state_stamp = -1
        self._next_status_check = 0.0
        self._stream: TextIO | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="dashboard", markup=False)
        with ControlBar(id="controls"):
            yield ControlButton("m Mute", id="mode-toggle", variant="warning")
            yield Label("q Quit viewer", id="quit-hint", markup=False)

    def on_mount(self) -> None:
        self._stream = self.path.open("r", encoding="utf-8", errors="replace")
        initial, self._pending_text = _read_complete_lines(
            self._stream, self._pending_text
        )
        self.history.extend(
            render_line(line, debug=self.debug_events) for line in initial
        )
        self._load_state()
        self._sync_controls()
        self.set_interval(0.1, self._poll_activity)
        self.set_interval(0.4, self._animate_dashboard)
        self.call_after_refresh(self._redraw)

    def on_unmount(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _layout_changed(self) -> None:
        self._disarm_control_buttons()
        self._sync_controls()
        self._redraw()

    def _disarm_control_buttons(
        self, except_button: ControlButton | None = None
    ) -> None:
        for button in self.query(ControlButton):
            if button is not except_button:
                button.disarm()

    def _button_mouse_down(self, button: ControlButton) -> None:
        """Disarm peers before the button stops MouseDown propagation."""
        self._disarm_control_buttons(except_button=button)

    def on_resize(self, _event: events.Resize) -> None:
        # A hidden control bar cannot observe the resize that makes it safe to
        # show again, so the app also watches terminal size changes.
        self.call_after_refresh(self._layout_changed)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        # Clear an abandoned press when another widget receives the next press.
        self._disarm_control_buttons(
            event.widget if isinstance(event.widget, ControlButton) else None
        )

    def on_mouse_up(self, _event: events.MouseUp) -> None:
        # A release outside the originally pressed button must not arm a replay.
        self._disarm_control_buttons()

    def _load_state(self) -> bool:
        state = _read_state(self.state_path)
        changed = state != self.state
        self.state = state
        self._apply_activity_safety_state(state)
        inferred = _inferred_daemon_mode(state)
        if inferred is not None and not self.control_outcome_unknown:
            self.daemon_mode = inferred
        try:
            self._last_state_stamp = self.state_path.stat().st_mtime_ns
        except OSError:
            self._last_state_stamp = 0
        return changed

    def _apply_activity_safety_state(self, state: dict[str, Any]) -> None:
        """Merge file state; never clear socket-confirmed frozen/pending flags."""
        phase = str(state.get("phase") or "").lower()
        if phase == "muting" or state.get("mic_transition_pending") is True:
            self.mic_transition_pending = True
        elif state.get("mic_transition_pending") is False and phase in {
            "muted",
            "shutting_down",
        }:
            # Only a terminal post-mute phase proves a newer file revision
            # finished the transition (older snapshots can race the RPC).
            self.mic_transition_pending = False
        # File snapshots may strengthen frozen state but must not clear it;
        # only a later live Status response may.
        if phase == "shutting_down" or state.get("delivery_closed") is True:
            self.mic_controls_frozen = True

    def _poll_activity(self) -> None:
        if self._stream is None or not self.is_running:
            return
        changed = False
        appended, self._pending_text = _read_complete_lines(
            self._stream, self._pending_text
        )
        if appended:
            self.history.extend(
                render_line(line, debug=self.debug_events) for line in appended
            )
            changed = True
        try:
            state_stamp = self.state_path.stat().st_mtime_ns
        except OSError:
            state_stamp = 0
        if state_stamp != self._last_state_stamp:
            changed = self._load_state() or changed
            self._sync_controls()
        if changed:
            self._redraw()
        self._reconcile_controls()

    def _reconcile_controls(self) -> None:
        """Confirm completion/restart over the socket before unlocking controls."""
        if (
            self.pending_action is not None
            or time.monotonic() < self._next_status_check
        ):
            return
        transition_finished = (
            self.mic_transition_pending
            and self.state.get("mic_transition_pending") is False
        )
        daemon_restarted = (
            self.mic_controls_frozen and self.state.get("delivery_closed") is False
        )
        if not (transition_finished or daemon_restarted):
            return
        self._next_status_check = time.monotonic() + 1.0
        self.pending_action = "status"
        self._sync_controls()
        self._send_control("status")

    def _animate_dashboard(self) -> None:
        if not self.is_mounted:
            return
        self.animation_tick += 1
        self._redraw()

    def _redraw(self) -> None:
        if not self.is_mounted or not self.is_running:
            return
        dashboard = self.query_one("#dashboard", Static)
        if dashboard.size.width < 1 or dashboard.size.height < 1:
            return
        display_state = dict(self.state)
        if self.control_outcome_unknown:
            display_state["control_outcome_unknown"] = True
            display_state["mode"] = "unknown"
        if self.feedback:
            display_state["last_result"] = self.feedback
        dashboard.update(
            render_dashboard(
                display_state,
                list(self.history),
                columns=dashboard.size.width,
                rows=dashboard.size.height,
                animation_tick=self.animation_tick,
                allow_art=True,
            )
        )

    def _sync_controls(self) -> None:
        if not self.is_mounted or not self.is_running:
            return
        controls = self.query_one("#controls", ControlBar)
        controls_visible = self.size.width >= MIN_CONTROL_WIDTH
        controls.display = controls_visible
        narrow = self.size.width < 32
        button = self.query_one("#mode-toggle", ControlButton)
        action = self._toggle_action(controls_visible=controls_visible, narrow=narrow)
        previous_action = getattr(button, "control_action", None)
        if action != previous_action:
            # A press begun under one authoritative mode cannot be released
            # after a state/layout change to activate the opposite operation.
            button.disarm()
        button.control_action = action
        button.label = "m Unmute" if action == "listen" else "m Mute"
        button.variant = "success" if action == "listen" else "warning"
        button.set_class(narrow, "narrow")
        button.disabled = action is None
        if not controls_visible:
            button.disarm()
        quit_hint = self.query_one("#quit-hint", Label)
        quit_hint.update("q" if narrow else "q Quit viewer")
        quit_hint.set_class(narrow, "narrow")

    def _toggle_action(self, *, controls_visible: bool, narrow: bool) -> str | None:
        """Return only the action that is safe in the current authoritative state."""
        if (
            not controls_visible
            or self.pending_action is not None
            or self.mic_transition_pending
            or self.mic_controls_frozen
        ):
            return None

        phase = str(self.state.get("phase") or "").lower()
        unsafe_for_listen = (
            narrow
            or self.control_outcome_unknown
            or phase in {"error", "starting", "muting", "shutting_down"}
        )
        if not unsafe_for_listen and self.daemon_mode == "mute":
            return "listen"
        if self.daemon_mode == "mute":
            return None
        # Unknown/stale/error may request mute, never Listen.
        return "mute"

    def action_quit_viewer(self) -> None:
        self.exit()

    def action_toggle_mode(self) -> None:
        button = self.query_one("#mode-toggle", ControlButton)
        action = getattr(button, "control_action", None)
        if action not in {"mute", "listen"} or button.disabled:
            return
        self.pending_action = action
        self.feedback = f"Requesting {action}; awaiting authoritative response."
        self._sync_controls()
        self._redraw()
        self._send_control(action)

    def on_button_pressed(self, message: Button.Pressed) -> None:
        if message.button.id == "mode-toggle":
            self.action_toggle_mode()

    def _send_control(self, action: str) -> None:
        threading.Thread(
            target=self._control_worker,
            args=(action,),
            name=f"voicerdr-activity-{action}",
            daemon=True,
        ).start()

    def _control_worker(self, action: str) -> None:
        try:
            if action == "status":
                result = self.control_client.status(notify=False)
            else:
                operation = getattr(self.control_client, action)
                result = operation()
        except Exception as exc:  # noqa: BLE001 - errors belong in the activity UI
            reconciliation = None
            reconciliation_error = None
            if action in {"mute", "listen"}:
                try:
                    reconciliation = self.control_client.status()
                except Exception as status_exc:  # noqa: BLE001 - shown in the UI
                    reconciliation_error = str(status_exc)
            self.post_message(
                ControlFinished(
                    action,
                    None,
                    str(exc),
                    reconciliation,
                    reconciliation_error,
                )
            )
        else:
            self.post_message(ControlFinished(action, result, None))

    def _apply_response_state(self, result: dict[str, Any]) -> bool:
        live_capture = result.get("live_capture")
        if isinstance(live_capture, dict):
            self.state = live_capture
            self._apply_activity_safety_state(live_capture)
        else:
            self._load_state()
        if "mic_transition_pending" in result:
            self.mic_transition_pending = bool(result["mic_transition_pending"])
        if "delivery_closed" in result:
            self.mic_controls_frozen = bool(result["delivery_closed"])
        returned_mode = result.get("mode")
        if returned_mode not in {"listen", "mute"}:
            return False
        self.daemon_mode = str(returned_mode)
        self.control_outcome_unknown = False
        return True

    def on_control_finished(self, message: ControlFinished) -> None:
        if message.action != self.pending_action:
            return
        self.pending_action = None
        if message.error is not None:
            reconciled = bool(
                message.reconciliation is not None
                and self._apply_response_state(message.reconciliation)
            )
            if reconciled:
                summary = self._response_summary("status", message.reconciliation or {})
                self.feedback = (
                    f"{message.action.title()} response failed: {message.error}; "
                    f"follow-up {summary.lower()}"
                )
            elif message.action in {"mute", "listen"}:
                self.daemon_mode = None
                self.control_outcome_unknown = True
                status_detail = (
                    f" Follow-up status failed: {message.reconciliation_error}."
                    if message.reconciliation_error
                    else " Follow-up status did not report a valid mic mode."
                )
                self.feedback = (
                    f"{message.action.title()} outcome unknown: {message.error}."
                    f"{status_detail} Use m to request a safe mute; CLI status "
                    "remains available for inspection."
                )
            else:
                self.feedback = f"{message.action.title()} failed: {message.error}."
            payload: dict[str, Any] = {
                "time": datetime.now().astimezone().isoformat(),
                "event": "viewer_control",
                "action": message.action,
                "error": message.error,
                "result": {"message": self.feedback},
            }
        else:
            result = message.result or {}
            # The completed socket response is authoritative over a state file
            # snapshot that may have been read just before its atomic replace.
            authoritative = self._apply_response_state(result)
            if not authoritative:
                self.daemon_mode = None
                self.control_outcome_unknown = True
                self.feedback = (
                    f"{message.action.title()} response did not report a valid "
                    "microphone mode. Use m to request a safe mute; CLI status "
                    "remains available for inspection."
                )
            else:
                self.feedback = self._response_summary(message.action, result)
            payload = {
                "time": datetime.now().astimezone().isoformat(),
                "event": "viewer_control",
                "action": message.action,
                "result": {"message": self.feedback},
            }
        self.history.append(render_event(payload, debug=self.debug_events))
        self._sync_controls()
        self._redraw()

    @staticmethod
    def _response_summary(action: str, result: dict[str, Any]) -> str:
        mode = result.get("mode")
        voice = result.get("voice_running")
        fields = []
        if mode is not None:
            fields.append(f"mic={mode}")
        if voice is not None:
            fields.append(f"voice={'on' if voice else 'off'}")
        if result.get("mic_transition_pending"):
            fields.append("mute=closing")
        if result.get("delivery_closed"):
            fields.append("delivery=closed")
        if result.get("voice_error"):
            fields.append(f"error={result['voice_error']}")
        preference = result.get("mic_preference")
        if isinstance(preference, dict) and preference.get("error"):
            fields.append(f"preference_error={preference['error']}")
        details = " · ".join(fields) or "no state fields returned"
        return f"{action.title()} response: {details}."


@contextmanager
def _graceful_viewer_signals(app: ActivityApp):
    """Route termination signals through Textual's terminal cleanup path."""
    previous: dict[signal.Signals, Any] = {}

    def exit_viewer(_signum: int, _frame: FrameType | None) -> None:
        app.exit()

    try:
        for handled in (signal.SIGHUP, signal.SIGTERM):
            previous[handled] = signal.getsignal(handled)
            signal.signal(handled, exit_viewer)
        yield
    finally:
        for handled, handler in previous.items():
            signal.signal(handled, handler)


def follow(
    path: Path,
    *,
    state_path: Path | None = None,
    history_lines: int = 250,
    debug: bool = False,
) -> None:
    state_path = state_path or path.with_name("activity_state.json")
    while not path.exists():
        time.sleep(0.2)
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        app = ActivityApp(
            path,
            state_path=state_path,
            history_lines=history_lines,
            debug=debug,
        )
        with _graceful_viewer_signals(app):
            app.run(mouse=True)
        return

    # Redirected output remains a deterministic, escape-free snapshot stream.
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        history: deque[str] = deque(maxlen=max(1, history_lines))
        pending = ""
        initial, pending = _read_complete_lines(stream, pending)
        history.extend(render_line(line, debug=debug) for line in initial)
        last_state_stamp = -1
        last_size: os.terminal_size | None = None
        animation_tick = 0
        while True:
            changed = False
            appended, pending = _read_complete_lines(stream, pending)
            if appended:
                history.extend(render_line(line, debug=debug) for line in appended)
                changed = True
            try:
                state_stamp = state_path.stat().st_mtime_ns
            except OSError:
                state_stamp = 0
            if state_stamp != last_state_stamp:
                last_state_stamp = state_stamp
                changed = True
            size = shutil.get_terminal_size((100, 32))
            if size != last_size:
                last_size = size
                changed = True
            if changed:
                dashboard = render_dashboard(
                    _read_state(state_path),
                    list(history),
                    columns=size.columns,
                    rows=size.lines,
                    animation_tick=animation_tick,
                    allow_art=False,
                )
                sys.stdout.write(dashboard + "\n")
                sys.stdout.flush()
            time.sleep(0.1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Follow voicerdr activity history")
    parser.add_argument("path", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--lines", type=int, default=250)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show complete raw JSON payloads instead of concise event fields",
    )
    args = parser.parse_args(argv)
    try:
        follow(
            args.path,
            state_path=args.state,
            history_lines=max(1, args.lines),
            debug=args.debug,
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
