import time
from dataclasses import dataclass, field

from voicerdr.config import AppConfig
from voicerdr.space_labels import strip_number_prefix

TALK_MODES = frozenset({"silent", "blocked_only", "milestones", "verbose"})


@dataclass
class TalkPolicy:
    config: AppConfig
    mic_mode: str = "mute"  # listen | mute
    speak_enabled: bool | None = None
    talk_mode: str | None = None
    quiet_workspaces: set[str] = field(default_factory=set)
    _last_spoken_at: dict[str, float] = field(default_factory=dict)
    _last_spoken_status: dict[str, str] = field(default_factory=dict)
    _prompted_targets: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.speak_enabled is None:
            self.speak_enabled = self.config.speak_enabled
        if self.talk_mode is None:
            self.talk_mode = self.config.talk_mode
        if self.talk_mode not in TALK_MODES:
            self.talk_mode = "blocked_only"
        quiet = (*self.quiet_workspaces, *self.config.quiet_workspaces)
        self.quiet_workspaces = {
            normalized for item in quiet if (normalized := _normalize_workspace(item))
        }
        if self.config.mute_on_start:
            self.mic_mode = "mute"
        else:
            self.mic_mode = "listen"

    def set_mic_mode(self, mode: str) -> None:
        if mode not in ("listen", "mute"):
            raise ValueError(f"unknown mode: {mode}")
        self.mic_mode = mode

    def set_talk_mode(self, mode: str) -> None:
        if mode not in TALK_MODES:
            raise ValueError(f"unknown talk mode: {mode}")
        self.talk_mode = mode

    def set_workspace_quiet(self, workspace: str, *, quiet: bool) -> None:
        key = _normalize_workspace(workspace)
        if not key:
            raise ValueError("workspace required")
        if quiet:
            self.quiet_workspaces.add(key)
        else:
            self.quiet_workspaces.discard(key)

    def mark_prompted(self, target: str) -> None:
        self._prompted_targets.add(target)

    def should_announce(
        self,
        *,
        pane_id: str,
        status: str,
        workspace_id: str | None = None,
        workspace_label: str | None = None,
    ) -> tuple[bool, str]:
        if not self.speak_enabled:
            return False, "speak_disabled"
        quiet_keys = {
            _normalize_workspace(workspace_id or ""),
            _normalize_workspace(workspace_label or ""),
        }
        if self.quiet_workspaces.intersection(quiet_keys):
            return False, "workspace_quiet"
        if self.talk_mode == "silent":
            return False, "talk_mode:silent"
        if self.talk_mode == "blocked_only" and status != "blocked":
            return False, f"talk_mode:blocked_only:{status}"
        if self.talk_mode == "milestones" and status != "blocked":
            if pane_id not in self._prompted_targets:
                return False, "not_secretary_prompted"
            if status not in self.config.interesting_statuses:
                return False, f"status_not_interesting:{status}"
        if self.talk_mode == "verbose" and not status:
            return False, "empty_status"
        now = time.monotonic()
        last = self._last_spoken_at.get(pane_id, 0.0)
        urgent_transition = (
            status == "blocked" and self._last_spoken_status.get(pane_id) != "blocked"
        )
        if not urgent_transition and now - last < self.config.min_interval_secs:
            return False, "rate_limited"
        return True, "ok"

    def mark_spoken(self, pane_id: str, status: str | None = None) -> None:
        self._last_spoken_at[pane_id] = time.monotonic()
        if status:
            self._last_spoken_status[pane_id] = status
            if status in {"idle", "done"}:
                self._prompted_targets.discard(pane_id)

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.talk_mode,
            "quiet_workspaces": sorted(self.quiet_workspaces),
            "min_interval_secs": self.config.min_interval_secs,
        }

    def clamp_speech(self, text: str) -> str:
        text = " ".join(text.split())
        limit = self.config.max_spoken_chars
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"


def _normalize_workspace(value: str) -> str:
    value = strip_number_prefix(value)
    return "".join(ch for ch in value.casefold().strip() if ch.isalnum() or ch in "-_")
