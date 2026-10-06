import os
from dataclasses import dataclass
from pathlib import Path

PLUGIN_ID = "voicerdr"


@dataclass(frozen=True)
class RuntimePaths:
    plugin_root: Path
    config_dir: Path
    state_dir: Path
    herdr_bin: str
    herdr_socket: str | None

    @property
    def control_socket(self) -> Path:
        return self.state_dir / "control.sock"

    @property
    def pidfile(self) -> Path:
        return self.state_dir / "daemon.pid"

    @property
    def session_file(self) -> Path:
        return self.state_dir / "session.json"

    @property
    def daemon_log(self) -> Path:
        return self.state_dir / "daemon.log"

    @property
    def activity_log(self) -> Path:
        return self.state_dir / "activity.jsonl"

    @property
    def activity_state(self) -> Path:
        return self.state_dir / "activity_state.json"

    @property
    def activity_workspace_file(self) -> Path:
        return self.state_dir / "activity_workspace.json"

    @property
    def activity_workspace_transaction(self) -> Path:
        return self.state_dir / "activity_workspace_transaction.json"

    @property
    def utterance_ledger(self) -> Path:
        return self.state_dir / "consumed_utterances.json"

    @property
    def utterance_ledger_lock(self) -> Path:
        return self.state_dir / "consumed_utterances.lock"

    @property
    def daemon_lock(self) -> Path:
        return self.state_dir / "daemon.lock"

    @property
    def mic_preference(self) -> Path:
        return self.state_dir / "mic_preference.json"

    @property
    def mic_preference_pending(self) -> Path:
        return self.state_dir / "mic_preference.pending"

    @property
    def mic_handoff(self) -> Path:
        return self.state_dir / "mic_handoff.json"

    @property
    def mic_handoff_pending(self) -> Path:
        return self.state_dir / "mic_handoff.pending"

    @property
    def config_toml(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def aliases_toml(self) -> Path:
        return self.config_dir / "aliases.toml"


def resolve_paths(
    *,
    plugin_root: Path | None = None,
    config_dir: Path | None = None,
    state_dir: Path | None = None,
) -> RuntimePaths:
    """Resolve Herdr-injected dirs, with local-dev fallbacks under ~/.config/herdr."""
    root = Path(
        plugin_root
        or os.environ.get("HERDR_PLUGIN_ROOT")
        or Path(__file__).resolve().parents[2]
    )
    home_cfg = Path.home() / ".config" / "herdr" / "plugins"
    cfg = Path(
        config_dir
        or os.environ.get("HERDR_PLUGIN_CONFIG_DIR")
        or (home_cfg / "config" / PLUGIN_ID)
    )
    # Herdr 0.8 seeds state under ~/.local/state/herdr/plugins/<id>.
    default_state = Path.home() / ".local" / "state" / "herdr" / "plugins" / PLUGIN_ID
    st = Path(state_dir or os.environ.get("HERDR_PLUGIN_STATE_DIR") or default_state)
    herdr_bin = os.environ.get("HERDR_BIN_PATH") or "herdr"
    herdr_socket = os.environ.get("HERDR_SOCKET_PATH")
    return RuntimePaths(
        plugin_root=root.resolve(),
        config_dir=cfg,
        state_dir=st,
        herdr_bin=herdr_bin,
        herdr_socket=herdr_socket,
    )


def ensure_dirs(paths: RuntimePaths) -> None:
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths.state_dir.chmod(0o700)
    except OSError:
        pass
