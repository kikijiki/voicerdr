"""Operator-facing lifecycle helpers for optional systemd supervision."""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from voicerdr.config import seed_config_files
from voicerdr.control_client import ControlClient, ControlClientError
from voicerdr.ensure import _startup_transaction, ensure_daemon, stop_daemon
from voicerdr.paths import RuntimePaths, ensure_dirs

SERVICE_NAME = "voicerdr.service"
MANAGED_MARKER = "# Managed by voicerdr; rerun `voicerdr service install` to update."


class OperationsError(RuntimeError):
    """An actionable lifecycle/configuration failure."""


@dataclass(frozen=True)
class ServiceFiles:
    unit: Path
    environment: Path


def service_files(paths: RuntimePaths) -> ServiceFiles:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return ServiceFiles(
        unit=config_home / "systemd" / "user" / SERVICE_NAME,
        environment=paths.state_dir / "systemd.env",
    )


def service_is_installed(paths: RuntimePaths) -> bool:
    unit = service_files(paths).unit
    if not unit.is_file():
        return False
    try:
        return MANAGED_MARKER in unit.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _escape_systemd_unit_path(value: str) -> str:
    """Escape a path for WorkingDirectory=/EnvironmentFile= style settings.

    Those parsers treat surrounding quotes as part of the path (systemd 261),
    so values must remain unquoted. Specifiers and env expansion still apply.
    """
    if "\0" in value or "\n" in value or "\r" in value:
        raise OperationsError("systemd values may not contain NUL or newlines")
    return value.replace("%", "%%").replace("$", "$$")


def _quote_systemd(value: str, *, unit: bool = False) -> str:
    if "\0" in value or "\n" in value or "\r" in value:
        raise OperationsError("systemd values may not contain NUL or newlines")
    # ExecStart argv and EnvironmentFile *contents* accept C-style quoting.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    if unit:
        # Unit files expand specifiers and environment variables even in quotes.
        escaped = escaped.replace("%", "%%").replace("$", "$$")
    return '"' + escaped + '"'


def _template(paths: RuntimePaths) -> str:
    source = paths.plugin_root / "systemd" / "voicerdr.service.in"
    if source.is_file():
        return source.read_text(encoding="utf-8")
    try:
        from importlib.resources import files

        packaged = files("voicerdr").joinpath("resources/voicerdr.service.in")
        return packaged.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise OperationsError(
            f"systemd template not found (looked for {source})"
        ) from exc


def _launcher_executable(paths: RuntimePaths) -> Path:
    launcher = paths.plugin_root / "scripts" / "run.sh"
    if launcher.is_file() and os.access(launcher, os.X_OK):
        return launcher.absolute()
    raise OperationsError(f"runtime launcher is missing or not executable: {launcher}")


def render_service(paths: RuntimePaths) -> str:
    files = service_files(paths)
    replacements = {
        "@MANAGED_MARKER@": MANAGED_MARKER,
        "@WORKING_DIRECTORY@": _escape_systemd_unit_path(str(paths.plugin_root)),
        "@LAUNCHER@": _quote_systemd(str(_launcher_executable(paths)), unit=True),
        "@ENVIRONMENT_FILE@": _escape_systemd_unit_path(str(files.environment)),
    }
    rendered = _template(paths)
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    leftovers = [token for token in replacements if token in rendered]
    if leftovers:
        raise OperationsError(f"unexpanded systemd template tokens: {leftovers}")
    return rendered


def _atomic_write(path: Path, content: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_service_environment(paths: RuntimePaths) -> Path:
    ensure_dirs(paths)
    values = {
        "HERDR_PLUGIN_ROOT": str(paths.plugin_root),
        "HERDR_PLUGIN_CONFIG_DIR": str(paths.config_dir),
        "HERDR_PLUGIN_STATE_DIR": str(paths.state_dir),
        "HERDR_PLUGIN_ID": "voicerdr",
        "HERDR_BIN_PATH": paths.herdr_bin,
        "HERDR_ENV": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if paths.herdr_socket:
        values["HERDR_SOCKET_PATH"] = paths.herdr_socket
    for name in (
        "PATH",
        "LD_LIBRARY_PATH",
        "XDG_CACHE_HOME",
        "MOONSHINE_VOICE_CACHE",
        "NLTK_DATA",
        "PULSE_SERVER",
        "PIPEWIRE_RUNTIME_DIR",
    ):
        if os.environ.get(name):
            values[name] = os.environ[name]
    content = "".join(
        f"{key}={_quote_systemd(value)}\n" for key, value in values.items()
    )
    environment = service_files(paths).environment
    _atomic_write(environment, content, mode=0o600)
    return environment


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["systemctl", "--user", *args],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise OperationsError(f"cannot run systemctl --user: {exc}") from exc
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise OperationsError(
            f"systemctl --user {' '.join(args)} failed"
            + (f": {detail}" if detail else "")
        )
    return result


def _health(paths: RuntimePaths) -> dict | None:
    try:
        status = ControlClient(paths.control_socket, timeout=1.0).ping()
    except ControlClientError:
        return None
    if status.get("herdr_socket") != paths.herdr_socket:
        return None
    return status


def _wait_for_health(paths: RuntimePaths, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = _health(paths)
        if status is not None:
            return status
        time.sleep(0.15)
    raise OperationsError(
        f"systemd service did not become ready; inspect `journalctl --user -u {SERVICE_NAME}`"
    )


def install_service(
    paths: RuntimePaths, *, enable: bool = False, start: bool = False
) -> dict:
    """Install/update the unit without silently opting the user into autostart."""
    ensure_dirs(paths)
    seed_config_files(paths)
    files = service_files(paths)
    if files.unit.exists():
        existing = files.unit.read_text(encoding="utf-8", errors="replace")
        if MANAGED_MARKER not in existing:
            raise OperationsError(f"refusing to overwrite unmanaged unit {files.unit}")
    _atomic_write(files.unit, render_service(paths), mode=0o644)
    write_service_environment(paths)
    _systemctl("daemon-reload")
    if enable:
        _systemctl("enable", SERVICE_NAME)
    result: dict = {
        "installed": True,
        "enabled": enable,
        "unit": str(files.unit),
        "environment": str(files.environment),
    }
    if start:
        ensured = ensure_managed_daemon(paths)
        result["started"] = True
        result["status"] = ensured.get("status")
    return result


def ensure_systemd_service(paths: RuntimePaths, *, timeout: float = 20.0) -> dict:
    with _startup_transaction(paths):
        return _ensure_systemd_service_serialized(paths, timeout=timeout)


def _ensure_systemd_service_serialized(
    paths: RuntimePaths, *, timeout: float = 20.0
) -> dict:
    if not service_is_installed(paths):
        raise OperationsError(
            "systemd unit is not installed; run `voicerdr service install`"
        )
    write_service_environment(paths)
    active = (
        _systemctl("is-active", "--quiet", SERVICE_NAME, check=False).returncode == 0
    )
    healthy = _health(paths)
    if active and healthy is not None:
        return {
            "ensured": True,
            "already_running": True,
            "systemd": True,
            "status": healthy,
        }

    # Converge cleanly when migrating a detached daemon or rebinding a Herdr socket.
    # Snapshot the daemon-reported microphone mode before systemd can terminate
    # it and destroy that evidence with shutdown UI state.
    stopped = stop_daemon(paths, wait_secs=3.0, _serialized=False)
    if not stopped.get("ok"):
        raise OperationsError("existing daemon could not be stopped safely")
    if active:
        _systemctl("stop", SERVICE_NAME)
    _systemctl("start", SERVICE_NAME)
    healthy = _wait_for_health(paths, timeout)
    return {
        "ensured": True,
        "already_running": False,
        "systemd": True,
        "status": healthy,
    }


def ensure_managed_daemon(paths: RuntimePaths, *, timeout: float = 20.0) -> dict:
    """Use systemd when installed, otherwise retain the portable detached mode."""
    if service_is_installed(paths):
        return ensure_systemd_service(paths, timeout=timeout)
    return ensure_daemon(paths, timeout=timeout)


def enable_service(paths: RuntimePaths) -> dict:
    if not service_is_installed(paths):
        raise OperationsError(
            "systemd unit is not installed; run `voicerdr service install`"
        )
    _systemctl("enable", SERVICE_NAME)
    ensured = ensure_systemd_service(paths)
    return {"enabled": True, "started": True, "status": ensured.get("status")}


def disable_service(paths: RuntimePaths) -> dict:
    """Disable supervision and guarantee the install-owned daemon is gone."""
    files = service_files(paths)
    if files.unit.exists() and not service_is_installed(paths):
        raise OperationsError(f"refusing to disable unmanaged unit {files.unit}")
    stopped = stop_daemon(paths)
    if service_is_installed(paths):
        _systemctl("disable", "--now", SERVICE_NAME)
        # Catch a detached process or a race with a previous restart policy.
        stopped = stop_daemon(paths)
    return {"disabled": True, "stop": stopped, "ok": bool(stopped.get("ok"))}


def uninstall_service(paths: RuntimePaths) -> dict:
    files = service_files(paths)
    had_unit = files.unit.exists()
    if had_unit and not service_is_installed(paths):
        raise OperationsError(f"refusing to remove unmanaged unit {files.unit}")
    disabled = disable_service(paths)
    if not disabled.get("ok"):
        raise OperationsError("daemon did not stop; refusing to remove service files")
    if had_unit:
        existing = files.unit.read_text(encoding="utf-8", errors="replace")
        if MANAGED_MARKER not in existing:
            raise OperationsError(f"refusing to remove unmanaged unit {files.unit}")
        files.unit.unlink()
    try:
        files.environment.unlink()
    except FileNotFoundError:
        pass
    if had_unit:
        _systemctl("daemon-reload")
    return {"uninstalled": True, "unit": str(files.unit), "disabled": disabled}


def service_status(paths: RuntimePaths) -> dict:
    files = service_files(paths)
    health = _health(paths)
    if not files.unit.is_file():
        return {
            "installed": False,
            "managed": False,
            "unit": str(files.unit),
            "active": "inactive",
            "enabled": "disabled",
            "healthy": health is not None,
            "daemon": health,
        }
    active = _systemctl("is-active", SERVICE_NAME, check=False)
    enabled = _systemctl("is-enabled", SERVICE_NAME, check=False)
    return {
        "installed": files.unit.is_file(),
        "managed": service_is_installed(paths),
        "unit": str(files.unit),
        "active": active.stdout.strip() or "unknown",
        "enabled": enabled.stdout.strip() or "unknown",
        "healthy": health is not None,
        "daemon": health,
    }


def print_operation_result(result: dict) -> None:
    print(json.dumps({"ok": bool(result.get("ok", True)), "result": result}, indent=2))
