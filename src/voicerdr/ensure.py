import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from voicerdr.config import seed_config_files
from voicerdr.control_client import ControlClient, ControlClientError
from voicerdr.paths import RuntimePaths, ensure_dirs, resolve_paths


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(paths: RuntimePaths) -> int | None:
    if not paths.pidfile.is_file():
        return None
    try:
        text = paths.pidfile.read_text().strip()
        return int(text)
    except (OSError, ValueError):
        return None


def _read_session(paths: RuntimePaths) -> dict:
    if not paths.session_file.is_file():
        return {}
    try:
        value = json.loads(paths.session_file.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _clear_stale(paths: RuntimePaths) -> None:
    for p in (
        paths.pidfile,
        paths.session_file,
    ):
        try:
            if p.is_socket() or p.is_file():
                p.unlink()
        except OSError:
            pass


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_private_json(target: Path, value: dict) -> None:
    """Replace one launcher-owned state file with a durable private file."""
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd: int | None = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        payload = (json.dumps(value, sort_keys=True) + "\n").encode()
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except OSError:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # An uninspectable state entry is evidence, not a clean first start.
        return True
    return True


def _begin_mic_handoff(paths: RuntimePaths) -> None:
    """Install a crash marker before querying mutable daemon state."""
    _atomic_write_private_json(
        paths.mic_handoff_pending,
        {"version": 1, "state": "capturing"},
    )


def _complete_mic_handoff(paths: RuntimePaths, mode: str) -> None:
    if mode not in {"listen", "mute"}:
        raise ValueError(f"daemon reported invalid microphone mode: {mode!r}")
    _atomic_write_private_json(
        paths.mic_handoff,
        {"version": 1, "mode": mode},
    )
    paths.mic_handoff_pending.unlink()
    _fsync_directory(paths.state_dir)


@contextmanager
def _startup_transaction(paths: RuntimePaths):
    """Serialize launcher health/cleanup/spawn mutations for one state root."""

    # Lock sits in the parent so state files are untouched until we hold it.
    parent = paths.state_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    identity = sha256(str(paths.state_dir.absolute()).encode()).hexdigest()[:20]
    lock_path = parent / f".{paths.state_dir.name}.{identity}.startup.lock"
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        fd = os.open(lock_path, os.O_RDWR)
        created = False
    try:
        if created:
            os.fsync(fd)
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _daemon_healthy(paths: RuntimePaths) -> dict | None:
    client = ControlClient(paths.control_socket, timeout=1.0)
    try:
        status = client.ping()
    except ControlClientError:
        return None
    expected = paths.herdr_socket
    bound = status.get("herdr_socket")
    if bound != expected:
        return None
    return status


def _pgid_of(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def _kill_tree(pid: int, *, sig: int) -> None:
    """Signal a process and its process group (session from start_new_session)."""
    if pid <= 0:
        return
    pgid = _pgid_of(pid)
    targets: list[int] = []
    if pgid and pgid > 1:
        targets.append(-pgid)  # whole group
    targets.append(pid)
    for target in targets:
        try:
            os.kill(target, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def _find_voicerdr_pids(paths: RuntimePaths) -> list[int]:
    """Find leftover daemon PIDs for this install (uv parent + python child)."""
    found: set[int] = set()
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,args="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_s, args = line.split(None, 1)
            pid = int(pid_s)
        except ValueError:
            continue
        # Skip this stop/ensure helper itself.
        if "ctl quit" in args or "stop_daemon" in args:
            continue
        if "daemon" not in args:
            continue
        # Never fall back to every `voicerdr daemon` on the machine: multiple
        # checkouts/sessions may legitimately be running at once. The wrapper
        # exports both values, so argv or the process environment can identify
        # the owning install without relying on a generic process name.
        if _is_rooted_wrapper_command(args, paths) or _pid_has_runtime_paths(
            pid, paths
        ):
            found.add(pid)
    return sorted(found)


def _pid_has_runtime_paths(pid: int, paths: RuntimePaths) -> bool:
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except OSError:
        return False
    values = set(raw.split(b"\0"))
    expected = {
        f"HERDR_PLUGIN_ROOT={paths.plugin_root}".encode(),
        f"HERDR_PLUGIN_STATE_DIR={paths.state_dir}".encode(),
    }
    return expected.issubset(values)


def _is_rooted_wrapper_command(command: str, paths: RuntimePaths) -> bool:
    return (
        str(paths.plugin_root) in command
        and "scripts/run.sh" in command
        and "daemon" in command
    )


def _pid_belongs_to_install(pid: int, paths: RuntimePaths) -> bool:
    """Reject stale pid-file values that have since been reused by another process."""
    if not _pid_alive(pid):
        return False
    if _pid_has_runtime_paths(pid, paths):
        return True
    try:
        cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return False
    command = cmdline.replace(b"\0", b" ").decode(errors="replace")
    return _is_rooted_wrapper_command(command, paths)


def stop_daemon(
    paths: RuntimePaths | None = None,
    *,
    wait_secs: float = 8.0,
    _serialized: bool = True,
) -> dict:
    """Fully stop the daemon: soft quit, then TERM/KILL the whole process group.

    Refuses to begin shutdown if it cannot first install a durable mic handoff
    marker. Otherwise guarantees no leftover ``uv``/``voicerdr daemon`` processes
    for this install.
    """
    paths = paths or resolve_paths()
    if _serialized:
        with _startup_transaction(paths):
            return stop_daemon(paths, wait_secs=wait_secs, _serialized=False)
    ensure_dirs(paths)

    # Collect every related PID before querying or asking the daemon to mutate
    # its shutdown UI. Stale artifacts are evidence of an interrupted daemon,
    # while their total absence is a clean first start and needs no handoff.
    pids: set[int] = set()
    pid = _read_pid(paths)
    if pid and _pid_belongs_to_install(pid, paths):
        pids.add(pid)
    session = _read_session(paths)
    for key in ("pid", "spawn_pid"):
        try:
            val = int(session.get(key) or 0)
        except (TypeError, ValueError):
            val = 0
        if val > 1 and _pid_belongs_to_install(val, paths):
            pids.add(val)
    pids.update(_find_voicerdr_pids(paths))

    daemon_evidence = bool(pids) or any(
        _path_present(path)
        for path in (paths.control_socket, paths.pidfile, paths.session_file)
    )
    handoff_mode: str | None = None
    handoff_complete = False
    soft_ok = False
    if daemon_evidence:
        # This marker is durable before the transition RPC. A launcher crash,
        # a legacy/unresponsive daemon, or a malformed response therefore makes
        # the next daemon fail closed instead of trusting a stale snapshot.
        _begin_mic_handoff(paths)
        try:
            # A concurrent listen may be completing a first model load under the
            # same daemon transition lock. Wait long enough to obtain its final
            # linearized state instead of needlessly degrading a current daemon.
            reported = ControlClient(paths.control_socket, timeout=60.0).handoff_quit()
            mode = reported.get("mode")
            if (
                reported.get("handoff_version") != 1
                or reported.get("quitting") is not True
                or not isinstance(mode, str)
            ):
                raise TypeError("daemon did not confirm an atomic microphone handoff")
            _complete_mic_handoff(paths, mode)
            handoff_mode = mode
            handoff_complete = True
            soft_ok = True
        except OSError:
            # Marker may have been removed before the final directory fsync.
            # Re-assert mute evidence before teardown.
            try:
                _begin_mic_handoff(paths)
            except OSError:
                try:
                    _atomic_write_private_json(
                        paths.mic_handoff,
                        {"version": 1, "mode": "mute"},
                    )
                except OSError:
                    pass
        except (ControlClientError, TypeError, ValueError):
            # Older daemons lack atomic handoff; keep the pending marker and stop muted.
            pass

    if not soft_ok:
        try:
            ControlClient(paths.control_socket, timeout=1.5).quit()
            soft_ok = True
        except ControlClientError:
            pass

    deadline = time.monotonic() + min(2.0, wait_secs)
    while time.monotonic() < deadline:
        if not any(_pid_alive(p) for p in pids) and not _find_voicerdr_pids(paths):
            break
        time.sleep(0.1)

    # Escalate: TERM process groups, then KILL.
    survivors = {p for p in pids if _pid_alive(p)} | set(_find_voicerdr_pids(paths))
    for p in list(survivors):
        _kill_tree(p, sig=signal.SIGTERM)
    time.sleep(0.4)
    survivors = {p for p in survivors if _pid_alive(p)} | set(
        _find_voicerdr_pids(paths)
    )
    for p in list(survivors):
        _kill_tree(p, sig=signal.SIGKILL)
    time.sleep(0.2)

    # A failed final ps/proc discovery must not erase a verified PID from the
    # result. Recheck every known PID independently before claiming success.
    left = sorted(
        {p for p in pids | survivors if _pid_alive(p)} | set(_find_voicerdr_pids(paths))
    )
    if not left:
        _clear_stale(paths)
    return {
        "stopped": True,
        "soft_quit": soft_ok,
        "mic_handoff": handoff_mode if handoff_complete else None,
        "mic_handoff_fail_closed": daemon_evidence and not handoff_complete,
        "killed": sorted(pids),
        "remaining": left,
        "ok": not left,
    }


def _spawn_daemon(paths: RuntimePaths) -> None:
    ensure_dirs(paths)
    seed_config_files(paths)
    log = paths.daemon_log.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["HERDR_PLUGIN_ROOT"] = str(paths.plugin_root)
    env["HERDR_PLUGIN_CONFIG_DIR"] = str(paths.config_dir)
    env["HERDR_PLUGIN_STATE_DIR"] = str(paths.state_dir)
    env["HERDR_PLUGIN_ID"] = "voicerdr"
    if paths.herdr_socket:
        env["HERDR_SOCKET_PATH"] = paths.herdr_socket
    env["HERDR_BIN_PATH"] = paths.herdr_bin
    env["HERDR_ENV"] = "1"

    # Prefer the same uv wrapper Herdr uses so the venv stays consistent.
    run_sh = paths.plugin_root / "scripts" / "run.sh"
    if run_sh.is_file():
        cmd = ["bash", str(run_sh), "daemon", "--foreground"]
    else:
        cmd = [sys.executable, "-m", "voicerdr", "daemon", "--foreground"]

    proc = subprocess.Popen(
        cmd,
        cwd=str(paths.plugin_root),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    spawn_pid = proc.pid
    pgid = _pgid_of(spawn_pid) or spawn_pid
    paths.pidfile.write_text(str(spawn_pid) + "\n")
    try:
        paths.pidfile.chmod(0o600)
    except OSError:
        pass
    # Record spawn leader so stop can kill uv+python as a group.
    try:
        paths.session_file.write_text(
            json.dumps(
                {
                    "spawn_pid": spawn_pid,
                    "pgid": pgid,
                    "herdr_socket": paths.herdr_socket,
                    "started_at": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        paths.session_file.chmod(0o600)
    except OSError:
        pass


def ensure_daemon(paths: RuntimePaths | None = None, *, timeout: float = 20.0) -> dict:
    """Idempotent: leave exactly one healthy daemon for this Herdr socket."""
    paths = paths or resolve_paths()
    with _startup_transaction(paths):
        return _ensure_daemon_serialized(paths, timeout=timeout)


def _ensure_daemon_serialized(paths: RuntimePaths, *, timeout: float) -> dict:
    ensure_dirs(paths)
    seed_config_files(paths)

    healthy = _daemon_healthy(paths)
    if healthy is not None:
        return {"ensured": True, "already_running": True, "status": healthy}

    # Stale / half-dead — wipe the tree, then respawn.
    stopped = stop_daemon(paths, wait_secs=3.0, _serialized=False)
    if not stopped.get("ok"):
        raise RuntimeError("existing daemon could not be stopped safely")
    _clear_stale(paths)
    _spawn_daemon(paths)

    deadline = time.monotonic() + timeout
    last_err = "daemon did not become ready"
    while time.monotonic() < deadline:
        healthy = _daemon_healthy(paths)
        if healthy is not None:
            return {"ensured": True, "already_running": False, "status": healthy}
        time.sleep(0.15)
        pid = _read_pid(paths)
        if pid and not _pid_alive(pid):
            last_err = f"daemon exited early; see {paths.daemon_log}"
            break

    raise RuntimeError(last_err)


def print_ensure_result(result: dict) -> None:
    status = result.get("status") or {}
    print(
        json.dumps(
            {
                "ok": True,
                "already_running": result.get("already_running"),
                "pid": status.get("pid"),
                "mode": status.get("mode"),
                "herdr_socket": status.get("herdr_socket"),
                "subscribed": status.get("subscribed"),
            }
        )
    )
