import json
import os
import signal
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voicerdr.config import AppConfig
from voicerdr.control_protocol import ControlRequest
from voicerdr.daemon import Daemon, MicPreferenceError
from voicerdr.ensure import (
    _begin_mic_handoff,
    _complete_mic_handoff,
    ensure_daemon,
    stop_daemon,
)
from voicerdr.paths import RuntimePaths
from voicerdr.talk_policy import TalkPolicy
from voicerdr.voice import VoiceListener


def daemon_at(root: Path, *, mute_on_start: bool = False) -> Daemon:
    daemon = Daemon.__new__(Daemon)
    daemon.paths = RuntimePaths(
        plugin_root=root,
        config_dir=root / "config",
        state_dir=root / "state",
        herdr_bin="herdr",
        herdr_socket=None,
    )
    daemon.config = AppConfig(mute_on_start=mute_on_start)
    daemon.policy = TalkPolicy(daemon.config)
    daemon._mic_preference_lock = threading.Lock()
    daemon._mic_preference_explicit = False
    daemon._mic_preference_error = None
    daemon._mode_transition_lock = threading.RLock()
    daemon._authority_lock = threading.RLock()
    daemon._voice_lock = threading.Lock()
    daemon._listener_generation = 0
    daemon._typed_generation = 0
    daemon._global_generation = 0
    daemon._delivery_closed = False
    daemon._shutdown_scheduled = False
    daemon._stop = threading.Event()
    daemon.voice = None
    daemon.subscriber = None
    daemon.herdr = FakeHerdr()
    daemon._activity_state_lock = threading.Lock()
    daemon._activity_status = {}
    return daemon


class FakeHerdr:
    @staticmethod
    def workspace_list() -> list[dict[str, object]]:
        return []

    @staticmethod
    def agent_list() -> list[dict[str, object]]:
        return []

    @staticmethod
    def notification_show(_title: str, _body: str, *, sound: str = "none") -> None:
        del sound


class MicPreferenceTests(unittest.TestCase):
    def test_configured_vad_stop_threshold_reaches_voice_listener(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.config.vad_stop_secs = 1.8
            listener = Mock(running=True, listener_generation=1)
            listener.wait_until_ready.return_value = True
            with patch("voicerdr.daemon.VoiceListener", return_value=listener) as make:
                self.assertTrue(daemon._start_voice(wait_secs=0))

            self.assertEqual(make.call_args.kwargs["vad_stop_secs"], 1.8)

    def test_real_legacy_stop_ensure_fails_closed_without_atomic_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = RuntimePaths(
                plugin_root=root,
                config_dir=root / "config",
                state_dir=root / "state",
                herdr_bin="herdr",
                herdr_socket=None,
            )
            paths.state_dir.mkdir(parents=True)
            paths.activity_state.write_text(json.dumps({"phase": "muted"}))
            self.assertFalse(paths.mic_preference.exists())

            legacy_stopped = threading.Event()
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(paths.control_socket))
            server.listen()

            def serve_legacy() -> None:
                try:
                    while True:
                        conn, _ = server.accept()
                        with conn:
                            request = json.loads(conn.recv(65536).split(b"\n", 1)[0])
                            if request["method"] == "handoff_quit":
                                response = {
                                    "id": request["id"],
                                    "ok": False,
                                    "error": {
                                        "code": "unknown_method",
                                        "message": "unknown method handoff_quit",
                                    },
                                }
                            elif request["method"] == "quit":
                                paths.activity_state.write_text(
                                    json.dumps({"phase": "shutting_down"})
                                )
                                response = {
                                    "id": request["id"],
                                    "ok": True,
                                    "result": {"quitting": True},
                                }
                            else:
                                raise AssertionError(request)
                            conn.sendall((json.dumps(response) + "\n").encode())
                            if request["method"] == "quit":
                                return
                finally:
                    server.close()
                    paths.control_socket.unlink(missing_ok=True)
                    legacy_stopped.set()

            legacy_thread = threading.Thread(target=serve_legacy, daemon=True)
            legacy_thread.start()

            stopped = stop_daemon(paths, wait_secs=0)
            self.assertTrue(stopped["soft_quit"])
            self.assertIsNone(stopped["mic_handoff"])
            self.assertTrue(stopped["mic_handoff_fail_closed"])
            self.assertTrue(legacy_stopped.wait(2))
            self.assertEqual(
                json.loads(paths.activity_state.read_text()),
                {"phase": "shutting_down"},
            )

            spawned: dict[str, object] = {}

            def start_in_process(_paths: RuntimePaths) -> None:
                config = AppConfig(
                    mute_on_start=False,
                    speak_enabled=False,
                    activity_workspace_enabled=False,
                )
                daemon = Daemon(paths, config)
                daemon._install_signals = Mock()
                daemon._start_events = Mock()
                daemon._seed_focus = Mock()
                daemon._sync_space_labels = Mock()
                daemon._ensure_activity_workspace = Mock()
                thread = threading.Thread(target=daemon.run_forever, daemon=True)
                spawned.update(daemon=daemon, thread=thread)
                thread.start()

            with patch("voicerdr.ensure._spawn_daemon", side_effect=start_in_process):
                ensured = ensure_daemon(paths, timeout=5)

            daemon = spawned["daemon"]
            assert isinstance(daemon, Daemon)
            self.assertEqual(ensured["status"]["mode"], "mute")
            self.assertFalse(daemon.status()["voice_running"])
            self.assertIsNone(daemon.voice)
            self.assertEqual(
                json.loads(paths.mic_preference.read_text()),
                {"mode": "mute", "version": 1},
            )
            self.assertEqual(paths.mic_preference.stat().st_mode & 0o777, 0o600)
            self.assertFalse(paths.mic_handoff.exists())

            stop_daemon(paths, wait_secs=1)
            thread = spawned["thread"]
            assert isinstance(thread, threading.Thread)
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_absent_preference_preserves_each_configured_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            listening = daemon_at(root, mute_on_start=False)
            listening._restore_mic_preference()
            self.assertEqual(listening.policy.mic_mode, "listen")
            self.assertFalse(listening._mic_preference_explicit)

            muted = daemon_at(root, mute_on_start=True)
            muted._restore_mic_preference()
            self.assertEqual(muted.policy.mic_mode, "mute")
            self.assertFalse(muted._mic_preference_explicit)

    def test_legacy_muted_activity_is_migrated_before_listener_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.paths.activity_state.write_text(
                json.dumps(
                    {
                        "phase": "muted",
                        "waiting_for": "listen command",
                        "revision": 8,
                    }
                )
            )
            daemon._acquire_daemon_lock = Mock()
            daemon._initialize_replay_ledger = Mock(return_value=set())
            daemon._write_session = Mock()
            daemon._install_signals = Mock()
            daemon._bind_control = Mock()
            daemon._start_events = Mock()
            daemon._seed_focus = Mock()
            daemon._sync_space_labels = Mock()
            daemon._record_activity = Mock()
            daemon._set_activity_status = Mock()
            daemon._ensure_activity_workspace = Mock()
            daemon._start_voice = Mock()
            daemon._accept_loop = Mock()
            daemon.shutdown = Mock()
            daemon._replay_ledger_error = None
            daemon.speaker = Mock()
            daemon.policy.speak_enabled = False
            daemon.focused_workspace_id = None

            daemon.run_forever()

            self.assertEqual(daemon.policy.mic_mode, "mute")
            daemon._start_voice.assert_not_called()
            self.assertTrue(daemon._mic_preference_explicit)
            self.assertEqual(
                json.loads(daemon.paths.mic_preference.read_text()),
                {"mode": "mute", "version": 1},
            )

    def test_legacy_non_muted_activity_does_not_override_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.paths.activity_state.write_text(json.dumps({"phase": "ready"}))

            daemon._restore_mic_preference()

            self.assertEqual(daemon.policy.mic_mode, "listen")
            self.assertFalse(daemon._mic_preference_explicit)
            self.assertFalse(daemon.paths.mic_preference.exists())

    def test_unreadable_legacy_activity_fails_safe_to_mute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.paths.activity_state.write_text("{corrupt")

            daemon._restore_mic_preference()

            self.assertEqual(daemon.policy.mic_mode, "mute")
            self.assertTrue(daemon._mic_preference_explicit)
            self.assertIn("legacy activity state", daemon._mic_preference_error or "")

    def test_explicit_mute_survives_restart_until_explicit_listen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = daemon_at(root)
            first.herdr = FakeHerdr()
            first.voice = None
            first._voice_lock = threading.Lock()
            first._authority_lock = threading.RLock()
            first._listener_generation = 0
            first._activity_state_lock = threading.Lock()
            first._activity_status = {"live_transcript": "stale final"}

            response = json.loads(
                first._dispatch(ControlRequest("mute-1", "set_mode", {"mode": "mute"}))
            )
            self.assertEqual(response["result"]["mode"], "mute")
            self.assertEqual(first._activity_status["live_transcript"], "")
            self.assertEqual(first.paths.mic_preference.stat().st_mode & 0o777, 0o600)

            restarted = daemon_at(root)
            restarted._restore_mic_preference()
            self.assertEqual(restarted.policy.mic_mode, "mute")
            self.assertTrue(restarted._mic_preference_explicit)

            restarted._persist_mic_preference("listen")
            after_listen = daemon_at(root, mute_on_start=True)
            after_listen._restore_mic_preference()
            self.assertEqual(after_listen.policy.mic_mode, "listen")

    def test_interrupted_mute_transition_forces_mute_after_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon._persist_mic_preference("listen")
            real_replace = os.replace
            replacements = 0

            def crash_after_marker(source: Path, target: Path) -> None:
                nonlocal replacements
                replacements += 1
                if replacements == 2:
                    raise OSError("crash")
                real_replace(source, target)

            with (
                patch("voicerdr.daemon.os.replace", side_effect=crash_after_marker),
                self.assertRaises(MicPreferenceError),
            ):
                daemon._persist_mic_preference("mute")

            self.assertEqual(
                json.loads(daemon.paths.mic_preference.read_text()),
                {"mode": "listen", "version": 1},
            )
            self.assertEqual(
                list(daemon.paths.state_dir.glob(".mic_preference.json.*.tmp")), []
            )
            self.assertTrue(daemon.paths.mic_preference_pending.is_file())

            restarted = daemon_at(Path(temp_dir))
            restarted._restore_mic_preference()
            self.assertEqual(restarted.policy.mic_mode, "mute")
            self.assertIn("interrupted", restarted._mic_preference_error)

    def test_invalid_or_unreadable_state_fails_safe_to_mute(self) -> None:
        invalid_states = (
            "{corrupt",
            json.dumps({"version": 1, "mode": "listen", "extra": True}),
            json.dumps({"version": 1, "mode": "unknown"}),
        )
        for payload in invalid_states:
            with (
                self.subTest(payload=payload),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                daemon = daemon_at(Path(temp_dir))
                daemon.paths.state_dir.mkdir(parents=True)
                daemon.paths.mic_preference.write_text(payload)

                daemon._restore_mic_preference()

                self.assertEqual(daemon.policy.mic_mode, "mute")
                self.assertTrue(daemon._mic_preference_explicit)
                self.assertIsNotNone(daemon._mic_preference_error)

    def test_confirmed_listen_handoff_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir), mute_on_start=True)
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.paths.session_file.write_text("{}")

            class ListeningDaemonClient:
                def __init__(self, _path: Path, timeout: float = 2.0) -> None:
                    del timeout

                def handoff_quit(self) -> dict[str, object]:
                    return {
                        "handoff_version": 1,
                        "mode": "listen",
                        "quitting": True,
                    }

            with (
                patch("voicerdr.ensure.ControlClient", ListeningDaemonClient),
                patch("voicerdr.ensure._find_voicerdr_pids", return_value=[]),
                patch("voicerdr.ensure.time.sleep"),
            ):
                stopped = stop_daemon(daemon.paths, wait_secs=0)

            self.assertEqual(stopped["mic_handoff"], "listen")
            self.assertEqual(daemon.paths.mic_handoff.stat().st_mode & 0o777, 0o600)

            daemon._restore_mic_preference()

            self.assertEqual(daemon.policy.mic_mode, "listen")
            self.assertEqual(
                json.loads(daemon.paths.mic_preference.read_text()),
                {"mode": "listen", "version": 1},
            )
            self.assertFalse(daemon.paths.mic_handoff.exists())

    def test_handoff_file_failure_still_terminates_and_reaps_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.paths.pidfile.write_text("4242\n")
            calls: list[str] = []

            class AcceptedHandoffClient:
                def __init__(self, _path: Path, timeout: float = 2.0) -> None:
                    del timeout

                def handoff_quit(self) -> dict[str, object]:
                    calls.append("handoff_quit")
                    return {
                        "handoff_version": 1,
                        "mode": "listen",
                        "quitting": True,
                    }

                def quit(self) -> dict[str, object]:
                    calls.append("quit")
                    return {"quitting": True}

            alive = iter((True, True, False))
            fsync_calls = 0

            def fail_after_marker_removal(_path: Path) -> None:
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 3:
                    raise OSError("handoff directory fsync failed")

            with (
                patch("voicerdr.ensure.ControlClient", AcceptedHandoffClient),
                patch("voicerdr.ensure._pid_belongs_to_install", return_value=True),
                patch("voicerdr.ensure._find_voicerdr_pids", return_value=[]),
                patch(
                    "voicerdr.ensure._pid_alive", side_effect=lambda _pid: next(alive)
                ),
                patch("voicerdr.ensure._kill_tree") as kill_tree,
                patch("voicerdr.ensure.time.sleep"),
                patch(
                    "voicerdr.ensure._fsync_directory",
                    side_effect=fail_after_marker_removal,
                ),
            ):
                stopped = stop_daemon(daemon.paths, wait_secs=0)

            self.assertEqual(calls, ["handoff_quit", "quit"])
            self.assertEqual(
                kill_tree.call_args_list,
                [
                    unittest.mock.call(4242, sig=signal.SIGTERM),
                    unittest.mock.call(4242, sig=signal.SIGKILL),
                ],
            )
            self.assertTrue(stopped["ok"])
            self.assertTrue(stopped["mic_handoff_fail_closed"])
            self.assertIsNone(stopped["mic_handoff"])
            self.assertTrue(daemon.paths.mic_handoff_pending.exists())
            self.assertEqual(fsync_calls, 4)

    def test_stale_listen_handoff_cannot_override_durable_mute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon._persist_mic_preference("mute")
            _begin_mic_handoff(daemon.paths)
            _complete_mic_handoff(daemon.paths, "listen")

            restarted = daemon_at(Path(temp_dir))
            restarted._restore_mic_preference()

            self.assertEqual(restarted.policy.mic_mode, "mute")
            self.assertIn("superseded", restarted._mic_preference_error or "")
            self.assertFalse(restarted.paths.mic_handoff.exists())

    def test_stale_listen_handoff_cannot_override_pending_mute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon._persist_mic_preference("listen")
            _begin_mic_handoff(daemon.paths)
            _complete_mic_handoff(daemon.paths, "listen")
            daemon._atomic_write_mic_state(
                daemon.paths.mic_preference_pending,
                {"version": 1, "target_mode": "mute"},
            )

            restarted = daemon_at(Path(temp_dir))
            restarted._restore_mic_preference()

            self.assertEqual(restarted.policy.mic_mode, "mute")
            self.assertIn("interrupted", restarted._mic_preference_error or "")
            self.assertFalse(restarted.paths.mic_handoff.exists())

    def test_crash_between_handoff_write_and_marker_removal_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            _begin_mic_handoff(daemon.paths)
            real_unlink = Path.unlink

            def crash_on_pending(path: Path, *args: object, **kwargs: object) -> None:
                if path == daemon.paths.mic_handoff_pending:
                    raise OSError("crash before marker removal")
                real_unlink(path, *args, **kwargs)

            with (
                patch.object(
                    Path, "unlink", autospec=True, side_effect=crash_on_pending
                ),
                self.assertRaises(OSError),
            ):
                _complete_mic_handoff(daemon.paths, "listen")

            self.assertTrue(daemon.paths.mic_handoff.exists())
            self.assertTrue(daemon.paths.mic_handoff_pending.exists())
            restarted = daemon_at(Path(temp_dir))
            restarted._restore_mic_preference()
            self.assertEqual(restarted.policy.mic_mode, "mute")

    def test_ping_snapshot_then_concurrent_mute_quit_restarts_muted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon._persist_mic_preference("listen")
            stale_snapshot = json.loads(
                daemon._dispatch(ControlRequest("ping", "ping", {}))
            )["result"]["mode"]
            self.assertEqual(stale_snapshot, "listen")
            _begin_mic_handoff(daemon.paths)
            _complete_mic_handoff(daemon.paths, stale_snapshot)

            stop_entered = threading.Event()
            release_stop = threading.Event()
            mute_response: dict[str, object] = {}

            class Listener:
                running = True
                last_error = None

                def stop(self) -> bool:
                    stop_entered.set()
                    self.assert_release()
                    self.running = False
                    return True

                @staticmethod
                def assert_release() -> None:
                    if not release_stop.wait(2):
                        raise AssertionError("mute barrier was not released")

            daemon.voice = Listener()

            def mute() -> None:
                mute_response.update(
                    json.loads(
                        daemon._dispatch(
                            ControlRequest("mute", "set_mode", {"mode": "mute"})
                        )
                    )
                )

            mute_thread = threading.Thread(target=mute)
            mute_thread.start()
            self.assertTrue(stop_entered.wait(2))
            self.assertTrue(mute_thread.is_alive())
            release_stop.set()
            mute_thread.join(2)
            self.assertTrue(mute_response["ok"])

            daemon._schedule_shutdown = Mock()
            quit_response = json.loads(
                daemon._dispatch(ControlRequest("quit", "quit", {}))
            )
            self.assertTrue(quit_response["ok"])

            spawned: dict[str, object] = {}

            def start_in_process(_paths: RuntimePaths) -> None:
                restarted = Daemon(
                    daemon.paths,
                    AppConfig(
                        mute_on_start=False,
                        speak_enabled=False,
                        activity_workspace_enabled=False,
                    ),
                )
                restarted._install_signals = Mock()
                restarted._start_events = Mock()
                restarted._seed_focus = Mock()
                restarted._sync_space_labels = Mock()
                restarted._ensure_activity_workspace = Mock()
                thread = threading.Thread(target=restarted.run_forever, daemon=True)
                spawned.update(daemon=restarted, thread=thread)
                thread.start()

            with patch("voicerdr.ensure._spawn_daemon", side_effect=start_in_process):
                ensured = ensure_daemon(daemon.paths, timeout=5)

            self.assertEqual(ensured["status"]["mode"], "mute")
            restarted = spawned["daemon"]
            assert isinstance(restarted, Daemon)
            self.assertIsNone(restarted.voice)
            self.assertFalse(restarted.paths.mic_handoff.exists())
            stop_daemon(daemon.paths, wait_secs=1)
            daemon_thread = spawned["thread"]
            assert isinstance(daemon_thread, threading.Thread)
            daemon_thread.join(2)
            self.assertFalse(daemon_thread.is_alive())

    def test_atomic_handoff_waits_for_inflight_mute_and_reports_final_mode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon._persist_mic_preference("listen")
            stop_entered = threading.Event()
            release_stop = threading.Event()
            handoff_returned = threading.Event()
            responses: dict[str, dict[str, object]] = {}

            class Listener:
                running = True
                last_error = None

                def stop(self) -> bool:
                    stop_entered.set()
                    if not release_stop.wait(2):
                        raise AssertionError("mute barrier was not released")
                    self.running = False
                    return True

            daemon.voice = Listener()
            daemon._schedule_shutdown = Mock()

            def request(name: str, method: str, params: dict[str, str]) -> None:
                responses[name] = json.loads(
                    daemon._dispatch(ControlRequest(name, method, params))
                )
                if name == "handoff":
                    handoff_returned.set()

            mute_thread = threading.Thread(
                target=request,
                args=("mute", "set_mode", {"mode": "mute"}),
            )
            mute_thread.start()
            self.assertTrue(stop_entered.wait(2))
            handoff_thread = threading.Thread(
                target=request,
                args=("handoff", "handoff_quit", {}),
            )
            handoff_thread.start()
            self.assertFalse(handoff_returned.wait(0.05))
            release_stop.set()
            mute_thread.join(2)
            handoff_thread.join(2)

            self.assertTrue(responses["mute"]["ok"])
            self.assertTrue(responses["handoff"]["ok"])
            self.assertEqual(responses["handoff"]["result"]["mode"], "mute")
            self.assertTrue(daemon._delivery_closed)

    def test_unresponsive_stale_daemon_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.paths.session_file.write_text("{}")

            with patch("voicerdr.ensure._find_voicerdr_pids", return_value=[]):
                stopped = stop_daemon(daemon.paths, wait_secs=0)

            self.assertTrue(stopped["mic_handoff_fail_closed"])
            self.assertTrue(daemon.paths.mic_handoff_pending.exists())
            daemon._restore_mic_preference()
            self.assertEqual(daemon.policy.mic_mode, "mute")
            self.assertEqual(
                json.loads(daemon.paths.mic_preference.read_text()),
                {"mode": "mute", "version": 1},
            )

    def test_corrupt_or_interrupted_handoff_fails_closed_and_is_consumed(self) -> None:
        for interrupted in (False, True):
            with (
                self.subTest(interrupted=interrupted),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                daemon = daemon_at(Path(temp_dir))
                daemon.paths.state_dir.mkdir(parents=True)
                if interrupted:
                    daemon.paths.mic_handoff_pending.write_text("capture pending")
                else:
                    daemon.paths.mic_handoff.write_text("{corrupt")
                    daemon.paths.mic_handoff.chmod(0o600)

                daemon._restore_mic_preference()

                self.assertEqual(daemon.policy.mic_mode, "mute")
                self.assertIsNotNone(daemon._mic_preference_error)
                self.assertEqual(
                    json.loads(daemon.paths.mic_preference.read_text()),
                    {"mode": "mute", "version": 1},
                )
                self.assertFalse(daemon.paths.mic_handoff.exists())
                self.assertFalse(daemon.paths.mic_handoff_pending.exists())

    def test_concurrent_listen_then_mute_serializes_effective_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir), mute_on_start=True)
            daemon.paths.state_dir.mkdir(parents=True)
            listen_entered = threading.Event()
            release_listen = threading.Event()
            mute_returned = threading.Event()
            responses: dict[str, dict[str, object]] = {}

            class Listener:
                running = True
                last_error = None
                listener_generation = 0

                def stop(self) -> bool:
                    self.running = False
                    return True

            def blocked_start(*, wait_secs: float) -> bool:
                del wait_secs
                listen_entered.set()
                self.assertTrue(release_listen.wait(2))
                daemon.voice = Listener()
                return True

            def request(mode: str) -> None:
                responses[mode] = json.loads(
                    daemon._dispatch(ControlRequest(mode, "set_mode", {"mode": mode}))
                )
                if mode == "mute":
                    mute_returned.set()

            with patch.object(daemon, "_start_voice", side_effect=blocked_start):
                listen_thread = threading.Thread(target=request, args=("listen",))
                listen_thread.start()
                self.assertTrue(listen_entered.wait(2))
                mute_thread = threading.Thread(target=request, args=("mute",))
                mute_thread.start()
                self.assertFalse(mute_returned.wait(0.05))
                release_listen.set()
                listen_thread.join(2)
                mute_thread.join(2)

            self.assertTrue(responses["listen"]["ok"])
            self.assertEqual(responses["listen"]["result"]["mode"], "listen")
            self.assertTrue(responses["mute"]["ok"])
            self.assertEqual(responses["mute"]["result"]["mode"], "mute")
            self.assertEqual(daemon.policy.mic_mode, "mute")
            self.assertIsNone(daemon.voice)
            self.assertEqual(daemon._activity_status["phase"], "muted")

    def test_direct_mute_status_and_queued_listen_cannot_reopen_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            daemon = Daemon(
                RuntimePaths(
                    plugin_root=root,
                    config_dir=root / "config",
                    state_dir=root / "state",
                    herdr_bin="herdr",
                    herdr_socket=None,
                ),
                AppConfig(),
            )
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.herdr = FakeHerdr()
            stop_entered = threading.Event()
            release_stop = threading.Event()
            responses: dict[str, dict[str, object]] = {}

            class BlockingListener:
                running = True
                capture_eligible = True
                last_error = None
                listener_generation = 0

                def stop(self) -> bool:
                    stop_entered.set()
                    if not release_stop.wait(2):
                        raise AssertionError("mute barrier was not released")
                    self.running = False
                    self.capture_eligible = False
                    return True

            daemon.voice = BlockingListener()

            def request(name: str, mode: str) -> None:
                responses[name] = json.loads(
                    daemon._dispatch(ControlRequest(name, "set_mode", {"mode": mode}))
                )

            mute_thread = threading.Thread(target=request, args=("mute", "mute"))
            mute_thread.start()
            self.assertTrue(stop_entered.wait(2))

            status = daemon.status()
            self.assertEqual(status["mode"], "mute")
            self.assertTrue(status["mic_transition_pending"])
            self.assertTrue(status["live_capture"]["mic_transition_pending"])

            listen_thread = threading.Thread(target=request, args=("listen", "listen"))
            listen_thread.start()
            self.assertTrue(listen_thread.is_alive())
            release_stop.set()
            mute_thread.join(2)
            listen_thread.join(2)

            self.assertTrue(responses["mute"]["ok"])
            self.assertFalse(responses["listen"]["ok"])
            self.assertEqual(
                responses["listen"]["error"]["code"], "mic_transition_pending"
            )
            self.assertEqual(daemon.policy.mic_mode, "mute")
            self.assertIsNone(daemon.voice)
            self.assertFalse(daemon.status()["mic_transition_pending"])
            self.assertFalse(daemon._activity_status["mic_transition_pending"])
            self.assertFalse(
                json.loads(daemon.paths.activity_state.read_text())[
                    "mic_transition_pending"
                ]
            )

    def test_listen_during_listen_boot_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            daemon = Daemon(
                RuntimePaths(
                    plugin_root=root,
                    config_dir=root / "config",
                    state_dir=root / "state",
                    herdr_bin="herdr",
                    herdr_socket=None,
                ),
                AppConfig(),
            )
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.herdr = FakeHerdr()
            daemon.policy.set_mic_mode("listen")
            boot_entered = threading.Event()
            release_boot = threading.Event()
            responses: dict[str, dict[str, object]] = {}

            class BootListener:
                running = False
                capture_eligible = False
                last_error = None
                listener_generation = 0

                def __init__(self, *args: object, **kwargs: object) -> None:
                    del args
                    self.listener_generation = int(
                        kwargs.get("listener_generation") or 0
                    )

                def start(self) -> None:
                    boot_entered.set()
                    if not release_boot.wait(2):
                        raise AssertionError("boot barrier was not released")
                    self.running = True
                    self.capture_eligible = True

                def wait_until_ready(self, timeout: float = 45.0) -> bool:
                    del timeout
                    return self.running

                def stop(self) -> bool:
                    self.running = False
                    self.capture_eligible = False
                    return True

            def boot() -> None:
                responses["boot"] = {"ok": daemon._start_voice(wait_secs=2.0)}

            def listen() -> None:
                responses["listen"] = json.loads(
                    daemon._dispatch(
                        ControlRequest("listen", "set_mode", {"mode": "listen"})
                    )
                )

            with patch("voicerdr.daemon.VoiceListener", BootListener):
                boot_thread = threading.Thread(target=boot)
                boot_thread.start()
                self.assertTrue(boot_entered.wait(2))
                listen_thread = threading.Thread(target=listen)
                listen_thread.start()
                self.assertTrue(listen_thread.is_alive())
                release_boot.set()
                boot_thread.join(2)
                listen_thread.join(2)

            self.assertTrue(responses["boot"]["ok"])
            self.assertTrue(responses["listen"]["ok"])
            self.assertEqual(responses["listen"]["result"]["mode"], "listen")
            self.assertTrue(responses["listen"]["result"]["voice_running"])
            self.assertEqual(daemon.policy.mic_mode, "listen")

    def test_successful_mute_waits_for_listener_thread_capture_ineligibility(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            listener = VoiceListener(on_transcript=lambda _final: None)
            initialized = threading.Event()
            release_transport = threading.Event()

            def retain_transport() -> None:
                initialized.set()
                listener._stop.wait(2)
                release_transport.wait(2)

            listener._thread_main = retain_transport
            listener.start()
            self.assertTrue(initialized.wait(2))
            self.assertTrue(listener.capture_eligible)
            daemon.voice = listener
            response: dict[str, object] = {}

            def mute() -> None:
                response.update(
                    json.loads(
                        daemon._dispatch(
                            ControlRequest("mute-init", "set_mode", {"mode": "mute"})
                        )
                    )
                )

            mute_thread = threading.Thread(target=mute)
            mute_thread.start()
            self.assertTrue(listener._stop.wait(2))
            self.assertTrue(listener.capture_eligible)
            self.assertTrue(mute_thread.is_alive())
            release_transport.set()
            mute_thread.join(2)

            self.assertFalse(mute_thread.is_alive())
            self.assertFalse(listener.capture_eligible)
            self.assertTrue(response["ok"])
            self.assertEqual(response["result"]["mode"], "mute")

    def test_unproven_mute_closure_errors_and_shuts_down_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = daemon_at(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon._schedule_shutdown = Mock()
            daemon.voice = SimpleNamespace(
                running=False,
                last_error=None,
                stop=Mock(return_value=False),
            )

            response = json.loads(
                daemon._dispatch(
                    ControlRequest("mute-stuck", "set_mode", {"mode": "mute"})
                )
            )

            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["code"], "mic_closure")
            self.assertTrue(daemon._delivery_closed)
            daemon._schedule_shutdown.assert_called_once_with()
            self.assertNotEqual(daemon._activity_status["phase"], "muted")


if __name__ == "__main__":
    unittest.main()
