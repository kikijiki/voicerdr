import io
import json
import os
import pty
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from textual import events
from textual._xterm_parser import XTermParser

from voicerdr.activity import (
    ActivityApp,
    ControlButton,
    _display_width,
    _read_complete_lines,
    follow,
    render_dashboard,
    render_line,
)
from voicerdr.config import AppConfig
from voicerdr.control_client import ControlClientError
from voicerdr.daemon import ACTIVITY_UI_VERSION, Daemon
from voicerdr.dictation import DictationBuffer
from voicerdr.herdr_client import HerdrError
from voicerdr.paths import RuntimePaths


class SimulatedCrash(BaseException):
    pass


class FakeHerdr:
    def __init__(self) -> None:
        self.workspaces: list[dict[str, object]] = []
        self.panes: dict[str, list[dict[str, object]]] = {}
        self.created = 0
        self.closed: list[str] = []
        self.renamed: list[tuple[str, str]] = []
        self.commands: list[tuple[str, str]] = []
        self.operations: list[tuple[str, str]] = []

    def workspace_list(self) -> list[dict[str, object]]:
        return self.workspaces

    def pane_list(self, workspace_id: str | None = None) -> list[dict[str, object]]:
        if workspace_id is None:
            rows: list[dict[str, object]] = []
            for panes in self.panes.values():
                rows.extend(panes)
            return rows
        return list(self.panes.get(workspace_id, []))

    def add_workspace(
        self,
        workspace_id: str,
        *,
        label: str = "voicerdr activity",
        pane_id: str | None = None,
    ) -> None:
        self.workspaces.append({"workspace_id": workspace_id, "label": label})
        self.panes[workspace_id] = [{"pane_id": pane_id or f"{workspace_id}:p1"}]

    def workspace_create(self, *, cwd: str, label: str) -> dict[str, str]:
        self.created += 1
        workspace_id = (
            "activity-ws" if self.created == 1 else f"activity-ws-{self.created}"
        )
        self.operations.append(("create", workspace_id))
        pane_id = f"{workspace_id}:p1"
        self.workspaces.append({"workspace_id": workspace_id, "label": label})
        self.panes[workspace_id] = [{"pane_id": pane_id}]
        return {"workspace_id": workspace_id, "pane_id": pane_id}

    def workspace_close(self, workspace_id: str) -> None:
        self.operations.append(("close", workspace_id))
        self.closed.append(workspace_id)
        self.workspaces = [
            row for row in self.workspaces if row.get("workspace_id") != workspace_id
        ]
        self.panes.pop(workspace_id, None)

    def agent_list(self) -> list[dict[str, object]]:
        return []

    @staticmethod
    def notification_show(_title: str, _body: str, *, sound: str = "none") -> None:
        del sound

    def pane_rename(self, pane_id: str, label: str) -> None:
        self.operations.append(("rename", pane_id))
        self.renamed.append((pane_id, label))

    def pane_run(self, pane_id: str, command: str) -> None:
        self.operations.append(("run", pane_id))
        self.commands.append((pane_id, command))

    @staticmethod
    def activity_command(python: str, path: str, state_path: str, lines: int) -> str:
        return (
            f"{python} -m voicerdr.activity {path} --state {state_path} --lines {lines}"
        )


def paths_for(root: Path) -> RuntimePaths:
    return RuntimePaths(
        plugin_root=root,
        config_dir=root / "config",
        state_dir=root / "state",
        herdr_bin="herdr",
        herdr_socket="/tmp/herdr.sock",
    )


class ActivityWorkspaceTests(unittest.TestCase):
    def test_default_config_enables_activity_workspace(self) -> None:
        self.assertTrue(AppConfig().activity_workspace_enabled)

    def test_creates_and_reuses_owned_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            with patch("voicerdr.daemon.sys.executable", "/venv/bin/python"):
                daemon._ensure_activity_workspace()
                daemon._ensure_activity_workspace()

            self.assertEqual(daemon.herdr.created, 1)
            self.assertEqual(
                daemon.activity_workspace,
                {
                    "workspace_id": "activity-ws",
                    "pane_id": "activity-ws:p1",
                    "ui_version": ACTIVITY_UI_VERSION,
                },
            )
            self.assertIn("voicerdr.activity", daemon.herdr.commands[0][1])
            state = json.loads(daemon.paths.activity_workspace_file.read_text())
            self.assertEqual(state["workspace_id"], "activity-ws")
            self.assertEqual(state["ui_version"], ACTIVITY_UI_VERSION)
            self.assertTrue(daemon.paths.activity_log.is_file())

    def test_stale_foreign_workspace_pointer_creates_without_closing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            stale = {
                "workspace_id": "w1",
                "pane_id": "w1:p1",
                "ui_version": ACTIVITY_UI_VERSION,
            }
            daemon.paths.activity_workspace_file.write_text(json.dumps(stale))
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            # Recycled ID now belongs to an unrelated Herdr workspace; pane gone.
            daemon.herdr.add_workspace("w1", label="#3 nixos-config", pane_id="w1:pB")
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            with patch("voicerdr.daemon.sys.executable", "/venv/bin/python"):
                daemon._ensure_activity_workspace()

            self.assertEqual(daemon.herdr.closed, [])
            self.assertEqual(daemon.herdr.created, 1)
            self.assertEqual(daemon.activity_workspace["workspace_id"], "activity-ws")
            # Foreign workspace remains listed.
            self.assertTrue(
                any(row.get("workspace_id") == "w1" for row in daemon.herdr.workspaces)
            )

    def test_disabling_closes_only_the_tracked_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.paths.activity_workspace_file.write_text(
                json.dumps({"workspace_id": "activity-ws", "pane_id": "activity-ws:p1"})
            )
            daemon.config = AppConfig(activity_workspace_enabled=False)
            daemon.herdr = FakeHerdr()
            daemon.herdr.add_workspace("activity-ws")
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            daemon._ensure_activity_workspace()

            self.assertEqual(daemon.herdr.closed, ["activity-ws"])
            self.assertFalse(daemon.paths.activity_workspace_file.exists())

    def test_failed_stale_pointer_cleanup_never_closes_foreign_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            daemon.herdr.add_workspace("w1", label="user project", pane_id="w1:pB")
            stale = {
                "workspace_id": "w1",
                "pane_id": "w1:p1",
                "ui_version": ACTIVITY_UI_VERSION,
            }
            daemon.paths.activity_workspace_file.write_text(json.dumps(stale))
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            with patch.object(
                daemon,
                "_clear_activity_workspace_transaction",
                side_effect=OSError("cleanup unavailable"),
            ):
                daemon._ensure_activity_workspace()
            self.assertTrue(daemon.paths.activity_workspace_transaction.exists())

            daemon._ensure_activity_workspace()

            self.assertEqual(daemon.herdr.closed, [])
            self.assertEqual(daemon.herdr.created, 1)
            self.assertFalse(daemon.paths.activity_workspace_transaction.exists())

    def test_recovery_preserves_recycled_workspace_or_pane_ids(self) -> None:
        for committed in (False, True):
            for changed_identity in ("label", "pane"):
                with (
                    self.subTest(
                        committed=committed, changed_identity=changed_identity
                    ),
                    tempfile.TemporaryDirectory() as temp_dir,
                ):
                    daemon = Daemon.__new__(Daemon)
                    daemon.paths = paths_for(Path(temp_dir))
                    daemon.paths.state_dir.mkdir(parents=True)
                    daemon.config = AppConfig()
                    daemon.herdr = FakeHerdr()
                    old = {"workspace_id": "old", "pane_id": "old:p1"}
                    new = {"workspace_id": "new", "pane_id": "new:p1"}
                    orphan = old if committed else new
                    daemon.herdr.add_workspace(
                        orphan["workspace_id"],
                        label=(
                            "user project"
                            if changed_identity == "label"
                            else "voicerdr activity"
                        ),
                        pane_id=(
                            "replacement-pane"
                            if changed_identity == "pane"
                            else orphan["pane_id"]
                        ),
                    )
                    daemon._write_activity_workspace_transaction(old, new)
                    live_ids = {orphan["workspace_id"]}

                    recovered = daemon._recover_activity_workspace_transaction(
                        new if committed else old, live_ids, daemon.herdr.workspaces
                    )

                    self.assertTrue(recovered)
                    self.assertEqual(daemon.herdr.closed, [])
                    self.assertIn(orphan["workspace_id"], live_ids)
                    self.assertFalse(
                        daemon.paths.activity_workspace_transaction.exists()
                    )

    def test_version_mismatch_starts_replacement_before_closing_old_viewer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            old = {
                "workspace_id": "old-ws",
                "pane_id": "old-ws:p1",
                "ui_version": "3",
            }
            daemon.paths.activity_workspace_file.write_text(json.dumps(old))
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            daemon.herdr.add_workspace("old-ws")
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            daemon._ensure_activity_workspace()

            self.assertLess(
                daemon.herdr.operations.index(("run", "activity-ws:p1")),
                daemon.herdr.operations.index(("close", "old-ws")),
            )
            self.assertEqual(daemon.herdr.closed, ["old-ws"])
            replacement = json.loads(
                daemon.paths.activity_workspace_file.read_text(encoding="utf-8")
            )
            self.assertEqual(replacement["workspace_id"], "activity-ws")
            self.assertEqual(replacement["ui_version"], ACTIVITY_UI_VERSION)
            self.assertEqual(daemon.activity_workspace, replacement)

    def test_replacement_start_failure_preserves_old_viewer_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            old = {
                "workspace_id": "old-ws",
                "pane_id": "old-ws:p1",
                "ui_version": "3",
            }
            daemon.paths.activity_workspace_file.write_text(json.dumps(old))
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            daemon.herdr.add_workspace("old-ws")
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            with patch.object(
                daemon.herdr,
                "pane_run",
                side_effect=HerdrError("replacement would not start"),
            ):
                daemon._ensure_activity_workspace()

            self.assertNotIn("old-ws", daemon.herdr.closed)
            self.assertIn("activity-ws", daemon.herdr.closed)
            self.assertEqual(
                json.loads(daemon.paths.activity_workspace_file.read_text()), old
            )
            self.assertEqual(daemon.activity_workspace, old)

    def test_replacement_metadata_failure_rolls_back_new_and_keeps_old(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            old = {
                "workspace_id": "old-ws",
                "pane_id": "old-ws:p1",
                "ui_version": "3",
            }
            daemon.paths.activity_workspace_file.write_text(json.dumps(old))
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            daemon.herdr.add_workspace("old-ws")
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            with patch.object(
                daemon,
                "_write_activity_workspace_state",
                side_effect=OSError("metadata storage full"),
            ):
                daemon._ensure_activity_workspace()

            self.assertNotIn("old-ws", daemon.herdr.closed)
            self.assertIn("activity-ws", daemon.herdr.closed)
            self.assertEqual(
                json.loads(daemon.paths.activity_workspace_file.read_text()), old
            )
            self.assertEqual(daemon.activity_workspace, old)

    def test_old_viewer_cleanup_failure_keeps_valid_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            old = {
                "workspace_id": "old-ws",
                "pane_id": "old-ws:p1",
                "ui_version": "3",
            }
            daemon.paths.activity_workspace_file.write_text(json.dumps(old))
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            daemon.herdr.add_workspace("old-ws")
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            with patch.object(
                daemon.herdr,
                "workspace_close",
                side_effect=OSError("herdr cleanup unavailable"),
            ):
                daemon._ensure_activity_workspace()

            replacement = json.loads(
                daemon.paths.activity_workspace_file.read_text(encoding="utf-8")
            )
            self.assertEqual(replacement["workspace_id"], "activity-ws")
            self.assertEqual(replacement["ui_version"], ACTIVITY_UI_VERSION)
            self.assertEqual(daemon.activity_workspace, replacement)
            self.assertTrue(daemon.paths.activity_workspace_transaction.exists())

    def test_crash_before_replacement_commit_recovers_new_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            old = {
                "workspace_id": "old-ws",
                "pane_id": "old-ws:p1",
                "ui_version": "3",
            }
            daemon.paths.activity_workspace_file.write_text(json.dumps(old))
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            daemon.herdr.add_workspace("old-ws")
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            with (
                patch.object(daemon.herdr, "pane_rename", side_effect=SimulatedCrash),
                self.assertRaises(SimulatedCrash),
            ):
                daemon._ensure_activity_workspace()

            transaction = json.loads(
                daemon.paths.activity_workspace_transaction.read_text()
            )
            self.assertEqual(transaction["old_workspace"], old)
            self.assertEqual(
                transaction["new_workspace"]["workspace_id"], "activity-ws"
            )
            self.assertEqual(
                json.loads(daemon.paths.activity_workspace_file.read_text()), old
            )

            daemon._ensure_activity_workspace()

            self.assertIn("activity-ws", daemon.herdr.closed)
            self.assertIn("old-ws", daemon.herdr.closed)
            self.assertFalse(daemon.paths.activity_workspace_transaction.exists())
            self.assertEqual(daemon.activity_workspace["workspace_id"], "activity-ws-2")

    def test_crash_after_replacement_commit_recovers_old_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            old = {
                "workspace_id": "old-ws",
                "pane_id": "old-ws:p1",
                "ui_version": "3",
            }
            daemon.paths.activity_workspace_file.write_text(json.dumps(old))
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            daemon.herdr.add_workspace("old-ws")
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None
            real_close = daemon.herdr.workspace_close

            def crash_on_old(workspace_id: str) -> None:
                if workspace_id == "old-ws":
                    raise SimulatedCrash
                real_close(workspace_id)

            with (
                patch.object(daemon.herdr, "workspace_close", side_effect=crash_on_old),
                self.assertRaises(SimulatedCrash),
            ):
                daemon._ensure_activity_workspace()

            committed = json.loads(daemon.paths.activity_workspace_file.read_text())
            self.assertEqual(committed["workspace_id"], "activity-ws")
            self.assertTrue(daemon.paths.activity_workspace_transaction.exists())

            daemon._ensure_activity_workspace()

            self.assertEqual(daemon.herdr.closed, ["old-ws"])
            self.assertEqual(daemon.herdr.created, 1)
            self.assertFalse(daemon.paths.activity_workspace_transaction.exists())
            self.assertEqual(daemon.activity_workspace, committed)

    def test_crash_after_orphan_cleanup_only_replays_transaction_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            old = {
                "workspace_id": "old-ws",
                "pane_id": "old-ws:p1",
                "ui_version": "3",
            }
            daemon.paths.activity_workspace_file.write_text(json.dumps(old))
            daemon.config = AppConfig()
            daemon.herdr = FakeHerdr()
            daemon.herdr.add_workspace("old-ws")
            daemon._activity_lock = threading.Lock()
            daemon.activity_workspace = None

            with (
                patch.object(
                    daemon,
                    "_clear_activity_workspace_transaction",
                    side_effect=SimulatedCrash,
                ),
                self.assertRaises(SimulatedCrash),
            ):
                daemon._ensure_activity_workspace()

            self.assertEqual(daemon.herdr.closed, ["old-ws"])
            self.assertTrue(daemon.paths.activity_workspace_transaction.exists())

            daemon._ensure_activity_workspace()

            self.assertEqual(daemon.herdr.closed, ["old-ws"])
            self.assertEqual(daemon.herdr.created, 1)
            self.assertFalse(daemon.paths.activity_workspace_transaction.exists())

    def test_replacement_transaction_and_state_are_file_and_directory_fsynced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.paths.state_dir.mkdir(parents=True)
            old = {"workspace_id": "old", "pane_id": "old:p1", "ui_version": "3"}
            new = {
                "workspace_id": "new",
                "pane_id": "new:p1",
                "ui_version": ACTIVITY_UI_VERSION,
            }
            real_open = os.open
            real_fsync = os.fsync
            opened: dict[int, Path] = {}
            fsynced: list[Path] = []

            def tracked_open(path: object, flags: int, *args: object) -> int:
                fd = real_open(path, flags, *args)
                opened[fd] = Path(path)  # type: ignore[arg-type]
                return fd

            def tracked_fsync(fd: int) -> None:
                if fd in opened:
                    fsynced.append(opened[fd])
                real_fsync(fd)

            with (
                patch("voicerdr.daemon.os.open", side_effect=tracked_open),
                patch("voicerdr.daemon.os.fsync", side_effect=tracked_fsync),
            ):
                daemon._write_activity_workspace_transaction(old, new)
                daemon._write_activity_workspace_state(new)

            self.assertGreaterEqual(fsynced.count(daemon.paths.state_dir), 2)
            self.assertTrue(
                any(
                    path.name.startswith(".activity_workspace_transaction.json")
                    for path in fsynced
                )
            )
            self.assertTrue(
                any(
                    path.name.startswith(".activity_workspace.json") for path in fsynced
                )
            )

    def test_activity_lines_are_readable_without_losing_details(self) -> None:
        line = json.dumps(
            {
                "time": "2026-09-08T15:12:13+09:00",
                "event": "prompt_sent",
                "space": "two",
                "message": "run tests",
            }
        )
        rendered = render_line(line)
        self.assertIn("15:12:13", rendered)
        self.assertIn("SENT", rendered)
        self.assertIn("run tests", rendered)

    def test_activity_line_formats_semantic_fields_without_opaque_json(self) -> None:
        line = json.dumps(
            {
                "time": "2026-09-08T15:12:13+09:00",
                "event": "withheld",
                "transcript": "jenny send the fix",
                "target": {"workspace_id": "voice", "pane_id": "voice:p1"},
                "mode": "dictation",
                "chosen_action": "agent_prompt",
                "reason": "Verifier disagreed",
                "sent": False,
                "utterance_digest": "opaque-digest",
            }
        )

        rendered = render_line(line)

        self.assertEqual(
            rendered,
            "15:12:13  NOT SENT         transcript=jenny send the fix  "
            "target=voice/voice:p1  mode=dictation  action=agent_prompt  "
            "error=Verifier disagreed",
        )
        self.assertNotIn("{", rendered)
        self.assertNotIn("opaque-digest", rendered)
        self.assertIn("opaque-digest", render_line(line, debug=True))

    def test_malformed_and_partial_journal_lines_are_safe(self) -> None:
        complete, pending = _read_complete_lines(io.StringIO('{"event":"hea'), "")
        self.assertEqual(complete, [])
        self.assertEqual(pending, '{"event":"hea')

        complete, pending = _read_complete_lines(
            io.StringIO('rd","transcript":"hello"}\n'), pending
        )
        self.assertEqual(complete, ['{"event":"heard","transcript":"hello"}'])
        self.assertEqual(pending, "")
        self.assertEqual(
            render_line("{truncated"),
            "--:--:--  MALFORMED ENTRY (details hidden; use --debug)",
        )
        self.assertIn("{truncated", render_line("{truncated", debug=True))

    def test_dashboard_pins_live_mode_wait_condition_and_words(self) -> None:
        rendered = render_dashboard(
            {
                "phase": "listening",
                "mode": "dictation",
                "capture_active": True,
                "speech_active": True,
                "waiting_for": 'ending phrase "send it"',
                "live_transcript": "Jenny listen, this is visible live",
                "dictation_buffer": "this is visible live",
                "chosen_action": "dictation_append",
                "chosen_mode": None,
                "chosen_target": None,
            },
            ["15:12:13  HEARD hello"],
            columns=90,
            rows=24,
        )

        self.assertIn("[DICTATING]", rendered)
        self.assertIn("MODE DICTATION", rendered)
        self.assertIn('ending phrase "send it"', rendered)
        self.assertIn("Jenny listen", rendered)
        self.assertIn("action=dictation_append", rendered)
        self.assertIn("RECENT ACTIVITY", rendered)

    def test_forensic_wake_miss_clears_hearing_before_unrelated_redraw(self) -> None:
        text = "But they think one doesn't"
        utterance_id = "voice:dc4b1a0ff42743e9910e02130b2dae31:51"
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.config = AppConfig(feedback_heard=False)
            daemon.dictation = DictationBuffer()
            daemon.pending_clarification = None
            daemon.last_voice_action = None
            daemon.last_transcript = None
            daemon._activity_lock = threading.Lock()
            daemon._activity_state_lock = threading.Lock()
            daemon._activity_status = {}
            daemon._current_input_mode = "detecting"
            daemon._reload_aliases = lambda: None
            daemon._set_activity_status(
                phase="transcribing",
                mode="detecting",
                capture_active=True,
                live_transcript=text,
            )

            exact_time = "2026-09-09T11:53:10+09:00"
            with patch("voicerdr.daemon.datetime") as clock:
                clock.now.return_value = datetime.fromisoformat(exact_time)
                daemon._process_final_transcript(
                    text,
                    utterance_id=utterance_id,
                    listener_generation=0,
                    origin="voice",
                    global_generation=0,
                )
                daemon._record_activity("agent_status", pane_id="unrelated:p1")

            state = json.loads(daemon.paths.activity_state.read_text())
            events = [
                json.loads(line)
                for line in daemon.paths.activity_log.read_text().splitlines()
            ]
            heard = [event for event in events if event["event"] == "heard"]
            self.assertEqual(
                [(event["utterance_id"], event["transcript"]) for event in heard],
                [(utterance_id, text)],
            )
            self.assertEqual(heard[0]["time"], exact_time)
            self.assertEqual(
                daemon.last_voice_action["result"]["code"], "wake_not_matched"
            )
            self.assertEqual(state["live_transcript"], "")
            self.assertEqual(state["assistant_name"], "Jenny")
            self.assertEqual(state["assistant_avatar"], "woman")
            rendered = render_dashboard(
                state,
                [render_line(json.dumps(event)) for event in events],
                columns=100,
                rows=30,
            )
            self.assertNotIn("HEARING:", rendered)
            self.assertIn(text, rendered)  # durable HEARD/ACTION history remains

    def test_all_terminal_action_transitions_clear_live_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon.dictation = DictationBuffer()
            daemon.pending_clarification = None
            daemon.last_voice_action = None
            daemon._activity_lock = threading.Lock()
            daemon._activity_state_lock = threading.Lock()
            daemon._activity_status = {}
            daemon._current_input_mode = "one_shot"
            terminal_kinds = (
                "no_action",
                "agent_prompt",
                "verification_blocked",
                "error",
                "control",
                "dictation_finish",
                "dictation_cancel",
            )
            for kind in terminal_kinds:
                with self.subTest(kind=kind):
                    daemon._set_activity_status(
                        phase="interpreting",
                        live_transcript=f"current final for {kind}",
                        last_result="older durable result",
                    )
                    daemon._commit_voice_action(
                        {
                            "action_kind": kind,
                            "transcript": f"current final for {kind}",
                            "result": {
                                "ok": kind
                                not in {
                                    "no_action",
                                    "verification_blocked",
                                    "error",
                                },
                                "message": f"terminal {kind}",
                            },
                        }
                    )
                    state = daemon._activity_status_snapshot()
                    self.assertEqual(state["live_transcript"], "")
                    self.assertEqual(state["last_result"], f"terminal {kind}")

    def test_activity_state_always_publishes_mic_safety_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            daemon = Daemon.__new__(Daemon)
            daemon.paths = paths_for(Path(temp_dir))
            daemon._activity_state_lock = threading.Lock()
            daemon._activity_status = {}
            daemon._delivery_closed = True
            daemon._mic_transition_count = 1

            daemon._set_activity_status(phase="error", mode="idle")

            state = json.loads(daemon.paths.activity_state.read_text())
            self.assertTrue(state["delivery_closed"])
            self.assertTrue(state["mic_transition_pending"])
            snapshot = daemon._activity_status_snapshot()
            self.assertTrue(snapshot["delivery_closed"])
            self.assertTrue(snapshot["mic_transition_pending"])

    def test_status_socket_response_does_not_wait_for_blocked_toast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = paths_for(Path(temp_dir))
            daemon = Daemon(paths, AppConfig())
            daemon.paths.state_dir.mkdir(parents=True)
            toast_started = threading.Event()
            release_toast = threading.Event()

            class BlockingHerdr(FakeHerdr):
                def notification_show(
                    self, _title: str, _body: str, *, sound: str = "none"
                ) -> None:
                    del sound
                    toast_started.set()
                    release_toast.wait(2)

            daemon.herdr = BlockingHerdr()
            server, client = socket.socketpair()
            worker = threading.Thread(target=daemon._handle_conn, args=(server,))
            worker.start()
            try:
                client.settimeout(0.5)
                client.sendall(b'{"id":"status","method":"status","params":{}}\n')
                response = json.loads(client.recv(65536).split(b"\n", 1)[0])

                self.assertTrue(response["ok"])
                self.assertTrue(toast_started.wait(1))
                self.assertFalse(release_toast.is_set())
            finally:
                release_toast.set()
                client.close()
                worker.join(2)

    def test_renderer_rejects_legacy_stale_hearing_in_terminal_state(self) -> None:
        rendered = render_dashboard(
            {
                "phase": "ready",
                "mode": "idle",
                "capture_active": False,
                "live_transcript": "stale final",
                "last_result": "Wake word not detected; nothing was sent.",
            },
            ["11:53:11  AGENT STATUS unrelated"],
            columns=90,
            rows=24,
        )

        self.assertNotIn("HEARING:", rendered)
        self.assertNotIn("stale final", rendered)
        self.assertIn("RESULT:", rendered)

    def test_dashboard_uses_exact_terminal_height_and_bottom_status(self) -> None:
        rendered = render_dashboard(
            {
                "phase": "listening",
                "mode": "dictation",
                "capture_active": True,
                "speech_active": True,
                "waiting_for": 'ending phrase "send it"',
                "live_transcript": "Jenny listen, this is visible live",
                "dictation_buffer": "this is visible live",
                "chosen_action": "dictation_append",
                "chosen_mode": None,
                "chosen_target": None,
            },
            ["15:12:13  HEARD hello"],
            columns=90,
            rows=24,
        )
        lines = rendered.split("\n")
        self.assertEqual(len(lines), 24)
        self.assertIn("[DICTATING]", "\n".join(lines[-12:]))
        self.assertNotIn("[HEARING]", rendered)

    def test_status_labels_cover_capture_delivery_and_mute_states(self) -> None:
        cases = (
            ({"phase": "ready", "mode": "idle"}, "[LISTENING]"),
            (
                {"phase": "listening", "mode": "detecting", "speech_active": True},
                "[HEARING]",
            ),
            ({"phase": "interpreting", "mode": "idle"}, "[INTERPRETING]"),
            ({"phase": "verifying", "mode": "idle"}, "[VERIFYING]"),
            (
                {"phase": "ready", "mode": "idle", "delivery_status": "sent"},
                "[SENT]",
            ),
            (
                {"phase": "ready", "mode": "idle", "delivery_status": "not_sent"},
                "[NOT SENT]",
            ),
            ({"phase": "muted", "mode": "idle"}, "[MUTED]"),
        )
        for state, expected in cases:
            with self.subTest(expected=expected):
                rendered = render_dashboard(state, [], columns=50, rows=10)
                self.assertIn(expected, rendered)

    def test_dashboard_snapshot_anchors_status_below_history(self) -> None:
        state = {
            "phase": "verifying",
            "mode": "dictation",
            "waiting_for": "independent LLM approval",
            "live_transcript": "Jenny send 日本語 status",
            "dictation_buffer": "send 日本語 status",
            "chosen_action": "agent_prompt",
            "chosen_target": {"workspace_id": "voice", "pane_id": "voice:p1"},
        }
        history = [
            render_line(
                json.dumps(
                    {
                        "time": "2026-09-08T15:12:13+09:00",
                        "event": "heard",
                        "transcript": "Jenny send 日本語 status",
                    }
                )
            ),
            render_line(
                json.dumps(
                    {
                        "time": "2026-09-08T15:12:14+09:00",
                        "event": "intent_chosen",
                        "action": "agent_prompt",
                        "mode": "dictation",
                        "target": {
                            "workspace_id": "voice",
                            "pane_id": "voice:p1",
                        },
                        "reason": "exact request",
                    }
                )
            ),
        ]

        rendered = render_dashboard(state, history, columns=60, rows=14)

        self.assertEqual(
            rendered,
            " RECENT ACTIVITY\n"
            "\n"
            "\n"
            "\n"
            "15:12:13  HEARD            transcript=Jenny send 日本語\n"
            "status\n"
            "15:12:14  INTENT           target=voice/voice:p1\n"
            "mode=dictation  action=agent_prompt  error=exact request\n"
            "[VERIFYING]  MODE DICTATION · PHASE VERIFYING\n"
            " WAITING: independent LLM approval\n"
            " TRANSCRIPT: Jenny send 日本語 status\n"
            " BUFFER: send 日本語 status\n"
            " CHOSEN: action=agent_prompt · target=voice/voice:p1\n"
            " RESULT: —",
        )
        self.assertTrue(rendered.rsplit("\n", 1)[-1].startswith(" RESULT:"))
        self.assertTrue(
            all(_display_width(line) <= 60 for line in rendered.split("\n"))
        )

    def test_narrow_unicode_dashboard_stays_inside_terminal(self) -> None:
        rendered = render_dashboard(
            {
                "phase": "listening",
                "mode": "detecting",
                "speech_active": True,
                "live_transcript": "日本語 café e\N{COMBINING ACUTE ACCENT}",
            },
            ["12:00:00  HEARD  日本語 transcript"],
            columns=12,
            rows=7,
        )
        lines = rendered.split("\n")
        self.assertEqual(len(lines), 7)
        self.assertTrue(all(_display_width(line) <= 12 for line in lines))
        self.assertIn("[HEARING]", rendered)

    def test_roomy_tty_frame_has_large_animated_long_hair_avatar(self) -> None:
        state = {
            "phase": "listening",
            "mode": "detecting",
            "speech_active": True,
            "live_transcript": "hello",
        }
        first = render_dashboard(
            state, ["12:00:00  HEARD hello"], columns=100, rows=18, animation_tick=0
        )
        second = render_dashboard(
            state, ["12:00:00  HEARD hello"], columns=100, rows=18, animation_tick=1
        )

        self.assertIn("~~~~~~~~~", first)
        self.assertNotIn("JENNY", first)
        self.assertGreaterEqual(sum("|" in line for line in first.split("\n")), 6)
        self.assertIn("[HEARING]", "\n".join(first.split("\n")[-12:]))
        self.assertNotEqual(first, second)

    def test_non_tty_snapshot_collapses_art_and_keeps_chosen_mode(self) -> None:
        rendered = render_dashboard(
            {
                "phase": "verifying",
                "mode": "dictation",
                "chosen_action": "agent_prompt",
                "chosen_mode": "dictation",
                "chosen_target": {"workspace_id": "voice", "pane_id": "voice:p1"},
            },
            ["12:00:00  INTENT target=voice/voice:p1"],
            columns=100,
            rows=18,
            allow_art=False,
        )

        self.assertNotIn("~~~~~~~~~", rendered)
        self.assertIn("mode=dictation", rendered)

    def test_avatar_config_can_hide_portrait_and_never_embeds_name(self) -> None:
        hidden = render_dashboard(
            {
                "phase": "ready",
                "mode": "idle",
                "assistant_name": "Alexandria",
                "assistant_avatar": "none",
            },
            [],
            columns=100,
            rows=18,
        )
        visible = render_dashboard(
            {
                "phase": "ready",
                "mode": "idle",
                "assistant_name": "Alexandria",
                "assistant_avatar": "woman",
            },
            [],
            columns=100,
            rows=18,
        )

        self.assertNotIn("~~~~~~~~~", hidden)
        self.assertIn("~~~~~~~~~", visible)
        self.assertNotIn("ALEXANDRIA", visible)

    def test_short_history_fragment_keeps_event_prefix_and_marks_clipping(self) -> None:
        rendered = render_dashboard(
            {"phase": "verifying", "mode": "idle"},
            ["12:00:00  VERIFYING message=" + "careful validation " * 6],
            columns=32,
            rows=8,
        )

        self.assertIn("12:00:00  VERIFYING", rendered)
        self.assertIn("…", rendered)


class FakeControlClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.results: dict[str, dict[str, object]] = {
            "mute": {"mode": "mute", "voice_running": False},
            "listen": {"mode": "listen", "voice_running": True},
            "status": {
                "mode": "listen",
                "voice_running": True,
                "live_capture": {"phase": "ready", "mode": "idle"},
            },
        }
        self.errors: dict[str, str] = {}
        self.started: threading.Event | None = None
        self.release: threading.Event | None = None

    def _call(self, action: str) -> dict[str, object]:
        self.calls.append(action)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=2)
        if action in self.errors:
            raise ControlClientError(self.errors[action])
        return self.results[action]

    def mute(self) -> dict[str, object]:
        return self._call("mute")

    def listen(self) -> dict[str, object]:
        return self._call("listen")

    def status(self, *, notify: bool = True) -> dict[str, object]:
        return self._call("status")


class ActivityInteractiveTests(unittest.IsolatedAsyncioTestCase):
    def make_app(
        self,
        root: Path,
        client: FakeControlClient,
        *,
        state: dict[str, object] | None = None,
        debug: bool = False,
    ) -> ActivityApp:
        history = root / "activity.jsonl"
        history.write_text(
            json.dumps({"time": "2026-09-10T12:00:00+09:00", "event": "daemon_started"})
            + "\n"
        )
        state_path = root / "activity_state.json"
        state_path.write_text(json.dumps(state or {"phase": "starting"}))
        return ActivityApp(
            history,
            state_path=state_path,
            history_lines=25,
            debug=debug,
            control_client=client,
        )

    async def test_single_toggle_layout_labels_the_next_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self.make_app(
                Path(temp_dir),
                FakeControlClient(),
                state={"phase": "muted", "mode": "idle"},
            )
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                buttons = list(app.query(ControlButton))
                controls = app.query_one("#controls")
                dashboard = app.query_one("#dashboard")
                self.assertEqual(len(buttons), 1)
                self.assertEqual(buttons[0].label.plain, "m Unmute")
                self.assertEqual(controls.region.bottom, 24)
                self.assertEqual(dashboard.region.bottom, controls.region.y)
                self.assertTrue(buttons[0].region in app.screen.region)

                await pilot.resize_terminal(20, 8)
                await pilot.pause()
                self.assertEqual(buttons[0].label.plain, "m Mute")
                self.assertTrue(buttons[0].disabled)
                self.assertTrue(buttons[0].region in app.screen.region)

    async def test_controls_disappear_below_safe_width_down_to_one_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            app = self.make_app(Path(temp_dir), client)
            async with app.run_test(size=(10, 8)) as pilot:
                await pilot.pause()
                controls = app.query_one("#controls")
                button = app.query_one("#mode-toggle", ControlButton)
                self.assertTrue(controls.display)
                self.assertTrue(button.region in app.screen.region)

                for width in range(9, 0, -1):
                    with self.subTest(width=width):
                        await pilot.resize_terminal(width, 8)
                        await pilot.pause()
                        self.assertFalse(controls.display)
                        self.assertTrue(button.disabled)
                        await pilot.press("m", "ctrl+l", "s")
                        await pilot.click(offset=(0, 7))
                        self.assertEqual(client.calls, [])

    async def test_click_is_same_cell_left_release_only_and_toggles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            app = self.make_app(
                Path(temp_dir), client, state={"phase": "ready", "mode": "idle"}
            )
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                await pilot.click("#mode-toggle", offset=(2, 0), button=3)
                await pilot.mouse_down("#mode-toggle", offset=(2, 0))
                await pilot.mouse_up("#mode-toggle", offset=(3, 0))
                await pilot.mouse_up("#mode-toggle", offset=(2, 0))
                await pilot.pause()
                self.assertEqual(client.calls, [])

                await pilot.click("#mode-toggle", offset=(2, 0))
                await pilot.pause()
                self.assertEqual(client.calls, ["mute"])
                self.assertEqual(
                    app.query_one("#mode-toggle", ControlButton).label.plain,
                    "m Unmute",
                )

                await pilot.click("#mode-toggle", offset=(2, 0))
                await pilot.pause()
                self.assertEqual(client.calls, ["mute", "listen"])

    async def test_other_widget_press_disarms_abandoned_toggle_press(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            app = self.make_app(
                Path(temp_dir), client, state={"phase": "muted", "mode": "idle"}
            )
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                toggle = app.query_one("#mode-toggle", ControlButton)
                await pilot.mouse_down("#mode-toggle", offset=(2, 0))
                self.assertIsNotNone(toggle._armed_at)
                await pilot.mouse_down("#dashboard", offset=(0, 0))
                self.assertIsNone(toggle._armed_at)
                await pilot.mouse_up("#mode-toggle", offset=(2, 0))
                await pilot.pause()
                self.assertEqual(client.calls, [])

    async def test_state_or_layout_change_disarms_stale_press(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = FakeControlClient()
            app = self.make_app(root, client, state={"phase": "muted", "mode": "idle"})
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                toggle = app.query_one("#mode-toggle", ControlButton)
                await pilot.mouse_down("#mode-toggle", offset=(2, 0))
                self.assertIsNotNone(toggle._armed_at)

                app.state_path.write_text(
                    json.dumps({"phase": "ready", "mode": "idle"})
                )
                app._load_state()
                app._sync_controls()
                self.assertIsNone(toggle._armed_at)
                self.assertEqual(toggle.label.plain, "m Mute")
                await pilot.mouse_up("#mode-toggle", offset=(2, 0))
                await pilot.pause()
                self.assertEqual(client.calls, [])

                await pilot.mouse_down("#mode-toggle", offset=(2, 0))
                app._layout_changed()
                self.assertIsNone(toggle._armed_at)

    async def test_keyboard_m_toggles_and_removed_shortcuts_do_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            app = self.make_app(
                Path(temp_dir), client, state={"phase": "muted", "mode": "idle"}
            )
            async with app.run_test(size=(60, 18)) as pilot:
                await pilot.press("ctrl+l", "s", "l")
                await pilot.pause()
                self.assertEqual(client.calls, [])
                await pilot.press("m")
                await pilot.pause()
                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["listen", "mute"])
                await pilot.press("q")
            self.assertIsNone(app._stream)

    async def test_q_does_not_wait_for_or_cancel_inflight_control(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            client.started = threading.Event()
            client.release = threading.Event()
            app = self.make_app(
                Path(temp_dir), client, state={"phase": "ready", "mode": "idle"}
            )
            async with app.run_test(size=(60, 18)) as pilot:
                await pilot.press("m")
                self.assertTrue(client.started.wait(timeout=1))
                await pilot.press("q")
            self.assertEqual(client.calls, ["mute"])
            self.assertFalse(client.release.is_set())
            client.release.set()

    async def test_busy_toggle_rejects_keyboard_and_mouse_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            client.started = threading.Event()
            client.release = threading.Event()
            app = self.make_app(
                Path(temp_dir), client, state={"phase": "ready", "mode": "idle"}
            )
            async with app.run_test(size=(60, 18)) as pilot:
                await pilot.press("m")
                self.assertTrue(client.started.wait(timeout=1))
                toggle = app.query_one("#mode-toggle", ControlButton)
                self.assertTrue(toggle.disabled)
                await pilot.press("m")
                await pilot.click("#mode-toggle", offset=(2, 0))
                self.assertEqual(client.calls, ["mute"])
                client.release.set()
                await pilot.pause()

    async def test_unknown_and_error_states_only_offer_safe_mute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = FakeControlClient()
            app = self.make_app(root, client, state={"phase": "starting"})
            async with app.run_test(size=(60, 18)) as pilot:
                await pilot.pause()
                toggle = app.query_one("#mode-toggle", ControlButton)
                self.assertEqual(toggle.label.plain, "m Mute")
                self.assertFalse(toggle.disabled)
                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["mute"])

                app.state_path.write_text(
                    json.dumps({"phase": "error", "mode": "idle"})
                )
                app._load_state()
                app._sync_controls()
                self.assertTrue(toggle.disabled)
                await pilot.press("m")
                self.assertEqual(client.calls, ["mute"])

    async def test_error_with_stale_listen_fact_can_only_mute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = FakeControlClient()
            app = self.make_app(root, client, state={"phase": "ready", "mode": "idle"})
            async with app.run_test(size=(60, 18)) as pilot:
                await pilot.pause()
                app.state_path.write_text(
                    json.dumps({"phase": "error", "mode": "idle"})
                )
                app._load_state()
                app._sync_controls()
                toggle = app.query_one("#mode-toggle", ControlButton)
                self.assertEqual(toggle.label.plain, "m Mute")
                self.assertFalse(toggle.disabled)
                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["mute"])

    async def test_narrow_layout_never_issues_listen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            app = self.make_app(
                Path(temp_dir), client, state={"phase": "muted", "mode": "idle"}
            )
            async with app.run_test(size=(20, 10)) as pilot:
                await pilot.pause()
                toggle = app.query_one("#mode-toggle", ControlButton)
                self.assertTrue(toggle.disabled)
                await pilot.press("m")
                self.assertEqual(client.calls, [])

                app.state_path.write_text(
                    json.dumps({"phase": "ready", "mode": "idle"})
                )
                app._load_state()
                app._sync_controls()
                self.assertFalse(toggle.disabled)
                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["mute"])

    async def test_transition_and_frozen_states_disable_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = FakeControlClient()
            app = self.make_app(root, client, state={"phase": "muting", "mode": "idle"})
            async with app.run_test(size=(70, 20)) as pilot:
                await pilot.pause()
                toggle = app.query_one("#mode-toggle", ControlButton)
                self.assertTrue(app.mic_transition_pending)
                self.assertTrue(toggle.disabled)
                await pilot.press("m")
                self.assertEqual(client.calls, [])

                app.state_path.write_text(
                    json.dumps(
                        {
                            "phase": "error",
                            "mode": "idle",
                            "delivery_closed": False,
                            "mic_transition_pending": False,
                        }
                    )
                )
                app._load_state()
                app._sync_controls()
                self.assertTrue(app.mic_transition_pending)
                self.assertTrue(toggle.disabled)

                app.state_path.write_text(
                    json.dumps(
                        {
                            "phase": "muted",
                            "mode": "idle",
                            "mic_transition_pending": False,
                        }
                    )
                )
                app._load_state()
                app._sync_controls()
                self.assertFalse(app.mic_transition_pending)
                self.assertFalse(toggle.disabled)

                app.state_path.write_text(
                    json.dumps({"phase": "shutting_down", "delivery_closed": True})
                )
                app._load_state()
                app._sync_controls()
                self.assertTrue(app.mic_controls_frozen)
                self.assertTrue(toggle.disabled)

                app.state_path.write_text(
                    json.dumps({"phase": "error", "delivery_closed": False})
                )
                app._load_state()
                app._sync_controls()
                self.assertTrue(app.mic_controls_frozen)
                self.assertTrue(toggle.disabled)

    async def test_completed_listen_transition_restores_mute_via_live_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            ready = {
                "phase": "ready",
                "mode": "idle",
                "mic_transition_pending": False,
                "delivery_closed": False,
            }
            client.results["status"] = {
                "mode": "listen",
                "voice_running": True,
                "mic_transition_pending": False,
                "delivery_closed": False,
                "live_capture": ready,
            }
            app = self.make_app(
                Path(temp_dir),
                client,
                state={**ready, "mic_transition_pending": True},
            )
            async with app.run_test(size=(70, 20)) as pilot:
                await pilot.pause()
                toggle = app.query_one("#mode-toggle", ControlButton)
                self.assertTrue(toggle.disabled)
                app.state_path.write_text(json.dumps(ready))
                app._poll_activity()
                await pilot.pause()

                self.assertEqual(client.calls, ["status"])
                self.assertFalse(app.mic_transition_pending)
                self.assertFalse(toggle.disabled)
                self.assertEqual(toggle.label.plain, "m Mute")
                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["status", "mute"])

    async def test_reused_viewer_reconciles_daemon_restart_before_enabling_listen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            muted = {
                "phase": "muted",
                "mode": "idle",
                "mic_transition_pending": False,
                "delivery_closed": False,
            }
            client.results["status"] = {
                "mode": "mute",
                "voice_running": False,
                "mic_transition_pending": False,
                "delivery_closed": False,
                "live_capture": muted,
            }
            app = self.make_app(
                Path(temp_dir),
                client,
                state={"phase": "shutting_down", "delivery_closed": True},
            )
            async with app.run_test(size=(70, 20)) as pilot:
                await pilot.pause()
                toggle = app.query_one("#mode-toggle", ControlButton)
                self.assertTrue(toggle.disabled)
                client.errors["status"] = "daemon is still starting"
                app.state_path.write_text(json.dumps(muted))
                app._poll_activity()
                await pilot.pause()
                self.assertTrue(app.mic_controls_frozen)
                self.assertTrue(toggle.disabled)

                # Retry without requiring another state-file change.
                del client.errors["status"]
                app._next_status_check = 0.0
                app._poll_activity()
                await pilot.pause()
                self.assertEqual(client.calls, ["status", "status"])
                self.assertFalse(app.mic_controls_frozen)
                self.assertFalse(toggle.disabled)
                self.assertEqual(toggle.label.plain, "m Unmute")
                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["status", "status", "listen"])

    async def test_failed_listen_reconciles_to_current_server_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            client.errors["listen"] = "microphone permission denied"
            app = self.make_app(
                Path(temp_dir), client, state={"phase": "muted", "mode": "idle"}
            )
            async with app.run_test(size=(70, 20)) as pilot:
                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["listen", "status"])
                self.assertEqual(app.daemon_mode, "listen")
                self.assertIn("permission denied", app.feedback or "")
                self.assertIn("follow-up status", app.feedback or "")
                self.assertEqual(
                    app.query_one("#mode-toggle", ControlButton).label.plain,
                    "m Mute",
                )

    async def test_failed_mute_reconciles_to_current_server_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            client.errors["mute"] = "persistence failed after mutation"
            client.results["status"] = {
                "mode": "mute",
                "voice_running": False,
                "live_capture": {"phase": "muted", "mode": "idle"},
            }
            app = self.make_app(
                Path(temp_dir), client, state={"phase": "ready", "mode": "idle"}
            )
            async with app.run_test(size=(70, 20)) as pilot:
                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["mute", "status"])
                self.assertEqual(app.daemon_mode, "mute")
                self.assertEqual(app.state["phase"], "muted")
                self.assertIn("follow-up status", app.feedback or "")
                self.assertEqual(
                    app.query_one("#mode-toggle", ControlButton).label.plain,
                    "m Unmute",
                )

    async def test_transport_ambiguity_rejects_stale_unmute_until_safe_mute(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            client = FakeControlClient()
            client.errors["listen"] = "socket closed before response"
            client.errors["status"] = "daemon unavailable"
            app = self.make_app(root, client, state={"phase": "muted", "mode": "idle"})
            async with app.run_test(size=(70, 20)) as pilot:
                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["listen", "status"])
                self.assertIsNone(app.daemon_mode)
                self.assertTrue(app.control_outcome_unknown)

                app.state_path.write_text(
                    json.dumps({"phase": "muted", "mode": "idle"})
                )
                app._load_state()
                app._sync_controls()
                app._redraw()
                toggle = app.query_one("#mode-toggle", ControlButton)
                self.assertIsNone(app.daemon_mode)
                self.assertEqual(toggle.label.plain, "m Mute")
                self.assertFalse(toggle.disabled)
                self.assertIn("MIC UNKNOWN", str(app.query_one("#dashboard").content))

                await pilot.press("m")
                await pilot.pause()
                self.assertEqual(client.calls, ["listen", "status", "mute"])
                self.assertFalse(app.control_outcome_unknown)

    async def test_success_response_without_mode_becomes_fail_closed_unknown(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            client.results["mute"] = {
                "voice_running": False,
                "live_capture": {"phase": "starting", "mode": "idle"},
            }
            app = self.make_app(
                Path(temp_dir), client, state={"phase": "ready", "mode": "idle"}
            )
            async with app.run_test(size=(70, 20)) as pilot:
                await pilot.press("m")
                await pilot.pause()
                self.assertIsNone(app.daemon_mode)
                self.assertTrue(app.control_outcome_unknown)
                self.assertIn("did not report a valid", app.feedback or "")
                toggle = app.query_one("#mode-toggle", ControlButton)
                self.assertEqual(toggle.label.plain, "m Mute")
                self.assertFalse(toggle.disabled)

    async def test_debug_history_keeps_raw_toggle_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeControlClient()
            app = self.make_app(
                Path(temp_dir),
                client,
                state={"phase": "ready", "mode": "idle"},
                debug=True,
            )
            async with app.run_test(size=(70, 20)) as pilot:
                await pilot.press("m")
                await pilot.pause()
                self.assertIn("RAW VIEWER_CONTROL", app.history[-1])
                self.assertIn("mic=mute", app.history[-1])


class ActivityTerminalCompatibilityTests(unittest.TestCase):
    def test_textual_sgr_parser_preserves_cells_buttons_and_release(self) -> None:
        parser = XTermParser()
        press, release, right, scroll = [
            next(iter(parser.feed(code)))
            for code in (
                "\x1b[<0;17;24M",
                "\x1b[<0;17;24m",
                "\x1b[<2;18;24M",
                "\x1b[<64;19;24M",
            )
        ]
        self.assertIsInstance(press, events.MouseDown)
        self.assertIsInstance(release, events.MouseUp)
        self.assertEqual((press.x, press.y, press.button), (16, 23, 1))
        self.assertEqual((release.x, release.y, release.button), (16, 23, 1))
        self.assertEqual(right.button, 3)
        self.assertIsInstance(scroll, events.MouseScrollUp)

        pasted = list(XTermParser().feed("\x1b[200~mls\x1b[201~"))
        self.assertEqual(len(pasted), 1)
        self.assertIsInstance(pasted[0], events.Paste)

    def test_non_tty_follow_is_escape_free_and_does_not_start_app(self) -> None:
        class StopFollow(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history = root / "activity.jsonl"
            history.write_text('{"event":"daemon_started"}\n')
            state = root / "activity_state.json"
            state.write_text('{"phase":"muted","mode":"idle"}')
            output = io.StringIO()
            with (
                patch("voicerdr.activity.sys.stdin", io.StringIO()),
                patch("voicerdr.activity.sys.stdout", output),
                patch("voicerdr.activity.ActivityApp.run") as run,
                patch("voicerdr.activity.time.sleep", side_effect=StopFollow),
                self.assertRaises(StopFollow),
            ):
                follow(history, state_path=state)
            run.assert_not_called()
            self.assertNotIn("\x1b", output.getvalue())
            self.assertIn("[MUTED]", output.getvalue())

    def test_real_tty_exit_restores_mouse_cursor_and_application_modes(self) -> None:
        for exit_kind in ("keyboard", "signal"):
            with (
                self.subTest(exit_kind=exit_kind),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                root = Path(temp_dir)
                history = root / "activity.jsonl"
                history.write_text('{"event":"daemon_started"}\n')
                state = root / "activity_state.json"
                state.write_text('{"phase":"muted","mode":"idle"}')
                master, slave = pty.openpty()
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "voicerdr.activity",
                        str(history),
                        "--state",
                        str(state),
                    ],
                    stdin=slave,
                    stdout=slave,
                    stderr=slave,
                    close_fds=True,
                )
                os.close(slave)
                output = bytearray()
                try:
                    for _ in range(100):
                        ready, _, _ = select.select([master], [], [], 0.05)
                        if ready:
                            output.extend(os.read(master, 65536))
                        if b"?1006h" in output:
                            break
                    if exit_kind == "keyboard":
                        os.write(master, b"q")
                    else:
                        process.send_signal(signal.SIGTERM)
                    process.wait(timeout=5)
                    while True:
                        ready, _, _ = select.select([master], [], [], 0)
                        if not ready:
                            break
                        try:
                            output.extend(os.read(master, 65536))
                        except OSError:
                            break
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                    os.close(master)
                self.assertEqual(process.returncode, 0)
                self.assertIn(b"?1006h", output)
                self.assertIn(b"?1006l", output)
                self.assertIn(b"?25h", output)


if __name__ == "__main__":
    unittest.main()
