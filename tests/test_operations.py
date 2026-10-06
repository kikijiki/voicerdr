import contextlib
import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voicerdr.cli import main
from voicerdr.config import AppConfig
from voicerdr.models import bootstrap_models
from voicerdr.operations import (
    MANAGED_MARKER,
    OperationsError,
    _quote_systemd,
    ensure_managed_daemon,
    ensure_systemd_service,
    install_service,
    render_service,
    service_files,
    uninstall_service,
    write_service_environment,
)
from voicerdr.paths import RuntimePaths


def runtime_paths(root: Path) -> RuntimePaths:
    return RuntimePaths(
        plugin_root=root,
        config_dir=root / "config-state",
        state_dir=root / "runtime-state",
        herdr_bin="/opt/herdr bin/herdr",
        herdr_socket="/run/user/1000/herdr session.sock",
    )


class SystemdRenderingTests(unittest.TestCase):
    def test_unit_rendering_quotes_paths_and_marks_ownership(self) -> None:
        with tempfile.TemporaryDirectory(prefix="voice % $ ") as temp_dir:
            root = Path(temp_dir)
            launcher = root / "scripts" / "run.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            launcher.chmod(0o755)
            template = (
                "@MANAGED_MARKER@\nWorkingDirectory=@WORKING_DIRECTORY@\n"
                "EnvironmentFile=@ENVIRONMENT_FILE@\n"
                "ExecStart=@LAUNCHER@ daemon --foreground\n"
            )
            with (
                patch("voicerdr.operations._template", return_value=template),
                patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "xdg")}),
            ):
                rendered = render_service(runtime_paths(root))
        self.assertIn(MANAGED_MARKER, rendered)
        self.assertIn("%%", rendered)
        self.assertIn("$$", rendered)
        self.assertNotIn("@LAUNCHER@", rendered)
        self.assertIn("scripts/run.sh", rendered)
        # WorkingDirectory=/EnvironmentFile= reject quoted paths on systemd 261.
        self.assertRegex(rendered, r"(?m)^WorkingDirectory=/.+")
        self.assertNotRegex(rendered, r'(?m)^WorkingDirectory="')
        self.assertRegex(rendered, r"(?m)^EnvironmentFile=/.+")
        self.assertNotRegex(rendered, r'(?m)^EnvironmentFile="')
        self.assertRegex(rendered, r'(?m)^ExecStart="/.+/scripts/run\.sh" ')

    def test_environment_file_is_private_and_preserves_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = runtime_paths(root)
            with patch.dict(
                os.environ,
                {
                    "XDG_CONFIG_HOME": str(root / "xdg"),
                    "NLTK_DATA": str(root / "nltk data"),
                },
            ):
                environment = write_service_environment(paths)
            content = environment.read_text(encoding="utf-8")
            mode = stat.S_IMODE(environment.stat().st_mode)
        self.assertEqual(mode, 0o600)
        self.assertIn('HERDR_SOCKET_PATH="/run/user/1000/herdr session.sock"', content)
        self.assertIn('NLTK_DATA="', content)

    def test_quote_rejects_multiline_values(self) -> None:
        with self.assertRaises(OperationsError):
            _quote_systemd("unsafe\nvalue")


class SystemdLifecycleTests(unittest.TestCase):
    def test_install_refuses_to_overwrite_unmanaged_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = runtime_paths(root)
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "xdg")}):
                unit = service_files(paths).unit
                unit.parent.mkdir(parents=True)
                unit.write_text("[Service]\nExecStart=/something/else\n")
                with self.assertRaises(OperationsError):
                    install_service(paths)

    def test_uninstall_checks_ownership_before_stopping_anything(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = runtime_paths(root)
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "xdg")}):
                unit = service_files(paths).unit
                unit.parent.mkdir(parents=True)
                unit.write_text("[Service]\nExecStart=/something/else\n")
                with (
                    patch("voicerdr.operations.stop_daemon") as stop,
                    self.assertRaises(OperationsError),
                ):
                    uninstall_service(paths)
                stop.assert_not_called()

    def test_uninstall_preserves_service_files_when_daemon_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = runtime_paths(root)
            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "xdg")}):
                files = service_files(paths)
                files.unit.parent.mkdir(parents=True)
                files.unit.write_text(MANAGED_MARKER + "\n")
                files.environment.parent.mkdir(parents=True)
                files.environment.write_text("HERDR_ENV=1\n")
                with (
                    patch(
                        "voicerdr.operations.stop_daemon",
                        return_value={"ok": False, "remaining": [12345]},
                    ),
                    patch("voicerdr.operations._systemctl") as systemctl,
                    self.assertRaisesRegex(OperationsError, "daemon did not stop"),
                ):
                    uninstall_service(paths)
                self.assertEqual(files.unit.read_text(), MANAGED_MARKER + "\n")
                self.assertEqual(files.environment.read_text(), "HERDR_ENV=1\n")
                systemctl.assert_called_once_with(
                    "disable", "--now", "voicerdr.service"
                )

    def test_uninstall_without_unit_stops_detached_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = runtime_paths(root)
            with (
                patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "xdg")}),
                patch(
                    "voicerdr.operations.stop_daemon", return_value={"ok": True}
                ) as stop,
                patch("voicerdr.operations._systemctl") as systemctl,
            ):
                result = uninstall_service(paths)
            self.assertTrue(result["uninstalled"])
            stop.assert_called_once_with(paths)
            systemctl.assert_not_called()

    def test_service_disable_cli_reports_surviving_daemon_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = runtime_paths(root)
            output = io.StringIO()
            with (
                patch.dict(os.environ, {"XDG_CONFIG_HOME": str(root / "xdg")}),
                patch("voicerdr.cli.resolve_paths", return_value=paths),
                patch(
                    "voicerdr.operations.stop_daemon",
                    return_value={"ok": False, "remaining": [12345]},
                ),
                contextlib.redirect_stdout(output),
            ):
                exit_code = main(["service", "disable"])
            result = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["result"]["stop"]["remaining"], [12345])

    def test_managed_ensure_selects_systemd_only_for_managed_unit(self) -> None:
        paths = runtime_paths(Path("/tmp/voicerdr-ops-test"))
        with (
            patch("voicerdr.operations.service_is_installed", return_value=True),
            patch(
                "voicerdr.operations.ensure_systemd_service",
                return_value={"systemd": True},
            ) as systemd,
            patch("voicerdr.operations.ensure_daemon") as detached,
        ):
            result = ensure_managed_daemon(paths)
        self.assertTrue(result["systemd"])
        systemd.assert_called_once_with(paths, timeout=20.0)
        detached.assert_not_called()

    def test_unhealthy_active_service_is_stopped_cleaned_and_started(self) -> None:
        paths = runtime_paths(Path("/tmp/voicerdr-ops-test"))
        active = subprocess.CompletedProcess([], 0, "", "")
        order: list[str] = []

        def systemctl_side_effect(*args: str, **_kwargs: object):
            order.append("systemctl:" + " ".join(args))
            return active

        with (
            patch("voicerdr.operations.service_is_installed", return_value=True),
            patch("voicerdr.operations.write_service_environment"),
            patch(
                "voicerdr.operations._systemctl", side_effect=systemctl_side_effect
            ) as systemctl,
            patch("voicerdr.operations._health", return_value=None),
            patch(
                "voicerdr.operations.stop_daemon",
                side_effect=lambda *_args, **_kwargs: (
                    order.append("handoff-stop") or {"ok": True}
                ),
            ) as stop,
            patch("voicerdr.operations._wait_for_health", return_value={"pid": 42}),
        ):
            result = ensure_systemd_service(paths)
        self.assertEqual(result["status"], {"pid": 42})
        self.assertIn(
            unittest.mock.call("stop", "voicerdr.service"), systemctl.call_args_list
        )
        self.assertIn(
            unittest.mock.call("start", "voicerdr.service"), systemctl.call_args_list
        )
        stop.assert_called_once_with(paths, wait_secs=3.0, _serialized=False)
        self.assertLess(
            order.index("handoff-stop"), order.index("systemctl:stop voicerdr.service")
        )
        self.assertLess(
            order.index("systemctl:stop voicerdr.service"),
            order.index("systemctl:start voicerdr.service"),
        )


class ModelBootstrapTests(unittest.TestCase):
    def test_bootstrap_orchestrates_selected_public_provisioners(self) -> None:
        config = AppConfig(stt_model="base", tts_voice="af_heart")
        with (
            patch("voicerdr.models._verify_vad", return_value={"ready": True}) as vad,
            patch(
                "voicerdr.models._bootstrap_tokenizer", return_value={"ready": True}
            ) as tokenizer,
            patch(
                "voicerdr.models._bootstrap_stt", return_value={"ready": True}
            ) as stt,
            patch("voicerdr.models._bootstrap_tts") as tts,
        ):
            result = bootstrap_models(config, tts=False, progress=lambda _line: None)
        self.assertIn("silero_vad", result)
        self.assertIn("nltk", result)
        self.assertIn("moonshine_stt", result)
        self.assertNotIn("kokoro_tts", result)
        vad.assert_called_once_with(config.sample_rate)
        tokenizer.assert_called_once()
        stt.assert_called_once()
        tts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
