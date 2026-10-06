import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def write_recorder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['CALL_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')\n"
        "if os.environ.get('FAIL_CLEANUP') and sys.argv[1:3] == ['plugin', 'action']:\n"
        "    print('cleanup failed: daemon still running', file=sys.stderr)\n"
        "    sys.exit(23)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def recorded_calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


class PluginLifecycleScriptTests(unittest.TestCase):
    def test_cleanup_failure_aborts_disable_uninstall_and_unlink(self) -> None:
        for action, cleanup in (
            ("disable", "service-disable"),
            ("uninstall", "service-uninstall"),
            ("unlink", "service-uninstall"),
        ):
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                binary = root / "herdr"
                log = root / "calls.jsonl"
                write_recorder(binary)
                result = subprocess.run(
                    ["bash", str(SOURCE_ROOT / "scripts/plugin-lifecycle.sh"), action],
                    env={
                        **os.environ,
                        "HERDR_BIN_PATH": str(binary),
                        "CALL_LOG": str(log),
                        "FAIL_CLEANUP": "1",
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 23)
                self.assertIn("daemon still running", result.stderr)
                self.assertEqual(
                    recorded_calls(log),
                    [["herdr", "plugin", "action", "invoke", f"voicerdr.{cleanup}"]],
                )

    def test_successful_cleanup_precedes_plugin_mutation(self) -> None:
        for action, cleanup in (
            ("disable", "service-disable"),
            ("uninstall", "service-uninstall"),
            ("unlink", "service-uninstall"),
        ):
            with (
                self.subTest(action=action),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                binary = root / "herdr"
                log = root / "calls.jsonl"
                write_recorder(binary)
                result = subprocess.run(
                    ["bash", str(SOURCE_ROOT / "scripts/plugin-lifecycle.sh"), action],
                    env={
                        **os.environ,
                        "HERDR_BIN_PATH": str(binary),
                        "CALL_LOG": str(log),
                        "FAIL_CLEANUP": "",
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    recorded_calls(log),
                    [
                        ["herdr", "plugin", "action", "invoke", f"voicerdr.{cleanup}"],
                        ["herdr", "plugin", action, "voicerdr"],
                    ],
                )


@unittest.skipUnless(shutil.which("just"), "just is not installed")
class JustRecipeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict[str, str]]:
        shutil.copy2(SOURCE_ROOT / "justfile", root / "justfile")
        for path in (
            root / "fake-bin/uv",
            root / "fake-bin/herdr",
            root / "scripts/run.sh",
            root / "scripts/plugin-lifecycle.sh",
        ):
            write_recorder(path)
        log = root / "calls.jsonl"
        return log, {
            **os.environ,
            "PATH": f"{root / 'fake-bin'}:{os.environ['PATH']}",
            "CALL_LOG": str(log),
            "FAIL_CLEANUP": "",
        }

    def test_user_arguments_are_forwarded_literally_from_paths_with_spaces(
        self,
    ) -> None:
        payload = 'hey Jenny, it\'s $(touch injected-command); * "quoted"\nnext line'
        for recipe, flag in (
            ("ingest", "--text"),
            ("say", "--text"),
            ("resolve", "--space"),
        ):
            with (
                self.subTest(recipe=recipe),
                tempfile.TemporaryDirectory(
                    prefix="voicerdr recipes $ ' "
                ) as directory,
            ):
                root = Path(directory)
                log, env = self._fixture(root)
                result = subprocess.run(
                    ["just", recipe, payload],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    recorded_calls(log),
                    [["uv", "sync"], ["run.sh", "ctl", recipe, flag, payload]],
                )
                self.assertFalse((root / "injected-command").exists())

    def test_plugin_recipes_preserve_checkout_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="voicerdr recipes $ ' ") as directory:
            root = Path(directory)
            log, env = self._fixture(root)
            result = subprocess.run(
                ["just", "link", "unlink", "plugin-enable", "plugin-disable"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                recorded_calls(log),
                [
                    ["herdr", "plugin", "link", str(root)],
                    ["plugin-lifecycle.sh", "unlink"],
                    ["plugin-lifecycle.sh", "enable"],
                    ["plugin-lifecycle.sh", "disable"],
                ],
            )


if __name__ == "__main__":
    unittest.main()
