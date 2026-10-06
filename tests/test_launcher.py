import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class RuntimeLauncherTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        source_root = Path(__file__).resolve().parents[1]
        scripts = root / "scripts"
        scripts.mkdir(parents=True)
        launcher = scripts / "run.sh"
        shutil.copy2(source_root / "scripts" / "run.sh", launcher)
        (root / "flake.nix").write_text("{}\n", encoding="utf-8")
        (root / "flake.lock").write_text("{}\n", encoding="utf-8")

        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        nix_log = root / "nix.log"
        uv_log = root / "uv.log"
        self._write_executable(
            fake_bin / "nix",
            """#!/usr/bin/env bash
set -euo pipefail
printf 'call' >>"$NIX_LOG"
printf '\t%s' "$@" >>"$NIX_LOG"
printf '\n' >>"$NIX_LOG"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--profile" ]]; then
    ln -s / "${args[$((i + 1))]}"
    break
  fi
done
while [[ $# -gt 0 && "$1" != "--command" ]]; do shift; done
[[ $# -gt 0 ]]
shift
exec "$@"
""",
        )
        self._write_executable(
            fake_bin / "uv",
            """#!/usr/bin/env bash
set -euo pipefail
printf 'call' >>"$UV_LOG"
printf '\t%s' "$@" >>"$UV_LOG"
printf '\n' >>"$UV_LOG"
""",
        )
        return launcher, nix_log, uv_log

    @staticmethod
    def _write_executable(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    def test_reexecs_once_forwards_arguments_and_reuses_profile(self) -> None:
        with tempfile.TemporaryDirectory(prefix="voicerdr packaged root ") as directory:
            root = Path(directory)
            launcher, nix_log, uv_log = self._fixture(root)
            env = {
                **os.environ,
                "PATH": f"{root / 'fake-bin'}:{os.environ['PATH']}",
                "NIX_LOG": str(nix_log),
                "UV_LOG": str(uv_log),
            }
            env.pop("VOICERDR_NIX_RUNTIME", None)
            result = subprocess.run(
                [launcher, "ctl", "ingest", "--text", "hello * world"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            runtime_key = (
                (root / ".voicerdr-nix-runtime.stamp")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                nix_log.read_text(encoding="utf-8").splitlines(),
                [
                    "\t".join(
                        (
                            "call",
                            "develop",
                            "--profile",
                            str(root / ".voicerdr-nix-runtime"),
                            str(root),
                            "--command",
                            "env",
                            f"VOICERDR_NIX_RUNTIME={root}",
                            f"VOICERDR_NIX_PROFILE_STAMP={root / '.voicerdr-nix-runtime.stamp'}",
                            f"VOICERDR_NIX_PROFILE_KEY={runtime_key}",
                            "bash",
                            str(launcher),
                            "ctl",
                            "ingest",
                            "--text",
                            "hello * world",
                        )
                    )
                ],
            )
            self.assertEqual(
                uv_log.read_text(encoding="utf-8").splitlines(),
                [
                    "call\tsync\t--quiet\t--frozen",
                    "call\trun\tvoicerdr\tctl\tingest\t--text\thello * world",
                ],
            )

            second = subprocess.run(
                [launcher, "--version"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                nix_log.read_text(encoding="utf-8").splitlines()[1],
                "\t".join(
                    (
                        "call",
                        "develop",
                        str(root / ".voicerdr-nix-runtime"),
                        "--command",
                        "env",
                        f"VOICERDR_NIX_RUNTIME={root}",
                        "bash",
                        str(launcher),
                        "--version",
                    )
                ),
            )

    def test_runtime_marker_prevents_recursion_and_sync_only_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, nix_log, uv_log = self._fixture(root)
            env = {
                **os.environ,
                "PATH": f"{root / 'fake-bin'}:{os.environ['PATH']}",
                "NIX_LOG": str(nix_log),
                "UV_LOG": str(uv_log),
                "VOICERDR_NIX_RUNTIME": str(root),
            }
            result = subprocess.run(
                [launcher, "--sync-only"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(nix_log.exists())
            self.assertEqual(
                uv_log.read_text(encoding="utf-8").splitlines(),
                ["call\tsync\t--quiet\t--frozen"],
            )

    def test_without_nix_preserves_native_uv_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, nix_log, uv_log = self._fixture(root)
            fake_bin = root / "fake-bin"
            (fake_bin / "nix").unlink()
            bash = shutil.which("bash")
            dirname = shutil.which("dirname")
            self.assertIsNotNone(bash)
            self.assertIsNotNone(dirname)
            (fake_bin / "bash").symlink_to(bash)
            (fake_bin / "dirname").symlink_to(dirname)
            env = {
                **os.environ,
                "PATH": str(fake_bin),
                "NIX_LOG": str(nix_log),
                "UV_LOG": str(uv_log),
            }
            env.pop("VOICERDR_NIX_RUNTIME", None)

            result = subprocess.run(
                [bash, launcher, "--version"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(nix_log.exists())
            self.assertEqual(
                uv_log.read_text(encoding="utf-8").splitlines(),
                [
                    "call\tsync\t--quiet\t--frozen",
                    "call\trun\tvoicerdr\t--version",
                ],
            )


if __name__ == "__main__":
    unittest.main()
