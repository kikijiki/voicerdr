import json
import shlex
import socket
import subprocess
import uuid
from typing import Any


class HerdrError(RuntimeError):
    def __init__(self, message: str, *, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload


class HerdrClient:
    """Talk to Herdr via CLI (portable) and raw Unix socket (events)."""

    def __init__(self, bin_path: str = "herdr", socket_path: str | None = None) -> None:
        self.bin_path = bin_path
        self.socket_path = socket_path

    def cli_json(self, args: list[str], *, timeout: float = 30.0) -> Any:
        cmd = [self.bin_path, *args]
        # Many herdr commands already emit JSON; --json where supported.
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise HerdrError(f"herdr timed out: {' '.join(args)}") from exc
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            raise HerdrError(
                err or out or f"herdr exited {proc.returncode}",
                payload={"stdout": out, "stderr": err, "code": proc.returncode},
            )
        if not out:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            # Some commands print human text; return raw.
            return {"raw": out}

    def socket_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if not self.socket_path:
            raise HerdrError("HERDR_SOCKET_PATH is not set")
        req_id = f"voicerdr-{uuid.uuid4().hex[:10]}"
        payload = (
            json.dumps(
                {"id": req_id, "method": method, "params": params or {}}
            ).encode()
            + b"\n"
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(10.0)
            sock.connect(self.socket_path)
            sock.sendall(payload)
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
        if not buf:
            raise HerdrError(f"empty response for {method}")
        data = json.loads(buf.split(b"\n", 1)[0].decode())
        if "error" in data:
            raise HerdrError(str(data["error"]), payload=data)
        return data.get("result", data)

    def workspace_list(self) -> list[dict[str, Any]]:
        data = self.cli_json(["workspace", "list"])
        return _extract_list(data, "workspaces", "workspace")

    def pane_list(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        args = ["pane", "list"]
        if workspace_id:
            args.extend(["--workspace", workspace_id])
        data = self.cli_json(args)
        return _extract_list(data, "panes", "pane")

    def agent_list(self) -> list[dict[str, Any]]:
        data = self.cli_json(["agent", "list"])
        return _extract_list(data, "agents", "agent")

    def agent_prompt(self, target: str, text: str, *, wait: bool = False) -> Any:
        args = ["agent", "prompt", target, text]
        if wait:
            args.append("--wait")
        return self.cli_json(args, timeout=120.0 if wait else 30.0)

    def agent_get(self, target: str) -> dict[str, Any]:
        data = self.cli_json(["agent", "get", target])
        if isinstance(data, dict):
            agent = (data.get("result") or {}).get("agent")
            if isinstance(agent, dict):
                return agent
            if "agent" in data and isinstance(data["agent"], dict):
                return data["agent"]
        raise HerdrError(f"unexpected agent.get payload: {data!r}")

    def agent_read(
        self,
        target: str,
        *,
        source: str = "recent-unwrapped",
        lines: int = 40,
    ) -> str:
        data = self.cli_json(
            [
                "agent",
                "read",
                target,
                "--source",
                source,
                "--lines",
                str(lines),
            ]
        )
        if isinstance(data, dict) and "raw" in data:
            return str(data["raw"])
        if isinstance(data, str):
            return data
        # CLI often prints plain text, already wrapped as raw above; fallback.
        result = data.get("result") if isinstance(data, dict) else None
        if isinstance(result, dict) and "text" in result:
            return str(result["text"])
        return json.dumps(data)

    def notification_show(self, title: str, body: str = "", sound: str = "none") -> Any:
        args = ["notification", "show", title, "--sound", sound]
        if body:
            args.extend(["--body", body])
        return self.cli_json(args)

    def workspace_report_metadata(
        self,
        workspace_id: str,
        *,
        source: str,
        tokens: dict[str, str | None],
        ttl_ms: int | None = 86_400_000,
    ) -> Any:
        params: dict[str, Any] = {
            "workspace_id": workspace_id,
            "source": source,
            "tokens": tokens,
        }
        if ttl_ms is not None:
            params["ttl_ms"] = ttl_ms
        return self.socket_request("workspace.report_metadata", params)

    def workspace_rename(self, workspace_id: str, label: str) -> Any:
        return self.cli_json(["workspace", "rename", workspace_id, label])

    def workspace_focus(self, workspace_id: str) -> Any:
        """Select a workspace in the Herdr UI (best-effort UX, not routing)."""
        return self.cli_json(["workspace", "focus", workspace_id])

    def workspace_create(self, *, cwd: str, label: str) -> dict[str, str]:
        data = self.cli_json(
            ["workspace", "create", "--cwd", cwd, "--label", label, "--no-focus"]
        )
        result = data.get("result", data) if isinstance(data, dict) else {}
        workspace = result.get("workspace") if isinstance(result, dict) else None
        pane = result.get("root_pane") if isinstance(result, dict) else None
        workspace_id = (
            workspace.get("workspace_id") if isinstance(workspace, dict) else None
        )
        pane_id = pane.get("pane_id") if isinstance(pane, dict) else None
        if not workspace_id or not pane_id:
            raise HerdrError(f"unexpected workspace.create payload: {data!r}")
        return {"workspace_id": str(workspace_id), "pane_id": str(pane_id)}

    def workspace_close(self, workspace_id: str) -> Any:
        return self.cli_json(["workspace", "close", workspace_id])

    def pane_run(self, pane_id: str, command: str) -> Any:
        return self.cli_json(["pane", "run", pane_id, command])

    def pane_rename(self, pane_id: str, label: str) -> Any:
        return self.cli_json(["pane", "rename", pane_id, label])

    @staticmethod
    def activity_command(python: str, path: str, state_path: str, lines: int) -> str:
        return " ".join(
            (
                "exec",
                shlex.quote(python),
                "-u",
                "-m",
                "voicerdr.activity",
                shlex.quote(path),
                "--state",
                shlex.quote(state_path),
                "--lines",
                str(max(1, lines)),
            )
        )


def _extract_list(data: Any, plural: str, singular: str) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    result = data.get("result", data)
    if isinstance(result, dict):
        if plural in result and isinstance(result[plural], list):
            return [x for x in result[plural] if isinstance(x, dict)]
        if singular in result and isinstance(result[singular], dict):
            return [result[singular]]
        # nested type envelopes
        for key in (plural, "items"):
            if key in result and isinstance(result[key], list):
                return [x for x in result[key] if isinstance(x, dict)]
    return []
