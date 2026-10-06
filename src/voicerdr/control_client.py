import json
import socket
import uuid
from pathlib import Path
from typing import Any


class ControlClientError(RuntimeError):
    pass


class ControlClient:
    def __init__(self, sock_path: Path, timeout: float = 2.0) -> None:
        self.sock_path = sock_path
        self.timeout = timeout

    def ping(self) -> dict[str, Any]:
        return self.call("ping")

    def status(self, *, notify: bool = True) -> dict[str, Any]:
        return self.call("status", None if notify else {"notify": False})

    def listen(self) -> dict[str, Any]:
        # Moonshine may need to load ONNX weights on first listen.
        return self.call("set_mode", {"mode": "listen"}, timeout=60.0)

    def mute(self) -> dict[str, Any]:
        # A successful mute is a microphone-closure barrier. It may have to
        # wait behind a serialized first-time model load and then join capture.
        return self.call("set_mode", {"mode": "mute"}, timeout=60.0)

    def quit(self) -> dict[str, Any]:
        return self.call("quit")

    def handoff_quit(self) -> dict[str, Any]:
        """Atomically freeze microphone mode and begin daemon shutdown."""
        return self.call("handoff_quit")

    def ingest_transcript(
        self,
        text: str,
        *,
        utterance_id: str | None = None,
        timeout: float = 60.0,
        retries: int = 1,
    ) -> dict[str, Any]:
        """Generate one retry identity before transport and reuse it on retries."""
        stable_id = utterance_id or f"typed:{uuid.uuid4().hex}"
        params = {"text": text, "utterance_id": stable_id}
        last_error: ControlClientError | None = None
        for _attempt in range(retries + 1):
            try:
                return self.call("ingest_transcript", params, timeout=timeout)
            except ControlClientError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not self.sock_path.exists():
            raise ControlClientError(f"control socket missing: {self.sock_path}")
        req = {"id": "1", "method": method, "params": params or {}}
        payload = (json.dumps(req) + "\n").encode()
        wait = self.timeout if timeout is None else timeout
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(wait)
                sock.connect(str(self.sock_path))
                sock.sendall(payload)
                buf = b""
                while b"\n" not in buf:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    if len(buf) > 1_048_576:
                        raise ControlClientError("daemon response exceeded 1 MiB")
        except ControlClientError:
            raise
        except OSError as exc:
            raise ControlClientError(f"control request failed: {exc}") from exc
        if not buf:
            raise ControlClientError("empty response from daemon")
        try:
            line = buf.split(b"\n", 1)[0].decode()
            data = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlClientError("invalid response from daemon") from exc
        if not data.get("ok", False):
            err = data.get("error") or {}
            raise ControlClientError(err.get("message") or "daemon error")
        result = data.get("result")
        return result if isinstance(result, dict) else {}
