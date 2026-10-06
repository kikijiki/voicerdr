import json
import logging
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from voicerdr.herdr_client import HerdrClient, HerdrError

log = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], None]


class EventSubscriber:
    """Long-lived events.subscribe reader for unscoped Herdr events."""

    def __init__(
        self,
        client: HerdrClient,
        *,
        subscriptions: list[dict[str, Any]] | None = None,
        on_event: EventHandler | None = None,
        retry_secs: float = 2.0,
    ) -> None:
        self.client = client
        # Only unscoped subscription types here. pane.agent_status_changed
        # requires pane_id — use AgentStatusWatcher / plugin event hooks.
        self.subscriptions = subscriptions or [
            {"type": "workspace.focused"},
            {"type": "workspace.created"},
            {"type": "workspace.closed"},
            {"type": "workspace.renamed"},
            {"type": "workspace.reordered"},
            {"type": "pane.created"},
            {"type": "pane.closed"},
        ]
        self.on_event = on_event
        self.retry_secs = retry_secs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="herdr-events", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._session()
            except Exception as exc:  # noqa: BLE001 — reconnect loop
                self.connected = False
                log.warning("herdr event session ended: %s", exc)
                self._stop.wait(self.retry_secs)

    def _session(self) -> None:
        if not self.client.socket_path:
            raise HerdrError("no HERDR_SOCKET_PATH")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect(self.client.socket_path)
            sub_id = f"voicerdr-sub-{int(time.time())}"
            req = {
                "id": sub_id,
                "method": "events.subscribe",
                "params": {"subscriptions": self.subscriptions},
            }
            sock.sendall((json.dumps(req) + "\n").encode())
            sock.settimeout(2.0)
            self.connected = True
            log.info("subscribed to Herdr events: %s", self.subscriptions)
            buf = b""
            while not self._stop.is_set():
                try:
                    chunk = sock.recv(65536)
                except TimeoutError:
                    continue
                if not chunk:
                    raise HerdrError("herdr socket closed")
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode())
                    except json.JSONDecodeError:
                        continue
                    if msg.get("id") == sub_id and "result" in msg:
                        continue
                    if "error" in msg:
                        raise HerdrError(str(msg["error"]), payload=msg)
                    if self.on_event:
                        try:
                            self.on_event(msg)
                        except Exception:
                            log.exception("event handler failed")
