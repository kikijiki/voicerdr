import logging
import threading
from collections.abc import Callable
from typing import Any

from voicerdr.herdr_client import HerdrClient, HerdrError

log = logging.getLogger(__name__)

StatusHandler = Callable[[dict[str, Any]], None]


class AgentStatusWatcher:
    """Poll agent.list and emit transitions.

    Herdr's `pane.agent_status_changed` socket subscription requires a concrete
    `pane_id`, so a fleet-wide secretary uses polling (plus optional plugin
    event hooks) instead of a single unscoped subscribe.
    """

    def __init__(
        self,
        client: HerdrClient,
        *,
        on_transition: StatusHandler | None = None,
        interval_secs: float = 2.0,
    ) -> None:
        self.client = client
        self.on_transition = on_transition
        self.interval_secs = interval_secs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: dict[str, str] = {}
        self.connected = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="agent-status-watcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def poll_once(self) -> None:
        try:
            agents = self.client.agent_list()
        except HerdrError as exc:
            self.connected = False
            log.warning("agent.list failed: %s", exc)
            return
        self.connected = True
        current: dict[str, str] = {}
        for agent in agents:
            pane_id = str(agent.get("pane_id") or "")
            status = str(agent.get("agent_status") or "unknown")
            if not pane_id:
                continue
            current[pane_id] = status
            prev = self._last.get(pane_id)
            if prev is not None and prev != status and self.on_transition:
                payload = {
                    "type": "pane.agent_status_changed",
                    "pane_id": pane_id,
                    "agent_status": status,
                    "previous_status": prev,
                    "agent": agent,
                }
                try:
                    self.on_transition(payload)
                except Exception:
                    log.exception("status transition handler failed")
        self._last = current

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval_secs)
