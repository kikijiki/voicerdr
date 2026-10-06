import fcntl
import hashlib
import hmac
import json
import logging
import os
import re
import signal
import socket
import stat
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from voicerdr.config import AppConfig, load_aliases, seed_config_files
from voicerdr.control_protocol import ControlRequest, err, ok
from voicerdr.dictation import DictationBuffer
from voicerdr.events import EventSubscriber
from voicerdr.herdr_client import HerdrClient, HerdrError
from voicerdr.intent import (
    BoundIntentPlan,
    IntentEvidence,
    IntentPlan,
    IntentTarget,
    IntentValidationError,
    match_wake_phrase,
    plan_from_json,
)
from voicerdr.paths import RuntimePaths, ensure_dirs
from voicerdr.routing import (
    ResolveResult,
    agent_name,
    resolve_route,
    resolve_workspace,
)
from voicerdr.secretary import (
    VERIFICATION_REASON_PAYLOAD,
    SecretaryClient,
    SecretaryError,
    SecretaryOutputError,
    VerificationResult,
    canonical_digest,
    payload_evidence_facts,
)
from voicerdr.space_labels import strip_number_prefix, sync_space_number_labels
from voicerdr.status_watcher import AgentStatusWatcher
from voicerdr.talk_policy import TalkPolicy
from voicerdr.tts import Speaker
from voicerdr.voice import FinalTranscript, VoiceListener

log = logging.getLogger(__name__)
ACTIVITY_UI_VERSION = "6"
SAFE_PROMPT_AGENT_STATES = frozenset({"idle", "done", "working"})
LLM_CORRECTIVE_ATTEMPTS = 3


@dataclass(frozen=True)
class VerifiedPromptCapability:
    utterance_id: str
    utterance_digest: str
    catalog_digest: str
    plan_digest: str
    workspace_id: str
    pane_id: str
    message: str
    evidence_source: str
    evidence_quote: str
    source_bundle_digest: str
    workspace_evidence_source: str
    workspace_evidence_quote: str
    workspace_catalog_number: str
    agent_evidence_source: str
    agent_evidence_quote: str
    target_binding_digest: str
    origin: str
    origin_generation: int
    global_generation: int
    seal: str


class ReplayLedgerError(RuntimeError):
    """Durable replay protection is unavailable or untrustworthy."""


class MicPreferenceError(RuntimeError):
    """The explicit microphone preference could not be read or persisted."""


class MicClosureError(RuntimeError):
    """Microphone capture could not be proven closed within the bound."""


class MicModeFrozenError(RuntimeError):
    """Shutdown froze the reported mode before a safe mute plan completed."""


class Daemon:
    def __init__(self, paths: RuntimePaths, config: AppConfig) -> None:
        self.paths = paths
        self.config = config
        self.policy = TalkPolicy(config)
        self.herdr = HerdrClient(paths.herdr_bin, paths.herdr_socket)
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._started_at = time.time()
        self.last_event: dict[str, Any] | None = None
        self.last_summary: str | None = None
        self.last_transcript: str | None = None
        self.last_voice_action: dict[str, Any] | None = None
        self.activity_workspace: dict[str, str] | None = None
        self._activity_lock = threading.Lock()
        self._activity_state_lock = threading.Lock()
        self._activity_status: dict[str, Any] = {}
        self._current_input_mode = "idle"
        self.pending_clarification: dict[str, Any] | None = None
        self._voice_state_lock = threading.RLock()
        self._authority_lock = threading.RLock()
        self._listener_generation = 0
        self._typed_generation = 0
        self._global_generation = 0
        self._delivery_closed = False
        self._mode_transition_lock = threading.RLock()
        self._mic_transition_count = 0
        self._shutdown_scheduled = False
        self._shutdown_complete = threading.Event()
        self._capability_key = os.urandom(32)
        self._consumed_utterance_ids: set[str] = set()
        self._replay_ledger_error: str | None = None
        self._replay_ledger_ready = False
        self._daemon_lock_fd: int | None = None
        self._mic_preference_lock = threading.Lock()
        self._mic_preference_explicit = False
        self._mic_preference_error: str | None = None
        self._control_socket_identity: tuple[int, int] | None = None
        self._control_socket_path_fd: int | None = None
        self.subscriber: EventSubscriber | None = None
        self.status_watcher: AgentStatusWatcher | None = None
        self.focused_workspace_id: str | None = None
        self.voice: VoiceListener | None = None
        self._voice_lock = threading.Lock()
        self.dictation = DictationBuffer()
        self.speaker = Speaker(
            voice=config.tts_voice,
            enabled=bool(config.speak_enabled),
            speed=config.tts_speed,
        )
        self.secretary = SecretaryClient(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            timeout_secs=config.llm_timeout_secs,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            verify_ssl=config.llm_verify_ssl,
        )
        self.verifier = SecretaryClient(
            base_url=config.verifier_base_url or config.llm_base_url,
            api_key=config.verifier_api_key or config.llm_api_key,
            model=config.verifier_model or config.llm_model,
            timeout_secs=config.verifier_timeout_secs or config.llm_timeout_secs,
            temperature=(
                config.verifier_temperature
                if config.verifier_temperature is not None
                else config.llm_temperature
            ),
            max_tokens=config.verifier_max_tokens or config.llm_max_tokens,
            verify_ssl=(
                config.verifier_verify_ssl
                if config.verifier_verify_ssl is not None
                else config.llm_verify_ssl
            ),
        )

    def run_forever(self) -> int:
        self._acquire_daemon_lock()
        seed_config_files(self.paths)
        try:
            self._consumed_utterance_ids = self._initialize_replay_ledger()
            self._replay_ledger_ready = True
        except ReplayLedgerError as exc:
            self._replay_ledger_error = str(exc)
            self._replay_ledger_ready = False
            log.error("replay ledger unavailable: %s", exc)
        ensure_dirs(self.paths)
        self._restore_mic_preference()
        self._write_session()
        self._install_signals()
        self._bind_control()
        self._start_events()
        self._seed_focus()
        self._sync_space_labels()
        self._record_activity(
            "daemon_started",
            pid=os.getpid(),
            mic_mode=self.policy.mic_mode,
            message=f"{self.config.assistant_name} voice assistant started.",
        )
        if self._replay_ledger_error:
            self._record_activity(
                "replay_ledger_error",
                reason=self._replay_ledger_error,
                sent=False,
            )
        listening = self.policy.mic_mode == "listen"
        self._set_activity_status(
            phase=(
                "error"
                if self._replay_ledger_error
                else "ready"
                if listening
                else "muted"
            ),
            mode="idle",
            capture_active=False,
            speech_active=False,
            waiting_for="wake word" if listening else "listen command",
            live_transcript="",
            dictation_buffer="",
            last_result=(
                f"Replay protection unavailable: {self._replay_ledger_error}. Voice commands will be withheld."
                if self._replay_ledger_error
                else "Microphone is ready."
                if listening
                else "Microphone muted."
            ),
            delivery_status=None,
        )
        self._ensure_activity_workspace()
        if self.policy.mic_mode == "listen":
            # Don't block the control accept loop on Moonshine load.
            threading.Thread(
                target=lambda: self._start_voice(wait_secs=45.0),
                name="voicerdr-voice-boot",
                daemon=True,
            ).start()
        if self.policy.speak_enabled:
            self.speaker.enabled = True
            self.speaker.start()
            threading.Thread(
                target=self._warm_tts,
                name="voicerdr-tts-warm",
                daemon=True,
            ).start()
        log.info(
            "voicerdr daemon up pid=%s mode=%s socket=%s focus=%s speak=%s dictation=%s",
            os.getpid(),
            self.policy.mic_mode,
            self.paths.control_socket,
            self.focused_workspace_id,
            self.policy.speak_enabled,
            self.config.dictation_enabled,
        )
        try:
            self._accept_loop()
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        self._close_delivery_authority()
        shutdown_complete = getattr(self, "_shutdown_complete", None)
        if shutdown_complete is None:
            shutdown_complete = threading.Event()
            self._shutdown_complete = shutdown_complete

        # Bound teardown so a stuck Pipecat/TTS thread cannot keep the process alive.
        def _force_exit() -> None:
            if shutdown_complete.wait(4.0):
                return
            log.error("shutdown hung — forcing process exit")
            os._exit(0)

        watchdog = threading.Thread(
            target=_force_exit, name="voicerdr-shutdown-watchdog", daemon=True
        )
        watchdog.start()
        try:
            try:
                self._stop_voice()
            except Exception:
                log.exception("voice stop during shutdown failed")
            try:
                self.speaker.stop()
            except Exception:
                log.debug("speaker stop during shutdown failed", exc_info=True)
            if self.subscriber:
                try:
                    self.subscriber.stop()
                except Exception:
                    log.debug("event subscriber stop failed", exc_info=True)
            if self.status_watcher:
                try:
                    self.status_watcher.stop()
                except Exception:
                    log.debug("status watcher stop failed", exc_info=True)
            if self._server:
                try:
                    self._server.close()
                except OSError:
                    pass
            self._unlink_owned_control_socket()
            for p in (self.paths.pidfile,):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass
        finally:
            shutdown_complete.set()
        # Keep the singleton descriptor until process exit. If listener teardown
        # timed out, no replacement daemon can overlap the stale capture thread.

    def _delayed_stop(self) -> None:
        time.sleep(0.05)
        self._stop.set()
        try:
            self._stop_voice()
        except Exception:
            log.exception("voice stop on quit failed")
        try:
            self.speaker.stop()
        except Exception:
            log.debug("speaker stop on quit failed", exc_info=True)
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
        # Accept loop exits → run_forever finally → shutdown(). If anything
        # blocks past the watchdog, force the interpreter down.
        threading.Thread(
            target=self._hard_exit_soon, name="voicerdr-hard-exit", daemon=True
        ).start()

    def _hard_exit_soon(self) -> None:
        shutdown_complete = getattr(self, "_shutdown_complete", None)
        if shutdown_complete is not None:
            if shutdown_complete.wait(5.0):
                return
        else:
            time.sleep(5.0)
        log.error("quit did not finish — os._exit")
        os._exit(0)

    def status(self) -> dict[str, Any]:
        mic = self._mic_state_snapshot()
        return {
            "pid": os.getpid(),
            "mode": mic["mode"],
            "speak_enabled": self.policy.speak_enabled,
            "talk_policy": self.policy.snapshot(),
            "tts_voice": self.config.tts_voice,
            "tts_error": self.speaker.last_error,
            "tts_last": self.speaker.last_spoken,
            "dictation_active": self.dictation.active,
            "dictation_enabled": self.config.dictation_enabled,
            "delivery_closed": mic["delivery_closed"],
            "mic_transition_pending": mic["mic_transition_pending"],
            "focus_on_prompt": self.config.focus_on_prompt,
            "voice_running": mic["voice_running"],
            "voice_error": mic["voice_error"],
            "stt_timing": self._stt_timing_status(),
            "herdr_socket": self.paths.herdr_socket,
            "herdr_bin": self.paths.herdr_bin,
            "subscribed": bool(self.subscriber and self.subscriber.connected),
            "watching_agents": bool(
                self.status_watcher and self.status_watcher.connected
            ),
            "focused_workspace_id": self.focused_workspace_id,
            "uptime_secs": round(time.time() - self._started_at, 1),
            "config_dir": str(self.paths.config_dir),
            "state_dir": str(self.paths.state_dir),
            "replay_ledger": {
                "ready": bool(getattr(self, "_replay_ledger_ready", False)),
                "error": getattr(self, "_replay_ledger_error", None),
                "path": str(self.paths.utterance_ledger),
            },
            "mic_preference": {
                "explicit": bool(getattr(self, "_mic_preference_explicit", False)),
                "mode": mic["mode"],
                "error": getattr(self, "_mic_preference_error", None),
                "path": str(self.paths.mic_preference),
            },
            "last_transcript": self.last_transcript,
            "last_voice_action": self.last_voice_action,
            "activity_workspace": {
                "enabled": self.config.activity_workspace_enabled,
                "workspace_id": (self.activity_workspace or {}).get("workspace_id"),
                "pane_id": (self.activity_workspace or {}).get("pane_id"),
                "log": str(self.paths.activity_log),
                "state": str(self.paths.activity_state),
            },
            "live_capture": self._activity_status_snapshot(mic=mic),
            "last_summary": self.last_summary,
            "assistant_name": self.config.assistant_name,
            "assistant_avatar": self.config.assistant_avatar,
            "wake_phrases": self.config.wake_phrases,
            "require_wake": self.config.require_wake,
            "aliases": dict(sorted(self.config.aliases.items())),
            "spaces": self._space_directory(),
            "llm": {
                "base_url": self.config.llm_base_url,
                "model": self.config.llm_model,
                "min_confidence": self.config.llm_min_confidence,
                "prompt_verification_required": True,
                "verifier_base_url": (
                    self.config.verifier_base_url or self.config.llm_base_url
                ),
                "verifier_model": self.config.verifier_model or self.config.llm_model,
            },
        }

    def _stt_timing_status(self) -> dict[str, int | float | None] | None:
        observation = (
            getattr(self.voice, "last_stt_observation", None) if self.voice else None
        )
        if observation is not None:
            # Do not duplicate transcript contents in metrics: status already
            # exposes the deliberately bounded ``last_transcript`` field.
            return {
                "sequence": observation.sequence,
                "utterance_id": observation.utterance_id,
                "speech_started_at": observation.speech_started_at,
                "speech_stopped_at": observation.speech_stopped_at,
                "transcript_received_at": observation.transcript_received_at,
                "utterance_seconds": observation.utterance_seconds,
                "start_to_transcript_seconds": (
                    observation.start_to_transcript_seconds
                ),
                "stt_latency_seconds": observation.stt_latency_seconds,
            }
        return None

    def _record_activity(self, event: str, **fields: Any) -> None:
        if not getattr(self, "paths", None):
            return
        payload = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            **fields,
        }
        try:
            ensure_dirs(self.paths)
            line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
            lock = getattr(self, "_activity_lock", None)
            if lock is None:
                self._activity_lock = threading.Lock()
                lock = self._activity_lock
            with lock:
                with self.paths.activity_log.open("a", encoding="utf-8") as stream:
                    stream.write(line)
                self.paths.activity_log.chmod(0o600)
        except OSError:
            log.debug("could not write activity journal", exc_info=True)

    def _mic_state_snapshot(self) -> dict[str, Any]:
        lock = getattr(self, "_authority_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._authority_lock = lock
        with lock:
            voice = getattr(self, "voice", None)
            policy = getattr(self, "policy", None)
            return {
                "mode": getattr(policy, "mic_mode", "mute"),
                "delivery_closed": bool(getattr(self, "_delivery_closed", False)),
                "mic_transition_pending": bool(
                    getattr(self, "_mic_transition_count", 0)
                ),
                "voice_running": bool(voice and getattr(voice, "running", False)),
                "voice_error": getattr(voice, "last_error", None) if voice else None,
            }

    def _activity_status_snapshot(
        self, *, mic: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        lock = getattr(self, "_activity_state_lock", None)
        if lock is None:
            state = dict(getattr(self, "_activity_status", {}))
        else:
            with lock:
                state = dict(getattr(self, "_activity_status", {}))
        safety = mic or self._mic_state_snapshot()
        config = getattr(self, "config", None)
        if config is not None:
            state["assistant_name"] = config.assistant_name
            state["assistant_avatar"] = config.assistant_avatar
        state["delivery_closed"] = safety["delivery_closed"]
        state["mic_transition_pending"] = safety["mic_transition_pending"]
        return state

    def _set_activity_status(self, **fields: Any) -> None:
        if not getattr(self, "paths", None):
            return
        lock = getattr(self, "_activity_state_lock", None)
        if lock is None:
            self._activity_state_lock = threading.Lock()
            lock = self._activity_state_lock
        try:
            ensure_dirs(self.paths)
            safety = self._mic_state_snapshot()
            with lock:
                state = dict(getattr(self, "_activity_status", {}))
                state.update(fields)
                config = getattr(self, "config", None)
                if config is not None:
                    state["assistant_name"] = config.assistant_name
                    state["assistant_avatar"] = config.assistant_avatar
                # Always publish daemon-owned safety fields so the viewer can
                # tell a completed mute from closure still in progress.
                state["delivery_closed"] = safety["delivery_closed"]
                state["mic_transition_pending"] = safety["mic_transition_pending"]
                state["updated_at"] = (
                    datetime.now().astimezone().isoformat(timespec="milliseconds")
                )
                state["revision"] = int(state.get("revision") or 0) + 1
                self._activity_status = state
                target = self.paths.activity_state
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                temporary.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary.chmod(0o600)
                os.replace(temporary, target)
        except OSError:
            log.debug("could not write live activity state", exc_info=True)

    def _commit_voice_action(self, action: dict[str, Any]) -> None:
        self.last_voice_action = action
        try:
            self._record_activity("action", action=action)
        except Exception:
            log.exception("voice action journal failed")
        log.info("voice action: %s", action)
        kind = str(action.get("action_kind") or action.get("kind") or "unknown")
        result = action.get("result") if isinstance(action.get("result"), dict) else {}
        result_text = (
            result.get("summary")
            or result.get("message")
            or action.get("message")
            or kind.replace("_", " ")
        )
        target = action.get("target")
        common = {
            "chosen_action": action.get("chosen_action") or kind,
            "chosen_target": target,
            "chosen_mode": action.get("mode"),
            "delivery_status": (
                "sent"
                if result.get("sent") is True
                else "not_sent"
                if result.get("sent") is False
                else "unknown"
                if "sent" in result
                else None
            ),
        }
        if getattr(self, "_delivery_closed", False):
            listener = self.voice
            self._current_input_mode = "idle"
            self._set_activity_status(
                phase="shutting_down",
                mode="closed",
                capture_active=bool(
                    listener and getattr(listener, "capture_eligible", False)
                ),
                speech_active=False,
                waiting_for="process exit",
                live_transcript="",
                dictation_buffer="",
                last_result=str(result_text),
                **common,
            )
        elif kind in {"dictation_start", "dictation_append"}:
            self._current_input_mode = "dictation"
            self._set_activity_status(
                phase="awaiting_end_phrase",
                mode="dictation",
                capture_active=True,
                speech_active=False,
                waiting_for="more speech or an explicit finish request",
                live_transcript=str(action.get("transcript") or ""),
                dictation_buffer=str(action.get("text") or ""),
                last_result="",
                **common,
            )
        elif kind == "clarification":
            pending = getattr(self, "pending_clarification", None) or {}
            resuming_dictation = bool(
                self.dictation.active
                or pending.get("resume_phase") == "dictation_capture"
            )
            self._set_activity_status(
                phase="awaiting_clarification",
                mode="dictation" if resuming_dictation else "clarification",
                capture_active=bool(pending) or self.dictation.active,
                speech_active=False,
                waiting_for="clarification or retry; buffered dictation is retained",
                live_transcript="",
                dictation_buffer=(
                    self.dictation.joined()
                    if self.dictation.active
                    else str(pending.get("request_text") or "")
                ),
                last_result=str(result_text),
                **common,
            )
        elif kind in {"error", "no_action", "verification_blocked"}:
            pending = getattr(self, "pending_clarification", None) or {}
            retaining_dictation = self.dictation.active
            retaining_clarification = bool(pending)
            self._current_input_mode = (
                "dictation"
                if retaining_dictation
                else "clarification"
                if retaining_clarification
                else "idle"
            )
            try:
                self._set_activity_status(
                    phase=(
                        "awaiting_clarification"
                        if retaining_clarification
                        else "error"
                        if kind == "error"
                        else "blocked"
                    ),
                    mode=self._current_input_mode,
                    capture_active=retaining_dictation or retaining_clarification,
                    speech_active=False,
                    waiting_for=(
                        "clarification or retry; pending request is retained"
                        if retaining_clarification
                        else "retry or clarify; buffered dictation is retained"
                        if retaining_dictation
                        else "wake word"
                    ),
                    live_transcript="",
                    dictation_buffer=(
                        self.dictation.joined()
                        if retaining_dictation
                        else str(pending.get("request_text") or "")
                    ),
                    last_result=str(result_text),
                    **common,
                )
            except Exception:
                log.exception("voice activity status publication failed")
        elif kind == "control" and result.get("pending"):
            listener = self.voice
            self._current_input_mode = "idle"
            self._set_activity_status(
                phase="muting",
                mode="idle",
                capture_active=bool(
                    listener and getattr(listener, "capture_eligible", False)
                ),
                speech_active=False,
                waiting_for="microphone listener closure",
                live_transcript="",
                dictation_buffer="",
                last_result=str(result_text),
                **common,
            )
        elif kind == "control" and result.get("ok") and action.get("mode") == "mute":
            self._current_input_mode = "idle"
            self._set_activity_status(
                phase="muted",
                mode="idle",
                capture_active=False,
                speech_active=False,
                waiting_for="listen command",
                live_transcript="",
                dictation_buffer="",
                last_result=str(result_text),
                **common,
            )
        else:
            pending = getattr(self, "pending_clarification", None) or {}
            self._current_input_mode = "clarification" if pending else "idle"
            try:
                self._set_activity_status(
                    phase="awaiting_clarification" if pending else "ready",
                    mode=self._current_input_mode,
                    capture_active=bool(pending),
                    speech_active=False,
                    waiting_for=(
                        "clarification or retry; pending request is retained"
                        if pending
                        else "wake word"
                    ),
                    live_transcript="",
                    dictation_buffer=str(pending.get("request_text") or ""),
                    last_result=str(result_text),
                    **common,
                )
            except Exception:
                log.exception("voice activity status publication failed")

    def _load_activity_workspace_state(self) -> dict[str, str] | None:
        try:
            data = json.loads(self.paths.activity_workspace_file.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        workspace_id = data.get("workspace_id")
        pane_id = data.get("pane_id")
        if not workspace_id or not pane_id:
            return None
        return {
            "workspace_id": str(workspace_id),
            "pane_id": str(pane_id),
            "ui_version": str(data.get("ui_version") or "1"),
        }

    def _write_activity_workspace_state(self, state: dict[str, str]) -> None:
        self._atomic_write_mic_state(self.paths.activity_workspace_file, state)

    @staticmethod
    def _activity_workspace_identity(value: Any) -> dict[str, str] | None:
        if not isinstance(value, dict):
            return None
        workspace_id = value.get("workspace_id")
        pane_id = value.get("pane_id")
        if not workspace_id or not pane_id:
            return None
        return {
            "workspace_id": str(workspace_id),
            "pane_id": str(pane_id),
            "ui_version": str(value.get("ui_version") or "1"),
        }

    def _load_activity_workspace_transaction(
        self,
    ) -> dict[str, dict[str, str]] | None:
        try:
            raw = json.loads(
                self.paths.activity_workspace_transaction.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OSError(f"activity replacement transaction is unreadable: {exc}")
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise OSError("activity replacement transaction has an invalid version")
        old = self._activity_workspace_identity(raw.get("old_workspace"))
        new = self._activity_workspace_identity(raw.get("new_workspace"))
        if old is None or new is None:
            raise OSError("activity replacement transaction lacks workspace identities")
        return {"old_workspace": old, "new_workspace": new}

    def _write_activity_workspace_transaction(
        self, old: dict[str, str], new: dict[str, str]
    ) -> None:
        self._atomic_write_mic_state(
            self.paths.activity_workspace_transaction,
            {"version": 1, "old_workspace": old, "new_workspace": new},
        )

    def _clear_activity_workspace_transaction(self) -> None:
        target = self.paths.activity_workspace_transaction
        try:
            target.unlink()
        except FileNotFoundError:
            return
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _recover_activity_workspace_transaction(
        self,
        state: dict[str, str] | None,
        live_ids: set[str],
        workspaces: list[dict[str, Any]],
    ) -> bool:
        """Finish cleanup from an interrupted old/new viewer replacement."""
        try:
            transaction = self._load_activity_workspace_transaction()
        except OSError as exc:
            self._record_activity(
                "workspace_error",
                error=str(exc),
                message="Activity workspace recovery metadata is unavailable.",
            )
            log.warning("activity workspace recovery failed: %s", exc)
            return False
        if transaction is None:
            return True

        old = transaction["old_workspace"]
        new = transaction["new_workspace"]
        committed = bool(
            state
            and state["workspace_id"] == new["workspace_id"]
            and state["pane_id"] == new["pane_id"]
        )
        orphan = old if committed else new
        orphan_id = orphan["workspace_id"]
        if orphan_id in live_ids:
            try:
                # A saved ID can belong to another workspace after a Herdr
                # restart. Recovery needs the same ownership check as normal
                # replacement, plus the recorded pane identity.
                owned = self._activity_workspace_owned(orphan, workspaces)
                panes = self.herdr.pane_list(orphan_id) if owned else []
                owned = owned and any(
                    str(row.get("pane_id")) == orphan["pane_id"] for row in panes
                )
                if owned:
                    self.herdr.workspace_close(orphan_id)
            except (HerdrError, OSError) as exc:
                self._record_activity(
                    "workspace_error",
                    **orphan,
                    error=str(exc),
                    message="Interrupted activity workspace cleanup will retry.",
                )
                log.warning("activity workspace recovery cleanup failed: %s", exc)
                return False
            if owned:
                live_ids.discard(orphan_id)
            else:
                self._record_activity(
                    "workspace_stale",
                    **orphan,
                    message="Discarding stale cleanup metadata; workspace ownership changed.",
                )
        try:
            self._clear_activity_workspace_transaction()
        except OSError as exc:
            self._record_activity(
                "workspace_error",
                **orphan,
                error=str(exc),
                message="Activity workspace cleanup record could not be cleared.",
            )
            log.warning("activity workspace recovery commit failed: %s", exc)
            return False
        return True

    def _activity_workspace_label(self) -> str:
        return self.config.activity_workspace_label.strip() or "voicerdr activity"

    @staticmethod
    def _workspace_base_label(label: str) -> str:
        text = (label or "").strip()
        if text.startswith("#"):
            parts = text.split(None, 1)
            return parts[1].strip() if len(parts) > 1 else ""
        return text

    def _activity_workspace_owned(
        self, state: dict[str, str], workspaces: list[dict[str, Any]]
    ) -> bool:
        expected = self._activity_workspace_label().casefold()
        target = state.get("workspace_id")
        for row in workspaces:
            if str(row.get("workspace_id")) != target:
                continue
            base = self._workspace_base_label(str(row.get("label") or ""))
            return base.casefold() == expected
        return False

    def _activity_pane_alive(self, state: dict[str, str]) -> bool:
        try:
            panes = self.herdr.pane_list(state["workspace_id"])
        except (HerdrError, OSError) as exc:
            log.warning("activity pane probe failed: %s", exc)
            return False
        pane_id = state.get("pane_id")
        return any(str(row.get("pane_id")) == pane_id for row in panes)

    def _ensure_activity_workspace(self) -> None:
        """Create or reuse the optional, plugin-owned activity workspace."""
        ensure_dirs(self.paths)
        state = self._load_activity_workspace_state()
        created: dict[str, str] | None = None
        created_committed = False
        previous: dict[str, str] | None = None
        try:
            workspaces = self.herdr.workspace_list()
            live_ids = {str(row.get("workspace_id")) for row in workspaces}
            if not self._recover_activity_workspace_transaction(
                state, live_ids, workspaces
            ):
                self.activity_workspace = (
                    state if state and state["workspace_id"] in live_ids else None
                )
                return
            if not self.config.activity_workspace_enabled:
                if (
                    state
                    and state["workspace_id"] in live_ids
                    and self._activity_workspace_owned(state, workspaces)
                ):
                    self.herdr.workspace_close(state["workspace_id"])
                try:
                    self.paths.activity_workspace_file.unlink()
                except FileNotFoundError:
                    pass
                self.activity_workspace = None
                return

            owned = bool(state) and self._activity_workspace_owned(state, workspaces)
            reusable = (
                owned
                and state is not None
                and state.get("ui_version") == ACTIVITY_UI_VERSION
                and self._activity_pane_alive(state)
            )
            if reusable:
                assert state is not None
                self.activity_workspace = state
                self._record_activity(
                    "workspace_reused",
                    **state,
                    message="Reusing the voicerdr activity workspace.",
                )
                return

            # Only close a prior workspace we can still prove we own. Stale
            # pointers at recycled Herdr IDs must not destroy foreign spaces.
            if owned and state is not None and state["workspace_id"] in live_ids:
                previous = state
            elif state is not None:
                self._record_activity(
                    "workspace_stale",
                    **state,
                    message="Ignoring stale activity workspace pointer; creating a new viewer.",
                )

            created = self.herdr.workspace_create(
                cwd=str(self.paths.plugin_root),
                label=self._activity_workspace_label(),
            )
            command = self.herdr.activity_command(
                sys.executable,
                str(self.paths.activity_log),
                str(self.paths.activity_state),
                self.config.activity_history_lines,
            )
            created["ui_version"] = ACTIVITY_UI_VERSION
            if state is not None:
                self._write_activity_workspace_transaction(state, created)
            self.herdr.pane_rename(created["pane_id"], "voicerdr activity")
            self.herdr.pane_run(created["pane_id"], command)
            self._write_activity_workspace_state(created)
            created_committed = True
            self.activity_workspace = created
            self._record_activity(
                "workspace_created",
                **created,
                message="Activity workspace created; this pane follows activity.jsonl.",
            )
            if previous and previous["workspace_id"] != created["workspace_id"]:
                try:
                    self.herdr.workspace_close(previous["workspace_id"])
                except (HerdrError, OSError) as exc:
                    # The replacement is running and durably tracked. Keep it
                    # even when cleanup leaves the obsolete viewer visible.
                    self._record_activity(
                        "workspace_error",
                        **created,
                        message="Replacement started, but the old viewer stayed open.",
                        error=str(exc),
                    )
                    log.warning("old activity workspace cleanup failed: %s", exc)
                else:
                    self._clear_activity_workspace_transaction()
            elif state is not None:
                self._clear_activity_workspace_transaction()
        except (HerdrError, OSError) as exc:
            if created and not created_committed:
                try:
                    self.herdr.workspace_close(created["workspace_id"])
                except (HerdrError, OSError):
                    log.debug("could not clean up failed activity workspace")
                else:
                    try:
                        self._clear_activity_workspace_transaction()
                    except OSError:
                        log.warning(
                            "failed activity workspace cleanup record was retained",
                            exc_info=True,
                        )
            if created_committed:
                self.activity_workspace = created
                self._record_activity(
                    "workspace_error",
                    **created,
                    error=str(exc),
                    message="Replacement is active; cleanup will retry on startup.",
                )
                log.warning("activity workspace cleanup commit failed: %s", exc)
                return
            # Atomic metadata replacement preserves the previous record when
            # setup or persistence of its replacement fails, making retry safe.
            self.activity_workspace = previous
            self._record_activity(
                "workspace_error",
                error=str(exc),
                message="Could not create the optional activity workspace.",
            )
            log.warning("activity workspace setup failed: %s", exc)

    def resolve_only(self, space: str, agent: str | None = None) -> dict[str, Any]:
        workspaces = self.herdr.workspace_list()
        agents = self.herdr.agent_list()
        route = resolve_route(
            space,
            workspaces=workspaces,
            agents=agents,
            aliases=self.config.aliases,
            agent_query=agent,
        )
        message = route.message
        if not route.ok:
            message = self._route_failure_message(route, space=space, agent=agent)
        return {
            "ok": route.ok,
            "workspace_id": route.workspace_id,
            "workspace_label": route.workspace_label,
            "target": route.target,
            "pane_id": route.pane_id,
            "agent_status": route.agent_status,
            "terminal_title": route.terminal_title,
            "message": message,
            "candidates": route.candidates,
        }

    def route_and_prompt(
        self,
        space: str,
        text: str,
        *,
        agent: str | None = None,
        notify: bool = True,
    ) -> dict[str, Any]:
        """Rejected stub: prompts must go through ingest_transcript."""
        _ = notify
        message = (
            "Direct prompt routing is disabled. Submit a wake-addressed transcript "
            "through ingest_transcript so the planner and verifier can authorize it."
        )
        self._record_activity(
            "withheld",
            code="verified_planner_required",
            space=space,
            agent=agent,
            message=text,
            reason=message,
            sent=False,
        )
        return {
            "ok": False,
            "sent": False,
            "code": "verified_planner_required",
            "message": message,
        }

    def _route_failure_message(
        self,
        route: ResolveResult,
        *,
        space: str,
        agent: str | None,
    ) -> str:
        candidates = [str(item) for item in route.candidates or [] if item]
        choices = ", ".join(candidates[:4])
        if route.workspace_id and candidates:
            return self.policy.clamp_speech(
                f"Which agent in {route.workspace_label or space}? "
                f"Say one of: {choices}."
            )
        if route.workspace_id:
            return self.policy.clamp_speech(
                f"{route.workspace_label or space} has no active agent. "
                "Start or select an agent, then try again."
            )
        if agent:
            suffix = f" Available agents: {choices}." if choices else ""
            return self.policy.clamp_speech(
                f"I could not find agent or title {agent} in {space}.{suffix}"
            )
        if candidates:
            return self.policy.clamp_speech(
                f"I could not identify {space}. Say one of: {choices}."
            )
        return self.policy.clamp_speech(
            route.message or f"I could not find an available agent in {space}."
        )

    def summarize_agent(self, target: str, *, clamp: bool = True) -> str:
        """Inspect recent target output for an explicit status/review request."""
        title = target
        try:
            agent = self.herdr.agent_get(target)
            pane_id = str(agent.get("pane_id") or target)
            title = str(
                agent.get("terminal_title_stripped") or agent_name(agent) or target
            )
            status = str(agent.get("agent_status") or "unknown")
            workspace = self._workspace_label(
                str(agent.get("workspace_id") or "") or None
            )

            # Alternate-screen history cannot be read while an agent is active.
            # The visible source is passive and still captures its current step.
            source = (
                "recent-unwrapped"
                if status.casefold() in {"idle", "done"}
                else "visible"
            )
            excerpt = self.herdr.agent_read(pane_id, source=source, lines=40)
            clean = _clean_progress_excerpt(excerpt)
            if not clean:
                raise SecretaryError("pane has no readable recent output")
            summary = self.secretary.summarize_progress(
                workspace=workspace,
                title=title,
                status=status,
                excerpt=clean,
            )
            summary = " ".join(summary.split()).strip()
            return self.policy.clamp_speech(summary) if clamp else summary
        except (HerdrError, SecretaryError) as exc:
            log.info("detailed agent summary unavailable for %s: %s", target, exc)
            raise SecretaryError(
                f"could not inspect recent output for {title}: {exc}"
            ) from exc

    @staticmethod
    def _status_group_members(
        workspaces: list[dict[str, Any]], workspace_id: str
    ) -> list[dict[str, Any]]:
        """Parent first, then linked worktrees for the same repo; else just the row."""
        target = next(
            (
                row
                for row in workspaces
                if str(row.get("workspace_id") or "") == workspace_id
            ),
            None,
        )
        if target is None:
            return []
        worktree = target.get("worktree") or {}
        repo_key = str(worktree.get("repo_key") or "")
        if not repo_key or bool(worktree.get("is_linked_worktree")):
            return [target]

        def _number(row: dict[str, Any]) -> tuple[int, str]:
            try:
                return (int(row.get("number")), str(row.get("workspace_id") or ""))
            except (TypeError, ValueError):
                return (10**9, str(row.get("workspace_id") or ""))

        linked = [
            row
            for row in workspaces
            if str(row.get("workspace_id") or "") != workspace_id
            and str((row.get("worktree") or {}).get("repo_key") or "") == repo_key
            and bool((row.get("worktree") or {}).get("is_linked_worktree"))
        ]
        linked.sort(key=_number)
        return [target, *linked]

    def summarize_workspace_group(self, workspace_id: str) -> list[str]:
        """Summarize parent agents, then each linked worktree's agents."""
        try:
            workspaces = self.herdr.workspace_list()
            agents = self.herdr.agent_list()
        except HerdrError as exc:
            raise SecretaryError(
                f"could not list workspaces for status: {exc}"
            ) from exc

        members = self._status_group_members(workspaces, workspace_id)
        if not members:
            raise SecretaryError(f"workspace {workspace_id} is not in the live catalog")

        label_counts = len(members)
        parts: list[str] = []
        for row in members:
            wid = str(row.get("workspace_id") or "")
            label = strip_number_prefix(str(row.get("label") or wid)) or wid
            ws_agents = [
                agent
                for agent in agents
                if str(agent.get("workspace_id") or "") == wid
                and str(agent.get("pane_id") or "")
            ]
            if not ws_agents:
                parts.append(self.policy.clamp_speech(f"{label}: no agents."))
                continue
            multi = label_counts > 1 or len(ws_agents) > 1
            for agent in ws_agents:
                pane_id = str(agent.get("pane_id") or "")
                title = str(
                    agent.get("terminal_title_stripped") or agent_name(agent) or pane_id
                )
                summary = (
                    self.summarize_agent(pane_id, clamp=False)
                    if multi
                    else self.summarize_agent(pane_id)
                )
                if multi:
                    parts.append(
                        self.policy.clamp_speech(f"{label} ({title}): {summary}")
                    )
                else:
                    parts.append(self.policy.clamp_speech(summary))
        return parts

    def _speak_status_summaries(self, parts: list[str]) -> str:
        """Speak each part in order; return the joined activity/log summary."""
        spoken = [self.policy.clamp_speech(part) for part in parts if part.strip()]
        if not spoken:
            spoken = ["No agents found in that space."]
        summary = self.policy.clamp_speech(" ".join(spoken))
        try:
            self.herdr.notification_show("voicerdr", summary, sound="done")
        except Exception as exc:  # noqa: BLE001
            log.debug("notification failed: %s", exc)
        for part in spoken:
            try:
                self._speak(part, replace=False)
            except Exception:  # TTS acknowledgement must not alter action truth.
                log.exception("speech acknowledgement failed")
                break
        return summary

    def _summarize_agent_metadata(
        self,
        agent: dict[str, Any],
        *,
        target: str,
        status: str,
        workspace_label: str | None = None,
    ) -> str:
        """Cheap metadata-only summary for fallbacks and lifecycle events."""
        title = str(agent.get("terminal_title_stripped") or agent_name(agent) or target)
        subject = title if self.config.prefer_titles else workspace_label or title
        return self.policy.clamp_speech(f"{subject} is {status}.")

    def fleet_status(self) -> dict[str, Any]:
        workspaces = self.herdr.workspace_list()
        agents = self.herdr.agent_list()
        labels = {
            str(ws.get("workspace_id")): strip_number_prefix(
                str(ws.get("label") or ws.get("workspace_id") or "unknown")
            )
            for ws in workspaces
        }
        blocked: list[dict[str, str]] = []
        for agent in agents:
            if str(agent.get("agent_status") or "").casefold() != "blocked":
                continue
            workspace_id = str(agent.get("workspace_id") or "")
            blocked.append(
                {
                    "workspace_id": workspace_id,
                    "workspace_label": labels.get(workspace_id, "unknown workspace"),
                    "target": str(agent.get("name") or agent.get("pane_id") or ""),
                    "title": str(
                        agent.get("terminal_title_stripped")
                        or agent_name(agent)
                        or "blocked agent"
                    ),
                }
            )
        if not blocked:
            summary = "No agents are blocked."
        else:
            descriptions = [
                f"{item['workspace_label']}: {item['title']}" for item in blocked[:4]
            ]
            extra = len(blocked) - len(descriptions)
            suffix = f", plus {extra} more" if extra else ""
            summary = (
                f"{len(blocked)} blocked agent"
                f"{'s' if len(blocked) != 1 else ''}: "
                f"{'; '.join(descriptions)}{suffix}."
            )
        return {
            "ok": True,
            "blocked_count": len(blocked),
            "blocked": blocked,
            "summary": self.policy.clamp_speech(summary),
        }

    def set_talk_policy(
        self,
        *,
        mode: str | None = None,
        space: str | None = None,
        quiet: bool | None = None,
    ) -> dict[str, Any]:
        if mode:
            self.policy.set_talk_mode(mode)
        workspace_label: str | None = None
        if space is not None and quiet is not None:
            workspaces = self.herdr.workspace_list()
            resolved = resolve_workspace(space, workspaces, self.config.aliases)
            if not resolved.ok or not resolved.workspace_id:
                message = self._route_failure_message(resolved, space=space, agent=None)
                return {"ok": False, "message": message}
            workspace_label = resolved.workspace_label or space
            self.policy.set_workspace_quiet(resolved.workspace_id, quiet=quiet)
            self.policy.set_workspace_quiet(workspace_label, quiet=quiet)
        snapshot = self.policy.snapshot()
        if space is not None and quiet is not None:
            summary = (
                f"Announcements for {workspace_label or space} are "
                f"{'quiet' if quiet else 'enabled'}."
            )
        else:
            summary = f"Talk mode is {str(snapshot['mode']).replace('_', ' ')}."
        return {"ok": True, "summary": summary, **snapshot}

    def _write_session(self) -> None:
        # Preserve spawn_pid/pgid from ensure so quit can kill the uv parent too.
        prev = {}
        if self.paths.session_file.is_file():
            try:
                prev = json.loads(self.paths.session_file.read_text())
            except (OSError, json.JSONDecodeError):
                prev = {}
        payload = {
            "pid": os.getpid(),
            "spawn_pid": prev.get("spawn_pid"),
            "pgid": prev.get("pgid") or os.getpgid(0),
            "herdr_socket": self.paths.herdr_socket,
            "started_at": self._started_at,
        }
        self.paths.session_file.write_text(json.dumps(payload, indent=2) + "\n")
        self.paths.pidfile.write_text(str(os.getpid()) + "\n")
        for p in (self.paths.session_file, self.paths.pidfile):
            try:
                p.chmod(0o600)
            except OSError:
                pass

    def _install_signals(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            log.info("signal %s — shutting down", signum)
            self._close_delivery_authority()
            if self._server:
                try:
                    self._server.close()
                except OSError:
                    pass

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def _acquire_daemon_lock(self) -> None:
        """Exclusive flock on the state directory before startup mutations."""
        # Contenders that already exist must not chmod, seed, bind, or touch ledgers.
        self._bootstrap_authority_dir_durable()
        try:
            fd = os.open(
                self.paths.daemon_lock,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            created = True
        except FileExistsError:
            fd = os.open(self.paths.daemon_lock, os.O_RDWR)
            created = False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise RuntimeError("another voicerdr daemon owns this state directory")
        try:
            self._ensure_state_dir_durable()
            if created:
                os.fsync(fd)
                directory_fd = os.open(
                    self.paths.state_dir, os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except Exception:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            raise
        self._daemon_lock_fd = fd

    def _bootstrap_authority_dir_durable(self) -> None:
        """Create only a missing lock directory chain and sync each new entry."""

        state_dir = self.paths.state_dir
        try:
            missing: list[Any] = []
            cursor = state_dir
            while not cursor.exists():
                missing.append(cursor)
                if cursor.parent == cursor:
                    break
                cursor = cursor.parent
            for directory in reversed(missing):
                try:
                    directory.mkdir()
                except FileExistsError:
                    continue
                parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
        except OSError as exc:
            raise RuntimeError(
                f"daemon authority directory is not durable: {exc}"
            ) from exc

    def _release_daemon_lock(self) -> None:
        fd = getattr(self, "_daemon_lock_fd", None)
        if fd is None:
            return
        self._daemon_lock_fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @staticmethod
    def _socket_is_live(path: os.PathLike[str]) -> bool:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.25)
        try:
            probe.connect(os.fspath(path))
            return True
        except (ConnectionRefusedError, FileNotFoundError):
            return False
        except OSError:
            # Unknown socket conditions are not proof of staleness.
            return True
        finally:
            probe.close()

    def _bind_control(self) -> None:
        path = self.paths.control_socket
        stale_fd: int | None = None
        try:
            before = os.lstat(path)
        except FileNotFoundError:
            before = None
        if before is not None:
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISSOCK(before.st_mode):
                raise RuntimeError(
                    "control path is not an owned socket; refusing to unlink"
                )
            try:
                stale_fd = os.open(path, os.O_PATH | os.O_NOFOLLOW)
                held = os.fstat(stale_fd)
                expected = (before.st_dev, before.st_ino)
                if (held.st_dev, held.st_ino) != expected:
                    raise RuntimeError("control socket changed during stale check")
                if self._socket_is_live(path):
                    raise RuntimeError(
                        "control socket is already live; refusing to unlink"
                    )
                after = os.lstat(path)
                if (
                    not stat.S_ISSOCK(after.st_mode)
                    or (
                        after.st_dev,
                        after.st_ino,
                    )
                    != expected
                ):
                    raise RuntimeError("control socket changed during stale check")
                self._quarantine_socket_path(
                    path, expected, purpose="stale control socket"
                )
            except FileNotFoundError as exc:
                raise RuntimeError("control socket changed during stale check") from exc
            finally:
                if stale_fd is not None:
                    os.close(stale_fd)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        path_fd: int | None = None
        owned: os.stat_result | None = None
        try:
            server.bind(str(path))
            try:
                path.chmod(0o600)
            except OSError:
                pass
            path_fd = os.open(path, os.O_PATH | os.O_NOFOLLOW)
            owned = os.fstat(path_fd)
            if not stat.S_ISSOCK(owned.st_mode):
                raise RuntimeError("bound control path is not a socket")
            server.listen(8)
            server.settimeout(1.0)
        except Exception:
            server.close()
            try:
                current = os.lstat(path)
                if owned is not None and (
                    current.st_dev,
                    current.st_ino,
                ) == (owned.st_dev, owned.st_ino):
                    self._quarantine_socket_path(
                        path,
                        (owned.st_dev, owned.st_ino),
                        purpose="failed control socket bind",
                    )
            except (FileNotFoundError, OSError, RuntimeError):
                log.debug("could not clean failed control socket", exc_info=True)
            if path_fd is not None:
                os.close(path_fd)
            raise
        assert owned is not None and path_fd is not None
        self._control_socket_identity = (owned.st_dev, owned.st_ino)
        self._control_socket_path_fd = path_fd
        self._server = server

    def _unlink_owned_control_socket(self) -> None:
        identity = getattr(self, "_control_socket_identity", None)
        if identity is None:
            return
        path = self.paths.control_socket
        try:
            current = os.lstat(path)
            if (
                stat.S_ISSOCK(current.st_mode)
                and (
                    current.st_dev,
                    current.st_ino,
                )
                == identity
            ):
                self._quarantine_socket_path(
                    path, identity, purpose="owned control socket"
                )
        except FileNotFoundError:
            pass
        except (OSError, RuntimeError):
            log.debug("could not remove owned control socket", exc_info=True)
        finally:
            self._control_socket_identity = None
            path_fd = getattr(self, "_control_socket_path_fd", None)
            self._control_socket_path_fd = None
            if path_fd is not None:
                try:
                    os.close(path_fd)
                except OSError:
                    log.debug("could not close control socket identity", exc_info=True)

    @staticmethod
    def _quarantine_socket_path(
        path: os.PathLike[str],
        expected: tuple[int, int],
        *,
        purpose: str,
    ) -> None:
        """Atomically move, validate, then remove one exact socket inode.

        If the pathname was replaced between validation and rename, the moved
        replacement is restored and never unlinked.
        """

        original = os.fspath(path)
        quarantine = f"{original}.quarantine.{os.getpid()}.{uuid.uuid4().hex}"
        os.rename(original, quarantine)
        moved = os.lstat(quarantine)
        moved_identity = (moved.st_dev, moved.st_ino)
        if not stat.S_ISSOCK(moved.st_mode) or moved_identity != expected:
            try:
                # Hard-link restoration is an atomic no-replace operation. A
                # fresh owner that appeared at ``original`` is never clobbered.
                os.link(quarantine, original, follow_symlinks=False)
            except FileExistsError:
                pass
            else:
                os.unlink(quarantine)
            raise RuntimeError(f"{purpose} changed during quarantine")
        os.unlink(quarantine)

    def _start_events(self) -> None:
        if not self.paths.herdr_socket:
            log.warning("no HERDR_SOCKET_PATH — Herdr watchers disabled")
            return

        def on_status(payload: dict[str, Any]) -> None:
            self.last_event = payload
            pane_id = str(payload.get("pane_id") or "")
            status = str(payload.get("agent_status") or "")
            if not pane_id or not status:
                return
            agent = payload.get("agent")
            self._maybe_announce(
                pane_id,
                status,
                agent=agent if isinstance(agent, dict) else None,
            )

        def on_socket_event(msg: dict[str, Any]) -> None:
            self.last_event = msg
            event = msg.get("event") or msg.get("result") or msg
            if not isinstance(event, dict):
                return
            etype = str(event.get("type") or event.get("kind") or "")
            data = event.get("data") if isinstance(event.get("data"), dict) else event
            if "workspace.focused" in etype or data.get("type") == "workspace.focused":
                ws = data.get("workspace_id") or (data.get("workspace") or {}).get(
                    "workspace_id"
                )
                if ws:
                    self.focused_workspace_id = str(ws)
            if any(
                x in etype
                for x in (
                    "workspace.created",
                    "workspace.closed",
                    "workspace.renamed",
                    "workspace.reordered",
                    "workspace.moved",
                )
            ):
                self._sync_space_labels()
            if self.status_watcher and (
                "pane.created" in etype or "pane.closed" in etype
            ):
                self.status_watcher.poll_once()

        self.status_watcher = AgentStatusWatcher(
            self.herdr,
            on_transition=on_status,
            interval_secs=2.0,
        )
        self.status_watcher.start()

        self.subscriber = EventSubscriber(
            self.herdr,
            on_event=on_socket_event,
            retry_secs=self.config.subscribe_retry_secs,
        )
        self.subscriber.start()

    def _maybe_announce(
        self,
        pane_id: str,
        status: str,
        *,
        agent: dict[str, Any] | None = None,
    ) -> None:
        if agent is None:
            try:
                agent = self.herdr.agent_get(pane_id)
            except HerdrError:
                agent = None
        workspace_id = str((agent or {}).get("workspace_id") or "") or None
        workspace_label = self._workspace_label(workspace_id)
        allow, reason = self.policy.should_announce(
            pane_id=pane_id,
            status=status,
            workspace_id=workspace_id,
            workspace_label=workspace_label,
        )
        if not allow:
            log.debug("skip announce pane=%s status=%s (%s)", pane_id, status, reason)
            return
        if agent is not None:
            summary = self._summarize_agent_metadata(
                agent,
                target=pane_id,
                status=status,
                workspace_label=workspace_label,
            )
        else:
            summary = self.policy.clamp_speech(f"Agent on {pane_id} is {status}")
        self.policy.mark_spoken(pane_id, status)
        self.last_summary = summary
        self._record_activity(
            "announcement",
            pane_id=pane_id,
            agent_status=status,
            workspace_id=workspace_id,
            workspace_label=workspace_label,
            message=summary,
        )
        log.info("announce: %s", summary)
        self._notify(
            "voicerdr",
            summary,
            sound="request" if status == "blocked" else "done",
            speak=True,
            urgent=status == "blocked",
        )

    def _workspace_label(self, workspace_id: str | None) -> str | None:
        if not workspace_id:
            return None
        try:
            for workspace in self.herdr.workspace_list():
                if str(workspace.get("workspace_id")) == workspace_id:
                    label = workspace.get("label")
                    return strip_number_prefix(str(label)) if label else None
        except HerdrError:
            pass
        return None

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            threading.Thread(
                target=self._handle_conn,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(5.0)
            buf = b""
            try:
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                line = buf.split(b"\n", 1)[0].decode()
                req = ControlRequest.parse(line)
                response = self._dispatch(req)
                conn.sendall(response.encode())
            except Exception as exc:  # noqa: BLE001
                try:
                    conn.sendall(err("0", str(exc)).encode())
                except OSError:
                    pass

    def _show_notification_async(self, title: str, body: str, *, sound: str) -> None:
        """Keep best-effort Herdr UI work off the control response path."""

        def show() -> None:
            try:
                self.herdr.notification_show(title, body, sound=sound)
            except HerdrError:
                pass

        try:
            threading.Thread(
                target=show,
                name="voicerdr-control-notification",
                daemon=True,
            ).start()
        except RuntimeError:
            log.debug("could not start control notification worker", exc_info=True)

    def _dispatch(self, req: ControlRequest) -> str:
        if req.method not in {"ping", "status"}:
            self._record_activity(
                "control_request",
                method=req.method,
                params=req.params,
            )
        if req.method == "ping":
            return ok(
                req.id,
                {
                    "pong": True,
                    "pid": os.getpid(),
                    "mode": self.policy.mic_mode,
                    "herdr_socket": self.paths.herdr_socket,
                    "subscribed": bool(self.subscriber and self.subscriber.connected),
                },
            )
        if req.method == "status":
            st = self.status()
            # Herdr action menu has no live indicator — toast makes status visible.
            mode = st.get("mode")
            voice = "on" if st.get("voice_running") else "off"
            line = f"mic={mode} voice={voice}"
            if st.get("voice_error"):
                line += f" err={st['voice_error']}"
            last = st.get("last_transcript")
            if last:
                line += f" | heard: {str(last)[:140]}"
            action = st.get("last_voice_action") or {}
            if isinstance(action, dict) and action.get("kind"):
                line += f" | last={action.get('kind')}"
                if action.get("message"):
                    line += f"({action['message']})"
            if req.params.get("notify", True):
                self._show_notification_async("voicerdr status", line, sound="none")
            return ok(req.id, st)
        if req.method == "aliases":
            self._reload_aliases()
            directory = self._space_directory()
            try:
                bits = [
                    f"#{s['number']} {s.get('base') or s['label']}"
                    + (
                        f" [{', '.join(s['nicknames'][:3])}]"
                        if s.get("nicknames")
                        else ""
                    )
                    for s in directory
                ]
                body = " · ".join(bits) if bits else "no spaces"
                self.herdr.notification_show(
                    "voicerdr spaces", body[:280], sound="none"
                )
            except HerdrError:
                pass
            return ok(
                req.id,
                {
                    "assistant_name": self.config.assistant_name,
                    "aliases": dict(sorted(self.config.aliases.items())),
                    "spaces": directory,
                },
            )
        if req.method == "set_mode":
            mode = str(req.params.get("mode") or "")
            if mode not in {"listen", "mute"}:
                return err(req.id, f"unknown mode: {mode}", code="bad_mode")
            with self._authority_lock:
                arrived_during_transition = bool(
                    getattr(self, "_mic_transition_count", 0)
                )
            with self._mic_transition():
                with self._authority_lock:
                    if getattr(self, "_delivery_closed", False):
                        return err(
                            req.id,
                            "Daemon shutdown has begun; microphone mode is frozen.",
                            code="delivery_closed",
                        )
                    # A listen that raced a mute must not reopen capture after
                    # that mute commits. Another listen/boot startup is idempotent
                    # once the prior transition finishes with mode still listen.
                    if (
                        mode == "listen"
                        and arrived_during_transition
                        and self.policy.mic_mode != "listen"
                    ):
                        return err(
                            req.id,
                            "Microphone mute is still proving listener closure.",
                            code="mic_transition_pending",
                        )
                if mode == "listen":
                    try:
                        self._persist_mic_preference(mode)
                    except MicPreferenceError as exc:
                        closed = self._mute_voice()
                        if not closed:
                            self._publish_mic_transition_error(
                                "Microphone closure could not be proven; "
                                "daemon shutdown was initiated.",
                                waiting_for="process exit",
                            )
                            return err(
                                req.id,
                                "microphone closure could not be proven; "
                                "daemon shutdown was initiated",
                                code="mic_closure",
                            )
                        self._publish_mic_transition_error(
                            f"Microphone muted; preference save failed: {exc}",
                            waiting_for="a writable preference store and listen command",
                        )
                        return err(
                            req.id,
                            f"microphone preference was not saved: {exc}",
                            code="mic_preference",
                        )
                    with self._authority_lock:
                        if getattr(self, "_delivery_closed", False):
                            return err(
                                req.id,
                                "Daemon shutdown began during listen; "
                                "listener startup was withheld.",
                                code="delivery_closed",
                            )
                        self.policy.set_mic_mode("listen")
                    ready = self._start_voice(wait_secs=45.0)
                    if not ready:
                        voice_error = (
                            self.voice.last_error if self.voice is not None else None
                        )
                        closure_error: MicClosureError | None = None
                        try:
                            self._mute_voice_durably()
                        except MicClosureError as exc:
                            closure_error = exc
                            voice_error = voice_error or str(exc)
                        except MicPreferenceError as exc:
                            voice_error = voice_error or str(exc)
                        message = voice_error or "listener did not become ready"
                        result_message = (
                            "Voice failed to start and microphone closure could "
                            f"not be proven: {message}"
                            if closure_error
                            else f"Voice failed to start; microphone is muted: {message}"
                        )
                        self._publish_mic_transition_error(
                            result_message,
                            waiting_for=(
                                "process exit" if closure_error else "listen retry"
                            ),
                        )
                        return err(
                            req.id,
                            result_message,
                            code="mic_closure" if closure_error else "voice",
                        )
                    try:
                        self.herdr.notification_show(
                            "voicerdr",
                            f"Listening. Say: {self.config.assistant_name}, ask "
                            "<alias or number> <message>",
                            sound="none",
                        )
                    except HerdrError:
                        pass
                else:
                    try:
                        self._mute_voice_durably()
                    except MicClosureError as exc:
                        self._publish_mic_transition_error(
                            f"Microphone closure could not be proven: {exc}",
                            waiting_for="process exit",
                        )
                        return err(req.id, str(exc), code="mic_closure")
                    except MicPreferenceError as exc:
                        self._publish_mic_transition_error(
                            f"Microphone muted; preference save failed: {exc}",
                            waiting_for="a writable preference store",
                            phase="muted",
                        )
                        return err(
                            req.id,
                            f"microphone was muted but the preference was not saved: {exc}",
                            code="mic_preference",
                        )
                effective_mode = self.policy.mic_mode
                voice_running = bool(self.voice and self.voice.running)
                if effective_mode == "listen" and not voice_running:
                    return err(
                        req.id,
                        "listener is not running; microphone mode was not enabled",
                        code="voice",
                    )
                self._set_activity_status(
                    phase="ready" if effective_mode == "listen" else "muted",
                    mode="idle",
                    capture_active=False,
                    speech_active=False,
                    waiting_for=(
                        "wake word" if effective_mode == "listen" else "listen command"
                    ),
                    live_transcript="",
                    dictation_buffer="",
                    last_result=(
                        "Microphone is ready."
                        if effective_mode == "listen"
                        else "Microphone muted."
                    ),
                )
                log.info(
                    "mic mode -> %s voice_running=%s err=%s",
                    effective_mode,
                    voice_running,
                    self.voice.last_error if self.voice else None,
                )
                return ok(
                    req.id,
                    {
                        "mode": effective_mode,
                        "voice_running": voice_running,
                        "voice_error": self.voice.last_error if self.voice else None,
                        "wake_phrases": self.config.wake_phrases,
                        "hint": (
                            f'Say: "{self.config.assistant_name}, ask '
                            '<alias|number> <message>"'
                            if effective_mode == "listen"
                            else None
                        ),
                    },
                )
        if req.method == "set_speak":
            enabled = bool(req.params.get("enabled", True))
            self.policy.speak_enabled = enabled
            self.speaker.enabled = enabled
            if enabled:
                self.speaker.start()
            return ok(req.id, {"speak_enabled": enabled})
        if req.method == "set_talk_policy":
            mode_raw = req.params.get("mode")
            space_raw = req.params.get("space")
            quiet_raw = req.params.get("quiet")
            mode = str(mode_raw) if mode_raw is not None else None
            space = str(space_raw) if space_raw is not None else None
            quiet = quiet_raw if isinstance(quiet_raw, bool) else None
            if not mode and not (space and quiet is not None):
                return err(
                    req.id,
                    "mode or space plus quiet required",
                    code="bad_params",
                )
            try:
                result = self.set_talk_policy(mode=mode, space=space, quiet=quiet)
            except ValueError as exc:
                return err(req.id, str(exc), code="bad_mode")
            if not result.get("ok"):
                return err(req.id, str(result.get("message")), code="route")
            return ok(req.id, result)
        if req.method == "fleet_status":
            try:
                return ok(req.id, self.fleet_status())
            except HerdrError as exc:
                return err(req.id, str(exc), code="herdr")
        if req.method == "say":
            text = str(req.params.get("text") or "")
            if not text:
                return err(req.id, "text required", code="bad_params")
            self._speak(text, replace=True)
            return ok(
                req.id, {"spoken": text, "speak_enabled": self.policy.speak_enabled}
            )
        if req.method == "ingest_transcript":
            with self._authority_lock:
                if getattr(self, "_delivery_closed", False):
                    self._record_activity(
                        "withheld",
                        code="delivery_closed",
                        method="ingest_transcript",
                        reason="Daemon shutdown has begun; ingest was rejected.",
                        sent=False,
                    )
                    return err(
                        req.id,
                        "Daemon shutdown has begun; ingest was rejected and nothing was sent.",
                        code="delivery_closed",
                    )
            # Debug / typed stand-in for STT.
            text = str(req.params.get("text") or "")
            if not text:
                return err(req.id, "text required", code="bad_params")
            supplied_id = req.params.get("utterance_id")
            if (
                not isinstance(supplied_id, str)
                or not supplied_id
                or supplied_id != supplied_id.strip()
                or len(supplied_id) > 200
            ):
                return err(
                    req.id, "utterance_id is required and invalid", code="bad_params"
                )
            utterance_id = supplied_id
            self._handle_transcript(FinalTranscript(utterance_id, text, origin="typed"))
            return ok(
                req.id,
                {
                    "utterance_id": utterance_id,
                    "transcript": self.last_transcript,
                    "action": self.last_voice_action,
                },
            )
        if req.method == "resolve":
            space = str(req.params.get("space") or "")
            agent = str(req.params.get("agent") or "") or None
            if not space:
                return err(req.id, "space required", code="bad_params")
            try:
                return ok(req.id, self.resolve_only(space, agent))
            except HerdrError as exc:
                return err(req.id, str(exc), code="herdr")
        if req.method == "prompt":
            self._record_activity(
                "withheld",
                code="verified_planner_required",
                method="prompt",
                reason="Direct prompt RPC is disabled; nothing was sent.",
                sent=False,
            )
            return err(
                req.id,
                "Direct prompt RPC is disabled; use ingest_transcript with a wake-addressed utterance.",
                code="verified_planner_required",
            )
        if req.method == "notify_agent_event":
            # Optional plugin [[events]] forwarder path.
            pane_id = str(req.params.get("pane_id") or "")
            status = str(
                req.params.get("agent_status") or req.params.get("status") or ""
            )
            if pane_id and status:
                self._maybe_announce(pane_id, status)
            return ok(req.id, {"accepted": True})
        if req.method in {"quit", "handoff_quit"}:
            mode = self._linearize_shutdown()
            result: dict[str, Any] = {"quitting": True}
            if req.method == "handoff_quit":
                result.update(handoff_version=1, mode=mode)
            return ok(req.id, result)
        return err(req.id, f"unknown method {req.method}", code="unknown_method")

    def _start_voice(self, *, wait_secs: float = 45.0) -> bool:
        # Mode transitions are ordered before authority and voice lifecycle.
        # Authority is never held while joining a listener: its callbacks also
        # consult authority, so doing so would deadlock listener teardown.
        with self._mic_transition():
            with self._authority_lock:
                if self.policy.mic_mode != "listen" or getattr(
                    self, "_delivery_closed", False
                ):
                    return False
                current_generation = int(getattr(self, "_listener_generation", 0))
                if (
                    self.voice
                    and self.voice.running
                    and self.voice.listener_generation == current_generation
                ):
                    return True
                generation = current_generation + 1
                self._listener_generation = generation
            with self._voice_lock:
                if self.voice is not None and not self._stop_voice_unlocked(
                    authority_already_revoked=True
                ):
                    log.error("old voice listener did not stop; replacement withheld")
                    return False
                listener = VoiceListener(
                    on_transcript=self._handle_transcript,
                    on_speech_start=lambda: self._authorized_voice_callback(
                        generation, self._handle_speech_start
                    ),
                    on_speech_stop=lambda: self._authorized_voice_callback(
                        generation, self._handle_speech_stop
                    ),
                    on_partial_transcript=lambda text: self._authorized_voice_callback(
                        generation, self._handle_partial_transcript, text
                    ),
                    on_capture_start=lambda: self._authorized_voice_callback(
                        generation, self._handle_capture_start
                    ),
                    input_device_index=self.config.input_device_index,
                    stt_model=self.config.stt_model,
                    sample_rate=self.config.sample_rate,
                    vad_stop_secs=self.config.vad_stop_secs,
                    listener_generation=generation,
                )
                self.voice = listener
                listener.start()
            ready = listener.wait_until_ready(timeout=wait_secs)
            with self._authority_lock:
                still_authorized = (
                    self.policy.mic_mode == "listen"
                    and not getattr(self, "_delivery_closed", False)
                    and generation == getattr(self, "_listener_generation", 0)
                )
            return bool(ready and still_authorized and listener.running)

    def _handle_speech_start(self) -> None:
        """Give live user speech priority over current and queued TTS."""
        removed = self.speaker.interrupt(clear_queue=True)
        log.debug("speech start interrupted TTS; removed=%s", removed)

    def _active_pending_clarification(self) -> dict[str, Any] | None:
        pending = getattr(self, "pending_clarification", None)
        if not pending:
            return None
        now = time.monotonic()
        expires_at = pending.get("expires_at")
        if not isinstance(expires_at, (int, float)):
            started = float(pending.get("started_at") or now)
            expires_at = started + float(
                getattr(self.config, "clarification_max_secs", 45.0)
            )
            pending["expires_at"] = expires_at
        if now >= expires_at:
            self.pending_clarification = None
            self._record_activity(
                "clarification_expired",
                clarification_id=pending.get("clarification_id"),
                sent=False,
            )
            self._set_activity_status(
                phase="ready",
                mode="idle",
                capture_active=False,
                waiting_for="wake word and a complete request",
                last_result=(
                    "Clarification window expired; use the wake address and repeat "
                    "the complete request. Nothing was sent."
                ),
            )
            return None
        return pending

    def _stamp_pending(self, pending: dict[str, Any]) -> dict[str, Any]:
        stamped = dict(pending)
        stamped.setdefault("clarification_id", uuid.uuid4().hex)
        now = time.monotonic()
        stamped.setdefault("started_at", now)
        stamped.setdefault(
            "expires_at",
            float(stamped["started_at"])
            + float(getattr(self.config, "clarification_max_secs", 45.0)),
        )
        return stamped

    def _clear_pending_for_state(self, state: dict[str, Any]) -> None:
        clarification_id = state.get("clarification_id")
        pending = getattr(self, "pending_clarification", None)
        if (
            clarification_id
            and pending
            and pending.get("clarification_id") == clarification_id
        ):
            self.pending_clarification = None

    def _handle_capture_start(self) -> None:
        """Publish capture state synchronously before interim words can arrive."""
        dictation = getattr(self, "dictation", None)
        dictating = bool(dictation and dictation.active)
        pending = self._active_pending_clarification()
        self._current_input_mode = (
            "clarification" if pending else "dictation" if dictating else "detecting"
        )
        self._set_activity_status(
            phase="listening",
            mode=self._current_input_mode,
            capture_active=True,
            speech_active=True,
            waiting_for=(
                "your clarification answer"
                if pending
                else "more dictation or a finish request"
                if dictating
                else "wake word and command"
            ),
            live_transcript="",
            dictation_buffer=(
                str((pending or {}).get("request_text") or "")
                if pending
                else dictation.joined()
                if dictating
                else ""
            ),
            last_result="",
            delivery_status=None,
        )

    def _handle_speech_stop(self) -> None:
        self._set_activity_status(
            phase="transcribing",
            mode=self._current_input_mode,
            capture_active=True,
            speech_active=False,
            waiting_for="final transcript after silence",
        )

    def _handle_partial_transcript(self, text: str) -> None:
        """Publish streaming speech without guessing a post-wake command."""
        pending = self._active_pending_clarification()
        mode = (
            "clarification"
            if pending
            else "dictation"
            if self.dictation.active
            else "detecting"
        )
        waiting_for = (
            "your clarification answer"
            if pending
            else "more dictation or a finish request"
            if self.dictation.active
            else "wake word and command"
        )
        if not self.dictation.active and not pending:
            woke = match_wake_phrase(text, self.config.wake_phrases).matched
            if not self.config.require_wake:
                woke = True
            if woke:
                mode = "awaiting_plan"
                waiting_for = "silence, then LLM interpretation"
        self._current_input_mode = mode
        self._set_activity_status(
            phase="listening",
            mode=mode,
            capture_active=True,
            speech_active=True,
            waiting_for=waiting_for,
            live_transcript=text,
            dictation_buffer=(
                str((pending or {}).get("request_text") or "")
                if pending
                else self.dictation.joined()
                if self.dictation.active
                else ""
            ),
        )

    def _current_listener_generation(self) -> int:
        lock = getattr(self, "_authority_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._authority_lock = lock
        with lock:
            return int(getattr(self, "_listener_generation", 0))

    def _current_authority(self, origin: str) -> tuple[int, int]:
        lock = getattr(self, "_authority_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._authority_lock = lock
        with lock:
            generation = (
                getattr(self, "_listener_generation", 0)
                if origin == "voice"
                else getattr(self, "_typed_generation", 0)
            )
            return int(generation), int(getattr(self, "_global_generation", 0))

    def _close_delivery_authority(self) -> None:
        lock = getattr(self, "_authority_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._authority_lock = lock
        with lock:
            if not getattr(self, "_delivery_closed", False):
                self._delivery_closed = True
                self._global_generation = getattr(self, "_global_generation", 0) + 1
                self._listener_generation = getattr(self, "_listener_generation", 0) + 1
                self._typed_generation = getattr(self, "_typed_generation", 0) + 1
            stop = getattr(self, "_stop", None)
            if stop is not None:
                stop.set()

    def _revoke_listener_authority(self) -> int:
        lock = getattr(self, "_authority_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._authority_lock = lock
        with lock:
            self._listener_generation = (
                int(getattr(self, "_listener_generation", 0)) + 1
            )
            return self._listener_generation

    def _authorized_voice_callback(
        self, generation: int, callback: Any, *args: Any
    ) -> None:
        if generation != self._current_listener_generation():
            return
        callback(*args)

    def _mic_mode_transition_lock(self) -> threading.RLock:
        lock = getattr(self, "_mode_transition_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._mode_transition_lock = lock
        return lock

    @contextmanager
    def _mic_transition(self):
        """Serialize a transition and publish its lifetime to concurrent Status."""
        with self._mic_mode_transition_lock():
            with self._authority_lock:
                already_pending = bool(getattr(self, "_mic_transition_count", 0))
                self._mic_transition_count = (
                    int(getattr(self, "_mic_transition_count", 0)) + 1
                )
            try:
                yield already_pending
            finally:
                with self._authority_lock:
                    self._mic_transition_count = max(
                        0, int(getattr(self, "_mic_transition_count", 1)) - 1
                    )
                    completed = self._mic_transition_count == 0
                if completed:
                    self._set_activity_status()

    def _begin_async_mic_transition(self) -> None:
        with self._authority_lock:
            self._mic_transition_count = (
                int(getattr(self, "_mic_transition_count", 0)) + 1
            )

    def _end_async_mic_transition(self) -> None:
        with self._authority_lock:
            self._mic_transition_count = max(
                0, int(getattr(self, "_mic_transition_count", 1)) - 1
            )
            completed = self._mic_transition_count == 0
        if completed:
            self._set_activity_status()

    def _publish_mic_transition_error(
        self, message: str, *, waiting_for: str, phase: str = "error"
    ) -> None:
        self._set_activity_status(
            phase=phase,
            mode="idle",
            capture_active=False,
            speech_active=False,
            waiting_for=waiting_for,
            live_transcript="",
            dictation_buffer="",
            last_result=message,
        )

    def _schedule_shutdown(self) -> None:
        if getattr(self, "_shutdown_scheduled", False):
            return
        self._shutdown_scheduled = True
        threading.Thread(target=self._delayed_stop, daemon=True).start()

    def _linearize_shutdown(self) -> str:
        """Freeze mic mode and begin shutdown under the mode-transition lock."""
        with self._mic_mode_transition_lock():
            mode = self.policy.mic_mode
            # Close delivery first so later mode requests cannot pass.
            self._close_delivery_authority()
            try:
                self._record_activity(
                    "delivery_closed",
                    reason="quit acknowledged only after permanent delivery closure",
                    sent=False,
                )
                self._set_activity_status(
                    phase="shutting_down",
                    mode="closed",
                    capture_active=bool(
                        self.voice and getattr(self.voice, "capture_eligible", False)
                    ),
                    speech_active=False,
                    waiting_for="process exit",
                    last_result=(
                        "Delivery is permanently closed; no new ingest is accepted."
                    ),
                )
            except Exception:
                log.exception("shutdown activity publication failed")
            self._schedule_shutdown()
            return mode

    def _listener_callback_is_current(self) -> bool:
        listener = self.voice
        return bool(listener and getattr(listener, "in_callback_thread", False) is True)

    def _stop_voice(self) -> bool:
        self._revoke_listener_authority()
        with self._voice_lock:
            return self._stop_voice_unlocked(authority_already_revoked=True)

    def _mute_voice(self) -> bool:
        with self._mic_transition():
            # Revocation is the first state change so an in-flight verified plan
            # cannot slip through between the mute request and listener stop.
            with self._authority_lock:
                frozen = bool(getattr(self, "_delivery_closed", False))
                if not frozen:
                    self._listener_generation = (
                        int(getattr(self, "_listener_generation", 0)) + 1
                    )
                    self.policy.set_mic_mode("mute")
            if frozen:
                with self._voice_lock:
                    listener = self.voice
                    if listener is not None:
                        listener.request_stop()
                return False
            with self._voice_lock:
                stopped = self._stop_voice_unlocked(authority_already_revoked=True)
            if not stopped:
                self._fail_closed_mic_barrier()
            return stopped

    def _mute_voice_durably(self) -> bool:
        """Persist mute preference and stop the listener."""
        with self._mic_transition():
            preference_error: MicPreferenceError | None = None
            with self._authority_lock:
                try:
                    # Marker makes a crash during this write restart as mute.
                    self._persist_mic_preference("mute")
                except MicPreferenceError as exc:
                    preference_error = exc
                frozen = bool(getattr(self, "_delivery_closed", False))
                if not frozen:
                    self._listener_generation = (
                        int(getattr(self, "_listener_generation", 0)) + 1
                    )
                    self.policy.set_mic_mode("mute")
            if frozen:
                with self._voice_lock:
                    listener = self.voice
                    if listener is not None:
                        listener.request_stop()
                if preference_error is not None:
                    raise preference_error
                raise MicModeFrozenError(
                    "daemon shutdown froze the final mode; durable mute intent "
                    "was preserved for restart"
                )
            with self._voice_lock:
                stopped = self._stop_voice_unlocked(authority_already_revoked=True)
            if not stopped:
                self._fail_closed_mic_barrier()
                raise MicClosureError(
                    "listener did not close within the bounded mute barrier; "
                    "daemon shutdown was initiated"
                )
            if preference_error is not None:
                raise preference_error
            return True

    def _start_spoken_mute_completion(
        self,
        action: dict[str, Any],
        *,
        transcript: str,
        state: dict[str, Any],
    ) -> None:
        """Revoke on the listener thread and prove closure from a join-safe thread."""
        preference_error: MicPreferenceError | None = None
        with self._mic_mode_transition_lock():
            with self._authority_lock:
                try:
                    self._persist_mic_preference("mute")
                except MicPreferenceError as exc:
                    preference_error = exc
                frozen = bool(getattr(self, "_delivery_closed", False))
                if not frozen:
                    self._listener_generation = (
                        int(getattr(self, "_listener_generation", 0)) + 1
                    )
                    self.policy.set_mic_mode("mute")
                    self._begin_async_mic_transition()
            listener = self.voice
            if listener is not None:
                listener.request_stop()
            if frozen:
                if preference_error is not None:
                    raise preference_error
                raise MicModeFrozenError(
                    "daemon shutdown froze the final mode; durable mute intent "
                    "was preserved for restart"
                )

            pending_action = dict(action)
            pending_action["result"] = {
                "ok": None,
                "sent": False,
                "pending": True,
                "code": "mic_closure_pending",
                "message": "Mute requested; proving microphone listener closure.",
            }
            self._commit_voice_action(pending_action)

            try:
                threading.Thread(
                    target=self._complete_spoken_mute,
                    args=(action, transcript, state, preference_error),
                    name="voicerdr-mute-completion",
                    daemon=True,
                ).start()
            except Exception as exc:
                self._end_async_mic_transition()
                self._fail_closed_mic_barrier()
                raise MicClosureError(
                    f"could not start microphone closure worker: {exc}; "
                    "daemon shutdown was initiated"
                ) from exc

    def _complete_spoken_mute(
        self,
        action: dict[str, Any],
        transcript: str,
        state: dict[str, Any],
        preference_error: MicPreferenceError | None,
    ) -> None:
        with self._mic_mode_transition_lock():
            with self._voice_lock:
                stopped = self._stop_voice_unlocked(authority_already_revoked=True)
            if not stopped:
                self._fail_closed_mic_barrier()
                self._end_async_mic_transition()
                self._withhold(
                    None,
                    transcript=transcript,
                    code="mic_closure",
                    reason=(
                        "Microphone mute failed closed without a success result: "
                        "listener did not close within the bounded mute barrier; "
                        "daemon shutdown was initiated"
                    ),
                    error=True,
                )
                return
            self._end_async_mic_transition()
            if preference_error is not None:
                self._withhold(
                    None,
                    transcript=transcript,
                    code="mic_preference",
                    reason=(
                        "Microphone mute failed closed without a success result: "
                        f"{preference_error}"
                    ),
                    error=True,
                )
                self._publish_mic_transition_error(
                    f"Microphone muted; preference save failed: {preference_error}",
                    waiting_for="a writable preference store",
                    phase="muted",
                )
                return

            completed_action = dict(action)
            completed_action["result"] = {
                "ok": True,
                "sent": False,
                "mode": "mute",
            }
            self._clear_pending_for_state(state)
            self._commit_voice_action(completed_action)
            self._notify("voicerdr", "Muted.", speak=True)

    def _fail_closed_mic_barrier(self) -> None:
        """Close delivery and shut down when mute cannot prove listener exit."""
        self._close_delivery_authority()
        self._schedule_shutdown()

    def _restore_mic_preference(self) -> None:
        """Apply durable mic preference; untrusted state becomes mute."""
        preference_mode: str | None = None
        preference_error: str | None = None
        try:
            preference_mode = self._load_mic_preference()
        except MicPreferenceError as exc:
            # An interrupted preference write is newer state evidence than any
            # leftover handoff and must never be upgraded back to listen.
            preference_mode = "mute"
            preference_error = str(exc)

        handoff_mode: str | None = None
        handoff_error: str | None = None
        try:
            handoff_mode = self._load_mic_handoff()
        except MicPreferenceError as exc:
            # A partial, corrupt, or untrusted launcher handoff is evidence of a
            # prior daemon, never a clean first start.
            handoff_mode = "mute"
            handoff_error = str(exc)
        if handoff_mode is not None or preference_error is not None:
            selected_mode = handoff_mode or "mute"
            selected_error = handoff_error or preference_error
            if selected_mode == "listen" and preference_mode == "mute":
                selected_mode = "mute"
                selected_error = preference_error or (
                    "listen handoff was superseded by a durable mute preference"
                )
            self.policy.set_mic_mode(selected_mode)
            try:
                # Consume only after the selected safe mode is durable. If this
                # crashes, the preference marker itself forces mute next time.
                self._persist_mic_preference(selected_mode)
                self._remove_mic_handoff()
            except MicPreferenceError as exc:
                self.policy.set_mic_mode("mute")
                try:
                    # Cleanup may have removed a listen handoff after its
                    # preference was written but before the directory fsync.
                    # Force mute so the next start does not inherit listen.
                    self._persist_mic_preference("mute")
                except MicPreferenceError:
                    pass
                self._mic_preference_explicit = True
                self._mic_preference_error = str(exc)
                log.error(
                    "mic handoff could not be consumed; defaulting to mute: %s", exc
                )
                return
            self._mic_preference_explicit = True
            self._mic_preference_error = selected_error
            if selected_error:
                log.error("microphone state failed closed: %s", selected_error)
            else:
                log.info("consumed launcher microphone handoff: %s", selected_mode)
            return
        mode = preference_mode
        if mode is None:
            try:
                mode = self._load_legacy_mic_preference()
                if mode is not None:
                    # Migration happens synchronously before startup considers
                    # opening input. Persist it so the legacy evidence is only
                    # needed once and later status redraws cannot erase it.
                    self.policy.set_mic_mode(mode)
                    self._persist_mic_preference(mode)
                    log.info("migrated legacy microphone mode: %s", mode)
            except MicPreferenceError as exc:
                self._mic_preference_explicit = True
                self._mic_preference_error = str(exc)
                self.policy.set_mic_mode("mute")
                log.error(
                    "legacy mic preference unavailable; defaulting to mute: %s", exc
                )
                return
        self._mic_preference_error = None
        self._mic_preference_explicit = mode is not None
        if mode is not None:
            self.policy.set_mic_mode(mode)

    def _load_mic_handoff(self) -> str | None:
        try:
            self.paths.mic_handoff_pending.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise MicPreferenceError(
                f"handoff marker could not be checked: {exc}"
            ) from exc
        else:
            raise MicPreferenceError("an interrupted microphone handoff was detected")

        try:
            metadata = self.paths.mic_handoff.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise MicPreferenceError(f"handoff could not be inspected: {exc}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise MicPreferenceError("handoff is not a regular file")
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise MicPreferenceError("handoff is not a trusted private file")
        try:
            raw = json.loads(self.paths.mic_handoff.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MicPreferenceError(
                f"handoff is unreadable or corrupt: {exc}"
            ) from exc
        if (
            not isinstance(raw, dict)
            or raw.get("version") != 1
            or raw.get("mode") not in {"listen", "mute"}
            or set(raw) != {"version", "mode"}
        ):
            raise MicPreferenceError(
                "handoff must contain only version 1 and a valid mode"
            )
        return str(raw["mode"])

    def _remove_mic_handoff(self) -> None:
        try:
            self.paths.mic_handoff.unlink(missing_ok=True)
            self.paths.mic_handoff_pending.unlink(missing_ok=True)
            directory_fd = os.open(self.paths.state_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise MicPreferenceError(f"handoff cleanup was not durable: {exc}") from exc

    def _load_legacy_mic_preference(self) -> str | None:
        """Adopt a muted pre-preference activity snapshot during upgrade."""
        try:
            raw = json.loads(self.paths.activity_state.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # No snapshot is the first-install case; the configured default applies.
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MicPreferenceError(
                f"legacy activity state is unreadable or corrupt: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise MicPreferenceError("legacy activity state must be an object")
        return "mute" if raw.get("phase") == "muted" else None

    def _load_mic_preference(self) -> str | None:
        try:
            self.paths.mic_preference_pending.stat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise MicPreferenceError(
                f"interrupted-update marker could not be checked: {exc}"
            ) from exc
        else:
            raise MicPreferenceError("an interrupted preference update was detected")
        try:
            raw = json.loads(self.paths.mic_preference.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MicPreferenceError(
                f"preference is unreadable or corrupt: {exc}"
            ) from exc
        if (
            not isinstance(raw, dict)
            or raw.get("version") != 1
            or raw.get("mode") not in {"listen", "mute"}
            or set(raw) != {"version", "mode"}
        ):
            raise MicPreferenceError(
                "preference must contain only version 1 and a valid mode"
            )
        return str(raw["mode"])

    def _persist_mic_preference(self, mode: str) -> None:
        if mode not in {"listen", "mute"}:
            raise ValueError(f"unknown mode: {mode}")
        lock = getattr(self, "_mic_preference_lock", None)
        if lock is None:
            self._mic_preference_lock = threading.Lock()
            lock = self._mic_preference_lock
        target = self.paths.mic_preference
        with lock:
            try:
                self._atomic_write_mic_state(
                    self.paths.mic_preference_pending,
                    {"version": 1, "target_mode": mode},
                )
                self._atomic_write_mic_state(target, {"version": 1, "mode": mode})
                self.paths.mic_preference_pending.unlink()
                directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except (OSError, ReplayLedgerError) as exc:
                self._mic_preference_error = str(exc)
                raise MicPreferenceError(
                    f"durable preference write failed: {exc}"
                ) from exc
            self._mic_preference_explicit = True
            self._mic_preference_error = None

    def _atomic_write_mic_state(self, target: Any, value: dict[str, Any]) -> None:
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        fd: int | None = None
        try:
            self._ensure_state_dir_durable()
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            payload = (json.dumps(value, sort_keys=True) + "\n").encode()
            with os.fdopen(fd, "wb") as stream:
                fd = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
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

    def _stop_voice_unlocked(self, *, authority_already_revoked: bool = False) -> bool:
        if not authority_already_revoked:
            self._revoke_listener_authority()
        if self.voice is not None:
            listener = self.voice
            try:
                stopped = listener.stop()
            except Exception:
                log.exception("voice stop failed")
                stopped = False
            # An initializing listener is still eligible to open input. Only
            # thread termination proves the microphone can no longer be opened
            # or retained by this listener.
            if not stopped:
                return False
            self.voice = None
        return True

    def _seed_focus(self) -> None:
        try:
            for agent in self.herdr.agent_list():
                if agent.get("focused") and agent.get("workspace_id"):
                    self.focused_workspace_id = str(agent["workspace_id"])
                    log.info(
                        "seeded focus from agent list: %s", self.focused_workspace_id
                    )
                    return
            for ws in self.herdr.workspace_list():
                if ws.get("focused") and ws.get("workspace_id"):
                    self.focused_workspace_id = str(ws["workspace_id"])
                    log.info(
                        "seeded focus from workspace list: %s",
                        self.focused_workspace_id,
                    )
                    return
        except HerdrError as exc:
            log.debug("seed focus failed: %s", exc)

    def _reload_aliases(self) -> None:
        try:
            self.config.aliases = load_aliases(self.paths)
        except Exception:
            log.debug("alias reload failed", exc_info=True)

    def _label_state_path(self):
        return self.paths.state_dir / "space_bases.json"

    def _sync_space_labels(self) -> None:
        try:
            sync_space_number_labels(
                self.herdr,
                state_path=self._label_state_path(),
                enabled=self.config.show_space_numbers,
            )
            self._publish_space_tokens()
        except Exception:
            log.exception("space label sync failed")

    def _publish_space_tokens(self) -> None:
        """Best-effort metadata tokens (for custom sidebar layouts)."""
        try:
            for ws in self.herdr.workspace_list():
                wid = str(ws.get("workspace_id") or "")
                num = ws.get("number")
                if not wid or num is None:
                    continue
                self.herdr.workspace_report_metadata(
                    wid,
                    source="voicerdr",
                    tokens={"num": str(num), "voice": f"#{num}"},
                    ttl_ms=86_400_000,
                )
        except HerdrError as exc:
            log.debug("publish space tokens failed: %s", exc)

    def _number_labels(self) -> dict[int, str]:
        out: dict[int, str] = {}
        try:
            for ws in self.herdr.workspace_list():
                label = ws.get("label")
                num = ws.get("number")
                if label is None or num is None:
                    continue
                try:
                    # Prefer bare base name for downstream alias/routing clarity.
                    out[int(num)] = strip_number_prefix(str(label))
                except (TypeError, ValueError):
                    continue
        except HerdrError:
            pass
        return out

    def _space_directory(self) -> list[dict[str, Any]]:
        """Herdr slot number + label + spoken nicknames that map to it."""
        self._reload_aliases()
        by_label: dict[str, list[str]] = {}
        for nick, label in self.config.aliases.items():
            key = strip_number_prefix(label).casefold()
            by_label.setdefault(key, []).append(nick)
        rows: list[dict[str, Any]] = []
        try:
            workspaces = self.herdr.workspace_list()
        except HerdrError:
            workspaces = []
        try:
            agents = self.herdr.agent_list()
        except HerdrError:
            agents = []
        agents_by_workspace: dict[str, list[dict[str, Any]]] = {}
        for agent in agents:
            workspace_id = str(agent.get("workspace_id") or "")
            agents_by_workspace.setdefault(workspace_id, []).append(
                {
                    "name": agent_name(agent),
                    "pane_id": agent.get("pane_id"),
                    "title": agent.get("terminal_title_stripped"),
                    "status": agent.get("agent_status") or "unknown",
                }
            )
        for ws in workspaces:
            label = str(ws.get("label") or ws.get("workspace_id") or "")
            base = strip_number_prefix(label)
            try:
                number = int(ws.get("number"))
            except (TypeError, ValueError):
                number = None
            nicks = sorted(
                set(by_label.get(base.casefold(), [])),
                key=lambda s: (len(s.split()), len(s)),
            )
            rows.append(
                {
                    "number": number,
                    "label": label,
                    "base": base,
                    "workspace_id": ws.get("workspace_id"),
                    "nicknames": nicks,
                    "agents": agents_by_workspace.get(
                        str(ws.get("workspace_id") or ""), []
                    ),
                }
            )
        rows.sort(key=lambda r: (r["number"] is None, r["number"] or 0, r["label"]))
        return rows

    def _handle_transcript(self, final: FinalTranscript | str) -> None:
        """Run the voice state machine for one turn and consume its utterance ID."""
        if isinstance(final, FinalTranscript):
            utterance_id, text = final.utterance_id, final.text
            provenance_valid = final.provenance_valid
            provenance_error = final.provenance_error
            listener_generation = final.listener_generation
            origin = final.origin
        else:
            # Tests and internal callers get a distinct turn unless they explicitly
            # provide a FinalTranscript retry identity.
            utterance_id, text = f"internal:{uuid.uuid4().hex}", final
            provenance_valid = True
            provenance_error = None
            listener_generation = None
            origin = "voice"
        lock = getattr(self, "_voice_state_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._voice_state_lock = lock
        with lock:
            if getattr(self, "_delivery_closed", False):
                self._withhold(
                    None,
                    transcript=text,
                    code="delivery_closed",
                    reason="Daemon shutdown has begun; nothing was sent.",
                    error=True,
                    notify=False,
                )
                return
            if origin not in {"voice", "typed", "internal"}:
                self._withhold(
                    None,
                    transcript=text,
                    code="invalid_transcript_origin",
                    reason="Transcript origin was invalid; nothing was sent.",
                    error=True,
                    notify=False,
                )
                return
            current_generation, global_generation = self._current_authority(origin)
            if listener_generation is None:
                listener_generation = current_generation
            if listener_generation != current_generation:
                self._withhold(
                    None,
                    transcript=text,
                    code="listener_authority_revoked",
                    reason="This transcript came from a stale listener; nothing was sent.",
                    error=True,
                    notify=False,
                )
                return
            if not provenance_valid:
                self._withhold(
                    None,
                    transcript=text,
                    code="unmatched_final_provenance",
                    reason=(
                        f"Final transcript was not bound to a live VAD turn"
                        f" ({provenance_error or 'unknown provenance'}). Nothing was sent."
                    ),
                    error=True,
                    notify=False,
                )
                return
            try:
                fresh = self._consume_utterance_id(utterance_id)
            except ReplayLedgerError as exc:
                self._withhold(
                    None,
                    transcript=text,
                    code="replay_ledger_unavailable",
                    reason=(
                        f"Replay protection could not durably consume this turn: {exc}. "
                        "Nothing was sent; fix the state storage and retry with the "
                        "same utterance ID."
                    ),
                    error=True,
                    notify=False,
                )
                return
            if not fresh:
                self._duplicate_utterance(utterance_id, text)
                return
            self._process_final_transcript(
                text,
                utterance_id=utterance_id,
                listener_generation=listener_generation,
                origin=origin,
                global_generation=global_generation,
            )

    def _duplicate_utterance(self, utterance_id: str, text: str) -> None:
        action = {
            "action_kind": "no_action",
            "utterance_id": utterance_id,
            "transcript": text,
            "reason": "utterance identity was already consumed",
            "result": {
                "ok": False,
                "sent": False,
                "code": "utterance_already_consumed",
                "message": "This utterance turn was already consumed; nothing was sent.",
            },
        }
        self._record_activity(
            "withheld",
            code="utterance_already_consumed",
            utterance_id=utterance_id,
            transcript=text,
            sent=False,
        )
        self._commit_voice_action(action)

    def _consume_utterance_id(self, utterance_id: str) -> bool:
        ledger_error = getattr(self, "_replay_ledger_error", None)
        if ledger_error:
            raise ReplayLedgerError(ledger_error)
        if not getattr(self, "paths", None):
            raise ReplayLedgerError("no replay ledger path is configured")
        if getattr(self, "_consumed_utterance_ids", None) is None:
            raise ReplayLedgerError("replay ledger was not initialized")
        try:
            with self._ledger_transaction():
                consumed = self._load_consumed_utterances()
                if utterance_id in consumed:
                    self._consumed_utterance_ids = consumed
                    return False
                updated = {*consumed, utterance_id}
                self._persist_consumed_utterances(updated)
        except ReplayLedgerError as exc:
            self._replay_ledger_error = str(exc)
            self._replay_ledger_ready = False
            raise
        self._consumed_utterance_ids = updated
        self._replay_ledger_ready = True
        return True

    def _initialize_replay_ledger(self) -> set[str]:
        try:
            with self._ledger_transaction():
                try:
                    consumed = self._load_consumed_utterances()
                except FileNotFoundError:
                    self._persist_consumed_utterances(set())
                    consumed = set()
            self._replay_ledger_error = None
            self._replay_ledger_ready = True
            return consumed
        except ReplayLedgerError:
            raise
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ReplayLedgerError(f"ledger is unreadable or corrupt: {exc}") from exc

    def _load_consumed_utterances(self) -> set[str]:
        paths = getattr(self, "paths", None)
        if not paths:
            raise ReplayLedgerError("no replay ledger path is configured")
        try:
            raw = json.loads(paths.utterance_ledger.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise ReplayLedgerError(f"ledger read failed: {exc}") from exc
        if (
            not isinstance(raw, list)
            or any(not isinstance(item, str) or not item for item in raw)
            or len(set(raw)) != len(raw)
        ):
            raise ReplayLedgerError("ledger must be a unique list of non-empty IDs")
        return set(raw)

    def _persist_consumed_utterances(self, consumed: set[str]) -> None:
        paths = getattr(self, "paths", None)
        if not paths:
            raise ReplayLedgerError("no replay ledger path is configured")
        target = paths.utterance_ledger
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        fd: int | None = None
        try:
            self._ensure_state_dir_durable()
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            payload = (json.dumps(sorted(consumed)) + "\n").encode()
            with os.fdopen(fd, "wb") as stream:
                fd = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as close_exc:
                    raise ReplayLedgerError(
                        f"durable ledger temporary close failed: {close_exc}"
                    ) from close_exc
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ReplayLedgerError(f"durable ledger write failed: {exc}") from exc

    def _ensure_state_dir_durable(self) -> None:
        paths = getattr(self, "paths", None)
        if not paths:
            raise ReplayLedgerError("no replay ledger path is configured")
        state_dir = paths.state_dir
        try:
            missing: list[Any] = []
            cursor = state_dir
            while not cursor.exists():
                missing.append(cursor)
                if cursor.parent == cursor:
                    break
                cursor = cursor.parent
            for directory in reversed(missing):
                try:
                    directory.mkdir()
                except FileExistsError:
                    pass
                parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            state_dir.chmod(0o700)
        except OSError as exc:
            raise ReplayLedgerError(
                f"state directory durability failed: {exc}"
            ) from exc

    @contextmanager
    def _ledger_transaction(self):
        self._ensure_state_dir_durable()
        fd: int | None = None
        failure: ReplayLedgerError | None = None
        try:
            try:
                fd = os.open(
                    self.paths.utterance_ledger_lock,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                created = True
            except FileExistsError:
                fd = os.open(self.paths.utterance_ledger_lock, os.O_RDWR)
                created = False
            if created:
                os.fsync(fd)
                directory_fd = os.open(
                    self.paths.state_dir, os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        except ReplayLedgerError as exc:
            failure = exc
        except (OSError, json.JSONDecodeError, UnicodeError, ValueError) as exc:
            failure = ReplayLedgerError(f"replay transaction failed: {exc}")
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError as exc:
                    failure = ReplayLedgerError(f"replay unlock failed: {exc}")
                try:
                    os.close(fd)
                except OSError as exc:
                    failure = ReplayLedgerError(f"replay lock close failed: {exc}")
            if failure is not None:
                raise failure

    def _process_final_transcript(
        self,
        text: str,
        *,
        utterance_id: str,
        listener_generation: int,
        origin: str = "voice",
        global_generation: int = 0,
    ) -> None:
        self.last_transcript = text
        self._record_activity(
            "heard",
            utterance_id=utterance_id,
            transcript=text,
            dictation_active=self.dictation.active,
        )
        self._reload_aliases()

        wake = match_wake_phrase(text, self.config.wake_phrases)
        pending = self._active_pending_clarification()
        if pending and wake.matched:
            # A new explicit activation is a new transaction. Never feed stale
            # clarification context into a complete wake-addressed request.
            self.pending_clarification = None
            self._record_activity(
                "clarification_superseded",
                clarification_id=pending.get("clarification_id"),
                superseding_utterance_id=utterance_id,
                sent=False,
            )
            pending = None
        if pending:
            pending = dict(pending)
            pending.setdefault("clarification_id", uuid.uuid4().hex)
            answer = text
            followup = {
                "utterance_id": utterance_id,
                "raw_transcript": text,
                "content": answer,
                "unresolved_slots": list(pending.get("unresolved_slots") or []),
            }
            followups = [*(pending.get("followups") or []), followup]
            plan_state = {
                "phase": "clarification",
                "resume_phase": pending.get("resume_phase"),
                "pending_transaction": {
                    "clarification_id": pending["clarification_id"],
                    "request": pending.get("request_text") or "",
                    "unresolved_slots": list(pending.get("unresolved_slots") or []),
                    "question": pending.get("question"),
                    "previous_plan": pending.get("previous_plan"),
                    "followups": followups,
                },
                "clarification_id": pending["clarification_id"],
                "source_content": answer,
                "clarification_request": str(pending.get("request_text") or ""),
                "clarification_answer": answer,
                "clarification_sources": {
                    f"clarification_followup_{index}": item["content"]
                    for index, item in enumerate(followups, start=1)
                },
            }
            self._dispatch_utterance(
                answer,
                transcript=text,
                raw_transcript=text,
                utterance_id=utterance_id,
                activation_phrase=None,
                input_mode=str(pending.get("input_mode") or "clarification"),
                state=plan_state,
                listener_generation=listener_generation,
                origin=origin,
                global_generation=global_generation,
            )
            result = (
                self.last_voice_action.get("result")
                if isinstance(self.last_voice_action, dict)
                else None
            )
            if isinstance(result, dict) and result.get("ok"):
                self._clear_pending_for_state(
                    {"clarification_id": pending["clarification_id"]}
                )
            elif not (
                isinstance(result, dict) and (result.get("ok") or result.get("pending"))
            ):
                try:
                    # Planner/no-action failures leave the original bounded
                    # transaction in place; a new clarification plan may update it.
                    visible = self.pending_clarification or pending
                    self._set_activity_status(
                        phase="awaiting_clarification",
                        mode=(
                            "dictation"
                            if visible.get("resume_phase") == "dictation_capture"
                            else "clarification"
                        ),
                        capture_active=True,
                        waiting_for=(
                            "clarification or retry; pending request is retained"
                        ),
                        dictation_buffer=str(visible.get("request_text") or ""),
                        last_result=(
                            "Nothing was sent. The pending request was retained; "
                            "reply before it expires or repeat it with the wake address."
                        ),
                    )
                except Exception:
                    log.exception("clarification retry status publication failed")
            return

        if self.config.dictation_enabled and self.dictation.active:
            if self.dictation.age_secs() > self.config.dictation_max_secs:
                self.dictation.clear()
                self._withhold(
                    None,
                    transcript=text,
                    code="dictation_timeout",
                    reason="Dictation timed out; nothing was sent.",
                )
                return
            raw_source = "\n".join(
                part for part in (self.dictation.raw_joined(), text) if part
            )
            self._dispatch_utterance(
                text,
                transcript=text,
                raw_transcript=raw_source,
                utterance_id=utterance_id,
                activation_phrase=None,
                input_mode="dictation",
                state={
                    "phase": "dictation_capture",
                    "buffer": self.dictation.joined(),
                    "buffer_fragments": list(self.dictation.parts),
                    "raw_fragments": list(self.dictation.raw_transcripts),
                    "start_examples": self.config.dictation_start_phrases,
                    "finish_examples": self.config.dictation_closing_phrases,
                    "cancel_examples": self.config.dictation_cancel_phrases,
                },
                listener_generation=listener_generation,
                origin=origin,
                global_generation=global_generation,
            )
            return

        if self.config.require_wake:
            if not wake.matched:
                action = {
                    "action_kind": "no_action",
                    "transcript": text,
                    "reason": "wake word not detected",
                    "result": {
                        "ok": False,
                        "sent": False,
                        "code": "wake_not_matched",
                        "message": "Wake word not detected; nothing was sent.",
                    },
                }
                if self.config.feedback_heard:
                    self._notify(
                        "voicerdr heard",
                        f"(no wake) {text[:160]}",
                        sound="none",
                        speak=False,
                    )
                self._commit_voice_action(action)
                return
            post_wake = wake.remainder
        else:
            post_wake = text
            wake = match_wake_phrase(text, [])

        self._dispatch_utterance(
            post_wake,
            transcript=text,
            raw_transcript=text,
            utterance_id=utterance_id,
            activation_phrase=wake.phrase,
            input_mode="one_shot",
            state={
                "phase": "idle",
                "dictation_enabled": self.config.dictation_enabled,
                "start_examples": self.config.dictation_start_phrases,
                "finish_examples": self.config.dictation_closing_phrases,
                "cancel_examples": self.config.dictation_cancel_phrases,
            },
            listener_generation=listener_generation,
            origin=origin,
            global_generation=global_generation,
        )

    def _dispatch_utterance(
        self,
        text_for_llm: str,
        *,
        transcript: str,
        utterance_id: str | None = None,
        activation_phrase: str | None = None,
        raw_transcript: str | None = None,
        input_mode: str = "one_shot",
        state: dict[str, Any] | None = None,
        listener_generation: int | None = None,
        origin: str = "voice",
        global_generation: int | None = None,
    ) -> None:
        """Plan, preflight, verify, and only then perform an exact action."""

        utterance_id = utterance_id or f"internal:{uuid.uuid4().hex}"
        listener_generation = (
            self._current_listener_generation()
            if listener_generation is None
            else listener_generation
        )
        if global_generation is None:
            _, global_generation = self._current_authority(origin)
        raw_source = raw_transcript or transcript
        plan_state = dict(state or {"phase": "idle"})
        plan_state.setdefault("source_content", text_for_llm)
        self._current_input_mode = input_mode
        self._set_activity_status(
            phase="interpreting",
            mode=input_mode,
            capture_active=plan_state.get("phase") == "dictation_capture",
            speech_active=False,
            waiting_for="LLM intent plan",
            live_transcript=transcript,
            dictation_buffer=(
                self.dictation.joined()
                if plan_state.get("phase") == "dictation_capture"
                else ""
            ),
            chosen_action=None,
            chosen_target=None,
            chosen_mode=None,
            delivery_status=None,
        )
        self._record_activity(
            "interpreting",
            input_mode=input_mode,
            transcript=raw_source,
            utterance=text_for_llm,
        )
        catalog = self._space_directory()
        catalog_fingerprint = self._catalog_fingerprint(catalog)
        utterance_evidence = {
            "utterance_id": utterance_id,
            "raw_transcript": raw_source,
            "post_wake_content": text_for_llm,
            "activation_phrase": activation_phrase,
            "capture_state": plan_state,
        }
        source_bundle = {
            "post_wake_content": text_for_llm,
            "raw_transcript": raw_source,
        }
        complete_buffer = plan_state.get("complete_buffer")
        if isinstance(complete_buffer, str) and complete_buffer:
            source_bundle["dictation_buffer"] = complete_buffer
        for source_name in ("clarification_request", "clarification_answer"):
            source_value = plan_state.get(source_name)
            if isinstance(source_value, str) and source_value:
                source_bundle[source_name] = source_value
        source_bundle.update(plan_state.get("clarification_sources") or {})
        utterance_digest = canonical_digest(utterance_evidence)
        plan: IntentPlan | None = None
        problem: str | None = None
        correction: dict[str, Any] | None = None
        last_planner_error: Exception | None = None
        last_preflight_problem: str | None = None
        rejected_problems: list[tuple[int, str]] = []
        clarification_retry_required = False
        for attempt in range(1, LLM_CORRECTIVE_ATTEMPTS + 1):
            try:
                planner_kwargs: dict[str, Any] = {
                    "utterance_id": utterance_id,
                    "raw_transcript": raw_source,
                    "activation_phrase": activation_phrase,
                    "spaces": catalog,
                    "catalog_digest": catalog_fingerprint,
                    "utterance_digest": utterance_digest,
                    "state": plan_state,
                }
                if correction is not None:
                    planner_kwargs["correction"] = correction
                clarification_only = bool(
                    correction is not None
                    and clarification_retry_required
                    and attempt == LLM_CORRECTIVE_ATTEMPTS
                )
                if clarification_only:
                    planner_kwargs["clarification_only"] = True
                bound = self.secretary.plan(text_for_llm, **planner_kwargs)
                if not isinstance(bound, BoundIntentPlan):
                    raise IntentValidationError(
                        "planner did not return a bound intent plan"
                    )
                if (
                    bound.utterance_digest != utterance_digest
                    or bound.catalog_digest != catalog_fingerprint
                ):
                    raise IntentValidationError("planner digest binding mismatch")
                candidate = plan_from_json(bound.decision.as_dict())
                if clarification_only and candidate.action_kind != "clarification":
                    raise IntentValidationError(
                        "final corrective decision must be an explicit clarification"
                    )
                problem = self._preflight_plan_shape(
                    candidate,
                    catalog,
                    plan_state,
                    source_text=text_for_llm,
                    source_bundle=source_bundle,
                )
                if problem is None:
                    plan = candidate
                    break
                last_preflight_problem = problem
                rejected_problems.append((attempt, problem))
                self._record_activity(
                    "planner_proposal_rejected",
                    attempt=attempt,
                    failure=problem,
                    proposed_plan=candidate.as_dict(),
                    catalog_fingerprint=catalog_fingerprint,
                    utterance_digest=utterance_digest,
                    sent=False,
                )
                candidate_actions = (
                    candidate.actions
                    if candidate.action_kind == "batch"
                    else (candidate,)
                )
                multi_agent_workspace_ids = {
                    str(space.get("workspace_id") or "")
                    for space in catalog
                    if len(space.get("agents") or []) > 1
                }
                clarification_retry_required = any(
                    action.action_kind == "agent_prompt"
                    and action.target is not None
                    and action.target.workspace_id in multi_agent_workspace_ids
                    for action in candidate_actions
                )
                correction = {
                    "attempt": attempt + 1,
                    "failure": problem,
                    "instruction": (
                        "Make a new independent schema-valid decision. Do not repair "
                        "or preserve an unsupported target; clarify when evidence is absent."
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                last_planner_error = exc
                problem = None
                correction = {
                    "attempt": attempt + 1,
                    "failure": str(exc),
                    "instruction": (
                        "Return only the strict action-specific JSON contract with "
                        "source-qualified evidence objects and exact supplied digests."
                    ),
                }
            if attempt < LLM_CORRECTIVE_ATTEMPTS:
                self._record_activity(
                    "planner_retry",
                    attempt=attempt + 1,
                    reason=(problem or str(last_planner_error)),
                    sent=False,
                )

        if plan is None:
            terminal_error = (
                last_planner_error
                if problem is None and clarification_retry_required
                else None
            )
            problem = problem or (
                None if terminal_error is not None else last_preflight_problem
            )
            reason = (
                problem or f"Intent planner unavailable or invalid: {terminal_error}"
            )
            if terminal_error is not None and last_preflight_problem:
                reason += f" Last rejected proposal: {last_preflight_problem}"
            elif rejected_problems:
                attempts = "; ".join(
                    f"attempt {attempt}: {failure}"
                    for attempt, failure in rejected_problems
                )
                reason += f" Rejected planner proposals: {attempts}"
            self._withhold(
                None,
                transcript=transcript,
                code="plan_blocked" if problem else "planner_error",
                reason=f"{reason}. Nothing was sent.",
                error=problem is None,
            )
            return

        self._record_activity(
            "intent_chosen",
            action=plan.action_kind,
            target=plan.target.as_dict() if plan.target else None,
            mode=plan.mode,
            confidence=plan.confidence,
            reason=plan.reason,
            workspace_evidence=(
                plan.workspace_evidence.as_dict() if plan.workspace_evidence else None
            ),
            agent_evidence=(
                plan.agent_evidence.as_dict() if plan.agent_evidence else None
            ),
            proposed_plan=plan.as_dict(),
            catalog_fingerprint=catalog_fingerprint,
            utterance_digest=utterance_digest,
        )
        self._set_activity_status(
            chosen_action=plan.action_kind,
            chosen_target=plan.target.as_dict() if plan.target else None,
            chosen_mode=plan.mode,
        )

        capabilities = self._verify_all_prompts(
            plan,
            raw_transcript=raw_source,
            transcript=transcript,
            state=plan_state,
            catalog=catalog,
            catalog_fingerprint=catalog_fingerprint,
            utterance_id=utterance_id,
            utterance_digest=utterance_digest,
            post_wake_content=text_for_llm,
            activation_phrase=activation_phrase,
            input_mode=input_mode,
            source_bundle=source_bundle,
            listener_generation=listener_generation,
            origin=origin,
            global_generation=global_generation,
        )
        if capabilities is None:
            return

        fresh_catalog = self._space_directory()
        if self._catalog_fingerprint(fresh_catalog) != catalog_fingerprint:
            self._withhold(
                plan,
                transcript=transcript,
                code="catalog_changed",
                reason="The live workspace catalog changed during planning. Nothing was sent.",
            )
            return
        problem = self._preflight_plan_shape(
            plan,
            fresh_catalog,
            plan_state,
            source_text=text_for_llm,
            source_bundle=source_bundle,
        )
        if problem:
            self._withhold(
                plan,
                transcript=transcript,
                code="catalog_revalidation_failed",
                reason=f"Catalog revalidation failed: {problem} Nothing was sent.",
            )
            return

        try:
            self._execute_plan(
                plan,
                transcript=transcript,
                raw_transcript=raw_source,
                input_mode=input_mode,
                state=plan_state,
                catalog=fresh_catalog,
                catalog_fingerprint=catalog_fingerprint,
                utterance_id=utterance_id,
                utterance_digest=utterance_digest,
                activation_phrase=activation_phrase,
                capabilities=capabilities,
                source_bundle=source_bundle,
                listener_generation=listener_generation,
                origin=origin,
                global_generation=global_generation,
            )
        except Exception as exc:
            log.exception("planned action execution failed")
            self._withhold(
                plan,
                transcript=transcript,
                code="execution_error",
                reason=(
                    f"The selected action failed: {exc}. No successful send was "
                    "recorded; inspect activity before retrying."
                ),
                error=True,
            )

    def _verify_all_prompts(
        self,
        plan: IntentPlan,
        *,
        raw_transcript: str,
        transcript: str,
        state: dict[str, Any],
        catalog: list[dict[str, Any]],
        catalog_fingerprint: str,
        utterance_id: str,
        utterance_digest: str,
        post_wake_content: str,
        activation_phrase: str | None,
        input_mode: str,
        source_bundle: dict[str, str],
        listener_generation: int,
        origin: str,
        global_generation: int,
    ) -> tuple[VerifiedPromptCapability, ...] | None:
        """Run the verifier and mint in-process delivery capabilities."""

        capabilities: list[VerifiedPromptCapability] = []
        plan_digest = canonical_digest(
            {
                "utterance_digest": utterance_digest,
                "catalog_digest": catalog_fingerprint,
                "complete_plan": plan.as_dict(),
            }
        )
        for proposed in self._prompt_plans(plan):
            established_payload_facts = payload_evidence_facts(proposed, source_bundle)
            self._set_activity_status(
                phase="verifying",
                waiting_for="independent LLM approval",
                chosen_action=proposed.action_kind,
                chosen_target=proposed.target.as_dict() if proposed.target else None,
            )
            self._record_activity(
                "verifying",
                action=proposed.action_kind,
                target=proposed.target.as_dict() if proposed.target else None,
                message=proposed.message,
                catalog_fingerprint=catalog_fingerprint,
                utterance_digest=utterance_digest,
                plan_digest=plan_digest,
            )
            verification: VerificationResult | None = None
            verification_error: Exception | None = None
            correction: dict[str, Any] | None = None
            for attempt in range(1, LLM_CORRECTIVE_ATTEMPTS + 1):
                try:
                    verifier = getattr(self, "verifier", self.secretary)
                    verifier_kwargs: dict[str, Any] = {
                        "utterance_id": utterance_id,
                        "raw_transcript": raw_transcript,
                        "post_wake_content": post_wake_content,
                        "activation_phrase": activation_phrase,
                        "proposed": proposed,
                        "complete_plan": plan,
                        "spaces": catalog,
                        "utterance_digest": utterance_digest,
                        "catalog_digest": catalog_fingerprint,
                        "plan_digest": plan_digest,
                        "source_evidence": source_bundle,
                        "established_payload_facts": established_payload_facts,
                        "state": state,
                    }
                    if correction is not None:
                        verifier_kwargs["correction"] = correction
                    candidate = verifier.verify_prompt(**verifier_kwargs)
                    if not isinstance(candidate, VerificationResult):
                        raise SecretaryOutputError(
                            "verifier returned an invalid result"
                        )
                    if candidate.source_exact != established_payload_facts[
                        "source_exact"
                    ] or (
                        candidate.reason_kind == VERIFICATION_REASON_PAYLOAD
                        and established_payload_facts["source_exact"]
                    ):
                        raise SecretaryOutputError(
                            "verifier payload check or reason contradicts established "
                            "exact payload evidence facts"
                        )
                    if not candidate.internally_consistent:
                        raise SecretaryOutputError(
                            "verifier approval contradicts its structured checks"
                        )
                    expected_target = (
                        proposed.target.as_dict() if proposed.target else None
                    )
                    agrees = (
                        candidate.action_kind == proposed.action_kind
                        and candidate.target == expected_target
                        and candidate.message == proposed.message
                        and candidate.utterance_digest == utterance_digest
                        and candidate.catalog_digest == catalog_fingerprint
                        and candidate.plan_digest == plan_digest
                    )
                    if not agrees:
                        raise SecretaryOutputError(
                            "verifier echoed different action, target, content, or digests"
                        )
                    verification = candidate
                    break
                except SecretaryOutputError as exc:
                    verification_error = exc
                    correction = {
                        "attempt": attempt + 1,
                        "failure": str(exc),
                        "instruction": (
                            "Make a new independent verification. Echo immutable fields "
                            "exactly, preserve established exact payload facts, and keep "
                            "approved and reason_kind consistent with all checks."
                        ),
                    }
                    if attempt < LLM_CORRECTIVE_ATTEMPTS:
                        self._record_activity(
                            "verifier_retry",
                            attempt=attempt + 1,
                            reason=str(exc),
                            sent=False,
                        )
                except Exception as exc:  # noqa: BLE001
                    verification_error = exc
                    break
            if verification is None:
                self._withhold(
                    plan,
                    transcript=transcript,
                    code="verifier_error",
                    reason=(
                        f"Independent verification remained invalid: "
                        f"{verification_error}. Nothing was sent."
                    ),
                )
                return None
            self._record_activity(
                "verification",
                approved=verification.approved,
                action=verification.action_kind,
                target=verification.target,
                message=verification.message,
                reason=verification.reason,
                utterance_digest=verification.utterance_digest,
                catalog_digest=verification.catalog_digest,
                plan_digest=verification.plan_digest,
            )
            if not verification.approved:
                self._withhold(
                    plan,
                    transcript=transcript,
                    code="verifier_rejected",
                    reason=(
                        f"Verifier rejected the proposed delivery: {verification.reason}. "
                        "Nothing was sent."
                    ),
                )
                return None
            capabilities.append(
                self._mint_prompt_capability(
                    utterance_id=utterance_id,
                    utterance_digest=utterance_digest,
                    catalog_digest=catalog_fingerprint,
                    plan_digest=plan_digest,
                    plan=proposed,
                    source_bundle=source_bundle,
                    listener_generation=listener_generation,
                    origin=origin,
                    global_generation=global_generation,
                )
            )
        return tuple(capabilities)

    def _mint_prompt_capability(
        self,
        *,
        utterance_id: str,
        utterance_digest: str,
        catalog_digest: str,
        plan_digest: str,
        plan: IntentPlan,
        source_bundle: dict[str, str],
        listener_generation: int,
        origin: str,
        global_generation: int,
    ) -> VerifiedPromptCapability:
        assert (
            plan.target
            and plan.target.pane_id
            and plan.message
            and plan.evidence
            and plan.workspace_evidence
        )
        source_bundle_digest = canonical_digest(source_bundle)
        values = (
            utterance_id,
            utterance_digest,
            catalog_digest,
            plan_digest,
            plan.target.workspace_id,
            plan.target.pane_id,
            plan.message,
            plan.evidence.source,
            plan.evidence.quote,
            source_bundle_digest,
            plan.workspace_evidence.source,
            plan.workspace_evidence.quote,
            (
                str(plan.workspace_evidence.catalog_number)
                if plan.workspace_evidence.catalog_number is not None
                else ""
            ),
            plan.agent_evidence.source if plan.agent_evidence else "",
            plan.agent_evidence.quote if plan.agent_evidence else "",
            self._target_binding_digest(plan, source_bundle),
            origin,
            str(listener_generation),
            str(global_generation),
        )
        key = getattr(self, "_capability_key", None)
        if key is None:
            key = os.urandom(32)
            self._capability_key = key
        seal = hmac.new(
            key, self._canonical_capability_bytes(values), hashlib.sha256
        ).hexdigest()
        return VerifiedPromptCapability(
            *values[:-3],
            origin,
            listener_generation,
            global_generation,
            seal,
        )

    @staticmethod
    def _canonical_capability_bytes(values: tuple[str, ...]) -> bytes:
        """JSON-encode capability fields with length-safe separators."""

        return json.dumps(
            {"version": 1, "fields": list(values)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _preflight_plan_shape(
        self,
        plan: IntentPlan,
        catalog: list[dict[str, Any]],
        state: dict[str, Any],
        *,
        source_text: str = "",
        source_bundle: dict[str, str] | None = None,
    ) -> str | None:
        plans = (plan, *plan.actions) if plan.action_kind == "batch" else (plan,)
        if sum(child.action_kind == "agent_prompt" for child in plans) > 1:
            return "Multiple agent prompts cannot be committed atomically and are disabled."
        minimum = self.config.llm_min_confidence
        for child in plans:
            if child.action_kind == "talk_policy" and child.target is not None:
                return "Voice talk-policy actions do not support scoped targets."
            if child.confidence < minimum:
                return (
                    f"Planner confidence {child.confidence:.2f} is below the "
                    f"required {minimum:.2f}; please clarify."
                )
            target_problem = self._validate_catalog_target(child, catalog)
            if target_problem:
                return target_problem
            evidence_sources = source_bundle or {"post_wake_content": source_text}
            evidence_problem = self._validate_target_evidence(
                child, catalog, evidence_sources
            )
            if evidence_problem:
                return evidence_problem
            if child.action_kind == "agent_prompt" and not child.evidence:
                return "agent_prompt has no exact transcript evidence."
            if child.action_kind == "agent_prompt":
                payload_facts = payload_evidence_facts(child, evidence_sources)
                if not payload_facts["message_equals_evidence_quote"]:
                    return "Prompt message does not exactly equal its evidence quote."
                if not payload_facts["evidence_source_present"]:
                    return "Prompt evidence names an unavailable transcript source."
                if not payload_facts["evidence_quote_is_unique_contiguous"]:
                    return (
                        "Prompt evidence is not one unique exact contiguous transcript "
                        "excerpt."
                    )
            if (
                child.action_kind
                in {"dictation_start", "dictation_append", "dictation_finish"}
                and child.message
                and source_text.count(child.message) != 1
            ):
                return (
                    "The proposed dictation content is not one exact post-wake "
                    "transcript excerpt."
                )
        phase = str(state.get("resume_phase") or state.get("phase") or "idle")
        if phase == "dictation_capture":
            allowed = {
                "dictation_append",
                "dictation_finish",
                "dictation_cancel",
                "clarification",
                "no_action",
            }
            if plan.action_kind not in allowed:
                return (
                    "The proposed action is invalid while dictation capture is active."
                )
        elif phase == "dictation_ready" and plan.action_kind.startswith("dictation_"):
            return "The completed dictation requires a final non-capture action."
        elif phase not in {
            "dictation_capture",
            "dictation_ready",
        } and plan.action_kind in {
            "dictation_append",
            "dictation_finish",
            "dictation_cancel",
        }:
            return "The proposed dictation transition does not match capture state."
        return None

    @staticmethod
    def _evidence_exists(evidence: IntentEvidence, sources: dict[str, str]) -> bool:
        source = sources.get(evidence.source)
        return isinstance(source, str) and source.count(evidence.quote) == 1

    @staticmethod
    def _identity_text(value: Any) -> str:
        return " ".join(str(value or "").casefold().split()).strip(" ,.:;!?—-")

    @staticmethod
    def _number_forms(value: Any) -> set[str]:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return set()
        words = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
            19: "nineteen",
            20: "twenty",
        }
        ordinals = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
            7: "seventh",
            8: "eighth",
            9: "ninth",
            10: "tenth",
            11: "eleventh",
            12: "twelfth",
            13: "thirteenth",
            14: "fourteenth",
            15: "fifteenth",
            16: "sixteenth",
            17: "seventeenth",
            18: "eighteenth",
            19: "nineteenth",
            20: "twentieth",
        }
        bases = {str(number), f"#{number}"}
        if number in words:
            bases.add(words[number])
        if number in ordinals:
            bases.add(ordinals[number])
            suffix = (
                "th"
                if 10 <= number % 100 <= 20
                else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
            )
            bases.add(f"{number}{suffix}")
        return bases | {
            f"{prefix} {base}"
            for prefix in ("workspace", "space", "index", "number")
            for base in tuple(bases)
        }

    @classmethod
    def _workspace_identity_forms(cls, row: dict[str, Any]) -> set[str]:
        forms = cls._number_forms(row.get("number"))
        for value in (
            row.get("label"),
            row.get("base"),
            *(row.get("nicknames") or []),
        ):
            normalized = cls._identity_text(value)
            if normalized:
                forms.add(normalized)
        return forms

    @classmethod
    def _validate_target_evidence(
        cls,
        plan: IntentPlan,
        catalog: list[dict[str, Any]],
        sources: dict[str, str],
    ) -> str | None:
        if plan.action_kind not in {"agent_prompt", "status"} or not plan.target:
            return None
        if not plan.workspace_evidence or not cls._evidence_exists(
            plan.workspace_evidence, sources
        ):
            return "Selected-workspace evidence is not one exact named-source excerpt."
        workspace_quote = cls._identity_text(plan.workspace_evidence.quote)
        quote_matches = [
            row
            for row in catalog
            if workspace_quote in cls._workspace_identity_forms(row)
        ]
        catalog_number = plan.workspace_evidence.catalog_number
        if catalog_number is None:
            workspace_matches = quote_matches
        else:
            workspace_matches = []
            for row in catalog:
                try:
                    row_number = int(row.get("number"))
                except (TypeError, ValueError):
                    continue
                if row_number == catalog_number:
                    workspace_matches.append(row)
            if len(workspace_matches) == 1 and any(
                str(row.get("workspace_id") or "")
                != str(workspace_matches[0].get("workspace_id") or "")
                for row in quote_matches
            ):
                return (
                    "Selected-workspace quote conflicts with its declared frozen "
                    "catalog number."
                )
        if len(workspace_matches) != 1:
            return (
                "Selected-workspace evidence does not uniquely identify one catalog "
                "workspace."
            )
        if (
            str(workspace_matches[0].get("workspace_id") or "")
            != plan.target.workspace_id
        ):
            return (
                "Selected-workspace evidence identifies a different catalog workspace."
            )
        workspace = next(
            (
                row
                for row in catalog
                if str(row.get("workspace_id") or "") == plan.target.workspace_id
            ),
            None,
        )
        if not workspace:
            return None
        if plan.action_kind == "agent_prompt" and plan.evidence:
            payload_identity = plan.evidence.quote.casefold()
            for routing in (plan.workspace_evidence, plan.agent_evidence):
                if not routing:
                    continue
                routing_identity = routing.quote.casefold()
                if (
                    routing_identity in payload_identity
                    or payload_identity in routing_identity
                ):
                    return "Routing evidence overlaps prompt payload evidence."
        if (
            plan.action_kind == "agent_prompt"
            and plan.evidence
            and plan.workspace_evidence.source == plan.evidence.source
        ):
            source = sources[plan.evidence.source]
            payload_start = source.index(plan.evidence.quote)
            route_start = source.index(plan.workspace_evidence.quote)
            if max(payload_start, route_start) < min(
                payload_start + len(plan.evidence.quote),
                route_start + len(plan.workspace_evidence.quote),
            ):
                return "Routing evidence overlaps prompt payload evidence."
        agents = [
            agent
            for agent in workspace.get("agents") or []
            if str(agent.get("pane_id") or "")
        ]
        if len(agents) == 1:
            # This does not choose a default pane: the planner still had to name
            # the one exact pane in the frozen row. Redundant agent evidence is
            # retained for audit, but it cannot make that unique binding ambiguous.
            if str(agents[0].get("pane_id") or "") != plan.target.pane_id:
                return (
                    "Selected sole-agent pane differs from the exact frozen catalog "
                    "pane."
                )
            if plan.agent_evidence and not cls._evidence_exists(
                plan.agent_evidence, sources
            ):
                return "Redundant sole-agent evidence is not an exact named-source excerpt."
            return None
        selected = next(
            (
                agent
                for agent in agents
                if str(agent.get("pane_id") or "") == plan.target.pane_id
            ),
            None,
        )
        if not selected:
            return "Selected agent pane is absent from the frozen catalog workspace."
        if plan.action_kind == "status" and not plan.agent_evidence:
            # Workspace-only status expands server-side to the parent plus linked
            # worktrees; the pane_id is only a catalog anchor, not a sole target.
            return None
        if not plan.agent_evidence:
            return (
                "The chosen workspace has multiple agents, but the plan has no "
                "explicit source evidence for the selected agent."
            )
        if not cls._evidence_exists(plan.agent_evidence, sources):
            return "Selected-agent evidence is not one exact post-wake excerpt."
        normalized_evidence = cls._identity_text(plan.agent_evidence.quote)
        matching_agents = [
            agent
            for agent in agents
            if normalized_evidence
            in {
                cls._identity_text(agent.get("name")),
                cls._identity_text(agent.get("title")),
            }
            - {""}
        ]
        if len(matching_agents) != 1:
            return (
                "Selected-agent evidence does not uniquely identify one catalog pane."
            )
        if str(matching_agents[0].get("pane_id") or "") != plan.target.pane_id:
            return "Selected-agent evidence identifies a different catalog pane."
        if plan.action_kind == "agent_prompt" and plan.evidence:
            for routing in (plan.agent_evidence,):
                if not routing or routing.source != plan.evidence.source:
                    continue
                source = sources[routing.source]
                payload_start = source.index(plan.evidence.quote)
                route_start = source.index(routing.quote)
                payload_span = range(
                    payload_start, payload_start + len(plan.evidence.quote)
                )
                route_span = range(route_start, route_start + len(routing.quote))
                if max(payload_span.start, route_span.start) < min(
                    payload_span.stop, route_span.stop
                ):
                    return "Routing evidence overlaps prompt payload evidence."
        return None

    @classmethod
    def _target_binding_digest(cls, plan: IntentPlan, sources: dict[str, str]) -> str:
        assert plan.target and plan.workspace_evidence
        return canonical_digest(
            {
                "target": plan.target.as_dict(),
                "workspace_evidence": plan.workspace_evidence.as_dict(),
                "agent_evidence": (
                    plan.agent_evidence.as_dict() if plan.agent_evidence else None
                ),
                "source_bundle_digest": canonical_digest(sources),
            }
        )

    @staticmethod
    def _validate_catalog_target(
        plan: IntentPlan, catalog: list[dict[str, Any]]
    ) -> str | None:
        if plan.action_kind not in {"agent_prompt", "status", "talk_policy"}:
            return None
        if not plan.target:
            if plan.action_kind == "talk_policy":
                return None
            return f"{plan.action_kind} has no exact catalog target."
        workspaces = [
            row
            for row in catalog
            if str(row.get("workspace_id") or "") == plan.target.workspace_id
        ]
        if len(workspaces) != 1:
            return "The selected workspace is missing or ambiguous in the live catalog."
        if plan.action_kind == "talk_policy":
            return None
        agents = [
            agent
            for agent in workspaces[0].get("agents") or []
            if str(agent.get("pane_id") or "") == plan.target.pane_id
        ]
        if len(agents) != 1:
            if not workspaces[0].get("agents"):
                label = str(workspaces[0].get("label") or plan.target.workspace_id)
                return (
                    f"Workspace {label!r} exists, but it has no live agent pane to "
                    "receive a prompt"
                )
            return (
                "The selected agent pane is missing or ambiguous in the live catalog."
            )
        if plan.action_kind == "agent_prompt":
            status = agents[0].get("status")
            if not isinstance(status, str) or status not in SAFE_PROMPT_AGENT_STATES:
                shown = status if isinstance(status, str) and status else "absent"
                return f"The selected agent pane has unsafe or unknown state {shown!r}."
        return None

    @staticmethod
    def _prompt_plans(plan: IntentPlan) -> tuple[IntentPlan, ...]:
        plans = plan.actions if plan.action_kind == "batch" else (plan,)
        return tuple(child for child in plans if child.action_kind == "agent_prompt")

    @staticmethod
    def _catalog_fingerprint(catalog: list[dict[str, Any]]) -> str:
        identity = []
        for row in catalog:
            identity.append(
                {
                    "workspace_id": row.get("workspace_id"),
                    "number": row.get("number"),
                    "label": row.get("label"),
                    "base": row.get("base"),
                    "nicknames": row.get("nicknames") or [],
                    "agents": [
                        {
                            "pane_id": agent.get("pane_id"),
                            "name": agent.get("name"),
                            "title": agent.get("title"),
                            "status": agent.get("status"),
                        }
                        for agent in row.get("agents") or []
                    ],
                }
            )
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _withhold(
        self,
        plan: IntentPlan | None,
        *,
        transcript: str,
        code: str,
        reason: str,
        pending: dict[str, Any] | None = None,
        error: bool = False,
        notify: bool = True,
    ) -> None:
        if pending:
            current = getattr(self, "pending_clarification", None)
            if current is None:
                if "request_text" not in pending:
                    pending["request_text"] = str(pending.pop("message", ""))
                if "request_raw_transcript" not in pending:
                    pending["request_raw_transcript"] = str(
                        pending.pop("raw_transcript", transcript)
                    )
                pending.setdefault("unresolved_slots", ["action"])
                self.pending_clarification = self._stamp_pending(pending)
            else:
                # A failed child may not overwrite or restart the pre-existing
                # clarification transaction. It remains bounded by its old expiry.
                pending = None
        blocked_kind = (
            "error" if error else "clarification" if pending else "verification_blocked"
        )
        action = {
            "action_kind": blocked_kind,
            "chosen_action": plan.action_kind if plan else None,
            "target": plan.target.as_dict() if plan and plan.target else None,
            "mode": plan.mode if plan else None,
            "message": reason,
            "transcript": transcript,
            "result": {
                "ok": False,
                "sent": False,
                "withheld": True,
                "code": code,
                "message": reason,
            },
        }
        self._record_activity(
            "withheld",
            code=code,
            chosen_action=action["chosen_action"],
            target=action["target"],
            reason=reason,
            sent=False,
        )
        if notify:
            self._notify("voicerdr — not sent", reason, sound="request", speak=True)
        self._commit_voice_action(action)

    def _execute_plan(
        self,
        plan: IntentPlan,
        *,
        transcript: str,
        raw_transcript: str,
        input_mode: str,
        state: dict[str, Any],
        catalog: list[dict[str, Any]],
        catalog_fingerprint: str,
        utterance_id: str,
        utterance_digest: str,
        activation_phrase: str | None,
        capabilities: tuple[VerifiedPromptCapability, ...],
        source_bundle: dict[str, str],
        listener_generation: int,
        origin: str = "voice",
        global_generation: int = 0,
    ) -> None:
        action = {
            **plan.as_dict(),
            "transcript": transcript,
            "utterance_id": utterance_id,
            "utterance_digest": utterance_digest,
            "catalog_digest": catalog_fingerprint,
        }
        kind = plan.action_kind
        if kind == "dictation_start":
            if not self.config.dictation_enabled:
                self._withhold(
                    plan,
                    transcript=transcript,
                    code="dictation_disabled",
                    reason="Dictation is disabled; nothing was sent.",
                )
                return
            self.dictation.start(plan.message or "", raw_transcript=raw_transcript)
            action["text"] = self.dictation.joined()
            action["result"] = {"ok": True, "sent": False, "mode": "dictation"}
            self._record_activity("dictation_started", message=plan.message, sent=False)
        elif kind == "dictation_append":
            self.dictation.append(plan.message or "", raw_transcript=transcript)
            action["text"] = self.dictation.joined()
            action["result"] = {"ok": True, "sent": False, "mode": "dictation"}
            self._record_activity(
                "dictation_appended", message=plan.message, sent=False
            )
        elif kind == "dictation_finish":
            if plan.message:
                self.dictation.append(plan.message, raw_transcript=transcript)
            else:
                self.dictation.record_raw(transcript)
            complete = self.dictation.joined()
            source = self.dictation.raw_joined() or raw_transcript
            self._record_activity("dictation_finished", message=complete, sent=False)
            if not complete:
                self._withhold(
                    plan,
                    transcript=transcript,
                    code="empty_dictation",
                    reason="The completed dictation was empty; nothing was sent.",
                )
                return
            self._dispatch_utterance(
                complete,
                transcript=transcript,
                raw_transcript=source,
                utterance_id=utterance_id,
                activation_phrase=activation_phrase,
                input_mode="dictation",
                state={
                    "phase": "dictation_ready",
                    "complete_buffer": complete,
                    "complete_fragments": list(self.dictation.parts),
                    "raw_fragments": list(self.dictation.raw_transcripts),
                    "clarification_id": state.get("clarification_id"),
                },
                listener_generation=listener_generation,
                origin=origin,
                global_generation=global_generation,
            )
            final_result = (
                self.last_voice_action.get("result")
                if isinstance(self.last_voice_action, dict)
                else None
            )
            if isinstance(final_result, dict) and final_result.get("ok"):
                self.dictation.clear()
            return
        elif kind == "dictation_cancel":
            self.dictation.clear()
            action["result"] = {"ok": True, "sent": False, "mode": "idle"}
            self._notify(
                "voicerdr", "Dictation cancelled. Nothing was sent.", speak=True
            )
        elif kind == "clarification":
            question = plan.clarification or "Please clarify."
            transaction = state.get("pending_transaction")
            existing = self.pending_clarification
            if (
                isinstance(transaction, dict)
                and existing
                and existing.get("clarification_id")
                == transaction.get("clarification_id")
            ):
                pending = dict(existing)
                pending["question"] = question
                pending["previous_plan"] = plan.as_dict()
                pending["unresolved_slots"] = list(plan.unresolved_slots)
                pending["followups"] = list(transaction.get("followups") or [])
            else:
                request_text = str(
                    state.get("complete_buffer")
                    or state.get("source_content")
                    or (
                        self.dictation.joined()
                        if state.get("phase") == "dictation_capture"
                        else ""
                    )
                )
                pending = {
                    "request_text": request_text,
                    "request_raw_transcript": raw_transcript,
                    "question": question,
                    "unresolved_slots": list(plan.unresolved_slots),
                    "followups": [],
                    "previous_plan": plan.as_dict(),
                    "input_mode": input_mode,
                    "resume_phase": (
                        state.get("phase")
                        if state.get("phase")
                        in {"dictation_capture", "dictation_ready"}
                        else None
                    ),
                }
            self.pending_clarification = self._stamp_pending(pending)
            action["message"] = question
            action["result"] = {
                "ok": False,
                "sent": False,
                "code": "clarification_required",
                "message": question,
            }
            self._record_activity(
                "clarification", question=question, reason=plan.reason, sent=False
            )
            self._notify("voicerdr — not sent", question, sound="request", speak=True)
        elif kind == "no_action":
            message = f"No action taken: {plan.reason}. Nothing was sent."
            action["result"] = {
                "ok": False,
                "sent": False,
                "code": "no_action",
                "message": message,
            }
            self._record_activity("no_action", reason=plan.reason, sent=False)
            self._notify("voicerdr — not sent", message, sound="none", speak=False)
        elif kind == "agent_prompt":
            result = self._deliver_verified_prompt(
                capabilities[0],
                catalog=catalog,
                source_bundle=source_bundle,
            )
            action["result"] = result
            if not result.get("ok"):
                if result.get("code") == "unsupported_message_rewrite":
                    self._retain_prompt_confirmation(
                        plan,
                        raw_transcript=raw_transcript,
                        input_mode=input_mode,
                        state=state,
                        question=str(result.get("message") or "Please clarify."),
                    )
                self._report_delivery_withheld(plan, result)
        elif kind == "status":
            assert plan.target and plan.target.pane_id
            if plan.agent_evidence:
                parts = [self.summarize_agent(plan.target.pane_id)]
            else:
                parts = self.summarize_workspace_group(plan.target.workspace_id)
            summary = self._speak_status_summaries(parts)
            action["result"] = {
                "ok": True,
                "sent": False,
                "summary": summary,
                "parts": parts,
                "target": plan.target.as_dict(),
            }
            self.last_summary = summary
        elif kind == "fleet_status":
            result = self.fleet_status()
            result["sent"] = False
            action["result"] = result
            self.last_summary = str(result["summary"])
            self._notify("voicerdr", str(result["summary"]), speak=True)
        elif kind == "talk_policy":
            if plan.target:
                raise IntentValidationError(
                    "voice talk_policy does not support workspace-scoped modes"
                )
            result = self.set_talk_policy(mode=plan.mode)
            result["sent"] = False
            action["result"] = result
            self._notify("voicerdr", str(result.get("summary")), speak=True)
        elif kind == "control":
            self.dictation.clear()
            try:
                if self._listener_callback_is_current():
                    self._start_spoken_mute_completion(
                        action,
                        transcript=transcript,
                        state=state,
                    )
                    return
                self._mute_voice_durably()
            except MicModeFrozenError as exc:
                self._withhold(
                    plan,
                    transcript=transcript,
                    code="delivery_closed",
                    reason=f"{exc}. Nothing was sent.",
                    error=True,
                    notify=False,
                )
                return
            except (MicPreferenceError, MicClosureError) as exc:
                self._withhold(
                    plan,
                    transcript=transcript,
                    code=(
                        "mic_closure"
                        if isinstance(exc, MicClosureError)
                        else "mic_preference"
                    ),
                    reason=(
                        f"Microphone mute failed closed without a success result: {exc}"
                    ),
                    error=True,
                )
                return
            action["result"] = {"ok": True, "sent": False, "mode": "mute"}
            self._notify("voicerdr", "Muted.", speak=True)
        elif kind == "batch":
            # Evaluate read-only children before any prompt delivery.
            child_results: list[dict[str, Any] | None] = [None] * len(plan.actions)
            for index, child in enumerate(plan.actions):
                if child.action_kind == "status":
                    assert child.target and child.target.pane_id
                    if child.agent_evidence:
                        parts = [self.summarize_agent(child.target.pane_id)]
                    else:
                        parts = self.summarize_workspace_group(
                            child.target.workspace_id
                        )
                    summary = " ".join(parts)
                    child_results[index] = {
                        "ok": True,
                        "sent": False,
                        "summary": summary,
                        "parts": parts,
                    }
                elif child.action_kind == "fleet_status":
                    result = self.fleet_status()
                    result["sent"] = False
                    child_results[index] = result
            capability_index = 0
            for index, child in enumerate(plan.actions):
                if child.action_kind == "agent_prompt":
                    result = self._deliver_verified_prompt(
                        capabilities[capability_index],
                        catalog=catalog,
                        source_bundle=source_bundle,
                        notify=False,
                    )
                    capability_index += 1
                    if not result.get("ok"):
                        if result.get("code") == "unsupported_message_rewrite":
                            self._retain_prompt_confirmation(
                                child,
                                raw_transcript=raw_transcript,
                                input_mode=input_mode,
                                state=state,
                                question=str(
                                    result.get("message") or "Please clarify."
                                ),
                            )
                        self._report_delivery_withheld(child, result, notify=False)
                    child_results[index] = result
            results = [
                {"action": child.as_dict(), "result": result}
                for child, result in zip(plan.actions, child_results, strict=True)
                if result is not None
            ]
            succeeded = all(result["result"].get("ok") for result in results)
            sent_values = [result["result"].get("sent") for result in results]
            sent: bool | None = (
                True if True in sent_values else None if None in sent_values else False
            )
            summary = (
                f"Completed {len(results)} planned actions."
                if succeeded
                else "A delivery outcome is unknown; inspect activity before retrying."
                if sent is None
                else "A planned batch action failed before prompt delivery."
            )
            summary_parts = [
                str(part)
                for item in results
                if item["action"]["action_kind"] in {"status", "fleet_status"}
                for part in (
                    item["result"].get("parts") or [item["result"].get("summary")]
                )
                if part
            ]
            if summary_parts:
                summary_parts.append(summary)
                summary = " ".join(summary_parts)
            action["result"] = {
                "ok": succeeded,
                "sent": sent,
                "summary": summary,
                "results": results,
            }
            try:
                if summary_parts:
                    self.last_summary = self._speak_status_summaries(summary_parts)
                else:
                    self._notify(
                        "voicerdr", summary, sound="none" if succeeded else "request"
                    )
            except Exception:
                if sent is True:
                    log.exception("post-send batch notification failed")
                else:
                    raise
        result = action.get("result")
        if isinstance(result, dict) and result.get("ok"):
            if (state.get("resume_phase") or state.get("phase")) == "dictation_ready":
                self.dictation.clear()
            self._clear_pending_for_state(state)
        self._commit_voice_action(action)

    def _retain_prompt_confirmation(
        self,
        plan: IntentPlan,
        *,
        raw_transcript: str,
        input_mode: str,
        state: dict[str, Any],
        question: str,
    ) -> None:
        self.pending_clarification = self._stamp_pending(
            {
                "request_text": str(state.get("source_content") or ""),
                "request_raw_transcript": raw_transcript,
                "question": question,
                "unresolved_slots": ["confirmation"],
                "previous_plan": plan.as_dict(),
                "input_mode": input_mode,
                "resume_phase": (
                    "dictation_capture"
                    if state.get("phase") == "dictation_capture"
                    else None
                ),
            }
        )

    def _report_delivery_withheld(
        self,
        plan: IntentPlan,
        result: dict[str, Any],
        *,
        notify: bool = True,
    ) -> None:
        message = str(
            result.get("message") or "Delivery was withheld; nothing was sent."
        )
        unknown = result.get("sent") is None
        self._record_activity(
            "delivery_unknown" if unknown else "withheld",
            code=result.get("code") or "delivery_blocked",
            chosen_action=plan.action_kind,
            target=plan.target.as_dict() if plan.target else None,
            reason=message,
            sent=None if unknown else False,
        )
        if notify:
            title = "voicerdr — delivery unknown" if unknown else "voicerdr — not sent"
            self._notify(title, message, sound="request", speak=True)

    def _deliver_verified_prompt(
        self,
        capability: VerifiedPromptCapability,
        *,
        catalog: list[dict[str, Any]],
        source_bundle: dict[str, str],
        notify: bool = True,
    ) -> dict[str, Any]:
        """Deliver one verified prompt after fresh catalog and seal checks."""

        values = (
            capability.utterance_id,
            capability.utterance_digest,
            capability.catalog_digest,
            capability.plan_digest,
            capability.workspace_id,
            capability.pane_id,
            capability.message,
            capability.evidence_source,
            capability.evidence_quote,
            capability.source_bundle_digest,
            capability.workspace_evidence_source,
            capability.workspace_evidence_quote,
            capability.workspace_catalog_number,
            capability.agent_evidence_source,
            capability.agent_evidence_quote,
            capability.target_binding_digest,
            capability.origin,
            str(capability.origin_generation),
            str(capability.global_generation),
        )
        key = getattr(self, "_capability_key", b"")
        expected = hmac.new(
            key, self._canonical_capability_bytes(values), hashlib.sha256
        ).hexdigest()
        if not key or not hmac.compare_digest(expected, capability.seal):
            return {
                "ok": False,
                "sent": False,
                "code": "invalid_verified_capability",
                "message": "Prompt capability was invalid; nothing was sent.",
            }
        spent = getattr(self, "_spent_capability_seals", None)
        if spent is None:
            spent = set()
            self._spent_capability_seals = spent
        if capability.seal in spent:
            return {
                "ok": False,
                "sent": False,
                "code": "capability_consumed",
                "message": "Prompt capability was already consumed; nothing was sent.",
            }
        if canonical_digest(source_bundle) != capability.source_bundle_digest:
            return {
                "ok": False,
                "sent": False,
                "code": "source_evidence_changed",
                "message": "Transcript evidence changed before delivery; nothing was sent.",
            }
        source = source_bundle.get(capability.evidence_source)
        quote = capability.evidence_quote
        if (
            not isinstance(source, str)
            or not quote
            or source.count(quote) != 1
            or capability.message != quote
        ):
            return {
                "ok": False,
                "sent": False,
                "code": "unsupported_message_rewrite",
                "message": (
                    "The proposed prompt was not one exact, uniquely identified "
                    "transcript excerpt. Nothing was sent; explicitly confirm the "
                    "desired wording."
                ),
            }
        delivered_message = quote
        current = self._space_directory()
        if self._catalog_fingerprint(current) != capability.catalog_digest:
            return {
                "ok": False,
                "sent": False,
                "code": "catalog_changed",
                "message": "Catalog changed before delivery; nothing was sent.",
            }
        plan = IntentPlan(
            "agent_prompt",
            IntentTarget(capability.workspace_id, capability.pane_id),
            delivered_message,
            None,
            None,
            1.0,
            "verified capability",
            evidence=IntentEvidence(capability.evidence_source, quote),
            workspace_evidence=IntentEvidence(
                capability.workspace_evidence_source,
                capability.workspace_evidence_quote,
                (
                    int(capability.workspace_catalog_number)
                    if capability.workspace_catalog_number
                    else None
                ),
            ),
            agent_evidence=(
                IntentEvidence(
                    capability.agent_evidence_source,
                    capability.agent_evidence_quote,
                )
                if capability.agent_evidence_source
                else None
            ),
        )
        evidence_problem = self._validate_target_evidence(plan, current, source_bundle)
        binding_changed = (
            self._target_binding_digest(plan, source_bundle)
            != capability.target_binding_digest
        )
        if evidence_problem or binding_changed:
            return {
                "ok": False,
                "sent": False,
                "code": "target_evidence_changed",
                "message": (
                    f"Target evidence failed sealed revalidation: {evidence_problem or 'binding changed'}. "
                    "Nothing was sent."
                ),
            }
        if self._validate_catalog_target(plan, current):
            return {
                "ok": False,
                "sent": False,
                "code": "target_missing",
                "message": "Exact target is no longer available; nothing was sent.",
            }
        matching = [
            agent
            for agent in self.herdr.agent_list()
            if str(agent.get("workspace_id") or "") == plan.target.workspace_id
            and str(agent.get("pane_id") or "") == capability.pane_id
        ]
        if len(matching) != 1:
            return {
                "ok": False,
                "sent": False,
                "code": "target_missing",
                "message": "Exact target is no longer available; nothing was sent.",
            }
        agent = matching[0]
        status = agent.get("agent_status")
        if not isinstance(status, str) or status not in SAFE_PROMPT_AGENT_STATES:
            return {
                "ok": False,
                "sent": False,
                "code": "unsafe_agent_state",
                "message": "Exact target state is not explicitly safe; nothing was sent.",
            }
        self._set_activity_status(
            phase="delivering",
            waiting_for=f"exact pane {capability.pane_id}",
            chosen_action="agent_prompt",
            chosen_target={
                "workspace_id": capability.workspace_id,
                "pane_id": capability.pane_id,
            },
        )
        self._record_activity(
            "prompt_requested",
            target={
                "workspace_id": capability.workspace_id,
                "pane_id": capability.pane_id,
            },
            message=delivered_message,
            catalog_fingerprint=capability.catalog_digest,
            utterance_digest=capability.utterance_digest,
            plan_digest=capability.plan_digest,
        )
        final_catalog = self._space_directory()
        if self._catalog_fingerprint(final_catalog) != capability.catalog_digest:
            return {
                "ok": False,
                "sent": False,
                "code": "catalog_changed",
                "message": "Catalog changed at the delivery boundary; nothing was sent.",
            }
        final_evidence_problem = self._validate_target_evidence(
            plan, final_catalog, source_bundle
        )
        final_target_problem = self._validate_catalog_target(plan, final_catalog)
        if final_evidence_problem or final_target_problem:
            return {
                "ok": False,
                "sent": False,
                "code": "target_revalidation_failed",
                "message": (
                    f"Target failed final sealed revalidation: "
                    f"{final_evidence_problem or final_target_problem}. Nothing was sent."
                ),
            }
        authority_lock = getattr(self, "_authority_lock", None)
        if authority_lock is None:
            authority_lock = threading.RLock()
            self._authority_lock = authority_lock
        with authority_lock:
            if getattr(self, "_delivery_closed", False):
                return {
                    "ok": False,
                    "sent": False,
                    "code": "delivery_closed",
                    "message": "Daemon shutdown has begun; nothing was sent.",
                }
            current_origin, current_global = self._current_authority(capability.origin)
            if (
                capability.origin_generation != current_origin
                or capability.global_generation != current_global
            ):
                return {
                    "ok": False,
                    "sent": False,
                    "code": (
                        "listener_authority_revoked"
                        if capability.origin == "voice"
                        else "command_authority_revoked"
                    ),
                    "message": (
                        "Listener authority was revoked; nothing was sent."
                        if capability.origin == "voice"
                        else "Command authority was revoked; nothing was sent."
                    ),
                }
            if capability.seal in spent:
                return {
                    "ok": False,
                    "sent": False,
                    "code": "capability_consumed",
                    "message": "Prompt capability was already consumed; nothing was sent.",
                }
            spent.add(capability.seal)
            try:
                self.herdr.agent_prompt(
                    capability.pane_id, delivered_message, wait=False
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "sent": None,
                    "code": "delivery_outcome_unknown",
                    "message": (
                        f"Herdr did not acknowledge delivery ({exc}); delivery outcome is "
                        "unknown, so it will not be retried automatically."
                    ),
                }
        try:
            self._record_activity(
                "prompt_sent",
                target={
                    "workspace_id": capability.workspace_id,
                    "pane_id": capability.pane_id,
                },
                message=delivered_message,
                catalog_fingerprint=capability.catalog_digest,
                utterance_digest=capability.utterance_digest,
                plan_digest=capability.plan_digest,
                sent=True,
            )
        except Exception:
            log.exception("post-send activity bookkeeping failed")
        try:
            self.policy.mark_prompted(capability.pane_id)
        except Exception:
            log.exception("post-send policy bookkeeping failed")
        if self.config.focus_on_prompt:
            try:
                self.herdr.workspace_focus(capability.workspace_id)
                self.focused_workspace_id = capability.workspace_id
            except Exception as exc:  # noqa: BLE001
                log.debug("workspace focus failed: %s", exc)
        label = next(
            (
                str(row.get("label") or capability.workspace_id)
                for row in catalog
                if str(row.get("workspace_id") or "") == capability.workspace_id
            ),
            capability.workspace_id,
        )
        title = agent.get("terminal_title_stripped") or capability.pane_id
        try:
            summary = self.policy.clamp_speech(f"Sent to {label} ({title}).")
        except Exception:
            log.exception("post-send acknowledgement formatting failed")
            summary = "Prompt sent to the verified target."
        self.last_summary = summary
        if notify:
            try:
                self._notify("voicerdr", summary, speak=self.config.speak_acks)
            except Exception:
                log.exception("post-send notification failed")
        return {
            "ok": True,
            "sent": True,
            "workspace_id": capability.workspace_id,
            "pane_id": capability.pane_id,
            "target": capability.pane_id,
            "summary": summary,
        }

    def _notify(
        self,
        title: str,
        body: str = "",
        *,
        sound: str = "none",
        speak: bool = False,
        urgent: bool = False,
    ) -> None:
        try:
            self.herdr.notification_show(title, body, sound=sound)
        except Exception as exc:  # noqa: BLE001
            log.debug("notification failed: %s", exc)
        if speak and body:
            try:
                self._speak(body, replace=urgent)
            except Exception:  # TTS acknowledgement must not alter action truth.
                log.exception("speech acknowledgement failed")

    def _speak(self, text: str, *, replace: bool = False) -> None:
        if not self.policy.speak_enabled:
            return
        clipped = self.policy.clamp_speech(text)
        self.speaker.enabled = True
        self.speaker.speak(clipped, replace=replace)

    def _warm_tts(self) -> None:
        try:
            self.speaker._ensure_engine()
        except Exception as exc:  # noqa: BLE001
            log.warning("TTS warm-up failed: %s", exc)
            self.speaker.last_error = str(exc)


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)


def _clean_progress_excerpt(text: str, *, max_chars: int = 6000) -> str:
    """Clean and bound recent pane output, retaining the freshest complete lines."""
    clean = _strip_ansi(text)
    clean = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", clean)
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", clean)
    lines = [" ".join(line.split()) for line in clean.splitlines()]
    lines = [line for line in lines if line]
    if not lines or max_chars <= 0:
        return ""

    kept: list[str] = []
    remaining = max_chars
    for line in reversed(lines):
        separator = 1 if kept else 0
        available = remaining - separator
        if available <= 0:
            break
        if len(line) > available:
            kept.append(line[-available:])
            break
        kept.append(line)
        remaining -= len(line) + separator
    return "\n".join(reversed(kept))


def run_daemon(paths: RuntimePaths, config: AppConfig) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return Daemon(paths, config).run_forever()
