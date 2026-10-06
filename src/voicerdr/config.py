import math
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from voicerdr.paths import RuntimePaths

DEFAULT_ASSISTANT_NAME = "Jenny"
DEFAULT_ASSISTANT_AVATAR = "woman"
ASSISTANT_AVATARS = frozenset({"woman", "none"})
DEFAULT_VAD_STOP_SECS = 1.3
MIN_VAD_STOP_SECS = 0.8
MAX_VAD_STOP_SECS = 3.0


def default_wake_phrases(assistant_name: str) -> list[str]:
    """Build natural wake forms for a newly configured assistant identity."""
    spoken_name = " ".join(assistant_name.split()).casefold()
    return [f"hey {spoken_name}", f"okay {spoken_name}", spoken_name]


_DEFAULT_WAKE_PHRASES = default_wake_phrases(DEFAULT_ASSISTANT_NAME)
_DEFAULT_DICTATION_CLOSERS = [
    "send it",
    "send that",
    "that's all",
    "thats all",
    "over",
    "go ahead",
]
_DEFAULT_DICTATION_CANCELS = [
    "never mind",
    "nevermind",
    "cancel that",
    "scratch that",
    "forget it",
]
_DEFAULT_DICTATION_STARTS = [
    "listen",
    "dictate",
    "dictates",
    "dictation",
    "take a message",
    "take a note",
    "take a memo",
]


@dataclass
class AppConfig:
    assistant_name: str = DEFAULT_ASSISTANT_NAME
    assistant_avatar: str = DEFAULT_ASSISTANT_AVATAR
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_api_key: str = "local"
    llm_model: str = "local-model"
    llm_timeout_secs: float = 15.0
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_min_confidence: float = 0.85
    llm_verify_ssl: bool = False
    verifier_base_url: str | None = None
    verifier_api_key: str | None = None
    verifier_model: str | None = None
    verifier_timeout_secs: float | None = None
    verifier_temperature: float | None = None
    verifier_max_tokens: int | None = None
    verifier_verify_ssl: bool | None = None
    mute_on_start: bool = False
    input_device_index: int | None = None
    sample_rate: int = 16000
    stt_model: str = "base"
    vad_stop_secs: float = DEFAULT_VAD_STOP_SECS
    wake_phrases: list[str] = field(default_factory=lambda: list(_DEFAULT_WAKE_PHRASES))
    require_wake: bool = True
    feedback_heard: bool = False
    interesting_statuses: list[str] = field(default_factory=lambda: ["blocked", "done"])
    talk_mode: str = "blocked_only"
    quiet_workspaces: list[str] = field(default_factory=list)
    min_interval_secs: float = 20.0
    prefer_titles: bool = True
    max_spoken_chars: int = 280
    speak_enabled: bool = True
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    speak_acks: bool = True
    subscribe_retry_secs: float = 2.0
    focus_on_prompt: bool = True
    # Prefix Herdr space labels with #N so the sidebar shows speakable numbers.
    show_space_numbers: bool = True
    activity_workspace_enabled: bool = True
    activity_workspace_label: str = "voicerdr activity"
    activity_history_lines: int = 250
    dictation_enabled: bool = True
    dictation_closing_phrases: list[str] = field(
        default_factory=lambda: list(_DEFAULT_DICTATION_CLOSERS)
    )
    dictation_cancel_phrases: list[str] = field(
        default_factory=lambda: list(_DEFAULT_DICTATION_CANCELS)
    )
    dictation_start_phrases: list[str] = field(
        default_factory=lambda: list(_DEFAULT_DICTATION_STARTS)
    )
    dictation_max_secs: float = 120.0
    clarification_max_secs: float = 45.0
    aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, paths: RuntimePaths, *, seed: bool = True) -> Self:
        if seed:
            seed_config_files(paths)
        data: dict[str, Any] = {}
        if paths.config_toml.is_file():
            with paths.config_toml.open("rb") as f:
                data = tomllib.load(f)
        # Optional untracked overlay next to the example (not published).
        local_config = paths.plugin_root / "config" / "config.local.toml"
        if local_config.is_file():
            with local_config.open("rb") as f:
                _deep_merge(data, tomllib.load(f))

        aliases = load_aliases(paths)

        assistant = data.get("assistant") or {}
        llm = data.get("llm") or {}
        verifier = data.get("verifier") or {}
        audio = data.get("audio") or {}
        wake = data.get("wake") or {}
        talk = data.get("talk") or {}
        herdr = data.get("herdr") or {}
        ui = data.get("ui") or {}
        dictation = data.get("dictation") or {}
        device = audio.get("input_device_index", None)
        if device is not None:
            device = int(device)
        assistant_name = (
            " ".join(str(assistant.get("name", DEFAULT_ASSISTANT_NAME)).split())
            or DEFAULT_ASSISTANT_NAME
        )
        assistant_avatar = (
            str(assistant.get("avatar", DEFAULT_ASSISTANT_AVATAR)).strip().lower()
        )
        if assistant_avatar not in ASSISTANT_AVATARS:
            assistant_avatar = DEFAULT_ASSISTANT_AVATAR
        phrases = wake.get("phrases", default_wake_phrases(assistant_name))
        if not isinstance(phrases, list):
            phrases = default_wake_phrases(assistant_name)
        closers = dictation.get("closing_phrases", _DEFAULT_DICTATION_CLOSERS)
        if not isinstance(closers, list):
            closers = list(_DEFAULT_DICTATION_CLOSERS)
        cancels = dictation.get("cancel_phrases", _DEFAULT_DICTATION_CANCELS)
        if not isinstance(cancels, list):
            cancels = list(_DEFAULT_DICTATION_CANCELS)
        starts = dictation.get("start_phrases", _DEFAULT_DICTATION_STARTS)
        if not isinstance(starts, list):
            starts = list(_DEFAULT_DICTATION_STARTS)
        return cls(
            assistant_name=assistant_name,
            assistant_avatar=assistant_avatar,
            llm_base_url=str(llm.get("base_url", cls.llm_base_url)),
            llm_api_key=str(llm.get("api_key", cls.llm_api_key)),
            llm_model=str(llm.get("model", cls.llm_model)),
            llm_timeout_secs=float(llm.get("timeout_secs", 15)),
            llm_temperature=float(llm.get("temperature", 0.0)),
            llm_max_tokens=int(llm.get("max_tokens", 2048)),
            llm_min_confidence=min(
                1.0, max(0.0, float(llm.get("min_confidence", 0.85)))
            ),
            llm_verify_ssl=bool(llm.get("verify_ssl", False)),
            verifier_base_url=(
                str(verifier["base_url"]) if verifier.get("base_url") else None
            ),
            verifier_api_key=(
                str(verifier["api_key"]) if verifier.get("api_key") else None
            ),
            verifier_model=(str(verifier["model"]) if verifier.get("model") else None),
            verifier_timeout_secs=(
                float(verifier["timeout_secs"])
                if verifier.get("timeout_secs") is not None
                else None
            ),
            verifier_temperature=(
                float(verifier["temperature"])
                if verifier.get("temperature") is not None
                else None
            ),
            verifier_max_tokens=(
                int(verifier["max_tokens"])
                if verifier.get("max_tokens") is not None
                else None
            ),
            verifier_verify_ssl=(
                bool(verifier["verify_ssl"])
                if verifier.get("verify_ssl") is not None
                else None
            ),
            mute_on_start=bool(audio.get("mute_on_start", False)),
            input_device_index=device,
            sample_rate=int(audio.get("sample_rate", 16000)),
            stt_model=str(audio.get("stt_model", "base")),
            vad_stop_secs=_bounded_float(
                audio.get("vad_stop_secs", DEFAULT_VAD_STOP_SECS),
                default=DEFAULT_VAD_STOP_SECS,
                minimum=MIN_VAD_STOP_SECS,
                maximum=MAX_VAD_STOP_SECS,
            ),
            wake_phrases=[str(p) for p in phrases],
            require_wake=bool(wake.get("require_wake", True)),
            feedback_heard=bool(wake.get("feedback_heard", False)),
            interesting_statuses=[
                str(s) for s in talk.get("interesting_statuses", ["blocked", "done"])
            ],
            talk_mode=str(talk.get("mode", "blocked_only")),
            quiet_workspaces=[str(s) for s in talk.get("quiet_workspaces", [])]
            if isinstance(talk.get("quiet_workspaces", []), list)
            else [],
            min_interval_secs=float(talk.get("min_interval_secs", 20)),
            prefer_titles=bool(talk.get("prefer_titles", True)),
            max_spoken_chars=int(talk.get("max_spoken_chars", 280)),
            speak_enabled=bool(talk.get("speak_enabled", True)),
            tts_voice=str(talk.get("tts_voice", "af_heart")),
            tts_speed=float(talk.get("tts_speed", 1.0)),
            speak_acks=bool(talk.get("speak_acks", True)),
            subscribe_retry_secs=float(herdr.get("subscribe_retry_secs", 2)),
            focus_on_prompt=bool(herdr.get("focus_on_prompt", True)),
            show_space_numbers=bool(ui.get("show_space_numbers", True)),
            activity_workspace_enabled=bool(ui.get("activity_workspace", True)),
            activity_workspace_label=str(
                ui.get("activity_workspace_label", "voicerdr activity")
            ),
            activity_history_lines=max(20, int(ui.get("activity_history_lines", 250))),
            dictation_enabled=bool(dictation.get("enabled", True)),
            dictation_closing_phrases=[str(p) for p in closers],
            dictation_cancel_phrases=[str(p) for p in cancels],
            dictation_start_phrases=[str(p) for p in starts],
            dictation_max_secs=float(dictation.get("max_secs", 120)),
            clarification_max_secs=float(dictation.get("clarification_max_secs", 45)),
            aliases=aliases,
        )


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return min(maximum, max(minimum, parsed))


def _load_aliases_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        raw = tomllib.load(f)
    section = raw.get("aliases") or {}
    out: dict[str, str] = {}
    for k, v in section.items():
        key = re.sub(r"\s+", " ", str(k).strip().lower())
        if key:
            out[key] = str(v).strip()
    return out


def load_aliases(paths: RuntimePaths) -> dict[str, str]:
    """Reload spoken aliases from config dir + local overlay (no daemon restart)."""
    aliases: dict[str, str] = {}
    aliases.update(_load_aliases_file(paths.aliases_toml))
    aliases.update(
        _load_aliases_file(paths.plugin_root / "config" / "aliases.local.toml")
    )
    return aliases


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value


def seed_config_files(paths: RuntimePaths) -> None:
    """Copy example config into the Herdr config dir if missing.

    Never copies ``*.local.toml`` — those stay in the checkout and are gitignored.
    """
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    examples = paths.plugin_root / "config"
    mapping = {
        "config.example.toml": paths.config_toml,
        "aliases.example.toml": paths.aliases_toml,
    }
    for src_name, dest in mapping.items():
        src = examples / src_name
        if dest.exists() or not src.is_file():
            continue
        shutil.copyfile(src, dest)
        try:
            dest.chmod(0o600)
        except OSError:
            pass
