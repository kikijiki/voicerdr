"""Explicit model provisioning for machines that must later run offline."""

from collections.abc import Callable
from typing import Any

from voicerdr.config import AppConfig

Progress = Callable[[str], None]


def _bootstrap_tokenizer(progress: Progress) -> dict[str, Any]:
    import nltk

    try:
        location = nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        progress("Downloading NLTK punkt_tab tokenizer data…")
        if not nltk.download("punkt_tab", quiet=False, raise_on_error=True):
            raise RuntimeError("NLTK did not provision punkt_tab")
        location = nltk.data.find("tokenizers/punkt_tab")
    return {"ready": True, "path": str(location)}


def _bootstrap_stt(model: str, progress: Progress) -> dict[str, Any]:
    from moonshine_voice import (
        get_model_for_language,
        model_arch_to_string,
        string_to_model_arch,
    )

    requested = model.strip().lower().replace("_", "-")
    architecture = string_to_model_arch(requested)

    def report(fraction: float, filename: str) -> None:
        progress(f"Moonshine {fraction:>6.1%} {filename}".rstrip())

    model_path, resolved = get_model_for_language(
        "en", architecture, on_progress=report
    )
    return {
        "ready": True,
        "requested": requested,
        "resolved": model_arch_to_string(resolved),
        "path": str(model_path),
    }


def _bootstrap_tts(voice: str, progress: Progress) -> dict[str, Any]:
    from pipecat.services.kokoro.tts import KokoroTTSService

    progress("Provisioning Kokoro ONNX model and voice table…")
    # Pipecat's constructor provisions the cache voicerdr uses at runtime.
    KokoroTTSService(settings=KokoroTTSService.Settings(voice=voice))
    return {"ready": True, "voice": voice}


def _verify_vad(sample_rate: int) -> dict[str, Any]:
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    SileroVADAnalyzer(sample_rate=sample_rate)
    return {"ready": True, "bundled": True, "sample_rate": sample_rate}


def bootstrap_models(
    config: AppConfig,
    *,
    tokenizer: bool = True,
    stt: bool = True,
    tts: bool = True,
    progress: Progress = print,
) -> dict[str, Any]:
    """Provision all assets needed after Python dependencies are installed."""
    result: dict[str, Any] = {"silero_vad": _verify_vad(config.sample_rate)}
    if tokenizer:
        result["nltk"] = _bootstrap_tokenizer(progress)
    if stt:
        result["moonshine_stt"] = _bootstrap_stt(config.stt_model, progress)
    if tts:
        result["kokoro_tts"] = _bootstrap_tts(config.tts_voice, progress)
    return result
