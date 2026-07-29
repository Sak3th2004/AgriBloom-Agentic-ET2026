"""
Speech-to-text: faster-whisper (local, GPU) -> Groq Whisper (free API).

``transcribe`` accepts an audio file path (or raw bytes) and returns
``{"text", "language", "source"}``. It auto-detects the spoken language when
``language="auto"`` (or None), so a farmer can speak any of the supported Indian
languages. Both backends are optional; if neither is available it returns an
empty transcription with an ``error`` rather than raising.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

from voice.languages import normalize_lang, whisper_lang

logger = logging.getLogger(__name__)

# Local model size — "small" is a good accuracy/speed trade-off on an 8GB GPU.
FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "small")
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

_MODEL = None  # cached faster-whisper model


def _as_path(audio: Union[str, bytes, Path]) -> tuple[str, Optional[str]]:
    """Return (path, temp_path_to_cleanup). Writes bytes to a temp file."""
    if isinstance(audio, (str, Path)):
        return str(audio), None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio)
    tmp.close()
    return tmp.name, tmp.name


def _faster_whisper(path: str, language: Optional[str]) -> Optional[dict[str, Any]]:
    """Local transcription; returns None if faster-whisper isn't available."""
    global _MODEL
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    try:
        if _MODEL is None:
            try:
                _MODEL = WhisperModel(FASTER_WHISPER_MODEL, device="cuda", compute_type="int8")
            except Exception:  # no CUDA -> CPU
                _MODEL = WhisperModel(FASTER_WHISPER_MODEL, device="cpu", compute_type="int8")
        segments, info = _MODEL.transcribe(path, language=language, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()
        detected = getattr(info, "language", None) or language or "en"
        return {"text": text, "language": normalize_lang(detected), "source": "faster_whisper"}
    except Exception as e:
        logger.warning("faster-whisper failed: %s", e)
        return None


def _groq_whisper(path: str, language: Optional[str]) -> Optional[dict[str, Any]]:
    """Free Groq Whisper API fallback; needs GROQ_API_KEY."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return None
    try:
        import requests

        with open(path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            data = {"model": GROQ_WHISPER_MODEL}
            if language:
                data["language"] = language
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}", "User-Agent": "AgriBloom/2.0"},
                files=files,
                data=data,
                timeout=30,
            )
        resp.raise_for_status()
        body = resp.json()
        return {
            "text": (body.get("text") or "").strip(),
            "language": normalize_lang(body.get("language") or language or "en"),
            "source": "groq_whisper",
        }
    except Exception as e:
        logger.warning("Groq Whisper failed: %s", e)
        return None


def transcribe(audio: Union[str, bytes, Path], language: str = "auto") -> dict[str, Any]:
    """Transcribe speech to text.

    Args:
        audio: path to an audio file, or raw audio bytes.
        language: a supported code, or "auto" to detect.

    Returns:
        ``{"text", "language", "source"}``; on total failure ``text`` is "" and
        an ``error`` key is present.
    """
    wl = whisper_lang(language)
    path, cleanup = _as_path(audio)
    try:
        for backend in (_faster_whisper, _groq_whisper):
            result = backend(path, wl)
            if result is not None and result.get("text"):
                return result
        return {"text": "", "language": normalize_lang(language if language != "auto" else "en"),
                "source": "none", "error": "no STT backend available"}
    finally:
        if cleanup:
            try:
                os.unlink(cleanup)
            except OSError:
                pass
