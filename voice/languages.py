"""
Language maps for voice I/O across 10 Indian languages (+ English).

One place to translate our internal language code into whatever each engine
expects: Whisper ISO codes, edge-tts neural voice names, and gTTS codes.
"""
from __future__ import annotations

# Our canonical language codes.
SUPPORTED = ["en", "hi", "te", "ta", "kn", "bn", "mr", "gu", "ml", "pa"]

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
    "kn": "Kannada", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
    "ml": "Malayalam", "pa": "Punjabi",
}

# Whisper uses ISO-639-1 codes (same as ours here). "auto" => let Whisper detect.
WHISPER_LANG = {c: c for c in SUPPORTED}

# edge-tts neural voices (free, high quality). Indian (-IN) voices preferred.
EDGE_VOICES = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "ta": "ta-IN-PallaviNeural",
    "kn": "kn-IN-SapnaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "ml": "ml-IN-SobhanaNeural",
    # Punjabi has no -IN neural voice in edge-tts; TTS falls back to gTTS for pa.
}

# gTTS language codes (fallback engine).
GTTS_LANG = {
    "en": "en", "hi": "hi", "te": "te", "ta": "ta", "kn": "kn",
    "bn": "bn", "mr": "mr", "gu": "gu", "ml": "ml", "pa": "pa",
}


def normalize_lang(lang: str | None) -> str:
    """Coerce arbitrary input to a supported code; unknown -> 'en'."""
    if not lang:
        return "en"
    code = lang.strip().lower().split("-")[0]
    return code if code in SUPPORTED else "en"


def whisper_lang(lang: str | None) -> str | None:
    """Return the Whisper code, or None to request auto-detection."""
    if not lang or lang.strip().lower() in ("auto", ""):
        return None
    return WHISPER_LANG.get(normalize_lang(lang))


def edge_voice_for(lang: str | None) -> str | None:
    """Return an edge-tts voice name for the language, or None if unavailable."""
    return EDGE_VOICES.get(normalize_lang(lang))


def gtts_lang_for(lang: str | None) -> str:
    return GTTS_LANG.get(normalize_lang(lang), "en")
