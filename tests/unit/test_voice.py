"""Unit tests for the voice package (language maps + STT/TTS fallback logic)."""
from __future__ import annotations

from pathlib import Path

import voice.stt as stt
import voice.tts as tts
from voice.languages import (
    SUPPORTED,
    edge_voice_for,
    gtts_lang_for,
    normalize_lang,
    whisper_lang,
)


# ── language maps ──────────────────────────────────────────────────────────────
def test_normalize_lang():
    assert normalize_lang("hi") == "hi"
    assert normalize_lang("HI-IN") == "hi"
    assert normalize_lang("klingon") == "en"
    assert normalize_lang(None) == "en"


def test_whisper_lang_auto_is_none():
    assert whisper_lang("auto") is None
    assert whisper_lang(None) is None
    assert whisper_lang("te") == "te"


def test_edge_voice_for_known_langs():
    assert edge_voice_for("te") == "te-IN-ShrutiNeural"
    assert edge_voice_for("hi") == "hi-IN-SwaraNeural"
    # Punjabi intentionally has no edge voice -> None -> gTTS fallback.
    assert edge_voice_for("pa") is None


def test_gtts_lang_for():
    assert gtts_lang_for("ml") == "ml"
    assert gtts_lang_for("unknown") == "en"


def test_all_supported_have_gtts():
    for code in SUPPORTED:
        assert gtts_lang_for(code) == code


# ── TTS fallback logic ──────────────────────────────────────────────────────────
def test_tts_empty_text_returns_none(tmp_path):
    assert tts.synthesize("   ", "en", str(tmp_path / "o.mp3")) is None


def test_tts_falls_back_to_gtts_when_edge_fails(tmp_path, monkeypatch):
    out = str(tmp_path / "o.mp3")

    monkeypatch.setattr(tts, "_try_edge", lambda text, lang, path: False)

    def fake_gtts(text, lang, path):
        Path(path).write_bytes(b"ID3fake")
        return True

    monkeypatch.setattr(tts, "_try_gtts", fake_gtts)
    assert tts.synthesize("hello", "pa", out) == out


def test_tts_returns_none_when_all_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_try_edge", lambda *a: False)
    monkeypatch.setattr(tts, "_try_gtts", lambda *a: False)
    assert tts.synthesize("hello", "en", str(tmp_path / "o.mp3")) is None


def test_tts_prefers_edge(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tts, "_try_edge", lambda text, lang, path: calls.append("edge") or True)
    monkeypatch.setattr(tts, "_try_gtts", lambda *a: calls.append("gtts") or True)
    tts.synthesize("hi", "hi", str(tmp_path / "o.mp3"))
    assert calls == ["edge"]  # gTTS not attempted when edge succeeds


def test_clean_strips_emoji_and_truncates():
    out = tts._clean("Hello 🌾🚜 world", max_chars=100)
    assert "🌾" not in out and "Hello" in out and "world" in out


# ── STT fallback logic ──────────────────────────────────────────────────────────
def test_stt_no_backend_returns_error(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFFfake")
    monkeypatch.setattr(stt, "_faster_whisper", lambda path, lang: None)
    monkeypatch.setattr(stt, "_groq_whisper", lambda path, lang: None)
    out = stt.transcribe(str(audio), language="auto")
    assert out["text"] == ""
    assert "error" in out


def test_stt_uses_first_successful_backend(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFFfake")
    monkeypatch.setattr(
        stt, "_faster_whisper",
        lambda path, lang: {"text": "namaste", "language": "hi", "source": "faster_whisper"},
    )
    called = {"groq": False}
    monkeypatch.setattr(stt, "_groq_whisper", lambda path, lang: called.__setitem__("groq", True))
    out = stt.transcribe(str(audio), language="hi")
    assert out["text"] == "namaste"
    assert out["source"] == "faster_whisper"
    assert called["groq"] is False  # fallback not used when primary succeeds


def test_stt_bytes_input_is_handled(monkeypatch):
    monkeypatch.setattr(
        stt, "_faster_whisper",
        lambda path, lang: {"text": "hi", "language": "en", "source": "faster_whisper"},
    )
    monkeypatch.setattr(stt, "_groq_whisper", lambda path, lang: None)
    out = stt.transcribe(b"RAWAUDIOBYTES", language="auto")
    assert out["text"] == "hi"
