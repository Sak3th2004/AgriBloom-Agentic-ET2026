"""
Text-to-speech: edge-tts (free neural Indic voices) -> gTTS fallback.

``synthesize`` writes an MP3 for the given text/language and returns its path.
edge-tts gives natural neural voices for 9 of our 10 languages; Punjabi (and any
edge failure) falls back to gTTS. Returns None if no engine can produce audio.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from voice.languages import edge_voice_for, gtts_lang_for

logger = logging.getLogger(__name__)

# Strip emojis/pictographs so they aren't read aloud as "..." noise.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿✀-➿]",
    flags=re.UNICODE,
)


def _clean(text: str, max_chars: int = 1000) -> str:
    text = _EMOJI_RE.sub("", text or "").strip()
    return text[:max_chars]


async def _edge_save(text: str, voice: str, out_path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def _try_edge(text: str, lang: str, out_path: str) -> bool:
    voice = edge_voice_for(lang)
    if not voice:
        return False
    try:
        asyncio.run(_edge_save(text, voice, out_path))
        return Path(out_path).exists() and Path(out_path).stat().st_size > 0
    except Exception as e:
        logger.warning("edge-tts failed (%s): %s", lang, e)
        return False


def _try_gtts(text: str, lang: str, out_path: str) -> bool:
    try:
        from gtts import gTTS

        gTTS(text=text, lang=gtts_lang_for(lang), slow=False).save(out_path)
        return Path(out_path).exists() and Path(out_path).stat().st_size > 0
    except Exception as e:
        logger.warning("gTTS failed (%s): %s", lang, e)
        return False


def synthesize(text: str, lang: str = "en", out_path: Optional[str] = None) -> Optional[str]:
    """Render ``text`` to an MP3 in ``lang``; return the path or None.

    Tries edge-tts (neural) first, then gTTS. Text is cleaned of emojis and
    truncated for latency.
    """
    clean = _clean(text)
    if not clean:
        return None
    if out_path is None:
        out_path = str(Path("models/outputs") / "voice_output.mp3")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if _try_edge(clean, lang, out_path):
        return out_path
    if _try_gtts(clean, lang, out_path):
        return out_path
    logger.error("TTS: all engines failed for lang=%s", lang)
    return None
