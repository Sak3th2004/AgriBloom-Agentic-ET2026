"""
Translator - Multilingual Translation Wrapper
Uses deep-translator for text translation with caching for offline mode.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Translation cache (in-memory)
_TRANSLATION_CACHE: dict[str, str] = {}


def translate_text(
    text: str,
    target_lang: str,
    source_lang: str = "en",
) -> str:
    """
    Translate text to target language using deep-translator.

    Args:
        text: Text to translate
        target_lang: Target language code (hi, kn, te, ta, etc.)
        source_lang: Source language code

    Returns:
        Translated text, or original text if translation fails
    """
    if not text or not text.strip():
        return text

    if target_lang == source_lang:
        return text

    # Check cache first
    cache_key = f"{source_lang}:{target_lang}:{text[:100]}"
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    # Language code mapping for Google Translate
    lang_map = {
        "en": "en", "hi": "hi", "kn": "kn", "te": "te", "ta": "ta",
        "pa": "pa", "gu": "gu", "mr": "mr", "bn": "bn", "or": "or",
    }

    src = lang_map.get(source_lang, "en")
    tgt = lang_map.get(target_lang, target_lang)

    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=src, target=tgt)
        result = translator.translate(text)

        if result:
            _TRANSLATION_CACHE[cache_key] = result
            return result
        return text

    except ImportError:
        logger.warning("deep-translator not installed, returning original text")
        return text
    except Exception as e:
        logger.warning(f"Translation failed ({src}→{tgt}): {e}")
        return text


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text to English."""
    return translate_text(text, target_lang="en", source_lang=source_lang)


def clear_cache():
    """Clear the translation cache."""
    _TRANSLATION_CACHE.clear()


# Export
__all__ = ["translate_text", "translate_to_english", "clear_cache"]
