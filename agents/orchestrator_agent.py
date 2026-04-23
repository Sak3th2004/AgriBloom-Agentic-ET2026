"""
Orchestrator Agent - Request Router and Session Manager
Routes requests to appropriate agents based on input analysis
Maintains conversation history and session state
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Supported languages with metadata — 10 Indian languages
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English", "tts_code": "en"},
    "hi": {"name": "Hindi", "native": "हिंदी", "tts_code": "hi"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "tts_code": "kn"},
    "te": {"name": "Telugu", "native": "తెలుగు", "tts_code": "te"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "tts_code": "ta"},
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "tts_code": "pa"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "tts_code": "gu"},
    "mr": {"name": "Marathi", "native": "मराठी", "tts_code": "mr"},
    "bn": {"name": "Bengali", "native": "বাংলা", "tts_code": "bn"},
    "or": {"name": "Odia", "native": "ଓଡ଼ିଆ", "tts_code": "or"},
}

# Crop keywords for routing
CROP_KEYWORDS = {
    "maize": ["maize", "corn", "makka", "ಮೆಕ್ಕೆಜೋಳ", "మొక్కజొన్న", "சோளம்", "ভুট্টা"],
    "tomato": ["tomato", "tamatar", "ಟೊಮ್ಯಾಟೊ", "టమాటా", "தக்காளி", "টমেটো"],
    "potato": ["potato", "aloo", "ಆಲೂಗಡ್ಡೆ", "బంగాళదుంప", "உருளைக்கிழங்கு", "আলু"],
    "rice": ["rice", "paddy", "dhan", "chawal", "ಅಕ್ಕಿ", "వరి", "நெல்", "ধান", "ਝੋਨਾ"],
    "wheat": ["wheat", "gehun", "ಗೋಧಿ", "గోధుమ", "கோதுமை", "গম", "ਕਣਕ"],
    "ragi": ["ragi", "finger millet", "nachni", "ರಾಗಿ", "రాగి", "கேழ்வரகு"],
    "sugarcane": ["sugarcane", "ganna", "ಕಬ್ಬು", "చెరకు", "கரும்பு", "ইক্ষু", "ਗੰਨਾ"],
    "cotton": ["cotton", "kapas", "ಹತ್ತಿ", "పత్తి", "பருத்தி", "তুলা", "ਕਪਾਹ", "कापूस"],
    "soybean": ["soybean", "soya", "soyabean", "सोयाबीन"],
    "groundnut": ["groundnut", "peanut", "moongfali", "mungfali", "ಕಡಲೆಕಾಯಿ", "వేరుశెనగ"],
    "apple": ["apple", "seb", "ಸೇಬು", "ఆపిల్", "ஆப்பிள்"],
    "grape": ["grape", "angur", "angoor", "ಮುಂತಾಕ", "ద్రాక్ష", "திராட்சை"],
    "pepper": ["pepper", "shimla mirch", "capsicum", "ಕ್ಯಾಪ್ಸಿಕಂ"],
}

# Intent keywords
INTENT_KEYWORDS = {
    "disease": ["disease", "blight", "rust", "rot", "infection", "problem", "issue", "leaf", "yellowing", "spots"],
    "weather": ["weather", "rain", "temperature", "forecast", "mausam"],
    "market": ["price", "market", "mandi", "sell", "buy", "rate", "cost"],
    "treatment": ["treatment", "spray", "medicine", "fungicide", "pesticide", "cure", "solution"],
}


def _detect_language(text: str, default: str = "en") -> str:
    """Detect language from text based on Unicode script detection."""
    if not text:
        return default

    # Count characters per script to handle mixed text
    script_counts = {}
    for char in text:
        code = ord(char)
        # Devanagari (Hindi/Marathi)
        if 0x0900 <= code <= 0x097F:
            script_counts["hi"] = script_counts.get("hi", 0) + 1
        # Bengali
        elif 0x0980 <= code <= 0x09FF:
            script_counts["bn"] = script_counts.get("bn", 0) + 1
        # Gurmukhi (Punjabi)
        elif 0x0A00 <= code <= 0x0A7F:
            script_counts["pa"] = script_counts.get("pa", 0) + 1
        # Gujarati
        elif 0x0A80 <= code <= 0x0AFF:
            script_counts["gu"] = script_counts.get("gu", 0) + 1
        # Odia
        elif 0x0B00 <= code <= 0x0B7F:
            script_counts["or"] = script_counts.get("or", 0) + 1
        # Tamil
        elif 0x0B80 <= code <= 0x0BFF:
            script_counts["ta"] = script_counts.get("ta", 0) + 1
        # Telugu
        elif 0x0C00 <= code <= 0x0C7F:
            script_counts["te"] = script_counts.get("te", 0) + 1
        # Kannada
        elif 0x0C80 <= code <= 0x0CFF:
            script_counts["kn"] = script_counts.get("kn", 0) + 1

    if script_counts:
        return max(script_counts, key=script_counts.get)

    return default


def _detect_crop_from_text(text: str) -> str | None:
    """Detect crop type from text using keywords."""
    text_lower = text.lower()

    for crop, keywords in CROP_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return crop

    return None


def _detect_intent(text: str, has_image: bool) -> str:
    """Detect user intent from text and context."""
    text_lower = text.lower() if text else ""

    if has_image:
        return "disease_detection"

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return intent

    return "general_inquiry"


def _determine_route(
    has_image: bool,
    intent: str,
) -> Literal["vision_first", "knowledge_first"]:
    """Determine the processing route based on input analysis."""
    # Vision-first route when image is provided
    if has_image:
        return "vision_first"

    # Knowledge-first for non-image queries
    return "knowledge_first"


def run_orchestrator(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Orchestrate request routing and session management.

    Input state keys:
        - image: PIL Image (optional)
        - user_text: User's query text
        - user_language: Language code
        - offline: Offline mode flag
        - lat, lon: Location coordinates

    Output state keys:
        - chat_history: Updated conversation history
        - route: "vision_first" or "knowledge_first"
        - detected_intent: User intent classification
        - detected_crop: Crop mentioned in text (if any)
        - lang: Normalized language code
        - status: "orchestrated"
    """
    # Extract inputs
    user_text = (state.get("user_text") or "").strip()
    user_language = state.get("user_language", "")
    has_image = state.get("image") is not None
    offline = bool(state.get("offline", False))

    # Initialize or get chat history
    chat_history = state.get("chat_history", [])

    # Auto-detect language from text if not explicitly set
    if not user_language or user_language == "en":
        detected_lang = _detect_language(user_text, "en")
        if detected_lang != "en":
            user_language = detected_lang
        elif not user_language:
            user_language = "en"

    # Detect intent and crop
    intent = _detect_intent(user_text, has_image)
    detected_crop = _detect_crop_from_text(user_text)

    # Determine route
    route = _determine_route(has_image, intent)

    # Log session event
    session_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": "orchestrator_received",
        "text": user_text[:100] if user_text else None,  # Truncate for privacy
        "language": user_language,
        "has_image": has_image,
        "offline": offline,
        "route": route,
        "intent": intent,
        "detected_crop": detected_crop,
    }
    chat_history.append(session_event)

    logger.info(
        f"Orchestrator: route={route}, intent={intent}, "
        f"lang={user_language}, crop={detected_crop}, offline={offline}"
    )

    return {
        **state,
        "chat_history": chat_history,
        "route": route,
        "detected_intent": intent,
        "detected_crop": detected_crop,
        "lang": user_language,
        "status": "orchestrated",
    }


# Export
__all__ = ["run_orchestrator", "SUPPORTED_LANGUAGES", "CROP_KEYWORDS"]
