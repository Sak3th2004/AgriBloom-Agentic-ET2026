"""
GenAI Handler — Google Gemini API Integration
Provides:
1. LLM-powered farmer-friendly treatment explanations
2. GenAI Vision fallback for unknown crops
3. LLM-generated PDF audit narratives
4. Conversational follow-up with context memory

Uses gemini-2.0-flash (FREE tier: 15 RPM, 1M tokens/day)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Lazy-loaded Gemini
_GEMINI_CLIENT = None  # Keep client alive to prevent connection closure
_GEMINI_AVAILABLE = None
_GEMINI_SDK = None  # "new" or "old"
_API_KEY = None


def _get_gemini_model():
    """Lazy-load Gemini and check availability."""
    global _GEMINI_CLIENT, _GEMINI_AVAILABLE, _GEMINI_SDK, _API_KEY

    if _GEMINI_AVAILABLE is False:
        return None

    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — GenAI features disabled")
        _GEMINI_AVAILABLE = False
        return None

    _API_KEY = api_key

    try:
        # Try new google.genai package first
        try:
            from google import genai as genai_new
            _GEMINI_CLIENT = genai_new.Client(api_key=api_key)
            _GEMINI_SDK = "new"
            _GEMINI_AVAILABLE = True
            logger.info("Gemini loaded via google.genai (new SDK)")
            return _GEMINI_CLIENT
        except ImportError:
            pass

        # Fall back to deprecated google.generativeai
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _GEMINI_CLIENT = genai.GenerativeModel("gemini-2.0-flash")
        _GEMINI_SDK = "old"
        _GEMINI_AVAILABLE = True
        logger.info("Gemini 2.0 Flash loaded via google.generativeai")
        return _GEMINI_CLIENT
    except Exception as e:
        logger.error(f"Failed to load Gemini model: {e}")
        _GEMINI_AVAILABLE = False
        return None


def _generate(prompt: str, image=None) -> str:
    """Unified Gemini API call — works with both new and old SDK. Auto-retries on rate limit."""
    _get_gemini_model()  # Ensure availability check ran
    if _GEMINI_AVAILABLE is not True:
        return ""

    import time

    for attempt in range(2):  # Try twice max
        try:
            if _GEMINI_SDK == "new":
                # ALWAYS create fresh client to avoid "client closed" errors
                from google import genai as genai_new
                fresh_client = genai_new.Client(api_key=_API_KEY)
                contents = [prompt]
                if image is not None:
                    contents = [image, prompt]
                response = fresh_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=contents,
                )
                return response.text.strip()
            else:
                # Old SDK: model.generate_content(prompt)
                if image is not None:
                    response = _GEMINI_CLIENT.generate_content([image, prompt])
                else:
                    response = _GEMINI_CLIENT.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt == 0:
                    logger.warning("Rate limited — waiting 15s and retrying...")
                    time.sleep(15)
                    continue
            logger.error(f"Gemini API call failed: {e}")
            return ""
    return ""


def is_genai_available() -> bool:
    """Check if GenAI features are available."""
    _get_gemini_model()
    return _GEMINI_AVAILABLE is True


def generate_treatment_advice(
    disease: str,
    crop: str,
    region: str = "India",
    season: str = "Kharif",
    language: str = "en",
    context: str = "",
) -> str:
    """
    Generate farmer-friendly treatment explanation using Gemini.

    Args:
        disease: Detected disease name
        crop: Crop type
        region: Farmer's region
        season: Current crop season
        language: Target language for response
        context: Additional RAG context from knowledge base

    Returns:
        Natural language treatment advice in farmer's language
    """
    model = _get_gemini_model()
    if model is None:
        return ""

    lang_names = {
        "en": "English", "hi": "Hindi", "kn": "Kannada",
        "te": "Telugu", "ta": "Tamil", "pa": "Punjabi",
        "gu": "Gujarati", "mr": "Marathi", "bn": "Bengali",
        "or": "Odia",
    }
    lang_name = lang_names.get(language, "English")

    context_section = ""
    if context:
        context_section = f"""
    Reference knowledge from ICAR advisories:
    {context}

    Use the above reference knowledge to provide accurate, specific advice.
    """

    prompt = f"""You are an expert Indian agricultural scientist (Krishi Vigyan Kendra advisor).

    A farmer in {region}, India has {crop} affected by {disease}.
    Current season: {season}.
    {context_section}
    Give advice in SIMPLE language that an uneducated farmer can understand:
    1. What is this disease? (2 sentences, no scientific jargon)
    2. Immediate action to take TODAY
    3. Safe treatment options (ONLY treatments approved by ICAR India)
    4. Prevention for next season
    5. When to harvest after treatment (waiting period)

    IMPORTANT: Do NOT recommend any banned pesticides in India.
    If Endosulfan, Monocrotophos, Methyl Parathion, Phorate, Dichlorvos,
    or any banned chemical would normally be recommended, suggest
    organic/bio alternatives instead (Neem oil, Trichoderma, etc.)

    Respond in {lang_name} language.
    Keep each point under 30 words. Use simple vocabulary.
    Format with bullet points."""

    try:
        result = _generate(prompt)
        if result:
            logger.info(f"Gemini treatment advice generated ({len(result)} chars)")
        return result
    except Exception as e:
        logger.error(f"Gemini treatment advice failed: {e}")
        return ""


def analyze_unknown_crop(image_path: str, language: str = "en") -> dict[str, Any]:
    """
    GenAI Vision fallback for crops not in the trained model.
    Uses Gemini Vision to analyze any plant disease.

    Args:
        image_path: Path to the crop image
        language: Target language

    Returns:
        Dict with crop, disease, confidence, treatment, is_fallback flag
    """
    model = _get_gemini_model()
    if model is None:
        return {
            "crop": "unknown",
            "disease": "unknown",
            "confidence": "low",
            "treatment": "Please visit your nearest KVK for diagnosis.",
            "is_fallback": True,
            "source": "genai_unavailable",
        }

    try:
        from PIL import Image as PILImage
        image = PILImage.open(image_path)

        prompt = """You are an expert Indian agricultural scientist.
    Analyze this crop leaf image:
    1. What crop is this?
    2. Is it diseased? If yes, what disease?
    3. How confident are you? (high/medium/low)
    4. What treatment do you recommend for Indian farmers?
    5. What is the severity? (mild/moderate/severe)

    If you cannot identify the crop or disease clearly, say so honestly.
    DO NOT guess if unsure. Recommend the farmer visit their nearest
    KVK (Krishi Vigyan Kendra).

    DO NOT recommend banned pesticides in India (Endosulfan, Monocrotophos, etc.)

    Respond in JSON format ONLY (no markdown):
    {"crop": "...", "disease": "...", "confidence": "high/medium/low", "severity": "mild/moderate/severe", "treatment": "...", "prevention": "..."}"""

        text = _generate(prompt, image=image)

        # Parse JSON from response
        # Handle case where response has markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        result["is_fallback"] = True
        result["source"] = "gemini_vision"

        logger.info(f"Gemini Vision fallback: crop={result.get('crop')}, disease={result.get('disease')}")
        return result

    except json.JSONDecodeError:
        logger.warning("Gemini returned non-JSON response, extracting text")
        return {
            "crop": "unknown",
            "disease": "unknown",
            "confidence": "low",
            "treatment": response.text.strip() if 'response' in dir() else "Visit nearest KVK",
            "is_fallback": True,
            "source": "gemini_vision_text",
        }
    except Exception as e:
        logger.error(f"Gemini Vision fallback failed: {e}")
        return {
            "crop": "unknown",
            "disease": "analysis_failed",
            "confidence": "low",
            "treatment": "Please visit your nearest Krishi Vigyan Kendra (KVK) for expert diagnosis.",
            "is_fallback": True,
            "source": "genai_error",
            "error": str(e),
        }


def analyze_unknown_crop_pil(image, language: str = "en") -> dict[str, Any]:
    """
    GenAI Vision fallback using PIL Image directly (no file path needed).
    """
    model = _get_gemini_model()
    if model is None:
        return {
            "crop": "unknown", "disease": "unknown", "confidence": "low",
            "treatment": "Visit nearest KVK.", "is_fallback": True,
            "source": "genai_unavailable",
        }

    try:
        prompt = """You are an expert Indian agricultural scientist.
    Analyze this crop leaf image:
    1. What crop is this?
    2. Is it diseased? If yes, what disease?
    3. How confident are you? (high/medium/low)
    4. What treatment do you recommend for Indian farmers?

    If unsure, say so honestly. Recommend visiting nearest KVK.
    DO NOT recommend banned pesticides (Endosulfan, Monocrotophos, etc.)

    Respond in JSON format ONLY:
    {"crop": "...", "disease": "...", "confidence": "high/medium/low", "treatment": "...", "prevention": "..."}"""

        text = _generate(prompt, image=image)

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        result["is_fallback"] = True
        result["source"] = "gemini_vision"
        return result

    except Exception as e:
        logger.error(f"Gemini Vision PIL fallback failed: {e}")
        return {
            "crop": "unknown", "disease": "analysis_failed", "confidence": "low",
            "treatment": "Visit nearest KVK.", "is_fallback": True,
            "source": "genai_error",
        }


def generate_audit_narrative(agent_trace: dict, language: str = "en") -> str:
    """
    Generate a professional audit report narrative using LLM.

    Args:
        agent_trace: Dict with all agent results
        language: Target language

    Returns:
        Natural language audit narrative
    """
    model = _get_gemini_model()
    if model is None:
        return ""

    lang_names = {
        "en": "English", "hi": "Hindi", "kn": "Kannada",
        "te": "Telugu", "ta": "Tamil",
    }

    prompt = f"""Write a professional but simple agricultural advisory report based on this analysis:
    {json.dumps(agent_trace, indent=2, default=str)}

    Include:
    - What disease was found and how confident the system is
    - What compliance checks were performed
    - What treatments were recommended and why
    - What was blocked and why (if any)
    - Disclaimer about consulting local KVK

    Keep it under 200 words. Language: {lang_names.get(language, 'English')}.
    Write in simple language a farmer can understand."""

    try:
        return _generate(prompt)
    except Exception as e:
        logger.error(f"Audit narrative generation failed: {e}")
        return ""


def conversational_followup(
    question: str,
    conversation_history: list[dict],
    crop: str = "",
    disease: str = "",
    language: str = "en",
) -> str:
    """
    Handle follow-up questions with conversation context.

    Args:
        question: Farmer's follow-up question
        conversation_history: List of previous messages
        crop: Current crop context
        disease: Current disease context
        language: Target language

    Returns:
        Contextual follow-up response
    """
    model = _get_gemini_model()
    if model is None:
        return "GenAI not available. Please consult your local KVK."

    lang_names = {
        "en": "English", "hi": "Hindi", "kn": "Kannada",
        "te": "Telugu", "ta": "Tamil",
    }

    # Build conversation context
    context_lines = []
    for msg in conversation_history[-5:]:  # Last 5 messages
        role = msg.get("role", "system")
        content = msg.get("content", "")
        context_lines.append(f"{role}: {content}")

    context = "\n".join(context_lines)

    prompt = f"""You are an agricultural advisor helping an Indian farmer.
    Current context:
    - Crop: {crop}
    - Disease detected: {disease}

    Previous conversation:
    {context}

    Farmer's new question: {question}

    Give a helpful, simple answer in {lang_names.get(language, 'English')}.
    Keep it under 100 words.
    Do NOT recommend banned pesticides.
    If unsure, suggest visiting nearest KVK."""

    try:
        result = _generate(prompt)
        return result or "Unable to process. Please consult your local KVK."
    except Exception as e:
        logger.error(f"Conversational followup failed: {e}")
        return "Unable to process question. Please consult your local KVK."


# Export
__all__ = [
    "is_genai_available",
    "generate_treatment_advice",
    "analyze_unknown_crop",
    "analyze_unknown_crop_pil",
    "generate_audit_narrative",
    "conversational_followup",
]
