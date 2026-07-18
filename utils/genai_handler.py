"""
GenAI Handler - configurable multi-backend LLM integration.

Provider order is controlled by AGRIBLOOM_LLM_ORDER. Cloud credentials must
come from environment variables; local Ollama remains the offline text fallback.
"""
from __future__ import annotations

import json
import logging
import os
import base64
import io
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_LLM_ORDER = "nvidia,openai,openai_compatible,gemini,ollama"

# ═════════════════════════════════════════════════════════════════════════════
# NVIDIA API (PRIMARY — free, powerful, reliable)
# Smart key segregation: Key1=Vision, Key2=Treatment, Key3=General/Followup
# Each task gets a dedicated key → 3000 total calls/day, no interference
# ═════════════════════════════════════════════════════════════════════════════
_NVIDIA_API_KEYS = [
    k.strip() for k in [
        os.getenv("NVIDIA_API_KEY", ""),
        os.getenv("NVIDIA_API_KEY_2", ""),
        os.getenv("NVIDIA_API_KEY_3", ""),
    ] if k.strip()
]
_NVIDIA_KEY_IDX = 0
_NVIDIA_API_KEY = _NVIDIA_API_KEYS[0] if _NVIDIA_API_KEYS else ""
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Task-to-key mapping (segregates load across keys)
_NVIDIA_TASK_KEY = {
    "vision": 0,       # Key 1: Vision identification (90B model)
    "treatment": 1,    # Key 2: Treatment advice (70B model)
    "general": 2,      # Key 3: Follow-up, chat, other
}

logger.info(f"NVIDIA API: {len(_NVIDIA_API_KEYS)} keys loaded for smart rotation")


def _nvidia_generate(prompt: str, image=None, model: str = None, task: str = "general") -> str:
    """Call NVIDIA's free API with smart key segregation. Supports text AND vision."""
    global _NVIDIA_KEY_IDX
    if not _NVIDIA_API_KEYS:
        return ""
    
    # Start with the dedicated key for this task type
    n_keys = len(_NVIDIA_API_KEYS)
    start_idx = _NVIDIA_TASK_KEY.get(task, 2) % n_keys
    
    # Try each key once, starting from the task-dedicated key
    for attempt in range(n_keys):
        key_idx = (start_idx + attempt) % n_keys
        key = _NVIDIA_API_KEYS[key_idx]
        try:
            import urllib.request
            import json as _json

            if model is None:
                model = "meta/llama-3.2-90b-vision-instruct" if image is not None else "meta/llama-3.3-70b-instruct"
            
            if image is not None:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            else:
                content = prompt

            payload = _json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 2048,
                "temperature": 0.4,
            }).encode("utf-8")

            req = urllib.request.Request(
                _NVIDIA_BASE_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = _json.loads(resp.read().decode())
                text = data["choices"][0]["message"]["content"].strip()
                if text:
                    logger.info(f"NVIDIA API response ({len(text)} chars, model={model.split('/')[-1]}, key={key_idx}, task={task})")
                return text
        except Exception as e:
            logger.warning(f"NVIDIA key {key_idx} failed ({task}): {e}")
            if attempt < n_keys - 1:
                logger.info(f"Rotating to NVIDIA key {(start_idx + attempt + 1) % n_keys}")
                continue
    
    logger.warning(f"All NVIDIA API keys failed for task={task}")
    return ""


def _provider_order() -> list[str]:
    """Return configured provider order with unknown names removed."""
    raw = os.getenv("AGRIBLOOM_LLM_ORDER", DEFAULT_LLM_ORDER)
    allowed = {"nvidia", "openai", "openai_compatible", "compatible", "gemini", "ollama"}
    order: list[str] = []
    for item in raw.split(","):
        provider = item.strip().lower().replace("-", "_")
        if provider in allowed and provider not in order:
            order.append(provider)
    return order or DEFAULT_LLM_ORDER.split(",")


def _openai_provider_config(provider: str) -> dict[str, str]:
    """
    Read OpenAI-compatible provider config from env.

    OpenAI requires OPENAI_API_KEY and OPENAI_MODEL. Custom providers such as
    Grok/xAI, hosted vLLM, LiteLLM, or internal gateways can use the generic
    OPENAI_COMPATIBLE_* variables.
    """
    provider = provider.strip().lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = (
            os.getenv("OPENAI_MODEL", "").strip()
            or os.getenv("OPENAI_CHAT_MODEL", "").strip()
        )
        base_url = os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1/chat/completions",
        ).strip()
        return {"name": "openai", "api_key": api_key, "model": model, "base_url": base_url}

    api_key = (
        os.getenv("OPENAI_COMPATIBLE_API_KEY", "").strip()
        or os.getenv("GROK_API_KEY", "").strip()
        or os.getenv("XAI_API_KEY", "").strip()
    )
    model = (
        os.getenv("OPENAI_COMPATIBLE_MODEL", "").strip()
        or os.getenv("GROK_MODEL", "").strip()
        or os.getenv("XAI_MODEL", "").strip()
    )
    base_url = (
        os.getenv("OPENAI_COMPATIBLE_BASE_URL", "").strip()
        or os.getenv("GROK_BASE_URL", "").strip()
        or os.getenv("XAI_BASE_URL", "").strip()
    )
    return {
        "name": "openai_compatible",
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
    }


def _openai_compatible_generate(prompt: str, provider: str = "openai", task: str = "general") -> str:
    """Generate text through any Chat Completions compatible endpoint."""
    config = _openai_provider_config(provider)
    api_key = config["api_key"]
    model = config["model"]
    base_url = config["base_url"]

    if not api_key or not model or not base_url:
        return ""

    try:
        import urllib.request
        import json as _json

        payload = _json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a practical agricultural advisor. "
                        "Give farmer-safe, concise, actionable guidance. "
                        "Do not recommend banned or restricted pesticides."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(os.getenv("AGRIBLOOM_LLM_MAX_TOKENS", "1600")),
            "temperature": float(os.getenv("AGRIBLOOM_LLM_TEMPERATURE", "0.3")),
        }).encode("utf-8")

        req = urllib.request.Request(
            base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        timeout = float(os.getenv("AGRIBLOOM_LLM_TIMEOUT_SECONDS", "30"))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            if text:
                logger.info(
                    "OpenAI-compatible response (%s chars, provider=%s, model=%s, task=%s)",
                    len(text),
                    config["name"],
                    model,
                    task,
                )
            return text
    except Exception as e:
        logger.warning(f"OpenAI-compatible provider failed ({provider}, task={task}): {e}")
        return ""


def get_configured_llm_providers(check_ollama: bool = False) -> list[str]:
    """Return configured LLM providers without making cloud API calls."""
    providers: list[str] = []
    if _NVIDIA_API_KEYS:
        providers.append("nvidia")
    if _openai_provider_config("openai").get("api_key") and _openai_provider_config("openai").get("model"):
        providers.append("openai")
    compatible = _openai_provider_config("openai_compatible")
    if compatible.get("api_key") and compatible.get("model") and compatible.get("base_url"):
        providers.append("openai_compatible")
    if API_KEYS:
        providers.append("gemini")
    if check_ollama:
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            providers.append("ollama")
        except Exception:
            pass
    return providers


# ═════════════════════════════════════════════════════════════════════════════
# Google Gemini (SECONDARY — often rate-limited)
# ═════════════════════════════════════════════════════════════════════════════
_GEMINI_CLIENT = None
_GEMINI_AVAILABLE = None
_GEMINI_SDK = None
_API_KEY = None

API_KEYS = [
    os.getenv("GEMINI_API_KEY", "").strip(),
    os.getenv("GEMINI_API_KEY_2", "").strip(),
    os.getenv("GEMINI_API_KEY_3", "").strip(),
]
API_KEYS = [k for k in API_KEYS if k]
_CURRENT_KEY_IDX = 0

def _get_gemini_model(force_reinit=False):
    """Lazy-load Gemini and check availability. Supports rotation."""
    global _GEMINI_CLIENT, _GEMINI_AVAILABLE, _GEMINI_SDK, _API_KEY, _CURRENT_KEY_IDX

    if _GEMINI_AVAILABLE is False and not force_reinit:
        return None

    if _GEMINI_CLIENT is not None and not force_reinit:
        return _GEMINI_CLIENT

    if not API_KEYS:
        logger.warning("GEMINI_API_KEY not set — GenAI features disabled")
        _GEMINI_AVAILABLE = False
        return None

    api_key = API_KEYS[_CURRENT_KEY_IDX]
    _API_KEY = api_key

    try:
        try:
            from google import genai as genai_new
            _GEMINI_CLIENT = genai_new.Client(api_key=api_key)
            _GEMINI_SDK = "new"
            _GEMINI_AVAILABLE = True
            logger.info(f"Gemini loaded (key {api_key[:10]}...) via google.genai")
            return _GEMINI_CLIENT
        except ImportError:
            pass

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _GEMINI_CLIENT = genai.GenerativeModel("gemini-2.0-flash")
        _GEMINI_SDK = "old"
        _GEMINI_AVAILABLE = True
        logger.info(f"Gemini loaded (key {api_key[:10]}...) via google.generativeai")
        return _GEMINI_CLIENT
    except Exception as e:
        logger.error(f"Failed to load Gemini model: {e}")
        _GEMINI_AVAILABLE = False
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Ollama (LOCAL FALLBACK — zero rate limits)
# ═════════════════════════════════════════════════════════════════════════════
def _ollama_generate(prompt: str) -> str:
    """Fallback: use local Ollama for text generation. Zero rate limits."""
    try:
        import urllib.request
        import json as _json

        payload = _json.dumps({
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 500},
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode())
            text = data.get("response", "").strip()
            if text:
                logger.info(f"Ollama generated response ({len(text)} chars)")
            return text
    except Exception as e:
        logger.warning(f"Ollama fallback failed: {e}")
        return ""


_GEMINI_COOLDOWN_UNTIL = 0

def _generate(prompt: str, image=None, task: str = "general") -> str:
    """Unified LLM call using AGRIBLOOM_LLM_ORDER."""
    global _CURRENT_KEY_IDX, _GEMINI_COOLDOWN_UNTIL
    import time

    for provider in _provider_order():
        if provider == "nvidia" and _NVIDIA_API_KEYS:
            result = _nvidia_generate(prompt, image, task=task)
            if result:
                return result

        if provider in {"openai", "openai_compatible", "compatible"} and image is None:
            result = _openai_compatible_generate(prompt, provider=provider, task=task)
            if result:
                return result

        if provider == "gemini":
            _get_gemini_model()
            if _GEMINI_AVAILABLE is True and time.time() > _GEMINI_COOLDOWN_UNTIL:
                # Try each Gemini key once only (fast fail)
                for attempt in range(len(API_KEYS)):
                    try:
                        if _GEMINI_SDK == "new":
                            from google import genai as genai_new
                            fresh_client = genai_new.Client(api_key=API_KEYS[_CURRENT_KEY_IDX])
                            contents = [prompt]
                            if image is not None:
                                contents = [image, prompt]
                            response = fresh_client.models.generate_content(
                                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                                contents=contents,
                            )
                            result = response.text.strip()
                            if result:
                                return result
                        else:
                            if image is not None:
                                response = _GEMINI_CLIENT.generate_content([image, prompt])
                            else:
                                response = _GEMINI_CLIENT.generate_content(prompt)
                            result = response.text.strip()
                            if result:
                                return result
                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                            logger.warning(f"Gemini Key {_CURRENT_KEY_IDX} rate limited.")
                            _CURRENT_KEY_IDX = (_CURRENT_KEY_IDX + 1) % len(API_KEYS)
                            _get_gemini_model(force_reinit=True)
                            continue
                        logger.warning(f"Gemini generation failed: {e}")
                        break
                _GEMINI_COOLDOWN_UNTIL = time.time() + 60
                logger.warning("All Gemini keys exhausted - cooldown 60s")
            elif _GEMINI_AVAILABLE is True:
                logger.info("Gemini in cooldown - skipping")

        if provider == "ollama" and image is None:
            result = _ollama_generate(prompt)
            if result:
                return result

    logger.error("All configured LLM backends failed")
    return ""


def is_genai_available() -> bool:
    """Check if any LLM backend is configured or locally available."""
    if _NVIDIA_API_KEYS:
        return True
    if "openai" in get_configured_llm_providers(check_ollama=False):
        return True
    if "openai_compatible" in get_configured_llm_providers(check_ollama=False):
        return True
    model = _get_gemini_model()
    if model is not None:
        return True
    # Check Ollama
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def generate_treatment_advice(
    disease: str,
    crop: str,
    region: str = "India",
    season: str = "Kharif",
    language: str = "en",
    context: str = "",
) -> str:
    """
    Generate farmer-friendly treatment explanation using the configured LLM provider chain.

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
    if not is_genai_available():
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
    You MUST respond ENTIRELY in {lang_name} language. Every single word must be in {lang_name}.

    A farmer in {region}, India has {crop} affected by {disease}.
    Current season: {season}.
    {context_section}
    Give DETAILED advice that a farmer can act on immediately:
    1. 🌿 What disease is this? (explain in simple words, no English jargon)
    2. ⚡ Immediate action to take TODAY
    3. 💊 Exact treatment: Name the specific pesticide/fungicide, dosage per liter of water, spray interval
       - Give 2 options: one chemical, one organic (Neem oil, Trichoderma, Pseudomonas)
    4. 🧪 Exact fertilizer to apply: NPK ratio, urea amount per acre
    5. 🛡️ Prevention for next season
    6. ⏰ When to harvest after treatment (waiting period in days)

    IMPORTANT: Do NOT recommend banned pesticides (Endosulfan, Monocrotophos, Methyl Parathion, Phorate).
    Use ICAR-approved treatments only.

    CRITICAL: Write your ENTIRE response in {lang_name}. NOT in English. Every word in {lang_name}."""

    try:
        result = _generate(prompt, task="treatment")
        if result:
            logger.info(f"Treatment advice generated ({len(result)} chars)")
        return result
    except Exception as e:
        logger.error(f"Treatment advice failed: {e}")
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


def _ollama_vision_analyze(image, prompt: str) -> str:
    """Analyze crop image using vision AI. 
    Priority: NVIDIA 90B Vision → Ollama LLaVA (local).
    
    This is the KEY self-learning feature: the vision model can SEE any image
    and diagnose any crop disease, even crops EfficientNet was never trained on.
    """
    # ── Try NVIDIA Vision first (90B model, extremely accurate) ────────
    if _NVIDIA_API_KEYS:
        result = _nvidia_generate(prompt, image, model="meta/llama-3.2-90b-vision-instruct", task="vision")
        if result:
            logger.info(f"NVIDIA Vision diagnosed ({len(result)} chars)")
            return result

    # ── Fallback to Ollama LLaVA (local, works offline) ────────────────
    try:
        import urllib.request
        import json as _json

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = _json.dumps({
            "model": "llava:7b",
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 500},
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode())
            text = data.get("response", "").strip()
            if text:
                logger.info(f"Ollama LLaVA vision response ({len(text)} chars)")
            return text
    except Exception as e:
        logger.warning(f"Ollama LLaVA vision failed: {e}")
        return ""


def analyze_unknown_crop_pil(image, language: str = "en") -> dict[str, Any]:
    """
    Vision fallback using PIL Image directly.
    Priority: Gemini Vision → Ollama LLaVA (local vision) → error fallback.
    """
    vision_prompt = """You are an expert Indian agricultural scientist.
Analyze this crop leaf image:
1. What crop is this?
2. Is it diseased? If yes, what disease?
3. How confident are you? (high/medium/low)
4. What treatment do you recommend for Indian farmers?

If unsure, say so honestly. Recommend visiting nearest KVK.
DO NOT recommend banned pesticides (Endosulfan, Monocrotophos, etc.)

Respond in JSON format ONLY:
{"crop": "...", "disease": "...", "confidence": "high/medium/low", "treatment": "...", "prevention": "..."}"""

    # ── Try Gemini Vision first ────────────────────────────────────────
    _get_gemini_model()
    if _GEMINI_AVAILABLE is True:
        try:
            text = _generate(vision_prompt, image=image)
            if text:
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                result = json.loads(text)
                result["is_fallback"] = True
                result["source"] = "gemini_vision"
                logger.info(f"Gemini Vision: crop={result.get('crop')}, disease={result.get('disease')}")
                return result
        except Exception as e:
            logger.warning(f"Gemini Vision failed: {e}")

    # ── Fallback: Ollama LLaVA (local vision — can SEE any image) ──────
    try:
        text = _ollama_vision_analyze(image, vision_prompt)
        if text:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            # Try to parse as JSON
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                # LLaVA sometimes returns plain text — extract key info
                import re
                crop_m = re.search(r'crop["\s:]+([a-zA-Z]+)', text, re.IGNORECASE)
                disease_m = re.search(r'disease["\s:]+([^",}\n]+)', text, re.IGNORECASE)
                result = {
                    "crop": crop_m.group(1).strip() if crop_m else "unknown",
                    "disease": disease_m.group(1).strip() if disease_m else "unknown",
                    "confidence": "medium",
                    "treatment": text[:300],
                }
            result["is_fallback"] = True
            result["source"] = "ollama_llava_vision"
            logger.info(f"LLaVA Vision: crop={result.get('crop')}, disease={result.get('disease')}")
            return result
    except Exception as e:
        logger.warning(f"LLaVA Vision fallback failed: {e}")

    return {
        "crop": "unknown", "disease": "analysis_failed", "confidence": "low",
        "treatment": "Visit nearest KVK.", "is_fallback": True,
        "source": "all_vision_failed",
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
    if not is_genai_available():
        return "AI not available. Please consult your local KVK."

    lang_names = {
        "en": "English", "hi": "Hindi", "kn": "Kannada",
        "te": "Telugu", "ta": "Tamil", "pa": "Punjabi",
        "gu": "Gujarati", "mr": "Marathi", "bn": "Bengali",
        "or": "Odia",
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
    "get_configured_llm_providers",
    "is_genai_available",
    "generate_treatment_advice",
    "analyze_unknown_crop",
    "analyze_unknown_crop_pil",
    "generate_audit_narrative",
    "conversational_followup",
]
