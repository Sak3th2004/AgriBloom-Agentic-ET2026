"""
Farmer-facing advice contract.

This module converts the internal agent state into a stable, frontend-friendly
object. The goal is to keep the farmer experience simple and predictable while
still preserving technical details for reports, audit, and debugging.
"""
from __future__ import annotations

import re
from typing import Any, Callable


SCHEMA_VERSION = "1.0"
UNCERTAIN_LABELS = {
    "uncertain_detection",
    "unknown",
    "error",
    "invalid_image",
    "model_unavailable",
    "model_error",
}
HELPLINE = {"name": "Kisan Call Center", "number": "1800-180-1551"}


def _humanize(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    text = re.sub(r"^gemini[_\s-]+", "", text, flags=re.IGNORECASE)
    text = text.replace("___", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() if text else "Unknown"


def _clean_item(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^[\s>*\-•●]+", "", text)
    text = re.sub(r"^\d+[\).:-]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe(items: list[str], limit: int = 5) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        text = _clean_item(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _split_treatment(treatment: str, limit: int = 4) -> list[str]:
    if not treatment:
        return []

    lines: list[str] = []
    for raw_line in str(treatment).splitlines():
        line = _clean_item(raw_line)
        if not line:
            continue

        # Very long paragraphs are hard for farmers. Split only when a line is
        # clearly sentence-heavy; keep dosage lines intact.
        if len(line) > 180:
            parts = re.split(r"(?<=[.!?])\s+", line)
            lines.extend(_clean_item(part) for part in parts if _clean_item(part))
        else:
            lines.append(line)

    return _dedupe(lines, limit=limit)


def _flatten_safe_alternatives(safe_alternatives: Any) -> list[str]:
    if not isinstance(safe_alternatives, dict):
        return []

    flattened: list[str] = []
    for chemical, alternatives in safe_alternatives.items():
        if isinstance(alternatives, list):
            for alternative in alternatives[:3]:
                flattened.append(f"Instead of {chemical}, ask about {alternative}")
        elif alternatives:
            flattened.append(f"Instead of {chemical}, ask about {alternatives}")
    return _dedupe(flattened, limit=5)


def _infer_crop(state: dict[str, Any], label: str) -> str:
    crop = str(state.get("crop_type") or "").strip()
    if crop and crop.lower() not in {"unknown", "none", "error"}:
        return _humanize(crop)

    if "___" in label:
        return _humanize(label.split("___", 1)[0])
    if "_" in label:
        return _humanize(label.split("_", 1)[0])
    return "Unknown"


def _risk_from_agronomy(severity: str, confidence: float, allowed: bool) -> str:
    if not allowed:
        return "high"

    severity_norm = str(severity or "").strip().lower()
    if severity_norm in {"critical", "severe"}:
        return "critical"
    if severity_norm == "high":
        return "high"
    if severity_norm == "medium":
        return "medium"
    if severity_norm in {"low", "none", "healthy"}:
        return "low"
    if confidence and confidence < 0.55:
        return "unknown"
    return "medium"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.55:
        return "medium"
    if confidence > 0:
        return "low"
    return "unknown"


def _default_next_questions(status: str) -> list[str]:
    if status == "needs_clear_photo":
        return [
            "Can you upload one close-up photo of the affected leaf in daylight?",
            "Can you also upload one photo of the full plant?",
        ]

    return [
        "How many days have you seen this problem?",
        "Is it spreading to new leaves or nearby plants?",
        "Did you spray any chemical or fertilizer recently?",
        "Was there rain or heavy watering in the last few days?",
    ]


def build_farmer_advice(
    state: dict[str, Any],
    disease_name: str | None = None,
    disease_name_fn: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    """
    Build the stable farmer-advice object consumed by UI, mobile, voice, and PDF.

    The returned object intentionally uses plain English field names. Localized
    labels should be handled by the frontend, while values can still come from
    translated treatment/recommendation text when upstream agents provide it.
    """
    lang = state.get("lang", state.get("user_language", "en")) or "en"
    disease = state.get("disease_prediction", {}) or {}
    label = str(disease.get("label") or "unknown").strip().lower()
    confidence = float(disease.get("confidence") or 0.0)
    compliance = state.get("compliance", {}) or {}
    knowledge = state.get("knowledge", {}) or {}
    agronomy = knowledge.get("agronomy", {}) or {}
    weather = knowledge.get("weather", {}) or {}
    market = knowledge.get("market", {}) or {}
    treatment = str(state.get("treatment") or "").strip()
    recommendations = state.get("recommendations", []) or []

    allowed = bool(compliance.get("allowed", True))
    crop = _infer_crop(state, label)
    problem = disease_name or ""
    if not problem and disease_name_fn:
        problem = disease_name_fn(label, lang)
    if not problem:
        problem = _humanize(label)

    if not allowed:
        status = "blocked"
    elif label in UNCERTAIN_LABELS:
        status = "needs_clear_photo"
    elif "healthy" in label:
        status = "healthy"
    else:
        status = "ready"

    risk_level = _risk_from_agronomy(
        str(agronomy.get("severity", "")),
        confidence,
        allowed,
    )
    if status == "needs_clear_photo":
        risk_level = "unknown"
    elif status == "healthy":
        risk_level = "low"

    treatment_actions = _split_treatment(treatment, limit=4)
    agronomy_actions = agronomy.get("actions", []) or []
    today_actions = _dedupe(
        [*agronomy_actions, *recommendations, *treatment_actions],
        limit=5,
    )

    if status == "blocked":
        today_actions = [
            "Do not use the blocked or restricted chemical.",
            "Ask your local agriculture officer or KVK for an approved alternative.",
            "Use protective clothing if you already handled the chemical.",
        ]
    elif status == "needs_clear_photo":
        today_actions = [
            "Take another photo in daylight.",
            "Keep the affected leaf close to the camera.",
            "Make sure spots, insects, or color changes are clearly visible.",
            "Add one full-plant photo if the problem is on many leaves.",
        ]
    elif status == "healthy":
        today_actions = _dedupe(
            [
                "Continue regular field monitoring every 5 to 7 days.",
                "Keep irrigation balanced and avoid water stress.",
                "Remove weeds and old infected plant debris from the field.",
                *today_actions,
            ],
            limit=5,
        )
    elif not today_actions:
        today_actions = [
            "Isolate badly affected leaves or plants if symptoms are spreading.",
            "Avoid watering directly on leaves.",
            "Keep the field clean and improve air movement around plants.",
            "Consult a local agriculture officer before spraying chemicals.",
        ]

    what_not_to_do = [
        "Do not spray banned, unlabelled, or unknown chemicals.",
        "Do not mix two or more pesticides without expert advice.",
        "Do not spray during strong wind, rain, or peak afternoon heat.",
    ]
    if "blight" in label or "rot" in label or "fung" in label or "mildew" in label:
        what_not_to_do.append("Do not compost infected leaves or stems.")
    if weather.get("rain_mm", 0) or weather.get("forecast_3day_rain", 0):
        what_not_to_do.append("Do not spray fungicide just before expected rain.")

    safe_alternatives = _flatten_safe_alternatives(compliance.get("safe_alternatives"))
    if status == "blocked" and safe_alternatives:
        treatment_guidance = " ".join(safe_alternatives[:3])
    elif treatment_actions:
        treatment_guidance = " ".join(treatment_actions[:3])
    elif status == "needs_clear_photo":
        treatment_guidance = "Treatment advice needs a clearer crop photo first."
    elif status == "healthy":
        treatment_guidance = "No treatment is suggested from this photo. Continue prevention and regular monitoring."
    else:
        treatment_guidance = "Use only crop-approved treatment after confirming with a local agriculture expert."

    if status == "blocked":
        summary = "The requested or generated treatment may be unsafe or restricted. Use approved alternatives only."
    elif status == "needs_clear_photo":
        summary = "The crop problem cannot be identified reliably from this photo."
    elif status == "healthy":
        summary = f"{crop} looks healthy from the available photo. Continue regular monitoring."
    else:
        summary = f"Likely {problem} in {crop}. Start with the steps below today and monitor closely."

    when_to_call_expert = (
        "Call your local KVK or agriculture officer if symptoms spread in 2 to 3 days, "
        "the crop is wilting fast, or more than one-fourth of the field is affected."
    )
    if status == "blocked":
        when_to_call_expert = "Call a local agriculture officer or KVK before using any chemical treatment."
    elif status == "needs_clear_photo":
        when_to_call_expert = "Call a local agriculture officer if the crop is wilting, drying, or spreading quickly."

    return {
        "schema_version": SCHEMA_VERSION,
        "language": lang,
        "status": status,
        "crop": crop,
        "problem": problem,
        "risk_level": risk_level,
        "confidence": round(confidence, 4),
        "confidence_label": _confidence_label(confidence),
        "summary": summary,
        "what_to_do_today": today_actions,
        "treatment_guidance": treatment_guidance,
        "what_not_to_do": _dedupe(what_not_to_do, limit=5),
        "safe_alternatives": safe_alternatives,
        "when_to_call_expert": when_to_call_expert,
        "helpline": HELPLINE,
        "next_questions": _default_next_questions(status),
        "technical": {
            "disease_label": label,
            "model_source": disease.get("source", "unknown"),
            "vision_top3": disease.get("top3", []),
            "yield_loss_range": agronomy.get("yield_loss_range", "varies"),
            "weather": {
                "temp_c": weather.get("temp_c"),
                "rain_mm": weather.get("rain_mm"),
                "forecast_3day_rain": weather.get("forecast_3day_rain"),
                "source": weather.get("source"),
            },
            "market": {
                "modal_price": market.get("modal_price"),
                "mandi": market.get("mandi"),
                "price_trend": market.get("price_trend"),
            },
            "compliance": {
                "allowed": allowed,
                "risk_level": compliance.get("risk_level"),
                "violations": compliance.get("violations", []),
                "rule_version": compliance.get("rule_version"),
            },
        },
    }


def format_farmer_advice_for_text(advice: dict[str, Any]) -> str:
    """Render farmer_advice as clean plain text for current Gradio textbox/TTS."""
    lines = [
        "Crop Health Result",
        "",
        f"Summary: {advice.get('summary', '')}",
        f"Crop: {advice.get('crop', 'Unknown')}",
        f"Likely problem: {advice.get('problem', 'Unknown')}",
        f"Risk level: {str(advice.get('risk_level', 'unknown')).title()}",
    ]

    confidence = advice.get("confidence")
    if isinstance(confidence, (int, float)) and confidence > 0:
        lines.append(
            f"Confidence: {confidence:.0%} ({str(advice.get('confidence_label', 'unknown')).title()})"
        )

    def add_section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.extend(["", title])
        for index, item in enumerate(items, 1):
            lines.append(f"{index}. {item}")

    add_section("What to do today", advice.get("what_to_do_today", []))

    treatment = advice.get("treatment_guidance")
    if treatment:
        lines.extend(["", "Treatment guidance", str(treatment)])

    add_section("Do not do this", advice.get("what_not_to_do", []))

    expert = advice.get("when_to_call_expert")
    if expert:
        lines.extend(["", "When to call an expert", str(expert)])

    helpline = advice.get("helpline", {}) or {}
    if helpline.get("number"):
        lines.extend(["", f"Help: {helpline.get('name', 'Farmer helpline')} - {helpline['number']}"])

    add_section("Questions to answer next", advice.get("next_questions", []))

    return "\n".join(lines).strip()


__all__ = [
    "SCHEMA_VERSION",
    "build_farmer_advice",
    "format_farmer_advice_for_text",
]
