"""
Serialize the internal pipeline state into the frontend API contract (§6).

The pipeline state is a rich dict (see KNOWLEDGE_GRAPH §11); the frontend expects
flattened shapes. These functions are the single, pure, unit-tested mapping
between the two — the integration glue documented in FRONTEND_PLAN.md §B3. No I/O
here, so it's trivially testable.
"""
from __future__ import annotations

import os
from typing import Any, Optional

UNCERTAIN_LABELS = {"uncertain_detection", "unknown", "error"}
_SOURCE_ENUM = {"ensemble", "efficientnet", "dinov2_knn", "llava", "fallback"}


def _pretty(label: str) -> str:
    return label.replace("___", " ").replace("_", " ").strip().title()


def _display_name(label: str, lang: str) -> str:
    """Localized disease name; falls back to a prettified label."""
    try:
        from agents.output_agent import _get_disease_name

        return _get_disease_name(label, lang)
    except Exception:
        return _pretty(label)


def _file_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"/api/v1/files/{os.path.basename(path)}"


def _map_source(source: str) -> str:
    if not source:
        return "fallback"
    for key in _SOURCE_ENUM:
        if key in source:
            return key
    return "fallback"


def serialize_disease(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    pred = state.get("disease_prediction") or {}
    label = pred.get("label")
    if not label:
        return None
    lang = state.get("lang", "en")
    confidence = float(pred.get("confidence", 0.0))
    is_uncertain = label in UNCERTAIN_LABELS or confidence < 0.45
    crop = pred.get("crop") or state.get("crop_type") or (label.split("_")[0] if "_" in label else "")

    top3 = []
    for item in pred.get("top3", [])[:3]:
        lbl = item.get("label", "")
        top3.append({
            "label": lbl,
            "display_name": _display_name(lbl, lang),
            "confidence": float(item.get("confidence", 0.0)),
        })

    return {
        "label": label,
        "display_name": _display_name(label, lang),
        "crop": crop,
        "confidence": confidence,
        "is_uncertain": is_uncertain,
        "top3": top3,
        "source": _map_source(str(pred.get("source", ""))),
    }


def serialize_compliance(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    comp = state.get("compliance")
    if not comp:
        return None
    violations = comp.get("violations", []) or []
    blocked = [
        (v.get("chemical") if isinstance(v, dict) else str(v))
        for v in violations
    ]
    # safe_alternatives is a dict {chem: [{name,...}]} -> unique flat name list.
    alts_out: list[str] = []
    seen = set()
    for alts in (comp.get("safe_alternatives") or {}).values():
        for a in alts:
            name = a.get("name") if isinstance(a, dict) else str(a)
            if name and name not in seen:
                seen.add(name)
                alts_out.append(name)
    disclaimers = comp.get("disclaimers") or []
    return {
        "allowed": bool(comp.get("allowed", True)),
        "risk_level": comp.get("risk_level", "low"),
        "status": comp.get("compliance_status", "safe"),
        "blocked_substances": [b for b in blocked if b],
        "safe_alternatives": alts_out,
        "disclaimer": disclaimers[0] if disclaimers else "",
    }


def serialize_weather(weather: Optional[dict]) -> Optional[dict[str, Any]]:
    if not weather:
        return None
    return {
        "temp_c": float(weather.get("temp_c", 0)),
        "humidity": float(weather.get("humidity", 0)),
        "rain_mm": float(weather.get("rain_mm", 0)),
        "desc": weather.get("weather_desc") or weather.get("desc") or "",
        "is_live": bool(weather.get("is_live", False)),
    }


def serialize_market(market: Optional[dict]) -> Optional[dict[str, Any]]:
    if not market:
        return None
    return {
        "crop": market.get("crop", ""),
        "modal_price": float(market.get("modal_price", 0)),
        "unit": market.get("unit", "quintal"),
        "mandi": market.get("mandi") or market.get("name") or "Local",
    }


def build_chart(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    """A simple 14-day recovery projection from the diagnosis (raw arrays)."""
    pred = state.get("disease_prediction") or {}
    label = pred.get("label", "")
    if not label or label in UNCERTAIN_LABELS:
        return None
    healthy = "healthy" in label.lower()
    start = 85.0 if healthy else 62.0
    days = list(range(14))
    without = []
    with_t = []
    for d in days:
        # Untreated declines; treated recovers toward ~92%.
        without.append(round(max(20.0, start - (0 if healthy else d * 3.2)), 1))
        with_t.append(round(min(94.0, start + d * (0.8 if healthy else 2.6)), 1))
    return {"days": days, "without_treatment": without, "with_treatment": with_t}


def _status(disease: Optional[dict], compliance: Optional[dict], state: dict) -> str:
    if (state.get("disease_prediction") or {}).get("label") == "invalid_image":
        return "invalid_image"
    if compliance and not compliance["allowed"]:
        return "blocked"
    if disease and disease["is_uncertain"]:
        return "uncertain"
    if state.get("status") == "error":
        return "error"
    return "ok"


def serialize_diagnosis(
    state: dict[str, Any],
    diag_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Full §6.1 DiagnoseResponse from a completed pipeline state."""
    lang = state.get("lang", state.get("user_language", "en"))
    disease = serialize_disease(state)
    compliance = serialize_compliance(state)
    knowledge = state.get("knowledge") or {}

    return {
        "id": diag_id,
        "session_id": session_id,
        "language": lang,
        "status": _status(disease, compliance, state),
        "disease": disease,
        "treatment": state.get("treatment") or None,
        "recommendations": state.get("recommendations", []) or [],
        "compliance": compliance,
        "knowledge": {
            "weather": serialize_weather(knowledge.get("weather")),
            "market": serialize_market(knowledge.get("market")),
        },
        "chart": build_chart(state),
        "audio_url": _file_url(state.get("voice_output_path")),
        "pdf_url": _file_url(state.get("audit_pdf_path")),
        "elapsed_seconds": float(state.get("elapsed_seconds", 0.0)),
    }
