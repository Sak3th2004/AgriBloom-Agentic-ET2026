"""Unit tests for the API serializer (backend/serializers.py) — pure mapping."""
from __future__ import annotations

from backend.serializers import (
    build_chart,
    serialize_compliance,
    serialize_diagnosis,
    serialize_disease,
    serialize_market,
    serialize_weather,
)


def _state(**over):
    base = {
        "lang": "en",
        "crop_type": "grape",
        "disease_prediction": {
            "label": "grape_downy_mildew",
            "confidence": 0.85,
            "source": "ensemble",
            "top3": [
                {"label": "grape_downy_mildew", "confidence": 0.85},
                {"label": "grape_black_rot", "confidence": 0.10},
            ],
        },
        "treatment": "Use copper oxychloride 3g/L.",
        "recommendations": ["Prune affected leaves", "Improve drainage"],
        "compliance": {
            "allowed": True,
            "risk_level": "low",
            "compliance_status": "safe",
            "violations": [],
            "safe_alternatives": {},
            "disclaimers": ["Consult your local KVK."],
        },
        "knowledge": {
            "weather": {"temp_c": 27, "humidity": 60, "rain_mm": 2, "weather_desc": "Overcast", "is_live": True},
            "market": {"crop": "grape", "modal_price": 4500, "unit": "quintal", "mandi": "Bengaluru"},
        },
        "elapsed_seconds": 3.1,
    }
    base.update(over)
    return base


def test_serialize_disease_basic():
    d = serialize_disease(_state())
    assert d["label"] == "grape_downy_mildew"
    assert d["crop"] == "grape"
    assert d["is_uncertain"] is False
    assert d["source"] == "ensemble"
    assert len(d["top3"]) == 2
    assert all("display_name" in t for t in d["top3"])


def test_serialize_disease_uncertain_low_conf():
    d = serialize_disease(_state(disease_prediction={"label": "grape_downy_mildew", "confidence": 0.2, "top3": []}))
    assert d["is_uncertain"] is True


def test_serialize_disease_none_when_missing():
    assert serialize_disease({"disease_prediction": {}}) is None


def test_serialize_compliance_flattens_blocked_and_alts():
    state = _state(compliance={
        "allowed": False, "risk_level": "high", "compliance_status": "unsafe",
        "violations": [{"chemical": "Endosulfan", "status": "BANNED"}],
        "safe_alternatives": {"Endosulfan": [{"name": "Neem oil"}, {"name": "Trichoderma"}]},
        "disclaimers": ["Consult KVK."],
    })
    c = serialize_compliance(state)
    assert c["allowed"] is False
    assert c["blocked_substances"] == ["Endosulfan"]
    assert c["safe_alternatives"] == ["Neem oil", "Trichoderma"]
    assert c["disclaimer"] == "Consult KVK."


def test_serialize_weather_prefers_desc_and_is_live():
    w = serialize_weather({"temp_c": 27, "humidity": 60, "rain_mm": 0, "weather_desc": "Clear sky", "is_live": True})
    assert w["desc"] == "Clear sky"
    assert w["is_live"] is True


def test_serialize_market():
    m = serialize_market({"crop": "rice", "modal_price": 2100, "unit": "quintal", "mandi": "Nashik"})
    assert m["crop"] == "rice" and m["modal_price"] == 2100.0


def test_build_chart_shape():
    ch = build_chart(_state())
    assert len(ch["days"]) == 14
    assert len(ch["with_treatment"]) == 14
    # Treated should end higher than untreated.
    assert ch["with_treatment"][-1] > ch["without_treatment"][-1]


def test_build_chart_none_for_uncertain():
    assert build_chart(_state(disease_prediction={"label": "uncertain_detection", "confidence": 0.2})) is None


def test_full_diagnosis_status_ok():
    out = serialize_diagnosis(_state(), "diag_1", "sess_1")
    assert out["status"] == "ok"
    assert out["id"] == "diag_1"
    assert out["disease"]["label"] == "grape_downy_mildew"
    assert out["knowledge"]["weather"]["is_live"] is True


def test_full_diagnosis_status_blocked():
    out = serialize_diagnosis(_state(compliance={
        "allowed": False, "risk_level": "high", "compliance_status": "unsafe",
        "violations": [{"chemical": "Endosulfan", "status": "BANNED"}],
        "safe_alternatives": {}, "disclaimers": [],
    }), "d", "s")
    assert out["status"] == "blocked"


def test_full_diagnosis_status_invalid_image():
    out = serialize_diagnosis(_state(disease_prediction={"label": "invalid_image", "confidence": 0.0}), "d", "s")
    assert out["status"] == "invalid_image"
