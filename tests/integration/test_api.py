"""TestClient tests for the FastAPI backend — pipeline mocked (no ML models)."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from backend.app import app, set_pipeline


@pytest.fixture()
def client():
    def fake_pipeline(**kwargs):
        # Mimic a completed pipeline state.
        return {
            "lang": kwargs.get("user_language", "en"),
            "crop_type": "grape",
            "disease_prediction": {
                "label": "grape_downy_mildew", "confidence": 0.85, "source": "ensemble",
                "top3": [{"label": "grape_downy_mildew", "confidence": 0.85}],
            },
            "treatment": "Use copper oxychloride 3g/L.",
            "recommendations": ["Prune affected leaves"],
            "compliance": {
                "allowed": True, "risk_level": "low", "compliance_status": "safe",
                "violations": [], "safe_alternatives": {}, "disclaimers": ["Consult KVK."],
            },
            "knowledge": {"weather": {"temp_c": 27, "humidity": 60, "rain_mm": 0,
                                      "weather_desc": "Overcast", "is_live": True},
                          "market": {"crop": "grape", "modal_price": 4500, "unit": "quintal", "mandi": "Bengaluru"}},
            "final_response": "",  # empty -> skip voice synth in test
            "elapsed_seconds": 2.0,
        }

    set_pipeline(fake_pipeline)
    yield TestClient(app)
    set_pipeline(None)


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_languages_has_ten(client):
    r = client.get("/api/v1/languages")
    data = r.json()
    assert len(data) == 10
    assert {"code", "name", "native"} <= set(data[0].keys())


def test_meta_shape(client):
    r = client.get("/api/v1/meta")
    body = r.json()
    assert "crops" in body and "quick_symptoms" in body and "districts" in body


def test_diagnose_text_only(client):
    r = client.post("/api/v1/diagnose", data={"text": "my grape has black spots", "language": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["disease"]["label"] == "grape_downy_mildew"
    assert body["compliance"]["allowed"] is True
    assert body["knowledge"]["weather"]["is_live"] is True
    assert body["id"].startswith("diag_")


def test_diagnose_empty_input_400(client):
    r = client.post("/api/v1/diagnose", data={"text": "", "language": "en"})
    assert r.status_code == 400


def test_diagnose_with_image(client):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (0, 128, 0)).save(buf, format="JPEG")
    buf.seek(0)
    r = client.post(
        "/api/v1/diagnose",
        data={"text": "", "language": "en"},
        files={"image": ("leaf.jpg", buf, "image/jpeg")},
    )
    assert r.status_code == 200
    assert r.json()["disease"]["crop"] == "grape"


def test_chat_requires_question(client):
    r = client.post("/api/v1/chat", json={"question": "", "language": "en", "history": []})
    assert r.status_code == 400


def test_sse_stream(client):
    r = client.get("/api/v1/diagnose/stream?job=diag_1")
    assert r.status_code == 200
    assert "event: step" in r.text
    assert "event: done" in r.text


def test_file_not_found(client):
    r = client.get("/api/v1/files/nonexistent.mp3")
    assert r.status_code == 404


def test_transcribe_uses_voice_module(client, monkeypatch):
    # The endpoint does `from voice.stt import transcribe` at call time, so
    # patching the module attribute is picked up.
    import voice.stt as stt

    monkeypatch.setattr(stt, "transcribe", lambda raw, language="auto": {"text": "namaste", "language": "hi"})
    r = client.post(
        "/api/v1/voice/transcribe",
        data={"language": "auto"},
        files={"audio": ("v.wav", io.BytesIO(b"RIFFfake"), "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["text"] == "namaste"
