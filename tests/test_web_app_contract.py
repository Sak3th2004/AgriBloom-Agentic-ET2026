from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_web_app_files_exist():
    for relative in [
        "web_app/index.html",
        "web_app/styles.css",
        "web_app/app.js",
        "web_app/manifest.webmanifest",
        "web_app/service-worker.js",
        "api/app.py",
    ]:
        assert (ROOT / relative).exists(), f"{relative} missing"


def test_frontend_calls_farmer_advice_api():
    script = (ROOT / "web_app/app.js").read_text(encoding="utf-8")
    assert 'fetch("/api/analyze"' in script
    assert "farmer_advice" in script
    assert "renderAdvice" in script


def test_api_exposes_required_routes():
    source = (ROOT / "api/app.py").read_text(encoding="utf-8")
    assert '@app.get("/api/health")' in source
    assert '@app.post("/api/analyze")' in source
    assert "farmer_advice" in source


def test_api_health_route_boots():
    from fastapi.testclient import TestClient

    from api.app import app

    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["farmer_advice_schema"] == "1.0"


def test_web_app_served_from_root():
    from fastapi.testclient import TestClient

    from api.app import app

    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "AgriBloom Crop Health" in response.text
