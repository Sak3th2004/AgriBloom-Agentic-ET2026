"""Unit tests for live-weather accuracy + honest flagging (knowledge_agent)."""
from __future__ import annotations

from agents.knowledge_agent import _fetch_weather, _wmo_desc


def test_wmo_desc_known_codes():
    assert _wmo_desc(0) == "Clear sky"
    assert _wmo_desc(61) == "Slight rain"
    assert _wmo_desc(95) == "Thunderstorm"


def test_wmo_desc_unknown_or_bad():
    assert _wmo_desc(1234) == ""
    assert _wmo_desc(None) == ""
    assert _wmo_desc("x") == ""


def test_offline_default_is_flagged_not_live():
    # Offline with no cache -> explicit estimate, never a silent guess.
    w = _fetch_weather(17.38, 78.48, offline=True)
    assert w["is_live"] is False
    assert "note" in w


def test_api_error_is_flagged_not_live(monkeypatch):
    # Force both API paths to fail -> must return an honest estimate flag.
    import agents.knowledge_agent as ka

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(ka.requests, "get", boom)
    monkeypatch.setattr(ka.os, "environ", {})  # no OWM key
    monkeypatch.setattr(ka.CACHE, "get", lambda *a, **k: None)  # no cache
    w = _fetch_weather(12.97, 77.59, offline=False)
    assert w["is_live"] is False
    assert w["source"] == "api_error"
    assert "note" in w


def test_offline_never_reports_a_cached_live_reading_as_live(monkeypatch):
    # Regression: a cached blob from a PAST live fetch still has is_live=True
    # stored in it. Serving it from the offline/cache path must force False —
    # it describes THIS response, not whether the value was once live. A
    # `setdefault` here was the actual bug (found via a real e2e run
    # simulating poor network): it only sets is_live when absent, so a
    # once-live cached reading kept claiming is_live=True forever.
    import agents.knowledge_agent as ka

    stale_but_marked_live = {
        "temp_c": 27, "humidity": 60, "rain_mm": 0,
        "weather_desc": "Clear sky", "is_live": True, "source": "open-meteo",
    }
    monkeypatch.setattr(ka.CACHE, "get", lambda *a, **k: dict(stale_but_marked_live))
    w = _fetch_weather(12.97, 77.59, offline=True)
    assert w["is_live"] is False
    assert w["source"] == "offline_cache"


def test_cache_fallback_never_reports_a_cached_live_reading_as_live(monkeypatch):
    import agents.knowledge_agent as ka

    def boom(*a, **k):
        raise RuntimeError("network down")

    stale_but_marked_live = {
        "temp_c": 27, "humidity": 60, "rain_mm": 0,
        "weather_desc": "Clear sky", "is_live": True, "source": "open-meteo",
    }
    monkeypatch.setattr(ka.requests, "get", boom)
    monkeypatch.setattr(ka.os, "environ", {})
    monkeypatch.setattr(ka.CACHE, "get", lambda *a, **k: dict(stale_but_marked_live))
    w = _fetch_weather(12.97, 77.59, offline=False)
    assert w["is_live"] is False
    assert w["source"] == "cache_fallback"
