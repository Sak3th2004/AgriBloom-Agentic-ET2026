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
