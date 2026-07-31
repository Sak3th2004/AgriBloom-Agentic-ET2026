"""
Regression test for a real bug found during Phase-7 integration testing:
orchestrator_agent detected `detected_crop` from text but never populated
`crop_type`, which knowledge_agent reads — so text-only queries always
silently fell back to a default crop ("maize") regardless of what the farmer
actually asked about.
"""
from __future__ import annotations

from agents.orchestrator_agent import run_orchestrator


def test_crop_type_bridged_from_text_for_text_only_query():
    state = {"user_text": "my rice leaves have brown spots", "user_language": "en",
              "image": None, "offline": False, "chat_history": []}
    out = run_orchestrator(state)
    assert out["detected_crop"] == "rice"
    assert out["crop_type"] == "rice"


def test_crop_type_not_overwritten_when_vision_already_set_it():
    # Vision (from an image) should take priority over text-detected crop.
    state = {"user_text": "wheat rust also present", "user_language": "en",
              "image": None, "offline": False, "chat_history": [], "crop_type": "grape"}
    out = run_orchestrator(state)
    assert out["crop_type"] == "grape"  # unchanged
    assert out["detected_crop"] == "wheat"  # still detected, just not applied


def test_no_crop_mentioned_leaves_crop_type_unset():
    state = {"user_text": "what is the weather today", "user_language": "en",
              "image": None, "offline": False, "chat_history": []}
    out = run_orchestrator(state)
    assert not out.get("crop_type")
