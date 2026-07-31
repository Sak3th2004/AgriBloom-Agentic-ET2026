"""
Integration-style unit tests: does the shared state builder (main.py) and the
downstream agents actually honor auto-detected offline mode, end to end at the
unit level (no real network, no real models)?
"""
from __future__ import annotations

import main
from agents.compliance_reflexion import run_compliance_reflexion
from agents.react_orchestrator import decide_next_action


# ── main.build_initial_state ────────────────────────────────────────────────
def test_poor_network_auto_upgrades_to_offline(monkeypatch):
    monkeypatch.setattr(main, "build_initial_state", main.build_initial_state)  # no-op, keep ref
    import utils.network as net

    monkeypatch.setattr(net, "is_network_healthy", lambda timeout=1.5: False)
    state = main.build_initial_state(user_text="hi", offline=False)
    assert state["offline"] is True


def test_healthy_network_stays_online(monkeypatch):
    import utils.network as net

    monkeypatch.setattr(net, "is_network_healthy", lambda timeout=1.5: True)
    state = main.build_initial_state(user_text="hi", offline=False)
    assert state["offline"] is False


def test_explicit_offline_request_skips_the_check_entirely(monkeypatch):
    import utils.network as net

    calls = {"n": 0}
    monkeypatch.setattr(net, "is_network_healthy", lambda timeout=1.5: calls.__setitem__("n", calls["n"] + 1) or True)
    state = main.build_initial_state(user_text="hi", offline=True)
    assert state["offline"] is True
    assert calls["n"] == 0  # explicit request never triggers a network probe


def test_auto_detect_can_be_disabled():
    # Caller explicitly opts out (e.g. tests, or a caller that already knows).
    state = main.build_initial_state(user_text="hi", offline=False, auto_detect_offline=False)
    assert state["offline"] is False


# ── downstream agents respect offline (no wasted network attempts) ─────────
def test_react_orchestrator_skips_llm_when_offline():
    calls = {"n": 0}

    def gen(prompt):
        calls["n"] += 1
        return '{"action":"knowledge"}'

    state = {"image": object(), "offline": True}  # vision+knowledge both valid -> would ask LLM
    from graph.state import KNOWLEDGE, VISION, deterministic_next

    result = decide_next_action(state, generate=gen)
    assert calls["n"] == 0  # LLM never called
    assert result == deterministic_next(state)
    assert result in (VISION, KNOWLEDGE)


def test_compliance_reflexion_skips_regeneration_when_offline(monkeypatch):
    import agents.compliance_reflexion as cr

    called = {"n": 0}

    def fake_regen(lang, crop):
        called["n"] += 1
        return lambda prev, fb: "should never run"

    monkeypatch.setattr(cr, "_default_regenerate", fake_regen)
    state = {"treatment": "Spray Endosulfan 2ml/L.", "crop_type": "cotton", "lang": "en", "offline": True}
    out = run_compliance_reflexion(state)
    assert called["n"] == 0  # regenerate() never even constructed
    assert out["compliance"]["allowed"] is False  # deterministic block still stands
    assert out["compliance"]["reflexion"]["reason"] == "offline"
