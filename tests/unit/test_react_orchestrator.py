"""Unit tests for the ReAct orchestrator, state helpers, and conditional graph.

All tests run without any real LLM or ML model: the LLM decision function is
either injected as a mock or disabled so routing falls back to the deterministic
policy, and tool nodes are replaced with lightweight stubs.
"""
from __future__ import annotations

import agents.react_orchestrator as orch
from agents.react_orchestrator import decide_next_action, parse_action
from graph.state import (
    COMPLIANCE,
    KNOWLEDGE,
    OUTPUT,
    VISION,
    deterministic_next,
    iteration_guard,
    mark_done,
    valid_actions,
)


# ── state helpers ──────────────────────────────────────────────────────────────
def test_valid_actions_progression_with_image():
    s = {"image": object()}
    assert VISION in valid_actions(s)
    assert KNOWLEDGE in valid_actions(s)
    assert COMPLIANCE not in valid_actions(s)  # needs knowledge first

    s = mark_done(s, VISION)
    s = mark_done(s, KNOWLEDGE)
    va = valid_actions(s)
    assert COMPLIANCE in va and OUTPUT in va and VISION not in va


def test_valid_actions_text_only_has_no_vision():
    s = {"user_text": "my wheat has rust"}
    assert VISION not in valid_actions(s)
    assert KNOWLEDGE in valid_actions(s)


def test_deterministic_order():
    s = {"image": object()}
    assert deterministic_next(s) == VISION
    s = mark_done(s, VISION)
    assert deterministic_next(s) == KNOWLEDGE
    s = mark_done(s, KNOWLEDGE)
    assert deterministic_next(s) == COMPLIANCE
    s = mark_done(s, COMPLIANCE)
    assert deterministic_next(s) == OUTPUT


def test_iteration_guard_trips():
    s = {"_react_iterations": 8}
    s2, exceeded = iteration_guard(s)
    assert exceeded is True
    assert s2["_react_iterations"] == 9


# ── action parsing ─────────────────────────────────────────────────────────────
def test_parse_action_from_json():
    allowed = [VISION, KNOWLEDGE, OUTPUT]
    assert parse_action('{"thought":"look","action":"vision"}', allowed) == VISION


def test_parse_action_from_fenced_json():
    allowed = [KNOWLEDGE, OUTPUT]
    txt = "```json\n{\"action\": \"knowledge\"}\n```"
    assert parse_action(txt, allowed) == KNOWLEDGE


def test_parse_action_bare_word_fallback():
    allowed = [COMPLIANCE, OUTPUT]
    assert parse_action("I think we should run compliance now.", allowed) == COMPLIANCE


def test_parse_action_rejects_out_of_scope():
    allowed = [OUTPUT]
    # 'vision' is not allowed here -> must return None (caller will fall back).
    assert parse_action('{"action":"vision"}', allowed) is None


def test_parse_action_garbage_returns_none():
    assert parse_action("no idea", [VISION, KNOWLEDGE]) is None


# ── decide_next_action ─────────────────────────────────────────────────────────
def test_decide_uses_llm_choice_when_valid():
    s = {"image": object()}  # vision + knowledge valid

    def gen(prompt):
        return '{"thought":"get facts first","action":"knowledge"}'

    assert decide_next_action(s, generate=gen) == KNOWLEDGE


def test_decide_falls_back_on_bad_llm_output():
    s = {"image": object()}

    def gen(prompt):
        return "banana boat"  # unparseable

    # Falls back to deterministic -> vision first.
    assert decide_next_action(s, generate=gen) == VISION


def test_decide_single_valid_action_shortcuts_llm():
    calls = []

    def gen(prompt):
        calls.append(prompt)
        return '{"action":"vision"}'

    s = {"user_text": "hi"}  # only knowledge valid at start
    s = mark_done(s, KNOWLEDGE)
    s = mark_done(s, COMPLIANCE)  # now only output valid
    assert decide_next_action(s, generate=gen) == OUTPUT
    assert calls == []  # LLM not consulted when there's a single option


def test_decide_llm_error_falls_back(monkeypatch):
    s = {"image": object()}

    def boom(prompt):
        raise RuntimeError("network down")

    assert decide_next_action(s, generate=boom) == VISION


# ── full ReAct graph with stub tools ───────────────────────────────────────────
def test_react_graph_runs_to_output(monkeypatch):
    from dataclasses import dataclass

    import agents.react_tools as rt

    order: list[str] = []

    @dataclass
    class _StubTool:
        name: str
        description: str = "stub"

        def run(self, state):
            order.append(self.name)
            return state

    stub_registry = {a: _StubTool(a) for a in (VISION, KNOWLEDGE, COMPLIANCE, OUTPUT)}
    monkeypatch.setattr(rt, "REGISTRY", stub_registry)
    # Force deterministic routing (no live LLM).
    monkeypatch.setattr(orch, "_default_generate", lambda: None)

    from graph.react_graph import build_react_graph

    g = build_react_graph()
    final = g.invoke({"image": object(), "user_text": "spots on leaf"})

    assert order == [VISION, KNOWLEDGE, COMPLIANCE, OUTPUT]
    assert final.get("next_action") == OUTPUT
