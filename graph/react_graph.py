"""
ReAct LangGraph with conditional edges.

Flow (a true agentic loop, not a fixed chain):

    orchestrator ──(next_action)──▶ vision ─────┐
                 │                  knowledge ───┤
                 │                  compliance ──┤
                 │                               └─▶ back to orchestrator
                 └──(next_action == output)────▶ output ──▶ END

The orchestrator picks the next action each turn (see
:func:`agents.react_orchestrator.run_react_orchestrator`). Tool nodes run and
return to the orchestrator; the loop ends when the orchestrator chooses
``output``. This is kept separate from V1's linear graph in ``main.py`` — enable
it with ``AGRIBLOOM_USE_REACT=1`` — so the winning pipeline is never disturbed.

Set ``use_reflexion=True`` (the backend API's default — see
``backend/pipeline.py``) to additionally swap in Reflexion self-correcting
compliance (Phase 4). This never mutates the shared tool registry: a fresh
registry is built per graph so the plain graph and the reflexion graph can
coexist in the same process (e.g. Gradio using plain, the API using reflexion).
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from agents import react_tools
from agents.react_orchestrator import run_react_orchestrator
from graph.state import COMPLIANCE, KNOWLEDGE, OUTPUT, VISION

logger = logging.getLogger(__name__)

ORCH = "orchestrator"


def _node(action: str, registry: dict):
    """Build a LangGraph node that executes one tool by name from ``registry``."""

    def _run(state: dict) -> dict:
        return react_tools.run_tool(action, state, registry=registry)

    _run.__name__ = f"tool_{action}"
    return _run


def _route(registry: dict):
    """Build a conditional-edge selector bound to ``registry``."""

    def _pick(state: dict) -> str:
        action = state.get("next_action", OUTPUT)
        return action if action in registry else OUTPUT

    return _pick


def build_react_graph(use_reflexion: bool = False) -> Any:
    """Compile the ReAct conditional-edge state graph.

    Args:
        use_reflexion: when True, compliance runs through the Reflexion
            self-correction loop (Phase 4) instead of the plain deterministic
            check. The deterministic guardrail still has final say either way.
    """
    logger.info("Building AgriBloom ReAct graph (use_reflexion=%s)...", use_reflexion)
    # Resolved at call time (module-attribute access) so test monkeypatching of
    # agents.react_tools.REGISTRY is respected even for the default registry.
    registry = react_tools.build_registry(use_reflexion) if use_reflexion else react_tools.REGISTRY

    graph = StateGraph(dict)

    graph.add_node(ORCH, run_react_orchestrator)
    for action in (VISION, KNOWLEDGE, COMPLIANCE, OUTPUT):
        graph.add_node(action, _node(action, registry))

    graph.set_entry_point(ORCH)

    # Orchestrator dynamically routes to the chosen tool.
    graph.add_conditional_edges(
        ORCH,
        _route(registry),
        {VISION: VISION, KNOWLEDGE: KNOWLEDGE, COMPLIANCE: COMPLIANCE, OUTPUT: OUTPUT},
    )

    # Every non-terminal tool loops back to the orchestrator for the next decision.
    graph.add_edge(VISION, ORCH)
    graph.add_edge(KNOWLEDGE, ORCH)
    graph.add_edge(COMPLIANCE, ORCH)
    # Output is terminal.
    graph.add_edge(OUTPUT, END)

    compiled = graph.compile()
    logger.info("ReAct graph compiled")
    return compiled
