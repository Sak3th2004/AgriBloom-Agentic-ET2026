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
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from agents.react_orchestrator import run_react_orchestrator
from agents.react_tools import REGISTRY, run_tool
from graph.state import COMPLIANCE, KNOWLEDGE, OUTPUT, VISION

logger = logging.getLogger(__name__)

ORCH = "orchestrator"


def _node(action: str):
    """Build a LangGraph node that executes one tool by name."""

    def _run(state: dict) -> dict:
        return run_tool(action, state)

    _run.__name__ = f"tool_{action}"
    return _run


def _route(state: dict) -> str:
    """Conditional-edge selector: read the orchestrator's chosen action."""
    action = state.get("next_action", OUTPUT)
    return action if action in REGISTRY else OUTPUT


def build_react_graph() -> Any:
    """Compile the ReAct conditional-edge state graph."""
    logger.info("Building AgriBloom ReAct graph...")
    graph = StateGraph(dict)

    graph.add_node(ORCH, run_react_orchestrator)
    for action in (VISION, KNOWLEDGE, COMPLIANCE, OUTPUT):
        graph.add_node(action, _node(action))

    graph.set_entry_point(ORCH)

    # Orchestrator dynamically routes to the chosen tool.
    graph.add_conditional_edges(
        ORCH,
        _route,
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
