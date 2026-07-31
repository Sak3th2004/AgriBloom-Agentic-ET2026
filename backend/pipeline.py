"""
The V2 pipeline used by the FastAPI backend: ReAct orchestrator (Phase 3) +
Reflexion self-correcting compliance (Phase 4), with the vision tool already
layering the ensemble (Phase 1) and knowledge running hybrid RAG (Phase 2).

This is deliberately a separate entry point from ``main.run_pipeline`` (which
stays on V1's linear graph for Gradio) — the backend API is where the full
advanced agentic pipeline is switched on by default, without touching or
risking the proven V1 path. Both share the exact same initial-state shape via
``main.build_initial_state`` so they never drift apart.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from main import build_initial_state

logger = logging.getLogger(__name__)

_V2_GRAPH: Optional[Any] = None


def get_v2_graph() -> Any:
    """Lazily build and cache the ReAct + Reflexion graph."""
    global _V2_GRAPH
    if _V2_GRAPH is None:
        from graph.react_graph import build_react_graph

        _V2_GRAPH = build_react_graph(use_reflexion=True)
    return _V2_GRAPH


def run_v2_pipeline(**kwargs: Any) -> dict[str, Any]:
    """Run the V2 agentic pipeline; same signature/shape as ``main.run_pipeline``."""
    start_time = time.time()
    initial_state = build_initial_state(**kwargs)

    logger.info(
        "V2 pipeline started: lang=%s, offline=%s, has_image=%s",
        initial_state["lang"], initial_state["offline"], initial_state["image"] is not None,
    )

    # The ReAct graph's tool set is vision/knowledge/compliance/output — it has
    # no NLU step, so language/crop/intent detection must run once up front
    # (same detector V1's orchestrator_agent uses) or crop_type would stay
    # unset for text-only queries and every downstream lookup would silently
    # default (e.g. knowledge_agent falling back to "maize").
    try:
        from agents.orchestrator_agent import run_orchestrator

        initial_state = run_orchestrator(initial_state)
    except Exception as e:
        logger.warning("V2 pipeline: NLU pre-step failed (%s) — continuing without it", e)

    try:
        final_state = get_v2_graph().invoke(initial_state)
        elapsed = time.time() - start_time
        final_state["elapsed_seconds"] = elapsed
        logger.info(
            "V2 pipeline completed: status=%s, elapsed=%.2fs",
            final_state.get("status", "unknown"), elapsed,
        )
        return final_state
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("V2 pipeline failed after %.2fs: %s", elapsed, e, exc_info=True)
        return {
            **initial_state,
            "status": "error",
            "error": str(e),
            "elapsed_seconds": elapsed,
            "final_response": f"Error: {e}",
        }
