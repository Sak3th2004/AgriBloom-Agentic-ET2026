"""
Shared state + helpers for the ReAct agent graph.

The pipeline state is a plain ``dict`` (kept from V1 to avoid Gradio's TypedDict
schema issues). This module centralises the small amount of *logic* the ReAct
orchestrator needs on top of that dict: which actions are currently valid, what
has already run, and a loop guard so the agent can never spin forever.

Actions the orchestrator can pick:
    vision      — analyse the crop image (only if an image is present)
    knowledge   — weather + market + agronomy + hybrid-RAG enrichment
    compliance  — regulatory guardrail (needs knowledge/treatment first)
    output      — render the final multilingual answer (terminal)
"""
from __future__ import annotations

from typing import Any

VISION = "vision"
KNOWLEDGE = "knowledge"
COMPLIANCE = "compliance"
OUTPUT = "output"

ALL_ACTIONS = (VISION, KNOWLEDGE, COMPLIANCE, OUTPUT)

# Safety net: never run more orchestrator turns than this.
MAX_ITERATIONS = 8


def _done(state: dict, key: str) -> bool:
    return bool(state.get(f"_{key}_done"))


def mark_done(state: dict, action: str) -> dict:
    """Return a new state with ``action`` flagged complete."""
    return {**state, f"_{action}_done": True}


def has_image(state: dict) -> bool:
    return state.get("image") is not None or bool(state.get("image_path"))


def valid_actions(state: dict) -> list[str]:
    """Actions that make sense to run next, given what's already happened.

    Encodes the data dependencies:
      * vision only if there's an image and it hasn't run
      * knowledge any time it hasn't run
      * compliance only after knowledge (it inspects treatment/recommendations)
      * output only once we have something to say (knowledge or a diagnosis)
    """
    actions: list[str] = []
    if has_image(state) and not _done(state, VISION):
        actions.append(VISION)
    if not _done(state, KNOWLEDGE):
        actions.append(KNOWLEDGE)
    if _done(state, KNOWLEDGE) and not _done(state, COMPLIANCE):
        actions.append(COMPLIANCE)
    # Output becomes available once we've gathered at least some information.
    ready = _done(state, KNOWLEDGE) or _done(state, VISION)
    if ready and not _done(state, OUTPUT):
        actions.append(OUTPUT)
    return actions


def deterministic_next(state: dict) -> str:
    """Fallback policy when no LLM is available: a sensible fixed order.

    vision (if image) -> knowledge -> compliance -> output.
    """
    actions = valid_actions(state)
    for preferred in (VISION, KNOWLEDGE, COMPLIANCE, OUTPUT):
        if preferred in actions:
            return preferred
    return OUTPUT  # always terminate on output


def iteration_guard(state: dict) -> tuple[dict, bool]:
    """Increment the turn counter; return (state, exceeded?)."""
    n = int(state.get("_react_iterations", 0)) + 1
    return {**state, "_react_iterations": n}, n > MAX_ITERATIONS


def summarize_progress(state: dict) -> dict[str, Any]:
    """Compact view of what's been done — used in the orchestrator prompt."""
    return {
        "has_image": has_image(state),
        "vision_done": _done(state, VISION),
        "knowledge_done": _done(state, KNOWLEDGE),
        "compliance_done": _done(state, COMPLIANCE),
        "detected_crop": state.get("detected_crop") or state.get("crop_type"),
        "diagnosis": (state.get("disease_prediction") or {}).get("label"),
    }
