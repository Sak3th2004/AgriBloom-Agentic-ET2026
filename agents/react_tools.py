"""
Tool registry for the ReAct orchestrator.

Each "tool" is one of the existing, complete agent nodes. Wrapping them here
gives the orchestrator (a) a natural-language description to reason over and
(b) a uniform ``run(state) -> state`` callable. Using the existing nodes means
the ReAct graph produces a *full* advisory (treatment, compliance, voice, PDF)
and V1 keeps working unchanged.

The vision tool prefers the Phase-1 ensemble for the prediction but still runs
the full vision agent so downstream treatment enrichment is preserved.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from graph.state import COMPLIANCE, KNOWLEDGE, OUTPUT, VISION

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[dict], dict]


def _run_vision(state: dict) -> dict:
    """Full vision agent, with the Phase-1 ensemble prediction layered in.

    We call the ensemble first (3-model weighted vote); if it produces a
    confident label we stash it, then run the existing vision agent which adds
    treatment enrichment. The ensemble detail is preserved for observability.
    """
    from agents.vision_agent import run_vision

    ensemble_pred = None
    try:
        from agents.vision_ensemble import ensemble_predict

        if state.get("image") is not None and not state.get("offline"):
            ensemble_pred = ensemble_predict(
                state["image"],
                lang=state.get("lang", "en"),
                use_genai="auto",
            )
    except Exception as e:  # ensemble is best-effort; vision agent still runs
        logger.warning("ReAct vision tool: ensemble failed, using base agent: %s", e)

    new_state = run_vision(state)
    if ensemble_pred is not None:
        base = new_state.get("disease_prediction") or {}
        # Keep the ensemble's cross-model detail alongside the agent's result.
        base_ensemble = ensemble_pred.get("ensemble")
        if isinstance(base, dict) and base_ensemble is not None:
            base = {**base, "ensemble": base_ensemble}
            new_state = {**new_state, "disease_prediction": base}
    return new_state


def _run_knowledge(state: dict) -> dict:
    from agents.knowledge_agent import run_knowledge

    return run_knowledge(state)


def _run_compliance(state: dict) -> dict:
    from agents.compliance_agent import run_compliance

    return run_compliance(state)


def _run_output(state: dict) -> dict:
    from agents.output_agent import run_output

    return run_output(state)


REGISTRY: dict[str, Tool] = {
    VISION: Tool(
        VISION,
        "Analyse the uploaded crop-leaf image to detect the disease "
        "(3-model ensemble). Use FIRST when an image is present.",
        _run_vision,
    ),
    KNOWLEDGE: Tool(
        KNOWLEDGE,
        "Fetch weather, market prices, agronomy facts and retrieve ICAR "
        "treatment guidance (hybrid RAG) for the crop/disease/question.",
        _run_knowledge,
    ),
    COMPLIANCE: Tool(
        COMPLIANCE,
        "Regulatory guardrail: check the proposed treatment against banned "
        "pesticides / MRL limits. Run AFTER knowledge, BEFORE output.",
        _run_compliance,
    ),
    OUTPUT: Tool(
        OUTPUT,
        "Compose the final multilingual answer (text + voice + chart + PDF). "
        "This ENDS the run.",
        _run_output,
    ),
}


def tool_descriptions(actions: list[str]) -> str:
    """Render the given actions as a bulleted menu for the LLM prompt."""
    return "\n".join(f"- {a}: {REGISTRY[a].description}" for a in actions if a in REGISTRY)


def build_registry(use_reflexion: bool = False) -> dict[str, Tool]:
    """Return a tool registry, optionally with Reflexion self-correcting compliance.

    Returns the shared module-level :data:`REGISTRY` unchanged when
    ``use_reflexion`` is False (so existing callers/tests that monkeypatch
    ``REGISTRY`` keep working). When True, returns a NEW dict with the
    compliance tool swapped for :func:`agents.compliance_reflexion.run_compliance_reflexion`
    — the shared registry is never mutated, so this is safe to use alongside
    the plain graph in the same process.
    """
    if not use_reflexion:
        return REGISTRY
    from agents.compliance_reflexion import run_compliance_reflexion

    reflexion_registry = dict(REGISTRY)
    base = REGISTRY[COMPLIANCE]
    reflexion_registry[COMPLIANCE] = Tool(
        COMPLIANCE,
        base.description + " (Reflexion self-correction enabled.)",
        run_compliance_reflexion,
    )
    return reflexion_registry


def run_tool(action: str, state: dict, registry: Optional[dict[str, Tool]] = None) -> dict:
    """Execute a tool by name, marking it done. Unknown action -> unchanged.

    ``registry`` defaults to the shared module-level :data:`REGISTRY` (looked
    up fresh on every call, so test monkeypatching of ``REGISTRY`` still works).
    """
    from graph.state import mark_done

    reg = registry if registry is not None else REGISTRY
    tool = reg.get(action)
    if tool is None:
        logger.warning("ReAct: unknown tool '%s' — skipping", action)
        return state
    return mark_done(tool.run(state), action)
