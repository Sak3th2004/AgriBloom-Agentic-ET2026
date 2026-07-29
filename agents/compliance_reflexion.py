"""
Reflexion-augmented compliance agent node.

Runs the deterministic compliance check (``agents.compliance_agent``); if the
proposed treatment names any banned/restricted substance, it drives the
:func:`compliance.reflexion.reflexion_loop` to regenerate a SAFE treatment via
the free LLM router, then re-checks. The deterministic verdict always has the
final say, so a farmer can never receive a banned recommendation even if
regeneration fails.

Enable in the ReAct graph / pipeline via ``AGRIBLOOM_USE_REFLEXION=1``. Kept
separate from the base ``run_compliance`` so V1 behaviour is untouched.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from agents.compliance_agent import _check_banned_substances, run_compliance
from compliance.reflexion import reflexion_loop

logger = logging.getLogger(__name__)


def _default_regenerate(lang: str, crop: str) -> Optional[Callable[[str, str], str]]:
    """Build an LLM regeneration function, or None if no LLM is available."""
    try:
        from utils import genai_handler

        if not genai_handler.is_genai_available():
            return None
    except Exception:
        return None

    def _regen(previous: str, feedback: str) -> str:
        prompt = (
            f"You are an ICAR agronomy advisor. Crop: {crop}. Language: {lang}.\n"
            f"{feedback}\n\nPrevious advice to correct:\n{previous}\n\nCorrected advice:"
        )
        from utils import genai_handler

        return genai_handler._generate(prompt) or previous

    return _regen


def run_compliance_reflexion(state: dict[str, Any]) -> dict[str, Any]:
    """Compliance node with Reflexion self-correction of the treatment text."""
    checked = run_compliance(state)
    report = checked.get("compliance", {})
    violations = report.get("violations", [])

    if not violations:
        return checked  # already clean

    treatment = state.get("treatment", "") or ""
    lang = state.get("lang", "en")
    crop = state.get("crop_type", "unknown")
    regenerate = _default_regenerate(lang, crop)

    if regenerate is None:
        # No LLM: can't rewrite, but deterministic block already protects the farmer.
        report["reflexion"] = {"success": False, "escalated": True, "reason": "no_llm"}
        logger.info("Reflexion skipped (no LLM); deterministic block stands")
        return checked

    new_treatment, meta = reflexion_loop(
        treatment,
        check=_check_banned_substances,
        regenerate=regenerate,
        max_iterations=3,
        safe_alternatives=report.get("safe_alternatives"),
    )

    # Re-run the deterministic check on the corrected treatment so the final
    # verdict reflects the rewrite. The block still stands if it somehow failed.
    final_state = {**state, "treatment": new_treatment}
    final = run_compliance(final_state)
    final_report = final.get("compliance", {})
    final_report["reflexion"] = meta
    final["treatment"] = new_treatment

    logger.info(
        "Reflexion compliance: success=%s escalated=%s iterations=%s",
        meta.get("success"), meta.get("escalated"), meta.get("iterations"),
    )
    return final
