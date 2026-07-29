"""
Reflexion self-correction loop (Algorithm 5).

Turns the compliance guardrail from a detector into a *fixer*. Given a piece of
advice, a ``check`` function (the deterministic banned-substance scanner) and a
``regenerate`` function (an LLM call), it iterates:

    check -> if violations: build feedback -> regenerate -> repeat (max N) -> escalate

The loop is pure and dependency-free — ``check`` and ``regenerate`` are injected,
so it's fully unit-testable and the same code works with any LLM backend.

IMPORTANT (safety): Reflexion improves the *quality* of the response, but the
100% banned-pesticide catch rate does NOT depend on it. The deterministic
checker still runs afterwards and blocks anything unsafe even if regeneration
fails. Reflexion just means the farmer usually gets a corrected, useful answer
instead of only a block.
"""
from __future__ import annotations

from typing import Any, Callable

from compliance.violation_formatter import format_violation_feedback

CheckFn = Callable[[str], list[dict[str, Any]]]
RegenerateFn = Callable[[str, str], str]


def reflexion_loop(
    response: str,
    check: CheckFn,
    regenerate: RegenerateFn,
    max_iterations: int = 3,
    safe_alternatives: dict[str, list] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Iteratively correct ``response`` until it passes ``check`` or we give up.

    Args:
        response: the advice text to sanitise.
        check: returns a list of violations (empty == clean).
        regenerate: ``(previous_text, feedback) -> new_text``.
        max_iterations: max regeneration attempts.
        safe_alternatives: optional map used to enrich the feedback.

    Returns:
        ``(final_text, meta)`` where meta has ``iterations``, ``success``,
        ``escalated``, ``history`` and (if escalated) ``remaining_violations``.
    """
    history: list[dict[str, Any]] = []
    current = response

    for i in range(max_iterations):
        violations = check(current)
        if not violations:
            return current, {
                "iterations": i,
                "success": True,
                "escalated": False,
                "history": history,
            }
        feedback = format_violation_feedback(violations, safe_alternatives)
        history.append({
            "iteration": i,
            "violations": [v.get("chemical") for v in violations],
            "feedback": feedback,
        })
        try:
            current = regenerate(current, feedback)
        except Exception as e:  # LLM failed — stop and escalate cleanly
            return current, {
                "iterations": i + 1,
                "success": False,
                "escalated": True,
                "error": str(e),
                "remaining_violations": [v.get("chemical") for v in violations],
                "history": history,
            }

    # Final verification after the last regeneration.
    remaining = check(current)
    if not remaining:
        return current, {
            "iterations": max_iterations,
            "success": True,
            "escalated": False,
            "history": history,
        }
    return current, {
        "iterations": max_iterations,
        "success": False,
        "escalated": True,
        "remaining_violations": [v.get("chemical") for v in remaining],
        "history": history,
    }
