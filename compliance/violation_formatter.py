"""
Format compliance violations into actionable feedback for the Reflexion loop.

The deterministic checker (``agents.compliance_agent._check_banned_substances``)
emits violation dicts shaped like::

    {"chemical": "Monocrotophos", "status": "BANNED", "regulation": "...", ...}

This module turns a list of those into a concise instruction the LLM can act on
when regenerating a safe treatment.
"""
from __future__ import annotations

from typing import Any, Optional


def format_violation_feedback(
    violations: list[dict[str, Any]],
    safe_alternatives: Optional[dict[str, list]] = None,
) -> str:
    """Return human/LLM-readable feedback naming each banned item + alternatives."""
    if not violations:
        return ""
    lines = ["The previous advice recommended substances that are ILLEGAL in India. "
             "Rewrite it WITHOUT any of the following, and do not name them at all:"]
    for v in violations:
        chem = v.get("chemical", "unknown")
        status = v.get("status", "BANNED")
        reg = v.get("regulation", "CIB&RC")
        lines.append(f"  - {chem} ({status}; {reg})")

    if safe_alternatives:
        lines.append("Prefer these approved / safe alternatives where relevant:")
        for banned, alts in safe_alternatives.items():
            names = ", ".join(
                (a.get("name") if isinstance(a, dict) else str(a)) for a in alts[:3]
            )
            if names:
                lines.append(f"  - instead of {banned}: {names}")

    lines.append(
        "Give practical, ICAR-aligned guidance (IPM, correct dosage, PHI, safety). "
        "Return only the corrected advice."
    )
    return "\n".join(lines)
