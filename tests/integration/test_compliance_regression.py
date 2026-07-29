"""
Compliance regression suite — proves a 100% banned-pesticide CATCH RATE.

The suite is generated directly from the regulatory database so it stays in sync
automatically: every banned name, every restricted name, and every search-term
synonym must be detected inside a realistic treatment sentence. It also checks
that genuinely safe advice is allowed, and that the Reflexion node produces a
safe final verdict even when the LLM is unavailable or uncooperative.

This is the non-negotiable safety gate: if any banned substance slips through,
this suite fails.
"""
from __future__ import annotations

import pytest

from agents.compliance_agent import (
    BANNED_NAMES,
    BANNED_SEARCH_TERMS,
    RESTRICTED_NAMES,
    _check_banned_substances,
    run_compliance,
)
from agents.compliance_reflexion import run_compliance_reflexion

# ── Build the parametrized case lists from the live DB ──────────────────────────
BANNED = sorted({e["name"] for e in BANNED_NAMES.values()})
RESTRICTED = sorted({e["name"] for e in RESTRICTED_NAMES.values()})
SEARCH_TERMS = sorted(set(BANNED_SEARCH_TERMS))

SAFE_TREATMENTS = [
    "Apply neem oil (Azadirachtin) 5ml per litre of water.",
    "Use Trichoderma viride 5g/L as a soil drench.",
    "Spray Pseudomonas fluorescens for bacterial blight.",
    "Copper oxychloride 3g/L is recommended for downy mildew.",
    "Follow IPM: install pheromone traps and use Beauveria bassiana.",
    "Mancozeb 75% WP at recommended dose with proper PHI.",
    "Bordeaux mixture 1% for fungal control on grapes.",
    "Apply well-decomposed farmyard manure and maintain field sanitation.",
]


# ── 1. Every banned name must be caught ─────────────────────────────────────────
@pytest.mark.parametrize("chemical", BANNED)
def test_every_banned_name_is_caught(chemical):
    text = f"To control the pest, spray {chemical} at 2ml per litre every week."
    violations = _check_banned_substances(text)
    names = {v["chemical"].lower() for v in violations}
    assert chemical.lower() in names, f"MISSED banned substance: {chemical}"
    assert any(v["status"] == "BANNED" for v in violations)


# ── 2. Every restricted name must be caught ─────────────────────────────────────
@pytest.mark.parametrize("chemical", RESTRICTED)
def test_every_restricted_name_is_caught(chemical):
    text = f"You could apply {chemical} but check the label first."
    violations = _check_banned_substances(text)
    names = {v["chemical"].lower() for v in violations}
    assert chemical.lower() in names, f"MISSED restricted substance: {chemical}"


# ── 3. Every search-term synonym must trigger a violation ───────────────────────
@pytest.mark.parametrize("term", SEARCH_TERMS)
def test_every_search_term_triggers(term):
    text = f"Old recommendation: use {term} for this disease."
    assert _check_banned_substances(text), f"search term did not trigger: {term}"


# ── 4. Banned treatment => run_compliance blocks it ─────────────────────────────
@pytest.mark.parametrize("chemical", BANNED[:15])
def test_banned_treatment_is_blocked_by_agent(chemical):
    state = {"treatment": f"Spray {chemical} 2ml/L.", "crop_type": "tomato", "lang": "en"}
    report = run_compliance(state)["compliance"]
    assert report["allowed"] is False
    assert report["compliance_status"] == "unsafe"


# ── 5. Safe advice is allowed ───────────────────────────────────────────────────
@pytest.mark.parametrize("treatment", SAFE_TREATMENTS)
def test_safe_treatment_allowed(treatment):
    state = {"treatment": treatment, "crop_type": "grape", "lang": "en"}
    report = run_compliance(state)["compliance"]
    assert report["allowed"] is True, f"false positive on safe advice: {treatment}"


# ── 6. Reflexion: even a hostile/absent LLM cannot leak a banned rec ─────────────
def test_reflexion_blocks_when_no_llm(monkeypatch):
    # Force "no LLM" path: reflexion can't rewrite, deterministic block must stand.
    import agents.compliance_reflexion as cr

    monkeypatch.setattr(cr, "_default_regenerate", lambda lang, crop: None)
    state = {"treatment": "Spray Endosulfan 2ml/L.", "crop_type": "cotton", "lang": "en"}
    out = run_compliance_reflexion(state)
    assert out["compliance"]["allowed"] is False
    assert out["compliance"]["reflexion"]["escalated"] is True


def test_reflexion_fixes_with_good_llm(monkeypatch):
    # A cooperative LLM rewrites banned advice to safe advice -> final allowed.
    import agents.compliance_reflexion as cr

    def fake_regen(lang, crop):
        return lambda prev, fb: "Use neem oil 5ml/L and Trichoderma; follow IPM and PHI."

    monkeypatch.setattr(cr, "_default_regenerate", fake_regen)
    state = {"treatment": "Spray Endosulfan 2ml/L.", "crop_type": "cotton", "lang": "en"}
    out = run_compliance_reflexion(state)
    assert out["compliance"]["allowed"] is True
    assert out["compliance"]["reflexion"]["success"] is True
    assert "endosulfan" not in out["treatment"].lower()


def test_reflexion_escalates_with_bad_llm(monkeypatch):
    # A hostile LLM keeps recommending banned -> escalate AND stay blocked.
    import agents.compliance_reflexion as cr

    def bad_regen(lang, crop):
        return lambda prev, fb: "Just keep using Endosulfan, it works."

    monkeypatch.setattr(cr, "_default_regenerate", bad_regen)
    state = {"treatment": "Spray Endosulfan 2ml/L.", "crop_type": "cotton", "lang": "en"}
    out = run_compliance_reflexion(state)
    assert out["compliance"]["allowed"] is False  # deterministic block still wins
    assert out["compliance"]["reflexion"]["escalated"] is True


def test_case_count_is_comprehensive():
    # Sanity: the generated suite really covers 100+ scenarios.
    total = len(BANNED) + len(RESTRICTED) + len(SEARCH_TERMS) + len(SAFE_TREATMENTS)
    assert total >= 100, f"regression suite only has {total} cases"
