"""Unit tests for the Reflexion self-correction loop (compliance/)."""
from __future__ import annotations

from compliance.reflexion import reflexion_loop
from compliance.violation_formatter import format_violation_feedback


def _banned(name):
    return {"chemical": name, "status": "BANNED", "regulation": "CIB&RC"}


def test_clean_input_zero_iterations():
    text, meta = reflexion_loop(
        "Use neem oil 5ml/L.",
        check=lambda t: [],
        regenerate=lambda prev, fb: prev,
    )
    assert meta["success"] is True
    assert meta["escalated"] is False
    assert meta["iterations"] == 0


def test_single_correction_succeeds():
    # First check finds a violation; after one regenerate the text is clean.
    calls = {"n": 0}

    def check(t):
        return [_banned("Monocrotophos")] if "monocrotophos" in t.lower() else []

    def regen(prev, feedback):
        calls["n"] += 1
        return "Use neem oil and Trichoderma instead."

    text, meta = reflexion_loop("Spray Monocrotophos.", check=check, regenerate=regen)
    assert meta["success"] is True
    assert meta["escalated"] is False
    assert meta["iterations"] == 1
    assert calls["n"] == 1
    assert "monocrotophos" not in text.lower()


def test_escalates_when_never_fixed():
    # regenerate keeps producing banned advice -> escalate after max_iterations.
    def check(t):
        return [_banned("Endosulfan")]  # always dirty

    def regen(prev, feedback):
        return "Still use Endosulfan."

    text, meta = reflexion_loop("Use Endosulfan.", check=check, regenerate=regen, max_iterations=3)
    assert meta["success"] is False
    assert meta["escalated"] is True
    assert meta["iterations"] == 3
    assert "Endosulfan" in meta["remaining_violations"]


def test_regenerate_exception_escalates_cleanly():
    def check(t):
        return [_banned("Phorate")]

    def regen(prev, feedback):
        raise RuntimeError("LLM down")

    text, meta = reflexion_loop("Use Phorate.", check=check, regenerate=regen)
    assert meta["success"] is False
    assert meta["escalated"] is True
    assert "error" in meta


def test_feedback_names_chemicals_and_alternatives():
    fb = format_violation_feedback(
        [_banned("Monocrotophos")],
        safe_alternatives={"Monocrotophos": [{"name": "Neem oil"}, {"name": "Trichoderma"}]},
    )
    assert "Monocrotophos" in fb
    assert "Neem oil" in fb
    assert "ILLEGAL" in fb


def test_feedback_empty_for_no_violations():
    assert format_violation_feedback([]) == ""
