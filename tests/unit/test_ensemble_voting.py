"""Unit tests for the weighted ensemble voting algorithm (models/ensemble.py)."""
from __future__ import annotations

import pytest

from models.ensemble import (
    DEFAULT_WEIGHTS,
    ModelPrediction,
    PlattParams,
    platt_scale,
    weighted_ensemble_vote,
)


def _preds(*triples):
    return [ModelPrediction(name=n, label=lbl, confidence=c) for n, lbl, c in triples]


def test_unanimous_agreement_keeps_full_confidence():
    preds = _preds(
        ("efficientnet", "grape_downy_mildew", 0.87),
        ("dinov2_knn", "grape_downy_mildew", 0.82),
        ("llava", "grape_downy_mildew", 0.79),
    )
    r = weighted_ensemble_vote(preds, DEFAULT_WEIGHTS)
    assert r["label"] == "grape_downy_mildew"
    assert r["agreement"] == 1.0
    # All models agree -> confidence is the full normalised score (== 1.0 here,
    # since there's only one class) with no agreement penalty.
    assert r["confidence"] == pytest.approx(1.0)


def test_disagreement_applies_penalty():
    # Two classes; winner has more weight but not unanimous -> penalty < 1.
    preds = _preds(
        ("efficientnet", "grape_downy_mildew", 0.80),
        ("dinov2_knn", "grape_black_rot", 0.80),
        ("llava", "grape_downy_mildew", 0.80),
    )
    r = weighted_ensemble_vote(preds, DEFAULT_WEIGHTS)
    assert r["label"] == "grape_downy_mildew"
    assert r["agreement"] == pytest.approx(2 / 3, abs=1e-3)  # rounded to 4dp for display
    # Confidence must be strictly below the raw score share due to the penalty.
    assert r["confidence"] < 1.0


def test_higher_weight_model_can_win_despite_lower_confidence():
    # efficientnet weight 0.5 beats llava weight 0.2 even at lower confidence.
    preds = _preds(
        ("efficientnet", "rice_blast", 0.60),
        ("llava", "rice_healthy", 0.85),
    )
    r = weighted_ensemble_vote(preds, DEFAULT_WEIGHTS)
    assert r["label"] == "rice_blast"


def test_no_available_models_returns_uncertain():
    preds = [ModelPrediction("efficientnet", "x", 0.9, available=False)]
    r = weighted_ensemble_vote(preds, DEFAULT_WEIGHTS)
    assert r["label"] == "uncertain_detection"
    assert r["confidence"] == 0.0


def test_zero_confidence_predictions_ignored():
    preds = _preds(
        ("efficientnet", "wheat_rust", 0.0),  # ignored (conf 0)
        ("dinov2_knn", "wheat_rust", 0.5),
    )
    r = weighted_ensemble_vote(preds, DEFAULT_WEIGHTS)
    assert r["label"] == "wheat_rust"
    assert r["agreement"] == 1.0  # only one available model


def test_single_model_no_penalty():
    preds = _preds(("efficientnet", "tomato_late_blight", 0.75))
    r = weighted_ensemble_vote(preds, DEFAULT_WEIGHTS)
    assert r["label"] == "tomato_late_blight"
    assert r["agreement"] == 1.0
    assert r["confidence"] == pytest.approx(1.0)


def test_historical_accuracy_multiplier_shifts_winner():
    weights = {"a": 0.5, "b": 0.5}
    preds = _preds(("a", "class_x", 0.6), ("b", "class_y", 0.6))
    # Tie on weight+conf; boost b's recent accuracy so it wins.
    r = weighted_ensemble_vote(preds, weights, historical_accuracy={"a": 0.5, "b": 1.0})
    assert r["label"] == "class_y"


def test_platt_scaling_identity_when_none():
    assert platt_scale(0.7, None) == pytest.approx(0.7)


def test_platt_scaling_monotonic():
    p = PlattParams(a=2.0, b=-1.0)
    assert platt_scale(0.2, p) < platt_scale(0.8, p)
    assert 0.0 <= platt_scale(0.5, p) <= 1.0


def test_calibration_can_change_outcome():
    preds = _preds(("a", "x", 0.55), ("b", "y", 0.6))
    weights = {"a": 0.5, "b": 0.5}
    # Without calibration, b wins. Strongly boost a via Platt scaling.
    calib = {"a": PlattParams(a=6.0, b=0.0)}  # sigmoid(6*0.55)=~0.965
    r = weighted_ensemble_vote(preds, weights, calibration=calib)
    assert r["label"] == "x"


def test_confidence_is_bounded():
    preds = _preds(
        ("efficientnet", "a", 0.9),
        ("dinov2_knn", "b", 0.9),
        ("llava", "c", 0.9),
    )
    r = weighted_ensemble_vote(preds, DEFAULT_WEIGHTS)
    assert 0.0 <= r["confidence"] <= 1.0
    # Three-way split -> low confidence.
    assert r["confidence"] < 0.6
