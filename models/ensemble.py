"""
Weighted ensemble voting for the vision agent (Algorithm 1).

Combines predictions from several independent vision models (EfficientNet-B4,
DINOv2+FAISS KNN, LLaVA) into a single calibrated decision. The maths here is
pure Python + math so it is fully unit-testable without a GPU, models, or any
network — the heavy models live behind the agent in `agents/vision_ensemble.py`.

Design:
  * Each model reports a raw confidence in [0, 1].
  * Raw confidences are optionally re-calibrated with **Platt scaling**
    (a logistic fit `sigmoid(a * conf + b)`) so an over/under-confident model
    can be corrected from historical validation data.
  * Each model has a static **weight** (trust) and an optional per-run
    **historical_accuracy** multiplier.
  * Scores are aggregated per predicted label; the label with the highest total
    weighted-calibrated score wins.
  * A **graded agreement penalty** lowers confidence when the models disagree:
    unanimous -> factor 1.0; only the winner votes for it -> factor = penalty.

Example:
    >>> preds = [
    ...     ModelPrediction("efficientnet", "grape_downy_mildew", 0.87),
    ...     ModelPrediction("dinov2_knn",   "grape_downy_mildew", 0.82),
    ...     ModelPrediction("llava",        "grape_downy_mildew", 0.79),
    ... ]
    >>> result = weighted_ensemble_vote(preds, {"efficientnet": 0.5, "dinov2_knn": 0.3, "llava": 0.2})
    >>> result["label"]
    'grape_downy_mildew'
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelPrediction:
    """A single model's vote.

    Attributes:
        name: model identifier, must match keys in ``model_weights``.
        label: predicted class label (e.g. ``"grape_downy_mildew"``).
        confidence: raw confidence in [0, 1].
        available: set False when the model could not run; it is ignored.
    """

    name: str
    label: str
    confidence: float
    available: bool = True


@dataclass
class PlattParams:
    """Logistic calibration parameters fit on validation data: sigmoid(a*x + b)."""

    a: float = 1.0
    b: float = 0.0


# Default fusion weights (trust) per model — tune from validation accuracy.
DEFAULT_WEIGHTS: dict[str, float] = {
    "efficientnet": 0.5,
    "dinov2_knn": 0.3,
    "llava": 0.2,
}

# How much to shrink confidence when models fully disagree (0 < penalty <= 1).
DEFAULT_AGREEMENT_PENALTY = 0.7


def _sigmoid(x: float) -> float:
    # Numerically stable logistic.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def platt_scale(confidence: float, params: Optional[PlattParams]) -> float:
    """Apply Platt scaling; identity when ``params`` is None."""
    c = min(max(confidence, 0.0), 1.0)
    if params is None:
        return c
    return _sigmoid(params.a * c + params.b)


def _crop_family(label: str) -> str:
    """Extract the crop prefix from a disease label for coarse comparison.

    ``grape___downy_mildew`` and ``grape_black_rot`` both map to ``grape``.
    """
    if "___" in label:
        return label.split("___")[0]
    if "_" in label:
        return label.split("_")[0]
    return label


@dataclass
class EnsembleResult:
    label: str
    confidence: float
    raw_agreement: float
    per_model: list[dict] = field(default_factory=list)
    class_scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "agreement": round(self.raw_agreement, 4),
            "per_model": self.per_model,
            "class_scores": {k: round(v, 4) for k, v in self.class_scores.items()},
            "source": "ensemble",
        }


def weighted_ensemble_vote(
    predictions: list[ModelPrediction],
    model_weights: Optional[dict[str, float]] = None,
    historical_accuracy: Optional[dict[str, float]] = None,
    calibration: Optional[dict[str, PlattParams]] = None,
    agreement_penalty: float = DEFAULT_AGREEMENT_PENALTY,
) -> dict:
    """Fuse model votes into one calibrated decision.

    Args:
        predictions: one :class:`ModelPrediction` per model.
        model_weights: static trust per model name; defaults to
            :data:`DEFAULT_WEIGHTS`, unknown models get weight 1.0.
        historical_accuracy: optional per-model multiplier (recent accuracy on
            similar classes), defaults to 1.0.
        calibration: optional per-model :class:`PlattParams`.
        agreement_penalty: confidence multiplier floor when models disagree.

    Returns:
        A dict (see :meth:`EnsembleResult.as_dict`) with the winning ``label``,
        calibrated ``confidence`` in [0, 1], ``agreement``, and per-model detail.
        Returns ``label="uncertain_detection"`` with confidence 0 when no model
        was available.
    """
    weights = model_weights if model_weights is not None else DEFAULT_WEIGHTS
    hist = historical_accuracy or {}
    calib = calibration or {}

    available = [p for p in predictions if p.available and p.confidence > 0]
    if not available:
        return EnsembleResult(
            label="uncertain_detection", confidence=0.0, raw_agreement=0.0
        ).as_dict()

    class_scores: dict[str, float] = {}
    per_model: list[dict] = []
    label_weight: dict[str, float] = {}  # total weight of voters per label

    for p in available:
        calibrated = platt_scale(p.confidence, calib.get(p.name))
        weight = weights.get(p.name, 1.0) * hist.get(p.name, 1.0)
        score = calibrated * weight
        class_scores[p.label] = class_scores.get(p.label, 0.0) + score
        label_weight[p.label] = label_weight.get(p.label, 0.0) + weight
        per_model.append(
            {
                "name": p.name,
                "label": p.label,
                "raw_confidence": round(p.confidence, 4),
                "calibrated": round(calibrated, 4),
                "weight": round(weight, 4),
                "score": round(score, 4),
            }
        )

    # Winner = highest aggregated weighted-calibrated score. Ties broken by the
    # label with the greater total voter weight, then alphabetically (stable).
    winner = max(
        class_scores,
        key=lambda lbl: (class_scores[lbl], label_weight[lbl], lbl),
    )

    total_score = sum(class_scores.values())
    base_confidence = class_scores[winner] / total_score if total_score > 0 else 0.0

    # Graded agreement: fraction of models that voted for the winner.
    n_models = len(available)
    n_agree = sum(1 for p in available if p.label == winner)
    if n_models <= 1 or n_agree == n_models:
        agreement_factor = 1.0
    else:
        # Ranges from ``agreement_penalty`` (only winner agrees) up to ~1.0.
        agreement_factor = agreement_penalty + (1.0 - agreement_penalty) * (
            (n_agree - 1) / (n_models - 1)
        )

    confidence = base_confidence * agreement_factor

    return EnsembleResult(
        label=winner,
        confidence=confidence,
        raw_agreement=n_agree / n_models,
        per_model=per_model,
        class_scores=class_scores,
    ).as_dict()
