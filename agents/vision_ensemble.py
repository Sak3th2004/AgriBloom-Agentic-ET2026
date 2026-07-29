"""
Vision Ensemble agent (Phase 1) — 3-model weighted voting.

Combines three independent vision models into one calibrated decision:

  1. **EfficientNet-B4** — the trained in-distribution classifier (weight 0.5).
     Reuses the engine loader in ``agents.vision_agent`` so we don't duplicate
     model loading or the OOD logic.
  2. **DINOv2 + FAISS KNN** — zero-shot novel-crop detector (weight 0.3).
  3. **GenAI vision** (LLaVA / NVIDIA-90B / Gemini via ``genai_handler``) — a
     reasoning vote (weight 0.2).

The fusion maths live in :mod:`models.ensemble`. This module only *gathers*
votes and maps the winner back to a ``disease_prediction`` dict compatible with
the rest of the pipeline. It is deliberately non-destructive: it does not touch
``main.py`` (the pipeline is rewired in Phase 3), so V1 keeps working.

The GenAI vote is the slow one, so by default it runs **only when the local
models are unsure or disagree** (``use_genai="auto"``). Set ``use_genai=True``
to always poll all three, or ``False`` to stay fully local/offline.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from models.ensemble import (
    DEFAULT_WEIGHTS,
    ModelPrediction,
    weighted_ensemble_vote,
)

logger = logging.getLogger(__name__)

# Below this ensemble confidence we report an uncertain detection rather than
# guessing — the graceful "not sure, take a clearer photo" path for novel crops.
UNCERTAIN_THRESHOLD = 0.45
# When the top local model is below this, it's worth spending a GenAI call.
GENAI_TRIGGER_CONFIDENCE = 0.60

_CONF_WORD_TO_NUM = {"high": 0.80, "medium": 0.55, "low": 0.30}


def _normalize(label: str) -> str:
    """Canonicalise a label for cross-model comparison: lower, unify separators."""
    return label.strip().lower().replace("___", "_").replace(" ", "_").replace("-", "_")


def _genai_to_prediction(result: dict) -> Optional[ModelPrediction]:
    """Turn a ``genai_handler`` vision dict into a :class:`ModelPrediction`."""
    disease = str(result.get("disease", "")).strip()
    crop = str(result.get("crop", "")).strip()
    if not disease or disease in {"unknown", "analysis_failed"}:
        return None
    conf_word = str(result.get("confidence", "low")).lower()
    confidence = _CONF_WORD_TO_NUM.get(conf_word, 0.4)
    label = _normalize(f"{crop}_{disease}" if crop and crop != "unknown" else disease)
    return ModelPrediction(name="llava", label=label, confidence=confidence)


def ensemble_predict(
    image,
    lang: str = "en",
    use_genai: Literal["auto", True, False] = "auto",
    weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Run the ensemble on a PIL image and return a ``disease_prediction`` dict.

    The returned dict is compatible with the downstream pipeline and adds an
    ``ensemble`` block with per-model detail for observability.
    """
    from agents.vision_agent import _resolve_and_load_engine

    weights = weights or DEFAULT_WEIGHTS
    predictions: list[ModelPrediction] = []
    detail: dict[str, Any] = {}

    # 1) EfficientNet-B4 -------------------------------------------------------
    effnet_raw: dict[str, Any] = {}
    engine = _resolve_and_load_engine()
    if engine is not None:
        try:
            effnet_raw = engine.predict(image)
            label = effnet_raw.get("label", "")
            conf = float(effnet_raw.get("confidence", 0.0))
            if label and label not in {"model_not_loaded", "model_error"}:
                predictions.append(
                    ModelPrediction(name="efficientnet", label=_normalize(label), confidence=conf)
                )
                detail["efficientnet"] = {"label": label, "confidence": conf,
                                          "is_ood": effnet_raw.get("is_ood")}
        except Exception as e:
            logger.warning("Ensemble: EfficientNet failed: %s", e)

    # 2) DINOv2 + FAISS KNN ----------------------------------------------------
    try:
        from models.dinov2_knn import get_dinov2_knn

        knn = get_dinov2_knn()
        knn_res = knn.predict(image)
        if knn_res.get("available"):
            predictions.append(
                ModelPrediction(
                    name="dinov2_knn",
                    label=_normalize(knn_res["label"]),
                    confidence=float(knn_res["confidence"]),
                )
            )
            detail["dinov2_knn"] = {"label": knn_res["label"], "confidence": knn_res["confidence"]}
        else:
            detail["dinov2_knn"] = {"available": False, "reason": knn_res.get("reason")}
    except Exception as e:
        logger.warning("Ensemble: DINOv2 KNN failed: %s", e)
        detail["dinov2_knn"] = {"available": False, "reason": str(e)}

    # 3) GenAI vision vote (conditional) --------------------------------------
    want_genai = use_genai is True or (
        use_genai == "auto" and _should_use_genai(predictions)
    )
    if want_genai:
        try:
            from utils import genai_handler

            g = genai_handler.analyze_unknown_crop_pil(image, language=lang)
            gp = _genai_to_prediction(g)
            if gp is not None:
                predictions.append(gp)
                detail["llava"] = {"label": gp.label, "confidence": gp.confidence,
                                   "source": g.get("source")}
        except Exception as e:
            logger.warning("Ensemble: GenAI vision failed: %s", e)

    # Fuse --------------------------------------------------------------------
    vote = weighted_ensemble_vote(predictions, model_weights=weights)
    vote["ensemble_detail"] = detail
    vote["models_used"] = [p.name for p in predictions]

    return _to_disease_prediction(vote, effnet_raw)


def _should_use_genai(predictions: list[ModelPrediction]) -> bool:
    """Decide whether a GenAI call is worth it, given the local votes so far."""
    if not predictions:
        return True  # nothing local worked — we need the reasoning model
    labels = {p.label for p in predictions}
    top = max(p.confidence for p in predictions)
    # Poll GenAI if models disagree or the best local confidence is weak.
    return len(labels) > 1 or top < GENAI_TRIGGER_CONFIDENCE


def _to_disease_prediction(vote: dict, effnet_raw: dict) -> dict[str, Any]:
    """Map an ensemble vote to the pipeline's ``disease_prediction`` schema."""
    label = vote["label"]
    confidence = float(vote["confidence"])

    if label == "uncertain_detection" or confidence < UNCERTAIN_THRESHOLD:
        return {
            "label": "uncertain_detection",
            "confidence": round(confidence, 4),
            "top3": effnet_raw.get("top3", []),
            "source": "ensemble",
            "ensemble": vote,
        }

    crop = label.split("_")[0] if "_" in label else label
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "crop": crop,
        "top3": effnet_raw.get("top3", []),
        "agreement": vote.get("agreement"),
        "source": "ensemble",
        "ensemble": vote,
    }


def run_vision_ensemble(state: dict) -> dict:
    """LangGraph node wrapper (wired into the graph in Phase 3)."""
    image = state.get("image")
    lang = state.get("lang", state.get("user_language", "en"))
    if image is None:
        return {**state, "disease_prediction": {"label": "no_image", "confidence": 0.0},
                "status": "no_image"}
    use_genai = False if state.get("offline") else "auto"
    pred = ensemble_predict(image, lang=lang, use_genai=use_genai)
    crop = pred.get("crop", state.get("crop_type", "unknown"))
    return {**state, "disease_prediction": pred, "crop_type": crop, "status": "vision_complete"}
