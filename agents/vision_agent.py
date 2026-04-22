"""
Vision Agent - Two-Tier Crop Disease Detection
Tier 1: EfficientNet-B4 (trained, fast, works offline)
Tier 2: Gemini Vision API (fallback for unknown crops)

Supports: Cotton, Rice, Wheat, Maize, Sugarcane, Ragi, Tomato, Potato,
          Pepper, Apple, Grape, Cherry, Peach, Orange, Soybean, Strawberry
          + ANY crop via Gemini Vision fallback
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.45      # Below this → uncertain
GEMINI_FALLBACK_THRESHOLD = 0.35 # Below this → trigger Gemini Vision


def get_device() -> torch.device:
    """Get optimal device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ── EfficientNet-B4 Engine ───────────────────────────────────────────────────
class EfficientNetEngine:
    """EfficientNet-B4 inference engine for trained Indian crop model."""

    def __init__(self, checkpoint_dir: str) -> None:
        self.device = get_device()
        self.checkpoint_dir = Path(checkpoint_dir)

        # Load class labels
        labels_path = self.checkpoint_dir / "class_labels.json"
        if labels_path.exists():
            raw = json.loads(labels_path.read_text(encoding="utf-8"))
            self.class_names = []
            self.id2label = {}
            for idx_str in sorted(raw.keys(), key=int):
                entry = raw[idx_str]
                name = entry if isinstance(entry, str) else entry.get("class_name", f"class_{idx_str}")
                self.class_names.append(name)
                self.id2label[int(idx_str)] = name
        else:
            logger.warning("class_labels.json not found, using generic labels")
            self.class_names = []
            self.id2label = {}

        num_classes = len(self.class_names) or 92  # default

        # Load model
        try:
            from efficientnet_pytorch import EfficientNet
            self.model = EfficientNet.from_name("efficientnet-b4")
            in_features = self.model._fc.in_features
            self.model._fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )

            ckpt_path = self.checkpoint_dir / "best_model.pth"
            if ckpt_path.exists():
                ckpt = torch.load(str(ckpt_path), map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                # Update class names from checkpoint if available
                if "class_names" in ckpt and not self.class_names:
                    self.class_names = ckpt["class_names"]
                    self.id2label = {i: n for i, n in enumerate(self.class_names)}
                logger.info(f"EfficientNet-B4 loaded from {ckpt_path}")
            else:
                logger.warning(f"No checkpoint found at {ckpt_path}")

            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"EfficientNet-B4: {num_classes} classes on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load EfficientNet-B4: {e}")
            self.model = None

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Run EfficientNet-B4 inference."""
        if self.model is None:
            return {"label": "model_not_loaded", "confidence": 0.0, "source": "error"}

        img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        outputs = self.model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self.id2label.get(idx, f"class_{idx}")

        # Get top 3
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [
            {"label": self.id2label.get(int(i), f"class_{i}"), "confidence": float(probs[i])}
            for i in top3_idx
        ]

        return {
            "label": label,
            "confidence": confidence,
            "class_index": idx,
            "top3": top3,
            "source": "efficientnet_b4_gpu" if "cuda" in str(self.device) else "efficientnet_b4_cpu",
        }


# ── ViT Engine (legacy support) ─────────────────────────────────────────────
class ViTEngine:
    """Legacy ViT engine for backward compatibility."""

    def __init__(self, model_path: str) -> None:
        self.device = get_device()
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            self.processor = ViTImageProcessor.from_pretrained(
                model_path, local_files_only=Path(model_path).exists()
            )
            self.model = ViTForImageClassification.from_pretrained(
                model_path, local_files_only=Path(model_path).exists(),
                ignore_mismatched_sizes=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label
            logger.info(f"ViT loaded: {self.model.config.num_labels} classes")
        except Exception as e:
            logger.error(f"ViT load failed: {e}")
            self.model = None
            self.id2label = {}

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        if self.model is None:
            return {"label": "model_error", "confidence": 0.0, "source": "error"}
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self.id2label.get(idx) or self.id2label.get(str(idx)) or f"class_{idx}"
        return {"label": label, "confidence": confidence, "source": "vit", "class_index": idx}


# ── Global engine instances ──────────────────────────────────────────────────
_ENGINE = None
_ENGINE_TYPE = ""


def _resolve_and_load_engine():
    """Load the best available model engine."""
    global _ENGINE, _ENGINE_TYPE

    if _ENGINE is not None:
        return _ENGINE

    # Priority 1: EfficientNet-B4 (new trained model)
    effnet_dir = Path(os.getenv("AGRIBLOOM_VISION_MODEL_DIR", "models/checkpoints/efficientnet_b4_indian"))
    if effnet_dir.exists() and (effnet_dir / "best_model.pth").exists():
        _ENGINE = EfficientNetEngine(str(effnet_dir))
        _ENGINE_TYPE = "efficientnet_b4"
        return _ENGINE

    # Priority 2: ViT fine-tuned checkpoint
    vit_dir = Path("models/checkpoints/vit_crop_disease")
    if vit_dir.exists() and (vit_dir / "config.json").exists():
        _ENGINE = ViTEngine(str(vit_dir))
        _ENGINE_TYPE = "vit"
        return _ENGINE

    # Priority 3: HuggingFace default
    try:
        _ENGINE = ViTEngine("google/vit-base-patch16-224")
        _ENGINE_TYPE = "vit_base"
    except Exception:
        _ENGINE = None
        _ENGINE_TYPE = "none"

    return _ENGINE


# ── Disease treatments (multilingual) ────────────────────────────────────────
DISEASE_TREATMENTS = {
    "healthy": {"en": "Crop is healthy! Continue regular monitoring and maintain proper irrigation."},
    "blight": {"en": "Apply Mancozeb 75% WP at 2.5g/L. Remove infected leaves. Improve drainage."},
    "rust": {"en": "Spray Propiconazole 25% EC at 1ml/L. Use resistant varieties next season."},
    "blast": {"en": "Apply Tricyclazole 75% WP at 0.6g/L. Avoid excess nitrogen fertilizer."},
    "rot": {"en": "Remove infected parts. Apply Carbendazim 50% WP at 1g/L as drench."},
    "spot": {"en": "Apply Chlorothalonil 75% WP at 2g/L. Maintain plant spacing."},
    "wilt": {"en": "Remove infected plants. Treat soil with Trichoderma viride 5g/L."},
    "mosaic": {"en": "Remove infected plants. Control whitefly vector. Use virus-free planting material."},
    "mildew": {"en": "Spray Karathane 1ml/L or Sulphur 80% WP 3g/L. Improve ventilation."},
    "curl": {"en": "Control whitefly with Imidacloprid 0.3ml/L. Remove infected plants."},
    "armyworm": {"en": "Spray Chlorantraniliprole 18.5% SC at 0.3ml/L. Release Trichogramma."},
    "bollworm": {"en": "Apply NPV 250 LE/ha or Spinosad 45% SC 0.3ml/L. Use pheromone traps."},
    "aphid": {"en": "Spray Neem oil 5ml/L or Thiamethoxam 25% WG 0.3g/L. Release ladybird beetles."},
    "thrips": {"en": "Apply Spinosad 45% SC 0.3ml/L. Use blue sticky traps."},
    "whitefly": {"en": "Apply Neem oil 5ml/L or Imidacloprid 17.8% SL 0.3ml/L. Use yellow sticky traps."},
    "scab": {"en": "Apply Mancozeb 75% WP 2.5g/L. Prune infected branches."},
}


def _get_treatment_for_label(label: str, lang: str = "en") -> str:
    """Get treatment recommendation by matching disease keywords in label."""
    label_lower = label.lower()

    # Check for healthy first
    if "healthy" in label_lower:
        return DISEASE_TREATMENTS["healthy"].get(lang, DISEASE_TREATMENTS["healthy"]["en"])

    # Match disease keywords
    for keyword, treatments in DISEASE_TREATMENTS.items():
        if keyword in label_lower:
            return treatments.get(lang, treatments.get("en", ""))

    return "Consult your nearest Krishi Vigyan Kendra (KVK) for expert advice. Helpline: 1800-180-1551"


def _extract_crop_type(label: str) -> str:
    """Extract crop name from label like 'tomato___early_blight' → 'tomato'."""
    if "___" in label:
        return label.split("___")[0].lower()
    if "_" in label:
        return label.split("_")[0].lower()
    return "unknown"


# ── Main Vision Node ─────────────────────────────────────────────────────────
def run_vision(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Two-tier vision inference.

    Tier 1: EfficientNet-B4 / ViT (local GPU)
    Tier 2: Gemini Vision API (fallback for unknown/low confidence)

    Input: image (PIL), offline (bool), lang (str)
    Output: disease_prediction, crop_type, treatment, status
    """
    image = state.get("image")
    timestamp = datetime.now().isoformat()

    # ── No image provided ────────────────────────────────────────────────
    if image is None:
        return {
            **state,
            "disease_prediction": {"label": "unknown", "confidence": 0.0, "source": "no_image"},
            "crop_type": "unknown",
            "treatment": "",
            "status": "vision_skipped",
        }

    # ── Image Validation ─────────────────────────────────────────────────
    lang = state.get("lang", "en")
    try:
        from utils.image_validator import validate_image
        validation = validate_image(image, lang)
        if not validation["is_valid"]:
            return {
                **state,
                "disease_prediction": {
                    "label": "invalid_image",
                    "confidence": 0.0,
                    "source": "image_validator",
                    "validation": validation,
                },
                "crop_type": "unknown",
                "treatment": validation.get("message", ""),
                "status": "vision_invalid_image",
            }
    except Exception as e:
        logger.warning(f"Image validation skipped: {e}")

    offline = bool(state.get("offline", False))

    # ── Tier 1: Local Model Inference ────────────────────────────────────
    try:
        engine = _resolve_and_load_engine()
        if engine is not None:
            pred = engine.predict(image)
        else:
            pred = {"label": "model_unavailable", "confidence": 0.0, "source": "no_model"}
    except Exception as e:
        logger.error(f"Tier 1 inference failed: {e}")
        pred = {"label": "error", "confidence": 0.0, "source": "inference_error", "error": str(e)}

    label = pred.get("label", "unknown")
    confidence = pred.get("confidence", 0.0)

    # ── Tier 2: Gemini Vision Fallback ───────────────────────────────────
    gemini_result = None
    if confidence < GEMINI_FALLBACK_THRESHOLD and not offline:
        try:
            from utils.genai_handler import analyze_unknown_crop_pil, is_genai_available
            if is_genai_available():
                logger.info(f"Tier 1 confidence {confidence:.2f} < {GEMINI_FALLBACK_THRESHOLD} → Gemini fallback")
                gemini_result = analyze_unknown_crop_pil(image, lang)

                if gemini_result and gemini_result.get("confidence") != "low":
                    # Use Gemini result
                    gemini_disease = gemini_result.get("disease", "unknown")
                    gemini_crop = gemini_result.get("crop", "unknown")
                    pred = {
                        "label": f"{gemini_crop}___{gemini_disease}".lower().replace(" ", "_"),
                        "confidence": 0.75 if gemini_result.get("confidence") == "high" else 0.55,
                        "source": "gemini_vision_fallback",
                        "gemini_raw": gemini_result,
                        "tier1_label": label,
                        "tier1_confidence": confidence,
                    }
                    label = pred["label"]
                    confidence = pred["confidence"]
        except Exception as e:
            logger.warning(f"Gemini fallback failed: {e}")

    # ── Apply confidence threshold ───────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD and gemini_result is None:
        pred["original_label"] = label
        pred["label"] = "uncertain_detection"
        label = "uncertain_detection"

    # ── Extract crop type and treatment ──────────────────────────────────
    crop_type = _extract_crop_type(label)
    treatment = _get_treatment_for_label(label, lang)

    # Use Gemini treatment if available
    if gemini_result and gemini_result.get("treatment"):
        treatment = gemini_result["treatment"]

    # ── GenAI treatment enhancement (if available and online) ────────────
    if confidence >= CONFIDENCE_THRESHOLD and not offline and "healthy" not in label.lower():
        try:
            from utils.genai_handler import generate_treatment_advice, is_genai_available
            if is_genai_available():
                genai_advice = generate_treatment_advice(
                    disease=label, crop=crop_type, language=lang,
                )
                if genai_advice:
                    treatment = genai_advice
        except Exception as e:
            logger.warning(f"GenAI treatment enhancement skipped: {e}")

    logger.info(
        f"Vision result: {label} ({confidence:.2%}) | crop={crop_type} | "
        f"source={pred.get('source', 'unknown')} | engine={_ENGINE_TYPE}"
    )

    return {
        **state,
        "disease_prediction": pred,
        "crop_type": crop_type,
        "treatment": treatment,
        "status": "vision_complete",
    }


# Export
__all__ = ["run_vision", "EfficientNetEngine", "ViTEngine"]
