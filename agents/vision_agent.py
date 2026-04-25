"""
Vision Agent — Two-tier crop disease detection.

Tier 1: EfficientNet-B4 on GPU (92 Indian crop diseases)
Tier 2: Gemini Vision / Ollama fallback for unknown crops
"""
from __future__ import annotations

import json
import logging
import math
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

        # ── Entropy-based OOD detection ────────────────────────────────
        # High entropy = model is unsure = likely out-of-distribution
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = math.log(len(probs))  # uniform distribution
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        # If entropy > 0.45, the model is guessing → flag as uncertain
        is_ood = normalized_entropy > 0.45

        # Get top 3
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [
            {"label": self.id2label.get(int(i), f"class_{i}"), "confidence": float(probs[i])}
            for i in top3_idx
        ]

        # Check if top-1 and top-2 are from DIFFERENT crops → uncertain
        top1_crop = label.split("___")[0].split("_")[0] if "___" in label or "_" in label else label
        if len(top3) >= 2:
            top2_label = top3[1]["label"]
            top2_crop = top2_label.split("___")[0].split("_")[0] if "___" in top2_label or "_" in top2_label else top2_label
            # If top-2 predictions are from different crops AND close in confidence → uncertain
            if top1_crop != top2_crop and top3[1]["confidence"] > confidence * 0.5:
                is_ood = True

        if is_ood:
            logger.info(f"OOD detected: entropy={normalized_entropy:.3f}, conf={confidence:.2%} → marking uncertain")
            confidence = min(confidence, 0.30)  # Cap at 30% to trigger fallback

        return {
            "label": label,
            "confidence": confidence,
            "class_index": idx,
            "top3": top3,
            "entropy": round(normalized_entropy, 3),
            "is_ood": is_ood,
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

    # ── User-text contradiction check ────────────────────────────────────
    # If the user explicitly describes a crop (e.g., "watermelon leaf") but
    # the model predicts a DIFFERENT crop (e.g., "cotton"), the model is
    # likely wrong. Force fallback to LLaVA which can actually see the image.
    user_text = state.get("user_text", "").lower()
    force_fallback = False
    user_mentioned_crop = None
    if user_text and confidence > GEMINI_FALLBACK_THRESHOLD:
        # Known crop names the user might type
        known_crops = [
            "mango", "watermelon", "banana", "coconut", "papaya", "guava",
            "pomegranate", "orange", "lemon", "chilli", "onion", "brinjal",
            "okra", "cauliflower", "cabbage", "pea", "mustard", "sunflower",
            "jute", "tea", "coffee", "cardamom", "turmeric", "ginger",
            "rice", "wheat", "maize", "cotton", "tomato", "potato",
            "apple", "grape", "corn", "soybean", "sugarcane", "pepper",
            "cherry", "peach", "strawberry", "blueberry", "raspberry",
            "squash", "cucumber", "pumpkin", "melon", "citrus",
        ]
        predicted_crop = label.split("___")[0].lower().replace(" ", "_") if "___" in label else label.lower().split("_")[0]
        user_mentioned_crop = None
        for crop in known_crops:
            if crop in user_text:
                user_mentioned_crop = crop
                break
        
        if user_mentioned_crop and user_mentioned_crop not in predicted_crop and predicted_crop not in user_mentioned_crop:
            logger.warning(
                f"CONTRADICTION: User said '{user_mentioned_crop}' but model predicted '{predicted_crop}' "
                f"({confidence:.0%}) → forcing LLaVA vision fallback"
            )
            force_fallback = True
            confidence = 0.20  # Force below threshold

        # Note: We intentionally DON'T force fallback for generic symptom text
        # (quick buttons). EfficientNet + NVIDIA treatment is faster and more reliable
        # than forcing NVIDIA vision, which can also misidentify crops.

    # ── Tier 2: AI Vision Fallback (LLaVA → Gemini → Ollama Text) ──────
    #
    # When EfficientNet is uncertain (mango, random crops, etc.), we need
    # a model that can actually SEE the image. Priority:
    #   1. Ollama LLaVA (local, free, sees images, no rate limits)
    #   2. Gemini Vision (cloud, rate-limited)
    #   3. Ollama text (can't see images, uses user description + top3 guesses)
    #
    vision_fallback_result = None
    if confidence < GEMINI_FALLBACK_THRESHOLD or force_fallback:
        logger.info(f"Tier 1 confidence {confidence:.2f} < {GEMINI_FALLBACK_THRESHOLD} → AI Vision fallback")

        # ── Step 1: Try Ollama LLaVA Vision (LOCAL, can SEE images) ────
        try:
            from utils.genai_handler import _ollama_vision_analyze
            top3_text = ", ".join([f"{t['label']}({t['confidence']:.0%})" for t in pred.get("top3", [])])
            user_text = state.get("user_text", "")
            user_hint = f" The farmer says: '{user_text}'." if user_text else ""

            # If user explicitly named a crop → trust them, only ask for disease
            if force_fallback and user_mentioned_crop:
                llava_prompt = (
                    f"You are an expert Indian agricultural scientist. "
                    f"The farmer told you this is a {user_mentioned_crop.upper()} leaf. TRUST the farmer — this IS {user_mentioned_crop}. "
                    f"Look at this {user_mentioned_crop} leaf image carefully. "
                    f"What disease or pest problem do you see on this {user_mentioned_crop} leaf? "
                    f"What treatment do you recommend? "
                    f"Reply STRICTLY: CROP: {user_mentioned_crop.capitalize()}, DISEASE: <specific disease name>, TREATMENT: <one practical sentence>"
                )
            else:
                llava_prompt = (
                    "You are an expert Indian agricultural scientist. "
                    "Look at this plant leaf image very carefully. "
                    "You MUST identify the EXACT crop species: Mango, Watermelon, Cotton, Tomato, Rice, Wheat, Banana, Coconut, Apple, Grape, Sugarcane, Maize, Pepper, Chilli, Groundnut, etc. "
                    "DO NOT say 'leafy green plant'. Say the EXACT species name. "
                    f"An AI model guessed (may be wrong): {top3_text}.{user_hint} "
                    "Reply STRICTLY: CROP: <exact species>, DISEASE: <specific disease>, TREATMENT: <one practical sentence>"
                )
            
            llava_text = _ollama_vision_analyze(image, llava_prompt)
            if llava_text and len(llava_text) > 10:
                import re
                crop_match = re.search(r'CROP:\s*([^,\n]+)', llava_text, re.IGNORECASE)
                disease_match = re.search(r'DISEASE:\s*([^,\n]+)', llava_text, re.IGNORECASE)
                treatment_match = re.search(r'TREATMENT:\s*(.+)', llava_text, re.IGNORECASE)

                # If user gave a crop name and LLaVA didn't parse, use user's crop
                final_crop = user_mentioned_crop.capitalize() if (force_fallback and user_mentioned_crop) else None

                if crop_match and disease_match:
                    vc = crop_match.group(1).strip()
                    # Override with user's crop if contradiction was detected
                    if final_crop:
                        vc = final_crop
                    vd = disease_match.group(1).strip()
                    vt = treatment_match.group(1).strip() if treatment_match else ""
                    pred = {
                        "label": f"{vc}___{vd}".lower().replace(" ", "_"),
                        "confidence": 0.72,
                        "source": "ollama_llava_vision",
                        "tier1_label": label,
                        "tier1_confidence": confidence,
                    }
                    label = pred["label"]
                    confidence = pred["confidence"]
                    vision_fallback_result = {"treatment": vt, "crop": vc, "disease": vd}
                    logger.info(f"LLaVA Vision diagnosed: {vc} - {vd}")
                else:
                    # LLaVA responded but not in expected format
                    # Use user's crop name if available, otherwise try to extract
                    diag_crop = final_crop or "Unknown"
                    # Try to extract disease keywords from freeform text
                    disease_keywords = ["blight", "spot", "rot", "wilt", "rust", "mildew", "mosaic",
                                       "anthracnose", "canker", "scab", "aphid", "mite", "whitefly",
                                       "leaf curl", "yellowing", "necrosis", "fungal", "bacterial"]
                    found_disease = "Leaf Disease"
                    for kw in disease_keywords:
                        if kw in llava_text.lower():
                            found_disease = kw.title()
                            break
                    
                    pred = {
                        "label": f"{diag_crop}___{found_disease}".lower().replace(" ", "_"),
                        "confidence": 0.60,
                        "source": "ollama_llava_vision",
                        "tier1_label": label,
                        "tier1_confidence": confidence,
                    }
                    label = pred["label"]
                    confidence = pred["confidence"]
                    vision_fallback_result = {"treatment": llava_text[:300], "crop": diag_crop, "disease": found_disease}
                    logger.info(f"LLaVA freeform → {diag_crop} - {found_disease}")
        except Exception as e:
            logger.warning(f"LLaVA Vision fallback failed: {e}")

        # ── Step 2: Try Gemini Vision (cloud, rate-limited) ────────────
        if vision_fallback_result is None and not offline:
            try:
                from utils.genai_handler import analyze_unknown_crop_pil
                logger.info("LLaVA unavailable, trying Gemini Vision...")
                gemini_result = analyze_unknown_crop_pil(image, lang)
                if gemini_result and gemini_result.get("confidence") != "low":
                    gc = gemini_result.get("crop", "unknown")
                    gd = gemini_result.get("disease", "unknown")
                    pred = {
                        "label": f"{gc}___{gd}".lower().replace(" ", "_"),
                        "confidence": 0.75 if gemini_result.get("confidence") == "high" else 0.55,
                        "source": gemini_result.get("source", "gemini_vision"),
                        "tier1_label": label,
                        "tier1_confidence": confidence,
                    }
                    label = pred["label"]
                    confidence = pred["confidence"]
                    vision_fallback_result = gemini_result
                    logger.info(f"Gemini/LLaVA Vision diagnosed: {gc} - {gd}")
            except Exception as e:
                logger.warning(f"Gemini Vision fallback failed: {e}")

        # ── Step 3: Ollama text-only (uses user description + top3) ────
        if vision_fallback_result is None:
            try:
                from utils.genai_handler import _ollama_generate
                top3_text = ", ".join([f"{t['label']}({t['confidence']:.0%})" for t in pred.get("top3", [])])
                user_text = state.get("user_text", "")
                user_ctx = f"\nThe farmer described: '{user_text}'." if user_text else ""
                ood_prompt = (
                    f"A crop disease AI analyzed a plant leaf image but was uncertain. "
                    f"Its confused guesses: {top3_text}.{user_ctx}\n"
                    f"What crop and disease is this MOST LIKELY? "
                    f"Reply: CROP: <name>, DISEASE: <name>, TREATMENT: <one sentence>"
                )
                ollama_text = _ollama_generate(ood_prompt)
                if ollama_text and "CROP:" in ollama_text.upper():
                    import re
                    cm = re.search(r'CROP:\s*([^,\n]+)', ollama_text, re.IGNORECASE)
                    dm = re.search(r'DISEASE:\s*([^,\n]+)', ollama_text, re.IGNORECASE)
                    tm = re.search(r'TREATMENT:\s*(.+)', ollama_text, re.IGNORECASE)
                    if cm and dm:
                        oc, od = cm.group(1).strip(), dm.group(1).strip()
                        ot = tm.group(1).strip() if tm else ""
                        pred = {
                            "label": f"{oc}___{od}".lower().replace(" ", "_"),
                            "confidence": 0.50,
                            "source": "ollama_text_diagnosis",
                            "tier1_label": label,
                            "tier1_confidence": pred.get("confidence", 0),
                        }
                        label = pred["label"]
                        confidence = pred["confidence"]
                        vision_fallback_result = {"treatment": ot, "crop": oc, "disease": od}
                        logger.info(f"Ollama text diagnosed: {oc} - {od}")
            except Exception as e:
                logger.warning(f"Ollama text OOD diagnosis failed: {e}")

    # ── Apply confidence threshold ───────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD and vision_fallback_result is None:
        pred["original_label"] = label
        pred["label"] = "uncertain_detection"
        label = "uncertain_detection"

    # ── Extract crop type and treatment ──────────────────────────────────
    crop_type = _extract_crop_type(label)
    treatment = _get_treatment_for_label(label, lang)

    # Use vision fallback treatment if available
    if vision_fallback_result and vision_fallback_result.get("treatment"):
        treatment = vision_fallback_result["treatment"]

    # ── GenAI treatment enhancement — ALWAYS call NVIDIA for detailed advice ──
    # Note: We always try NVIDIA even in "offline" mode — treatment quality is critical
    if label not in ["uncertain_detection", "unknown", "error"] and "healthy" not in label.lower():
        try:
            import concurrent.futures
            from utils.genai_handler import generate_treatment_advice, is_genai_available
            if is_genai_available():
                def _gen():
                    return generate_treatment_advice(
                        disease=label, crop=crop_type, language=lang,
                    )
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_gen)
                    genai_advice = future.result(timeout=30)  # 30s max for treatment
                if genai_advice and len(genai_advice) > 50:
                    treatment = genai_advice
                    logger.info(f"NVIDIA treatment advice: {len(genai_advice)} chars")
        except concurrent.futures.TimeoutError:
            logger.warning("GenAI treatment timed out after 25s — using local treatment")
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
