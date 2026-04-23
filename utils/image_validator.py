"""
Image Validator — Validates if uploaded image is a crop leaf
Uses green channel analysis, resolution check, and format validation.
Returns friendly error messages in farmer's language.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum requirements
MIN_RESOLUTION = 128
MAX_RESOLUTION = 8000
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP", "TIFF"}

# Error messages in multiple languages
VALIDATION_MESSAGES = {
    "invalid_format": {
        "en": "❌ Unsupported image format. Please upload a JPEG or PNG photo.",
        "hi": "❌ असमर्थित फोटो प्रारूप। कृपया JPEG या PNG फोटो अपलोड करें।",
        "kn": "❌ ಬೆಂಬಲವಿಲ್ಲದ ಫೋಟೋ ಫಾರ್ಮ್ಯಾಟ್. ದಯವಿಟ್ಟು JPEG ಅಥವಾ PNG ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "te": "❌ మద్దతు లేని ఫోటో ఫార్మాట్. దయచేసి JPEG లేదా PNG ఫోటో అప్‌లోడ్ చేయండి.",
        "ta": "❌ ஆதரிக்கப்படாத புகைப்பட வடிவம். JPEG அல்லது PNG புகைப்படம் பதிவேற்றவும்.",
    },
    "too_small": {
        "en": "❌ Image is too small. Please take a closer photo of the leaf (minimum 128x128 pixels).",
        "hi": "❌ फोटो बहुत छोटी है। कृपया पत्ती की नजदीकी फोटो लें।",
        "kn": "❌ ಫೋಟೋ ತುಂಬಾ ಚಿಕ್ಕದಾಗಿದೆ. ದಯವಿಟ್ಟು ಎಲೆಯ ಹತ್ತಿರದ ಫೋಟೋ ತೆಗೆಯಿರಿ.",
        "te": "❌ ఫోటో చాలా చిన్నది. దయచేసి ఆకు యొక్క దగ్గరి ఫోటో తీయండి.",
        "ta": "❌ புகைப்படம் மிகவும் சிறியது. இலையின் நெருக்கமான புகைப்படம் எடுக்கவும்.",
    },
    "not_leaf": {
        "en": "⚠️ This doesn't appear to be a crop leaf. Please take a clear photo of the affected leaf.\n\n"
              "📸 Tips:\n• Good lighting (daylight)\n• Close-up of the leaf\n• Avoid shadows\n• Show disease spots clearly",
        "hi": "⚠️ यह फसल की पत्ती नहीं लगती। कृपया प्रभावित पत्ती की स्पष्ट फोटो लें।\n\n"
              "📸 सुझाव:\n• अच्छी रोशनी (दिन की रोशनी)\n• पत्ती का क्लोज-अप\n• छाया से बचें\n• रोग के दाग स्पष्ट दिखाएं",
        "kn": "⚠️ ಇದು ಬೆಳೆ ಎಲೆಯಂತೆ ಕಾಣುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಪ್ರಭಾವಿತ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ತೆಗೆಯಿರಿ.\n\n"
              "📸 ಸಲಹೆ:\n• ಉತ್ತಮ ಬೆಳಕು\n• ಎಲೆಯ ಕ್ಲೋಸ್-ಅಪ್\n• ನೆರಳು ತಪ್ಪಿಸಿ",
        "te": "⚠️ ఇది పంట ఆకుగా కనిపించడం లేదు. దయచేసి ప్రభావిత ఆకు యొక్క స్పష్టమైన ఫోటో తీయండి.",
        "ta": "⚠️ இது பயிர் இலையாக தெரியவில்லை. பாதிக்கப்பட்ட இலையின் தெளிவான புகைப்படம் எடுக்கவும்.",
    },
    "valid": {
        "en": "✅ Valid crop leaf image",
        "hi": "✅ मान्य फसल पत्ती फोटो",
        "kn": "✅ ಮಾನ್ಯ ಬೆಳೆ ಎಲೆ ಫೋಟೋ",
        "te": "✅ చెల్లుబాటయ్యే పంట ఆకు ఫోటో",
        "ta": "✅ செல்லுபடியாகும் பயிர் இலை புகைப்படம்",
    },
}


def _analyze_green_channel(image: Image.Image) -> float:
    """
    Analyze green channel dominance to detect if image is a plant leaf.
    Returns greenness ratio (0-1). Typical leaves: 0.35+
    """
    try:
        img_array = np.array(image.convert("RGB"))

        # Calculate mean channel values
        r_mean = img_array[:, :, 0].mean()
        g_mean = img_array[:, :, 1].mean()
        b_mean = img_array[:, :, 2].mean()

        total = r_mean + g_mean + b_mean
        if total == 0:
            return 0.0

        green_ratio = g_mean / total

        # Also check if green is the dominant channel
        green_dominant = g_mean > r_mean and g_mean > b_mean

        # Combined score: green ratio + bonus for dominance
        score = green_ratio
        if green_dominant:
            score += 0.05

        # Also consider yellow-green (diseased leaves)
        # Diseased leaves often have higher red (yellowing)
        if r_mean > g_mean and r_mean < g_mean * 1.3:
            # Yellowish-green — could be diseased leaf
            score += 0.03

        return min(1.0, score)

    except Exception as e:
        logger.warning(f"Green channel analysis failed: {e}")
        return 0.33  # Default neutral score


def _check_color_variance(image: Image.Image) -> float:
    """
    Check color variance — uniform color images (solid backgrounds) are unlikely leaves.
    Returns variance score (0-1). Real photos: 0.3+
    """
    try:
        img_array = np.array(image.convert("RGB").resize((64, 64)))
        variance = img_array.std() / 255.0
        return min(1.0, variance)
    except Exception:
        return 0.5


def _detect_skin_tone(image: Image.Image) -> float:
    """
    Detect percentage of skin-tone pixels in the image.
    Uses HSV color space for reliable skin detection across skin tones.
    Returns ratio (0-1). Selfies typically: 0.3+, Leaves: <0.15
    """
    try:
        img = image.convert("RGB").resize((128, 128))
        img_array = np.array(img)

        # Convert RGB to HSV manually
        r, g, b = img_array[:,:,0].astype(float), img_array[:,:,1].astype(float), img_array[:,:,2].astype(float)
        r, g, b = r/255.0, g/255.0, b/255.0

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c

        # Hue calculation
        hue = np.zeros_like(max_c)
        mask = diff > 0
        # Red is max
        idx = mask & (max_c == r)
        hue[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
        # Green is max
        idx = mask & (max_c == g)
        hue[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
        # Blue is max
        idx = mask & (max_c == b)
        hue[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360

        sat = np.where(max_c > 0, diff / max_c, 0)
        val = max_c

        # Skin detection rules (works for all skin tones):
        # Hue: 0-50 (red to yellow range)
        # Saturation: 0.1-0.8
        # Value: 0.2-0.95
        skin_mask = (
            ((hue >= 0) & (hue <= 50)) &
            (sat >= 0.1) & (sat <= 0.8) &
            (val >= 0.2) & (val <= 0.95)
        )

        skin_ratio = skin_mask.sum() / skin_mask.size
        return float(skin_ratio)

    except Exception as e:
        logger.warning(f"Skin tone detection failed: {e}")
        return 0.0  # Safe default — don't reject


def validate_image(
    image: Image.Image | None,
    lang: str = "en",
) -> dict[str, Any]:
    """
    Validate if uploaded image is a crop leaf.

    Args:
        image: PIL Image to validate
        lang: Language code for error messages

    Returns:
        Dict with is_valid, message, and analysis details
    """
    if image is None:
        return {
            "is_valid": False,
            "reason": "no_image",
            "message": "Please upload a crop photo.",
        }

    # Check format
    fmt = (image.format or "").upper()
    if fmt and fmt not in SUPPORTED_FORMATS:
        return {
            "is_valid": False,
            "reason": "invalid_format",
            "message": VALIDATION_MESSAGES["invalid_format"].get(
                lang, VALIDATION_MESSAGES["invalid_format"]["en"]
            ),
        }

    # Check resolution
    w, h = image.size
    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        return {
            "is_valid": False,
            "reason": "too_small",
            "message": VALIDATION_MESSAGES["too_small"].get(
                lang, VALIDATION_MESSAGES["too_small"]["en"]
            ),
            "resolution": f"{w}x{h}",
        }

    # Green channel analysis
    green_score = _analyze_green_channel(image)
    variance_score = _check_color_variance(image)

    # Skin tone detection — detects selfies/human photos
    skin_ratio = _detect_skin_tone(image)

    # Decision logic:
    # 1. If skin tone > 40% of image → likely a selfie → reject
    # 2. If green score very low AND high skin → definitely not a leaf
    # 3. Otherwise → ACCEPT (let the ML model decide if it's a valid crop)
    is_likely_leaf = True

    if skin_ratio > 0.40:
        # More than 40% skin pixels → likely a selfie/human photo
        is_likely_leaf = False
    elif green_score < 0.25 and variance_score < 0.10:
        # Very low green AND very uniform → probably a solid background
        is_likely_leaf = False

    logger.debug(f"Image validation: green={green_score:.3f}, var={variance_score:.3f}, skin={skin_ratio:.3f}, valid={is_likely_leaf}")

    if not is_likely_leaf:
        return {
            "is_valid": False,
            "reason": "not_leaf",
            "message": VALIDATION_MESSAGES["not_leaf"].get(
                lang, VALIDATION_MESSAGES["not_leaf"]["en"]
            ),
            "green_score": round(green_score, 3),
            "variance_score": round(variance_score, 3),
            "skin_ratio": round(skin_ratio, 3),
        }

    return {
        "is_valid": True,
        "reason": "valid",
        "message": VALIDATION_MESSAGES["valid"].get(
            lang, VALIDATION_MESSAGES["valid"]["en"]
        ),
        "green_score": round(green_score, 3),
        "variance_score": round(variance_score, 3),
        "resolution": f"{w}x{h}",
    }


# Export
__all__ = ["validate_image", "VALIDATION_MESSAGES"]
