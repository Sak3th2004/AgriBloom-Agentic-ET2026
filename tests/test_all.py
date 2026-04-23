"""
AgriBloom Agentic — Full Integration Test Suite
Tests every feature end-to-end before hackathon submission.
"""
import pytest
import json
import os
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: MODEL & ONNX
# ═══════════════════════════════════════════════════════════════════════════════
class TestModel:
    """Verify trained model files exist and are valid."""

    def test_model_checkpoint_exists(self):
        path = ROOT / "models/checkpoints/efficientnet_b4_indian/best_model.pth"
        assert path.exists(), "Trained model checkpoint not found!"
        assert path.stat().st_size > 1_000_000, "Model file too small — likely corrupt"

    def test_class_labels_exist(self):
        path = ROOT / "models/checkpoints/efficientnet_b4_indian/class_labels.json"
        assert path.exists()
        labels = json.loads(path.read_text(encoding="utf-8"))
        assert len(labels) == 92, f"Expected 92 classes, got {len(labels)}"

    def test_onnx_model_exists(self):
        path = ROOT / "models/checkpoints/efficientnet_b4_indian/model.onnx"
        assert path.exists(), "ONNX model not found — run export_onnx.py"

    def test_training_curves_exist(self):
        path = ROOT / "models/checkpoints/efficientnet_b4_indian/training_curves.png"
        assert path.exists(), "Training curves not saved"

    def test_model_loads_on_gpu(self):
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        checkpoint = torch.load(
            str(ROOT / "models/checkpoints/efficientnet_b4_indian/best_model.pth"),
            map_location="cuda:0"
        )
        assert checkpoint["val_acc"] > 90.0, f"Model accuracy too low: {checkpoint['val_acc']}"
        assert checkpoint["num_classes"] == 92


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: COMPLIANCE ENGINE (CRITICAL)
# ═══════════════════════════════════════════════════════════════════════════════
class TestComplianceFull:
    """Exhaustive compliance testing — judges will probe this."""

    def test_endosulfan_blocked(self):
        from agents.compliance_agent import run_compliance
        state = {"treatment": "Apply Endosulfan 35% EC", "crop_type": "cotton", "lang": "en", "user_text": "", "recommendations": []}
        result = run_compliance(state)
        assert result["compliance"]["allowed"] is False

    def test_monocrotophos_blocked(self):
        from agents.compliance_agent import run_compliance
        state = {"treatment": "Spray Monocrotophos 36% SL", "crop_type": "tomato", "lang": "en", "user_text": "", "recommendations": []}
        result = run_compliance(state)
        assert len(result["compliance"]["violations"]) > 0

    def test_methyl_parathion_blocked(self):
        from agents.compliance_agent import run_compliance
        state = {"treatment": "Use Methyl Parathion", "crop_type": "rice", "lang": "en", "user_text": "", "recommendations": []}
        result = run_compliance(state)
        assert result["compliance"]["allowed"] is False

    def test_neem_oil_safe(self):
        from agents.compliance_agent import run_compliance
        state = {"treatment": "Neem oil 5ml/L spray", "crop_type": "tomato", "lang": "en", "user_text": "", "recommendations": []}
        result = run_compliance(state)
        assert result["compliance"]["allowed"] is True
        assert result["compliance"]["risk_level"] == "low"

    def test_trichoderma_safe(self):
        from agents.compliance_agent import run_compliance
        state = {"treatment": "Trichoderma viride soil application", "crop_type": "wheat", "lang": "en", "user_text": "", "recommendations": []}
        result = run_compliance(state)
        assert result["compliance"]["allowed"] is True

    def test_safe_alternatives_provided(self):
        from agents.compliance_agent import run_compliance
        state = {"treatment": "Use Endosulfan for pest control", "crop_type": "cotton", "lang": "en", "user_text": "", "recommendations": []}
        result = run_compliance(state)
        assert len(result["compliance"]["safe_alternatives"]) > 0

    def test_audit_trail_complete(self):
        from agents.compliance_agent import run_compliance
        state = {"treatment": "Apply Mancozeb 75% WP", "crop_type": "tomato", "lang": "en", "user_text": "", "recommendations": []}
        result = run_compliance(state)
        audit = result["compliance"]["audit_log"]
        assert len(audit) >= 3

    def test_multilingual_disclaimers(self):
        from agents.compliance_agent import run_compliance
        for lang in ["en", "hi", "kn", "te", "ta"]:
            state = {"treatment": "Neem oil spray", "crop_type": "rice", "lang": lang, "user_text": "", "recommendations": []}
            result = run_compliance(state)
            assert len(result["compliance"]["disclaimers"]) > 0

    def test_banned_db_has_46_entries(self):
        path = ROOT / "compliance/banned_pesticides.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["banned_for_manufacture_import_use"]) >= 46

    def test_mrl_limits_present(self):
        path = ROOT / "compliance/mrl_limits.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["mrl_limits"]) >= 10


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: MULTILINGUAL & LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
class TestMultilingual:
    """Test all 10 language support."""

    def test_kannada_detection(self):
        from agents.orchestrator_agent import _detect_language
        assert _detect_language("ನನ್ನ ಬೆಳೆ ಹಾನಿಯಾಗಿದೆ") == "kn"

    def test_hindi_detection(self):
        from agents.orchestrator_agent import _detect_language
        assert _detect_language("मेरी फसल खराब हो रही है") == "hi"

    def test_telugu_detection(self):
        from agents.orchestrator_agent import _detect_language
        assert _detect_language("నా పంట దెబ్బతిన్నది") == "te"

    def test_tamil_detection(self):
        from agents.orchestrator_agent import _detect_language
        assert _detect_language("என் பயிர் பாதிக்கப்பட்டது") == "ta"

    def test_bengali_detection(self):
        from agents.orchestrator_agent import _detect_language
        assert _detect_language("আমার ফসল নষ্ট হচ্ছে") == "bn"

    def test_punjabi_detection(self):
        from agents.orchestrator_agent import _detect_language
        assert _detect_language("ਮੇਰੀ ਫਸਲ ਖਰਾਬ ਹੋ ਰਹੀ ਹੈ") == "pa"

    def test_gujarati_detection(self):
        from agents.orchestrator_agent import _detect_language
        assert _detect_language("મારો પાક બગડી રહ્યો છે") == "gu"

    def test_english_default(self):
        from agents.orchestrator_agent import _detect_language
        assert _detect_language("My crop is damaged") == "en"

    def test_empty_text_default(self):
        from agents.orchestrator_agent import _detect_language
        assert _detect_language("") == "en"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: IMAGE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
class TestImageValidation:
    """Test image validator edge cases."""

    def test_none_rejected(self):
        from utils.image_validator import validate_image
        assert validate_image(None)["is_valid"] is False

    def test_tiny_image_rejected(self):
        from utils.image_validator import validate_image
        tiny = Image.new("RGB", (30, 30), (0, 128, 0))
        assert validate_image(tiny)["is_valid"] is False

    def test_valid_green_image_accepted(self):
        from utils.image_validator import validate_image
        # Create a green leaf-like image
        img = Image.new("RGB", (300, 300), (34, 139, 34))
        result = validate_image(img)
        assert result["is_valid"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: ORCHESTRATOR ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
class TestRouting:
    """Test request routing logic."""

    def test_image_routes_to_vision(self):
        from agents.orchestrator_agent import run_orchestrator
        state = {"image": Image.new("RGB", (224, 224)), "user_text": ""}
        result = run_orchestrator(state)
        assert result["route"] == "vision_first"

    def test_text_routes_to_knowledge(self):
        from agents.orchestrator_agent import run_orchestrator
        state = {"image": None, "user_text": "What is the price of rice?"}
        result = run_orchestrator(state)
        assert result["route"] == "knowledge_first"

    def test_crop_detection_cotton(self):
        from agents.orchestrator_agent import _detect_crop_from_text
        assert _detect_crop_from_text("My cotton leaves have spots") == "cotton"

    def test_crop_detection_rice(self):
        from agents.orchestrator_agent import _detect_crop_from_text
        assert _detect_crop_from_text("paddy field disease") == "rice"

    def test_crop_detection_hindi(self):
        from agents.orchestrator_agent import _detect_crop_from_text
        assert _detect_crop_from_text("मेरे कापूस में कीड़े") == "cotton"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6: KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════
class TestKnowledgeBase:
    """Test knowledge base integrity."""

    def test_diseases_complete(self):
        path = ROOT / "knowledge_base/crop_diseases.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["diseases"]) >= 8

    def test_every_disease_has_treatment(self):
        path = ROOT / "knowledge_base/crop_diseases.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, disease in data["diseases"].items():
            assert "treatment" in disease or "organic_treatment" in disease, f"{name} missing treatment"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 7: UNIQUE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
class TestUniqueFeatures:
    """Test crop calendar, fertilizer calculator, helpline."""

    def test_fertilizer_calculator(self):
        from utils.fertilizer_calc import calculate_fertilizer
        result = calculate_fertilizer("rice", 2.0, "medium")
        assert result["available"] is True
        assert result["products"]["urea_kg"] > 0
        assert result["estimated_cost_rs"] > 0

    def test_fertilizer_unknown_crop(self):
        from utils.fertilizer_calc import calculate_fertilizer
        result = calculate_fertilizer("dragon_fruit", 1.0)
        assert result["available"] is False

    def test_crop_calendar(self):
        from utils.crop_calendar import get_crop_advisory
        result = get_crop_advisory("rice")
        assert result["available"] is True
        assert result["sowing_period"] != ""

    def test_seasonal_warning(self):
        from utils.crop_calendar import get_seasonal_warning
        result = get_seasonal_warning("cotton")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_helpline_numbers(self):
        from utils.helpline import get_helplines
        lines = get_helplines("en")
        assert len(lines) >= 3
        assert any("1800" in h["number"] for h in lines)

    def test_nearest_kvk(self):
        from utils.helpline import get_nearest_kvk
        result = get_nearest_kvk("Karnataka", "Davangere")
        assert result["found"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 8: GENAI HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
class TestGenAI:
    """Test Gemini API integration."""

    def test_genai_availability(self):
        from utils.genai_handler import is_genai_available
        result = is_genai_available()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
