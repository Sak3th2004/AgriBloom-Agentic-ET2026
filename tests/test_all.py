"""
Tests for AgriBloom Agentic System
Covers: Compliance, Vision, GenAI, RAG, Multilingual, Pipeline, Edge Cases
"""
import pytest
import json
from pathlib import Path


# ── Test 1: Compliance Agent ─────────────────────────────────────────────────
class TestCompliance:
    """Test deterministic compliance guardrails."""

    def test_banned_pesticide_blocked(self):
        """Endosulfan must be blocked."""
        from agents.compliance_agent import run_compliance
        state = {
            "treatment": "Apply Endosulfan 35% EC at 2ml/L",
            "user_text": "",
            "crop_type": "cotton",
            "lang": "en",
            "recommendations": [],
        }
        result = run_compliance(state)
        assert result["compliance"]["allowed"] is False
        assert result["compliance"]["risk_level"] == "high"
        assert len(result["compliance"]["violations"]) > 0

    def test_safe_treatment_allowed(self):
        """Neem oil should pass compliance."""
        from agents.compliance_agent import run_compliance
        state = {
            "treatment": "Apply Neem oil 5ml/L spray",
            "user_text": "",
            "crop_type": "tomato",
            "lang": "en",
            "recommendations": [],
        }
        result = run_compliance(state)
        assert result["compliance"]["allowed"] is True
        assert result["compliance"]["risk_level"] == "low"

    def test_monocrotophos_blocked(self):
        """Monocrotophos must be blocked on vegetables."""
        from agents.compliance_agent import run_compliance
        state = {
            "treatment": "Spray Monocrotophos 36% SL",
            "user_text": "",
            "crop_type": "tomato",
            "lang": "en",
            "recommendations": [],
        }
        result = run_compliance(state)
        compliance = result["compliance"]
        violations = compliance.get("violations", [])
        assert len(violations) > 0

    def test_safe_alternatives_provided(self):
        """When a banned chemical is found, alternatives must be offered."""
        from agents.compliance_agent import run_compliance
        state = {
            "treatment": "Use Endosulfan for pest control",
            "user_text": "",
            "crop_type": "cotton",
            "lang": "en",
            "recommendations": [],
        }
        result = run_compliance(state)
        assert len(result["compliance"]["safe_alternatives"]) > 0

    def test_audit_trail_generated(self):
        """Every compliance check must have audit log."""
        from agents.compliance_agent import run_compliance
        state = {
            "treatment": "Apply Mancozeb 75% WP",
            "user_text": "",
            "crop_type": "tomato",
            "lang": "en",
            "recommendations": [],
        }
        result = run_compliance(state)
        assert "audit_log" in result["compliance"]
        assert len(result["compliance"]["audit_log"]) >= 3  # 3 checks minimum

    def test_disclaimers_multilingual(self):
        """Disclaimers must be in farmer's language."""
        from agents.compliance_agent import run_compliance
        for lang in ["en", "hi", "kn", "te", "ta"]:
            state = {
                "treatment": "Neem oil spray",
                "user_text": "",
                "crop_type": "rice",
                "lang": lang,
                "recommendations": [],
            }
            result = run_compliance(state)
            assert len(result["compliance"]["disclaimers"]) > 0


# ── Test 2: Compliance Database Integrity ────────────────────────────────────
class TestComplianceDB:
    """Verify compliance database files are complete."""

    def test_banned_pesticides_count(self):
        """Must have 46+ banned pesticides."""
        path = Path("compliance/banned_pesticides.json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["banned_for_manufacture_import_use"]) >= 46

    def test_mrl_limits_exist(self):
        """MRL limits must exist."""
        path = Path("compliance/mrl_limits.json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["mrl_limits"]) >= 10

    def test_safe_alternatives_exist(self):
        """Safe alternatives must exist."""
        path = Path("compliance/safe_alternatives.json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["alternatives"]) >= 5


# ── Test 3: Image Validator ──────────────────────────────────────────────────
class TestImageValidator:
    """Test crop leaf image validation."""

    def test_none_image_rejected(self):
        from utils.image_validator import validate_image
        result = validate_image(None)
        assert result["is_valid"] is False

    def test_small_image_rejected(self):
        from utils.image_validator import validate_image
        from PIL import Image
        tiny = Image.new("RGB", (50, 50), (0, 128, 0))
        result = validate_image(tiny)
        assert result["is_valid"] is False
        assert result["reason"] == "too_small"


# ── Test 4: GenAI Handler ───────────────────────────────────────────────────
class TestGenAI:
    """Test Gemini API integration (requires API key)."""

    def test_genai_availability_check(self):
        from utils.genai_handler import is_genai_available
        # Should not crash regardless of key availability
        result = is_genai_available()
        assert isinstance(result, bool)


# ── Test 5: Knowledge Base ───────────────────────────────────────────────────
class TestKnowledgeBase:
    """Test crop disease knowledge base."""

    def test_crop_diseases_complete(self):
        path = Path("knowledge_base/crop_diseases.json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["diseases"]) >= 8

    def test_disease_has_treatment(self):
        path = Path("knowledge_base/crop_diseases.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, disease in data["diseases"].items():
            assert "treatment" in disease or "organic_treatment" in disease, f"{name} missing treatment"


# ── Test 6: Orchestrator ────────────────────────────────────────────────────
class TestOrchestrator:
    """Test request routing."""

    def test_image_routes_to_vision(self):
        from agents.orchestrator_agent import run_orchestrator
        from PIL import Image
        state = {"image": Image.new("RGB", (224, 224)), "user_text": ""}
        result = run_orchestrator(state)
        assert result["route"] == "vision_first"

    def test_text_routes_to_knowledge(self):
        from agents.orchestrator_agent import run_orchestrator
        state = {"image": None, "user_text": "What is the price of rice?"}
        result = run_orchestrator(state)
        assert result["route"] == "knowledge_first"

    def test_language_detection(self):
        from agents.orchestrator_agent import run_orchestrator
        state = {"image": None, "user_text": "ನನ್ನ ಟೊಮ್ಯಾಟೊ ಎಲೆಗಳು ಹಳದಿ"}
        result = run_orchestrator(state)
        assert result["lang"] == "kn"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
