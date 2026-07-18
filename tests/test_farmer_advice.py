from utils.farmer_advice import build_farmer_advice, format_farmer_advice_for_text


def test_farmer_advice_for_disease_is_structured():
    state = {
        "lang": "en",
        "crop_type": "tomato",
        "disease_prediction": {
            "label": "tomato_early_blight",
            "confidence": 0.84,
            "source": "efficientnet_b4",
        },
        "knowledge": {
            "agronomy": {
                "severity": "medium",
                "yield_loss_range": "15-30%",
                "actions": [
                    "Remove affected leaves",
                    "Avoid overhead irrigation",
                ],
            },
            "weather": {"temp_c": 31, "rain_mm": 0, "source": "test"},
            "market": {"modal_price": 1800, "mandi": "Kolar", "price_trend": "stable"},
        },
        "recommendations": ["Scout field every 3 days"],
        "treatment": "Use crop-approved fungicide only after local expert confirmation.",
        "compliance": {"allowed": True, "risk_level": "low", "violations": []},
    }

    advice = build_farmer_advice(state, disease_name="Tomato Early Blight")

    assert advice["schema_version"] == "1.0"
    assert advice["status"] == "ready"
    assert advice["crop"] == "Tomato"
    assert advice["problem"] == "Tomato Early Blight"
    assert advice["risk_level"] == "medium"
    assert advice["confidence_label"] == "high"
    assert "what_to_do_today" in advice
    assert len(advice["what_to_do_today"]) >= 3
    assert advice["helpline"]["number"] == "1800-180-1551"
    assert advice["technical"]["disease_label"] == "tomato_early_blight"


def test_farmer_advice_for_unclear_photo_prioritizes_retaking_photo():
    state = {
        "lang": "en",
        "crop_type": "unknown",
        "disease_prediction": {"label": "uncertain_detection", "confidence": 0.22},
        "knowledge": {"agronomy": {}, "weather": {}, "market": {}},
        "recommendations": [],
        "treatment": "",
        "compliance": {"allowed": True, "risk_level": "low", "violations": []},
    }

    advice = build_farmer_advice(state)

    assert advice["status"] == "needs_clear_photo"
    assert advice["risk_level"] == "unknown"
    assert "photo" in advice["summary"].lower()
    assert any("daylight" in step.lower() for step in advice["what_to_do_today"])
    assert advice["next_questions"][0].lower().startswith("can you upload")


def test_farmer_advice_for_blocked_treatment_prioritizes_safety():
    state = {
        "lang": "en",
        "crop_type": "cotton",
        "disease_prediction": {"label": "cotton_leaf_spot", "confidence": 0.72},
        "knowledge": {"agronomy": {"severity": "high"}, "weather": {}, "market": {}},
        "recommendations": ["Spray Endosulfan"],
        "treatment": "Apply Endosulfan 35% EC",
        "compliance": {
            "allowed": False,
            "risk_level": "high",
            "violations": [{"chemical": "Endosulfan", "status": "BANNED"}],
            "safe_alternatives": {"Endosulfan": ["neem oil", "pheromone traps"]},
        },
    }

    advice = build_farmer_advice(state, disease_name="Cotton Leaf Spot")
    text = format_farmer_advice_for_text(advice)

    assert advice["status"] == "blocked"
    assert advice["risk_level"] == "high"
    assert advice["safe_alternatives"]
    assert "Do not use" in advice["what_to_do_today"][0]
    assert "Endosulfan" in advice["treatment_guidance"]
    assert "Crop Health Result" in text
