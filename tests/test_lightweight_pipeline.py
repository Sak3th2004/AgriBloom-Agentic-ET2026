from agents.output_agent import run_output


def test_output_agent_can_skip_heavy_artifacts():
    result = run_output({
        "lang": "en",
        "generate_artifacts": False,
        "crop_type": "tomato",
        "disease_prediction": {
            "label": "tomato_early_blight",
            "confidence": 0.82,
            "source": "test",
        },
        "knowledge": {
            "agronomy": {
                "severity": "medium",
                "yield_loss_range": "15-30%",
                "actions": ["Remove affected leaves"],
            },
            "weather": {},
            "market": {},
        },
        "recommendations": [],
        "treatment": "Avoid overhead watering.",
        "compliance": {"allowed": True, "risk_level": "low", "violations": []},
    })

    assert result["farmer_advice"]["crop"] == "Tomato"
    assert result["voice_output_path"] is None
    assert result["bloom_figure"] is None
    assert result["audit_pdf_path"] is None
    assert result["status"] == "output_complete"
