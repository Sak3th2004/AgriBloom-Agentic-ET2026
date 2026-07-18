from channels.whatsapp import format_whatsapp_reply, parse_whatsapp_messages
from utils.learning_store import load_recent_feedback, record_feedback
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_parse_whatsapp_text_message():
    payload = {
        "entry": [{
            "changes": [{
                "value": {
                    "metadata": {"phone_number_id": "123"},
                    "messages": [{
                        "id": "wamid.1",
                        "from": "919999999999",
                        "type": "text",
                        "text": {"body": "My tomato leaves have spots"},
                    }],
                }
            }]
        }]
    }

    messages = parse_whatsapp_messages(payload)

    assert messages == [{
        "message_id": "wamid.1",
        "from": "919999999999",
        "type": "text",
        "phone_number_id": "123",
        "text": "My tomato leaves have spots",
        "media_id": None,
    }]


def test_format_whatsapp_reply_is_farmer_readable():
    reply = format_whatsapp_reply({
        "crop": "Tomato",
        "problem": "Leaf spot",
        "risk_level": "medium",
        "what_to_do_today": ["Remove affected leaves", "Avoid overhead watering"],
        "treatment_guidance": "Use only crop-approved treatment after expert confirmation.",
        "when_to_call_expert": "Call KVK if it spreads.",
        "helpline": {"number": "1800-180-1551"},
    })

    assert "AgriBloom crop advice" in reply
    assert "Tomato" in reply
    assert "Remove affected leaves" in reply
    assert "1800-180-1551" in reply


def test_learning_store_records_jsonl():
    path = ROOT / "data" / "feedback" / "test_learning_store.jsonl"
    if path.exists():
        path.unlink()

    record = record_feedback(
        {
            "source": "test",
            "rating": "not_useful",
            "crop": "Rice",
            "problem": "Yellowing",
            "farmer_text": "leaves yellow",
            "correction": "soil nutrient issue",
        },
        path=path,
    )

    records = load_recent_feedback(path=path)
    assert record["id"]
    assert record["review_status"] == "pending"
    assert records[-1]["crop"] == "Rice"
    path.unlink()
