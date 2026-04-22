"""
Farmer Emergency Helpline & Nearest KVK Finder
Provides instant access to government helplines and nearest advisory centers.
"""
from __future__ import annotations
from typing import Any

# Government helplines
HELPLINES = {
    "en": {
        "kisan_call_center": {"number": "1800-180-1551", "desc": "Kisan Call Center (Free, 24x7)"},
        "crop_insurance": {"number": "1800-200-7710", "desc": "PM Fasal Bima Yojana"},
        "soil_health": {"number": "1800-180-1551", "desc": "Soil Health Card Helpline"},
        "pm_kisan": {"number": "155261", "desc": "PM-KISAN Support"},
        "water_helpline": {"number": "1800-180-2042", "desc": "Water Resource Helpline"},
    },
    "hi": {
        "kisan_call_center": {"number": "1800-180-1551", "desc": "किसान कॉल सेंटर (मुफ्त, 24x7)"},
        "crop_insurance": {"number": "1800-200-7710", "desc": "पीएम फसल बीमा योजना"},
        "soil_health": {"number": "1800-180-1551", "desc": "मृदा स्वास्थ्य कार्ड हेल्पलाइन"},
        "pm_kisan": {"number": "155261", "desc": "पीएम-किसान सहायता"},
        "water_helpline": {"number": "1800-180-2042", "desc": "जल संसाधन हेल्पलाइन"},
    },
    "kn": {
        "kisan_call_center": {"number": "1800-180-1551", "desc": "ಕಿಸಾನ್ ಕಾಲ್ ಸೆಂಟರ್ (ಉಚಿತ, 24x7)"},
        "crop_insurance": {"number": "1800-200-7710", "desc": "ಪಿಎಂ ಫಸಲ್ ಬೀಮಾ ಯೋಜನೆ"},
        "pm_kisan": {"number": "155261", "desc": "ಪಿಎಂ-ಕಿಸಾನ್ ಸಹಾಯ"},
    },
}

# Nearest KVK database (major ones)
KVK_CENTERS = {
    "Karnataka": [
        {"name": "KVK Davangere", "district": "Davangere", "phone": "08192-233044", "lat": 14.46, "lon": 75.92},
        {"name": "KVK Bengaluru Rural", "district": "Bengaluru", "phone": "080-23636711", "lat": 13.08, "lon": 77.57},
        {"name": "KVK Dharwad", "district": "Hubli", "phone": "0836-2447783", "lat": 15.46, "lon": 75.01},
        {"name": "KVK Mandya", "district": "Mysuru", "phone": "08232-220269", "lat": 12.52, "lon": 76.90},
    ],
    "Andhra Pradesh": [
        {"name": "KVK Guntur", "district": "Guntur", "phone": "0863-2234990", "lat": 16.31, "lon": 80.44},
        {"name": "KVK Tirupati", "district": "Tirupati", "phone": "0877-2248484", "lat": 13.63, "lon": 79.42},
    ],
    "Telangana": [
        {"name": "KVK Rangareddy", "district": "Hyderabad", "phone": "040-24015222", "lat": 17.32, "lon": 78.41},
        {"name": "KVK Warangal", "district": "Warangal", "phone": "0870-2455513", "lat": 17.97, "lon": 79.59},
    ],
    "Tamil Nadu": [
        {"name": "KVK Kancheepuram", "district": "Chennai", "phone": "044-27452371", "lat": 12.83, "lon": 79.70},
        {"name": "KVK Coimbatore", "district": "Coimbatore", "phone": "0422-6611356", "lat": 11.01, "lon": 76.97},
    ],
    "Maharashtra": [
        {"name": "KVK Pune", "district": "Pune", "phone": "020-25691217", "lat": 18.52, "lon": 73.86},
        {"name": "KVK Nagpur", "district": "Nagpur", "phone": "0712-2500279", "lat": 21.15, "lon": 79.09},
    ],
    "Punjab": [
        {"name": "KVK Ludhiana", "district": "Ludhiana", "phone": "0161-2401960", "lat": 30.90, "lon": 75.86},
    ],
    "Gujarat": [
        {"name": "KVK Ahmedabad", "district": "Ahmedabad", "phone": "079-26305065", "lat": 23.02, "lon": 72.57},
    ],
    "Uttar Pradesh": [
        {"name": "KVK Lucknow", "district": "Lucknow", "phone": "0522-2370781", "lat": 26.85, "lon": 80.95},
    ],
}


def get_helplines(lang: str = "en") -> list[dict]:
    """Get helplines in farmer's language."""
    lines = HELPLINES.get(lang, HELPLINES["en"])
    return [{"name": k, **v} for k, v in lines.items()]


def get_nearest_kvk(state: str, district: str = "") -> dict[str, Any]:
    """Find nearest KVK for farmer's location."""
    kvks = KVK_CENTERS.get(state, [])
    if not kvks:
        return {
            "found": False,
            "message": f"Contact Kisan Call Center: 1800-180-1551 for KVK in {state}",
        }

    # Try to match district
    for kvk in kvks:
        if kvk["district"].lower() == district.lower():
            return {"found": True, **kvk}

    # Return first KVK in state
    return {"found": True, **kvks[0]}


def format_helpline_card(state: str, district: str, lang: str = "en") -> str:
    """Generate a formatted helpline card for the farmer."""
    kvk = get_nearest_kvk(state, district)
    helplines = get_helplines(lang)

    card = "## 📞 Emergency Helplines\n\n"
    for h in helplines:
        card += f"- **{h['desc']}**: `{h['number']}`\n"

    card += "\n## 🏛️ Nearest KVK\n"
    if kvk["found"]:
        card += f"- **{kvk['name']}** — {kvk.get('phone', 'N/A')}\n"
    else:
        card += f"- {kvk['message']}\n"

    return card


__all__ = ["get_helplines", "get_nearest_kvk", "format_helpline_card"]
