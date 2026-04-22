"""
Indian Crop Calendar — Season-based advisory for farmers
Provides planting/harvesting schedules and seasonal disease warnings.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


# Indian crop seasons
CROP_CALENDAR = {
    "rice": {
        "kharif": {
            "sowing": "June-July",
            "harvesting": "October-November",
            "diseases_peak": ["Blast (August-September)", "Brown spot (September)", "BLB (August)"],
            "advisory": "Prepare nursery in May. Transplant 25-day old seedlings. Apply basal dose NPK.",
        },
        "rabi": {
            "sowing": "November-December",
            "harvesting": "March-April",
            "diseases_peak": ["Sheath blight (February)"],
            "advisory": "Use short-duration varieties. Ensure good drainage.",
        },
    },
    "wheat": {
        "rabi": {
            "sowing": "October-November",
            "harvesting": "March-April",
            "diseases_peak": ["Yellow rust (January-February)", "Brown rust (February-March)", "Powdery mildew (January)"],
            "advisory": "Sow before November 15 for best yield. First irrigation at 21 DAS (crown root stage).",
        },
    },
    "cotton": {
        "kharif": {
            "sowing": "April-May",
            "harvesting": "October-December",
            "diseases_peak": ["Bollworm (July-August)", "Leaf curl (June-July)", "Bacterial blight (August)"],
            "advisory": "Use Bt cotton seeds. Install pheromone traps. Spray neem oil at 35 DAS.",
        },
    },
    "maize": {
        "kharif": {
            "sowing": "June-July",
            "harvesting": "September-October",
            "diseases_peak": ["Fall armyworm (July-August)", "Northern leaf blight (August)", "Common rust (August-September)"],
            "advisory": "Seed treatment with Thiram 3g/kg. Plant spacing 60x20cm.",
        },
        "rabi": {
            "sowing": "October-November",
            "harvesting": "February-March",
            "diseases_peak": ["Stem borer (November-December)"],
            "advisory": "Use winter maize varieties. Irrigate at critical stages.",
        },
    },
    "sugarcane": {
        "spring": {
            "sowing": "February-March",
            "harvesting": "December-February (next year)",
            "diseases_peak": ["Red rot (August-September)", "Smut (June-July)", "Top borer (April-May)"],
            "advisory": "Hot water treatment of setts at 50°C for 2 hours. Plant 3-budded setts.",
        },
    },
    "tomato": {
        "kharif": {
            "sowing": "June-July",
            "harvesting": "September-October",
            "diseases_peak": ["Late blight (August-September)", "Leaf curl (July-August)"],
            "advisory": "Raise seedlings in pro-trays. Transplant at 4-leaf stage. Mulch with paddy straw.",
        },
        "rabi": {
            "sowing": "October-November",
            "harvesting": "January-March",
            "diseases_peak": ["Early blight (December)", "Bacterial spot (January)"],
            "advisory": "Ideal season for tomato. Stake plants. Apply Trichoderma at planting.",
        },
    },
    "ragi": {
        "kharif": {
            "sowing": "June-July",
            "harvesting": "October-November",
            "diseases_peak": ["Blast (August-September)", "Rust (September)"],
            "advisory": "Transplant 25-day seedlings. Apply FYM 10t/ha. Weed at 20 and 40 DAS.",
        },
    },
    "potato": {
        "rabi": {
            "sowing": "October-November",
            "harvesting": "January-February",
            "diseases_peak": ["Late blight (December-January)", "Early blight (November)"],
            "advisory": "Use certified seed tubers. Earthing up at 30 and 45 DAS. Dehaulm 10 days before harvest.",
        },
    },
    "groundnut": {
        "kharif": {
            "sowing": "June-July",
            "harvesting": "October-November",
            "diseases_peak": ["Tikka disease (August)", "Collar rot (July)"],
            "advisory": "Seed treatment with Carbendazim 2g/kg. Apply gypsum at flowering.",
        },
    },
    "soybean": {
        "kharif": {
            "sowing": "June-July",
            "harvesting": "October",
            "diseases_peak": ["Yellow mosaic (August)", "Rust (September)"],
            "advisory": "Rhizobium seed inoculation. Row spacing 45cm. Harvest at 95% pod maturity.",
        },
    },
}


def get_current_season() -> str:
    """Determine current Indian agricultural season."""
    month = datetime.now().month
    if 6 <= month <= 9:
        return "kharif"
    elif 10 <= month <= 2 or month <= 2:
        return "rabi"
    else:
        return "spring"


def get_crop_advisory(crop: str, lang: str = "en") -> dict[str, Any]:
    """Get seasonal advisory for a crop."""
    crop = crop.lower()
    season = get_current_season()

    calendar = CROP_CALENDAR.get(crop, {})
    if not calendar:
        return {
            "crop": crop,
            "season": season,
            "advisory": f"No specific calendar data for {crop}. Consult local KVK.",
            "available": False,
        }

    # Find the relevant season
    season_info = calendar.get(season)
    if not season_info:
        # Fall back to any available season
        season_key = list(calendar.keys())[0]
        season_info = calendar[season_key]
        season = season_key

    return {
        "crop": crop,
        "season": season,
        "sowing_period": season_info.get("sowing", ""),
        "harvest_period": season_info.get("harvesting", ""),
        "current_disease_risks": season_info.get("diseases_peak", []),
        "advisory": season_info.get("advisory", ""),
        "available": True,
    }


def get_seasonal_warning(crop: str) -> str:
    """Get current seasonal disease warning for a crop."""
    month = datetime.now().month
    month_name = datetime.now().strftime("%B")

    advisory = get_crop_advisory(crop)
    if not advisory["available"]:
        return ""

    risks = advisory.get("current_disease_risks", [])
    if risks:
        risk_text = ", ".join(risks)
        return f"⚠️ {month_name} warning for {crop}: Watch for {risk_text}"

    return f"✅ No major disease risks for {crop} this month."


# Export
__all__ = ["get_crop_advisory", "get_seasonal_warning", "get_current_season", "CROP_CALENDAR"]
