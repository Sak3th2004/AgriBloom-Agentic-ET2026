"""
Smart Fertilizer Calculator
Gives exact NPK dose based on crop, area, and soil type.
Based on ICAR recommendations for Indian crops.
"""
from __future__ import annotations
from typing import Any


# ICAR recommended NPK doses (kg/hectare)
CROP_NPK = {
    "rice":       {"N": 120, "P": 60, "K": 60, "urea": 261, "dap": 130, "mop": 100, "note": "Split N: 50% basal + 25% at tillering + 25% at panicle"},
    "wheat":      {"N": 120, "P": 60, "K": 40, "urea": 261, "dap": 130, "mop": 67, "note": "Full P&K basal. Split N: 50% basal + 50% at first irrigation"},
    "maize":      {"N": 120, "P": 60, "K": 40, "urea": 261, "dap": 130, "mop": 67, "note": "Split N into 3 doses at sowing, knee-high, and tasseling"},
    "cotton":     {"N": 150, "P": 60, "K": 60, "urea": 326, "dap": 130, "mop": 100, "note": "Apply in 3 splits. Use neem-coated urea for slow release"},
    "sugarcane":  {"N": 250, "P": 115, "K": 115, "urea": 543, "dap": 250, "mop": 192, "note": "Heavy feeder. Apply FYM 25t/ha at planting"},
    "tomato":     {"N": 150, "P": 100, "K": 75, "urea": 326, "dap": 217, "mop": 125, "note": "Fertigate if drip. Add micronutrients Zn, B"},
    "potato":     {"N": 150, "P": 80, "K": 100, "urea": 326, "dap": 174, "mop": 167, "note": "Full dose at planting. Earthing up at 30 DAS"},
    "ragi":       {"N": 50, "P": 40, "K": 25, "urea": 109, "dap": 87, "mop": 42, "note": "Low input crop. Apply FYM 5t/ha"},
    "groundnut":  {"N": 20, "P": 40, "K": 40, "urea": 43, "dap": 87, "mop": 67, "note": "Apply gypsum 500kg/ha at flowering"},
    "soybean":    {"N": 30, "P": 60, "K": 30, "urea": 65, "dap": 130, "mop": 50, "note": "Rhizobium inoculation. Low N due to N-fixation"},
    "pepper":     {"N": 100, "P": 50, "K": 50, "urea": 217, "dap": 109, "mop": 83, "note": "Apply in splits. Mulch to conserve moisture"},
    "apple":      {"N": 70, "P": 35, "K": 70, "urea": 152, "dap": 76, "mop": 117, "note": "Per tree basis. Apply in February and June"},
    "grape":      {"N": 100, "P": 60, "K": 100, "urea": 217, "dap": 130, "mop": 167, "note": "Fertigate. High K for fruit quality"},
}


def calculate_fertilizer(
    crop: str,
    area_acres: float = 1.0,
    soil_type: str = "medium",
    lang: str = "en",
) -> dict[str, Any]:
    """
    Calculate fertilizer requirement for a crop.

    Args:
        crop: Crop name
        area_acres: Farm area in acres
        soil_type: low_fertility / medium / high_fertility
        lang: Language code

    Returns:
        Dict with NPK requirements and product quantities
    """
    crop = crop.lower()
    npk = CROP_NPK.get(crop)

    if not npk:
        return {
            "available": False,
            "message": f"No fertilizer data for '{crop}'. Consult local KVK.",
        }

    # Soil adjustment factor
    soil_factor = {"low_fertility": 1.2, "medium": 1.0, "high_fertility": 0.8}.get(soil_type, 1.0)

    # Convert acres to hectares (1 acre = 0.4047 ha)
    area_ha = area_acres * 0.4047

    # Calculate
    n_total = npk["N"] * soil_factor * area_ha
    p_total = npk["P"] * soil_factor * area_ha
    k_total = npk["K"] * soil_factor * area_ha
    urea_kg = npk["urea"] * soil_factor * area_ha
    dap_kg = npk["dap"] * soil_factor * area_ha
    mop_kg = npk["mop"] * soil_factor * area_ha

    # Estimated cost (approximate 2025 prices)
    cost = (urea_kg * 6) + (dap_kg * 27) + (mop_kg * 17)

    return {
        "available": True,
        "crop": crop.title(),
        "area_acres": area_acres,
        "area_ha": round(area_ha, 2),
        "soil_type": soil_type,
        "npk": {
            "nitrogen_kg": round(n_total, 1),
            "phosphorus_kg": round(p_total, 1),
            "potassium_kg": round(k_total, 1),
        },
        "products": {
            "urea_kg": round(urea_kg, 1),
            "dap_kg": round(dap_kg, 1),
            "mop_kg": round(mop_kg, 1),
        },
        "estimated_cost_rs": round(cost),
        "application_note": npk["note"],
    }


def format_fertilizer_card(crop: str, area_acres: float = 1.0, soil_type: str = "medium") -> str:
    """Format fertilizer recommendation as readable text."""
    result = calculate_fertilizer(crop, area_acres, soil_type)

    if not result["available"]:
        return result["message"]

    return (
        f"## 🧪 Fertilizer for {result['crop']} ({result['area_acres']} acres)\n\n"
        f"**NPK Required:**\n"
        f"- Nitrogen: {result['npk']['nitrogen_kg']} kg\n"
        f"- Phosphorus: {result['npk']['phosphorus_kg']} kg\n"
        f"- Potassium: {result['npk']['potassium_kg']} kg\n\n"
        f"**What to Buy:**\n"
        f"- Urea: **{result['products']['urea_kg']} kg** (₹{round(result['products']['urea_kg']*6)})\n"
        f"- DAP: **{result['products']['dap_kg']} kg** (₹{round(result['products']['dap_kg']*27)})\n"
        f"- MOP: **{result['products']['mop_kg']} kg** (₹{round(result['products']['mop_kg']*17)})\n\n"
        f"**Total Cost: ≈ ₹{result['estimated_cost_rs']}**\n\n"
        f"📝 *{result['application_note']}*"
    )


__all__ = ["calculate_fertilizer", "format_fertilizer_card", "CROP_NPK"]
