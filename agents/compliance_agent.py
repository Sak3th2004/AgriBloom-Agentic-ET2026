"""
Compliance Agent - FSSAI/ICAR/CIB&RC Regulatory Guardrails
Enforces Indian agricultural chemical regulations and safety standards
Generates audit trails for traceability

CRITICAL: This agent is DETERMINISTIC — no LLM involved.
All decisions are based on rule-matching against official Indian government databases.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Load compliance databases ───────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent / "compliance"


def _load_json(filename: str) -> dict:
    """Load a JSON compliance file with fallback."""
    path = _ROOT / filename
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {filename}: {e}")
    logger.warning(f"Compliance file not found: {filename}")
    return {}


# Load all databases at module level
BANNED_DB = _load_json("banned_pesticides.json")
MRL_DB = _load_json("mrl_limits.json")
ALTERNATIVES_DB = _load_json("safe_alternatives.json")
LEGACY_RULES = {}  # Legacy rules removed — now split into banned/mrl/alternatives

# Build search index for banned substances (lowercase for matching)
BANNED_SEARCH_TERMS = [t.lower() for t in BANNED_DB.get("search_terms", [])]
BANNED_NAMES = {
    item["name"].lower(): item
    for item in BANNED_DB.get("banned_for_manufacture_import_use", [])
}
RESTRICTED_NAMES = {
    item["name"].lower(): item
    for item in BANNED_DB.get("restricted_use", [])
}

# Required disclaimers in multiple languages
DISCLAIMERS = {
    "en": [
        "Validate all recommendations with your local Krishi Vigyan Kendra (KVK).",
        "Follow Integrated Pest Management (IPM) practices.",
        "Wear protective equipment when handling agrochemicals.",
        "Observe pre-harvest interval (PHI) before crop harvesting.",
        "This is AI-generated advice. Consult agricultural officer for critical decisions.",
        "Helpline: 1800-180-1551 (Kisan Call Center, Toll-Free)",
    ],
    "hi": [
        "सभी सिफारिशों को अपने स्थानीय कृषि विज्ञान केंद्र (KVK) से सत्यापित करें।",
        "एकीकृत कीट प्रबंधन (IPM) प्रथाओं का पालन करें।",
        "कृषि रसायनों को संभालते समय सुरक्षा उपकरण पहनें।",
        "फसल कटाई से पहले प्रतीक्षा अवधि (PHI) का पालन करें।",
        "यह AI-जनित सलाह है। महत्वपूर्ण निर्णयों के लिए कृषि अधिकारी से परामर्श करें।",
        "हेल्पलाइन: 1800-180-1551 (किसान कॉल सेंटर, टोल-फ्री)",
    ],
    "kn": [
        "ಎಲ್ಲಾ ಶಿಫಾರಸುಗಳನ್ನು ನಿಮ್ಮ ಸ್ಥಳೀಯ KVK ಯಿಂದ ಮಾನ್ಯಮಾಡಿ.",
        "ಸಮಗ್ರ ಕೀಟ ನಿರ್ವಹಣೆ (IPM) ಅನುಸರಿಸಿ.",
        "ಕೃಷಿ ರಾಸಾಯನಿಕಗಳನ್ನು ನಿರ್ವಹಿಸುವಾಗ ರಕ್ಷಣಾ ಸಲಕರಣೆ ಧರಿಸಿ.",
        "ಬೆಳೆ ಕೊಯ್ಲಿನ ಮೊದಲು ಕಾಯುವ ಅವಧಿ (PHI) ಪಾಲಿಸಿ.",
        "ಇದು AI-ಉತ್ಪಾದಿಸಿದ ಸಲಹೆ. ನಿರ್ಣಾಯಕ ನಿರ್ಧಾರಗಳಿಗೆ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "ಹೆಲ್ಪ್‌ಲೈನ್: 1800-180-1551 (ಕಿಸಾನ್ ಕಾಲ್ ಸೆಂಟರ್, ಉಚಿತ)",
    ],
    "te": [
        "మీ స్థానిక KVK నుండి అన్ని సిఫార్సులను ధృవీకరించండి.",
        "సమగ్ర సస్యరక్షణ (IPM) పద్ధతులు అనుసరించండి.",
        "వ్యవసాయ రసాయనాలను నిర్వహించేటప్పుడు రక్షణ పరికరాలు ధరించండి.",
        "పంట కోతకు ముందు వేచి ఉండే వ్యవధి (PHI) పాటించండి.",
        "ఇది AI-ఉత్పత్తి సలహా. క్లిష్టమైన నిర్ణయాల కోసం వ్యవసాయ అధికారిని సంప్రదించండి.",
        "హెల్ప్‌లైన్: 1800-180-1551 (కిసాన్ కాల్ సెంటర్, టోల్-ఫ్రీ)",
    ],
    "ta": [
        "உங்கள் உள்ளூர் KVK இலிருந்து அனைத்து பரிந்துரைகளையும் சரிபார்க்கவும்.",
        "ஒருங்கிணைந்த பூச்சி மேலாண்மை (IPM) நடைமுறைகளை பின்பற்றவும்.",
        "வேளாண் ரசாயனங்களை கையாளும்போது பாதுகாப்பு உபகரணங்கள் அணியவும்.",
        "பயிர் அறுவடைக்கு முன் காத்திருப்பு காலம் (PHI) கடைப்பிடிக்கவும்.",
        "இது AI-உருவாக்கிய ஆலோசனை. முக்கிய முடிவுகளுக்கு வேளாண் அதிகாரியை அணுகவும்.",
        "ஹெல்ப்லைன்: 1800-180-1551 (கிசான் கால் சென்டர், இலவசம்)",
    ],
    "pa": [
        "ਆਪਣੇ ਸਥਾਨਕ KVK ਤੋਂ ਸਾਰੀਆਂ ਸਿਫਾਰਸ਼ਾਂ ਦੀ ਪੁਸ਼ਟੀ ਕਰੋ।",
        "ਇਹ AI-ਉਤਪਾਦਿਤ ਸਲਾਹ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਫੈਸਲਿਆਂ ਲਈ ਖੇਤੀ ਅਧਿਕਾਰੀ ਨਾਲ ਸਲਾਹ ਕਰੋ।",
        "ਹੈਲਪਲਾਈਨ: 1800-180-1551 (ਕਿਸਾਨ ਕਾਲ ਸੈਂਟਰ, ਟੋਲ-ਫ੍ਰੀ)",
    ],
    "gu": [
        "તમારા સ્થાનિક KVK માંથી બધી ભલામણોની ખાતરી કરો.",
        "આ AI-ઉત્પાદિત સલાહ છે. મહત્વના નિર્ણયો માટે કૃષિ અધિકારીનો સંપર્ક કરો.",
        "હેલ્પલાઇન: 1800-180-1551 (કિસાન કોલ સેન્ટર, ટોલ-ફ્રી)",
    ],
    "mr": [
        "तुमच्या स्थानिक KVK कडून सर्व शिफारसींची पडताळणी करा.",
        "हा AI-निर्मित सल्ला आहे. महत्त्वाच्या निर्णयांसाठी कृषी अधिकाऱ्याशी संपर्क साधा.",
        "हेल्पलाईन: 1800-180-1551 (किसान कॉल सेंटर, टोल-फ्री)",
    ],
    "bn": [
        "আপনার স্থানীয় KVK থেকে সমস্ত সুপারিশ যাচাই করুন।",
        "এটি AI-উৎপাদিত পরামর্শ। গুরুত্বপূর্ণ সিদ্ধান্তের জন্য কৃষি কর্মকর্তার সাথে পরামর্শ করুন।",
        "হেল্পলাইন: ১৮০০-১৮০-১৫৫১ (কিষাণ কল সেন্টার, টোল-ফ্রি)",
    ],
    "or": [
        "ଆପଣଙ୍କ ସ୍ଥାନୀୟ KVK ରୁ ସମସ୍ତ ସୁପାରିଶ ଯାଞ୍ଚ କରନ୍ତୁ।",
        "ଏହା AI-ଉତ୍ପାଦିତ ପରାମର୍ଶ। ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ନିଷ୍ପତ୍ତି ପାଇଁ କୃଷି ଅଧିକାରୀଙ୍କ ସହ ପରାମର୍ଶ କରନ୍ତୁ।",
        "ହେଲ୍ପଲାଇନ: ୧୮୦୦-୧୮୦-୧୫୫୧ (କିସାନ କଲ ସେଣ୍ଟର, ଟୋଲ-ଫ୍ରି)",
    ],
}


def _check_banned_substances(text: str) -> list[dict]:
    """
    Check text for any mention of banned substances.
    Returns list of violations with regulation references.
    """
    text_lower = text.lower()
    violations = []

    for term in BANNED_SEARCH_TERMS:
        if term in text_lower:
            # Find the full banned entry
            for name, entry in BANNED_NAMES.items():
                if term in name or name in text_lower:
                    violations.append({
                        "chemical": entry["name"],
                        "status": "BANNED",
                        "ban_date": entry.get("ban_date", "unknown"),
                        "regulation": entry.get("regulation", "CIB&RC"),
                        "category": entry.get("category", "pesticide"),
                    })
                    break
            else:
                # Check restricted list
                for name, entry in RESTRICTED_NAMES.items():
                    if term in name or name in text_lower:
                        violations.append({
                            "chemical": entry["name"],
                            "status": "RESTRICTED",
                            "restriction": entry.get("restriction", ""),
                            "regulation": entry.get("regulation", "CIB&RC"),
                            "crops_restricted": entry.get("crops_restricted", []),
                        })
                        break

    # Deduplicate by chemical name
    seen = set()
    unique_violations = []
    for v in violations:
        key = v["chemical"].lower()
        if key not in seen:
            seen.add(key)
            unique_violations.append(v)

    return unique_violations


def _check_crop_restrictions(treatment: str, crop: str) -> list[dict]:
    """Check if a treatment chemical is restricted for a specific crop."""
    warnings = []
    treatment_lower = treatment.lower()

    for name, entry in RESTRICTED_NAMES.items():
        if name in treatment_lower:
            restricted_crops = entry.get("crops_restricted", [])
            for rc in restricted_crops:
                if crop.lower() in rc.lower() or rc.lower() in crop.lower():
                    warnings.append({
                        "chemical": entry["name"],
                        "crop": crop,
                        "restriction": entry.get("restriction", ""),
                        "regulation": entry.get("regulation", "CIB&RC"),
                    })

    return warnings


def _check_mrl_compliance(treatment: str, crop: str) -> list[dict]:
    """Check MRL limits for chemicals mentioned in treatment."""
    mrl_limits = MRL_DB.get("mrl_limits", {})
    warnings = []
    treatment_lower = treatment.lower()

    for chemical, limits in mrl_limits.items():
        chemical_clean = chemical.replace("_", " ")
        if chemical_clean in treatment_lower or chemical in treatment_lower:
            crop_limit = limits.get(crop, limits.get("default", None))
            phi = limits.get("pre_harvest_interval_days", "unknown")

            if crop_limit is not None:
                warnings.append({
                    "chemical": chemical_clean.title(),
                    "mrl_limit_mgkg": crop_limit,
                    "crop": crop,
                    "pre_harvest_interval_days": phi,
                    "warning": f"Ensure {chemical_clean.title()} residue stays below {crop_limit} mg/kg. "
                               f"Wait {phi} days before harvesting.",
                })

    return warnings


def _get_safe_alternatives(chemical: str) -> list[dict]:
    """Get safe alternatives for a banned/restricted chemical."""
    alternatives = ALTERNATIVES_DB.get("alternatives", {})
    chemical_key = chemical.lower().replace(" ", "_").replace("-", "_")

    # Try exact match first
    alt_entry = alternatives.get(chemical_key)
    if not alt_entry:
        # Try partial match
        for key, entry in alternatives.items():
            if key in chemical_key or chemical_key in key:
                alt_entry = entry
                break

    if not alt_entry:
        alt_entry = alternatives.get("default", {})

    return alt_entry.get("safe_alternatives", [])


def _get_disclaimers(lang: str = "en") -> list[str]:
    """Get disclaimers in the appropriate language."""
    return DISCLAIMERS.get(lang, DISCLAIMERS["en"])


def run_compliance(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Check recommendations against FSSAI/ICAR/CIB&RC compliance.

    DETERMINISTIC — no LLM involved. All decisions are rule-based.

    Input state keys:
        - knowledge: Dict with recommendations
        - treatment: Disease treatment recommendation
        - user_text: User's query text
        - crop_type: Detected crop type
        - lang: Language code

    Output state keys:
        - compliance: Full compliance report with audit trail
        - status: "compliance_complete" or "compliance_blocked"
    """
    lang = state.get("lang", "en")
    crop = state.get("crop_type", "unknown")
    treatment = state.get("treatment", "")
    user_text = state.get("user_text", "")
    recommendations = state.get("recommendations", [])

    # Combine all text for scanning
    all_text = " ".join([user_text, treatment, " ".join(recommendations)])

    # Timestamp for audit
    check_time = datetime.now().isoformat()
    audit_log = []

    # ── Step 1: Check for banned substances ──────────────────────────────
    violations = _check_banned_substances(all_text)
    audit_log.append({
        "timestamp": check_time,
        "agent": "compliance",
        "action": "check_banned_substances",
        "chemicals_scanned": len(BANNED_SEARCH_TERMS),
        "violations_found": len(violations),
        "result": "BLOCKED" if violations else "PASSED",
    })

    # ── Step 2: Check crop-specific restrictions ─────────────────────────
    crop_warnings = _check_crop_restrictions(treatment, crop)
    audit_log.append({
        "timestamp": check_time,
        "agent": "compliance",
        "action": "check_crop_restrictions",
        "crop": crop,
        "warnings_found": len(crop_warnings),
        "result": "WARNING" if crop_warnings else "PASSED",
    })

    # ── Step 3: Check MRL compliance ─────────────────────────────────────
    mrl_warnings = _check_mrl_compliance(treatment, crop)
    audit_log.append({
        "timestamp": check_time,
        "agent": "compliance",
        "action": "check_mrl_limits",
        "crop": crop,
        "chemicals_checked": len(mrl_warnings),
        "result": "MRL_INFO_ATTACHED",
    })

    # ── Step 4: Get safe alternatives for any violations ─────────────────
    safe_alternatives = {}
    for violation in violations:
        chem = violation["chemical"]
        alts = _get_safe_alternatives(chem)
        if alts:
            safe_alternatives[chem] = alts
            audit_log.append({
                "timestamp": check_time,
                "agent": "compliance",
                "action": "find_safe_alternative",
                "blocked_chemical": chem,
                "alternatives_found": len(alts),
                "result": "ALTERNATIVES_PROVIDED",
            })

    # ── Step 5: Build compliance report ──────────────────────────────────
    has_banned = any(v.get("status") == "BANNED" for v in violations)
    has_restricted = any(v.get("status") == "RESTRICTED" for v in violations)

    if has_banned:
        compliance_status = "unsafe"
        risk_level = "high"
    elif has_restricted or crop_warnings:
        compliance_status = "warning"
        risk_level = "medium"
    else:
        compliance_status = "safe"
        risk_level = "low"

    allowed = compliance_status != "unsafe"

    # Get disclaimers in farmer's language
    disclaimers = _get_disclaimers(lang)

    compliance_report = {
        "allowed": allowed,
        "compliance_status": compliance_status,
        "risk_level": risk_level,
        "violations": violations,
        "crop_warnings": crop_warnings,
        "mrl_warnings": mrl_warnings,
        "safe_alternatives": safe_alternatives,
        "disclaimers": disclaimers,
        "helpline": "1800-180-1551",
        "icar_guidelines": {
            "spray_timing": "Avoid spraying during flowering to protect pollinators.",
            "water_interval": "Maintain 24-48 hour interval between irrigation and spraying.",
            "mixing": "Never mix more than 2 pesticides. Test compatibility first.",
            "ppe": "Always wear mask, gloves, and full-sleeve clothing while spraying.",
        },
        "rule_version": BANNED_DB.get("metadata", {}).get("version", "3.0.0"),
        "checked_at": check_time,
        "audit_log": audit_log,
    }

    # Log compliance check result
    logger.info(
        f"Compliance check: status={compliance_status}, risk={risk_level}, "
        f"violations={len(violations)}, crop_warnings={len(crop_warnings)}, "
        f"mrl_checks={len(mrl_warnings)}"
    )

    if violations:
        for v in violations:
            logger.warning(
                f"COMPLIANCE VIOLATION: {v['chemical']} — {v.get('status')} — "
                f"{v.get('regulation', 'CIB&RC')}"
            )

    return {
        **state,
        "compliance": compliance_report,
        "status": "compliance_complete" if allowed else "compliance_blocked",
    }


# Export
__all__ = ["run_compliance", "BANNED_DB", "MRL_DB", "ALTERNATIVES_DB"]
