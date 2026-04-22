"""
Output Agent - Multilingual Voice & Visual Response Generation
Supports Kannada, Hindi, Telugu, Tamil with gTTS
Generates Bloom Simulator visualizations and PDF audit reports
"""
from __future__ import annotations

import logging
import math
import struct
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from gtts import gTTS

from utils.bloom_simulator import build_bloom_figure
from utils.pdf_audit import generate_audit_pdf

logger = logging.getLogger(__name__)

# Language mapping for gTTS
LANGUAGE_MAP = {
    "en": "en",
    "hi": "hi",
    "kn": "kn",
    "te": "te",
    "ta": "ta",
    "pa": "pa",
    "gu": "gu",
    "mr": "mr",
    "bn": "bn",
    "or": "or",
}

# Multilingual response templates
RESPONSE_TEMPLATES = {
    "en": {
        "header": "AgriBloom Advisory Report",
        "disease_detected": "Detected Condition: {disease}",
        "confidence": "Confidence Level: {confidence}",
        "severity": "Severity: {severity} - Potential Yield Loss: {yield_loss}",
        "weather": "Current Weather: {temp}°C, Rainfall: {rain}mm",
        "market": "Market Price: ₹{price}/quintal at {mandi} mandi",
        "actions": "Recommended Actions:",
        "disclaimer": "Important Disclaimers:",
        "compliance_alert": "COMPLIANCE ALERT: Your query mentions restricted agricultural chemicals.",
        "consult": "Please consult your local Krishi Vigyan Kendra (KVK) for approved alternatives.",
    },
    "hi": {
        "header": "एग्रीब्लूम सलाहकार रिपोर्ट",
        "disease_detected": "पहचानी गई स्थिति: {disease}",
        "confidence": "विश्वास स्तर: {confidence}",
        "severity": "गंभीरता: {severity} - संभावित उपज हानि: {yield_loss}",
        "weather": "वर्तमान मौसम: {temp}°C, वर्षा: {rain}mm",
        "market": "बाजार भाव: ₹{price}/क्विंटल ({mandi} मंडी में)",
        "actions": "अनुशंसित कार्य:",
        "disclaimer": "महत्वपूर्ण अस्वीकरण:",
        "compliance_alert": "अनुपालन चेतावनी: आपकी क्वेरी में प्रतिबंधित कृषि रसायनों का उल्लेख है।",
        "consult": "कृपया स्वीकृत विकल्पों के लिए अपने स्थानीय कृषि विज्ञान केंद्र से संपर्क करें।",
    },
    "kn": {
        "header": "ಅಗ್ರಿಬ್ಲೂಮ್ ಸಲಹಾ ವರದಿ",
        "disease_detected": "ಪತ್ತೆಯಾದ ಸ್ಥಿತಿ: {disease}",
        "confidence": "ವಿಶ್ವಾಸ ಮಟ್ಟ: {confidence}",
        "severity": "ತೀವ್ರತೆ: {severity} - ಸಂಭಾವ್ಯ ಇಳುವರಿ ನಷ್ಟ: {yield_loss}",
        "weather": "ಪ್ರಸ್ತುತ ಹವಾಮಾನ: {temp}°C, ಮಳೆ: {rain}mm",
        "market": "ಮಾರುಕಟ್ಟೆ ಬೆಲೆ: ₹{price}/ಕ್ವಿಂಟಲ್ ({mandi} ಮಂಡಿ)",
        "actions": "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮಗಳು:",
        "disclaimer": "ಮುಖ್ಯ ಹಕ್ಕು ನಿರಾಕರಣೆಗಳು:",
        "compliance_alert": "ಅನುಸರಣೆ ಎಚ್ಚರಿಕೆ: ನಿಮ್ಮ ಪ್ರಶ್ನೆಯಲ್ಲಿ ನಿಷೇಧಿತ ಕೃಷಿ ರಾಸಾಯನಿಕಗಳ ಉಲ್ಲೇಖವಿದೆ.",
        "consult": "ದಯವಿಟ್ಟು ಅನುಮೋದಿತ ಪರ್ಯಾಯಗಳಿಗಾಗಿ ನಿಮ್ಮ ಸ್ಥಳೀಯ KVK ಅನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    },
    "te": {
        "header": "అగ్రిబ్లూమ్ సలహా నివేదిక",
        "disease_detected": "గుర్తించిన స్థితి: {disease}",
        "confidence": "నమ్మకం స్థాయి: {confidence}",
        "severity": "తీవ్రత: {severity} - సంభావ్య దిగుబడి నష్టం: {yield_loss}",
        "weather": "ప్రస్తుత వాతావరణం: {temp}°C, వర్షపాతం: {rain}mm",
        "market": "మార్కెట్ ధర: ₹{price}/క్వింటాల్ ({mandi} మండి)",
        "actions": "సిఫార్సు చేసిన చర్యలు:",
        "disclaimer": "ముఖ్యమైన హక్కు నిరాకరణలు:",
        "compliance_alert": "అనుసరణ హెచ్చరిక: మీ ప్రశ్నలో నిషేధిత వ్యవసాయ రసాయనాలు ఉన్నాయి.",
        "consult": "దయచేసి ఆమోదించిన ప్రత్యామ్నాయాల కోసం మీ స్థానిక KVK ని సంప్రదించండి.",
    },
    "ta": {
        "header": "அக்ரிப்ளூம் ஆலோசனை அறிக்கை",
        "disease_detected": "கண்டறியப்பட்ட நிலை: {disease}",
        "confidence": "நம்பகத்தன்மை அளவு: {confidence}",
        "severity": "தீவிரம்: {severity} - சாத்தியமான விளைச்சல் இழப்பு: {yield_loss}",
        "weather": "தற்போதைய வானிலை: {temp}°C, மழை: {rain}mm",
        "market": "சந்தை விலை: ₹{price}/குவிண்டால் ({mandi} மண்டி)",
        "actions": "பரிந்துரைக்கப்பட்ட நடவடிக்கைகள்:",
        "disclaimer": "முக்கிய மறுப்புகள்:",
        "compliance_alert": "இணக்க எச்சரிக்கை: உங்கள் கேள்வியில் தடைசெய்யப்பட்ட வேளாண் இரசாயனங்கள் உள்ளன.",
        "consult": "அங்கீகரிக்கப்பட்ட மாற்றுகளுக்கு உங்கள் உள்ளூர் KVK ஐ அணுகவும்.",
    },
    "pa": {
        "header": "ਐਗਰੀਬਲੂਮ ਸਲਾਹ ਰਿਪੋਰਟ",
        "disease_detected": "ਪਛਾਣੀ ਗਈ ਸਥਿਤੀ: {disease}",
        "confidence": "ਭਰੋਸਾ ਪੱਧਰ: {confidence}",
        "severity": "ਗੰਭੀਰਤਾ: {severity} - ਸੰਭਾਵੀ ਝਾੜ ਨੁਕਸਾਨ: {yield_loss}",
        "weather": "ਮੌਜੂਦਾ ਮੌਸਮ: {temp}°C, ਮੀਂਹ: {rain}mm",
        "market": "ਮੰਡੀ ਭਾਅ: ₹{price}/ਕੁਇੰਟਲ ({mandi} ਮੰਡੀ)",
        "actions": "ਸਿਫਾਰਸ਼ ਕੀਤੀਆਂ ਕਾਰਵਾਈਆਂ:",
        "disclaimer": "ਮਹੱਤਵਪੂਰਨ ਬੇਦਾਅਵਾ:",
        "compliance_alert": "ਪਾਲਣਾ ਚੇਤਾਵਨੀ: ਤੁਹਾਡੀ ਪੁੱਛ-ਗਿੱਛ ਵਿੱਚ ਪਾਬੰਦੀਸ਼ੁਦਾ ਖੇਤੀ ਰਸਾਇਣ ਹਨ।",
        "consult": "ਪ੍ਰਵਾਨਿਤ ਵਿਕਲਪਾਂ ਲਈ ਆਪਣੇ ਸਥਾਨਕ KVK ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
    },
    "gu": {
        "header": "એગ્રીબ્લૂમ સલાહ રિપોર્ટ",
        "disease_detected": "ઓળખાયેલી સ્થિતિ: {disease}",
        "confidence": "વિશ્વાસ સ્તર: {confidence}",
        "severity": "ગંભીરતા: {severity} - સંભવિત ઉપજ નુકસાન: {yield_loss}",
        "weather": "હાલનું હવામાન: {temp}°C, વરસાદ: {rain}mm",
        "market": "બજાર ભાવ: ₹{price}/ક્વિન્ટલ ({mandi} માર્કેટ)",
        "actions": "ભલામણ કરેલ પગલાં:",
        "disclaimer": "મહત્વપૂર્ણ અસ્વીકરણ:",
        "compliance_alert": "અનુપાલન ચેતવણી: તમારી ક્વેરીમાં પ્રતિબંધિત કૃષિ રસાયણો છે.",
        "consult": "મંજૂર વિકલ્પો માટે તમારા સ્થાનિક KVK નો સંપર્ક કરો.",
    },
    "mr": {
        "header": "ऍग्रीब्लूम सल्ला अहवाल",
        "disease_detected": "ओळखलेली स्थिती: {disease}",
        "confidence": "विश्वास पातळी: {confidence}",
        "severity": "तीव्रता: {severity} - संभाव्य उत्पादन नुकसान: {yield_loss}",
        "weather": "सध्याचे हवामान: {temp}°C, पाऊस: {rain}mm",
        "market": "बाजार भाव: ₹{price}/क्विंटल ({mandi} मंडी)",
        "actions": "शिफारस केलेल्या कृती:",
        "disclaimer": "महत्त्वाचे अस्वीकरण:",
        "compliance_alert": "अनुपालन इशारा: तुमच्या प्रश्नात प्रतिबंधित कृषी रसायने आहेत.",
        "consult": "मंजूर पर्यायांसाठी तुमच्या स्थानिक KVK शी संपर्क साधा.",
    },
    "bn": {
        "header": "এগ্রিব্লুম পরামর্শ প্রতিবেদন",
        "disease_detected": "সনাক্ত অবস্থা: {disease}",
        "confidence": "আত্মবিশ্বাস স্তর: {confidence}",
        "severity": "তীব্রতা: {severity} - সম্ভাব্য ফলন ক্ষতি: {yield_loss}",
        "weather": "বর্তমান আবহাওয়া: {temp}°C, বৃষ্টি: {rain}mm",
        "market": "বাজার দর: ₹{price}/কুইন্টাল ({mandi} মান্ডি)",
        "actions": "সুপারিশকৃত পদক্ষেপ:",
        "disclaimer": "গুরুত্বপূর্ণ দাবিত্যাগ:",
        "compliance_alert": "অনুসরণ সতর্কতা: আপনার প্রশ্নে নিষিদ্ধ কৃষি রাসায়নিক রয়েছে।",
        "consult": "অনুমোদিত বিকল্পের জন্য আপনার স্থানীয় KVK এর সাথে যোগাযোগ করুন।",
    },
    "or": {
        "header": "ଏଗ୍ରିବ୍ଲୁମ୍ ପରାମର୍ଶ ରିପୋର୍ଟ",
        "disease_detected": "ଚିହ୍ନଟ ହୋଇଥିବା ଅବସ୍ଥା: {disease}",
        "confidence": "ବିଶ୍ୱାସ ସ୍ତର: {confidence}",
        "severity": "ତୀବ୍ରତା: {severity} - ସମ୍ଭାବ୍ୟ ଅମଳ କ୍ଷତି: {yield_loss}",
        "weather": "ବର୍ତ୍ତମାନ ପାଣିପାଗ: {temp}°C, ବର୍ଷା: {rain}mm",
        "market": "ବଜାର ଦର: ₹{price}/କ୍ୱିଣ୍ଟାଲ ({mandi} ମଣ୍ଡି)",
        "actions": "ସୁପାରିଶ କରାଯାଇଥିବା କାର୍ଯ୍ୟ:",
        "disclaimer": "ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ଦାବିତ୍ୟାଗ:",
        "compliance_alert": "ଅନୁପାଳନ ସତର୍କତା: ଆପଣଙ୍କ ପ୍ରଶ୍ନରେ ନିଷିଦ୍ଧ କୃଷି ରାସାୟନିକ ଅଛି।",
        "consult": "ଅନୁମୋଦିତ ବିକଳ୍ପ ପାଇଁ ଆପଣଙ୍କ ସ୍ଥାନୀୟ KVK ସହ ଯୋଗାଯୋଗ କରନ୍ତୁ।",
    },
}

# Disease name translations
DISEASE_NAMES = {
    "uncertain_detection": {"en": "Uncertain - Need clearer photo", "hi": "अनिश्चित - स्पष्ट फोटो चाहिए", "kn": "ಅನಿಶ್ಚಿತ - ಸ್ಪಷ್ಟ ಫೋಟೋ ಬೇಕು", "te": "అనిశ్చితం - స్పష్టమైన ఫోటో కావాలి", "ta": "நிச்சயமற்றது - தெளிவான புகைப்படம் தேவை"},
    "unknown": {"en": "Unknown - Upload crop photo", "hi": "अज्ञात - फसल फोटो अपलोड करें", "kn": "ಅಜ್ಞಾತ - ಬೆಳೆ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ", "te": "తెలియదు - పంట ఫోటో అప్‌లోడ్ చేయండి", "ta": "தெரியாது - பயிர் புகைப்படம் பதிவேற்றவும்"},
    "maize_blight": {"en": "Maize Blight", "hi": "मक्का झुलसा", "kn": "ಮೆಕ್ಕೆಜೋಳ ಬ್ಲೈಟ್", "te": "మొక్కజొన్న బ్లైట్", "ta": "சோள தீக்காயம்"},
    "maize_(maize)_northern_leaf_blight": {"en": "Maize Northern Leaf Blight", "hi": "मक्का उत्तरी पत्ती झुलसा", "kn": "ಮೆಕ್ಕೆಜೋಳ ಉತ್ತರ ಎಲೆ ಬ್ಲೈಟ್", "te": "మొక్కజొన్న ఉత్తర ఆకు బ్లైట్", "ta": "சோள வடக்கு இலை தீக்காயம்"},
    "maize_(maize)_common_rust_": {"en": "Maize Common Rust", "hi": "मक्का सामान्य रस्ट", "kn": "ಮೆಕ್ಕೆಜೋಳ ತುಕ್ಕು", "te": "మొక్కజొన్న తుప్పు", "ta": "சோள துரு"},
    "maize_(maize)_healthy": {"en": "Healthy Maize", "hi": "स्वस्थ मक्का", "kn": "ಆರೋಗ್ಯಕರ ಮೆಕ್ಕೆಜೋಳ", "te": "ఆరోగ్యకరమైన మొక్కజొన్న", "ta": "ஆரோக்கியமான சோளம்"},
    "tomato_early_blight": {"en": "Tomato Early Blight", "hi": "टमाटर अगेती झुलसा", "kn": "ಟೊಮ್ಯಾಟೊ ಮುಂಚೆಯ ಬ್ಲೈಟ್", "te": "టమాటా ముందస్తు బ్లైట్", "ta": "தக்காளி ஆரம்ப தீக்காயம்"},
    "tomato_late_blight": {"en": "Tomato Late Blight", "hi": "टमाटर पछेती झुलसा", "kn": "ಟೊಮ್ಯಾಟೊ ತಡವಾದ ಬ್ಲೈಟ್", "te": "టమాటా చివరి బ్లైట్", "ta": "தக்காளி தாமதமான தீக்காயம்"},
    "tomato_healthy": {"en": "Healthy Tomato", "hi": "स्वस्थ टमाटर", "kn": "ಆರೋಗ್ಯಕರ ಟೊಮ್ಯಾಟೊ", "te": "ఆరోగ్యకరమైన టమాటా", "ta": "ஆரோக்கியமான தக்காளி"},
    "potato_early_blight": {"en": "Potato Early Blight", "hi": "आलू अगेती झुलसा", "kn": "ಆಲೂಗಡ್ಡೆ ಮುಂಚೆಯ ಬ್ಲೈಟ್", "te": "బంగాళదుంప ముందస్తు బ్లైట్", "ta": "உருளைக்கிழங்கு ஆரம்ப தீக்காயம்"},
    "potato_late_blight": {"en": "Potato Late Blight", "hi": "आलू पछेती झुलसा", "kn": "ಆಲೂಗಡ್ಡೆ ತಡವಾದ ಬ್ಲೈಟ್", "te": "బంగాళదుంప చివరి బ్లైట్", "ta": "உருளைக்கிழங்கு தாமதமான தீக்காயம்"},
    "potato_healthy": {"en": "Healthy Potato", "hi": "स्वस्थ आलू", "kn": "ಆರೋಗ್ಯಕರ ಆಲೂಗಡ್ಡೆ", "te": "ఆరోగ్యకరమైన బంగాళదుంప", "ta": "ஆரோக்கியமான உருளைக்கிழங்கு"},
    "sugarcane_redrot": {"en": "Sugarcane Red Rot", "hi": "गन्ना लाल सड़न", "kn": "ಕಬ್ಬು ಕೆಂಪು ಕೊಳೆತ", "te": "చెరకు ఎరుపు కుళ్ళు", "ta": "கரும்பு சிவப்பு அழுகல்"},
    "sugarcane_rust": {"en": "Sugarcane Rust", "hi": "गन्ना रस्ट", "kn": "ಕಬ್ಬು ತುಕ್ಕು", "te": "చెరకు తుప్పు", "ta": "கரும்பு துரு"},
    "sugarcane_healthy": {"en": "Healthy Sugarcane", "hi": "स्वस्थ गन्ना", "kn": "ಆರೋಗ್ಯಕರ ಕಬ್ಬು", "te": "ఆరోగ్యకరమైన చెరకు", "ta": "ஆரோக்கியமான கரும்பு"},
    "wheat_yellow_rust": {"en": "Wheat Yellow Rust", "hi": "गेहूं पीला रस्ट", "kn": "ಗೋಧಿ ಹಳದಿ ತುಕ್ಕು", "te": "గోధుమ పసుపు తుప్పు", "ta": "கோதுமை மஞ்சள் துரு"},
    "wheat_healthy": {"en": "Healthy Wheat", "hi": "स्वस्थ गेहूं", "kn": "ಆರೋಗ್ಯಕರ ಗೋಧಿ", "te": "ఆరోగ్యకరమైన గోధుమ", "ta": "ஆரோக்கியமான கோதுமை"},
    "ragi_blast": {"en": "Ragi Blast", "hi": "रागी ब्लास्ट", "kn": "ರಾಗಿ ಬ್ಲಾಸ್ಟ್", "te": "రాగి బ్లాస్ట్", "ta": "கேழ்வரகு வெடிப்பு"},
    "ragi_healthy": {"en": "Healthy Ragi", "hi": "स्वस्थ रागी", "kn": "ಆರೋಗ್ಯಕರ ರಾಗಿ", "te": "ఆరోగ్యకరమైన రాగి", "ta": "ஆரோக்கியமான கேழ்வரகு"},
}


def _get_disease_name(disease_label: str, lang: str) -> str:
    """Get localized disease name."""
    names = DISEASE_NAMES.get(disease_label, {})
    return names.get(lang, names.get("en", disease_label.replace("_", " ").title()))


def _format_response(state: dict[str, Any], lang: str = "en") -> str:
    """Format comprehensive response in the target language."""
    templates = RESPONSE_TEMPLATES.get(lang, RESPONSE_TEMPLATES["en"])
    compliance = state.get("compliance", {})

    # Check compliance first
    if not compliance.get("allowed", True):
        return (
            f"⚠️ {templates['compliance_alert']}\n\n"
            f"❌ {templates['consult']}\n\n"
            f"Blocked substances: {', '.join(compliance.get('violations', []))}"
        )

    # Build response
    disease = state.get("disease_prediction", {})
    disease_label = disease.get("label", "unknown")
    confidence = disease.get("confidence", 0.0)

    knowledge = state.get("knowledge", {})
    weather = knowledge.get("weather", {})
    market = knowledge.get("market", {})
    agronomy = knowledge.get("agronomy", {})

    recommendations = state.get("recommendations", [])
    treatment = state.get("treatment", "")
    
    # Handle uncertain/unknown detection
    if disease_label in ["uncertain_detection", "unknown", "error"]:
        uncertain_msg = {
            "en": "⚠️ Could not identify crop disease clearly.\n\n📸 Tips for better photo:\n• Use good lighting\n• Take close-up of affected leaf\n• Avoid shadows and blur\n• Show both healthy and diseased parts\n\nPlease upload a clearer photo of the affected crop leaf.",
            "hi": "⚠️ फसल की बीमारी स्पष्ट रूप से पहचान नहीं हो पाई।\n\n📸 बेहतर फोटो के लिए:\n• अच्छी रोशनी का उपयोग करें\n• प्रभावित पत्ती का क्लोज-अप लें\n• छाया और धुंधलापन से बचें\n\nकृपया प्रभावित फसल की पत्ती की स्पष्ट फोटो अपलोड करें।",
            "kn": "⚠️ ಬೆಳೆ ರೋಗವನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.\n\n📸 ಉತ್ತಮ ಫೋಟೋಗಾಗಿ:\n• ಉತ್ತಮ ಬೆಳಕು ಬಳಸಿ\n• ಪ್ರಭಾವಿತ ಎಲೆಯ ಕ್ಲೋಸ್-ಅಪ್ ತೆಗೆಯಿರಿ\n• ನೆರಳು ಮತ್ತು ಮಸುಕು ತಪ್ಪಿಸಿ\n\nದಯವಿಟ್ಟು ಪ್ರಭಾವಿತ ಬೆಳೆ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
            "te": "⚠️ పంట వ్యాధిని స్పష్టంగా గుర్తించలేకపోయాము.\n\n📸 మంచి ఫోటో కోసం:\n• మంచి వెలుతురు ఉపయోగించండి\n• ప్రభావిత ఆకు క్లోజ్-అప్ తీయండి\n• నీడలు మరియు అస్పష్టత నివారించండి\n\nదయచేసి ప్రభావిత పంట ఆకు యొక్క స్పష్టమైన ఫోటో అప్‌లోడ్ చేయండి.",
            "ta": "⚠️ பயிர் நோயை தெளிவாக அடையாளம் காண முடியவில்லை.\n\n📸 சிறந்த புகைப்படத்திற்கு:\n• நல்ல ஒளியைப் பயன்படுத்தவும்\n• பாதிக்கப்பட்ட இலையின் க்ளோஸ்-அப் எடுக்கவும்\n• நிழல் மற்றும் மங்கலைத் தவிர்க்கவும்\n\nபாதிக்கப்பட்ட பயிர் இலையின் தெளிவான புகைப்படத்தை பதிவேற்றவும்.",
        }
        return uncertain_msg.get(lang, uncertain_msg["en"])

    lines = [
        f"🌾 {templates['header']}",
        "=" * 40,
        "",
        f"🔬 {templates['disease_detected'].format(disease=_get_disease_name(disease_label, lang))}",
        f"📊 {templates['confidence'].format(confidence=f'{confidence:.0%}')}",
    ]

    # Add severity if available and disease is valid
    if agronomy and disease_label not in ["uncertain_detection", "unknown"]:
        severity = agronomy.get("severity", "unknown")
        yield_loss = agronomy.get("yield_loss_range", "varies")
        lines.append(f"⚠️ {templates['severity'].format(severity=severity, yield_loss=yield_loss)}")

    lines.append("")

    # Weather info
    lines.append(
        f"🌤️ {templates['weather'].format(temp=weather.get('temp_c', 'N/A'), rain=weather.get('rain_mm', 0))}"
    )

    # Only show market info for valid crop detection
    if disease_label not in ["uncertain_detection", "unknown", "error"]:
        lines.append(
            f"💰 {templates['market'].format(price=market.get('modal_price', 'N/A'), mandi=market.get('mandi', 'Local'))}"
        )

    lines.append("")

    # Recommended actions
    lines.append(f"📋 {templates['actions']}")

    # Add treatment first
    if treatment:
        lines.append(f"   💊 {treatment}")

    # Add other recommendations (filter out market-related if unknown crop)
    for i, rec in enumerate(recommendations[:5], 1):
        if disease_label in ["uncertain_detection", "unknown"] and "market" in rec.lower():
            continue
        lines.append(f"   {i}. {rec}")

    lines.append("")

    # Disclaimers
    disclaimers = compliance.get("disclaimers", [])
    if disclaimers:
        lines.append(f"📢 {templates['disclaimer']}")
        for disclaimer in disclaimers[:3]:
            lines.append(f"   • {disclaimer}")

    return "\n".join(lines)


def _generate_voice(text: str, lang: str, output_dir: Path) -> Path:
    """Generate voice output using gTTS."""
    tts_lang = LANGUAGE_MAP.get(lang, "en")
    audio_path = output_dir / f"response_{lang}_{datetime.now().strftime('%H%M%S')}.mp3"

    try:
        # Simplify text for TTS (remove emojis)
        clean_text = text.replace("🌾", "").replace("🔬", "").replace("📊", "")
        clean_text = clean_text.replace("⚠️", "").replace("🌤️", "").replace("💰", "")
        clean_text = clean_text.replace("📋", "").replace("📢", "").replace("💊", "")
        clean_text = clean_text.replace("❌", "").replace("=", "")

        tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
        tts.save(str(audio_path))
        logger.info(f"Generated voice output: {audio_path}")
        return audio_path

    except Exception as e:
        logger.warning(f"gTTS failed: {e}. Generating fallback audio.")
        return _generate_fallback_audio(output_dir)


def _generate_fallback_audio(output_dir: Path) -> Path:
    """Generate fallback audio (beep tone) when TTS is unavailable."""
    audio_path = output_dir / "fallback_response.wav"

    sample_rate = 16000
    duration = 2.0
    frequencies = [440, 550, 660]  # A4, C#5, E5 chord

    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = []
        for i in range(int(duration * sample_rate)):
            value = sum(
                int(4000 * math.sin(2 * math.pi * f * (i / sample_rate)))
                for f in frequencies
            )
            value = max(-32767, min(32767, value))
            frames.append(struct.pack("<h", value))

        wav_file.writeframes(b"".join(frames))

    logger.info(f"Generated fallback audio: {audio_path}")
    return audio_path


def _calculate_health_trajectory(disease_label: str, confidence: float) -> tuple[float, float]:
    """Calculate before/after health scores for Bloom Simulator."""
    # Base health depends on disease severity
    severity_map = {
        "healthy": (85, 95),
        "blight": (35, 75),
        "rust": (45, 80),
        "rot": (25, 65),
        "blast": (40, 78),
        "spot": (50, 82),
        "default": (50, 80),
    }

    before_health = 50.0
    after_potential = 85.0

    for keyword, (before, after) in severity_map.items():
        if keyword in disease_label.lower():
            before_health = before
            after_potential = after
            break

    # Adjust based on confidence (higher confidence = more accurate projection)
    improvement = (after_potential - before_health) * confidence
    after_health = before_health + improvement

    return before_health, min(95.0, after_health)


def run_output(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Generate final output (text, voice, chart, PDF).

    Input state keys:
        - disease_prediction: Disease info from vision
        - knowledge: Weather, market, agronomy info
        - recommendations: Action recommendations
        - treatment: Disease treatment
        - compliance: Compliance report
        - lang/user_language: Language code

    Output state keys:
        - final_response: Formatted text response
        - voice_output_path: Path to audio file
        - bloom_figure: Plotly figure object
        - audit_pdf_path: Path to PDF audit log
        - status: "output_complete"
    """
    lang = state.get("lang", state.get("user_language", "en"))
    compliance = state.get("compliance", {})

    # Generate text response
    response_text = _format_response(state, lang)

    # Setup output directory
    out_dir = Path("models/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate voice
    audio_path = _generate_voice(response_text, lang, out_dir)

    # Generate Bloom Simulator chart
    disease = state.get("disease_prediction", {})
    disease_label = disease.get("label", "unknown")
    confidence = disease.get("confidence", 0.0)

    before_health, after_health = _calculate_health_trajectory(disease_label, confidence)
    bloom_figure = build_bloom_figure(
        before_health=before_health,
        after_health=after_health,
        days=14,
    )

    # Generate audit PDF
    audit_payload = {
        "disease": disease_label,
        "disease_localized": _get_disease_name(disease_label, lang),
        "confidence": f"{confidence:.2%}",
        "crop_type": state.get("crop_type", "unknown"),
        "compliance_allowed": compliance.get("allowed", True),
        "risk_level": compliance.get("risk_level", "low"),
        "violations": ", ".join(compliance.get("violations", [])) or "None",
        "disclaimers": " | ".join(compliance.get("disclaimers", [])[:3]),
        "recommendations": state.get("recommendations", [])[:5],
        "weather": state.get("knowledge", {}).get("weather", {}),
        "market": state.get("knowledge", {}).get("market", {}),
        "language": lang,
        "timestamp": datetime.now().isoformat(),
    }
    audit_pdf_path = generate_audit_pdf(audit_payload)

    return {
        **state,
        "final_response": response_text,
        "voice_output_path": str(audio_path),
        "bloom_figure": bloom_figure,
        "audit_pdf_path": audit_pdf_path,
        "status": "output_complete",
    }


# Export
__all__ = ["run_output", "RESPONSE_TEMPLATES", "DISEASE_NAMES"]
