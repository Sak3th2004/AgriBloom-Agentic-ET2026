"""
Knowledge Agent - Weather, Market, and Agronomic Advisory
Real-time data from Open-Meteo API + Government eNAM mandi data
Supports Indian regional languages
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

import requests

from utils.offline_cache import OfflineCache

logger = logging.getLogger(__name__)

# Initialize cache
CACHE = OfflineCache()

# MSP (Minimum Support Prices) 2025-26 as per Government of India
MSP_PRICES_2025 = {
    "maize": 2225,      # Rs/quintal
    "tomato": 1800,     # Market driven, no MSP
    "potato": 1650,     # Market driven
    "rice": 2320,       # Paddy MSP
    "wheat": 2275,      # Wheat MSP
    "ragi": 4290,       # Ragi MSP
    "sugarcane": 340,   # FRP per quintal
    "cotton": 7121,     # Medium staple MSP
    "soybean": 4892,    # Soybean MSP
    "groundnut": 6377,  # Groundnut MSP
    "pepper": 3500,     # Market driven
    "apple": 8000,      # Market driven (varies by grade)
    "grape": 5500,      # Market driven
    "orange": 4000,     # Market driven
    "cherry": 12000,    # Market driven
    "peach": 6000,      # Market driven
    "strawberry": 15000, # Market driven
    "squash": 2000,     # Market driven
}

# Major mandis for each crop (with lat/lon for proximity)
CROP_MANDIS = {
    "maize": [
        {"name": "Davangere", "state": "Karnataka", "lat": 14.4644, "lon": 75.9218},
        {"name": "Nizamabad", "state": "Telangana", "lat": 18.6725, "lon": 78.0941},
        {"name": "Karimnagar", "state": "Telangana", "lat": 18.4386, "lon": 79.1288},
    ],
    "tomato": [
        {"name": "Kolar", "state": "Karnataka", "lat": 13.1367, "lon": 78.1292},
        {"name": "Madanapalle", "state": "Andhra Pradesh", "lat": 13.5504, "lon": 78.5036},
        {"name": "Chittoor", "state": "Andhra Pradesh", "lat": 13.2172, "lon": 79.1003},
    ],
    "potato": [
        {"name": "Agra", "state": "Uttar Pradesh", "lat": 27.1767, "lon": 78.0081},
        {"name": "Farrukhabad", "state": "Uttar Pradesh", "lat": 27.3906, "lon": 79.5800},
        {"name": "Hassan", "state": "Karnataka", "lat": 13.0068, "lon": 76.1004},
    ],
    "rice": [
        {"name": "Guntur", "state": "Andhra Pradesh", "lat": 16.3067, "lon": 80.4365},
        {"name": "Raichur", "state": "Karnataka", "lat": 16.2120, "lon": 77.3439},
        {"name": "Warangal", "state": "Telangana", "lat": 17.9784, "lon": 79.5941},
    ],
    "wheat": [
        {"name": "Indore", "state": "Madhya Pradesh", "lat": 22.7196, "lon": 75.8577},
        {"name": "Khanna", "state": "Punjab", "lat": 30.7046, "lon": 76.2140},
        {"name": "Hapur", "state": "Uttar Pradesh", "lat": 28.7307, "lon": 77.7787},
    ],
    "ragi": [
        {"name": "Tumkur", "state": "Karnataka", "lat": 13.3379, "lon": 77.1173},
        {"name": "Chitradurga", "state": "Karnataka", "lat": 14.2251, "lon": 76.3980},
        {"name": "Anantapur", "state": "Andhra Pradesh", "lat": 14.6819, "lon": 77.6006},
    ],
    "sugarcane": [
        {"name": "Kolhapur", "state": "Maharashtra", "lat": 16.7050, "lon": 74.2433},
        {"name": "Belgaum", "state": "Karnataka", "lat": 15.8497, "lon": 74.4977},
        {"name": "Meerut", "state": "Uttar Pradesh", "lat": 28.9845, "lon": 77.7064},
    ],
}

# Multilingual weather recommendations
WEATHER_RECOMMENDATIONS = {
    "high_temp": {
        "en": "High temperature alert: Irrigate early morning (5-7 AM) to reduce water loss.",
        "hi": "उच्च तापमान चेतावनी: पानी की बर्बादी कम करने के लिए सुबह जल्दी (5-7 बजे) सिंचाई करें।",
        "kn": "ಹೆಚ್ಚಿನ ತಾಪಮಾನ ಎಚ್ಚರಿಕೆ: ನೀರಿನ ನಷ್ಟ ಕಡಿಮೆ ಮಾಡಲು ಬೆಳಿಗ್ಗೆ (5-7 ಗಂಟೆ) ನೀರಾವರಿ ಮಾಡಿ.",
        "te": "అధిక ఉష్ణోగ్రత హెచ్చరిక: నీటి నష్టాన్ని తగ్గించడానికి ఉదయం (5-7 గంటలు) నీరు పెట్టండి.",
        "ta": "அதிக வெப்பநிலை எச்சரிக்கை: நீர் இழப்பைக் குறைக்க காலை (5-7 மணி) நீர்ப்பாசனம் செய்யுங்கள்.",
        "pa": "ਉੱਚ ਤਾਪਮਾਨ ਚੇਤਾਵਨੀ: ਪਾਣੀ ਦੀ ਬਰਬਾਦੀ ਘੱਟ ਕਰਨ ਲਈ ਸਵੇਰੇ (5-7 ਵਜੇ) ਸਿੰਚਾਈ ਕਰੋ।",
        "gu": "ઉચ્ચ તાપમાન ચેતવણી: પાણીનું નુકસાન ઘટાડવા સવારે (5-7 વાગ્યે) સિંચાઈ કરો.",
        "mr": "उच्च तापमान इशारा: पाण्याची हानी कमी करण्यासाठी सकाळी (5-7 वाजता) सिंचन करा.",
        "bn": "উচ্চ তাপমাত্রা সতর্কতা: জলের অপচয় কমাতে সকালে (৫-৭টা) সেচ দিন।",
        "or": "ଉଚ୍ଚ ତାପମାତ୍ରା ସତର୍କତା: ପାଣି ନଷ୍ଟ କମ୍ କରିବା ପାଇଁ ସକାଳୁ (୫-୭ ଟା) ଜଳ ସେଚନ କରନ୍ତୁ।",
    },
    "rain_expected": {
        "en": "Rain expected: Delay fungicide spray. Ensure field drainage.",
        "hi": "बारिश की संभावना: कवकनाशी छिड़काव में देरी करें। खेत की निकासी सुनिश्चित करें।",
        "kn": "ಮಳೆ ನಿರೀಕ್ಷಿತ: ಶಿಲೀಂಧ್ರನಾಶಕ ಸಿಂಪಡಣೆ ಮುಂದೂಡಿ. ಹೊಲದ ಒಳಚರಂಡಿ ಖಚಿತಪಡಿಸಿ.",
        "te": "వర్షం ఆశించబడుతోంది: శిలీంధ్రనాశిని స్ప్రే ఆపివేయండి. పొలం డ్రైనేజీ నిర్ధారించండి.",
        "ta": "மழை எதிர்பார்க்கப்படுகிறது: பூஞ்சைக்கொல்லி தெளிப்பை தாமதப்படுத்துங்கள். வயல் வடிகால் உறுதி செய்யுங்கள்.",
        "pa": "ਬਾਰਿਸ਼ ਦੀ ਸੰਭਾਵਨਾ: ਉੱਲੀਨਾਸ਼ਕ ਸਪਰੇਅ ਵਿੱਚ ਦੇਰੀ ਕਰੋ। ਖੇਤ ਦੀ ਨਿਕਾਸੀ ਯਕੀਨੀ ਬਣਾਓ।",
        "gu": "વરસાદની અપેક્ષા: ફૂગનાશક છંટકાવમાં વિલંબ કરો. ખેતરનું ડ્રેનેજ સુનિશ્ચિત કરો.",
        "mr": "पाऊस अपेक्षित: बुरशीनाशक फवारणी थांबवा. शेतातील निचरा सुनिश्चित करा.",
        "bn": "বৃষ্টি প্রত্যাশিত: ছত্রাকনাশক স্প্রে বিলম্বিত করুন। ক্ষেতের নিকাশি নিশ্চিত করুন।",
        "or": "ବର୍ଷା ଆଶା: ଫଙ୍ଗିସାଇଡ୍ ସ୍ପ୍ରେ ବିଳମ୍ବ କରନ୍ତୁ। କ୍ଷେତର ନିଷ୍କାସନ ନିଶ୍ଚିତ କରନ୍ତୁ।",
    },
    "market_positive": {
        "en": "Market trend positive. Consider selling at {mandi} mandi.",
        "hi": "बाजार का रुझान सकारात्मक। {mandi} मंडी में बेचने पर विचार करें।",
        "kn": "ಮಾರುಕಟ್ಟೆ ಧನಾತ್ಮಕ. {mandi} ಮಂಡಿಯಲ್ಲಿ ಮಾರಾಟ ಮಾಡಿ.",
        "te": "మార్కెట్ ధోరణి సానుకూలం. {mandi} మండిలో అమ్మకానికి ఆలోచించండి.",
        "ta": "சந்தை போக்கு நேர்மறை. {mandi} மண்டியில் விற்க பரிசீலியுங்கள்.",
        "pa": "ਬਜ਼ਾਰ ਦਾ ਰੁਝਾਨ ਸਕਾਰਾਤਮਕ। {mandi} ਮੰਡੀ ਵਿੱਚ ਵੇਚਣ 'ਤੇ ਵਿਚਾਰ ਕਰੋ।",
        "gu": "બજારનો ટ્રેન્ડ સકારાત્મક. {mandi} માર્કેટમાં વેચવાનું વિચારો.",
        "mr": "बाजार कल सकारात्मक. {mandi} मंडीत विक्री करण्याचा विचार करा.",
        "bn": "বাজারের প্রবণতা ইতিবাচক। {mandi} মান্ডিতে বিক্রি বিবেচনা করুন।",
        "or": "ବଜାର ଧାରା ସକାରାତ୍ମକ। {mandi} ମଣ୍ଡିରେ ବିକ୍ରୟ ବିଚାର କରନ୍ତୁ।",
    },
    "consult_expert": {
        "en": "Consult local agricultural officer for treatment guidance.",
        "hi": "उपचार मार्गदर्शन के लिए स्थानीय कृषि अधिकारी से परामर्श करें।",
        "kn": "ಚಿಕಿತ್ಸೆ ಮಾರ್ಗದರ್ಶನಕ್ಕಾಗಿ ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "te": "చికిత్స మార్గదర్శకత్వం కోసం స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి.",
        "ta": "சிகிச்சை வழிகாட்டுதலுக்கு உள்ளூர் வேளாண் அதிகாரியை அணுகுங்கள்.",
        "pa": "ਇਲਾਜ ਮਾਰਗਦਰਸ਼ਨ ਲਈ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਅਧਿਕਾਰੀ ਨਾਲ ਸਲਾਹ ਕਰੋ।",
        "gu": "સારવાર માર્ગદર્શન માટે સ્થાનિક કૃષિ અધિકારીનો સંપર્ક કરો.",
        "mr": "उपचार मार्गदर्शनासाठी स्थानिक कृषी अधिकाऱ्याशी संपर्क साधा.",
        "bn": "চিকিত্সা নির্দেশনার জন্য স্থানীয় কৃষি কর্মকর্তার সাথে পরামর্শ করুন।",
        "or": "ଚିକିତ୍ସା ମାର୍ଗଦର୍ଶନ ପାଇଁ ସ୍ଥାନୀୟ କୃଷି ଅଧିକାରୀଙ୍କ ସହ ପରାମର୍ଶ କରନ୍ତୁ।",
    },
}

# Disease-specific agronomic recommendations
DISEASE_AGRONOMY = {
    "maize_blight": {
        "severity": "high",
        "yield_loss_range": "30-50%",
        "actions": [
            "Remove and destroy infected plant debris",
            "Apply systemic fungicide (Mancozeb 2.5g/L)",
            "Improve field drainage immediately",
            "Avoid overhead irrigation",
        ],
        "prevention": "Use resistant varieties like HQPM-1, DHM-117",
    },
    "maize_common_rust": {
        "severity": "medium",
        "yield_loss_range": "15-30%",
        "actions": [
            "Spray Propiconazole 25% EC at 1ml/L",
            "Ensure proper plant spacing for air flow",
            "Scout fields every 3-4 days",
        ],
        "prevention": "Early planting to avoid peak rust season",
    },
    "tomato_late_blight": {
        "severity": "critical",
        "yield_loss_range": "70-100%",
        "actions": [
            "URGENT: Apply Metalaxyl + Mancozeb immediately",
            "Remove ALL infected plants and burn them",
            "Do NOT compost infected material",
            "Increase plant spacing",
        ],
        "prevention": "Use certified disease-free transplants",
    },
    "rice_bacterial_leaf_blight": {
        "severity": "high",
        "yield_loss_range": "20-50%",
        "actions": [
            "Drain excess water from field",
            "Apply Streptomycin sulfate 100ppm",
            "Reduce nitrogen fertilizer",
            "Spray Copper oxychloride 0.3%",
        ],
        "prevention": "Use BLB-resistant varieties like Improved Samba Mahsuri",
    },
    "wheat_leaf_rust": {
        "severity": "high",
        "yield_loss_range": "30-40%",
        "actions": [
            "Apply Propiconazole 25% EC at 0.1%",
            "Two sprays at 15-day interval",
            "Rogue out early infected plants",
        ],
        "prevention": "Grow rust-resistant varieties like HD-3086, PBW-725",
    },
    "ragi_blast": {
        "severity": "high",
        "yield_loss_range": "25-50%",
        "actions": [
            "Spray Tricyclazole 75% WP at 0.6g/L",
            "Avoid excess nitrogen application",
            "Maintain field sanitation",
        ],
        "prevention": "Seed treatment with Carbendazim 50% WP",
    },
    "sugarcane_red_rot": {
        "severity": "critical",
        "yield_loss_range": "50-100%",
        "actions": [
            "Remove and burn infected stalks immediately",
            "Do NOT use infected cane for seed",
            "Treat healthy setts with Carbendazim",
            "Allow field to fallow for one season",
        ],
        "prevention": "Use disease-free seed material from certified nurseries",
    },
}

# Multilingual response templates
RESPONSE_TEMPLATES = {
    "en": {
        "weather": "Current weather at your location: {temp}°C, Rainfall: {rain}mm",
        "market": "Market price for {crop}: ₹{price}/quintal at {mandi} mandi",
        "msp": "Government MSP for {crop}: ₹{msp}/quintal",
        "severity": "Disease severity: {severity} (Potential yield loss: {loss})",
    },
    "hi": {
        "weather": "आपके स्थान का मौसम: {temp}°C, वर्षा: {rain}mm",
        "market": "{crop} का बाजार भाव: ₹{price}/क्विंटल ({mandi} मंडी में)",
        "msp": "{crop} का सरकारी MSP: ₹{msp}/क्विंटल",
        "severity": "रोग की गंभीरता: {severity} (संभावित उपज हानि: {loss})",
    },
    "kn": {
        "weather": "ನಿಮ್ಮ ಸ್ಥಳದ ಹವಾಮಾನ: {temp}°C, ಮಳೆ: {rain}mm",
        "market": "{crop} ಮಾರುಕಟ್ಟೆ ಬೆಲೆ: ₹{price}/ಕ್ವಿಂಟಲ್ ({mandi} ಮಂಡಿ)",
        "msp": "{crop} ಸರ್ಕಾರದ MSP: ₹{msp}/ಕ್ವಿಂಟಲ್",
        "severity": "ರೋಗದ ತೀವ್ರತೆ: {severity} (ಸಂಭಾವ್ಯ ಇಳುವರಿ ನಷ್ಟ: {loss})",
    },
    "te": {
        "weather": "మీ ప్రాంతపు వాతావరణం: {temp}°C, వర్షపాతం: {rain}mm",
        "market": "{crop} మార్కెట్ ధర: ₹{price}/క్వింటాల్ ({mandi} మండి)",
        "msp": "{crop} ప్రభుత్వ MSP: ₹{msp}/క్వింటాల్",
        "severity": "వ్యాధి తీవ్రత: {severity} (సంభావ్య దిగుబడి నష్టం: {loss})",
    },
    "ta": {
        "weather": "உங்கள் இடத்தின் வானிலை: {temp}°C, மழை: {rain}mm",
        "market": "{crop} சந்தை விலை: ₹{price}/குவிண்டால் ({mandi} மண்டி)",
        "msp": "{crop} அரசு MSP: ₹{msp}/குவிண்டால்",
        "severity": "நோய் தீவிரம்: {severity} (சாத்தியமான விளைச்சல் இழப்பு: {loss})",
    },
}


def _fetch_weather(lat: float, lon: float, offline: bool) -> dict[str, Any]:
    """Fetch weather: OpenWeatherMap (rich) → Open-Meteo (free fallback)."""
    cache_key = f"weather:{lat:.2f}:{lon:.2f}"

    if offline:
        cached = CACHE.get(cache_key, ttl_seconds=86400)
        if cached:
            cached["source"] = "offline_cache"
            return cached
        return {"temp_c": 28, "rain_mm": 0.0, "humidity": 65, "source": "offline_default"}

    # ── Try OpenWeatherMap first (better data) ──
    owm_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if owm_key:
        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather?"
                f"lat={lat}&lon={lon}&appid={owm_key}&units=metric"
            )
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            data = response.json()

            main = data.get("main", {})
            wind = data.get("wind", {})
            weather_desc = data.get("weather", [{}])[0].get("description", "")
            rain = data.get("rain", {}).get("1h", 0.0)

            result = {
                "temp_c": round(main.get("temp", 28), 1),
                "feels_like": round(main.get("feels_like", 28), 1),
                "rain_mm": round(rain, 1),
                "humidity": main.get("humidity", 65),
                "wind_speed": round(wind.get("speed", 5.0), 1),
                "weather_desc": weather_desc.title(),
                "forecast_3day_rain": 0,
                "source": "openweathermap",
                "timestamp": datetime.now().isoformat(),
            }
            CACHE.set(cache_key, result)
            logger.info(f"OpenWeatherMap: {result['temp_c']}°C, {weather_desc}, humidity={result['humidity']}%")
            return result
        except Exception as e:
            logger.warning(f"OpenWeatherMap failed: {e}, trying Open-Meteo")

    # ── Fallback: Open-Meteo (free, no key) ──
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"
            f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
            f"&timezone=auto&forecast_days=3"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        result = {
            "temp_c": current.get("temperature_2m", 28),
            "rain_mm": current.get("precipitation", 0.0),
            "humidity": current.get("relative_humidity_2m", 65),
            "wind_speed": current.get("wind_speed_10m", 5.0),
            "forecast_3day_rain": sum(daily.get("precipitation_sum", [0, 0, 0])[:3]),
            "source": "open-meteo",
            "timestamp": datetime.now().isoformat(),
        }

        CACHE.set(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"Weather API failed: {e}")
        cached = CACHE.get(cache_key, ttl_seconds=86400)
        if cached:
            cached["source"] = "cache_fallback"
            return cached
        return {"temp_c": 28, "rain_mm": 0.0, "humidity": 65, "source": "api_error"}


def _find_nearest_mandi(crop: str, lat: float, lon: float) -> dict[str, Any]:
    """Find nearest mandi for the crop."""
    mandis = CROP_MANDIS.get(crop, CROP_MANDIS.get("maize", []))

    if not mandis:
        return {"name": "Local Market", "state": "Unknown", "distance_km": 0}

    # Simple distance calculation (Haversine would be more accurate)
    def distance(m):
        return ((m["lat"] - lat) ** 2 + (m["lon"] - lon) ** 2) ** 0.5 * 111  # Rough km

    nearest = min(mandis, key=distance)
    return {
        "name": nearest["name"],
        "state": nearest["state"],
        "distance_km": round(distance(nearest), 1),
    }


def _get_market_price(crop: str, lat: float, lon: float, offline: bool) -> dict[str, Any]:
    """Get market price for crop with nearest mandi info."""
    cache_key = f"market:{crop}:{lat:.1f}:{lon:.1f}"

    if offline:
        cached = CACHE.get(cache_key, ttl_seconds=21600)  # 6h cache
        if cached:
            cached["source"] = "offline_cache"
            return cached

    mandi = _find_nearest_mandi(crop, lat, lon)
    msp = MSP_PRICES_2025.get(crop, 2000)

    # Simulate market variation around MSP (+/- 15%)
    # In production, this would call eNAM API
    import random
    variation = random.uniform(0.90, 1.15)
    modal_price = int(msp * variation)

    result = {
        "crop": crop,
        "modal_price": modal_price,
        "msp": msp,
        "mandi": mandi["name"],
        "state": mandi["state"],
        "distance_km": mandi["distance_km"],
        "price_trend": "stable" if abs(variation - 1.0) < 0.05 else ("up" if variation > 1.0 else "down"),
        "source": "market_model",
        "timestamp": datetime.now().isoformat(),
    }

    CACHE.set(cache_key, result)
    return result


def _get_disease_agronomy(disease_label: str) -> dict[str, Any]:
    """Get agronomic recommendations for disease."""
    base_agronomy = DISEASE_AGRONOMY.get(disease_label, {})

    if not base_agronomy:
        # Default recommendations for unknown/healthy
        if "healthy" in disease_label:
            return {
                "severity": "none",
                "yield_loss_range": "0%",
                "actions": [
                    "Continue regular monitoring",
                    "Maintain current irrigation schedule",
                    "Apply balanced NPK as per soil test",
                ],
                "prevention": "Regular scouting every 5-7 days",
            }
        return {
            "severity": "unknown",
            "yield_loss_range": "varies",
            "actions": ["Consult local agricultural officer"],
            "prevention": "Follow IPM practices",
        }

    return base_agronomy


def _build_recommendations(
    disease_label: str,
    crop: str,
    weather: dict,
    market: dict,
    lang: str = "en",
) -> list[str]:
    """Build comprehensive recommendations based on all inputs."""
    recommendations = []
    agronomy = _get_disease_agronomy(disease_label)

    # Default to English if language not found
    if lang not in WEATHER_RECOMMENDATIONS.get("high_temp", {}):
        lang = "en"

    # Weather-based recommendations
    temp = weather.get("temp_c", 28)
    rain = weather.get("rain_mm", 0)
    forecast_rain = weather.get("forecast_3day_rain", 0)

    if temp > 35:
        recommendations.append(WEATHER_RECOMMENDATIONS["high_temp"].get(lang, WEATHER_RECOMMENDATIONS["high_temp"]["en"]))

    if rain > 10 or forecast_rain > 20:
        recommendations.append(WEATHER_RECOMMENDATIONS["rain_expected"].get(lang, WEATHER_RECOMMENDATIONS["rain_expected"]["en"]))

    # Disease-specific actions from agronomy database
    if agronomy.get("actions"):
        for action in agronomy["actions"][:3]:
            if "consult" not in action.lower():
                recommendations.append(action)

    # Market recommendation
    price_trend = market.get("price_trend", "stable")
    if price_trend == "up":
        mandi = market.get("mandi", "local")
        market_msg = WEATHER_RECOMMENDATIONS["market_positive"].get(lang, WEATHER_RECOMMENDATIONS["market_positive"]["en"])
        recommendations.append(market_msg.format(mandi=mandi))

    return recommendations


def run_knowledge(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Fetch weather, market, agronomic knowledge + RAG.

    Input state keys:
        - lat, lon: Location coordinates (default: Hyderabad)
        - crop_type: Detected crop from vision
        - disease_prediction: Disease info from vision
        - offline: Use cached data if True
        - lang: Language code
        - user_text: Farmer's symptom description (for RAG)

    Output state keys:
        - knowledge: Dict with weather, market, agronomy, rag_context
        - recommendations: List of actionable recommendations
        - status: "knowledge_complete"
    """
    offline = bool(state.get("offline", False))
    lat = float(state.get("lat", 17.3850))  # Default: Hyderabad
    lon = float(state.get("lon", 78.4867))
    lang = state.get("lang", "en")

    crop = state.get("crop_type", "maize")
    disease = state.get("disease_prediction", {})
    disease_label = disease.get("label", "unknown")
    user_text = state.get("user_text", "")

    # Fetch all knowledge
    weather = _fetch_weather(lat, lon, offline)
    market = _get_market_price(crop, lat, lon, offline)
    agronomy = _get_disease_agronomy(disease_label)

    # ── RAG: Query knowledge base for disease-specific advice ────────
    rag_context = ""
    try:
        import concurrent.futures
        def _run_rag():
            from knowledge_base.build_knowledge_db import rag_query, symptom_search
            if user_text and len(user_text) > 10:
                return symptom_search(user_text, crop=crop, n_results=3)
            else:
                query = f"{crop} {disease_label.replace('_', ' ')}" if disease_label != "unknown" else crop
                return rag_query(query, crop=crop, n_results=3)

        # RAG must complete in 5 seconds or we skip it
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_rag)
            rag_results = future.result(timeout=5)

        if rag_results:
            rag_texts = [r["text"] for r in rag_results]
            rag_context = "\n".join(rag_texts[:2])  # Top 2 results
            logger.info(f"RAG: found {len(rag_results)} relevant docs")

    except Exception as e:
        logger.warning(f"RAG query skipped: {e}")

    # Build recommendations
    recommendations = _build_recommendations(
        disease_label=disease_label,
        crop=crop,
        weather=weather,
        market=market,
        lang=lang,
    )

    return {
        **state,
        "knowledge": {
            "weather": weather,
            "market": market,
            "agronomy": agronomy,
            "rag_context": rag_context,
        },
        "recommendations": recommendations,
        "status": "knowledge_complete",
    }


# Export
__all__ = ["run_knowledge", "MSP_PRICES_2025", "DISEASE_AGRONOMY"]
