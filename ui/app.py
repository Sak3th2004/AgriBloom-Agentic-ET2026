"""
AgriBloom Agentic - Premium Farmer-Friendly Gradio UI
Clean, Simple, Multilingual — Every feature working
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import gradio as gr
from PIL import Image

logger = logging.getLogger(__name__)

# ── Language Configuration ───────────────────────────────────────────────────
LANGUAGE_MAP = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Telugu (తెలుగు)": "te",
    "Tamil (தமிழ்)": "ta",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Gujarati (ગુજરાતી)": "gu",
    "Marathi (मराठी)": "mr",
    "Bengali (বাংলা)": "bn",
    "Odia (ଓଡ଼ିଆ)": "or",
}

# UI labels in each language
UI_LABELS = {
    "en": {
        "title": "🌾 AgriBloom — AI Advisory for Farmers",
        "subtitle": "Upload crop photo → Get disease diagnosis + treatment in your language",
        "upload": "📷 Take Photo or Upload Crop Leaf",
        "describe": "📝 Describe Problem (Optional)",
        "placeholder": "Example: My tomato leaves have brown spots...",
        "language": "🌐 Your Language",
        "state": "State",
        "district": "District",
        "offline": "📶 Offline Mode (No Internet)",
        "submit": "🌾 GET ADVICE",
        "result": "🌾 Diagnosis & Treatment",
        "listen": "🔊 Listen to Advice",
        "report": "📄 Download Report",
        "followup": "💬 Ask Follow-up Question",
        "followup_placeholder": "Ask anything about the treatment...",
        "followup_btn": "Ask",
        "calendar": "📅 Crop Calendar",
        "status_ready": "✅ Ready! Upload a crop photo to get advice.",
    },
    "hi": {
        "title": "🌾 एग्रीब्लूम — किसानों के लिए AI सलाहकार",
        "subtitle": "फसल की फोटो अपलोड करें → अपनी भाषा में रोग निदान + उपचार पाएं",
        "upload": "📷 फोटो लें या अपलोड करें",
        "describe": "📝 समस्या बताएं (वैकल्पिक)",
        "placeholder": "उदाहरण: मेरे टमाटर की पत्तियां पीली हो रही हैं...",
        "language": "🌐 आपकी भाषा",
        "state": "राज्य",
        "district": "जिला",
        "offline": "📶 ऑफलाइन मोड (बिना इंटरनेट)",
        "submit": "🌾 सलाह लें",
        "result": "🌾 रोग और उपचार",
        "listen": "🔊 सलाह सुनें",
        "report": "📄 रिपोर्ट डाउनलोड",
        "followup": "💬 और पूछें",
        "followup_placeholder": "उपचार के बारे में कुछ भी पूछें...",
        "followup_btn": "पूछें",
        "calendar": "📅 फसल कैलेंडर",
        "status_ready": "✅ तैयार! सलाह पाने के लिए फसल की फोटो अपलोड करें।",
    },
    "kn": {
        "title": "🌾 ಅಗ್ರಿಬ್ಲೂಮ್ — ರೈತರಿಗೆ AI ಸಲಹೆಗಾರ",
        "subtitle": "ಬೆಳೆ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ → ನಿಮ್ಮ ಭಾಷೆಯಲ್ಲಿ ರೋಗ ನಿರ್ಣಯ + ಚಿಕಿತ್ಸೆ",
        "upload": "📷 ಫೋಟೋ ತೆಗೆಯಿರಿ ಅಥವಾ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "describe": "📝 ಸಮಸ್ಯೆ ವಿವರಿಸಿ (ಐಚ್ಛಿಕ)",
        "placeholder": "ಉದಾ: ನನ್ನ ಟೊಮ್ಯಾಟೊ ಎಲೆಗಳು ಹಳದಿ ಆಗುತ್ತಿವೆ...",
        "language": "🌐 ನಿಮ್ಮ ಭಾಷೆ",
        "state": "ರಾಜ್ಯ",
        "district": "ಜಿಲ್ಲೆ",
        "offline": "📶 ಆಫ್‌ಲೈನ್ ಮೋಡ್",
        "submit": "🌾 ಸಲಹೆ ಪಡೆಯಿರಿ",
        "result": "🌾 ರೋಗ ಮತ್ತು ಚಿಕಿತ್ಸೆ",
        "listen": "🔊 ಸಲಹೆ ಕೇಳಿ",
        "report": "📄 ವರದಿ ಡೌನ್‌ಲೋಡ್",
        "followup": "💬 ಇನ್ನಷ್ಟು ಕೇಳಿ",
        "followup_placeholder": "ಚಿಕಿತ್ಸೆ ಬಗ್ಗೆ ಏನಾದರೂ ಕೇಳಿ...",
        "followup_btn": "ಕೇಳಿ",
        "calendar": "📅 ಬೆಳೆ ಕ್ಯಾಲೆಂಡರ್",
        "status_ready": "✅ ಸಿದ್ಧ! ಸಲಹೆ ಪಡೆಯಲು ಬೆಳೆ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
    },
    "te": {
        "title": "🌾 అగ్రిబ్లూమ్ — రైతులకు AI సలహాదారు",
        "subtitle": "పంట ఫోటో అప్‌లోడ్ → మీ భాషలో వ్యాధి నిర్ధారణ + చికిత్స",
        "upload": "📷 ఫోటో తీయండి లేదా అప్‌లోడ్ చేయండి",
        "describe": "📝 సమస్య వివరించండి",
        "placeholder": "ఉదా: నా టమాటా ఆకులు పసుపు రంగులోకి మారుతున్నాయి...",
        "language": "🌐 మీ భాష", "state": "రాష్ట్రం", "district": "జిల్లా",
        "offline": "📶 ఆఫ్‌లైన్ మోడ్", "submit": "🌾 సలహా పొందండి",
        "result": "🌾 వ్యాధి & చికిత్స", "listen": "🔊 సలహా వినండి",
        "report": "📄 రిపోర్ట్", "followup": "💬 మరింత అడగండి",
        "followup_placeholder": "చికిత్స గురించి అడగండి...", "followup_btn": "అడగండి",
        "calendar": "📅 పంట క్యాలెండర్",
        "status_ready": "✅ సిద్ధం! సలహా కోసం పంట ఫోటో అప్‌లోడ్ చేయండి.",
    },
    "ta": {
        "title": "🌾 அக்ரிப்ளூம் — விவசாயிகளுக்கு AI ஆலோசகர்",
        "subtitle": "பயிர் புகைப்படம் பதிவேற்றம் → உங்கள் மொழியில் நோய் கண்டறிதல் + சிகிச்சை",
        "upload": "📷 புகைப்படம் எடுக்கவும்",
        "describe": "📝 பிரச்சனையை விவரிக்கவும்",
        "placeholder": "எ.கா: என் தக்காளி இலைகள் மஞ்சள் நிறமாக...",
        "language": "🌐 உங்கள் மொழி", "state": "மாநிலம்", "district": "மாவட்டம்",
        "offline": "📶 ஆஃப்லைன்", "submit": "🌾 ஆலோசனை பெறுங்கள்",
        "result": "🌾 நோய் & சிகிச்சை", "listen": "🔊 ஆலோசனை கேளுங்கள்",
        "report": "📄 அறிக்கை", "followup": "💬 மேலும் கேளுங்கள்",
        "followup_placeholder": "சிகிச்சை பற்றி கேளுங்கள்...", "followup_btn": "கேளுங்கள்",
        "calendar": "📅 பயிர் காலண்டர்",
        "status_ready": "✅ தயார்! ஆலோசனைக்கு பயிர் புகைப்படம் பதிவேற்றவும்.",
    },
}

# Fall back to English for languages without full UI translation
for code in ["pa", "gu", "mr", "bn", "or"]:
    UI_LABELS[code] = UI_LABELS["en"].copy()

# ── Location Data ────────────────────────────────────────────────────────────
INDIAN_LOCATIONS = {
    "Karnataka": ["Bengaluru", "Mysuru", "Davangere", "Belgaum", "Hubli", "Mangaluru", "Tumkur", "Shimoga"],
    "Andhra Pradesh": ["Vijayawada", "Guntur", "Visakhapatnam", "Tirupati", "Kurnool", "Nellore"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Thanjavur"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Kolhapur"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Allahabad"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur"],
}

DISTRICT_COORDS = {
    "Bengaluru": (12.97, 77.59), "Mysuru": (12.30, 76.64), "Davangere": (14.46, 75.92),
    "Belgaum": (15.85, 74.50), "Hubli": (15.36, 75.12), "Mangaluru": (12.91, 74.86),
    "Vijayawada": (16.51, 80.65), "Guntur": (16.31, 80.44), "Visakhapatnam": (17.69, 83.22),
    "Hyderabad": (17.39, 78.49), "Warangal": (17.97, 79.59), "Chennai": (13.08, 80.27),
    "Coimbatore": (11.02, 76.96), "Madurai": (9.93, 78.12), "Pune": (18.52, 73.86),
    "Nagpur": (21.15, 79.09), "Ahmedabad": (23.02, 72.57), "Surat": (21.17, 72.83),
    "Ludhiana": (30.90, 75.86), "Amritsar": (31.63, 74.87), "Jaipur": (26.91, 75.79),
    "Lucknow": (26.85, 80.95), "Patna": (25.59, 85.14), "Kolkata": (22.57, 88.36),
    "Bhubaneswar": (20.30, 85.82), "Kochi": (9.93, 76.27),
}


def _get_coords(district: str) -> tuple:
    return DISTRICT_COORDS.get(district, (14.46, 75.92))


def _get_labels(lang_code: str) -> dict:
    return UI_LABELS.get(lang_code, UI_LABELS["en"])


# ── Seasonal Warning ────────────────────────────────────────────────────────
def _get_seasonal_info(lang_code: str) -> str:
    """Get current seasonal crop advisory."""
    try:
        from utils.crop_calendar import get_current_season, CROP_CALENDAR
        season = get_current_season()
        season_name = {"kharif": "Kharif (Monsoon)", "rabi": "Rabi (Winter)", "spring": "Spring"}.get(season, season)

        crops_this_season = []
        for crop, seasons in CROP_CALENDAR.items():
            if season in seasons:
                info = seasons[season]
                risks = ", ".join(info.get("diseases_peak", [])[:2])
                crops_this_season.append(f"**{crop.title()}**: Sow {info['sowing']} | ⚠️ {risks}")

        if crops_this_season:
            header = f"### 📅 Current Season: {season_name}\n"
            return header + "\n".join([f"- {c}" for c in crops_this_season[:6]])
        return ""
    except Exception:
        return ""


def _get_farmer_instructions(lang_code: str) -> str:
    """Get farmer instructions in their language."""
    instructions = {
        "en": """**📸 How to take a good photo:**
1. 🌿 Pick the **most affected leaf** from your crop
2. ☀️ Take photo in **daylight** (not at night)
3. 📏 Hold phone **close to the leaf** (20-30 cm)
4. 🎯 Make sure the **disease spots are visible**
5. 📱 You can take a **live photo** with camera OR **upload** from gallery

**✅ Good:** Close-up of leaf, disease visible, good light
**❌ Bad:** Full field view, blurry, too dark, not a plant

**🔄 Steps:** Select language → Take photo → Select state → Click GET ADVICE → Listen 🔊""",

        "hi": """**📸 अच्छी फोटो कैसे लें:**
1. 🌿 अपनी फसल से **सबसे ज्यादा प्रभावित पत्ती** चुनें
2. ☀️ **दिन की रोशनी** में फोटो लें (रात में नहीं)
3. 📏 फोन को **पत्ती के पास** रखें (20-30 सेमी)
4. 🎯 **रोग के दाग** साफ दिखाई दें
5. 📱 **कैमरे से तुरंत** फोटो लें या **गैलरी से** अपलोड करें

**✅ सही:** पत्ती का क्लोज-अप, रोग दिखे, अच्छी रोशनी
**❌ गलत:** पूरा खेत, धुंधली, बहुत अंधेरी

**🔄 तरीका:** भाषा चुनें → फोटो लें → राज्य चुनें → सलाह लें बटन दबाएं → सुनें 🔊""",

        "kn": """**📸 ಉತ್ತಮ ಫೋಟೋ ಹೇಗೆ ತೆಗೆಯುವುದು:**
1. 🌿 ನಿಮ್ಮ ಬೆಳೆಯಿಂದ **ಹೆಚ್ಚು ಹಾನಿಗೊಳಗಾದ ಎಲೆ** ಆಯ್ಕೆ ಮಾಡಿ
2. ☀️ **ಹಗಲು ಬೆಳಕಿನಲ್ಲಿ** ಫೋಟೋ ತೆಗೆಯಿರಿ
3. 📏 ಫೋನ್ ಅನ್ನು **ಎಲೆಗೆ ಹತ್ತಿರ** ಇರಿಸಿ (20-30 ಸೆಂ.ಮೀ)
4. 🎯 **ರೋಗದ ಚಿಹ್ನೆಗಳು** ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣಲಿ
5. 📱 **ಕ್ಯಾಮೆರಾ** ಅಥವಾ **ಗ್ಯಾಲರಿ** ಯಿಂದ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ

**🔄 ಹಂತಗಳು:** ಭಾಷೆ ಆಯ್ಕೆ → ಫೋಟೋ → ರಾಜ್ಯ → ಸಲಹೆ ಪಡೆಯಿರಿ → ಕೇಳಿ 🔊""",

        "te": """**📸 మంచి ఫోటో ఎలా తీయాలి:**
1. 🌿 మీ పంట నుండి **ఎక్కువగా ప్రభావితమైన ఆకు** ఎంచుకోండి
2. ☀️ **పగటి వెలుతురులో** ఫోటో తీయండి
3. 📏 ఫోన్‌ను **ఆకుకు దగ్గరగా** పట్టుకోండి
4. 🎯 **వ్యాధి మచ్చలు** స్పష్టంగా కనిపించాలి

**🔄 స్టెప్స్:** భాష ఎంచుకోండి → ఫోటో → రాష్ట్రం → సలహా పొందండి → వినండి 🔊""",

        "ta": """**📸 நல்ல புகைப்படம் எடுப்பது எப்படி:**
1. 🌿 உங்கள் பயிரிலிருந்து **மிகவும் பாதிக்கப்பட்ட இலையை** தேர்ந்தெடுக்கவும்
2. ☀️ **பகல் வெளிச்சத்தில்** புகைப்படம் எடுக்கவும்
3. 📏 ஃபோனை **இலைக்கு அருகில்** பிடிக்கவும்
4. 🎯 **நோய் புள்ளிகள்** தெளிவாக தெரிய வேண்டும்

**🔄 படிகள்:** மொழி → புகைப்படம் → மாநிலம் → ஆலோசனை → கேளுங்கள் 🔊""",
    }
    return instructions.get(lang_code, instructions["en"])


# ── Main UI ──────────────────────────────────────────────────────────────────
def launch_app(run_pipeline: Callable[..., dict[str, Any]]) -> None:
    """Launch the AgriBloom Gradio app."""

    # Conversation state
    conversation_history = []

    custom_css = """
    .main-header {
        background: linear-gradient(135deg, #052e16 0%, #166534 50%, #14532d 100%);
        border-radius: 16px; padding: 32px; margin-bottom: 16px;
        text-align: center; color: #ffffff !important; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .main-header h1 { font-size: 2.2em; margin: 0; font-weight: 700; color: #ffffff !important; }
    .main-header p { opacity: 0.9; margin-top: 8px; font-size: 1.1em; color: #e0f2e0 !important; }
    .main-header span { color: #ffffff !important; }
    .feature-badges { display: flex; gap: 8px; justify-content: center; margin-top: 16px; flex-wrap: wrap; }
    .feature-badges span {
        background: rgba(255,255,255,0.15); padding: 4px 14px; border-radius: 20px;
        font-size: 0.85em; backdrop-filter: blur(4px);
    }
    .green-card {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1px solid #86efac; border-radius: 12px; padding: 16px;
    }
    .submit-btn {
        background: linear-gradient(135deg, #166534, #14532d) !important;
        color: white !important; font-weight: 700 !important; font-size: 1.1em !important;
        padding: 14px !important; border-radius: 10px !important; border: none !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .submit-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(22,101,52,0.4) !important;
    }
    .footer {
        text-align: center; padding: 12px; margin-top: 16px;
        background: #f0fdf4; border-radius: 12px; color: #166534; font-size: 0.9em;
    }
    """

    with gr.Blocks(
        title="AgriBloom Agentic — AI for Indian Farmers",
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald", neutral_hue="slate"),
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────
        header_html = gr.HTML("""
        <div class="main-header">
            <h1>🌾 AgriBloom Agentic</h1>
            <p>AI Advisory for Indian Farmers — Upload crop photo, get treatment in your language</p>
            <div class="feature-badges">
                <span>🔬 92-Class Disease AI</span>
                <span>🛡️ CIB&RC Compliance</span>
                <span>🌍 10 Languages</span>
                <span>📶 Offline Ready</span>
                <span>🧠 Gemini GenAI</span>
                <span>📚 ICAR Knowledge</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ── LEFT: Inputs ─────────────────────────────────────────────
            with gr.Column(scale=5):
                # Language selector — TOP, prominent
                language = gr.Dropdown(
                    choices=list(LANGUAGE_MAP.keys()),
                    value="English",
                    label="🌐 Select Your Language / अपनी भाषा चुनें / ನಿಮ್ಮ ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ",
                    interactive=True,
                )

                image_input = gr.Image(
                    type="pil",
                    label="📷 Take Photo or Upload Crop Leaf",
                    height=260,
                    sources=["upload", "webcam"],
                )

                text_input = gr.Textbox(
                    lines=2,
                    label="📝 Describe Problem (Optional)",
                    placeholder="Example: My tomato leaves have brown spots...",
                )

                with gr.Accordion("🎤 Speak Your Problem / बोलकर बताएं / ಮಾತನಾಡಿ ಹೇಳಿ", open=False):
                    voice_input = gr.Audio(
                        label="🎤 Record Voice (speak in your language)",
                        type="filepath",
                        format="wav",
                        sources=["microphone"],
                    )
                    transcribe_btn = gr.Button("🎤 Transcribe Voice → Text", size="sm")


                with gr.Row():
                    state_input = gr.Dropdown(
                        choices=list(INDIAN_LOCATIONS.keys()),
                        value="Karnataka", label="📍 State",
                    )
                    district_input = gr.Dropdown(
                        choices=INDIAN_LOCATIONS["Karnataka"],
                        value="Davangere", label="📍 District",
                    )

                # Hidden textbox to receive JS geolocation result
                geo_result = gr.Textbox(visible=False, elem_id="geo_result")
                detect_btn = gr.Button("📍 Auto-Detect My Location", size="sm", variant="secondary")
                location_status = gr.Markdown("", elem_id="location_status")

                offline_mode = gr.Checkbox(value=False, label="📶 Offline Mode (No Internet)")

                # Farmer Instructions
                with gr.Accordion("📖 How to Use / कैसे इस्तेमाल करें / ಹೇಗೆ ಬಳಸುವುದು", open=False):
                    farmer_instructions = gr.Markdown("""
**📸 How to take a good photo:**
1. 🌿 Pick the **most affected leaf** from your crop
2. ☀️ Take photo in **daylight** (not at night)
3. 📏 Hold phone **close to the leaf** (20-30 cm)
4. 🎯 Make sure the **disease spots are visible**
5. 📱 You can take a **live photo** with camera OR **upload** from gallery

**✅ Good photos:** Close-up of leaf, disease visible, good light
**❌ Bad photos:** Full field view, blurry, too dark, not a plant

**🔄 Steps:**
1. Select your **language** above
2. **Take photo** of diseased leaf (camera or upload)
3. Select your **state and district**
4. Click **GET ADVICE** button
5. **Listen** to the voice advice 🔊
""")

                submit_btn = gr.Button(
                    "🌾 GET ADVICE",
                    variant="primary", size="lg",
                    elem_classes=["submit-btn"],
                )

            # ── RIGHT: Results ───────────────────────────────────────────
            with gr.Column(scale=5):
                response_output = gr.Textbox(
                    lines=14,
                    label="🌾 Diagnosis & Treatment",
                )

                voice_output = gr.Audio(label="🔊 Listen to Advice", type="filepath")

                with gr.Accordion("💬 Ask Follow-up Question", open=False):
                    followup_input = gr.Textbox(
                        lines=1, label="Your question",
                        placeholder="Ask anything about the treatment...",
                    )
                    followup_btn = gr.Button("Ask", size="sm")
                    followup_output = gr.Textbox(lines=4, label="Answer", interactive=False)

                with gr.Row():
                    bloom_plot = gr.Plot(label="📊 Confidence")
                    audit_file = gr.File(label="📄 Report PDF")

                status_output = gr.Markdown("✅ *Ready! Upload a crop photo to get advice.*")

        # ── Extra Feature Tabs ───────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("🧪 Fertilizer Calculator", open=False):
                    gr.Markdown("*Calculate exact fertilizer for your crop and area*")
                    fert_crop = gr.Dropdown(
                        choices=["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Tomato",
                                 "Potato", "Ragi", "Groundnut", "Soybean", "Pepper", "Apple", "Grape"],
                        value="Rice", label="Crop",
                    )
                    fert_area = gr.Number(value=1.0, label="Area (acres)", minimum=0.1, maximum=100)
                    fert_soil = gr.Radio(
                        choices=["low_fertility", "medium", "high_fertility"],
                        value="medium", label="Soil Type",
                    )
                    fert_btn = gr.Button("🧪 Calculate", size="sm")
                    fert_output = gr.Markdown("")

            with gr.Column(scale=1):
                with gr.Accordion("📅 Current Season Advisory", open=False):
                    seasonal_info = gr.Markdown(_get_seasonal_info("en"))

                with gr.Accordion("📞 Helplines & Nearest KVK", open=False):
                    helpline_output = gr.Markdown(
                        "## 📞 Kisan Call Center: `1800-180-1551` (Free, 24x7)\n"
                        "## 🏛️ PM-KISAN: `155261`\n"
                        "## 🌾 Crop Insurance: `1800-200-7710`\n\n"
                        "*Select your state above to find nearest KVK*"
                    )

        # ── Footer ───────────────────────────────────────────────────────
        gr.HTML("""
        <div class="footer">
            <b>🌾 AgriBloom Agentic</b> — ET AI Hackathon 2026 |
            92-Class Disease AI • CIB&RC Compliance • 10 Languages • Offline Ready |
            Helpline: 1800-180-1551
        </div>
        """)

        # ── Hidden state ─────────────────────────────────────────────────
        chat_state = gr.State(value=[])  # conversation history

        # ── Event Handlers ───────────────────────────────────────────────
        def update_districts(state):
            districts = INDIAN_LOCATIONS.get(state, ["Select"])
            return gr.Dropdown(choices=districts, value=districts[0])

        def update_ui_labels(language_name):
            """Update UI labels when language changes."""
            lang_code = LANGUAGE_MAP.get(language_name, "en")
            labels = _get_labels(lang_code)
            seasonal = _get_seasonal_info(lang_code)
            instructions = _get_farmer_instructions(lang_code)
            return (
                gr.update(label=labels["upload"]),
                gr.update(label=labels["describe"], placeholder=labels["placeholder"]),
                gr.update(label=labels["result"]),
                gr.update(label=labels["listen"]),
                gr.update(value=labels["status_ready"]),
                seasonal,
                instructions,
            )

        def process_query(image, text_value, language_name, state, district, offline, history):
            """Run the full agent pipeline."""
            lang_code = LANGUAGE_MAP.get(language_name, "en")
            lat, lon = _get_coords(district)

            try:
                import concurrent.futures

                def _run():
                    return run_pipeline(
                        image=image,
                        image_path="",
                        user_text=text_value or "",
                        user_language=lang_code,
                        lang=lang_code,
                        offline=bool(offline),
                        lat=float(lat),
                        lon=float(lon),
                    )

                # 120s timeout — pipeline must complete in 120 seconds
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run)
                    result = future.result(timeout=120)

                conf = result.get("disease_prediction", {}).get("confidence", 0)
                source = result.get("disease_prediction", {}).get("source", "")
                crop = result.get("crop_type", "unknown")
                tier = "🧠 Gemini Vision" if "gemini" in source else "🔬 EfficientNet-B4"

                status = (
                    f"✅ **Analysis Complete!** | Confidence: {conf*100:.0f}% | "
                    f"Crop: {crop.title()} | Engine: {tier}"
                )

                # Update conversation history
                new_history = list(history) if history else []
                new_history.append({"role": "system", "content": result.get("final_response", "")})

                return (
                    result.get("final_response", "No response generated."),
                    result.get("voice_output_path"),
                    result.get("bloom_figure"),
                    result.get("audit_pdf_path"),
                    status,
                    new_history,
                )
            except concurrent.futures.TimeoutError:
                logger.error("Pipeline timed out after 120 seconds")
                return (
                    "⏱️ Analysis is taking too long (exceeded 2 minutes). Please try again or enable Offline Mode.",
                    None, None, None,
                    "❌ **Timeout** — Try Offline Mode or retry",
                    history or [],
                )
            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
                return (
                    f"Error: {str(e)}",
                    None, None, None,
                    f"❌ **Error**: {str(e)}",
                    history or [],
                )

        def handle_followup(question, language_name, history):
            """Handle follow-up questions using Gemini."""
            if not question or not question.strip():
                return "Please type a question."

            lang_code = LANGUAGE_MAP.get(language_name, "en")
            try:
                from utils.genai_handler import conversational_followup, is_genai_available
                if not is_genai_available():
                    return "GenAI not available. Please check internet connection."

                # Extract context from history
                crop = ""
                disease = ""
                if history:
                    last = history[-1].get("content", "")
                    crop = last[:50]

                answer = conversational_followup(
                    question=question,
                    conversation_history=history or [],
                    crop=crop,
                    disease=disease,
                    language=lang_code,
                )
                return answer
            except Exception as e:
                return f"Error: {str(e)}"

        # ── Connect Events ───────────────────────────────────────────────
        state_input.change(fn=update_districts, inputs=[state_input], outputs=[district_input])

        language.change(
            fn=update_ui_labels,
            inputs=[language],
            outputs=[image_input, text_input, response_output, voice_output, status_output, seasonal_info, farmer_instructions],
        )

        # ── Geolocation Detection ─────────────────────────────────────────
        # ── Geolocation Detection ─────────────────────────────────────────
        def detect_location_from_coords(geo_text):
            """Reverse geocode lat/lon to nearest Indian state/district."""
            try:
                source = "GPS"
                if not geo_text or geo_text.strip() == "0,0" or "," not in geo_text:
                    # Python-side IP Geolocation fallback (works flawlessly for local host)
                    import urllib.request, json
                    try:
                        req = urllib.request.Request("http://ip-api.com/json/", headers={"User-Agent": "AgriBloom/1.0"})
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            ip_data = json.loads(resp.read().decode())
                            lat = float(ip_data.get("lat", 0))
                            lon = float(ip_data.get("lon", 0))
                            source = "IP"
                    except Exception as e:
                        logger.error(f"IP Geo failed: {e}")
                        return gr.update(), gr.update(), gr.update(), "❌ Could not detect location. Please select manually."
                else:
                    parts = geo_text.strip().split(",")
                    lat, lon = float(parts[0]), float(parts[1])
                    if len(parts) > 2 and parts[2] == "ip":
                        source = "IP"

                if abs(lat) < 0.1 and abs(lon) < 0.1:
                    return gr.update(), gr.update(), gr.update(), "❌ Location access denied. Please select manually."

                # Reverse geocode using Nominatim (free, no API key)
                import urllib.request, json
                url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&addressdetails=1"
                req = urllib.request.Request(url, headers={"User-Agent": "AgriBloom/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())

                address = data.get("address", {})
                state = address.get("state", "")
                district = address.get("state_district", "") or address.get("county", "")

                # Match to our location list
                matched_state = None
                for s in INDIAN_LOCATIONS:
                    if s.lower() in state.lower() or state.lower() in s.lower():
                        matched_state = s
                        break

                if not matched_state:
                    return gr.update(), gr.update(), gr.update(), f"📍 Detected: {state}, {district} (not in list — select manually)"

                matched_district = INDIAN_LOCATIONS[matched_state][0]
                for d in INDIAN_LOCATIONS[matched_state]:
                    if d.lower() in district.lower() or district.lower() in d.lower():
                        matched_district = d
                        break

                return (
                    gr.update(value=matched_state),
                    gr.update(choices=INDIAN_LOCATIONS[matched_state], value=matched_district),
                    gr.update(),
                    f"✅ **Location detected ({source}):** {matched_state} → {matched_district}",
                )
            except Exception as e:
                logger.error(f"Geolocation failed: {e}")
                return gr.update(), gr.update(), gr.update(), f"❌ Location detection failed: {str(e)[:50]}"

        # JS: Try GPS first, if it fails, send 0,0 to let Python handle IP-based geolocation
        geo_js = """
        async () => {
            try {
                const pos = await new Promise((resolve, reject) => {
                    if (!navigator.geolocation) { reject('no_geo'); return; }
                    navigator.geolocation.getCurrentPosition(resolve, reject, {timeout: 5000});
                });
                return pos.coords.latitude + "," + pos.coords.longitude;
            } catch(e) {
                // Return 0,0 so Python backend uses its reliable IP-based geolocation
                return "0,0";
            }
        }
        """

        detect_btn.click(
            fn=None, inputs=None, outputs=[geo_result],
            js=geo_js,
        ).then(
            fn=detect_location_from_coords,
            inputs=[geo_result],
            outputs=[state_input, district_input, geo_result, location_status],
        )

        submit_btn.click(
            fn=process_query,
            inputs=[image_input, text_input, language, state_input, district_input, offline_mode, chat_state],
            outputs=[response_output, voice_output, bloom_plot, audit_file, status_output, chat_state],
            show_progress="full",
        )

        followup_btn.click(
            fn=handle_followup,
            inputs=[followup_input, language, chat_state],
            outputs=[followup_output],
        )

        # Voice transcription
        def transcribe_audio(audio_path, language_name):
            """Transcribe farmer's voice — uses Google Speech API (free, no key needed)."""
            if audio_path is None:
                return ""

            def _do_transcribe():
                lang_code = LANGUAGE_MAP.get(language_name, "en")
                lang_map = {"en": "en-IN", "hi": "hi-IN", "kn": "kn-IN",
                            "te": "te-IN", "ta": "ta-IN", "pa": "pa-IN",
                            "gu": "gu-IN", "mr": "mr-IN", "bn": "bn-IN", "or": "or-IN"}
                speech_lang = lang_map.get(lang_code, "en-IN")

                import speech_recognition as sr
                recognizer = sr.Recognizer()
                recognizer.energy_threshold = 300  # Adjust for noisy environments

                # Open the WAV audio file directly (Gradio format='wav' ensures this)
                with sr.AudioFile(audio_path) as source:
                    audio = recognizer.record(source, duration=30)  # Max 30s

                text = recognizer.recognize_google(audio, language=speech_lang)
                logger.info(f"Voice transcribed: '{text[:80]}' (lang={speech_lang})")
                return text

            # Run with timeout to prevent UI freeze
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_transcribe)
                    return future.result(timeout=15)
            except concurrent.futures.TimeoutError:
                return "⚠️ Transcription timed out. Please try shorter audio."
            except Exception as e:
                err = str(e)
                if "UnknownValueError" in err or "could not understand" in err.lower():
                    return "⚠️ Could not understand. Please speak clearly and try again."
                logger.error(f"Transcription failed: {e}")
                return "⚠️ Could not transcribe. Please type your problem instead."

        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[voice_input, language],
            outputs=[text_input],
        )

        # Fertilizer calculator
        def calc_fertilizer(crop_name, area, soil):
            try:
                from utils.fertilizer_calc import format_fertilizer_card
                return format_fertilizer_card(crop_name.lower(), float(area), soil)
            except Exception as e:
                return f"Error: {str(e)}"

        fert_btn.click(
            fn=calc_fertilizer,
            inputs=[fert_crop, fert_area, fert_soil],
            outputs=[fert_output],
        )

        # Update helpline when state changes
        def update_helpline(state_name, language_name):
            try:
                from utils.helpline import format_helpline_card
                lang_code = LANGUAGE_MAP.get(language_name, "en")
                return format_helpline_card(state_name, "", lang_code)
            except Exception as e:
                return f"📞 Kisan Call Center: `1800-180-1551` (Free, 24x7)"

        state_input.change(
            fn=update_helpline,
            inputs=[state_input, language],
            outputs=[helpline_output],
        )

    # Launch
    logger.info("Launching AgriBloom UI on http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    def mock_pipeline(**kwargs):
        return {"final_response": "Test response", "status": "output_complete"}
    launch_app(mock_pipeline)
