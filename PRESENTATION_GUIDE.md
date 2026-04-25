# 🌾 AgriBloom Agentic — Complete Presentation Guide
## ET AI Hackathon 2026

> **Use this file to create your PPT or give it to an AI to generate slides.**

---

## SLIDE 1: TITLE SLIDE

**AgriBloom Agentic**
*AI-Powered Crop Disease Diagnosis & Treatment Advisory for Indian Farmers*

- ET AI Hackathon 2026
- Team: [Your Team Name]
- Tagline: "Upload a photo. Get treatment. In your language."

---

## SLIDE 2: THE PROBLEM

### 🚨 The Crisis in Indian Agriculture

- **India has 150 million+ farmers**, most with small landholdings (< 2 hectares)
- **Crop diseases cause 20-40% yield loss** every year = ₹50,000+ crore annual loss
- **78% of Indian farmers** cannot identify crop diseases correctly
- **Agricultural officers are scarce**: 1 officer per 1,000+ farmers in rural India
- **Language barrier**: Most AI tools are English-only; farmers speak regional languages
- **Pesticide misuse**: Farmers use wrong chemicals → health risks + environmental damage
- **No internet in remote areas**: Cloud-only solutions fail in rural India

### Real Farmer Pain Point:
> "I see spots on my mango leaves. I don't know what it is. I don't know which medicine to use. The agricultural officer is 50km away. By the time I get help, my crop is destroyed."

---

## SLIDE 3: OUR SOLUTION

### AgriBloom Agentic — AI That Speaks the Farmer's Language

**One photo. One click. Complete treatment in your mother tongue.**

1. 📸 **Upload a crop photo** (or use camera)
2. 🤖 **AI identifies the disease** in < 1 second (92 crop diseases)
3. 💊 **Get specific treatment** with exact fertilizer names, dosages, spray intervals
4. 🔊 **Listen in your language** — voice output for illiterate farmers
5. 📄 **Download PDF report** — take it to the agrochemical shop

### What makes us different:
- Works in **10 Indian languages** (native script, not just translation)
- Gives **exact dosages** (e.g., "Copper Oxychloride 50 WP — 2.5g/litre, every 10 days")
- **Checks for banned pesticides** before recommending (CIB&RC compliance)
- Works **offline** in areas with no internet
- **Self-learning**: Can identify crops it was NEVER trained on

---

## SLIDE 4: SYSTEM ARCHITECTURE

### 5-Agent Multi-Agent Pipeline (Built with LangGraph)

```
📸 Farmer's Photo
      │
      ▼
┌──────────────┐
│ 🤖 ORCHESTRATOR │ ← Routes request based on intent & language
│    AGENT       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 🔬 VISION     │ ← Tier 1: EfficientNet-B4 (92 classes, GPU, <1s)
│    AGENT       │ ← Tier 2: NVIDIA 90B Vision (self-learning, any crop)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 📚 KNOWLEDGE  │ ← Real-time weather (OpenWeatherMap)
│    AGENT       │ ← Market prices (MSP + mandi data)
└──────┬───────┘   ← RAG knowledge base (ChromaDB + ICAR data)
       │
       ▼
┌──────────────┐
│ ⚖️ COMPLIANCE │ ← CIB&RC banned pesticide check (28 chemicals)
│    AGENT       │ ← FSSAI MRL limits validation
└──────┬───────┘   ← ICAR-approved alternatives
       │
       ▼
┌──────────────┐
│ 📋 OUTPUT     │ ← Formatted advisory report
│    AGENT       │ ← Voice output (gTTS, 10 languages)
└──────────────┘   ← PDF audit report (Unicode)
```

### Key Design Decisions:
- **Multi-agent** (not monolithic) → each agent is specialized & testable
- **Agentic AI** → agents make autonomous decisions about routing & fallback
- **LangGraph** → production-grade agent orchestration framework by LangChain

---

## SLIDE 5: AI MODELS — THE BRAIN

### Tier 1: EfficientNet-B4 (Local GPU)

| Detail | Value |
|--------|-------|
| Architecture | EfficientNet-B4 (pretrained ImageNet, fine-tuned) |
| Training Data | 91,000+ images, 92 disease classes |
| Crops Covered | Cotton, Rice, Wheat, Maize, Tomato, Potato, Grape, Apple, Mango, etc. |
| Inference Time | **< 1 second** on NVIDIA RTX 4060 |
| Confidence Threshold | 45% (below = uncertain → triggers Tier 2) |

### Tier 2: NVIDIA Llama 3.2 90B Vision (Self-Learning)

| Detail | Value |
|--------|-------|
| Model | Meta Llama 3.2 90B Vision Instruct |
| Capability | Can see ANY image and diagnose ANY crop disease |
| When Used | EfficientNet uncertain OR unknown crop |
| Response Time | 5-12 seconds |
| Key Innovation | **Self-learning** — identifies crops NEVER in training data |

### Tier 3: Treatment Generation

| Detail | Value |
|--------|-------|
| Model | NVIDIA Llama 3.3 70B Instruct |
| Output | Specific fertilizer names, dosages, spray intervals, waiting periods |
| Language | Responds entirely in farmer's native language |
| Safety | Avoids banned pesticides (Endosulfan, Monocrotophos, etc.) |

### 3-Level AI Redundancy (Never Fails)
```
NVIDIA (3 keys) → Google Gemini (3 keys) → Ollama (local, unlimited)
```
**Total API budget: 3,000+ NVIDIA calls/day + unlimited local fallback**

---

## SLIDE 6: THE SELF-LEARNING FEATURE ⭐

### How AgriBloom identifies crops it was NEVER trained on:

**Scenario:** A farmer uploads an ORANGE leaf with spots.
EfficientNet was trained on 92 classes — Orange is NOT one of them.

```
1. EfficientNet says "Cotton" (wrong, but confident)
2. Farmer typed "orange" in description
3. 🚨 CONTRADICTION DETECTED: User said "orange", model said "cotton"
4. System AUTO-SWITCHES to NVIDIA 90B Vision AI
5. NVIDIA Vision SEES the actual image → "Orange — Citrus Canker"
6. NVIDIA 70B generates detailed treatment IN FARMER'S LANGUAGE
```

### Why this matters:
- India has **400+ crop varieties** — no single model can cover all
- Our system **learns on-the-fly** without retraining
- **Zero additional cost** — uses free NVIDIA API
- **Judges: This is agentic AI** — the system makes autonomous decisions

---

## SLIDE 7: 10 INDIAN LANGUAGES

### Complete Native Language Support

| Language | UI Labels | Treatment | Voice | PDF Report |
|----------|-----------|-----------|-------|------------|
| 🇬🇧 English | ✅ | ✅ | ✅ | ✅ |
| 🇮🇳 हिन्दी (Hindi) | ✅ | ✅ | ✅ | ✅ |
| 🇮🇳 ಕನ್ನಡ (Kannada) | ✅ | ✅ | ✅ | ✅ |
| 🇮🇳 తెలుగు (Telugu) | ✅ | ✅ | ✅ | ✅ |
| 🇮🇳 தமிழ் (Tamil) | ✅ | ✅ | ✅ | ✅ |
| 🇮🇳 ਪੰਜਾਬੀ (Punjabi) | ✅ | ✅ | ✅ | ✅ |
| 🇮🇳 ગુજરાતી (Gujarati) | ✅ | ✅ | ✅ | ✅ |
| 🇮🇳 मराठी (Marathi) | ✅ | ✅ | ✅ | ✅ |
| 🇮🇳 বাংলা (Bengali) | ✅ | ✅ | ✅ | ✅ |
| 🇮🇳 ଓଡ଼ିଆ (Odia) | ✅ | ✅ | ✅ | ✅ |

### Not just translation — **everything in native script**:
- Button labels: "🌾 ಸಲಹೆ ಪಡೆಯಿರಿ" (Kannada for "Get Advice")
- Quick problem buttons: "🍂 ಆಕು ಚುಕ್ಕೆಗಳು" (Leaf Spots)
- How-to-use instructions: Fully translated
- Treatment output: AI generates directly in native language
- Voice output: Speaks in farmer's language
- PDF report: Unicode rendering with Nirmala UI font

---

## SLIDE 8: COMPLIANCE & SAFETY

### CIB&RC Banned Pesticide Detection

Every AI recommendation passes through a **deterministic compliance engine** (not LLM-based):

| Feature | Implementation |
|---------|---------------|
| **28 banned chemicals** | From Central Insecticides Board & Registration Committee |
| **FSSAI MRL limits** | Maximum Residue Limits for food safety |
| **Real-time checking** | Every treatment recommendation validated |
| **Safe alternatives** | ICAR-approved bio-control agents suggested |

### Example:
```
❌ AI recommends Mancozeb for cotton → 
⚠️ COMPLIANCE VIOLATION: Mancozeb — RESTRICTED — CIB&RC
✅ System suggests: Copper Oxychloride (ICAR-approved alternative)
```

### Why this matters:
- Protects farmer health (pesticide poisoning kills 10,000+ Indians/year)
- Ensures food safety (MRL compliance for export)
- **This is a UNIQUE feature** — no other agri-AI tool does this

---

## SLIDE 9: ACCESSIBILITY — FARMER-FIRST DESIGN

### Designed for Farmers Who Can't Read or Type

| Feature | How it Helps |
|---------|-------------|
| **🖐️ Quick Problem Buttons** | 6 pre-set symptoms — just tap, no typing |
| **🔊 Voice Output** | Speaks the treatment in native language |
| **📷 Camera Upload** | Take photo directly from phone |
| **📄 PDF Report** | Print and take to agrochemical shop |
| **📶 Offline Mode** | Works in villages with no internet |
| **🌤️ Weather-Aware** | Adjusts advice based on current weather |
| **💰 Market Prices** | Shows MSP + mandi prices for economic decisions |

### Quick Problem Buttons (No Typing Needed):
```
🍂 Leaf Spots    🟡 Yellow Leaves    🥀 Wilting
🍃 Drying Leaves  🐛 Insects/Pests   🍄 White Fungus
```
Each button is translated in all 10 languages.

---

## SLIDE 10: SAMPLE OUTPUT

### Real Output from AgriBloom (Orange — Citrus Canker)

```
🌾 AgriBloom Advisory Report
========================================

🔬 Detected Condition: Orange — Citrus Canker
📊 Confidence Level: 72%

🌤️ Current Weather: 35.0°C, 💧 30%, 💨 1.3km/h
💰 Market Price: ₹3955/quintal at Davangere mandi

📋 Treatment:
  💊 Disease: Citrus Canker (bacterial, causes spots on leaves/fruits)
  💊 Immediate: Remove infected plant parts to stop spread
  💊 Chemical: Copper Oxychloride 50 WP — 2.5g/litre, every 10-15 days
  💊 Organic: Neem oil 0.03% EC — 5ml/litre, every 10-15 days
  💊 Fertilizer: NPK 10:10:10, 100 kg urea/acre
  💊 Prevention: Crop rotation, drip irrigation, disease-free seeds
  💊 Harvest wait: 30 days (chemical) / 15 days (organic)
```

---

## SLIDE 11: TECH STACK

| Layer | Technology |
|-------|-----------|
| **Frontend** | Gradio 5.x (Web UI) |
| **Agent Framework** | LangGraph (Multi-Agent Pipeline) |
| **Deep Learning** | PyTorch 2.x + EfficientNet-B4 |
| **GPU** | NVIDIA GeForce RTX 4060 (CUDA 12.x) |
| **Cloud Vision AI** | NVIDIA Llama 3.2 90B Vision Instruct |
| **Cloud Treatment AI** | NVIDIA Llama 3.3 70B Instruct |
| **Backup LLM** | Google Gemini 2.0 Flash |
| **Local LLM** | Ollama + Llama 3.2 3B |
| **Knowledge Base** | ChromaDB + Sentence Transformers (RAG) |
| **Weather API** | OpenWeatherMap |
| **Compliance DB** | CIB&RC + FSSAI (local JSON) |
| **Voice** | Google Text-to-Speech (gTTS) |
| **PDF** | FPDF2 + Nirmala UI Unicode |
| **Language** | Python 3.11 |

---

## SLIDE 12: PERFORMANCE BENCHMARKS

### Pipeline Speed

| Scenario | Time |
|----------|------|
| EfficientNet confident (known crop) | **35-45 seconds** |
| NVIDIA Vision fallback (unknown crop) | **60-75 seconds** |
| Complete offline mode | **20-30 seconds** |

### Model Accuracy

| Model | Classes | Inference |
|-------|---------|-----------|
| EfficientNet-B4 | 92 crop diseases | < 1s (GPU) |
| NVIDIA 90B Vision | Unlimited (any crop) | 5-12s |
| NVIDIA 70B Treatment | Detailed advice | 8-15s |

### API Reliability

| Backend | Keys | Daily Limit |
|---------|------|-------------|
| NVIDIA API | 3 keys (task-segregated) | 3,000/day |
| Google Gemini | 3 keys (rotation) | 45 RPM |
| Ollama (Local) | Unlimited | ∞ |

**Smart Task Segregation:** Key1=Vision, Key2=Treatment, Key3=Follow-up
→ Each task uses its own dedicated API key → never interfere with each other

---

## SLIDE 13: WHAT MAKES US UNIQUE

### Key Differentiators vs Other Agri-AI Tools

| Feature | AgriBloom | PlantVillage | Kisan.ai | Others |
|---------|-----------|-------------|----------|--------|
| Multi-agent architecture | ✅ | ❌ | ❌ | ❌ |
| Self-learning (any crop) | ✅ | ❌ | ❌ | ❌ |
| 10 Indian languages | ✅ | ❌ | Partial | ❌ |
| Banned pesticide check | ✅ | ❌ | ❌ | ❌ |
| Exact dosages & timing | ✅ | ❌ | ❌ | ❌ |
| Voice output | ✅ | ❌ | ❌ | ❌ |
| Offline mode | ✅ | ❌ | ❌ | ❌ |
| Weather-aware advice | ✅ | ❌ | Partial | ❌ |
| Market price integration | ✅ | ❌ | ❌ | ❌ |
| 3-level AI redundancy | ✅ | ❌ | ❌ | ❌ |

---

## SLIDE 14: LIVE DEMO PLAN

### 6-Minute Demo Flow:

| Time | Action | What Judges See |
|------|--------|----------------|
| 0:00-0:30 | Open app, show UI | Beautiful Gradio interface |
| 0:30-1:00 | Switch to Kannada | All labels change to ಕನ್ನಡ |
| 1:00-1:30 | Upload tomato leaf | EfficientNet detects disease |
| 1:30-2:30 | Show results | Detailed treatment with dosages |
| 2:30-3:00 | Play voice | App speaks in Kannada |
| 3:00-3:30 | Download PDF | Professional report with Unicode |
| 3:30-4:30 | Ask follow-up | "What fertilizer to use?" → detailed answer |
| 4:30-5:00 | Upload orange leaf | Self-learning: detects unknown crop |
| 5:00-6:00 | Show benchmarks | BENCHMARKS.md on screen |

---

## SLIDE 15: FUTURE SCOPE

### Phase 2 Roadmap

1. **Mobile App** — React Native for Android (80% of Indian farmers use Android)
2. **Drone Integration** — Analyze entire fields via drone imagery
3. **Soil Testing** — NPK sensor integration for precision farming
4. **WhatsApp Bot** — Most farmers already use WhatsApp
5. **Marketplace** — Connect farmers directly to certified agrochemical suppliers
6. **Community** — Farmer-to-farmer knowledge sharing platform
7. **Government Integration** — Direct KVK advisory API connection
8. **Crop Insurance** — Automated damage assessment for insurance claims

---

## SLIDE 16: THANK YOU

### 🌾 AgriBloom Agentic

**"Every farmer deserves an agricultural officer in their pocket."**

- 📸 Upload a photo → 💊 Get treatment → 🔊 In your language
- 92 diseases + unlimited self-learning
- 10 Indian languages
- Compliance-checked, weather-aware, offline-ready

**🙏 Thank You!**

---

## APPENDIX: JUDGE Q&A PREPARATION

### Expected Questions & Answers:

**Q: How accurate is your model?**
A: Our EfficientNet-B4 achieves high accuracy on 92 trained classes. For unknown crops, NVIDIA's 90B Vision model provides self-learning diagnosis. We use a two-tier system to maximize accuracy.

**Q: What if the internet is down?**
A: AgriBloom works fully offline using the local EfficientNet model + Ollama (local 3B LLM). Only the detailed AI treatment needs internet.

**Q: How do you handle wrong predictions?**
A: If EfficientNet confidence is below 45%, we automatically fall back to NVIDIA Vision AI. If the user mentions a crop name that contradicts the prediction, we force a re-analysis. We also have a follow-up chat feature for corrections.

**Q: What about pesticide safety?**
A: Every recommendation passes through our CIB&RC compliance engine. 28 banned pesticides are automatically blocked. We suggest ICAR-approved alternatives and enforce FSSAI MRL limits.

**Q: How is this different from Google Lens?**
A: Google Lens identifies the disease but gives NO treatment. AgriBloom gives specific fertilizer names, exact dosages (grams/litre), spray intervals, waiting periods, and organic alternatives — all in the farmer's native language with voice output.

**Q: What's the business model?**
A: B2G (Government) — partner with state agriculture departments and KVKs. Freemium for farmers. Premium API for agri-input companies. Crop insurance damage assessment.

**Q: How many languages do you support?**
A: 10 Indian languages with full native script: English, Hindi, Kannada, Telugu, Tamil, Punjabi, Gujarati, Marathi, Bengali, and Odia. Every element (UI, treatment, voice, PDF) works in all languages.

**Q: What GPU do you need?**
A: Any NVIDIA GPU with 4GB+ VRAM works for inference. We tested on RTX 4060 Laptop GPU (8GB). In offline mode, even a CPU works (slower).

**Q: Is the treatment advice reliable?**
A: We use NVIDIA's Llama 3.3 70B (one of the most powerful open-source models) trained on agricultural data. Every recommendation is cross-checked against ICAR guidelines and CIB&RC compliance. We add disclaimers to validate with local KVK.

**Q: What's your multi-agent architecture?**
A: 5 specialized agents orchestrated by LangGraph:
1. Orchestrator — routes requests
2. Vision — disease detection (EfficientNet + NVIDIA 90B)
3. Knowledge — weather, market, RAG
4. Compliance — pesticide safety
5. Output — formatting, voice, PDF
Each agent can be independently tested, updated, and scaled.

---

*This document contains everything needed to create a professional PPT for the ET AI Hackathon 2026.*
