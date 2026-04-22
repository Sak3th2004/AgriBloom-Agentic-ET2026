# 🌾 AgriBloom Agentic — AI Advisory for Indian Farmers

**ET AI Hackathon 2026** | Multi-Agent Agricultural Advisory System

> Upload a crop photo → Get disease diagnosis + treatment in your language

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🔬 **92-Class Disease AI** | EfficientNet-B4 trained on 91,000+ images |
| 🧠 **Gemini Vision Fallback** | Identifies ANY crop not in trained model |
| 🛡️ **CIB&RC Compliance** | 46 banned pesticides blocked automatically |
| 🌍 **10 Indian Languages** | Hindi, Kannada, Telugu, Tamil, Punjabi, Gujarati, Marathi, Bengali, Odia, English |
| 📶 **Offline Ready** | Works without internet using local GPU |
| 📚 **ICAR Knowledge Base** | RAG-powered disease advisories from ChromaDB |
| 🧪 **Fertilizer Calculator** | ICAR NPK recommendations with cost estimate |
| 📞 **Helpline & KVK Finder** | Emergency contacts + nearest advisory center |
| 📅 **Crop Calendar** | Seasonal disease warnings and sowing schedules |
| 🎙️ **Voice Output** | Text-to-speech in farmer's language |
| 📄 **PDF Reports** | Professional advisory reports for record |

## 🌾 Crops Supported

**Tier 1 (Trained — 92 classes):** Cotton, Rice, Wheat, Maize, Sugarcane, Ragi, Tomato, Potato, Pepper, Apple, Grape, Orange, Peach, Cherry, Strawberry, Soybean, Squash, Blueberry, Raspberry

**Tier 2 (Gemini Vision — unlimited):** ANY crop — Pulses, Groundnut, Mustard, Jute, Coconut, Mango, Banana, etc.

## 🏗️ Architecture

```
5-Agent LangGraph Pipeline:
┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐
│ Orchestrator │──▶│   Vision    │──▶│  Knowledge   │──▶│ Compliance │──▶│   Output   │
│  (Router)    │   │ (EfficientNet│   │ (Weather+RAG)│   │ (CIB&RC)   │   │ (Format)   │
└─────────────┘   │ + Gemini)    │   └──────────────┘   └────────────┘   └────────────┘
                  └─────────────┘
```

## 📂 Project Structure

```
AgriBloom-Agentic-ET2026/
├── agents/                    # 5 LangGraph agents
│   ├── orchestrator_agent.py  # Request router + language detection
│   ├── vision_agent.py        # Two-tier disease detection
│   ├── knowledge_agent.py     # Weather + market + RAG
│   ├── compliance_agent.py    # CIB&RC pesticide guardrails
│   └── output_agent.py        # Response formatting + voice
├── compliance/                # Regulatory databases
│   ├── banned_pesticides.json # 46 banned chemicals
│   ├── mrl_limits.json        # FSSAI limits
│   └── safe_alternatives.json # ICAR-approved bio-control
├── knowledge_base/            # RAG knowledge
│   ├── crop_diseases.json     # ICAR disease advisories
│   └── build_knowledge_db.py  # ChromaDB builder
├── models/
│   └── train_model.py         # EfficientNet-B4 training pipeline
├── data/
│   └── prepare_dataset.py     # Dataset unification script
├── utils/                     # Utility modules
│   ├── genai_handler.py       # Gemini API integration
│   ├── image_validator.py     # Crop leaf validation
│   ├── translator.py          # Multilingual translation
│   ├── crop_calendar.py       # Seasonal advisory
│   ├── fertilizer_calc.py     # NPK calculator
│   └── helpline.py            # Emergency contacts
├── ui/
│   └── app.py                 # Gradio web interface
├── tests/
│   └── test_all.py            # 14-test compliance suite
├── docs/                      # Documentation
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
└── .env.example               # Environment template
```

## ⚡ Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Sak3th2004/AgriBloom-Agentic-ET2026.git
cd AgriBloom-Agentic-ET2026
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your Gemini API key

# 3. Install PyTorch with CUDA (for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Run
python main.py
# Open http://localhost:7860
```

## 🏋️ Training

```bash
# Download datasets to data/raw/ then:
python data/prepare_dataset.py
python models/train_model.py --data_dir data/unified --epochs 50 --batch_size 32
```

## 🧪 Tests

```bash
python -m pytest tests/test_all.py -v  # 14/14 passing
```

## 🛡️ Compliance

All recommendations pass through a **deterministic** (non-LLM) compliance engine:
- 46 banned pesticides from CIB&RC are blocked
- FSSAI MRL limits enforced
- Safe ICAR-approved alternatives suggested
- Full audit trail for every check

## 👥 Team

**ET AI Hackathon 2026** — AgriBloom Team

---

*Helpline: Kisan Call Center 1800-180-1551 (Free, 24x7)*
