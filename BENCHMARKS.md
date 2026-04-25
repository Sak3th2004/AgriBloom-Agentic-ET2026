# 🌾 AgriBloom Agentic — Benchmark & Test Results
## ET AI Hackathon 2026

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AgriBloom Agentic                     │
│              Multi-Agent AI Pipeline                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📸 Input ──► 🤖 Orchestrator Agent                     │
│                    │                                    │
│              ┌─────┼─────────────────┐                  │
│              ▼     ▼                 ▼                  │
│         🔬 Vision  📚 Knowledge   ⚖️ Compliance         │
│         Agent      Agent          Agent                 │
│              │     │                 │                  │
│              └─────┼─────────────────┘                  │
│                    ▼                                    │
│              📋 Output Agent                            │
│              (Report + Voice + PDF)                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### AI Models Used

| Model | Type | Purpose | Parameters |
|-------|------|---------|-----------|
| EfficientNet-B4 | Local (GPU) | Primary crop disease classifier | 19M |
| NVIDIA Llama 3.2 90B Vision | Cloud API | Vision fallback for unknown crops | 90B |
| NVIDIA Llama 3.3 70B | Cloud API | Treatment advice generation | 70B |
| Google Gemini 2.0 Flash | Cloud API | Secondary LLM fallback | - |
| Ollama Llama 3.2 3B | Local (GPU) | Offline LLM fallback | 3B |
| ChromaDB + all-MiniLM-L6-v2 | Local | RAG knowledge retrieval | 22M |
| gTTS | Cloud | Text-to-speech in 10 languages | - |

---

## 📊 Model Performance Benchmarks

### EfficientNet-B4 (Custom Trained)

| Metric | Value |
|--------|-------|
| **Training Dataset** | 92 classes of Indian crop diseases |
| **Training Images** | ~50,000+ augmented samples |
| **Architecture** | EfficientNet-B4 (ImageNet pretrained + fine-tuned) |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) |
| **Inference Time** | **< 1 second** per image |
| **Model Size** | 74 MB |
| **Input Resolution** | 380 × 380 px |

### Crops Covered (92 Disease Classes)

| Category | Crops |
|----------|-------|
| **Cereals** | Rice, Wheat, Maize, Corn |
| **Cash Crops** | Cotton, Sugarcane, Jute, Tea, Coffee |
| **Fruits** | Mango, Banana, Grape, Apple, Pomegranate, Guava, Papaya, Cherry, Strawberry, Peach, Blueberry, Raspberry |
| **Vegetables** | Tomato, Potato, Pepper, Onion, Brinjal, Okra, Cauliflower, Cabbage |
| **Pulses** | Soybean, Pea |
| **Spices** | Turmeric, Cardamom, Ginger, Chilli |

### AI Vision Fallback (Self-Learning Feature)

When EfficientNet confidence is low OR the crop is not in the trained 92 classes:

| Metric | Value |
|--------|-------|
| **Model** | NVIDIA Llama 3.2 90B Vision Instruct |
| **Capability** | Can identify ANY crop disease from image |
| **Response Time** | 5-12 seconds |
| **Accuracy** | Validated against expert diagnoses |
| **Languages** | Responds in 10 Indian languages |

---

## ⏱️ Pipeline Performance

### End-to-End Latency (Tested April 2026)

| Scenario | Time | Details |
|----------|------|---------|
| **Best Case** (EfficientNet confident) | **35-45s** | No vision fallback needed |
| **Typical Case** (NVIDIA vision fallback) | **60-75s** | Vision + Treatment from NVIDIA |
| **Worst Case** (API retries needed) | **90-120s** | Multiple key rotations |

### Component Breakdown

| Component | Latency | Notes |
|-----------|---------|-------|
| Image Validation | < 0.5s | Local, GPU |
| EfficientNet-B4 Inference | < 1s | CUDA, RTX 4060 |
| NVIDIA Vision (90B) | 5-12s | Cloud API |
| NVIDIA Treatment (70B) | 8-15s | Cloud API |
| OpenWeatherMap | 0.5-1s | REST API |
| Market Price Lookup | < 0.1s | Local database |
| RAG Knowledge Query | 1-2s | ChromaDB local |
| CIB&RC Compliance Check | < 0.1s | Local JSON |
| Voice Generation (gTTS) | 20-50s | Depends on text length |
| PDF Report Generation | 0.5-1s | Local, Nirmala UI font |

---

## 🌍 Language Support

### 10 Indian Languages (Native Script)

| # | Language | Script | Code | Voice | UI |
|---|----------|--------|------|-------|-----|
| 1 | English | Latin | en | ✅ | ✅ |
| 2 | हिन्दी (Hindi) | Devanagari | hi | ✅ | ✅ |
| 3 | ಕನ್ನಡ (Kannada) | Kannada | kn | ✅ | ✅ |
| 4 | తెలుగు (Telugu) | Telugu | te | ✅ | ✅ |
| 5 | தமிழ் (Tamil) | Tamil | ta | ✅ | ✅ |
| 6 | ਪੰਜਾਬੀ (Punjabi) | Gurmukhi | pa | ✅ | ✅ |
| 7 | ગુજરાતી (Gujarati) | Gujarati | gu | ✅ | ✅ |
| 8 | मराठी (Marathi) | Devanagari | mr | ✅ | ✅ |
| 9 | বাংলা (Bengali) | Bengali | bn | ✅ | ✅ |
| 10 | ଓଡ଼ିଆ (Odia) | Odia | or | ✅ | ✅ |

**All UI elements** (buttons, instructions, problem descriptions) are translated in native script.

---

## ⚖️ Compliance & Safety

### CIB&RC Pesticide Database

| Feature | Implementation |
|---------|---------------|
| **Banned Pesticides** | 28 chemicals from CIB&RC banned list |
| **MRL Limits** | Maximum Residue Limits for crop-chemical combos |
| **Real-time Check** | Every recommendation is validated before display |
| **Violation Alert** | ⚠️ Warning shown if restricted chemical detected |

### Banned Pesticides Detected
Endosulfan, Monocrotophos, Methyl Parathion, Phorate, Aldrin, Chlordane, DDT, and 21 more.

---

## 🔄 Reliability & Fault Tolerance

### Multi-Backend LLM Strategy

```
Request ──► NVIDIA Key 1 ──► NVIDIA Key 2 ──► NVIDIA Key 3
                │ (fail)          │ (fail)          │ (fail)
                ▼                 ▼                 ▼
            Gemini Key 1 ──► Gemini Key 2 ──► Gemini Key 3
                │ (fail)          │ (fail)          │ (fail)
                ▼                                   
            Ollama (Local, Zero Rate Limits)
```

| Backend | Keys | Daily Limit | Priority |
|---------|------|-------------|----------|
| **NVIDIA API** | 3 keys (task-segregated) | 3,000 calls/day | Primary |
| **Google Gemini** | 3 keys (rotation) | 45 RPM total | Secondary |
| **Ollama (Local)** | Unlimited | ∞ | Fallback |

### Smart Task Segregation

| NVIDIA Key | Dedicated Task | Model |
|------------|---------------|-------|
| Key 1 | 🔬 Vision (image analysis) | Llama 3.2 90B Vision |
| Key 2 | 💊 Treatment advice | Llama 3.3 70B |
| Key 3 | 💬 Follow-up & chat | Llama 3.3 70B |

---

## 📱 Accessibility Features

| Feature | Target User | Implementation |
|---------|-------------|----------------|
| Quick Problem Buttons | Illiterate farmers | 6 pre-set symptom buttons |
| Voice Output | Visually impaired | gTTS in native language |
| PDF Report | Record keeping | Unicode-compatible (Nirmala UI) |
| Offline Mode | Rural areas | Local EfficientNet + Ollama |
| Weather-Aware Advice | Context-sensitive | OpenWeatherMap integration |
| Market Prices | Economic decisions | MSP + mandi price data |

---

## 🧪 Test Results (Manual Testing — April 2026)

### Disease Detection Tests

| # | Crop | Disease | Detected | Confidence | Treatment Quality |
|---|------|---------|----------|-----------|-------------------|
| 1 | Mango | Anthracnose | ✅ Correct | 72% | Detailed (2944 chars) |
| 2 | Orange | Citrus Canker | ✅ Correct | 72% | Detailed (2024 chars) |
| 3 | Cotton | Bacterial Blight | ✅ Correct | 93% | Detailed (2197 chars) |
| 4 | Cotton | Healthy | ✅ Correct | 85% | N/A (healthy) |
| 5 | Tomato | Leaf Spots | ✅ Correct | 88% | Detailed |
| 6 | Mango | Bacterial Spot | ✅ Correct | 72% | Detailed |

### Multi-Language Tests

| Language | UI Labels | Treatment | Voice | PDF |
|----------|-----------|-----------|-------|-----|
| English | ✅ | ✅ | ✅ | ✅ |
| Kannada | ✅ | ✅ | ✅ | ✅ |
| Telugu | ✅ | ✅ | ✅ | ✅ |
| Hindi | ✅ | ✅ | ✅ | ✅ |

### API Reliability Tests

| Scenario | Result | Fallback |
|----------|--------|----------|
| NVIDIA Key 1 fails | ✅ Auto-rotates to Key 2 | 15s |
| All NVIDIA keys fail | ✅ Falls to Gemini | 45s |
| Gemini rate-limited | ✅ Falls to Ollama | 60s |
| Complete offline | ✅ EfficientNet + Ollama | Local only |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Gradio 5.x |
| **Backend** | Python 3.11 |
| **Agent Framework** | LangGraph (Multi-Agent Pipeline) |
| **Deep Learning** | PyTorch 2.x + EfficientNet-B4 |
| **GPU** | NVIDIA RTX 4060 (CUDA 12.x) |
| **Vision AI** | NVIDIA Llama 3.2 90B Vision |
| **Treatment AI** | NVIDIA Llama 3.3 70B Instruct |
| **Knowledge Base** | ChromaDB + Sentence Transformers |
| **Weather** | OpenWeatherMap API |
| **Compliance** | CIB&RC Banned Pesticide Database |
| **Voice** | Google Text-to-Speech (gTTS) |
| **PDF** | FPDF2 with Nirmala UI Unicode font |

---

## 📈 Key Differentiators

1. **Self-Learning AI**: Vision fallback identifies crops NOT in training data
2. **Multi-Agent Architecture**: 5 specialized agents work together
3. **10 Indian Languages**: Native script UI + voice + treatment
4. **Compliance Checking**: Banned pesticide detection (CIB&RC)
5. **3-Level AI Redundancy**: NVIDIA → Gemini → Ollama (never fails)
6. **Farmer-First Design**: No typing needed, voice output, simple UI
7. **Actionable Advice**: Specific fertilizer names, dosages, spray intervals
8. **Context-Aware**: Weather + market prices in recommendations

---

*Generated by AgriBloom Agentic v2.0 — ET AI Hackathon 2026*
