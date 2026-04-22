# AgriBloom Agentic - System Architecture

## Multi-Agent Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
│                    [Image] + [Text] + [Language] + [Location]                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR AGENT                                   │
│  Role: Request routing, session management, state initialization            │
│  Input: Raw user request                                                     │
│  Output: Initialized state with routing decision                             │
│  Error Handling: Returns safe defaults if input validation fails            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│         VISION AGENT              │   │       KNOWLEDGE AGENT             │
│                                   │   │                                   │
│  Role: Disease detection          │   │  Role: Context enrichment         │
│  Model: ViT (54 classes)          │   │  APIs: Open-Meteo, eNAM           │
│  Offline: ONNX Runtime            │   │  Offline: JSON cache              │
│                                   │   │                                   │
│  Tools:                           │   │  Tools:                           │
│  - PyTorch/ONNX inference         │   │  - Weather API (Open-Meteo)       │
│  - Image preprocessing            │   │  - Market prices (MSP/eNAM)       │
│  - Confidence thresholding        │   │  - Agronomy database              │
│                                   │   │                                   │
│  Error Handling:                  │   │  Error Handling:                  │
│  - No image → skip vision         │   │  - API fail → use cached data     │
│  - Low confidence → uncertain     │   │  - No cache → use defaults        │
│  - Model error → return error     │   │  - Invalid coords → fallback      │
└───────────────────────────────────┘   └───────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLIANCE AGENT                                     │
│  Role: Regulatory guardrails (FSSAI/ICAR)                                   │
│  Checks: Banned pesticides, MRL limits, safety warnings                     │
│  Output: Compliance status + required disclaimers                           │
│                                                                              │
│  Tools:                                                                      │
│  - compliance_rules.json (30+ banned substances)                            │
│  - MRL database                                                              │
│  - Disclaimer templates (10 languages)                                      │
│                                                                              │
│  Error Handling:                                                             │
│  - Unknown substance → allow with warning                                   │
│  - Missing rules → use strictest defaults                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT AGENT                                       │
│  Role: Response formatting, voice synthesis, visualization                  │
│  Languages: EN, HI, KN, TE, TA, PA, GU, MR, BN, OR (10 total)               │
│  Output: Text + Audio + Chart + PDF                                          │
│                                                                              │
│  Tools:                                                                      │
│  - gTTS (text-to-speech)                                                    │
│  - Plotly (Bloom Simulator chart)                                           │
│  - ReportLab (PDF audit report)                                             │
│                                                                              │
│  Error Handling:                                                             │
│  - TTS fail → generate fallback beep audio                                  │
│  - Chart fail → skip visualization                                          │
│  - PDF fail → return text-only response                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FINAL OUTPUT                                    │
│              [Advisory Text] + [Voice Audio] + [Chart] + [PDF]              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Communication (LangGraph State)

All agents communicate through a shared state dictionary:

```python
State = {
    # Input
    "image": PIL.Image,
    "user_text": str,
    "lang": str,              # "kn", "hi", "te", etc.
    "lat": float,
    "lon": float,
    "offline": bool,
    
    # Vision Agent output
    "disease_prediction": {"label": str, "confidence": float},
    "crop_type": str,
    "treatment": str,
    
    # Knowledge Agent output
    "knowledge": {
        "weather": {"temp_c": float, "rain_mm": float},
        "market": {"modal_price": int, "mandi": str},
        "agronomy": {"severity": str, "yield_loss_range": str}
    },
    "recommendations": [str],
    
    # Compliance Agent output
    "compliance": {"allowed": bool, "violations": [], "disclaimers": []},
    
    # Output Agent output
    "final_response": str,
    "voice_output_path": str,
    "bloom_figure": plotly.Figure,
    "audit_pdf_path": str,
    
    # Status
    "status": str
}
```

---

## Tool Integrations

| Agent | Tool | Purpose | Offline Fallback |
|-------|------|---------|------------------|
| Vision | PyTorch ViT | Disease classification | ONNX Runtime |
| Vision | Pillow | Image preprocessing | Same |
| Knowledge | Open-Meteo API | Real-time weather | JSON cache (24hr TTL) |
| Knowledge | eNAM/AgMarknet | Market prices | MSP defaults |
| Compliance | JSON rules file | Banned substances | Built-in |
| Output | gTTS | Voice synthesis | Beep audio |
| Output | Plotly | Health chart | Skip |
| Output | ReportLab | PDF report | Skip |

---

## Error Handling Summary

| Error Type | Handling Strategy |
|------------|-------------------|
| No image uploaded | Skip vision, show "upload photo" message |
| Low confidence (<50%) | Mark as "uncertain_detection", ask for clearer photo |
| Model inference error | Return error state, show generic advice |
| Weather API timeout | Use 24-hour cached data |
| No cached weather | Use regional averages |
| Invalid location | Default to state capital coordinates |
| TTS language unsupported | Fall back to English |
| PDF generation error | Return text response only |
| Banned substance detected | Block recommendation, show compliance alert |

---

## Pipeline Flow

```
START → orchestrator → vision → knowledge → compliance → output → END
             │                                               │
             └──────────── State Passed Through ─────────────┘
```

---

## Performance

| Metric | Value |
|--------|-------|
| Inference time (GPU) | ~2 seconds |
| Inference time (CPU/ONNX) | ~5 seconds |
| Model accuracy | 90.64% |
| Supported languages | 10 |
| Disease classes | 54 |
| Crop types | 12 |
