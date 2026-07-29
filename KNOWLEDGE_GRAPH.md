# AgriBloom Agentic — Project Knowledge Graph

> **Purpose of this file:** A machine- and human-readable map of the entire codebase so any
> LLM agent (or future me) can understand the system without re-reading every file.
> Nodes = components. Edges = "calls / depends on / produces". Last mapped: 2026-07-28.

---

## 1. One-Line Summary

**AgriBloom Agentic** is a multi-agent agricultural advisory system for Indian farmers. A farmer
uploads a crop-leaf photo (+ optional text, in any of 10 Indian languages); the system detects the
disease, enriches with weather/market/agronomy data, runs a deterministic regulatory-compliance
check, and returns a multilingual text + voice + chart + PDF advisory. Orchestrated as a **5-node
LangGraph pipeline**.

- **Entry point:** `main.py` → builds `GRAPH` → launches Gradio UI (`ui/app.py`)
- **Hackathon:** ET AI Hackathon 2026
- **Current branch:** `feature/provider-router-v2` (this branch is reworking the LLM provider routing)

---

## 2. The Pipeline (core control flow)

```
USER INPUT (image + text + lang + lat/lon + offline)
        │
        ▼
[orchestrator] ──route──┬─ "vision_first"    ─▶ [vision] ─▶ [knowledge]
                        └─ "knowledge_first" ─────────────▶ [knowledge]
                                                                │
                                                                ▼
                                                          [compliance]
                                                                │
                                                                ▼
                                                            [output] ─▶ END
```

- Graph is defined in `main.py:build_graph()`, compiled once at module load into `GRAPH`.
- State is a **plain `dict`** (not TypedDict — deliberately, to avoid Gradio schema parser issues; see `main.py:51`).
- `run_pipeline(...)` in `main.py:116` is the public function the UI calls. It seeds initial state and invokes `GRAPH`.
- Routing function: `main.py:_route_after_orchestrator()` reads `state["route"]`.
  - ⚠️ **Note:** Even for `knowledge_first`, the graph edges go `vision→knowledge→compliance→output`. Knowledge-first simply skips the vision node. `output` is always terminal.

---

## 3. Node Reference (the 5 agents)

Each agent is a function `run_<name>(state: dict) -> dict` that reads some keys and merges new keys via `{**state, ...}`.

### 3.1 `agents/orchestrator_agent.py` → `run_orchestrator`
- **Role:** Router + language/intent/crop detection + session log.
- **Reads:** `user_text`, `user_language`, `image`, `offline`.
- **Writes:** `route` (`vision_first`|`knowledge_first`), `detected_intent`, `detected_crop`, `lang`, `chat_history`, `status="orchestrated"`.
- **Key logic:**
  - `_detect_language()` — Unicode-block script counting (Devanagari→hi, Bengali→bn, Gurmukhi→pa, Gujarati, Odia, Tamil, Telugu, Kannada).
  - `_detect_crop_from_text()` — keyword match against `CROP_KEYWORDS` (multilingual synonyms).
  - `_detect_intent()` — keyword match against `INTENT_KEYWORDS`; image present ⇒ `disease_detection`.
  - `_determine_route()` — image present ⇒ `vision_first`, else `knowledge_first`.
- **Constants exported:** `SUPPORTED_LANGUAGES` (10 langs w/ TTS codes), `CROP_KEYWORDS`.

### 3.2 `agents/vision_agent.py` → `run_vision`  ⭐ (most complex node)
- **Role:** Two-tier crop-disease detection.
- **Reads:** `image` (PIL), `offline`, `lang`, `user_text`.
- **Writes:** `disease_prediction` (dict: label, confidence, source, top3, entropy…), `crop_type`, `treatment`, `status`.
- **Tier 1 — local model** (`_resolve_and_load_engine()`, cached in global `_ENGINE`):
  1. `EfficientNetEngine` — EfficientNet-B4, loads `models/checkpoints/efficientnet_b4_indian/best_model.pth` + `class_labels.json`. Path override via env `AGRIBLOOM_VISION_MODEL_DIR`.
  2. `ViTEngine` — legacy fallback, `models/checkpoints/vit_crop_disease/` or HF `google/vit-base-patch16-224`.
  - **OOD detection** inside `EfficientNetEngine.predict()`: normalized entropy > 0.45 OR top-1/top-2 from different crops ⇒ caps confidence at 0.30 to force fallback.
- **Contradiction check:** if `user_text` names a crop that differs from predicted crop ⇒ `force_fallback=True`, confidence set to 0.20.
- **Tier 2 — AI vision fallback** (triggered when `confidence < GEMINI_FALLBACK_THRESHOLD=0.35` or forced):
  1. Ollama LLaVA / NVIDIA 90B vision (can SEE image) → `genai_handler._ollama_vision_analyze`
  2. Gemini Vision → `genai_handler.analyze_unknown_crop_pil`
  3. Ollama text-only diagnosis → `genai_handler._ollama_generate`
- **Thresholds:** `CONFIDENCE_THRESHOLD=0.45` (below ⇒ `uncertain_detection`), `GEMINI_FALLBACK_THRESHOLD=0.35`.
- **Treatment enhancement:** for valid non-healthy labels, always calls `genai_handler.generate_treatment_advice()` (30s timeout via ThreadPoolExecutor) to replace the local `DISEASE_TREATMENTS` string with a detailed LLM answer.
- **Image validation:** calls `utils/image_validator.validate_image()` first; invalid ⇒ short-circuits with `invalid_image`.
- **Local treatment fallback table:** `DISEASE_TREATMENTS` (keyword→English treatment string).

### 3.3 `agents/knowledge_agent.py` → `run_knowledge`
- **Role:** Weather + market + agronomy + RAG enrichment.
- **Reads:** `lat`,`lon` (default Hyderabad 17.3850/78.4867), `crop_type`, `disease_prediction`, `offline`, `lang`, `user_text`.
- **Writes:** `knowledge={weather, market, agronomy, rag_context}`, `recommendations` (list[str]), `status="knowledge_complete"`.
- **Weather** (`_fetch_weather`): OpenWeatherMap (needs `OPENWEATHER_API_KEY`) → Open-Meteo (free, no key) → cache → defaults. Uses `utils/offline_cache.OfflineCache` (24h TTL).
- **Market** (`_get_market_price`): `MSP_PRICES_2025` table × random ±15% variation (simulated; comment says real eNAM would go here). `_find_nearest_mandi()` uses `CROP_MANDIS` w/ crude Euclidean-→km distance.
- **Agronomy** (`_get_disease_agronomy`): static `DISEASE_AGRONOMY` table (severity, yield_loss, actions, prevention).
- **RAG** (5s timeout): `knowledge_base/build_knowledge_db.symptom_search()` (if user_text>10 chars) or `rag_query()`.
- **Exports:** `MSP_PRICES_2025`, `DISEASE_AGRONOMY`.

### 3.4 `agents/compliance_agent.py` → `run_compliance`  🛡️ **DETERMINISTIC — NO LLM**
- **Role:** Regulatory guardrail (CIB&RC / FSSAI / ICAR). Pure rule-matching.
- **Reads:** `treatment`, `user_text`, `recommendations`, `crop_type`, `lang`.
- **Writes:** `compliance` (full report w/ audit_log), `status` = `compliance_complete` | `compliance_blocked`.
- **Databases (loaded at import from `compliance/`):**
  - `banned_pesticides.json` → `BANNED_DB` (banned + restricted + search_terms)
  - `mrl_limits.json` → `MRL_DB`
  - `safe_alternatives.json` → `ALTERNATIVES_DB`
- **Checks:** (1) banned-substance scan, (2) crop-specific restriction, (3) MRL limit + PHI, (4) safe-alternative lookup for violations.
- **Verdict:** banned ⇒ `unsafe`/high/`allowed=False`; restricted or crop-warning ⇒ `warning`/medium; else `safe`/low.
- **Disclaimers:** `DISCLAIMERS` dict, 10 languages.
- **Exports:** `BANNED_DB`, `MRL_DB`, `ALTERNATIVES_DB`.

### 3.5 `agents/output_agent.py` → `run_output`
- **Role:** Render final text + voice + chart + PDF.
- **Reads:** `disease_prediction`, `knowledge`, `recommendations`, `treatment`, `compliance`, `lang`.
- **Writes:** `final_response` (str), `voice_output_path` (mp3), `bloom_figure` (Plotly), `audit_pdf_path`, `status="output_complete"`.
- **Text:** `_format_response()` uses `RESPONSE_TEMPLATES` (5 langs fully: en/hi/kn/te/ta; others partial) + `DISEASE_NAMES` localization. Handles `uncertain_detection`/`unknown`/`error` and `compliance.allowed=False` branches.
- **Voice:** `_generate_voice()` via **gTTS** (strips emojis, truncates to 500 chars). Fallback: `_generate_fallback_audio()` writes a WAV beep chord.
- **Chart:** `utils/bloom_simulator.build_bloom_figure()` — health-recovery trajectory (before/after over 14 days). `_calculate_health_trajectory()` maps disease keyword → health %.
- **PDF:** `utils/pdf_audit.generate_audit_pdf()` (ReportLab).
- **Output dir:** `models/outputs/`.

---

## 4. GenAI / LLM Layer  🔌 (where APIs live — likely bug/enhancement zone)

### `utils/genai_handler.py` — the multi-backend LLM router
**Priority order: NVIDIA API → Google Gemini → local Ollama.**

- **NVIDIA** (`_nvidia_generate`) — PRIMARY. Base URL `https://integrate.api.nvidia.com/v1/chat/completions`.
  - Keys from env `NVIDIA_API_KEY`, `NVIDIA_API_KEY_2`, `NVIDIA_API_KEY_3` → `_NVIDIA_API_KEYS`.
  - **Smart key segregation** via `_NVIDIA_TASK_KEY`: vision→key0, treatment→key1, general→key2. Rotates on failure.
  - Models: `meta/llama-3.2-90b-vision-instruct` (vision), `meta/llama-3.3-70b-instruct` (text).
- **Gemini** (`_get_gemini_model`, `_generate`) — SECONDARY, often rate-limited. Supports new `google.genai` and old `google.generativeai` SDKs. Model `gemini-2.0-flash`. Has key rotation + 60s cooldown on 429.
  - ⚠️ **SECURITY ISSUE:** two **hardcoded Gemini API keys** in source at `utils/genai_handler.py:125-126` (`API_KEYS` list). Should be removed / moved to env. Flag before any commit/push.
- **Ollama** (`_ollama_generate`, `_ollama_vision_analyze`) — LOCAL fallback, zero rate limits. `http://localhost:11434`. Models `llama3.2:3b` (text), `llava:7b` (vision).
- **Public functions (called by agents):**
  - `is_genai_available()`
  - `generate_treatment_advice(disease, crop, region, season, language, context)` ← vision_agent
  - `analyze_unknown_crop(image_path, language)` — path-based Gemini vision
  - `analyze_unknown_crop_pil(image, language)` ← vision_agent (PIL-based)
  - `_ollama_vision_analyze(image, prompt)` ← vision_agent
  - `generate_audit_narrative(agent_trace, language)`
  - `conversational_followup(question, history, crop, disease, language)` ← UI follow-up
- **Known bug:** `analyze_unknown_crop()` JSONDecodeError branch references undefined `response` var (`genai_handler.py:424`).

### `utils/onnx_inference.py` — `ONNXVisionEngine`
- Alternate/offline inference path via ONNX Runtime (`get_optimal_providers()` picks CUDA/CPU). `get_engine()` singleton. **Note:** `run_vision` currently uses PyTorch engines, not this — ONNX path appears wired for offline but not invoked by the main graph. Verify before relying on it.

---

## 5. UI Layer

### `ui/app.py` (915 lines) — Gradio `Blocks` app, `launch_app(run_pipeline)`
- Quick-symptom buttons (Leaf Spots, Yellow Leaves, Insects, White Fungus, Wilting, Healthy Check).
- Language selector (updates labels via `update_ui_labels`), district → coords (`update_districts`, `_get_coords`), offline toggle.
- `process_query(...)` → wraps and calls the injected `run_pipeline`.
- `handle_followup(...)` → `genai_handler.conversational_followup`.
- `detect_location_from_coords`, `transcribe_audio` (voice input), `update_helpline` (→ `utils/helpline`).
- `mock_pipeline` fallback exists for isolated UI testing.
- **Also referenced in README but NOT in current tree:** WhatsApp channel, learning queue, `mobile_app/` (React Native — App.js + screens + ModelService.js exists as a separate client).

---

## 6. Support Utilities (`utils/`)

| File | Key funcs | Purpose |
|------|-----------|---------|
| `offline_cache.py` | `OfflineCache.get/set` | JSON file cache (`models/offline_cache.json`) w/ TTL. Used by knowledge agent. |
| `image_validator.py` | `validate_image` | Rejects non-leaf images (green-channel, color-variance, skin-tone heuristics). |
| `bloom_simulator.py` | `build_bloom_figure`, `build_comparison_figure` | Plotly health-recovery charts. |
| `crop_calendar.py` | `get_current_season`, `get_crop_advisory`, `get_seasonal_warning` | Seasonal advisory. |
| `fertilizer_calc.py` | `calculate_fertilizer`, `format_fertilizer_card` | ICAR NPK calculator. |
| `helpline.py` | `get_helplines`, `get_nearest_kvk`, `format_helpline_card` | Emergency contacts + KVK finder. |
| `pdf_audit.py` | `generate_audit_pdf` | ReportLab PDF advisory report. |
| `translator.py` | `translate_text`, `translate_to_english` | Multilingual translation w/ cache. |
| `onnx_inference.py` | `ONNXVisionEngine`, `get_engine` | ONNX offline inference (see §4). |

---

## 7. Knowledge Base / RAG

### `knowledge_base/build_knowledge_db.py`
- `build_knowledge_db()` — builds a **ChromaDB** vector store from `knowledge_base/crop_diseases.json` (ICAR advisories).
- `get_collection()`, `rag_query(query, crop, n_results)`, `symptom_search(...)` — queried by knowledge agent (5s timeout).

---

## 8. Data & Model Assets

- **Datasets (per MEMORY.md):** 121,388 images, 54 classes, 7 crops. Manifests in `data/manifests/{train,val,test}.csv`, `label_index.json`.
  - ⚠️ README claims "92-class EfficientNet-B4 / 91,000+ images"; MEMORY.md says 54 classes / 121k images. **Numbers disagree — reconcile which model is actually deployed.**
- **Checkpoints:** `models/checkpoints/efficientnet_b4_indian/` (expects `best_model.pth` + `class_labels.json`), `models/checkpoints/vit_crop_disease/`, `models/checkpoints/day4_vit/`.
- **Training:** `models/train_model.py`, `utils/train_vision.py`, `utils/train_eval_pipeline.py`, `models/export_onnx.py`.
- **Compliance DBs:** `compliance/*.json`.

---

## 9. Config & Environment

- `.env` keys used in code: `GEMINI_API_KEY`, `NVIDIA_API_KEY`(+`_2`,`_3`), `OPENWEATHER_API_KEY`, `AGRIBLOOM_VISION_MODEL_DIR`, `AGRIBLOOM_OFFLINE_DEFAULT`.
  - ⚠️ `.env.example` documents only `GEMINI_API_KEY` + model dir + offline — **missing NVIDIA and OpenWeather keys**. Update it.
- Logging → console + `agribloom.log`.
- `requirements.txt` (+ `requirements-stable-lock.txt` GPU lock).

---

## 10. Tests

- `tests/test_all.py` (291 lines) — README claims 14-test compliance suite. Run: `python -m pytest tests/test_all.py -v`.

---

## 11. State Dictionary Schema (the shared bus)

```
# Input (seeded by run_pipeline)
image, image_path, user_text, user_language, lang, offline, lat, lon,
chat_history, status, allow_path_hints, model_dir

# + orchestrator
route, detected_intent, detected_crop

# + vision
disease_prediction={label, confidence, class_index, top3, entropy, is_ood, source, ...}
crop_type, treatment

# + knowledge
knowledge={weather, market, agronomy, rag_context}, recommendations

# + compliance
compliance={allowed, compliance_status, risk_level, violations, crop_warnings,
            mrl_warnings, safe_alternatives, disclaimers, audit_log, ...}

# + output
final_response, voice_output_path, bloom_figure, audit_pdf_path

# + run_pipeline
elapsed_seconds  (error path adds: error)
```

---

## 12. Known Issues / Watch-List (for the bug-fix session)

1. **Hardcoded Gemini API keys** in `utils/genai_handler.py:125-126` — security leak; remove before push.
2. **Undefined `response`** in `analyze_unknown_crop()` JSON-error branch (`genai_handler.py:424`).
3. **Class-count mismatch** — README (92) vs MEMORY/manifests (54). Confirm deployed model.
4. **`.env.example` incomplete** — missing NVIDIA/OpenWeather keys.
5. **ONNX engine wired but unused** by main graph despite "offline" flag — verify offline path.
6. **`_format_response` compliance-blocked branch** joins `violations` as strings, but violations are dicts (`output_agent.py:201`) — will render `[object]`-ish. Cross-check with `run_output`'s own dict-safe handling at line 434.
7. **Market prices are simulated** (`random ±15%`), not real eNAM — fine for demo, note for production.
8. **Repo hygiene:** `.git-backup/` and `.venv-gpu-windows-backup/` are committed/present and pollute searches. Confirm they're gitignored.

---

## 13. How to Extend (quick pointers)

- **Add a language:** update `SUPPORTED_LANGUAGES` (orchestrator), `RESPONSE_TEMPLATES` + `DISEASE_NAMES` (output), `DISCLAIMERS` (compliance), `LANGUAGE_MAP` (output gTTS).
- **Add a crop:** `CROP_KEYWORDS` (orchestrator), `MSP_PRICES_2025` + `CROP_MANDIS` + `DISEASE_AGRONOMY` (knowledge), retrain vision model + `class_labels.json`.
- **Swap/route LLM provider:** everything funnels through `utils/genai_handler._generate()` and `_nvidia_generate` — this is the current focus of branch `feature/provider-router-v2`.
- **Add a pipeline stage:** register a node + edges in `main.py:build_graph()`.
