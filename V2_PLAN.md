# AgriBloom V2 — Build Plan (Fully Open-Source · Free · Scalable · Multi-Channel)

> Guiding rule from Saketh: **100% open source, free to deploy AND free to run, reachable by
> any farmer (Web + WhatsApp + Android app), fully advanced, and able to serve many farmers at
> once (scale/load).** No paid APIs are a *required* dependency anywhere.

---

## 0. The one big change vs the V2 doc

Your `Project_1_AgriBloom_V2.md` assumes **paid Claude API + paid OpenAI Whisper** for production.
That breaks the "everything free + open source" rule. So we swap every paid piece for a free/OSS
equivalent, keeping the *same advanced algorithms*:

| V2 doc (paid) | AgriBloom V2 (this plan, free/OSS) |
|---|---|
| Claude API (reasoning) | **Groq free tier** (Llama-3.3-70B) → **NVIDIA NIM** free → **Ollama** local |
| OpenAI Whisper API | **faster-whisper** (local, OSS) + **Groq Whisper** free |
| edge-tts / gTTS | **edge-tts** (free) + gTTS fallback |
| LangSmith (paid obs) | **Langfuse** (open source, self-hostable, free) |
| Qdrant Cloud | **Qdrant** self-hosted/local (OSS) or free 1 GB tier; Chroma stays as fallback |
| ChromaDB cosine | Hybrid retrieval on Qdrant (still OSS) |

All model weights (EfficientNet-B4, DINOv2, LLaVA, cross-encoder, MiniLM) are open weights.

---

## 1. Scale strategy (new — the "any number of farmers" requirement)

V1 today = single-process Gradio, mostly one-user-at-a-time. To serve many farmers:

- **Split into a stateless FastAPI backend** = the agent pipeline as an async HTTP/JSON API.
  Every channel (Web, WhatsApp, Android) calls this same API. No channel holds business logic.
- **Async everywhere** + a **job queue** for heavy vision inference (RQ/Redis or in-proc async),
  so slow requests don't block others.
- **Caching layer**: cache disease→advice, weather, and RAG results → cuts LLM calls ~60–80%,
  which is what actually keeps free tiers alive under load.
- **Horizontal scale**: containerized workers; run N replicas on free HF Spaces / Fly.io free / a
  friend's GPU box. Stateless design = add replicas freely behind a load balancer.
- **Rate-limit + graceful degradation**: if Groq is busy → NVIDIA → Ollama → cached/deterministic
  answer. Farmer always gets *something* safe.
- **Load tested** with Locust before we call it done.

> Honest note: "free" + "unlimited farmers" has real ceilings on free tiers. This design scales
> *horizontally for free as far as the free tiers allow*, degrades gracefully past that, and is
> one config change away from paid scale if you ever want it. I'll flag the concrete limits at deploy.

---

## 2. Target architecture

```
   Farmers
   ├─ Web (Gradio, free HF Space)
   ├─ WhatsApp (OSS: Baileys / whatsapp-web.js, or Meta Cloud API free tier)
   └─ Android APK (Expo + React Native / TypeScript, shared via WhatsApp)
                    │  all call the same API
                    ▼
        ┌───────────────────────────────┐
        │   FastAPI backend (async)     │
        │   + queue + cache + limits    │
        └───────────────┬───────────────┘
                        ▼
        ╔═══════════════════════════════╗
        ║  ReAct Orchestrator (LangGraph║  ← Groq/NVIDIA/Ollama (free)
        ║  conditional edges)           ║
        ╚═══╤═══════════╤═══════════╤═══╝
            ▼           ▼           ▼
      Vision Ensemble  Hybrid RAG  Reflexion Compliance
      (EffNet+DINOv2   (Qdrant+BM25 (46 rules + self-
       +LLaVA vote)     +RRF+rerank  correct loop)
                        +HyDE)
            └───────────┼───────────┘
                        ▼
              Output (multilingual)
              + Voice (faster-whisper in / edge-tts out)
                        │
              Langfuse traces (OSS observability) run alongside
```

Keeps V1's dict-state + LangGraph foundation; upgrades each node.

---

## 3. Open-source building blocks (prebuilt from GitHub — used as libraries)

LangGraph · Qdrant · rank_bm25 · sentence-transformers (embeddings + cross-encoder) · FAISS ·
DINOv2 (facebookresearch) · timm/EfficientNet · Ollama · faster-whisper · edge-tts · Gradio ·
FastAPI · Langfuse · Baileys/whatsapp-web.js · Expo/React Native · Locust. We *import* these and
write all orchestration, tool defs, voting, fusion, and business logic ourselves.

---

## 4. The 6 advanced algorithms (kept from your doc, implemented ourselves)

1. **Weighted ensemble voting** w/ Platt-scaled confidence + agreement penalty (vision).
2. **Reciprocal Rank Fusion** (BM25 + dense) for hybrid retrieval.
3. **HyDE** — multi-hypothesis hypothetical-document embeddings.
4. **ReAct** — Thought→Action→Observation loop via tool use + conditional edges.
5. **Reflexion** — compliance self-correction loop (max 3 iters, then escalate).
6. **DINOv2 + FAISS KNN** — zero-shot novel-crop detection.

---

## 5. Phased build — you approve each phase, I build it, test edge cases, show you, then next

**Phase 0 — Foundations & safety (do first)**
- Branch `agribloom-v2`; folder scaffolding; CI (ruff + pytest); `.env.example` completed.
- Fix known V1 bugs so we build on solid ground:
  - Remove hardcoded Gemini keys (`genai_handler.py:125-126`) → env only.
  - Fix undefined `response` in `analyze_unknown_crop()`.
  - Reconcile class-count mismatch (54 vs 92) — confirm deployed model.
  - Fix compliance-blocked dict-vs-string render bug (`output_agent.py:201`).
  - Gitignore `.git-backup/`, `.venv-gpu-windows-backup/`.

**Phase 1 — Vision Ensemble** (Algorithms 1 + 6)
- Add DINOv2+FAISS KNN model + LLaVA reasoning; weighted voting w/ calibration; "not sure"
  graceful path for novel crops.

**Phase 2 — Hybrid RAG** (Algorithms 2 + 3)
- Qdrant dense + BM25 + RRF + cross-encoder rerank + HyDE. Ingest ICAR docs. Eval vs V1.

**Phase 3 — ReAct Orchestrator** (Algorithm 4)
- LangGraph conditional edges, dynamic tool selection; replaces the fixed linear order.

**Phase 4 — Reflexion Compliance** (Algorithm 5)
- Self-correction loop over the deterministic 46-rule engine. **100% banned-pesticide catch rate**
  enforced by a 100-case regression suite (non-negotiable).

**Phase 5 — Voice I/O**
- faster-whisper (in) + edge-tts (out), 10 Indic languages, auto language detect.

**Phase 6 — Scale backend**
- FastAPI async service, job queue, caching, rate limiting, Langfuse observability, provider
  fallback chain (Groq→NVIDIA→Ollama).

**Phase 7 — Channels (reach every farmer) — FRONTEND DECISION LOCKED**
- **Public web = Next.js + TypeScript + Tailwind + shadcn/ui on Vercel (free)** — premium,
  professional, mobile-first, animated. This is the live link for resume/recruiters.
- **Mobile = PWA (install the web app to home screen)** + **Expo/React Native APK** for WhatsApp sharing.
- **WhatsApp** bot (OSS: Baileys / whatsapp-web.js).
- **Gradio** kept as internal dev/test tool only — not the public face.
- All channels call the same **FastAPI** backend.

**Phase 8 — Deploy free + load test**
- HF Spaces (web/API), Qdrant, WhatsApp webhook on Cloudflare Workers free, APK via WhatsApp.
  Locust load test; document real free-tier limits + degradation behavior.

**Phase 9 — Merge to `main` + docs + farmer case study**
- Only after V2 is proven end-to-end. V1/main stays working the whole time.

---

## 6. Testing & edge cases (every phase)
- Unit: voting math, RRF, HyDE, reflexion loop.
- Integration: full pipeline w/ mocked LLM.
- Edge cases: novel/unknown crop, blurry/non-leaf image, no internet (offline path), all providers
  down, banned-pesticide traps, mixed-language input, empty/huge input, concurrent load.
- Compliance regression: 100 scenarios, zero misses required.

---

## 7. Ground rules while building
- Keep `main` (V1) working until V2 is proven; all work on `agribloom-v2`.
- No paid API is ever a hard dependency.
- Commit/push only when you ask.
- After each phase: I show you what changed + how I tested it, you approve, then I continue.

---

## 8. GPU usage (RTX 4060 8 GB · Ryzen 9 · 24 GB RAM · 1 TB SSD)

Your laptop does the **vision + embedding + voice** heavy lifting on CUDA; the free cloud does the
**70B reasoning** (too big for 8 GB). Clean split, all free.

| Workload | Runs on | VRAM budget |
|---|---|---|
| EfficientNet-B4 inference / fine-tune | RTX 4060 (CUDA) | ~2–4 GB |
| DINOv2 embeddings (ViT-S/B) | RTX 4060 (CUDA) | ~1–2 GB |
| LLaVA:7b vision (Ollama, Q4) | RTX 4060 (CUDA) | ~5–6 GB |
| faster-whisper (small/medium int8) | RTX 4060 (CUDA) | ~1–2 GB |
| FAISS KNN | GPU or CPU | small |
| Reasoning brain (Llama-3.3-70B) | **Groq/NVIDIA free cloud** (won't fit 8 GB) | n/a |

Note: run vision + LLaVA sequentially (not both peaking at once) to stay under 8 GB. Your PC can
also act as a **self-hosted GPU worker** behind the FastAPI backend for extra free capacity.

---

## 9. Research paper (deliverable)

- **Angle:** "A fully open-source, free-tier, horizontally-scalable multi-agent system (ReAct +
  Reflexion + weighted vision ensemble + hybrid RAG) for regulatory-safe agricultural advisory in
  10 Indian languages."
- **Contributions:** the free/OSS scalable architecture; 100%-catch Reflexion compliance guardrail;
  ensemble + DINOv2 zero-shot novel-crop handling; multilingual voice reach.
- **Evidence collected during the build:** retrieval precision (V1 70% → V2 target 92%), ensemble
  accuracy, compliance catch-rate on 100 cases, latency under load (Locust), farmer case study.
- **Output:** `docs/PAPER/` with figures auto-generated from eval notebooks → IEEE-format draft.

---

## 10. Repository skeleton (created in Phase 0)

```
agribloom-v2/                     # on branch agribloom-v2
├── agents/            # base_agent, orchestrator(ReAct), vision_ensemble, knowledge, compliance, voice, output
├── models/            # efficientnet_wrapper, dinov2_knn, ollama_llava, calibration
├── rag/               # ingestion, bm25, dense(qdrant), rrf_fusion, reranker, hyde
├── graph/             # state, react_graph, conditional_edges
├── voice/             # faster_whisper_client, edge_tts_client, language_detect
├── compliance/        # rules_engine, banned/mrl/alternatives json, reflexion, violation_formatter
├── backend/           # FastAPI app, queue, cache, rate_limit, provider_router (Groq→NVIDIA→Ollama)
├── observability/     # langfuse_setup, metrics, request_logger
├── channels/          # gradio_app, whatsapp_bot (OSS), sms (optional)
├── mobile/            # Expo + React Native (TypeScript) APK
├── data/              # icar_docs, banned_pesticides.json, crops_metadata, reference_gallery (DINOv2)
├── tests/             # unit / integration / e2e + compliance_regression (100 cases) + load (locust)
├── deployment/        # Dockerfile, docker-compose, HF Space config, .github/workflows (CI/CD)
├── notebooks/         # model_evaluation, error_analysis, paper_figures
├── docs/              # ARCHITECTURE, EVALUATION, CASE_STUDY, DEPLOYMENT, PAPER/
├── .env.example  requirements.txt  pyproject.toml  README.md  LICENSE(MIT)
```

---

## 11. How we actually work — Scrum / Agile (planning → testing → deploy)

**Cadence:** short phase-sprints (each Phase in §5 = one mini-sprint). You are Product Owner
(approve/prioritize); I'm the dev team. We track work with the task list.

**Per-sprint flow (repeats for every phase):**
1. **Plan** — I write the sprint backlog (user stories + acceptance criteria) and show you.
2. **Build** — I implement on `agribloom-v2`, small commits, typed + linted.
3. **Test** — unit + integration + edge cases + (for compliance) the 100-case regression. I run
   the `/verify` skill to prove it works end-to-end, not just "tests pass."
4. **Review (Sprint Review)** — I show you: what changed, how I tested it, live demo/screenshot.
   You approve or send it back.
5. **Retro** — one line: what worked / what to change next sprint.
6. **Deploy** — merge the phase, redeploy the HF Space, verify the live link still works.

**Definition of Done (every phase):**
- Code typed + `ruff` clean + tests green (CI passes)
- Edge cases covered (novel crop, blurry image, offline, all-providers-down, banned-pesticide traps)
- Demo works on the live link
- You approved it

**Sprint board (high level):**
| Sprint | Phase(s) | Goal |
|---|---|---|
| S0 | Phase 0 | Foundations + V1 bug fixes + skeleton + CI |
| S1 | Phases 1–2 | Vision ensemble + Hybrid RAG (GPU work) |
| S2 | Phases 3–4 | ReAct orchestrator + Reflexion compliance |
| S3 | Phases 5–6 | Voice I/O + scalable FastAPI backend |
| S4 | Phases 7–8 | WhatsApp + Android + deploy + load test |
| S5 | Phase 9 | Merge to main + docs + paper draft + case study |

