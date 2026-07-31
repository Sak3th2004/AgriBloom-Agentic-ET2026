---
title: AgriBloom V2 API
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AgriBloom V2 — Backend API

Free, open-source agentic agricultural advisory backend. Serves the
`/api/v1/*` contract consumed by the AgriBloom web app, WhatsApp bot, and
Android app.

- Health check: `GET /api/v1/health`
- Full contract: see `FRONTEND_PLAN.md` §6 in the main repo.
- Source: https://github.com/Sak3th2004/AgriBloom-Agentic-ET2026

This Space is built from `deployment/Dockerfile` at the main repo root (see
`deployment/README.md` for the exact push steps). Runs CPU-only — the vision
ensemble degrades gracefully (EfficientNet CPU inference; DINOv2/LLaVA votes
are skipped without a GPU, with cloud vision fallback via the free-tier
NVIDIA/Gemini APIs still available).

**Required Space secrets** (Settings → Repository secrets):
`GEMINI_API_KEY`, `NVIDIA_API_KEY` (+`_2`,`_3`), `GROQ_API_KEY`,
`OPENWEATHER_API_KEY` (optional — falls back to free Open-Meteo),
`ENVIRONMENT=prod`.
