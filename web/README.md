# AgriBloom V2 — Web Frontend

Farmer-first Next.js 14 app (TypeScript + Tailwind + framer-motion + react-query + zod).
Built against the **FRONTEND_PLAN.md §6 API contract** with a built-in mock API, so it runs
with **zero backend**.

## Run

```bash
npm install
npm run dev        # http://localhost:3000
```

## Mock API

The §6 contract is implemented as Next.js route handlers under `app/api/v1/*`
(diagnose, SSE stream, chat, voice transcribe, languages, meta, health, files).
`lib/api.ts` + `lib/schemas.ts` (zod) form the typed client.

**Switching to the real backend:** set `NEXT_PUBLIC_API_BASE_URL` in `.env.local`
to the FastAPI URL. Empty = same-origin mocks.

### Mock scenario triggers (type these words in the question box)

| Trigger in text | Scenario shown |
|---|---|
| *(anything else)* | Happy path — Grape Downy Mildew, safe treatment, chart |
| `insect`, `ridomil`, `banned`, `monocrotophos` | **Red blocked compliance card** with safe alternatives |
| `healthy` | Healthy crop, no treatment needed |
| `unknown`, `strange`, `not sure` (or very short text without photo) | Uncertain — retake-photo guidance |
| `blur`, `invalid` | Invalid image — retake tips |

Voice mic button returns a canned transcription in the selected language.
Audio player plays a generated placeholder chime (`.wav`); PDF downloads a
generated one-page advisory.

## Features

- Home: camera capture / upload (client-side compression), text + mic input,
  quick-symptom chips, district selector, offline toggle
- Live 5-agent progress stepper driven by SSE (with timed fallback)
- Result: diagnosis + confidence meter, loud compliance card (green/red),
  treatment, recommendations, weather + mandi price, 14-day bloom recovery
  chart, listen / PDF / WhatsApp share / follow-up chat
- Follow-up chat, history (localStorage), about, offline fallback
- 10 Indic languages (switcher persists), light/dark, installable PWA
