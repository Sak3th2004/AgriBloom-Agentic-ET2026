# AgriBloom V2 — FRONTEND BUILD PLAN (standalone brief for a parallel builder)

> **Read this if you are building ONLY the frontend** (e.g. in a separate Fable 5 window).
> The backend + REST API is being built in parallel by another session. You do **not** need the
> backend running — this doc contains the full **API contract**, so you build against **mocks**
> and flip to the real API later by changing one env var (`NEXT_PUBLIC_API_BASE_URL`).
>
> Frontend and backend agree on **§6 API contract** as the single source of truth. Do not change
> field names without syncing.

---

## 1. Mission & hard constraints

- **Users:** small Indian farmers (often low-literacy, low-bandwidth, phone-first) + KVK officers.
- **Must be:** premium & smooth (recruiter-grade), yet dead-simple for a farmer.
- **100% free hosting:** deploy on **Vercel free tier**. No paid services.
- **Mobile-first + installable PWA** (add-to-home-screen behaves like an app).
- **Multilingual:** 10 Indic languages (en, hi, kn, te, ta, bn, mr, gu, ml, pa) + big icons so
  text isn't the only guide.
- **Voice-first option:** record a question (mic), get spoken answer (audio player).

---

## 2. Tech stack (use exactly this — it's free, modern, resume-grade)

| Concern | Choice | Why |
|---|---|---|
| Framework | **Next.js 14 (App Router) + TypeScript** | premium, SSR/SSG, free on Vercel |
| Styling | **Tailwind CSS** | fast, consistent |
| Components | **shadcn/ui** (Radix under the hood) | polished, accessible, animated |
| Animation | **framer-motion** | smooth micro-interactions |
| Data fetching | **@tanstack/react-query** | caching, retries, loading states |
| Validation | **zod** | validate API responses at runtime |
| API mocking | **MSW (Mock Service Worker)** | build with zero backend |
| i18n | **next-intl** | 10-language routing + messages |
| Icons | **lucide-react** | clean icon set |
| PWA | **next-pwa** (or manual manifest + SW) | installable app |
| Charts | **Recharts** | the "bloom" health-recovery chart |
| Forms/upload | native + shadcn, **react-dropzone** for image | camera + file |

> If any chart/graph work is done, follow the **dataviz** skill's palette/mark rules.

---

## 3. Design principles (farmer-first premium)

1. **One primary action per screen.** Big buttons, thumb-reachable, min 48px targets.
2. **Icon + short text + optional voice** for every key element (low-literacy safe).
3. **Camera is king** — the home screen's main CTA is "Take/Upload leaf photo".
4. **Show progress, not spinners** — because the agent "thinks" in steps (see SSE stream in §6.3),
   show a live "Analyzing image → Looking up treatment → Checking safety" stepper.
5. **Safety is loud** — banned-pesticide / compliance warnings use a distinct red alert card.
6. **Works offline-ish** — cache last result; show a clear offline banner.
7. **Light + dark mode**, calm agricultural palette (greens/earth), high contrast.

---

## 4. Screens / routes to build

| Route | Screen | Key elements |
|---|---|---|
| `/` | **Home / Ask** | Language switcher, big "Scan Leaf" camera/upload, text box, mic button, district selector, "offline" toggle, quick-symptom chips (Leaf Spots, Yellow Leaves, Insects, White Fungus, Wilting, Healthy Check) |
| `/result/[id]` | **Advisory result** | Disease name + confidence meter, affected-crop badge, treatment card, **compliance/safety card** (green safe / red blocked + safe alternatives), weather + market cards, bloom recovery chart, audio player (spoken answer), "Ask a follow-up" box, download PDF button, share button |
| `/chat/[session]` | **Follow-up chat** | Conversational Q&A thread tied to a diagnosis |
| `/about` | **About / trust** | What it is, free & open-source, KVK helpline, disclaimer |
| `/offline` | PWA offline fallback | cached content + retry |

Global: top bar (logo, language, light/dark), bottom nav on mobile (Home, History, Help).

---

## 5. Full feature checklist

- [ ] Image upload **and** live camera capture (mobile `capture="environment"`)
- [ ] Text question input (any of 10 languages)
- [ ] **Voice input**: record mic → send audio → show transcription
- [ ] **Voice output**: play returned audio; language matches input
- [ ] Language switcher (persists to localStorage + URL)
- [ ] District → lat/lon selector (dropdown; ship a JSON of districts→coords)
- [ ] Offline toggle (passes `offline:true` to API)
- [ ] Quick-symptom chips → prefill query
- [ ] **Live "agent thinking" stepper** via SSE (graceful fallback to spinner)
- [ ] Result: disease, confidence meter, treatment, recommendations list
- [ ] **Compliance card**: safe (green) / blocked (red) + safe alternatives + disclaimer
- [ ] Weather + market info cards
- [ ] Bloom health-recovery chart (Recharts, 14-day trajectory)
- [ ] Follow-up chat
- [ ] Download advisory PDF (link from API)
- [ ] Share (WhatsApp deep link `https://wa.me/?text=...`)
- [ ] History (localStorage list of past diagnoses)
- [ ] PWA install prompt + offline fallback
- [ ] Light/dark mode
- [ ] Loading / error / empty states everywhere (react-query)
- [ ] Accessibility: labels, focus rings, ARIA, keyboard nav

---

## 6. THE API CONTRACT  ⭐ (shared source of truth — do not diverge)

Base URL from env: `NEXT_PUBLIC_API_BASE_URL` (e.g. `https://<hf-space>.hf.space`).
All JSON. CORS enabled. Errors use shape in §6.6.

### 6.1 `POST /api/v1/diagnose`  — main endpoint
`multipart/form-data`:
| field | type | required | notes |
|---|---|---|---|
| `image` | file | no* | leaf photo (jpg/png). *either image or text required |
| `text` | string | no* | farmer's question |
| `language` | string | yes | one of the 10 codes; `auto` allowed |
| `lat` | number | no | default backend Hyderabad |
| `lon` | number | no | |
| `offline` | boolean | no | default false |

**200 response** (`application/json`):
```json
{
  "id": "diag_abc123",
  "session_id": "sess_xyz",
  "language": "te",
  "status": "ok",                         // ok | uncertain | invalid_image | blocked | error
  "disease": {
    "label": "grape_downy_mildew",
    "display_name": "Grape Downy Mildew",
    "crop": "grape",
    "confidence": 0.85,                    // 0..1
    "is_uncertain": false,
    "top3": [
      {"label": "grape_downy_mildew", "display_name": "Grape Downy Mildew", "confidence": 0.85},
      {"label": "grape_black_rot", "display_name": "Grape Black Rot", "confidence": 0.09}
    ],
    "source": "ensemble"                   // ensemble | efficientnet | dinov2_knn | llava | fallback
  },
  "treatment": "Apply ...",                // plain text, in `language`
  "recommendations": ["...", "..."],
  "compliance": {
    "allowed": true,                       // false => show red blocked card
    "risk_level": "low",                   // low | medium | high
    "status": "safe",                      // safe | warning | unsafe
    "blocked_substances": ["Ridomil MZ"],  // names only, may be empty
    "safe_alternatives": ["Copper oxychloride ..."],
    "disclaimer": "Consult your local KVK ..."
  },
  "knowledge": {
    "weather": {"temp_c": 29, "humidity": 70, "rain_mm": 2, "desc": "Partly cloudy"},
    "market": {"crop": "grape", "modal_price": 4500, "unit": "quintal", "mandi": "Bengaluru"}
  },
  "chart": {                               // for Recharts; may be null
    "days": [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    "without_treatment": [80,74,68,...],
    "with_treatment": [80,82,85,...]
  },
  "audio_url": "/api/v1/files/diag_abc123.mp3",   // spoken answer, may be null
  "pdf_url": "/api/v1/files/diag_abc123.pdf",     // may be null
  "elapsed_seconds": 4.2
}
```
Notes for FE: if `status==="invalid_image"` show a "retake photo" tip card; if
`disease.is_uncertain` show the "not sure, tips for better photo" state; if
`compliance.allowed===false` lead with the **red safety card**.

### 6.2 `POST /api/v1/chat` — follow-up Q&A
```json
// request
{ "session_id": "sess_xyz", "question": "How often should I spray?",
  "language": "te", "history": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}] }
// 200
{ "answer": "Spray every 7 days ...", "language": "te", "audio_url": null }
```

### 6.3 `GET /api/v1/diagnose/stream?job=<id>` — SSE agent progress (optional but nice)
Server-sent events; each event:
```
event: step
data: {"stage":"vision","message":"Analyzing image","progress":0.3}

event: step
data: {"stage":"knowledge","message":"Finding treatment","progress":0.6}

event: done
data: {"id":"diag_abc123"}   // then GET the full result, or the result is inlined
```
FE: render a stepper; if SSE unsupported, fall back to a normal POST + spinner.

### 6.4 `POST /api/v1/voice/transcribe` — mic → text
`multipart/form-data`: `audio` (file), optional `language` (`auto` default).
```json
// 200
{ "text": "ఈ ఆకుపై నల్ల మచ్చలు", "language": "te" }
```

### 6.5 metadata endpoints
- `GET /api/v1/health` → `{"status":"ok","version":"2.0.0"}`
- `GET /api/v1/languages` → `[{"code":"en","name":"English","native":"English"}, {"code":"te","name":"Telugu","native":"తెలుగు"}, ...]`
- `GET /api/v1/meta` → `{ "crops":[...], "quick_symptoms":[{"id":"leaf_spots","label":"Leaf Spots","icon":"..."}], "districts":[{"name":"Bengaluru","lat":12.97,"lon":77.59}] }`
- `GET /api/v1/files/{name}` → serves mp3/pdf (or use direct absolute URLs).

### 6.6 error shape (any non-2xx)
```json
{ "status": "error", "code": "IMAGE_TOO_LARGE", "message": "Human-readable, already localized when possible" }
```

---

## 7. Build against mocks (so you don't wait for the backend)

- Add **MSW** handlers that return the §6 example payloads (happy path + `blocked` + `uncertain`
  + `invalid_image` + `error`). Toggle with `NEXT_PUBLIC_USE_MOCKS=true`.
- Define **zod schemas** matching §6 and parse every response — this catches contract drift the
  moment the real API returns something off.
- Ship a `districts.json` and a `languages.ts` constant so `/meta` and `/languages` can be mocked
  or hardcoded until wired.

---

## 8. Project structure

```
web/                              # separate folder / repo; deploys to Vercel
├── app/
│   ├── (marketing)/about/page.tsx
│   ├── page.tsx                  # Home / Ask
│   ├── result/[id]/page.tsx
│   ├── chat/[session]/page.tsx
│   ├── offline/page.tsx
│   └── layout.tsx                # top bar, theme, i18n provider
├── components/
│   ├── ui/                       # shadcn generated
│   ├── ScanCard.tsx  UploadDropzone.tsx  MicRecorder.tsx
│   ├── ConfidenceMeter.tsx  ComplianceCard.tsx  TreatmentCard.tsx
│   ├── WeatherCard.tsx  MarketCard.tsx  BloomChart.tsx
│   ├── AgentStepper.tsx  LanguageSwitcher.tsx  AudioPlayer.tsx
│   └── QuickSymptomChips.tsx  HistoryList.tsx
├── lib/
│   ├── api.ts                    # typed client for §6 (fetch + zod)
│   ├── schemas.ts                # zod schemas
│   ├── mocks/                    # MSW handlers
│   └── constants/ (languages.ts, districts.json, symptoms.ts)
├── messages/                     # next-intl: en.json, hi.json, te.json, ...
├── public/ (manifest.json, icons, service worker)
├── .env.example                 # NEXT_PUBLIC_API_BASE_URL, NEXT_PUBLIC_USE_MOCKS
├── tailwind.config.ts  next.config.js  package.json  tsconfig.json
```

---

## 9. Frontend build order (parallel-safe milestones)

1. Scaffold Next.js + TS + Tailwind + shadcn + react-query; theme + layout + i18n skeleton.
2. `lib/schemas.ts` + `lib/api.ts` + MSW mocks for **all** §6 endpoints (contract-first).
3. Home screen: language switcher, upload/camera, text box, quick-symptom chips, submit.
4. Result screen with **mock data**: disease + confidence meter + treatment + **compliance card**.
5. Weather/market cards + BloomChart (Recharts).
6. Voice: MicRecorder → transcribe; AudioPlayer for `audio_url`.
7. AgentStepper via SSE (with spinner fallback).
8. Follow-up chat screen.
9. History (localStorage), share (wa.me), PDF download link.
10. PWA (manifest + service worker + offline route) + light/dark polish + a11y pass.
11. Flip `NEXT_PUBLIC_USE_MOCKS=false`, point at real API, smoke test.
12. Deploy to Vercel (free), set `NEXT_PUBLIC_API_BASE_URL` secret.

---

## 10. Definition of Done (frontend)
- All §5 features work against mocks, then against the real API.
- Every response parsed with zod; loading/error/empty states everywhere.
- Mobile-first, installable PWA, light/dark, 10 languages switch correctly.
- Lighthouse: PWA installable, a11y ≥ 90.
- Deployed to Vercel with a live URL.

---

## 11. Division of labor (so the two windows don't collide)
- **Backend/API window (the other session):** implements §6 endpoints, agents, vision, RAG,
  compliance, voice, deploys FastAPI to HF Spaces. Owns the contract's server side.
- **This frontend window (Fable 5):** everything under `web/`. Owns the contract's client side.
- **Shared rule:** §6 is the contract. If either side must change a field, note it and tell the
  other. Frontend never blocks on backend thanks to mocks.
```
