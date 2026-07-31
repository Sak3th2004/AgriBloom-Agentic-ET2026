# AgriBloom V2 — Deployment

Everything here is prepared and verified as far as possible without your
credentials. Going live requires YOUR login (HuggingFace + Vercel) — I can't
authenticate as you, so these are the exact commands to run.

## What's ready
- `deployment/Dockerfile` — CPU backend image, builds from repo root
- `deployment/requirements-cpu.txt` — trimmed CPU-only deps
- `deployment/docker-compose.yml` — local full-stack test
- `deployment/huggingface_space/README.md` — HF Space config (Docker SDK)
- `.github/workflows/deploy.yml` — manual-trigger CI/CD (needs secrets, see below)
- Frontend (`web/`) — already verified: `npm run build` compiles clean,
  Playwright smoke test passes in a real browser.

## 1. Backend → HuggingFace Spaces (free)

```bash
# One-time login (opens browser / asks for a token from
# https://huggingface.co/settings/tokens — needs "write" scope)
huggingface-cli login

# Create the Space (Docker SDK, free CPU tier)
huggingface-cli repo create agribloom-v2-api --type space --space-sdk docker

# Copy the Space's own README (has the required YAML frontmatter) then push
cp deployment/huggingface_space/README.md /tmp/agribloom-space-readme.md
git remote add hf https://huggingface.co/spaces/<YOUR_USERNAME>/agribloom-v2-api
git push hf feature/phase-8-deploy-prep:main
# Then, in the Space's own git working copy (or via the HF web UI file editor),
# replace README.md with deployment/huggingface_space/README.md's content,
# and move/copy deployment/Dockerfile to the Space repo ROOT as `Dockerfile`
# (HF Docker Spaces require the Dockerfile at repo root).

# The trained checkpoint (models/checkpoints/efficientnet_b4_indian/best_model.pth,
# 213MB) is gitignored on GitHub for size — it must be added directly to the
# Space repo via Git LFS (HF Spaces support large files natively):
cd <space-clone>
git lfs install
git lfs track "*.pth"
cp <path-to>/best_model.pth models/checkpoints/efficientnet_b4_indian/
git add .gitattributes models/checkpoints/efficientnet_b4_indian/best_model.pth
git commit -m "Add trained checkpoint via LFS"
git push

# Add secrets: Space Settings -> Repository secrets ->
#   GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3
#   NVIDIA_API_KEY, NVIDIA_API_KEY_2, NVIDIA_API_KEY_3
#   GROQ_API_KEY, OPENWEATHER_API_KEY (optional)
#   ENVIRONMENT=prod
```

Verify: `https://<your-username>-agribloom-v2-api.hf.space/api/v1/health`
should return `{"status":"ok","version":"2.0.0"}`.

## 2. Frontend → Vercel (free)

```bash
cd web
npx vercel login        # opens browser
npx vercel               # first deploy, links the project
npx vercel --prod        # promote to production
# In the Vercel dashboard, set the env var:
#   NEXT_PUBLIC_API_BASE_URL = https://<your-username>-agribloom-v2-api.hf.space
#   NEXT_PUBLIC_USE_MOCKS = false
# Redeploy after setting env vars (Vercel dashboard -> Redeploy).
```

## 3. Local full-stack verification (no cloud accounts needed)

```bash
docker compose -f deployment/docker-compose.yml up --build
curl http://localhost:7860/api/v1/health

cd web
NEXT_PUBLIC_API_BASE_URL=http://localhost:7860 NEXT_PUBLIC_USE_MOCKS=false npm run dev
# open http://localhost:3000 — now talking to the real local backend
```

## 4. CI/CD (optional, once secrets are set)
`.github/workflows/deploy.yml` is manual-trigger only (`workflow_dispatch`) —
runs the test suite, then pushes to the HF Space. Add repo secrets `HF_TOKEN`
(write-scope token) and `HF_SPACE` (e.g.
`https://huggingface.co/spaces/<user>/agribloom-v2-api`) before running it
from the Actions tab.

## Known free-tier limits (be upfront about these)
- HF free Spaces: CPU only, sleeps after inactivity (cold start ~30-60s),
  16GB RAM. Vision ensemble runs EfficientNet on CPU (slower than your local
  GPU) and skips DINOv2/LLaVA local votes (falls back to cloud vision APIs).
- Vercel free: generous for a portfolio/demo; fine for field testing with
  10s-100s of farmers.
- This is "free as far as free goes" — see V2_PLAN.md §1 for the scaling
  strategy beyond that ceiling.
