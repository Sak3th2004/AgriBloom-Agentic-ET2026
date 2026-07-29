"""
FastAPI service implementing the FRONTEND_PLAN.md §6 contract.

All channels (Next.js web, WhatsApp, Android) call these endpoints. The heavy
agent pipeline is loaded lazily and can be swapped for tests via
``set_pipeline()`` so the API layer is testable without any ML model.

Run locally:
    uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations

import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from backend.serializers import serialize_diagnosis
from voice.languages import LANGUAGE_NAMES, SUPPORTED

logger = logging.getLogger(__name__)

app = FastAPI(title="AgriBloom V2 API", version="2.0.0")

# CORS: allow the web/app origins (tighten in production via env).
_origins = os.getenv("AGRIBLOOM_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("models/outputs")

# ── Pipeline injection (swappable for tests) ────────────────────────────────────
PipelineFn = Callable[..., dict[str, Any]]
_pipeline: Optional[PipelineFn] = None


def set_pipeline(fn: Optional[PipelineFn]) -> None:
    """Override the pipeline function (used by tests)."""
    global _pipeline
    _pipeline = fn


def _get_pipeline() -> PipelineFn:
    global _pipeline
    if _pipeline is None:
        from main import run_pipeline  # lazy: avoids loading torch at import

        _pipeline = run_pipeline
    return _pipeline


# ── Metadata endpoints ──────────────────────────────────────────────────────────
@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/v1/languages")
def languages() -> list[dict[str, str]]:
    native = {
        "en": "English", "hi": "हिन्दी", "te": "తెలుగు", "ta": "தமிழ்",
        "kn": "ಕನ್ನಡ", "bn": "বাংলা", "mr": "मराठी", "gu": "ગુજરાતી",
        "ml": "മലയാളം", "pa": "ਪੰਜਾਬੀ",
    }
    return [{"code": c, "name": LANGUAGE_NAMES[c], "native": native.get(c, c)} for c in SUPPORTED]


@app.get("/api/v1/meta")
def meta() -> dict[str, Any]:
    quick = [
        {"id": "leaf_spots", "label": "Leaf Spots", "icon": "spots"},
        {"id": "yellow_leaves", "label": "Yellow Leaves", "icon": "yellow"},
        {"id": "insects", "label": "Insects", "icon": "bug"},
        {"id": "white_fungus", "label": "White Fungus", "icon": "fungus"},
        {"id": "wilting", "label": "Wilting", "icon": "wilt"},
        {"id": "healthy", "label": "Healthy Check", "icon": "leaf"},
    ]
    districts = [
        {"name": "Bengaluru", "lat": 12.97, "lon": 77.59},
        {"name": "Hyderabad", "lat": 17.385, "lon": 78.4867},
        {"name": "Pune", "lat": 18.52, "lon": 73.86},
        {"name": "Nashik", "lat": 19.99, "lon": 73.79},
    ]
    crops = ["grape", "rice", "wheat", "tomato", "potato", "maize", "sugarcane", "ragi"]
    return {"crops": crops, "quick_symptoms": quick, "districts": districts}


# ── Core diagnose endpoint ──────────────────────────────────────────────────────
def _load_image(raw: bytes):
    try:
        from PIL import Image

        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        logger.warning("diagnose: bad image (%s)", e)
        return None


@app.post("/api/v1/diagnose")
async def diagnose(
    image: Optional[UploadFile] = File(default=None),
    text: str = Form(default=""),
    language: str = Form(default="auto"),
    lat: Optional[float] = Form(default=None),
    lon: Optional[float] = Form(default=None),
    offline: bool = Form(default=False),
) -> JSONResponse:
    if image is None and not text.strip():
        raise HTTPException(status_code=400, detail={"status": "error",
                            "code": "EMPTY_INPUT", "message": "Provide an image or text."})

    pil = _load_image(await image.read()) if image is not None else None
    lang = "en" if language == "auto" else language

    kwargs: dict[str, Any] = {
        "image": pil, "user_text": text.strip(), "user_language": lang, "offline": offline,
    }
    if lat is not None:
        kwargs["lat"] = lat
    if lon is not None:
        kwargs["lon"] = lon

    state = _get_pipeline()(**kwargs)

    diag_id = f"diag_{uuid.uuid4().hex[:10]}"
    session_id = f"sess_{uuid.uuid4().hex[:10]}"

    # Best-effort neural voice from the final response (Phase-5 edge-tts).
    if not state.get("voice_output_path") and state.get("final_response"):
        try:
            from voice.tts import synthesize

            out = str(OUTPUT_DIR / f"{diag_id}.mp3")
            if synthesize(state["final_response"], state.get("lang", lang), out):
                state["voice_output_path"] = out
        except Exception as e:
            logger.warning("diagnose: voice synth failed: %s", e)

    return JSONResponse(serialize_diagnosis(state, diag_id, session_id))


# ── Follow-up chat ──────────────────────────────────────────────────────────────
@app.post("/api/v1/chat")
async def chat(payload: dict[str, Any]) -> dict[str, Any]:
    question = (payload.get("question") or "").strip()
    language = payload.get("language", "en")
    history = payload.get("history", [])
    if not question:
        raise HTTPException(status_code=400, detail={"status": "error",
                            "code": "EMPTY_QUESTION", "message": "Question is required."})
    try:
        from utils import genai_handler

        answer = genai_handler.conversational_followup(
            question=question, history=history, crop="", disease="", language=language,
        )
    except Exception as e:
        logger.warning("chat failed: %s", e)
        answer = "Please consult your nearest Krishi Vigyan Kendra (KVK) for detailed help."
    return {"answer": answer, "language": language, "audio_url": None}


# ── Voice transcription ─────────────────────────────────────────────────────────
@app.post("/api/v1/voice/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form(default="auto"),
) -> dict[str, str]:
    from voice.stt import transcribe

    raw = await audio.read()
    result = transcribe(raw, language=language)
    return {"text": result.get("text", ""), "language": result.get("language", "en")}


# ── SSE agent-progress stream (simple, real-shaped) ─────────────────────────────
@app.get("/api/v1/diagnose/stream")
async def diagnose_stream(job: str = "") -> StreamingResponse:
    def _events():
        stages = [
            ("vision", "Analyzing image", 0.3),
            ("knowledge", "Finding treatment", 0.6),
            ("compliance", "Checking safety", 0.85),
        ]
        for stage, msg, prog in stages:
            yield f"event: step\ndata: {json.dumps({'stage': stage, 'message': msg, 'progress': prog})}\n\n"
        yield f"event: done\ndata: {json.dumps({'id': job})}\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


# ── File serving (audio / pdf) ──────────────────────────────────────────────────
@app.get("/api/v1/files/{name}")
def get_file(name: str) -> FileResponse:
    # Prevent path traversal — basename only, must exist under OUTPUT_DIR.
    safe = os.path.basename(name)
    path = OUTPUT_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail={"status": "error",
                            "code": "NOT_FOUND", "message": "File not found."})
    return FileResponse(str(path))
