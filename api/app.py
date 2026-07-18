"""
AgriBloom HTTP API.

This is a thin production-facing bridge over the existing advanced agent
pipeline. It intentionally returns the structured farmer_advice contract so a
modern frontend does not need to parse long generated text.
"""
from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web_app"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
SUPPORTED_LANGUAGES = {"en", "hi", "kn", "te", "ta", "pa", "gu", "mr", "bn", "or"}


def _safe_float(value: str | float | int | None, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_language(language: str | None) -> str:
    code = (language or "en").strip().lower()
    return code if code in SUPPORTED_LANGUAGES else "en"


async def _read_crop_image(upload: UploadFile | None) -> Image.Image | None:
    if upload is None:
        return None

    data = await upload.read()
    if not data:
        return None
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Please upload a photo under 8 MB.",
        )

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read the image. Please upload a clear JPG or PNG photo.",
        ) from exc


def create_app() -> FastAPI:
    app = FastAPI(
        title="AgriBloom API",
        version="1.0.0",
        description="Farmer-facing API bridge for crop health advice.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        providers: list[str] = []
        try:
            from utils.genai_handler import get_configured_llm_providers

            providers = get_configured_llm_providers(check_ollama=False)
        except Exception as exc:
            logger.warning("Provider status unavailable: %s", exc)

        return {
            "status": "ok",
            "service": "agribloom-api",
            "farmer_advice_schema": "1.0",
            "configured_llm_providers": providers,
        }

    @app.post("/api/analyze")
    async def analyze(
        image: UploadFile | None = File(default=None),
        text: str = Form(default=""),
        language: str = Form(default="en"),
        state: str = Form(default=""),
        district: str = Form(default=""),
        offline: bool = Form(default=False),
        lat: str | None = Form(default=None),
        lon: str | None = Form(default=None),
    ) -> JSONResponse:
        crop_image = await _read_crop_image(image)
        clean_text = (text or "").strip()
        if crop_image is None and not clean_text:
            raise HTTPException(
                status_code=400,
                detail="Please add a crop photo or describe the crop problem.",
            )

        lang_code = _normalize_language(language)
        latitude = _safe_float(lat, 14.4644)
        longitude = _safe_float(lon, 75.9218)

        def _run_pipeline() -> dict[str, Any]:
            from main import run_pipeline

            return run_pipeline(
                image=crop_image,
                image_path="",
                user_text=clean_text,
                user_language=lang_code,
                lang=lang_code,
                offline=bool(offline),
                lat=latitude,
                lon=longitude,
            )

        result = await asyncio.to_thread(_run_pipeline)
        advice = result.get("farmer_advice")

        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("error") or "Crop analysis failed.",
            )

        payload = {
            "status": result.get("status", "complete"),
            "farmer_advice": advice,
            "final_response": result.get("final_response", ""),
            "voice_output_path": result.get("voice_output_path"),
            "audit_pdf_path": result.get("audit_pdf_path"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "request": {
                "language": lang_code,
                "state": state,
                "district": district,
                "offline": bool(offline),
                "lat": latitude,
                "lon": longitude,
            },
        }
        return JSONResponse(content=jsonable_encoder(payload))

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


app = create_app()

