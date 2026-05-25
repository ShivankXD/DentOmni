"""
main.py — DentOmni FastAPI backend
Run: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from .model_loader import get_model, CLASS_NAMES

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dentomni")

# ---------------------------------------------------------------------------
app = FastAPI(
    title="DentOmni API",
    version="1.0.0",
    description="AI-powered dental disease detection — caries & periapical lesions",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Preload models at startup so first request is instant
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def preload_models():
    """Load both models into GPU memory when the server starts."""
    for model_name in ("resnet50", "faster_rcnn"):
        try:
            logger.info("Preloading model: %s ...", model_name)
            get_model(model_name).load()
            logger.info("Preloaded model: %s OK", model_name)
        except Exception as exc:
            logger.error("Failed to preload %s: %s", model_name, exc)


# ---------------------------------------------------------------------------
# Allowed image types
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
MAX_FILE_SIZE = 20 * 1024 * 1024   # 20 MB


# ---------------------------------------------------------------------------
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "DentOmni API", "version": "1.0.0"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy", "classes": CLASS_NAMES}


# ---------------------------------------------------------------------------
@app.post("/predict", tags=["inference"])
async def predict(
    file:  UploadFile = File(..., description="Dental X-ray image"),
    model: Literal["faster_rcnn", "resnet50"] = Form(
        ...,
        description="Model to use: 'faster_rcnn' (Detectron2) or 'resnet50' (classifier)"
    ),
):
    """
    Run dental disease detection on the uploaded X-ray image.

    - **file**: JPEG / PNG / WEBP / BMP / TIFF image (max 20 MB)
    - **model**: `faster_rcnn` → bounding boxes + confidence scores
                 `resnet50`    → classification confidence scores only

    Returns JSON with `caries` and `periapical_lesion` detected/confidence fields,
    optional `bounding_boxes` list, and optional base64 `annotated_image`.
    """
    # ── validate content type ──────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    # ── read & size-check ──────────────────────────────────────────────────
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)/1e6:.1f} MB). Max allowed: 20 MB.",
        )

    # ── decode image ───────────────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    logger.info("Predict | model=%s  image=%dx%d  file=%s",
                model, img.width, img.height, file.filename)

    # ── run inference ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = get_model(model).predict(img)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── format & return ─────────────────────────────────────────────────────
    response = {
        "model_used":          model,
        "inference_time_ms":   elapsed_ms,
        "image_size":          {"width": img.width, "height": img.height},
        "caries":              result["caries"],
        "periapical_lesion":   result["periapical_lesion"],
        "bounding_boxes":      result.get("bounding_boxes", []),
        "annotated_image":     result.get("annotated_image"),  # base64 PNG or None
    }

    logger.info(
        "Result | caries=%.1f%%  lesion=%.1f%%  boxes=%d  time=%sms",
        result["caries"]["confidence"] * 100,
        result["periapical_lesion"]["confidence"] * 100,
        len(result.get("bounding_boxes", [])),
        elapsed_ms,
    )

    return JSONResponse(content=response)
