
from pathlib import Path
import os
import io
import logging

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile
)

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image, UnidentifiedImageError

from app.xai import XRayExplainer


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

STATIC_DIR = BASE_DIR / "app" / "static"

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "chest_xray.keras")
    )
).expanduser().resolve()

MAX_IMAGE_SIZE = 12 * 1024 * 1024


# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
)

logger = logging.getLogger("PulmoVision")


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="PulmoVision XAI",
    description="Chest X-ray classification with Grad-CAM.",
    version="2.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# ==========================================================
# GLOBAL MODEL STATE
# ==========================================================

explainer = None
model_error = None


# ==========================================================
# MODEL INITIALIZATION
# ==========================================================

@app.on_event("startup")
def load_model():

    global explainer, model_error

    logger.info("======================================")
    logger.info("STARTING PULMOVISION")
    logger.info("======================================")

    logger.info("Model path: %s", MODEL_PATH)

    if not MODEL_PATH.exists():

        model_error = (
            f"Model file not found: {MODEL_PATH}"
        )

        logger.error(model_error)

        return

    try:

        logger.info("Initializing XRayExplainer...")

        explainer = XRayExplainer(
            model_path=str(MODEL_PATH),
            image_size=(128, 128)
        )

        model_error = None

        logger.info("MODEL LOADED SUCCESSFULLY")

    except Exception as exc:

        explainer = None

        model_error = (
            f"{type(exc).__name__}: {exc}"
        )

        logger.exception(
            "MODEL INITIALIZATION FAILED"
        )

    logger.info("======================================")


# ==========================================================
# HOME PAGE
# ==========================================================

@app.get("/", response_class=HTMLResponse)
def home():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=500,
            detail="Frontend index.html was not found."
        )

    return index_file.read_text(
        encoding="utf-8"
    )


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/api/health")
def health():

    return {
        "application": "PulmoVision XAI",

        "status": (
            "ok"
            if explainer is not None
            else "model_unavailable"
        ),

        "model_loaded": explainer is not None,

        "model_exists": MODEL_PATH.exists(),

        "message": (
            "Model ready"
            if explainer is not None
            else model_error
        ),

        "gradcam_layer": (
            explainer.last_conv_name
            if explainer is not None
            else None
        )
    }


# ==========================================================
# PREDICTION API
# ==========================================================

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):

    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {model_error}"
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select an image."
        )

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(raw) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 12 MB."
        )

    try:
        # Read image using its actual contents
        image_buffer = io.BytesIO(raw)

        with Image.open(image_buffer) as source:
            actual_format = source.format

            if actual_format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported image format: {actual_format}"
                )

            source.load()
            image = source.convert("RGB")

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please upload JPG, PNG, or WEBP."
        )

    try:
        logger.info(
            "Received file: %s | MIME: %s | Format: %s | Size: %s bytes",
            file.filename,
            file.content_type,
            actual_format,
            len(raw)
        )

        result = explainer.predict(image)

        return result

    except Exception as exc:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(exc).__name__}: {exc}"
        )



# ==========================================================
# APPLICATION INFO
# ==========================================================

@app.get("/api")
def api_info():

    return {
        "name": "PulmoVision XAI",
        "version": "2.0.0",
        "description": (
            "Chest X-ray classification and Grad-CAM visualization"
        ),
        "endpoints": {
            "home": "/",
            "health": "/api/health",
            "predict": "/api/predict",
            "docs": "/docs"
        }
    }
