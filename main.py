import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image

from detector import load_model, predict_image


# ── Lifespan: runs once when server starts ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-loads the model at server startup so the first API request
    is not slow. Model stays in memory for all future requests.
    """
    print("[STARTUP] Warming up model...")
    load_model()
    print("[STARTUP] Server is ready to accept requests.")
    yield
    print("[SHUTDOWN] Server shutting down.")


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Image Detector",
    description="""
    ## Deepfake Image Detection API
    Upload any face image and get back a **REAL** or **FAKE** prediction.

    ### How it works
    - Uses a **Vision Transformer (ViT)** model fine-tuned on 56,000+ real/fake face images
    - Model: `prithivMLmods/Deep-Fake-Detector-v2-Model` (free, from HuggingFace)
    - Returns confidence score and per-class probabilities

    ### Supported formats
    - JPG / JPEG
    - PNG
    - WEBP
    - Max file size: 10MB
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed image MIME types
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """Basic health check — confirms server is running."""
    return {
        "status": "running",
        "message": "Deepfake Detector API is live.",
        "tip": "Go to /docs to test the API in your browser.",
    }


@app.get("/health", tags=["General"])
def health():
    """Health check endpoint for monitoring / deployment checks."""
    return {"status": "ok"}


@app.post("/detect", tags=["Detection"])
async def detect_deepfake(
    file: UploadFile = File(..., description="Face image file (JPG, PNG, WEBP)")
):
    """
    ## Detect if an image is a deepfake

    **Steps:**
    1. Upload a face image
    2. The ViT model analyzes it
    3. You get back: REAL or FAKE + confidence score

    **Returns:**
```json
    {
        "result": "FAKE",
        "confidence": 0.9821,
        "confidence_percent": "98.21%",
        "probabilities": {
            "Deepfake": 0.9821,
            "Realism": 0.0179
        },
        "model_used": "prithivMLmods/Deep-Fake-Detector-v2-Model",
        "device": "cpu"
    }
```
    """

    # ── Step 1: Validate file type ─────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Please upload a JPG, PNG, or WEBP image."
            ),
        )

    # ── Step 2: Read file bytes ────────────────────────────────────────────
    try:
        file_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # ── Step 3: Check file size ────────────────────────────────────────────
    if len(file_bytes) > MAX_SIZE_BYTES:
        size_mb = round(len(file_bytes) / (1024 * 1024), 2)
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb}MB). Maximum allowed size is 10MB.",
        )

    # ── Step 4: Open as PIL image ──────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open image. Make sure it's a valid image file.",
        )

    # ── Step 5: Run prediction ─────────────────────────────────────────────
    try:
        result = predict_image(image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}",
        )

    return result


# ── Run directly with: python main.py ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)