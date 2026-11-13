import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

APP_VERSION = "0.1.0"
MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
TARGET_CLASS_NAME = os.getenv("TARGET_CLASS_NAME", "Fruit on Tree").strip()
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="SANTRAZEST Orchard Detection API", version=APP_VERSION)
# Local prototype only. Restrict this list to the real website origin before deployment.
origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

model: YOLO | None = None
model_error: str | None = None


@app.on_event("startup")
def load_model() -> None:
    global model, model_error
    if not MODEL_PATH:
        model_error = "MODEL_PATH is not configured."
        return
    path = Path(MODEL_PATH).expanduser()
    if not path.is_file():
        model_error = f"Model file does not exist: {path}"
        return
    try:
        model = YOLO(str(path))
        model_error = None
    except Exception as exc:
        model_error = f"Model could not be loaded: {type(exc).__name__}: {exc}"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "not_ready",
        "model_loaded": model is not None,
        "model_name": Path(MODEL_PATH).name if MODEL_PATH else None,
        "model_version": APP_VERSION,
        "message": model_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Detection model is not configured or could not be loaded.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG or WEBP image.")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded file could not be decoded as an image.")

    started = time.perf_counter()
    try:
        result = model.predict(image, conf=CONFIDENCE, verbose=False)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}") from exc
    inference_ms = round((time.perf_counter() - started) * 1000, 1)

    names = result.names
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = str(names[class_id])
        if TARGET_CLASS_NAME and class_name.casefold() != TARGET_CLASS_NAME.casefold():
            continue
        x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0].tolist()]
        detections.append({
            "class_name": class_name,
            "confidence": round(float(box.conf.item()), 4),
            "box": [x1, y1, x2, y2],
        })

    return {
        "status": "success",
        "count": len(detections),
        "target_class": TARGET_CLASS_NAME or "all classes",
        "detections": detections,
        "inference_ms": inference_ms,
        "model_name": Path(MODEL_PATH).name,
        "model_version": APP_VERSION,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }
