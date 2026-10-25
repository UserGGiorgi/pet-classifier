import io
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ARTIFACTS / "pet_resnet18_int8.onnx"))
CLASSES = json.loads((ARTIFACTS / "classes.json").read_text(encoding="utf-8"))

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_BYTES = 10 * 1024 * 1024

# Loaded once at startup, not per request
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

app = FastAPI(title="Pet Breed Classifier", version="1.0")


def preprocess(img: Image.Image) -> np.ndarray:
    """Must match src/data.py get_transforms(train=False): resize 224x224 + ImageNet normalize."""
    img = ImageOps.exif_transpose(img)  # phone photos carry rotation in EXIF metadata
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD  # HWC
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # NCHW, batch of 1


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "num_classes": len(CLASSES)}


@app.post("/classify")
def classify(file: UploadFile = File(...), top_k: int = 5):
    top_k = max(1, min(top_k, len(CLASSES)))
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    t0 = time.perf_counter()
    try:
        x = preprocess(Image.open(io.BytesIO(data)))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="File is not a valid image")
    t1 = time.perf_counter()

    logits = session.run(None, {INPUT_NAME: x})[0][0]
    t2 = time.perf_counter()

    probs = softmax(logits)
    best = np.argsort(probs)[::-1][:top_k]
    return {
        "top_prediction": CLASSES[best[0]],
        "confidence": round(float(probs[best[0]]), 4),
        "predictions": [{"breed": CLASSES[i], "probability": round(float(probs[i]), 4)} for i in best],
        "timing_ms": {"preprocess": round((t1 - t0) * 1000, 1), "inference": round((t2 - t1) * 1000, 1)},
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")