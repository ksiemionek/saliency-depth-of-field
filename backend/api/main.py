import io
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from backend.pipeline.models import Models, load_models
from backend.pipeline.run import run_pipeline

models: Models | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    models = load_models()
    yield
    models = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": models is not None}


@app.post("/process")
async def process(image: UploadFile):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "The uploaded file is not a valid image.")

    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    img = np.asarray(pil)

    result = run_pipeline(img, models)

    _, png = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    return Response(content=png.tobytes(), media_type="image/png")
