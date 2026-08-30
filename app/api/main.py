import io

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.inference.unet_inference import (
    UNetInference
)

app = FastAPI(
    title="Urban Growth AI API"
)

model = UNetInference()


@app.get("/")
def home():

    return {
        "message": (
            "Urban Growth AI API Running"
        )
    }


@app.get("/health")
def health():

    return {"status": "ok"}


async def _read_upload_as_rgb(upload: UploadFile) -> np.ndarray:

    raw_bytes = await upload.read()

    np_buffer = np.frombuffer(raw_bytes, np.uint8)

    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode '{upload.filename}' as an image."
        )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@app.post("/predict")
async def predict(
    image1: UploadFile,
    image2: UploadFile
):

    # Models are trained on RGB arrays (see app/preprocessing/image_loader.py),
    # so uploads decoded by OpenCV (BGR) must be converted before inference.
    img1 = await _read_upload_as_rgb(image1)
    img2 = await _read_upload_as_rgb(image2)

    prediction = model.predict(img1, img2)

    success, encoded = cv2.imencode(".png", prediction)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode prediction as PNG."
        )

    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/png"
    )
