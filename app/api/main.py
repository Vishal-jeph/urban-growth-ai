import cv2
import numpy as np

from fastapi import FastAPI, UploadFile

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

@app.post("/predict")
async def predict(
    image1: UploadFile,
    image2: UploadFile
):

    # Read images
    image1_bytes = await image1.read()

    image2_bytes = await image2.read()

    # Convert to OpenCV
    np_image1 = np.frombuffer(
        image1_bytes,
        np.uint8
    )

    np_image2 = np.frombuffer(
        image2_bytes,
        np.uint8
    )

    img1 = cv2.imdecode(
        np_image1,
        cv2.IMREAD_COLOR
    )

    img2 = cv2.imdecode(
        np_image2,
        cv2.IMREAD_COLOR
    )

    # Predict
    prediction = model.predict(
        img1,
        img2
    )

    return {
        "prediction_shape": (
            prediction.shape
        )
    }