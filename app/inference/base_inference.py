import cv2
import numpy as np
import torch

import torchvision.transforms as transforms

from app.utils.config import load_config

_IMAGE_SIZE = load_config()["model"]["image_size"]


class BaseChangeDetector:
    """Shared device/transform/predict pipeline for the change-detection models."""

    def __init__(self, model, checkpoint_path):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def predict(self, image1, image2):

        image1_tensor = self.transform(
            image1
        ).unsqueeze(0).to(self.device)

        image2_tensor = self.transform(
            image2
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            prediction = self.model(
                image1_tensor,
                image2_tensor
            )

        prediction = prediction.squeeze().cpu().numpy()

        prediction = cv2.resize(
            prediction,
            (image1.shape[1], image1.shape[0])
        )

        prediction = (
            prediction * 255
        ).astype(np.uint8)

        return prediction
