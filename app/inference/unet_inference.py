import cv2
import torch
import numpy as np

import torchvision.transforms as transforms

from app.models.unet_change_detector import (
    UNetChangeDetector
)


class UNetInference:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = UNetChangeDetector().to(
            self.device
        )

        self.model.load_state_dict(
            torch.load(
                "checkpoints/unet_model.pth",
                map_location=self.device
            )
        )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
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

        prediction = prediction.squeeze()

        prediction = prediction.cpu().numpy()

        prediction = (
            prediction * 255
        ).astype(np.uint8)

        prediction = cv2.resize(
            prediction,
            (
                image1.shape[1],
                image1.shape[0]
            )
        )

        return prediction