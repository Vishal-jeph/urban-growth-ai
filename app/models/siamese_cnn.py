import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):

        return self.encoder(x)


class SiameseChangeDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_extractor = FeatureExtractor()

        self.classifier = nn.Sequential(

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image1, image2):

        features1 = self.feature_extractor(image1)

        features2 = self.feature_extractor(image2)

        diff = torch.abs(features1 - features2)

        output = self.classifier(diff)

        return output