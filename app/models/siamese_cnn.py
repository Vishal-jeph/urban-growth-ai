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

        # The encoder downsamples 8x (three stride-2 pools); mirror that
        # with three upsampling stages so the output matches the input
        # resolution instead of a coarse 1/8-scale map naively resized up.
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(

            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image1, image2):

        features1 = self.feature_extractor(image1)

        features2 = self.feature_extractor(image2)

        diff = torch.abs(features1 - features2)

        diff = self.decoder(diff)

        output = self.classifier(diff)

        return output
