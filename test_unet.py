import torch

from app.models.unet_change_detector import (
    UNetChangeDetector
)

model = UNetChangeDetector()

image1 = torch.randn(
    2,
    3,
    256,
    256
)

image2 = torch.randn(
    2,
    3,
    256,
    256
)

output = model(
    image1,
    image2
)

print(output.shape)