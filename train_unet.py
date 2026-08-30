from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.models.unet_change_detector import (
    UNetChangeDetector
)

from app.models.dataset import (
    ChangeDetectionDataset
)

from app.models.trainer import (
    CombinedLoss
)

# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# -----------------------------------
# Dataset Paths
# -----------------------------------

train_A = sorted(
    Path("data/train/A").glob("*")
)

train_B = sorted(
    Path("data/train/B").glob("*")
)

train_masks = sorted(
    Path("data/train/masks").glob("*")
)

# -----------------------------------
# Dataset
# -----------------------------------

train_dataset = ChangeDetectionDataset(
    train_A,
    train_B,
    train_masks
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

# -----------------------------------
# Model
# -----------------------------------

model = UNetChangeDetector().to(device)

# -----------------------------------
# Loss
# -----------------------------------

criterion = CombinedLoss()

# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------------
# Training Loop
# -----------------------------------

epochs = 3

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for image1, image2, mask in train_loader:

        image1 = image1.to(device)

        image2 = image2.to(device)

        mask = mask.to(device)

        optimizer.zero_grad()

        output = model(
            image1,
            image2
        )

        loss = criterion(
            output,
            mask
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {epoch_loss:.4f}"
    )

# -----------------------------------
# Save Model
# -----------------------------------

Path("checkpoints").mkdir(
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "checkpoints/unet_model.pth"
)

print("UNet model saved successfully.")