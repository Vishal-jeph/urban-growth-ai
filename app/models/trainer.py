from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.models.dataset import ChangeDetectionDataset


def dice_loss(pred, target, smooth=1):

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    dice = (
        2. * intersection + smooth
    ) / (
        pred.sum() + target.sum() + smooth
    )

    return 1 - dice


class CombinedLoss(nn.Module):

    def __init__(self, pos_weight=1.0):
        super().__init__()

        # Change-detection masks are heavily imbalanced (few change pixels),
        # so plain BCE can converge to "predict background everywhere" and
        # never clear a fixed decision threshold. pos_weight upweights the
        # positive (changed) class to counter that; 1.0 keeps prior behavior.
        self.pos_weight = pos_weight

    def forward(self, pred, target):

        weight = target * self.pos_weight + (1 - target)

        bce_loss = F.binary_cross_entropy(pred, target, weight=weight)

        d_loss = dice_loss(pred, target)

        return bce_loss + d_loss


def train_model(
    model,
    checkpoint_name,
    config,
    data_dir="data/train",
    checkpoint_dir="checkpoints",
    pos_weight=1.0
):
    """Shared training loop for the Siamese and U-Net change detectors."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using Device:", device)

    data_dir = Path(data_dir)

    train_A = sorted((data_dir / "A").glob("*"))
    train_B = sorted((data_dir / "B").glob("*"))
    train_masks = sorted((data_dir / "masks").glob("*"))

    model_cfg = config["model"]

    train_dataset = ChangeDetectionDataset(
        train_A,
        train_B,
        train_masks,
        image_size=model_cfg["image_size"],
        augment=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_cfg["batch_size"],
        shuffle=True
    )

    model = model.to(device)

    criterion = CombinedLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg["learning_rate"]
    )

    epochs = model_cfg["epochs"]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for image1, image2, mask in train_loader:

            image1 = image1.to(device)
            image2 = image2.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            output = model(image1, image2)

            if output.shape[2:] != mask.shape[2:]:
                mask = torch.nn.functional.interpolate(
                    mask,
                    size=output.shape[2:]
                )

            loss = criterion(output, mask)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    Path(checkpoint_dir).mkdir(exist_ok=True)

    checkpoint_path = Path(checkpoint_dir) / checkpoint_name

    torch.save(model.state_dict(), checkpoint_path)

    print(f"Model saved to {checkpoint_path}")

    return model
