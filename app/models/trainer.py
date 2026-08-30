import torch
import torch.nn as nn


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

    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, pred, target):

        bce_loss = self.bce(pred, target)

        d_loss = dice_loss(pred, target)

        return bce_loss + d_loss