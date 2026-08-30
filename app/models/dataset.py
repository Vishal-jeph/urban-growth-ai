import random

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

from app.utils.config import load_config

_DEFAULT_IMAGE_SIZE = load_config()["model"]["image_size"]


class ChangeDetectionDataset(Dataset):

    def __init__(
        self,
        image1_paths,
        image2_paths,
        mask_paths,
        image_size=_DEFAULT_IMAGE_SIZE,
        augment=False
    ):

        self.image1_paths = image1_paths
        self.image2_paths = image2_paths
        self.mask_paths = mask_paths
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.image1_paths)

    def _augment(self, image1, image2, mask):
        # Flips/rotations applied identically to image1, image2, and mask so
        # the change mask stays spatially aligned with both images.

        if random.random() < 0.5:
            image1 = torch.flip(image1, dims=[2])
            image2 = torch.flip(image2, dims=[2])
            mask = torch.flip(mask, dims=[2])

        if random.random() < 0.5:
            image1 = torch.flip(image1, dims=[1])
            image2 = torch.flip(image2, dims=[1])
            mask = torch.flip(mask, dims=[1])

        k = random.randint(0, 3)

        if k:
            image1 = torch.rot90(image1, k, dims=[1, 2])
            image2 = torch.rot90(image2, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        return image1, image2, mask

    def __getitem__(self, idx):

        image1 = Image.open(
            self.image1_paths[idx]
        ).convert("RGB")

        image2 = Image.open(
            self.image2_paths[idx]
        ).convert("RGB")

        mask = Image.open(
            self.mask_paths[idx]
        ).convert("L")

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        mask = self.transform(mask)

        mask = (mask > 0.5).float()

        if self.augment:
            image1, image2, mask = self._augment(image1, image2, mask)

        return image1, image2, mask
