from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class ChangeDetectionDataset(Dataset):

    def __init__(
        self,
        image1_paths,
        image2_paths,
        mask_paths
    ):

        self.image1_paths = image1_paths
        self.image2_paths = image2_paths
        self.mask_paths = mask_paths

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.image1_paths)

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

        return image1, image2, mask