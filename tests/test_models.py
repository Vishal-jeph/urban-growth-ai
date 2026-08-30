import torch
from PIL import Image

from app.models.dataset import ChangeDetectionDataset
from app.models.siamese_cnn import SiameseChangeDetector
from app.models.unet_change_detector import UNetChangeDetector


def test_siamese_forward_shape():
    model = SiameseChangeDetector()

    image1 = torch.randn(2, 3, 256, 256)
    image2 = torch.randn(2, 3, 256, 256)

    output = model(image1, image2)

    assert output.shape == (2, 1, 256, 256)
    assert torch.all((output >= 0) & (output <= 1))


def test_unet_forward_shape():
    model = UNetChangeDetector()

    image1 = torch.randn(2, 3, 256, 256)
    image2 = torch.randn(2, 3, 256, 256)

    output = model(image1, image2)

    assert output.shape == (2, 1, 256, 256)
    assert torch.all((output >= 0) & (output <= 1))


def test_dataset_augmentation_keeps_pair_aligned(tmp_path):
    # image1, image2, and the mask must undergo the *same* flip/rotation,
    # otherwise the mask silently desyncs from the images it labels.
    size = 32

    marker = Image.new("L", (size, size), 0)

    for x in range(size // 4):
        for y in range(size // 4):
            marker.putpixel((x, y), 255)

    image_rgb = Image.merge("RGB", (marker, marker, marker))

    image1_path = tmp_path / "img1.png"
    image2_path = tmp_path / "img2.png"
    mask_path = tmp_path / "mask.png"

    image_rgb.save(image1_path)
    image_rgb.save(image2_path)
    marker.save(mask_path)

    dataset = ChangeDetectionDataset(
        [image1_path], [image2_path], [mask_path],
        image_size=size, augment=True
    )

    for _ in range(20):
        image1, image2, mask = dataset[0]

        mask_marker = mask[0] > 0.5

        assert torch.equal(image1[0] > 0.5, mask_marker)
        assert torch.equal(image2[0] > 0.5, mask_marker)
