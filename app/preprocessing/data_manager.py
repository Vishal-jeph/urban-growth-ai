from pathlib import Path


def get_sample_images():

    sample_dir = Path("data/samples")

    image_files = list(sample_dir.glob("*"))

    return image_files