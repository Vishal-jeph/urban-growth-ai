from pathlib import Path

from app.utils.config import load_config


def get_sample_images():

    config = load_config()

    sample_dir = Path(config["paths"]["sample_data"])

    image_files = list(sample_dir.glob("*"))

    return image_files
