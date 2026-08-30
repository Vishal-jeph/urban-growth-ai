from app.models.unet_change_detector import UNetChangeDetector
from app.models.trainer import train_model
from app.utils.config import load_config

config = load_config()

train_model(
    UNetChangeDetector(),
    checkpoint_name="unet_model.pth",
    config=config
)
