from pathlib import Path

from app.inference.base_inference import BaseChangeDetector
from app.models.unet_change_detector import UNetChangeDetector

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "checkpoints" / "unet_model.pth"
)


class UNetInference(BaseChangeDetector):

    def __init__(self):
        super().__init__(UNetChangeDetector(), CHECKPOINT_PATH)
