from pathlib import Path

from app.inference.base_inference import BaseChangeDetector
from app.models.siamese_cnn import SiameseChangeDetector

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "checkpoints" / "siamese_model.pth"
)


class AIChangeDetector(BaseChangeDetector):

    def __init__(self):
        super().__init__(SiameseChangeDetector(), CHECKPOINT_PATH)
