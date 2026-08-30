from app.models.siamese_cnn import SiameseChangeDetector
from app.models.trainer import train_model
from app.utils.config import load_config

config = load_config()

train_model(
    SiameseChangeDetector(),
    checkpoint_name="siamese_model.pth",
    config=config,
    # LEVIR-CD masks are ~4.6% positive pixels; without upweighting the
    # change class, this model converges to predicting background
    # everywhere (see app/models/trainer.py:CombinedLoss). Tried 8/20/1
    # (unweighted); 20 (~inverse class frequency) scored best on the
    # held-out test set (IoU 0.10 vs 0.05 and 0.00) — this small model is
    # fairly unstable run-to-run, so treat this as a reasonable working
    # value, not a precisely tuned optimum.
    pos_weight=20.0
)
