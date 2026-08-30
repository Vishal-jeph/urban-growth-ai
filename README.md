# Urban Growth AI

Satellite-imagery change detection: compare two images of the same location
taken at different times and highlight urban development between them.

The app runs a classical CV diff (grayscale threshold + morphology) alongside
two learned models — a lightweight Siamese CNN and a U-Net — and reports
change heatmaps, overlays, an explainability view, and simple urban-growth
analytics.

## Project layout

- `app/models/` — Siamese CNN and U-Net architectures, dataset, training loop
- `app/inference/` — inference wrappers, classical CV detector, metrics
- `app/preprocessing/` — image loading/resizing, sample-data discovery
- `app/visualization/` — matplotlib/OpenCV plotting helpers
- `app/frontend/streamlit_app.py` — the Streamlit UI (primary product)
- `app/api/main.py` — a standalone FastAPI `/predict` endpoint (U-Net only)
- `train.py` / `train_unet.py` — training entry points
- `configs/config.yaml` — model/paths/visualization settings

The Streamlit app and the FastAPI service are independent entry points — the
UI calls the models in-process and does not go through the API.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For running tests, use `requirements-dev.txt` instead (it includes
`requirements.txt`):

```bash
pip install -r requirements-dev.txt
```

## Running

Streamlit UI:

```bash
streamlit run app/frontend/streamlit_app.py
```

FastAPI service:

```bash
uvicorn app.api.main:app --reload
```

Both together (also what the Docker image runs):

```bash
./start.sh
```

## Testing

```bash
pytest
```

## Training

Training/validation data lives under `data/train/{A,B,masks}` and
`data/val/{A,B,masks}` — these are gitignored (third-party, academic-use-only
imagery) and fetched locally instead of committed. Hyperparameters come from
`configs/config.yaml` (`model.*`).

Fetch the [LEVIR-CD](https://justchenhao.github.io/LEVIR/) building
change-detection dataset (445 train / 64 val / 128 test pairs, 1024x1024,
0.5m/pixel, academic use only):

```bash
python scripts/prepare_levir_cd.py               # train + val
python scripts/prepare_levir_cd.py --splits test  # held-out test set
```

Then train:

```bash
python train.py        # Siamese CNN -> checkpoints/siamese_model.pth
python train_unet.py   # U-Net       -> checkpoints/unet_model.pth
```

On CPU, the Siamese CNN trains in a few minutes for the full 20 configured
epochs; the U-Net is heavier (~7-8 minutes/epoch on a single CPU core) — cut
`model.epochs` in `configs/config.yaml` for a faster turnaround, or pass an
override via `train_model(..., config={**config, "model": {**config["model"], "epochs": N}})`.

### Results

Evaluated on the held-out LEVIR-CD test split (128 pairs, never seen during
training), thresholding predictions at 127/255:

| Model | Epochs | IoU | Precision | Recall |
|---|---|---|---|---|
| U-Net | 35 | 0.439 | 0.571 | 0.571 |
| Siamese CNN | 20 | 0.101 | 0.110 | 0.570 |

The Siamese CNN's original architecture downsampled 8x with no decoder,
naively upscaling a coarse 32x32 map — that alone capped it well below the
U-Net. Adding a matching decoder (`app/models/siamese_cnn.py`) wasn't
sufficient on its own: with LEVIR-CD's ~4.6% positive-pixel imbalance, plain
BCE+Dice converged to predicting "no change" everywhere (IoU 0.0). Upweighting
the change class in the loss (`pos_weight` in `CombinedLoss`, tuned against
the measured class ratio) fixed the collapse, landing at the numbers above —
still well behind the U-Net, which remains the model worth using in practice.

For reference, published LEVIR-CD benchmarks with longer training,
augmentation, and stronger backbones reach ~0.80-0.85 IoU; these are
from-scratch, short training runs on a single CPU core, not tuned to match
that.

## Docker

```bash
docker build -t urban-growth-ai .
docker run -p 8501:8501 -p 8000:8000 urban-growth-ai
```
