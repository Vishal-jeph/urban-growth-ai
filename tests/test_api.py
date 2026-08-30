from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
SAMPLE_IMAGES = sorted(SAMPLE_DIR.glob("*"))


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Urban Growth AI API Running"}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_png():
    image1_path, image2_path = SAMPLE_IMAGES[0], SAMPLE_IMAGES[1]

    with open(image1_path, "rb") as image1, open(image2_path, "rb") as image2:
        response = client.post(
            "/predict",
            files={
                "image1": (image1_path.name, image1, "image/png"),
                "image2": (image2_path.name, image2, "image/png"),
            }
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_predict_rejects_invalid_image():
    bad_file = b"this is not an image"

    response = client.post(
        "/predict",
        files={
            "image1": ("bad.png", bad_file, "image/png"),
            "image2": ("bad.png", bad_file, "image/png"),
        }
    )

    assert response.status_code == 400
