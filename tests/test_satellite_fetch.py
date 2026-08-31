from app.preprocessing.satellite_fetch import fetch_sentinel2_image

# Small bbox over Bangalore, India — matches the existing sample imagery.
BANGALORE_BBOX = (77.55, 12.90, 77.65, 13.00)


def test_fetch_sentinel2_image_returns_valid_rgb_array():
    # Hits the live Planetary Computer STAC API — needs network access.
    image, item = fetch_sentinel2_image(BANGALORE_BBOX, 2020, image_size=128)

    assert image.shape == (128, 128, 3)
    assert image.dtype.name == "uint8"
    assert item.properties["eo:cloud_cover"] < 100
