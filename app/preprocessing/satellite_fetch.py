"""Fetch real satellite imagery for a map area + year via Microsoft's
Planetary Computer (Sentinel-2 L2A) — free, keyless STAC API.

Note: Sentinel-2 is 10m/pixel, far coarser than the 0.5m/pixel imagery the
change-detection models were trained on (LEVIR-CD). Individual buildings
are often smaller than a single pixel here, so predictions on this imagery
read as regional development trends, not building-level change detection.
"""
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Widen the cloud-cover filter progressively rather than failing outright
# just because the cleanest scene for a given year is a bit cloudy.
CLOUD_COVER_STEPS = (20, 50, 80, 100)


class SatelliteFetchError(Exception):
    pass


def _search_best_item(bbox, year):
    catalog = pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace
    )

    for max_cloud_cover in CLOUD_COVER_STEPS:

        search = catalog.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}}
        )

        items = list(search.item_collection())

        if items:
            items.sort(key=lambda item: item.properties["eo:cloud_cover"])
            return items[0]

    return None


def fetch_sentinel2_image(bbox, year, image_size=256):
    """
    bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84 (EPSG:4326).
    year: calendar year to search within.

    Returns (image, item): image is an (image_size, image_size, 3) uint8
    RGB array; item is the STAC item used, for surfacing its acquisition
    date/cloud cover to the user.

    Raises SatelliteFetchError if no usable scene is found.
    """
    try:
        item = _search_best_item(bbox, year)
    except Exception as exc:
        raise SatelliteFetchError(
            f"Could not search satellite imagery: {exc}"
        ) from exc

    if item is None:
        raise SatelliteFetchError(
            f"No Sentinel-2 imagery found for {year} over this area. "
            "Try a different year (global Sentinel-2 coverage starts "
            "around 2017) or draw a different area."
        )

    try:
        href = item.assets["visual"].href

        with rasterio.open(href) as dataset:

            warped_bbox = transform_bounds(
                "EPSG:4326", dataset.crs, *bbox
            )

            window = from_bounds(*warped_bbox, transform=dataset.transform)

            if window.width <= 0 or window.height <= 0:
                raise SatelliteFetchError(
                    "The selected area falls outside this scene's coverage."
                )

            data = dataset.read(
                [1, 2, 3],
                window=window,
                out_shape=(3, image_size, image_size),
                resampling=Resampling.bilinear
            )

    except SatelliteFetchError:
        raise
    except Exception as exc:
        raise SatelliteFetchError(
            f"Could not download satellite imagery: {exc}"
        ) from exc

    image = np.transpose(data, (1, 2, 0))

    return image, item
