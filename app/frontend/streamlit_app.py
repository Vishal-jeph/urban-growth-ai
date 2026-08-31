import sys
from datetime import date
from pathlib import Path

# -----------------------------------
# Add Project Root to Python Path
# -----------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# -----------------------------------
# Imports
# -----------------------------------

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from app.utils.config import load_config
from app.utils.logger import setup_logger

from app.preprocessing.image_loader import load_image
from app.preprocessing.preprocess import resize_image
from app.preprocessing.satellite_fetch import (
    fetch_sentinel2_image,
    SatelliteFetchError
)

from app.visualization.compare_view import plot_comparison
from app.visualization.heatmap_view import plot_change_heatmap
from app.visualization.overlay_view import create_overlay

from app.visualization.ai_prediction_view import (
    plot_ai_prediction
)

from app.inference.change_detection import detect_changes

from app.inference.urban_analytics import (
    calculate_change_percentage,
    estimate_vegetation_loss,
    infrastructure_growth_score,
    estimate_density
)

from app.inference.unet_inference import (
    UNetInference
)

from app.visualization.explainability_view import (
    generate_attention_overlay
)

# -----------------------------------
# Load Config + Logger
# -----------------------------------

config = load_config()

logger = setup_logger()

logger.info("Starting Streamlit App")

# -----------------------------------
# Initialize AI Model
# -----------------------------------
# Cached so checkpoints load once per process instead of on every widget
# interaction (Streamlit reruns this whole script on every rerun).


@st.cache_resource(show_spinner="Loading U-Net...")
def load_unet_detector():
    return UNetInference()


unet_detector = load_unet_detector()

# -----------------------------------
# Cached Inference
# -----------------------------------
# Keyed on image content, so re-running the app on the same images doesn't
# recompute every model's forward pass from scratch.


@st.cache_data(show_spinner=False)
def run_classical_diff(image1, image2):
    return detect_changes(image1, image2)


@st.cache_data(show_spinner="Running U-Net inference...")
def run_unet_prediction(_detector, image1, image2):
    return _detector.predict(image1, image2)


@st.cache_data(show_spinner="Fetching satellite imagery...")
def run_satellite_fetch(bbox, year):
    return fetch_sentinel2_image(
        bbox, year, image_size=config["model"]["image_size"]
    )

# -----------------------------------
# Streamlit Config
# -----------------------------------

st.set_page_config(
    page_title=config["streamlit"]["page_title"],
    layout="wide"
)

# -----------------------------------
# App Header
# -----------------------------------

st.title(config["streamlit"]["page_title"])

st.write(
    "AI-powered satellite imagery analysis platform "
    "for urban growth detection."
)

# -----------------------------------
# Sidebar: Image Source
# -----------------------------------

st.sidebar.header("Input Images")

input_mode = st.sidebar.radio(
    "Image source",
    ["Upload your own", "Pick on map"]
)

if input_mode == "Upload your own":

    image1_source = st.sidebar.file_uploader(
        "Historical image",
        type=["png", "jpg", "jpeg"],
        key="upload_image1"
    )

    image2_source = st.sidebar.file_uploader(
        "Recent image",
        type=["png", "jpg", "jpeg"],
        key="upload_image2"
    )

    if image1_source is None or image2_source is None:
        st.info("Upload both a historical and a recent image to begin.")
        st.stop()

    st.sidebar.caption(
        "Models were trained on aerial/satellite building imagery "
        "(LEVIR-CD) — predictions are most meaningful on similar imagery."
    )

else:
    image1_source = image2_source = None

# -----------------------------------
# Load Images
# -----------------------------------

logger.info("Loading satellite images")

if input_mode == "Pick on map":

    # Handled further down, once the user has drawn an area and fetched
    # imagery — the map needs the full-width main area, not the sidebar.
    pass

else:

    try:
        image1 = load_image(image1_source)
        image2 = load_image(image2_source)
    except Exception:
        st.error(
            "Could not read one of the images — please provide valid "
            "PNG/JPG image files."
        )
        st.stop()

    image1 = resize_image(image1)
    image2 = resize_image(image2)

# -----------------------------------
# Pick on Map
# -----------------------------------

if input_mode == "Pick on map":

    st.subheader("Pick an Area and Year Range")

    st.caption(
        "Draw a rectangle, choose two years, then fetch. Uses free "
        "Sentinel-2 imagery (10m/pixel) — this shows regional development "
        "trends, not building-level detail like the other two modes "
        "(the models were trained on 0.5m/pixel imagery)."
    )

    map_col, controls_col = st.columns([2, 1])

    with map_col:

        area_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

        Draw(
            export=False,
            draw_options={
                "rectangle": True,
                "polygon": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "polyline": False
            },
            edit_options={"edit": False}
        ).add_to(area_map)

        map_state = st_folium(
            area_map, height=350, width=550, key="area_map"
        )

    with controls_col:

        current_year = date.today().year

        start_year = st.number_input(
            "Historical year",
            min_value=2017,
            max_value=current_year,
            value=2018
        )

        end_year = st.number_input(
            "Recent year",
            min_value=2017,
            max_value=current_year,
            value=current_year
        )

        if end_year <= start_year:
            st.warning("Pick a recent year later than the historical year.")

        fetch_clicked = st.button(
            "Fetch satellite imagery", width='stretch'
        )

    drawing = map_state.get("last_active_drawing") if map_state else None

    if drawing is None:
        st.info("Draw a rectangle on the map to select an area.")
        st.stop()

    coords = drawing["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    bbox = (min(lons), min(lats), max(lons), max(lats))

    if (bbox[2] - bbox[0] > 1.0) or (bbox[3] - bbox[1] > 1.0):
        st.warning(
            "That area is quite large — the fetched image is a single "
            "downsampled crop of it, so detail will be limited. Consider "
            "drawing a smaller box."
        )

    if fetch_clicked:
        st.session_state["map_bbox"] = bbox
        st.session_state["map_start_year"] = start_year
        st.session_state["map_end_year"] = end_year

    if "map_bbox" not in st.session_state:
        st.info(
            "Draw an area and click \"Fetch satellite imagery\" to "
            "continue."
        )
        st.stop()

    try:
        image1, item1 = run_satellite_fetch(
            st.session_state["map_bbox"], st.session_state["map_start_year"]
        )
        image2, item2 = run_satellite_fetch(
            st.session_state["map_bbox"], st.session_state["map_end_year"]
        )
    except SatelliteFetchError as exc:
        st.error(str(exc))
        st.stop()

    st.caption(
        f"Historical: {item1.properties['datetime'][:10]} "
        f"(cloud cover {item1.properties['eo:cloud_cover']:.1f}%) — "
        f"Recent: {item2.properties['datetime'][:10]} "
        f"(cloud cover {item2.properties['eo:cloud_cover']:.1f}%)"
    )

    image1 = resize_image(image1)
    image2 = resize_image(image2)

# -----------------------------------
# Satellite Comparison
# -----------------------------------

st.subheader("Satellite Image Comparison")

comparison_fig = plot_comparison(
    image1,
    image2
)

st.pyplot(comparison_fig)

# -----------------------------------
# Classical Change Detection
# -----------------------------------

logger.info("Running classical CV change detection")

diff_map, cleaned_map = run_classical_diff(
    image1,
    image2
)

overlay_image = create_overlay(
    image2,
    cleaned_map,
    alpha=config["visualization"]["heatmap_opacity"]
)

st.subheader("Classical Change Detection")

st.caption(
    "A simple, non-AI baseline: raw pixel differences between the two "
    "images, cleaned up and highlighted."
)

classical_col1, classical_col2, classical_col3 = st.columns(3)

with classical_col1:
    st.pyplot(plot_change_heatmap(diff_map))
    st.caption("Raw difference heatmap")

with classical_col2:
    st.image(
        cleaned_map,
        caption="Detected changes (cleaned up)",
        width='stretch'
    )

with classical_col3:
    st.image(
        overlay_image,
        caption="Highlighted on recent image",
        width='stretch'
    )

# -----------------------------------
# AI-Based Prediction (U-Net)
# -----------------------------------

st.subheader("AI-Based Urban Change Prediction")

st.caption(
    "U-Net, trained from scratch on LEVIR-CD (445 pairs) — test-set IoU "
    "0.44. See the plain-language summary below for what that means."
)

logger.info("Running AI inference")

unet_prediction = run_unet_prediction(
    unet_detector,
    image1,
    image2
)

attention_overlay = generate_attention_overlay(
    image2,
    unet_prediction,
    alpha=config["visualization"]["heatmap_opacity"]
)

unet_col1, unet_col2, unet_col3 = st.columns(3)

with unet_col1:
    st.pyplot(plot_ai_prediction(unet_prediction))
    st.caption("Confidence heatmap")

with unet_col2:
    st.image(
        unet_prediction,
        caption="Segmentation prediction",
        width='stretch'
    )

with unet_col3:
    st.image(
        attention_overlay,
        caption="Attention overlay",
        width='stretch'
    )

# -----------------------------------
# Urban Analytics
# -----------------------------------

st.subheader("Urban Intelligence Analytics")

change_percentage = calculate_change_percentage(
    cleaned_map
)

vegetation_loss = estimate_vegetation_loss(
    image1,
    image2
)

growth_score = infrastructure_growth_score(
    change_percentage
)

density_score = estimate_density(
    cleaned_map
)

analytics_col1, analytics_col2 = st.columns(2)

with analytics_col1:

    st.metric(
        "Urban Change %",
        f"{change_percentage}%"
    )

    st.metric(
        "Vegetation Reduction",
        f"{vegetation_loss}"
    )

with analytics_col2:

    st.metric(
        "Infrastructure Growth Score",
        f"{growth_score}/100"
    )

    st.metric(
        "Development Density",
        f"{density_score}"
    )

# -----------------------------------
# Plain-Language Summary
# -----------------------------------

st.divider()

st.subheader("📋 What Does This Mean?")

if change_percentage < 5:
    level_icon = "🟢"
    level_text = "very little change"
    level_detail = (
        "The two images look largely the same — no notable development "
        "was detected in this area."
    )
elif change_percentage < 15:
    level_icon = "🟡"
    level_text = "a moderate amount of change"
    level_detail = (
        "Some parts of this area look different between the two images — "
        "likely a mix of new construction, roadwork, or land clearing."
    )
else:
    level_icon = "🔴"
    level_text = "a significant amount of change"
    level_detail = (
        "Large parts of this area look different between the two images "
        "— likely substantial new construction or land-use change."
    )

st.info(
    f"{level_icon} **Overall: {level_text}.** About "
    f"**{change_percentage}%** of the area appears different between the "
    f"historical and recent image. {level_detail}"
)

if vegetation_loss > 5:
    st.info(
        "🌳 **Greenery appears to have decreased** in the more recent "
        "image — a common sign that trees, farmland, or open land is "
        "being cleared, often to make way for construction."
    )
else:
    st.info(
        "🌳 **No meaningful loss of greenery** was detected between the "
        "two images."
    )

st.warning(
    "🤖 **How much should you trust this?** In testing on real satellite "
    "imagery the AI model had never seen before, it correctly identified "
    "genuine changes about **44 out of 100 times** (in ML terms, a "
    "test-set IoU of 0.44). Treat every result on this page as a "
    "**helpful pointer toward areas worth a closer look** — not a "
    "certain, verified fact."
)

# -----------------------------------
# Footer Logging
# -----------------------------------

logger.info(
    "Urban analysis pipeline executed successfully"
)