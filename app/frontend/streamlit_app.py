import sys
from pathlib import Path

# -----------------------------------
# Add Project Root to Python Path
# -----------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# -----------------------------------
# Imports
# -----------------------------------

import streamlit as st
import numpy as np

from app.utils.config import load_config
from app.utils.logger import setup_logger

from app.preprocessing.image_loader import load_image
from app.preprocessing.preprocess import resize_image
from app.preprocessing.data_manager import get_sample_images

from app.visualization.compare_view import plot_comparison
from app.visualization.heatmap_view import plot_change_heatmap
from app.visualization.overlay_view import create_overlay

from app.visualization.ai_prediction_view import (
    plot_ai_prediction
)

from app.inference.change_detection import detect_changes

from app.inference.model_inference import (
    AIChangeDetector
)

from app.inference.urban_analytics import (
    calculate_change_percentage,
    estimate_vegetation_loss,
    infrastructure_growth_score,
    estimate_density
)

from app.inference.evaluation_metrics import (
    calculate_iou,
    calculate_precision,
    calculate_recall
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

ai_detector = AIChangeDetector()
unet_detector = UNetInference()

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
# Sidebar
# -----------------------------------

st.sidebar.header("Sample Satellite Dataset")

sample_images = get_sample_images()

image_names = [img.name for img in sample_images]

if len(image_names) < 2:

    st.error(
        "Please add at least 2 images inside data/samples/"
    )

    st.stop()

selected_image1 = st.sidebar.selectbox(
    "Select Historical Image",
    image_names
)

selected_image2 = st.sidebar.selectbox(
    "Select Recent Image",
    image_names,
    index=min(1, len(image_names) - 1)
)

# -----------------------------------
# Image Paths
# -----------------------------------

image1_file = Path(
    "data/samples"
) / selected_image1

image2_file = Path(
    "data/samples"
) / selected_image2

# -----------------------------------
# Load Images
# -----------------------------------

logger.info("Loading satellite images")

image1 = load_image(image1_file)

image2 = load_image(image2_file)

# -----------------------------------
# Resize Images
# -----------------------------------

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

diff_map, cleaned_map = detect_changes(
    image1,
    image2
)

# -----------------------------------
# Heatmap Visualization
# -----------------------------------

st.subheader("Urban Growth Heatmap")

heatmap_fig = plot_change_heatmap(
    diff_map
)

st.pyplot(heatmap_fig)

# -----------------------------------
# Binary Change Regions
# -----------------------------------

st.subheader("Detected Development Regions")

st.image(
    cleaned_map,
    caption="Detected Urban Changes",
    use_container_width=True
)

# -----------------------------------
# Overlay Visualization
# -----------------------------------

st.subheader("Urban Growth Overlay Visualization")

overlay_image = create_overlay(
    image2,
    cleaned_map
)

st.image(
    overlay_image,
    caption="Highlighted Development Regions",
    use_container_width=True
)

# -----------------------------------
# AI-Based Prediction
# -----------------------------------

st.subheader("AI-Based Urban Change Prediction")

logger.info("Running AI inference")

# Interactive threshold slider
threshold = st.slider(
    "AI Detection Threshold",
    min_value=0,
    max_value=255,
    value=127
)

# Run prediction
ai_prediction = ai_detector.predict(
    image1,
    image2
)

# Create thresholded prediction
thresholded_prediction = (
    ai_prediction > threshold
).astype(np.uint8) * 255

# -----------------------------------
# U-Net Prediction
# -----------------------------------

st.subheader("U-Net Urban Change Prediction")

unet_prediction = unet_detector.predict(
    image1,
    image2
)

unet_fig = plot_ai_prediction(
    unet_prediction
)

st.pyplot(unet_fig)

st.image(
    unet_prediction,
    caption="U-Net Segmentation Prediction",
    use_container_width=True
)

# -----------------------------------
# Explainable AI Visualization
# -----------------------------------

st.subheader("Explainable AI Attention Visualization")

attention_overlay = generate_attention_overlay(
    image2,
    unet_prediction
)

st.image(
    attention_overlay,
    caption="AI Attention Overlay",
    use_container_width=True
)

# -----------------------------------
# AI Confidence Heatmap
# -----------------------------------

ai_fig = plot_ai_prediction(
    ai_prediction
)

st.pyplot(ai_fig)

# -----------------------------------
# Thresholded Prediction
# -----------------------------------

st.image(
    thresholded_prediction,
    caption="Thresholded AI Prediction",
    use_container_width=True
)

# -----------------------------------
# AI Evaluation Metrics
# -----------------------------------

st.subheader("AI Evaluation Metrics")

iou_score = calculate_iou(
    thresholded_prediction,
    cleaned_map
)

precision_score = calculate_precision(
    thresholded_prediction,
    cleaned_map
)

recall_score = calculate_recall(
    thresholded_prediction,
    cleaned_map
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:

    st.metric(
        "IoU Score",
        iou_score
    )

with metric_col2:

    st.metric(
        "Precision",
        precision_score
    )

with metric_col3:

    st.metric(
        "Recall",
        recall_score
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

# -----------------------------------
# Metrics Layout
# -----------------------------------

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Urban Change %",
        f"{change_percentage}%"
    )

    st.metric(
        "Vegetation Reduction",
        f"{vegetation_loss}"
    )

with col2:

    st.metric(
        "Infrastructure Growth Score",
        f"{growth_score}/100"
    )

    st.metric(
        "Development Density",
        f"{density_score}"
    )

# -----------------------------------
# Footer Logging
# -----------------------------------

logger.info(
    "Urban analysis pipeline executed successfully"
)