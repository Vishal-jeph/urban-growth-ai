import cv2
import numpy as np


def generate_attention_overlay(
    original_image,
    prediction_map
):

    heatmap = cv2.applyColorMap(
        prediction_map,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay