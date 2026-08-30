import cv2


def generate_attention_overlay(
    original_image,
    prediction_map,
    alpha=0.4
):

    heatmap = cv2.applyColorMap(
        prediction_map,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original_image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay
