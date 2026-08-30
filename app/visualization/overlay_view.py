import cv2
import numpy as np


def create_overlay(image, change_mask, alpha=0.35):

    overlay = image.copy()

    contours, _ = cv2.findContours(
        change_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours
    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 0, 0),
        2
    )

    # Create red overlay
    red_mask = np.zeros_like(image)

    red_mask[:, :, 0] = change_mask

    overlay = cv2.addWeighted(
        overlay,
        1,
        red_mask,
        alpha,
        0
    )

    return overlay