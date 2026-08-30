import cv2
import numpy as np


def detect_changes(image1, image2):

    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    _, threshold = cv2.threshold(
        diff,
        30,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    cleaned = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel
    )

    return diff, cleaned