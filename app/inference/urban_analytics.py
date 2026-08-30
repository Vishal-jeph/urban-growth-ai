import numpy as np
import cv2


def calculate_change_percentage(change_mask):

    changed_pixels = np.sum(change_mask > 0)

    total_pixels = change_mask.shape[0] * change_mask.shape[1]

    percentage = (changed_pixels / total_pixels) * 100

    return round(percentage, 2)


def estimate_vegetation_loss(image1, image2):

    green1 = image1[:, :, 1]
    green2 = image2[:, :, 1]

    vegetation_diff = np.mean(green1) - np.mean(green2)

    return round(max(vegetation_diff, 0), 2)


def infrastructure_growth_score(change_percentage):

    score = min(change_percentage * 4, 100)

    return round(score, 2)


def estimate_density(change_mask):

    density = cv2.GaussianBlur(change_mask, (11, 11), 0)

    density_score = np.mean(density)

    return round(density_score / 2.55, 2)