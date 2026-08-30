import numpy as np


def calculate_iou(prediction, target):

    prediction = prediction > 127
    target = target > 127

    intersection = np.logical_and(
        prediction,
        target
    ).sum()

    union = np.logical_or(
        prediction,
        target
    ).sum()

    if union == 0:
        return 0

    return round(intersection / union, 4)


def calculate_precision(prediction, target):

    prediction = prediction > 127
    target = target > 127

    tp = np.logical_and(
        prediction,
        target
    ).sum()

    fp = np.logical_and(
        prediction,
        np.logical_not(target)
    ).sum()

    if tp + fp == 0:
        return 0

    return round(tp / (tp + fp), 4)


def calculate_recall(prediction, target):

    prediction = prediction > 127
    target = target > 127

    tp = np.logical_and(
        prediction,
        target
    ).sum()

    fn = np.logical_and(
        np.logical_not(prediction),
        target
    ).sum()

    if tp + fn == 0:
        return 0

    return round(tp / (tp + fn), 4)