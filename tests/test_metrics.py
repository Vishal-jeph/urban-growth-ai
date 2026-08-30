import numpy as np

from app.inference.evaluation_metrics import (
    calculate_iou,
    calculate_precision,
    calculate_recall
)
from app.inference.urban_analytics import (
    calculate_change_percentage,
    infrastructure_growth_score
)


def test_metrics_perfect_match():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 255

    assert calculate_iou(mask, mask) == 1.0
    assert calculate_precision(mask, mask) == 1.0
    assert calculate_recall(mask, mask) == 1.0


def test_metrics_no_overlap():
    prediction = np.zeros((10, 10), dtype=np.uint8)
    prediction[0:3, 0:3] = 255

    target = np.zeros((10, 10), dtype=np.uint8)
    target[7:10, 7:10] = 255

    assert calculate_iou(prediction, target) == 0
    assert calculate_precision(prediction, target) == 0
    assert calculate_recall(prediction, target) == 0


def test_metrics_empty_masks_do_not_divide_by_zero():
    empty = np.zeros((10, 10), dtype=np.uint8)

    assert calculate_iou(empty, empty) == 0
    assert calculate_precision(empty, empty) == 0
    assert calculate_recall(empty, empty) == 0


def test_calculate_change_percentage():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :5] = 255

    assert calculate_change_percentage(mask) == 50.0


def test_infrastructure_growth_score_caps_at_100():
    assert infrastructure_growth_score(100) == 100
    assert infrastructure_growth_score(10) == 40
