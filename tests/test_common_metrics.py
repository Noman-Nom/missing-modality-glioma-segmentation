"""Spec 0001, AC-8: the empty region rule and toy cases with known answers."""
import math

import numpy as np
import pytest
from common.metrics import HD95_EMPTY_MM, region_metrics

SP = (1.0, 1.0, 1.0)


def vol(*points, shape=(12, 12, 12)):
    v = np.zeros(shape, bool)
    for p in points:
        v[p] = True
    return v


def test_hd95_empty_constant():
    assert HD95_EMPTY_MM == 373.13
    assert round(math.sqrt(240**2 + 240**2 + 155**2), 2) == HD95_EMPTY_MM


def test_both_empty():
    m = region_metrics(vol(), vol(), SP)
    assert (m["dice"], m["hd95_mm"]) == (1.0, 0.0)
    assert math.isnan(m["sensitivity"]) and math.isnan(m["precision"])
    assert m["gt_empty"] and m["pred_empty"]


def test_truth_empty():
    m = region_metrics(vol((1, 1, 1)), vol(), SP)
    assert (m["dice"], m["hd95_mm"]) == (0.0, HD95_EMPTY_MM)
    assert math.isnan(m["sensitivity"]) and m["precision"] == 0.0
    assert m["gt_empty"] and not m["pred_empty"]


def test_prediction_empty():
    m = region_metrics(vol(), vol((1, 1, 1)), SP)
    assert (m["dice"], m["hd95_mm"]) == (0.0, HD95_EMPTY_MM)
    assert m["sensitivity"] == 0.0 and math.isnan(m["precision"])
    assert not m["gt_empty"] and m["pred_empty"]


def test_dice_075():
    gt = vol((5, 5, 5), (5, 5, 6), (5, 6, 5), (5, 6, 6))
    pred = vol((5, 5, 5), (5, 5, 6), (5, 6, 5), (6, 6, 6))
    m = region_metrics(pred, gt, SP)
    assert m["dice"] == pytest.approx(0.75)
    assert m["sensitivity"] == pytest.approx(0.75) and m["precision"] == pytest.approx(0.75)


def test_hd95_3mm():
    m = region_metrics(vol((5, 5, 8)), vol((5, 5, 5)), SP)
    assert m["hd95_mm"] == pytest.approx(3.0)
    assert m["dice"] == 0.0


def test_hd95_uses_spacing():
    m = region_metrics(vol((5, 5, 8)), vol((5, 5, 5)), (1.0, 1.0, 2.0))
    assert m["hd95_mm"] == pytest.approx(6.0)
