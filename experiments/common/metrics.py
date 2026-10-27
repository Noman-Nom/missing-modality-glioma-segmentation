"""Per region metrics with the one empty region rule (BraTS convention; spec 0001, AC-8).

Both empty: Dice 1, HD95 0. Exactly one empty: Dice 0, HD95 = HD95_EMPTY_MM.
Sensitivity is NaN when truth is empty; precision is NaN when prediction is empty.
"""
import math

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance

# volume diagonal of the 240 x 240 x 155 BraTS grid at 1 mm: sqrt(240^2 + 240^2 + 155^2)
HD95_EMPTY_MM = 373.13


def region_metrics(pred, gt, spacing, with_hd95=True):
    """Binary pred and gt (X, Y, Z), voxel spacing in mm. Returns a dict of the metrics.

    with_hd95=False skips the distance (slow) and gives hd95_mm NaN; validation needs Dice only.
    """
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    tp = int((pred & gt).sum())
    gt_empty, pred_empty = n_gt == 0, n_pred == 0
    if gt_empty and pred_empty:
        dice, hd95 = 1.0, 0.0
    elif gt_empty or pred_empty:
        dice, hd95 = 0.0, HD95_EMPTY_MM
    else:
        dice = 2 * tp / (n_pred + n_gt)
        hd95 = math.nan if not with_hd95 else float(compute_hausdorff_distance(
            torch.from_numpy(pred)[None, None], torch.from_numpy(gt)[None, None],
            include_background=True, percentile=95, directed=False,
            spacing=[float(s) for s in spacing]).item())
    return {
        "dice": dice,
        "hd95_mm": hd95,
        "sensitivity": math.nan if gt_empty else tp / n_gt,
        "precision": math.nan if pred_empty else tp / n_pred,
        "gt_empty": gt_empty,
        "pred_empty": pred_empty,
    }
