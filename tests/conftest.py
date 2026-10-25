"""Shared fixtures: tiny synthetic BraTS style cases (no real data, no GPU)."""
import csv

import nibabel as nib
import numpy as np
import pytest
from common.data import MODALITIES, SUFFIX

SHAPE = (40, 40, 40)


def write_case(data_dir, case, seed=0, shape=SHAPE):
    """A ball shaped 'brain' with random intensities and a 3 label tumour; returns the label."""
    rng = np.random.default_rng(seed)
    folder = data_dir / case
    folder.mkdir(parents=True)
    grid = np.indices(shape) - (np.array(shape) // 2)[:, None, None, None]
    r = np.sqrt((grid**2).sum(0))
    brain = r < 15
    label = np.zeros(shape, np.int16)
    label[r < 7] = 2
    label[r < 5] = 3
    label[r < 2] = 1
    for i, m in enumerate(MODALITIES):
        img = np.where(brain, rng.normal(100 + 20 * i, 10, shape) + 30 * (label > 0), 0)
        nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)),
                 folder / f"{case}-{SUFFIX[m]}.nii.gz")
    nib.save(nib.Nifti1Image(label, np.eye(4)), folder / f"{case}-seg.nii.gz")
    return label


def write_splits(path, rows):
    """rows: (case, fold, timepoint, flair_mask_flag)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "subject", "timepoint", "fold", "flair_mask_flag"])
        for case, fold, tp, flag in rows:
            w.writerow([case, case.split("-")[2], tp, fold, str(flag)])


@pytest.fixture
def synthetic_cases(tmp_path):
    """3 train and 1 val synthetic cases plus a mini splits.csv (with a test row never used)."""
    data = tmp_path / "data"
    train = [f"BraTS-GLI-0000{i}-000" for i in range(1, 4)]
    val = ["BraTS-GLI-00005-000"]
    for i, c in enumerate(train + val):
        write_case(data, c, seed=i)
    splits = tmp_path / "splits.csv"
    write_splits(splits, [(c, "train", 0, False) for c in train]
                 + [(c, "val", 0, False) for c in val] + [("BraTS-GLI-00009-000", "test", 0, False)])
    return data, splits, train, val
