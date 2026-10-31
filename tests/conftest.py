"""Shared fixtures: tiny synthetic BraTS style cases and the spec 0003 CPU setup (no real data, no GPU)."""
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from common.data import MODALITIES, SUFFIX
from common.runlog import dump_config, load_config

SHAPE = (40, 40, 40)
BASELINE_EXP = Path(__file__).resolve().parents[1] / "experiments" / "0003-baseline"


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


@pytest.fixture
def setup(tmp_path, synthetic_cases):
    """A tiny CPU config over the synthetic cases, its own cache dir; returns a small namespace."""
    data, splits, train_cases, val_cases = synthetic_cases
    cfg = load_config(BASELINE_EXP / "config.toml")
    cfg["data"].update(splits_csv=str(splits), cache_dir=str(tmp_path / "cache"))
    cfg["train"].update(iterations=2, val_every=2, patch=[32, 32, 32], num_workers=0)
    cfg["model"].update(channels=[4, 8], strides=[2], num_res_units=1)
    cfg["infer"]["roi"] = [32, 32, 32]
    path = tmp_path / "config.toml"
    dump_config(cfg, path)

    class S:
        pass

    s = S()
    s.cfg, s.config, s.data, s.splits = cfg, path, data, splits
    s.train, s.val, s.cache, s.out = train_cases, val_cases, tmp_path / "cache", tmp_path / "results"
    s.pre_argv = ["--config", str(path), "--data-dir", str(data)]
    s.train_argv = ["--config", str(path), "--out", str(s.out), "--device", "cpu"]
    return s


def only_run(out):
    (run,) = out.iterdir()
    return run
