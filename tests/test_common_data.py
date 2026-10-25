"""Spec 0001: subject selection (AC-1), channel and label mapping (AC-2), preprocessing (AC-3)."""
import csv
import os
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from common.data import (
    MODALITIES,
    SUFFIX,
    Preprocess,
    case_paths,
    check_inputs,
    crop_box,
    label_to_regions,
    load_case,
    normalise,
    select_subjects,
    subject_of,
)
from conftest import write_case, write_splits

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "data" / "splits" / "splits.csv"


# AC-1
def test_select_subjects_real_splits():
    train, val = select_subjects(SPLITS, 16, 4)
    assert (len(train), len(val)) == (16, 4)
    with open(SPLITS, newline="") as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    for c in train + val:
        assert rows[c]["fold"] != "test"
        assert rows[c]["timepoint"] == "0" and rows[c]["flair_mask_flag"] == "False"
    assert all(rows[c]["fold"] == "train" for c in train)
    assert all(rows[c]["fold"] == "val" for c in val)
    assert not {subject_of(c) for c in train} & {subject_of(c) for c in val}
    assert train == sorted(train, key=subject_of) and val == sorted(val, key=subject_of)
    assert select_subjects(SPLITS, 16, 4) == (train, val)


def test_select_subjects_excludes_flagged_and_timepoint1(tmp_path):
    p = tmp_path / "splits.csv"
    write_splits(p, [
        ("BraTS-GLI-00001-000", "train", 0, True),   # flagged
        ("BraTS-GLI-00002-001", "train", 1, False),  # timepoint 1
        ("BraTS-GLI-00003-000", "train", 0, False),
        ("BraTS-GLI-00000-000", "test", 0, False),   # test fold, lowest ID
        ("BraTS-GLI-00004-000", "val", 0, False),
    ])
    assert select_subjects(p, 1, 1) == (["BraTS-GLI-00003-000"], ["BraTS-GLI-00004-000"])
    with pytest.raises(AssertionError):
        select_subjects(p, 2, 1)
    assert select_subjects(p, 2, 1, exclude_flair_flag=False)[0] == [
        "BraTS-GLI-00001-000", "BraTS-GLI-00003-000"]


# AC-2
def test_label_to_regions():
    wt, tc, et = label_to_regions(np.array([0, 1, 2, 3]))
    assert wt.tolist() == [0, 1, 1, 1]
    assert tc.tolist() == [0, 1, 0, 1]
    assert et.tolist() == [0, 0, 0, 1]


def test_channel_order_by_suffix(tmp_path):
    case = "BraTS-GLI-00011-000"
    folder = tmp_path / case
    folder.mkdir()
    (folder / f"{case}-t1n.nii").mkdir()  # stray folder like cases 00011 and 00014
    value = {"T1": 1.0, "T1ce": 2.0, "T2": 3.0, "FLAIR": 4.0}
    for m in ("FLAIR", "T1ce", "T1", "T2"):  # written out of order
        nib.save(nib.Nifti1Image(np.full((4, 4, 4), value[m], np.float32), np.eye(4)),
                 folder / f"{case}-{SUFFIX[m]}.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.eye(4)), folder / f"{case}-seg.nii.gz")
    image, _, _ = load_case(case_paths(tmp_path, case))
    assert [float(image[i].mean()) for i in range(4)] == [value[m] for m in MODALITIES]
    assert MODALITIES == ("T1", "T1ce", "T2", "FLAIR")
    assert SUFFIX == {"T1": "t1n", "T1ce": "t1c", "T2": "t2w", "FLAIR": "t2f"}


def test_check_inputs_lists_missing(tmp_path):
    write_case(tmp_path, "BraTS-GLI-00001-000")
    os.remove(tmp_path / "BraTS-GLI-00001-000" / "BraTS-GLI-00001-000-t2w.nii.gz")
    assert check_inputs(tmp_path, ["BraTS-GLI-00001-000"]) == [
        "BraTS-GLI-00001-000: BraTS-GLI-00001-000-t2w.nii.gz"]
    assert len(check_inputs(tmp_path, ["BraTS-GLI-00002-000"])) == 5


# AC-3
def test_normalise_zscore_over_nonzero_background_zero():
    rng = np.random.default_rng(0)
    x = np.zeros((4, 10, 10, 10), np.float32)
    x[:, 2:8, 2:8, 2:8] = rng.normal(50, 5, (4, 6, 6, 6))
    x[0, 5, 5, 5] = -3.0  # negatives count as brain
    out = normalise(x, 0.5, 99.5)
    mask = x != 0
    assert np.all(out[~mask] == 0)
    for c in range(4):
        assert abs(out[c][mask[c]].mean()) < 1e-5
        assert abs(out[c][mask[c]].std() - 1) < 1e-4


def test_normalise_clips_to_percentiles():
    x = np.zeros((4, 1, 1, 1001), np.float32)
    x[:, 0, 0, :1000] = np.arange(1, 1001)
    x[:, 0, 0, 1000] = 1e6  # outlier
    out = normalise(x, 0.5, 99.5)
    v = x[0, 0, 0]
    lo, hi = np.percentile(v, [0.5, 99.5])
    c = np.clip(v, lo, hi)
    np.testing.assert_allclose(out[0, 0, 0], (c - c.mean()) / c.std(), rtol=1e-5, atol=1e-5)


def test_normalise_empty_and_constant_channel_stay_zero():
    x = np.zeros((4, 5, 5, 5), np.float32)
    x[1, 1:4, 1:4, 1:4] = 7.0  # constant
    x[2:, 1:4, 1:4, 1:4] = np.arange(27).reshape(3, 3, 3) + 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = normalise(x, 0.5, 99.5, "toy")
    assert np.all(out[0] == 0) and np.all(out[1] == 0)
    assert len(w) == 2 and "no nonzero" in str(w[0].message) and "std" in str(w[1].message)


def test_crop_box_any_channel():
    x = np.zeros((4, 10, 10, 10))
    x[0, 2, 3, 4] = 1
    x[3, 6, 7, 8] = -1
    assert crop_box(x) == (slice(2, 7), slice(3, 8), slice(4, 9))


def test_crop_keeps_label_count(tmp_path):
    case = "BraTS-GLI-00001-000"
    label = write_case(tmp_path, case)
    d = Preprocess(0.5, 99.5)({"case": case, "paths": case_paths(tmp_path, case)})
    assert d["image"].shape[0] == 4 and d["image"].shape[1:] == d["label"].shape[1:]
    assert d["image"].shape[1:] == (29, 29, 29)  # the r < 15 ball
    assert int(d["wt"].sum()) == int(d["label"][0].sum()) == int((label > 0).sum())
    assert d["image"].dtype == np.float32
    assert tuple(d["pixdim"]) == (1.0, 1.0, 1.0)


@pytest.mark.slow
def test_real_case_through_eval_transforms():
    data = Path(os.environ.get("BRATS_TRAIN_DIR",
                               REPO / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"))
    case = "BraTS-GLI-00002-000"
    d = Preprocess(0.5, 99.5)({"case": case, "paths": case_paths(data, case)})
    assert d["image"].shape[0] == 4 and d["label"].shape[0] == 3 and d["wt"].shape[0] == 1
    assert d["image"].shape[1:] == d["label"].shape[1:]
    assert all(s <= o for s, o in zip(d["image"].shape[1:], (240, 240, 155)))
    assert np.isfinite(d["image"]).all()


def write_seg(tmp_path, values, dtype):
    """A synthetic case whose seg holds `values` (other voxels 0), saved with `dtype`."""
    case = "BraTS-GLI-00001-000"
    write_case(tmp_path, case)
    seg = np.zeros((40, 40, 40), dtype)
    seg.flat[:len(values)] = values
    nib.save(nib.Nifti1Image(seg, np.eye(4)), tmp_path / case / f"{case}-seg.nii.gz")
    return case, case_paths(tmp_path, case)


# spec 0002, AC-6
@pytest.mark.parametrize(("values", "dtype", "bad"), [
    ([1, 2, 3, 4], np.int16, "[4]"),
    ([1.0, 1.5, 3.0], np.float32, "[1.5]"),
])
def test_load_case_rejects_labels_outside_allowed(tmp_path, values, dtype, bad):
    case, paths = write_seg(tmp_path, values, dtype)
    with pytest.raises(ValueError, match=rf"{case}: label values outside \[0, 1, 2, 3\]: \{bad}"):
        load_case(paths)


# spec 0002, AC-6: integer by value, not by dtype
def test_load_case_accepts_float_labels_with_integer_values(tmp_path):
    _, paths = write_seg(tmp_path, [1.0, 2.0, 3.0], np.float32)
    _, label, _ = load_case(paths)
    assert label.dtype == np.int16 and set(np.unique(label)) == {0, 1, 2, 3}


# spec 0002, AC-7
def test_subject_of_valid_id():
    assert subject_of("BraTS-GLI-00002-001") == "00002"


@pytest.mark.parametrize("case", ["BRATS_001", "BraTS-GLI-0002-000", "BraTS-GLI-00002-000.nii.gz"])
def test_subject_of_rejects_other_ids(case):
    with pytest.raises(ValueError, match="not a BraTS 2023 GLI case ID"):
        subject_of(case)
