"""BraTS 2023 GLI: channel order, label to region mapping, subject selection, preprocessing.

Spec 0001, AC-1, AC-2, AC-3. Channel order and labels are per dataset; this file is BraTS 2023 GLI.
"""
import csv
import re
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import Compose, RandCropByPosNegLabeld, SpatialPadd, Transform

# model channel order, fixed; files matched by suffix name, never by position
MODALITIES = ("T1", "T1ce", "T2", "FLAIR")
SUFFIX = {"T1": "t1n", "T1ce": "t1c", "T2": "t2w", "FLAIR": "t2f"}
SEG_SUFFIX = "seg"
# BraTS 2023 GLI labels: 1 NCR, 2 ED, 3 ET
REGIONS = ("WT", "TC", "ET")
REGION_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
# every value a BraTS 2023 GLI seg may hold (0 background); anything else raises in load_case
ALLOWED_LABELS = frozenset({0, 1, 2, 3})
# BraTS-GLI-<5 digit subject>-<3 digit timepoint>
BRATS_ID_RE = re.compile(r"^BraTS-GLI-(\d{5})-(\d{3})$")
STD_EPS = 1e-8


def label_to_regions(label):
    """Integer label (X, Y, Z) to region masks (3, X, Y, Z) in REGIONS order, uint8."""
    return np.stack([np.isin(label, REGION_LABELS[r]) for r in REGIONS]).astype(np.uint8)


def select_subjects(splits_csv, n_train, n_val, timepoint=0, exclude_flair_flag=True):
    """First N subjects by sorted subject ID per fold, one case each. Returns (train, val) case IDs.

    "First N by sorted ID" is deterministic but not random; revisit when sizing E1.
    """
    with open(splits_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if int(r["timepoint"]) == timepoint
            and not (exclude_flair_flag and r["flair_mask_flag"] == "True")]

    def first(fold, n):
        picked = sorted((r for r in keep if r["fold"] == fold), key=lambda r: r["subject"])[:n]
        assert len(picked) == n, f"fold {fold}: wanted {n} subjects, found {len(picked)}"
        assert len({r["subject"] for r in picked}) == n, f"fold {fold}: a subject appears twice"
        return [r["case"] for r in picked]

    train, val = first("train", n_train), first("val", n_val)
    fold_of = {r["case"]: r["fold"] for r in rows}
    assert all(fold_of[c] != "test" for c in train + val), "a test fold case was selected"
    assert not {subject_of(c) for c in train} & {subject_of(c) for c in val}, "subject overlap"
    return train, val


def subject_of(case):
    """BraTS-GLI-00002-000 -> 00002. Any other string raises ValueError."""
    m = BRATS_ID_RE.match(case)
    if not m:
        raise ValueError(f"not a BraTS 2023 GLI case ID: {case!r}")
    return m.group(1)


def case_paths(data_dir, case):
    """Map {modality or 'seg': path} by file suffix. Only *.nii.gz files count; absent keys are missing."""
    folder = Path(data_dir) / case
    by_suffix = {}
    if folder.is_dir():
        for p in folder.iterdir():
            if p.is_file() and p.name.startswith(case + "-") and p.name.endswith(".nii.gz"):
                by_suffix[p.name[len(case) + 1:-len(".nii.gz")]] = p
    out = {m: by_suffix[SUFFIX[m]] for m in MODALITIES if SUFFIX[m] in by_suffix}
    if SEG_SUFFIX in by_suffix:
        out["seg"] = by_suffix[SEG_SUFFIX]
    return out


def check_inputs(data_dir, cases):
    """Every expected file that is missing, as 'case: name' strings (empty list = all present)."""
    missing = []
    for c in cases:
        have = case_paths(data_dir, c)
        for key in (*MODALITIES, "seg"):
            if key not in have:
                suffix = SUFFIX.get(key, SEG_SUFFIX)
                missing.append(f"{c}: {c}-{suffix}.nii.gz")
    return missing


def crop_box(image):
    """Bounding box of voxels != 0 in any channel of a raw (C, X, Y, Z) stack, margin 0."""
    nz = np.argwhere(np.any(image != 0, axis=0))
    if len(nz) == 0:
        return tuple(slice(0, s) for s in image.shape[1:])
    lo, hi = nz.min(0), nz.max(0) + 1
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def normalise(image, low_pct, high_pct, name=""):
    """Per channel: mask = raw != 0, clip to percentiles over mask, z-score over mask, outside = 0.

    An empty channel or one with std below 1e-8 stays all 0 and logs a warning.
    """
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        x = image[c].astype(np.float32)
        mask = x != 0
        if not mask.any():
            warnings.warn(f"{name} channel {MODALITIES[c]}: no nonzero voxels, left at 0")
            continue
        lo, hi = np.percentile(x[mask], [low_pct, high_pct])
        v = np.clip(x[mask], lo, hi)
        std = v.std()
        if std < STD_EPS:
            warnings.warn(f"{name} channel {MODALITIES[c]}: std {std:.2e} below {STD_EPS}, left at 0")
            continue
        out[c][mask] = (v - v.mean()) / std
    return out


def load_case(paths):
    """Raw float32 (4, X, Y, Z) stack in MODALITIES order, int label, voxel spacing from seg header."""
    image = np.stack([np.asarray(nib.load(paths[m]).dataobj, dtype=np.float32) for m in MODALITIES])
    seg = nib.load(paths["seg"])
    raw = np.asarray(seg.dataobj)
    # by value, not dtype: a scaled NIfTI loads integer labels as float
    values = np.unique(raw)
    bad = values[(values != np.round(values)) | ~np.isin(values, list(ALLOWED_LABELS))]
    if bad.size:
        case = Path(paths["seg"]).parent.name
        raise ValueError(f"{case}: label values outside {sorted(ALLOWED_LABELS)}: "
                         f"{sorted(bad.tolist())}")
    label = raw.astype(np.int16)
    pixdim = tuple(float(z) for z in seg.header.get_zooms()[:3])
    return image, label, pixdim


class Preprocess(Transform):
    """{'case', 'paths'} -> cropped, normalised image, 3 region label, 1 channel wt mask, pixdim.

    Deterministic, so CacheDataset caches its output.
    """

    def __init__(self, low_pct, high_pct):
        self.low_pct, self.high_pct = low_pct, high_pct

    def __call__(self, d):
        image, label, pixdim = load_case(d["paths"])
        box = crop_box(image)
        n_label = int(np.count_nonzero(label))
        image, label = image[(slice(None), *box)], label[box]
        assert int(np.count_nonzero(label)) == n_label, f"{d['case']}: crop cut tumour voxels"
        return {
            "case": d["case"],
            "image": normalise(image, self.low_pct, self.high_pct, d["case"]),
            "label": label_to_regions(label),
            "wt": (label > 0)[None].astype(np.uint8),
            "pixdim": np.array(pixdim, dtype=np.float32),
        }


def items(data_dir, cases):
    return [{"case": c, "paths": case_paths(data_dir, c)} for c in cases]


def eval_transforms(cfg):
    p = cfg["preprocess"]
    return Compose([Preprocess(p["clip_low_pct"], p["clip_high_pct"])])


def train_transforms(cfg):
    p, t = cfg["preprocess"], cfg["train"]
    keys = ["image", "label", "wt"]
    return Compose([
        Preprocess(p["clip_low_pct"], p["clip_high_pct"]),
        SpatialPadd(keys=keys, spatial_size=t["patch"], value=0),
        # the only random step; label_key is the 1 channel wt mask (MONAI treats channel 0 of a
        # multichannel label as background)
        RandCropByPosNegLabeld(keys=keys, label_key="wt", spatial_size=t["patch"],
                               pos=t["pos"], neg=t["neg"], num_samples=1,
                               image_key="image", image_threshold=0),
    ])
