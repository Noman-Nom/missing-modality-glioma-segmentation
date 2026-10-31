"""The preprocessed float16 disk cache (spec 0003, AC-2, AC-3, AC-4).

One uncompressed <case>.npz per case, plus manifest.csv (one row per case) and cache.json (one per
cache dir). Every file is written to a temp name and renamed, so a crash never leaves a half file.
"""
import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from common.data import crop_box, load_case, normalise
from common.runlog import config_hash

# raise by hand whenever the preprocessing code changes its output
PREPROCESS_VERSION = 1
REPO = Path(__file__).resolve().parents[2]
MANIFEST_COLUMNS = ("case", "subject", "fold", "timepoint", "shape_d", "shape_h", "shape_w",
                    "sha256", "preprocess_hash")
DEV_FOLDS = ("train", "val")


def repo_path(p):
    """A config path: relative paths are from the repo root, absolute paths stay as they are."""
    return REPO / p


def cache_dir(cfg):
    return repo_path(cfg["data"]["cache_dir"])


def preprocess_hash(cfg):
    """Only [preprocess] and the code version invalidate the cache, never training settings."""
    return config_hash({"preprocess": cfg["preprocess"], "version": PREPROCESS_VERSION})


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def preprocess_case(paths, low_pct, high_pct, case=""):
    """Raw NIfTI -> (image float32 (4, D, H, W) cropped and normalised as spec 0001 AC-3,
    label uint8 (D, H, W) raw values cropped, spacing float32 (3,), crop box int (3, 2))."""
    image, label, pixdim = load_case(paths)
    box = crop_box(image)
    n_label = int(np.count_nonzero(label))
    image, label = image[(slice(None), *box)], label[box]
    assert int(np.count_nonzero(label)) == n_label, f"{case}: crop cut tumour voxels"
    return (normalise(image, low_pct, high_pct, case), label.astype(np.uint8),
            np.array(pixdim, dtype=np.float32), np.array([[s.start, s.stop] for s in box]))


def write_case(cdir, case, image, label, spacing, box):
    """Write <case>.npz (image as float16) through a temp file. Returns its sha256."""
    path = Path(cdir) / f"{case}.npz"
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, image=image.astype(np.float16), label=label, spacing=spacing, crop_box=box)
    os.replace(tmp, path)
    return sha256_file(path)


def read_manifest(cdir):
    """{case: row} of manifest.csv, empty if there is none."""
    path = Path(cdir) / "manifest.csv"
    if not path.is_file():
        return {}
    with open(path, newline="") as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def read_cache_json(cdir):
    path = Path(cdir) / "cache.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _replace_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_manifest(cdir, manifest, info):
    """Rewrite manifest.csv, then cache.json with n_cases and folds recomputed from the manifest."""
    cdir = Path(cdir)
    tmp = cdir / "manifest.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for case in sorted(manifest):
            w.writerow(manifest[case])
    os.replace(tmp, cdir / "manifest.csv")
    info = dict(info, n_cases=len(manifest),
                folds=sorted({r["fold"] for r in manifest.values()}),
                updated_utc=datetime.now(UTC).isoformat(timespec="seconds"))
    _replace_text(cdir / "cache.json", json.dumps(info, indent=2))
    return info


def check_cache(cfg, cases, allowed_folds=DEV_FOLDS):
    """The guard every E1 script runs before any GPU work. Returns error strings (empty = ok).

    cache.json present with this config's preprocess_hash, every case in the manifest with its
    file present, and every case in an allowed fold (E1: train or val, never test).
    """
    cdir = cache_dir(cfg)
    info = read_cache_json(cdir)
    if info is None:
        return [f"no cache.json in {cdir}; run preprocess.py first"]
    want = preprocess_hash(cfg)
    if info["preprocess_hash"] != want:
        return [(f"cache {cdir} has preprocess hash {info['preprocess_hash'][:12]}, "
                 f"config needs {want[:12]}")]
    manifest = read_manifest(cdir)
    errors = []
    for c in cases:
        row = manifest.get(c)
        if row is None or not (cdir / f"{c}.npz").is_file():
            errors.append(f"{c}: not in the cache")
        elif row["fold"] not in allowed_folds:
            errors.append(f"{c}: fold {row['fold']!r} not allowed here ({', '.join(allowed_folds)})")
    return errors
