"""E0 data audit: read every BraTS 2023 GLI volume once and write a per-case table.

Numpy only (no nibabel). Raw data is read only. Outputs go to results/0000-data-audit/.

Run (slow, ~10-25 min, human runs it):
    python3 experiments/0000-data-audit/audit.py \
        --train ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
        --val   ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData
"""
import argparse
import gzip
import json
import re
import struct
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

MODS = ["t1n", "t1c", "t2w", "t2f"]  # BraTS 2023 file suffixes; order here is only for tables
DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
          256: np.int8, 512: np.uint16, 768: np.uint32}
ID_RE = re.compile(r"^BraTS-GLI-(\d{5})-(\d{3})$")


def read_nifti(path):
    """Return (data float32/ints as stored, spacing, affine, dtype name). Plain .nii.gz only."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if struct.unpack("<i", raw[:4])[0] != 348:
        raise ValueError("not little-endian NIfTI-1")
    dim = struct.unpack("<8h", raw[40:56])
    datatype = struct.unpack("<h", raw[70:72])[0]
    pixdim = struct.unpack("<8f", raw[76:108])
    vox_offset = int(struct.unpack("<f", raw[108:112])[0])
    slope, inter = struct.unpack("<2f", raw[112:120])
    sform_code = struct.unpack("<h", raw[254:256])[0]
    qform_code = struct.unpack("<h", raw[252:254])[0]
    if sform_code > 0:
        srow = np.array(struct.unpack("<12f", raw[280:328])).reshape(3, 4)
        affine = np.vstack([srow, [0, 0, 0, 1]])
    else:  # some seg files carry only a qform (sform_code 0)
        b, c, d, qx, qy, qz = struct.unpack("<6f", raw[256:280])
        a_ = np.sqrt(max(0.0, 1.0 - (b * b + c * c + d * d)))
        qfac = -1.0 if pixdim[0] < 0 else 1.0
        R = np.array([[a_*a_+b*b-c*c-d*d, 2*(b*c-a_*d), 2*(b*d+a_*c)],
                      [2*(b*c+a_*d), a_*a_+c*c-b*b-d*d, 2*(c*d-a_*b)],
                      [2*(b*d-a_*c), 2*(c*d+a_*b), a_*a_+d*d-c*c-b*b]])
        zoom = np.array([pixdim[1], pixdim[2], pixdim[3] * qfac])
        affine = np.eye(4)
        affine[:3, :3] = R * zoom
        affine[:3, 3] = [qx, qy, qz]
    shape = tuple(dim[1:1 + dim[0]])
    arr = np.frombuffer(raw, dtype=DTYPES[datatype], count=int(np.prod(shape)), offset=vox_offset)
    arr = arr.reshape(shape, order="F")
    if slope not in (0.0, 1.0) or inter != 0.0:
        arr = arr.astype(np.float32) * (slope or 1.0) + inter
    return arr, pixdim[1:1 + dim[0]], affine, np.dtype(DTYPES[datatype]).name, max(sform_code, qform_code)


def axcodes(aff):
    """Orientation letters (e.g. RAS) from an affine, no nibabel."""
    labels = [("L", "R"), ("P", "A"), ("I", "S")]
    out = []
    m = aff[:3, :3]
    used = set()
    for i in range(3):
        j = int(np.argmax(np.abs(m[:, i]) * [1 if k not in used else 0 for k in range(3)]))
        used.add(j)
        out.append(labels[j][1] if m[j, i] > 0 else labels[j][0])
    return "".join(out)


def case_id_parts(cid):
    m = ID_RE.match(cid)
    return (m.group(1), m.group(2)) if m else (None, None)


def audit_case(args):
    split, case_dir = args
    case_dir = Path(case_dir)
    cid = case_dir.name
    subj, tp = case_id_parts(cid)
    row = {"split": split, "case": cid, "subject": subj, "timepoint": tp, "problems": []}
    prob = row["problems"]
    brains = {}
    ref_shape = ref_aff = None
    for m in MODS + ["seg"]:
        p = case_dir / f"{cid}-{m}.nii.gz"
        if not p.is_file():
            if m != "seg" or split == "train":
                prob.append(f"missing {m}")
            continue
        try:
            a, sp, aff, dt, sf = read_nifti(p)
        except Exception as e:  # corrupt gzip, bad header
            prob.append(f"unreadable {m}: {type(e).__name__}")
            continue
        row[f"{m}_dtype"] = dt
        if ref_shape is None:
            ref_shape, ref_aff = a.shape, aff
            row["shape"] = "x".join(map(str, a.shape))
            row["spacing"] = "x".join(f"{v:.3g}" for v in sp)
            row["orient"] = axcodes(aff)
            row["sform_code"] = sf
        else:
            if a.shape != ref_shape:
                prob.append(f"shape mismatch {m}")
            if not np.allclose(aff, ref_aff, atol=1e-3):
                prob.append(f"affine mismatch {m}")
        if m == "seg":
            vals, cnts = np.unique(a, return_counts=True)
            row["seg_labels"] = ",".join(str(int(v)) for v in vals)
            c = dict(zip(vals.astype(int).tolist(), cnts.tolist()))
            vox_ml = float(np.prod(sp[:3])) / 1000.0
            n1, n2, n3 = c.get(1, 0), c.get(2, 0), c.get(3, 0)
            row.update(vox_NCR=n1, vox_ED=n2, vox_ET=n3,
                       ml_WT=(n1 + n2 + n3) * vox_ml, ml_TC=(n1 + n3) * vox_ml, ml_ET=n3 * vox_ml)
            row["ET_empty"] = n3 == 0
            row["TC_empty"] = (n1 + n3) == 0
            row["WT_empty"] = (n1 + n2 + n3) == 0
            if set(c) - {0, 1, 2, 3}:
                prob.append("unexpected seg labels")
            tum = a > 0
            if tum.any():
                idx = np.argwhere(tum)
                lo, hi = idx.min(0), idx.max(0)
                row["tumor_bbox"] = "x".join(str(int(h - l + 1)) for l, h in zip(lo, hi))
                row["tumor_centroid"] = ";".join(f"{v:.0f}" for v in idx.mean(0))
                # number of connected-ish pieces is skipped on purpose (slow); WT bbox only
            row["_seg"] = None
            continue
        f = a.astype(np.float32)
        if not np.isfinite(f).all():
            prob.append(f"nan/inf {m}")
        nz = f != 0
        brains[m] = nz
        n = int(nz.sum())
        row[f"{m}_brain_vox"] = n
        if n == 0:
            prob.append(f"all zero {m}")
            continue
        v = f[nz]
        p1, p50, p99 = np.percentile(v, [1, 50, 99])
        row.update({f"{m}_min": float(v.min()), f"{m}_max": float(v.max()),
                    f"{m}_mean": float(v.mean()), f"{m}_std": float(v.std()),
                    f"{m}_p1": float(p1), f"{m}_p50": float(p50), f"{m}_p99": float(p99)})
        if v.min() < 0:
            prob.append(f"negative values {m}")
    if len(brains) == 4:
        base = brains["t1n"]
        for m in MODS[1:]:
            row[f"brain_dice_t1n_vs_{m}"] = float(2 * (base & brains[m]).sum() / (base.sum() + brains[m].sum()))
        row["brain_vox_min_over_max"] = min(b.sum() for b in brains.values()) / max(b.sum() for b in brains.values())
    row.pop("_seg", None)
    row["problems"] = "; ".join(prob)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="results/0000-data-audit")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="only first N cases per split (fast gate)")
    a = ap.parse_args()
    import pandas as pd

    jobs = []
    for split, root in (("train", a.train), ("val", a.val)):
        dirs = sorted(d for d in Path(root).iterdir() if d.is_dir())
        jobs += [(split, str(d)) for d in (dirs[:a.limit] if a.limit else dirs)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap(audit_case, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "cases.csv", index=False)
    print(f"wrote {out/'cases.csv'} ({len(df)} cases); run summarize.py next")


if __name__ == "__main__":
    main()
