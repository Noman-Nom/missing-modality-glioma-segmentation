"""Fast gate: reader works on 3 real cases, synthetic NIfTI round-trips, orientation code is right."""
import gzip
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audit import audit_case, axcodes, read_nifti


def write_nifti(path, arr, sp=(1, 1, 1)):
    h = bytearray(348)
    struct.pack_into("<i", h, 0, 348)
    struct.pack_into("<8h", h, 40, 3, *arr.shape, 1, 1, 1, 1)
    struct.pack_into("<h", h, 70, 16)
    struct.pack_into("<8f", h, 76, 1, *sp, 0, 0, 0, 0)
    struct.pack_into("<f", h, 108, 352.0)
    struct.pack_into("<f", h, 112, 1.0)
    struct.pack_into("<h", h, 254, 1)
    struct.pack_into("<12f", h, 280, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    with gzip.open(path, "wb") as f:
        f.write(bytes(h) + b"\0\0\0\0" + np.asfortranarray(arr.astype(np.float32)).tobytes(order="F"))


def main():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "x.nii.gz"
        write_nifti(p, a)
        b, _, _, dt, _ = read_nifti(p)
        assert np.array_equal(a, b) and dt == "float32", "round trip"
    assert axcodes(np.diag([1, 1, 1, 1.0])) == "RAS"
    assert axcodes(np.diag([-1, -1, 1, 1.0])) == "LPS"
    # toy case with known answer: 10 NCR, 20 ED, 30 ET voxels at 1 mm
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "BraTS-GLI-00001-000"
        d.mkdir()
        img = np.ones((6, 6, 6), np.float32)
        for m in ["t1n", "t1c", "t2w", "t2f"]:
            write_nifti(d / f"{d.name}-{m}.nii.gz", img)
        seg = np.zeros(216, np.float32)
        seg[:10], seg[10:30], seg[30:60] = 1, 2, 3
        write_nifti(d / f"{d.name}-seg.nii.gz", seg.reshape(6, 6, 6))
        r = audit_case(("train", str(d)))
        assert (r["vox_NCR"], r["vox_ED"], r["vox_ET"]) == (10, 20, 30), r
        assert abs(r["ml_WT"] - 0.06) < 1e-9 and abs(r["ml_TC"] - 0.04) < 1e-9, r
        assert r["problems"] == "", r["problems"]
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if root and root.is_dir():
        for d in sorted(root.iterdir())[:3]:
            r = audit_case(("train", str(d)))
            assert r["problems"] == "" and r["shape"] == "240x240x155", r
            print(d.name, r["shape"], r["orient"], r["seg_labels"], round(r["ml_WT"], 1))
    print("OK")


if __name__ == "__main__":
    main()
