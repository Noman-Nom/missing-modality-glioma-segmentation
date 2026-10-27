"""Build the E1 float16 disk cache from raw BraTS 2023 GLI NIfTI (spec 0003, AC-2, AC-3, AC-4).

Run from the repo root:
    uv run python experiments/0003-baseline/preprocess.py            # train and val folds (human)
    uv run python experiments/0003-baseline/preprocess.py --quick    # only the quick run cases
Writes <cache_dir>/<case>.npz, manifest.csv and cache.json. Caches every case of the requested
folds (no timepoint or flag filter); test only with --folds test. Cases already in the manifest
with the same preprocess hash and their file present are skipped, so a rerun only adds missing
cases. Exit 2: no data dir, a missing input file, or the cache dir holds a different hash.
"""
import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cache import (
    PREPROCESS_VERSION,
    cache_dir,
    preprocess_case,
    preprocess_hash,
    read_cache_json,
    read_manifest,
    repo_path,
    sha256_file,
    write_case,
    write_manifest,
)
from common.data import BRATS_ID_RE, case_paths, check_inputs, fold_cases, subject_of
from common.runlog import git_info, load_config

HERE = Path(__file__).resolve().parent
FOLDS = ("train", "val", "test")


def quick_cases(cfg, splits):
    """The first 2 train and first 1 val case of the filtered sorted lists (train.py --quick)."""
    d = cfg["data"]
    return (fold_cases(splits, "train", d["train_timepoints"], d["exclude_flair_flag"])[:2]
            + fold_cases(splits, "val", d["val_timepoints"], d["exclude_flair_flag"])[:1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=HERE / "config.toml")
    ap.add_argument("--data-dir", default=os.environ.get("BRATS_TRAIN_DIR"))
    ap.add_argument("--splits", default=None, help="default: [data] splits_csv of the config")
    ap.add_argument("--folds", default="train,val", help="comma separated, of train, val, test")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    folds = [f for f in a.folds.split(",") if f]
    if not folds or set(folds) - set(FOLDS):
        print(f"ERROR: --folds {a.folds!r}: each must be one of {', '.join(FOLDS)}")
        return 2
    if not a.data_dir or not Path(a.data_dir).is_dir():
        print(f"ERROR: data dir not found: {a.data_dir!r} (set BRATS_TRAIN_DIR or --data-dir)")
        return 2
    splits = repo_path(a.splits or cfg["data"]["splits_csv"])
    cdir = cache_dir(cfg)
    want = preprocess_hash(cfg)
    info = read_cache_json(cdir)
    if info is not None and info["preprocess_hash"] != want:
        print(f"ERROR: {cdir} holds a cache with preprocess hash {info['preprocess_hash'][:12]}, "
              f"this config needs {want[:12]}. Nothing deleted; use another cache_dir.")
        return 2

    fold_of = {c: f for f in FOLDS for c in fold_cases(splits, f)}
    cases = quick_cases(cfg, splits) if a.quick else sorted(fold_of)
    cases = [c for c in cases if fold_of[c] in folds]
    missing = check_inputs(a.data_dir, cases)
    if missing:
        print(f"ERROR: {len(missing)} input file(s) missing, nothing cached:")
        print("\n".join("  " + m for m in missing))
        return 2

    cdir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(cdir)
    commit, _ = git_info()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    info = info or {"preprocess_hash": want, "preprocess_version": PREPROCESS_VERSION,
                    "preprocess": cfg["preprocess"], "created_utc": now}
    info.update(git_commit=commit, splits_sha256=sha256_file(splits))
    p = cfg["preprocess"]
    written = skipped = 0
    for i, c in enumerate(cases, start=1):
        row = manifest.get(c)
        if row and row["preprocess_hash"] == want and (cdir / f"{c}.npz").is_file():
            skipped += 1
            continue
        image, label, spacing, box = preprocess_case(case_paths(a.data_dir, c),
                                                     p["clip_low_pct"], p["clip_high_pct"], c)
        sha = write_case(cdir, c, image, label, spacing, box)
        manifest[c] = {"case": c, "subject": subject_of(c), "fold": fold_of[c],
                       "timepoint": int(BRATS_ID_RE.match(c).group(2)), "shape_d": label.shape[0],
                       "shape_h": label.shape[1], "shape_w": label.shape[2], "sha256": sha,
                       "preprocess_hash": want}
        # the row goes in only after the rename, so a rerun picks up where a crash stopped
        info = write_manifest(cdir, manifest, info)
        written += 1
        print(f"[{i}/{len(cases)}] {c}")
    if not written:
        info = write_manifest(cdir, manifest, info)
    gb = sum((cdir / f"{c}.npz").stat().st_size for c in manifest) / 1e9
    print(f"{written} written, {skipped} skipped, {len(manifest)} cases in {cdir} ({gb:.2f} GB)")
    print(f"preprocess hash {want}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
