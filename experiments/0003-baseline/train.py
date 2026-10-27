"""E1 complete input baseline training (spec 0003): the full train fold, chosen on val only.

Run from the repo root, after preprocess.py:
    uv run python experiments/0003-baseline/train.py              # full run, seed 42 (human)
    uv run python experiments/0003-baseline/train.py --seed 43    # another seed, its own hash
    uv run python experiments/0003-baseline/train.py --quick      # fast gate
Writes results/0003-baseline/<run_id>/ with config.toml, run.json, train_log.csv and
val_log.csv (Dice per val point, pattern, case and region), checkpoints/best.pt (best val score)
and last.pt (latest val point). Never reads the test fold.
Exit 0 ok, 1 failed (OOM, non finite loss), 2 bad config, inputs or cache guard (before training),
130 Ctrl+C. Test hook: BASELINE_INJECT_INF=<iteration> feeds inf at that iteration.
"""
import argparse
import copy
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cache import check_cache, repo_path
from common.data import fold_cases
from common.runlog import config_hash, load_config
from common.training import check_config, make_run_dir, train

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
QUICK = {"train": {"iterations": 10, "val_every": 5, "warmup": 2, "patch": [64, 64, 64],
                   "num_workers": 0},
         "infer": {"roi": [64, 64, 64]}}
QUICK_TRAIN, QUICK_VAL = 2, 1


def apply_quick(cfg):
    cfg = copy.deepcopy(cfg)
    for section, values in QUICK.items():
        cfg[section].update(values)
    return cfg


def effective_config(path, quick=False, seed=None):
    """The config after --quick and --seed, before hashing."""
    cfg = load_config(path)
    if quick:
        cfg = apply_quick(cfg)
    if seed is not None:
        cfg["seed"] = seed
    return cfg


def run_cases(cfg, quick=False):
    """(train, val) case IDs of the config: fold_cases() filtered, the first few when quick."""
    d = cfg["data"]
    splits = repo_path(d["splits_csv"])
    train_cases = fold_cases(splits, "train", d["train_timepoints"], d["exclude_flair_flag"])
    val_cases = fold_cases(splits, "val", d["val_timepoints"], d["exclude_flair_flag"])
    if quick:
        train_cases, val_cases = train_cases[:QUICK_TRAIN], val_cases[:QUICK_VAL]
    return train_cases, val_cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=HERE / "config.toml")
    ap.add_argument("--seed", type=int, default=None, help="overrides seed before hashing")
    ap.add_argument("--out", default=REPO / "results" / "0003-baseline")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    a = ap.parse_args(argv)

    cfg = effective_config(a.config, a.quick, a.seed)
    bad = check_config(cfg)
    if bad:
        print("ERROR: bad config, nothing trained:\n" + "\n".join("  " + e for e in bad))
        return 2
    train_cases, val_cases = run_cases(cfg, a.quick)
    errors = check_cache(cfg, train_cases + val_cases)
    if errors:
        print(f"ERROR: cache guard, nothing trained ({len(errors)} problem(s)):")
        print("\n".join("  " + e for e in errors[:20]))
        return 2

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu")
    run_id, run_dir = make_run_dir(a.out, f"{datetime.now(UTC):%Y%m%d-%H%M%S}-"
                                          f"{config_hash(cfg)[:8]}")
    return train(cfg, run_dir, run_id, train_cases, val_cases, device=device, quick=a.quick)


if __name__ == "__main__":
    sys.exit(main())
