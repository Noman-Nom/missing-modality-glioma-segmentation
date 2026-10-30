"""E1 complete input baseline training (spec 0003): the full train fold, chosen on val only.

Run from the repo root, after preprocess.py:
    uv run python experiments/0003-baseline/train.py              # full run, seed 42 (human)
    uv run python experiments/0003-baseline/train.py --seed 43    # another seed, its own hash
    uv run python experiments/0003-baseline/train.py --quick      # fast gate
    uv run python experiments/0003-baseline/train.py --resume <run dir>   # after a failure
Writes results/0003-baseline/<run_id>/ with config.toml, run.json, train_log.csv and
val_log.csv (Dice per val point, pattern, case and region), checkpoints/best.pt (best val score)
and last.pt (full training state at the latest val point). Never reads the test fold.
--resume continues a failed, interrupted or crashed run from last.pt with the run's own
config.toml (only --device may be given with it); at most val_every iterations are redone.
Exit 0 ok, 1 failed (OOM, non finite loss), 2 bad config, inputs, cache guard or resume refused
(before training), 130 Ctrl+C. Test hook: BASELINE_INJECT_INF=<iteration> feeds inf at that iteration.
"""
import argparse
import copy
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cache import cache_dir, check_cache, preprocess_hash, read_cache_json, repo_path
from common.data import fold_cases
from common.runlog import config_hash, load_config, load_run
from common.training import RUN_KEYS, check_config, make_run_dir, train

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


def pid_alive(path):
    """True when train.pid names a process that is still running."""
    try:
        pid = int(Path(path).read_text())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resume_problems(run_dir):
    """Why --resume must refuse this run (AC-11), as strings; empty = it may continue.
    Returns (problems, config, run.json)."""
    run_dir = Path(run_dir)
    if not (run_dir / "run.json").is_file() or not (run_dir / "config.toml").is_file():
        return [f"{run_dir} has no run.json or config.toml"], None, None
    rec, cfg = load_run(run_dir), load_config(run_dir / "config.toml")
    if set(rec) != set(RUN_KEYS):
        return [f"run.json keys are not the E1 keys: {sorted(set(rec) ^ set(RUN_KEYS))}"], cfg, rec
    if rec["status"] == "ok":
        return ["the run finished (status ok); start a new run instead"], cfg, rec
    if rec["status"] == "running" and pid_alive(run_dir / "train.pid"):
        return ["the run is still training (process in train.pid is alive)"], cfg, rec
    problems = []
    if config_hash(cfg) != rec["config_hash"]:
        problems.append("the run's config.toml was changed (hash differs from run.json)")
    problems += check_config(cfg)
    info = read_cache_json(cache_dir(cfg))
    if preprocess_hash(cfg) != rec["cache_hash"] or (info and info["preprocess_hash"]
                                                     != rec["cache_hash"]):
        problems.append("the cache hash differs from run.json cache_hash")
    if not (run_dir / "checkpoints" / "last.pt").is_file():
        problems.append("no last.pt (it failed before the first val point); start a new run")
    problems += check_cache(cfg, rec["train_cases"] + rec["val_cases"])
    return problems, cfg, rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="default: config.toml next to this script")
    ap.add_argument("--seed", type=int, default=None, help="overrides seed before hashing")
    ap.add_argument("--out", default=None, help="default: results/0003-baseline")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--resume", type=Path, default=None, metavar="RUN_DIR")
    a = ap.parse_args(argv)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu")

    if a.resume:
        if a.config or a.seed is not None or a.out or a.quick:
            print("ERROR: --resume takes only --device; the run's own config.toml is used")
            return 2
        problems, cfg, rec = resume_problems(a.resume)
        if problems:
            print("ERROR: resume refused, nothing trained:\n"
                  + "\n".join("  " + p for p in problems[:20]))
            return 2
        return train(cfg, a.resume, rec["run_id"], rec["train_cases"], rec["val_cases"],
                     device=device, quick=rec["quick"], resume=True)

    cfg = effective_config(a.config or HERE / "config.toml", a.quick, a.seed)
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

    run_id, run_dir = make_run_dir(a.out or REPO / "results" / "0003-baseline",
                                   f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{config_hash(cfg)[:8]}")
    return train(cfg, run_dir, run_id, train_cases, val_cases, device=device, quick=a.quick)


if __name__ == "__main__":
    sys.exit(main())
