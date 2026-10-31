"""E1 GPU memory and speed test (spec 0003, AC-6): the full config on real cached cases.

    uv run python experiments/0003-baseline/memory_test.py    # human, on the GPU, before train.py
In one process, through the same common/training.py code as train.py: [memory_test] steps
training steps on the train fold, a val phase on the 2 largest val cases, then 20 more steps.
PASS when the peak reserved VRAM over the whole process is below [memory_test] limit_mib.
Writes results/0003-baseline/memory_test.json (peaks, s/iter over steps 51 on, data wait per
step, val seconds, projected hours for the full run). Never reads the test fold.
Exit 0 pass, 1 peak at or above the limit, OOM or no GPU (a CPU run never passes), 2 bad config
or cache guard.
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cache import cache_dir, check_cache, read_manifest, repo_path
from common.data import fold_cases
from common.runlog import config_hash, load_config
from common.seed import set_seed
from common.training import SELECTION_PATTERN, Trainer, check_config, validate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXTRA_STEPS = 20
SKIP_STEPS = 50  # warm up: s/iter is the mean over the steps after these
N_VAL_CASES = 2
KEYS = ("peak_vram_allocated_mib", "peak_vram_reserved_mib", "seconds_per_iteration",
        "data_wait_s_per_step", "val_cases", "val_seconds", "projected_hours", "config_hash",
        "pass")


def config_cases(cfg):
    """(train, val) case IDs of the config: fold_cases() with the [data] filters."""
    d = cfg["data"]
    splits = repo_path(d["splits_csv"])
    return (fold_cases(splits, "train", d["train_timepoints"], d["exclude_flair_flag"]),
            fold_cases(splits, "val", d["val_timepoints"], d["exclude_flair_flag"]))


def largest_val_cases(cfg, val_cases, n=N_VAL_CASES):
    """The n val cases with the most cropped voxels in manifest.csv, ties by sorted case ID."""
    manifest = read_manifest(cache_dir(cfg))
    size = lambda c: math.prod(int(manifest[c][k]) for k in ("shape_d", "shape_h", "shape_w"))
    return sorted(sorted(val_cases), key=size, reverse=True)[:n]


def projected_hours(s_per_it, val_seconds, n_val, t):
    """Training plus val time of the full run; val uses the largest cases, so it overestimates."""
    return (s_per_it * t["iterations"]
            + val_seconds * n_val * t["iterations"] / t["val_every"]) / 3600


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=HERE / "config.toml")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--out", default=REPO / "results" / "0003-baseline")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    bad = check_config(cfg)
    if bad:
        print("ERROR: bad config:\n" + "\n".join("  " + e for e in bad))
        return 2
    train_cases, val_cases = config_cases(cfg)
    errors = check_cache(cfg, train_cases + val_cases)
    if errors:
        print(f"ERROR: cache guard, nothing run ({len(errors)} problem(s)):")
        print("\n".join("  " + e for e in errors[:20]))
        return 2
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu")
    t, n = cfg["train"], cfg["memory_test"]["steps"]
    limit = cfg["memory_test"]["limit_mib"]
    val_big = largest_val_cases(cfg, val_cases)
    print(f"memory test: {n} + {EXTRA_STEPS} steps, val on {', '.join(val_big)}, "
          f"device {device.type}, config hash {config_hash(cfg)[:12]}")

    gen = set_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    steps, val_times, failure = [], [], None
    try:
        tr = Trainer(cfg, device, train_cases, gen)
        steps = [tr.step(i) for i in range(1, n + 1)]
        for c in val_big:
            v0 = time.time()
            validate(tr.model, [c], (SELECTION_PATTERN,), cfg, device)
            val_times.append(time.time() - v0)
        for i in range(n + 1, n + EXTRA_STEPS + 1):
            tr.step(i)
    except torch.cuda.OutOfMemoryError as e:
        failure = f"CUDA out of memory: {str(e).splitlines()[0]}"

    timed = steps[SKIP_STEPS:] if len(steps) > SKIP_STEPS else steps
    s_it = statistics.mean(s["seconds"] for s in timed) if timed else None
    wait = statistics.mean(s["data_wait"] for s in timed) if timed else None
    val_s = statistics.mean(val_times) if val_times else None
    peaks = ((torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20)
             if device.type == "cuda" else (None, None))
    if failure is None and peaks[1] is None:
        failure = "no GPU, peak VRAM not measured (a CPU run never passes)"
    if failure is None and peaks[1] >= limit:
        failure = f"peak reserved VRAM {peaks[1]:.1f} MiB >= {limit}"
    hours = (projected_hours(s_it, val_s, len(val_cases), t)
             if s_it is not None and val_s is not None else None)
    result = dict(zip(KEYS, (*peaks, s_it, wait, val_big, val_s, hours, config_hash(cfg),
                             failure is None)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "memory_test.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out / "memory_test.json")
    fmt = lambda v, d: "n/a" if v is None else f"{v:.{d}f}"
    print(f"peak reserved {fmt(peaks[1], 1)} MiB (allocated {fmt(peaks[0], 1)}), "
          f"{fmt(s_it, 3)} s/iter, data wait {fmt(wait, 3)} s/step, "
          f"val {fmt(val_s, 1)} s/case, projected {fmt(hours, 1)} h")
    print("PASS" if failure is None else f"FAIL: {failure}")
    return 0 if failure is None else 1


if __name__ == "__main__":
    sys.exit(main())
