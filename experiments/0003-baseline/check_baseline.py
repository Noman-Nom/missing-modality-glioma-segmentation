"""Experiment check for an E1 run folder (spec 0003, AC-9, AC-16).

    uv run python experiments/0003-baseline/check_baseline.py <run dir>
Prints PASS (exit 0) or FAIL: <reason> (exit 1). Run it after evaluate.py. A quick run never
passes: after the structure checks it always prints FAIL: quick run.
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cache import cache_dir, preprocess_hash, read_cache_json, repo_path
from common.data import REGIONS, fold_cases
from common.runlog import config_hash, load_config, load_run
from common.training import RUN_KEYS

COLUMNS = ["run_id", "case", "subject", "region", "dice", "hd95_mm", "sensitivity", "precision",
           "gt_empty", "pred_empty"]
FULL_ITERATIONS = 40000  # AC-7
VAL_POINTS = 10  # AC-8: every 4,000 iterations
VRAM_LIMIT_MIB = 3584  # AC-6
BEST_SCORE_MIN = 0.60  # AC-9, set before any E1 number was seen
LOSS_WINDOW = 50  # AC-9
N_TRAIN, N_VAL = 863, 114  # AC-1


class Fail(Exception):
    pass


def need(ok, reason):
    if not ok:
        raise Fail(reason)


def read_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def structure(run):
    """Checks that hold for quick and full runs. Returns (run.json, config, metrics rows)."""
    need((run / "run.json").is_file(), "run.json missing")
    rec = load_run(run)
    need(set(rec) == set(RUN_KEYS), f"run.json keys differ: {sorted(set(rec) ^ set(RUN_KEYS))}")
    need(rec["status"] != "running",
         "status running with no result (crashed run; train.py --resume <run dir>)")
    need(rec["status"] == "ok", f"status {rec['status']}: {rec['reason']}")
    need((run / "config.toml").is_file(), "config.toml missing")
    cfg = load_config(run / "config.toml")
    need(config_hash(cfg) == rec["config_hash"], "config.toml hash differs from run.json")
    info = read_cache_json(cache_dir(cfg))
    need(rec["cache_hash"] == preprocess_hash(cfg), "cache hash differs from the run's config")
    need(info is not None and info["preprocess_hash"] == rec["cache_hash"],
         "cache.json hash differs from run.json cache_hash")
    _, log = read_csv(run / "train_log.csv")
    need([int(r["iteration"]) for r in log] == list(range(1, rec["iterations"] + 1)),
         "train_log.csv does not hold one row per iteration from 1 to iterations")
    need((run / "checkpoints" / "best.pt").is_file(), "checkpoints/best.pt missing")
    test = set(fold_cases(repo_path(cfg["data"]["splits_csv"]), "test"))
    leaked = sorted(test & set(rec["train_cases"] + rec["val_cases"]))
    need(not leaked, f"AC-4: test fold cases used: {leaked[:3]}")
    need((run / "metrics_val.csv").is_file(), "metrics_val.csv missing (run evaluate.py)")
    cols, rows = read_csv(run / "metrics_val.csv")
    need(cols == COLUMNS, f"metrics_val.csv columns {cols}")
    expected = len(rec["val_cases"]) * len(REGIONS)
    need(len(rows) == expected, f"{len(rows)} metrics rows, expected {expected}")
    need({r["case"] for r in rows} == set(rec["val_cases"]), "metrics cases differ from val_cases")
    for r in rows:
        for k in ("dice", "hd95_mm"):
            need(r[k] != "" and not math.isnan(float(r[k])), f"{k} NaN in {r['case']} {r['region']}")
    return rec, cfg, log


def full(rec, cfg, log):
    """The full run thresholds (AC-6, AC-7, AC-8, AC-9, AC-1)."""
    need(rec["iterations"] == FULL_ITERATIONS,
         f"{rec['iterations']} iterations, expected {FULL_ITERATIONS}")
    peak = rec["peak_vram_reserved_mib"]
    need(peak is not None, "peak VRAM not logged (not a GPU run?)")
    need(peak < VRAM_LIMIT_MIB, f"peak reserved VRAM {peak:.1f} MiB >= {VRAM_LIMIT_MIB}")
    points = [v["iteration"] for v in rec["val_history"]]
    need(len(points) == VAL_POINTS, f"{len(points)} val points, expected {VAL_POINTS}")
    need(rec["best_iteration"] in points, f"best_iteration {rec['best_iteration']} not a val point")
    losses = [float(r["loss"]) for r in log]
    w = LOSS_WINDOW
    start, end = sum(losses[:w]) / w, sum(losses[-w:]) / w
    need(math.isclose(start, rec["loss_start_mean"], rel_tol=1e-9)
         and math.isclose(end, rec["loss_end_mean"], rel_tol=1e-9),
         "loss means in run.json do not match train_log.csv")
    need(end < start, f"AC-9: loss did not fall (first {w}: {start:.4f}, last {w}: {end:.4f})")
    need(rec["best_score"] > BEST_SCORE_MIN,
         f"AC-9: best score {rec['best_score']:.4f} <= {BEST_SCORE_MIN}")
    d = cfg["data"]
    splits = repo_path(d["splits_csv"])
    train = fold_cases(splits, "train", d["train_timepoints"], d["exclude_flair_flag"])
    val = fold_cases(splits, "val", d["val_timepoints"], d["exclude_flair_flag"])
    need(rec["train_cases"] == train and len(train) == N_TRAIN,
         f"train cases: {len(rec['train_cases'])} used, expected the {N_TRAIN} of fold_cases()")
    need(rec["val_cases"] == val and len(val) == N_VAL,
         f"val cases: {len(rec['val_cases'])} used, expected the {N_VAL} of fold_cases()")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: check_baseline.py <run dir>")
        return 1
    run = Path(argv[0])
    try:
        rec, cfg, log = structure(run)
        if rec["quick"]:
            print("structure checks passed")
            raise Fail("quick run")
        full(rec, cfg, log)
    except Fail as e:
        print(f"FAIL: {e}")
        return 1
    print(f"run {rec['run_id']}: peak reserved {rec['peak_vram_reserved_mib']:.1f} MiB, "
          f"best score {rec['best_score']:.4f} at iteration {rec['best_iteration']}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
