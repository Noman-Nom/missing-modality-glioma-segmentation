"""Experiment check for a tracer run folder (spec 0001, AC-13).

    uv run python experiments/0002-tracer-bullet/check_tracer.py <run dir>
Prints PASS (exit 0) or FAIL: <reason> (exit 1). A quick run never passes: after the structure
checks it always prints FAIL: quick run.
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import REGIONS
from common.missing import PATTERNS
from common.runlog import RUN_KEYS, load_run

COLUMNS = ["run_id", "case", "subject", "pattern_id", "pattern", "region", "dice", "hd95_mm",
           "sensitivity", "precision", "gt_empty", "pred_empty"]
FULL_ITERATIONS = 2000  # AC-5
VRAM_LIMIT_MIB = 4096  # AC-5
WT_DICE_MIN = 0.3  # AC-6


class Fail(Exception):
    pass


def need(ok, reason):
    if not ok:
        raise Fail(reason)


def structure(run):
    """Checks that hold for quick and full runs. Returns (run.json, metrics rows)."""
    need((run / "run.json").is_file(), "run.json missing")
    rec = load_run(run)
    need(set(rec) == set(RUN_KEYS), f"run.json keys differ: {sorted(set(rec) ^ set(RUN_KEYS))}")
    need(rec["status"] != "running", "status running with no result (crashed run)")
    need(rec["status"] == "ok", f"status {rec['status']}: {rec['reason']}")
    need((run / "config.toml").is_file(), "config.toml missing")
    need((run / "metrics.csv").is_file(), "metrics.csv missing (run evaluate.py)")
    with open(run / "metrics.csv", newline="") as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames == COLUMNS, f"metrics.csv columns {reader.fieldnames}")
        rows = list(reader)
    n_val = len(rec["val_cases"])
    expected = n_val * len(PATTERNS) * len(REGIONS)
    need(len(rows) == expected, f"{len(rows)} metrics rows, expected {expected}")
    need({int(r["pattern_id"]) for r in rows} == set(PATTERNS), "pattern IDs are not 1 to 15")
    need({r["region"] for r in rows} == set(REGIONS), "regions are not WT, TC, ET")
    need({r["case"] for r in rows} == set(rec["val_cases"]), "metrics cases differ from val_cases")
    need({r["run_id"] for r in rows} == {rec["run_id"]}, "metrics run_id differs from run.json")
    for r in rows:
        for k in ("dice", "hd95_mm"):
            need(r[k] != "" and not math.isnan(float(r[k])), f"{k} NaN in {r['case']} {r['region']}")
    return rec, rows


def full(run, rec, rows):
    """AC-5 and AC-6 thresholds, only for a full run."""
    need(rec["iterations"] == FULL_ITERATIONS,
         f"AC-5: {rec['iterations']} iterations, expected {FULL_ITERATIONS}")
    peak = rec["peak_vram_reserved_mib"]
    need(peak is not None, "AC-5: peak VRAM not logged (not a GPU run?)")
    need(peak < VRAM_LIMIT_MIB, f"AC-5: peak reserved VRAM {peak:.1f} MiB >= {VRAM_LIMIT_MIB}")
    with open(run / "train_log.csv", newline="") as f:
        losses = [float(r["loss"]) for r in csv.DictReader(f)]
    w = min(50, len(losses) // 2)
    start, end = sum(losses[:w]) / w, sum(losses[-w:]) / w
    need(math.isclose(start, rec["loss_start_mean"], rel_tol=1e-9)
         and math.isclose(end, rec["loss_end_mean"], rel_tol=1e-9),
         "AC-6: loss means in run.json do not match train_log.csv")
    need(end < start, f"AC-6: loss did not fall (first {w}: {start:.4f}, last {w}: {end:.4f})")
    wt = [float(r["dice"]) for r in rows if int(r["pattern_id"]) == 1 and r["region"] == "WT"]
    mean_wt = sum(wt) / len(wt)
    need(mean_wt > WT_DICE_MIN, f"AC-6: pattern 1 WT Dice {mean_wt:.4f} <= {WT_DICE_MIN}")
    return mean_wt


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: check_tracer.py <run dir>")
        return 1
    run = Path(argv[0])
    try:
        rec, rows = structure(run)
        if rec["quick"]:
            print("structure checks passed")
            raise Fail("quick run")
        mean_wt = full(run, rec, rows)
    except Fail as e:
        print(f"FAIL: {e}")
        return 1
    print(f"run {rec['run_id']}: peak reserved {rec['peak_vram_reserved_mib']:.1f} MiB, "
          f"pattern 1 WT Dice {mean_wt:.4f}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
