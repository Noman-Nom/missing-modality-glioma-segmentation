"""The check of the confirmation study (spec 0006, AC-23).

Built so far: only rescore_problems, the proof of AC-26, so this file has no command yet. The rest
of the check and its command come with build step 7.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import REGIONS
from common.missing import PATTERNS

MAX_DICE_DIFF = 0.0001  # AC-26, per row
MAX_HD95_DIFF = 0.01  # AC-26, per row, in mm
MAX_TRUE_ML_DIFF = 0.1  # AC-26, WT, against ml_WT in splits.csv
WT = "WT"


def _rows(path):
    """metrics.csv as {(case, pattern_id, region): row}."""
    with open(path, newline="") as f:
        return {(r["case"], int(r["pattern_id"]), r["region"]): r for r in csv.DictReader(f)}


def rescore_problems(new_eval, old_eval, splits_csv):
    """Every failed rule of AC-26 for one seed 42 model, as error strings. The new val eval must
    match the old one on all 15 patterns, row by row, and its true WT volume must match the data
    audit. Empty means the rescoring changed nothing it should not have."""
    new, old = _rows(Path(new_eval) / "metrics.csv"), _rows(Path(old_eval) / "metrics.csv")
    if not new or set(new) != set(old):
        return [f"rows differ from the old eval: {len(set(new) ^ set(old))} key(s)"]
    want = {(c, p, g) for c, _, _ in new for p in PATTERNS for g in REGIONS}
    if set(new) != want:
        return [f"{len(want - set(new))} (case, pattern, region) row(s) missing"]
    errors = []
    for column, limit in (("dice", MAX_DICE_DIFF), ("hd95_mm", MAX_HD95_DIFF)):
        far = sorted(k for k in new if abs(float(new[k][column]) - float(old[k][column])) > limit)
        if far:
            errors.append(f"{len(far)} row(s) of {column} differ by over {limit}, first {far[0]}")
    with open(splits_csv, newline="") as f:
        audit = {r["case"]: float(r["ml_WT"]) for r in csv.DictReader(f)}
    wt = {k[0]: float(r["true_ml"]) for k, r in new.items() if k[2] == WT}
    unknown = sorted(set(wt) - set(audit))
    if unknown:
        errors.append(f"{len(unknown)} case(s) not in {splits_csv}, first {unknown[0]}")
    far = sorted(c for c in wt if c in audit and abs(wt[c] - audit[c]) > MAX_TRUE_ML_DIFF)
    if far:
        errors.append(f"true_ml of {len(far)} case(s) differs from ml_WT by over "
                      f"{MAX_TRUE_ML_DIFF} mL, first {far[0]}")
    return errors
