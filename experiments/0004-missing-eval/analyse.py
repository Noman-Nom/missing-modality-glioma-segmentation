"""metrics.csv of one eval folder -> stats.csv: each incomplete pattern against pattern 1 (spec 0004).

    uv run python experiments/0004-missing-eval/analyse.py --eval <eval dir>
Reads only metrics.csv. Uses no GPU. Exit 1 if any (case, pattern, region) row is missing. Dice
only for now, and ci_low and ci_high stay empty: HD95, Friedman, the marginal effect, the
bootstrap and the verdict come with the full statistics milestone.
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import REGIONS
from common.missing import PATTERNS
from common.stats import holm, paired_wilcoxon

COLUMNS = ["metric", "region", "pattern_id", "pattern_name", "n", "mean", "median", "mean_diff",
           "ci_low", "ci_high", "n_nonzero", "wilcoxon_stat", "p_raw", "p_holm", "n_pred_empty"]
METRICS = ("dice",)
BASE = 1  # complete input
TESTED = [p for p in PATTERNS if p != BASE]


def load_metrics(path):
    """metrics.csv as a DataFrame. Raises SystemExit unless every row is there exactly once."""
    if not Path(path).is_file():
        raise SystemExit(f"ERROR: {path} missing")
    df = pd.read_csv(path)
    have = list(zip(df["case"], df["pattern_id"], df["region"], strict=True))
    want = {(c, p, r) for c in df["case"].unique() for p in PATTERNS for r in REGIONS}
    missing = sorted(want - set(have))
    if len(have) != len(want) or set(have) != want:
        raise SystemExit(f"ERROR: metrics.csv has {len(have)} rows, expected {len(want)}; "
                         f"missing {missing[:3]}")
    return df


def region_rows(df, metric, region):
    """The 15 stats.csv rows of one (metric, region), pattern IDs ascending."""
    sub = df[df["region"] == region]
    wide = sub.pivot(index="case", columns="pattern_id", values=metric)
    empty = sub.groupby("pattern_id")["pred_empty"].sum()
    d = {p: (wide[p] - wide[BASE]).to_numpy() for p in TESTED}
    tests = {p: paired_wilcoxon(d[p]) for p in TESTED}
    p_holm = dict(zip(TESTED, holm([tests[p][1] for p in TESTED]), strict=True))
    rows = []
    for p in PATTERNS:
        row = dict.fromkeys(COLUMNS, "") | {
            "metric": metric, "region": region, "pattern_id": p,
            "pattern_name": PATTERNS[p].name, "n": len(wide), "mean": f"{wide[p].mean():.6f}",
            "median": f"{wide[p].median():.6f}", "n_pred_empty": int(empty[p])}
        if p != BASE:
            row |= {"mean_diff": f"{d[p].mean():.6f}", "n_nonzero": int((d[p] != 0).sum()),
                    "wilcoxon_stat": f"{tests[p][0]:.6f}", "p_raw": f"{tests[p][1]:.6g}",
                    "p_holm": f"{p_holm[p]:.6g}"}
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", required=True, type=Path)
    a = ap.parse_args(argv)
    df = load_metrics(a.eval / "metrics.csv")
    rows = [r for m in METRICS for g in REGIONS for r in region_rows(df, m, g)]
    with open(a.eval / "stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{df['case'].nunique()} cases -> {len(rows)} rows in {a.eval / 'stats.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
