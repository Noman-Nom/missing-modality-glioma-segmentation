"""metrics.csv of one eval folder -> the statistics of spec 0004 (AC-12, AC-13).

    uv run python experiments/0004-missing-eval/analyse.py --eval <eval dir>
Reads only metrics.csv, config.toml and run.json, and writes stats.csv (90 rows), friedman.csv
(18 rows), marginal.csv (24 rows) and verdict.txt. Uses no GPU. Exit 1, writing nothing, unless
run.json is there, its status is ok, no (case, pattern, region) row is missing and the case count
equals n_cases, checked in that order, and then unless config.toml still hashes to the config_hash
in run.json. The rules are the Statistics protocol of the spec: changing this file changes
protocol_hash, so it costs a val rerun.
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import MODALITIES, REGIONS
from common.missing import BIT, PATTERNS
from common.runlog import config_hash, load_config, load_run
from common.stats import bootstrap_ci, bootstrap_indices, friedman, holm, paired_wilcoxon

STATS_COLUMNS = ["metric", "region", "pattern_id", "pattern_name", "n", "mean", "median",
                 "mean_diff", "ci_low", "ci_high", "n_nonzero", "wilcoxon_stat", "p_raw", "p_holm",
                 "n_pred_empty"]
FRIEDMAN_COLUMNS = ["metric", "region", "n_missing", "n", "statistic", "p_raw", "p_holm",
                    "kendall_w"]
MARGINAL_COLUMNS = ["metric", "region", "modality", "mean_present", "mean_absent", "mean_diff",
                    "ci_low", "ci_high"]
P_COLUMNS = {"p_raw", "p_holm"}
METRICS = ("dice", "hd95_mm")
BASE = 1  # complete input
TESTED = [p for p in PATTERNS if p != BASE]
# patterns of equal size, so Friedman compares which modalities are missing, not how many
GROUPS = {n: [p.id for p in PATTERNS.values() if len(p.missing) == n] for n in (1, 2, 3)}
ID_OF_MASK = {p.mask: p.id for p in PATTERNS.values()}


def load_finished(eval_dir):
    """run.json of a finished sweep. A sweep that died between two cases leaves every remaining
    row whole, so only the status can refuse it (AC-12)."""
    if not (eval_dir / "run.json").is_file():
        raise SystemExit(f"ERROR: {eval_dir / 'run.json'} missing")
    rec = load_run(eval_dir)
    if rec.get("status") != "ok":
        raise SystemExit(f"ERROR: eval status is {rec.get('status')!r}, expected 'ok'")
    return rec


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


def wide_table(df, metric, region):
    """One row per case (sorted, so every bootstrap row is paired), one column per pattern."""
    return df[df["region"] == region].pivot(index="case", columns="pattern_id", values=metric)


def stats_rows(df, metric, region, indices):
    """The 15 stats.csv rows of one (metric, region), pattern IDs ascending, as plain numbers."""
    wide = wide_table(df, metric, region)
    empty = df[df["region"] == region].groupby("pattern_id")["pred_empty"].sum()
    d = {p: (wide[p] - wide[BASE]).to_numpy() for p in TESTED}
    tests = {p: paired_wilcoxon(d[p]) for p in TESTED}
    p_holm = dict(zip(TESTED, holm([tests[p][1] for p in TESTED]), strict=True))
    rows = []
    for p in PATTERNS:
        row = dict.fromkeys(STATS_COLUMNS) | {
            "metric": metric, "region": region, "pattern_id": p,
            "pattern_name": PATTERNS[p].name, "n": len(wide), "mean": wide[p].mean(),
            "median": wide[p].median(), "n_pred_empty": int(empty[p])}
        if p != BASE:
            low, high = bootstrap_ci(d[p], indices)
            row |= {"mean_diff": d[p].mean(), "ci_low": low, "ci_high": high,
                    "n_nonzero": int((d[p] != 0).sum()), "wilcoxon_stat": tests[p][0],
                    "p_raw": tests[p][1], "p_holm": p_holm[p]}
        rows.append(row)
    return rows


def friedman_rows(df, metric, region):
    """The 3 friedman.csv rows of one (metric, region), with Holm over the 3 groups."""
    wide = wide_table(df, metric, region)
    tests = {g: friedman(wide[ids].to_numpy()) for g, ids in GROUPS.items()}
    p_holm = holm([tests[g][1] for g in GROUPS])
    return [{"metric": metric, "region": region, "n_missing": g, "n": len(wide),
             "statistic": tests[g][0], "p_raw": tests[g][1], "p_holm": adjusted,
             "kendall_w": tests[g][2]}
            for g, adjusted in zip(GROUPS, p_holm, strict=True)]


def marginal_pairs(modality):
    """The 7 (S + m, S) pattern pairs of a modality: each pair differs by exactly that modality."""
    bit = BIT[modality]
    return [(ID_OF_MASK[m], ID_OF_MASK[m ^ bit]) for m in ID_OF_MASK if m & bit and m != bit]


def marginal_rows(df, metric, region, indices):
    """The 4 marginal.csv rows of one (metric, region). Descriptive, with a CI and no test."""
    wide = wide_table(df, metric, region)
    rows = []
    for m in MODALITIES:
        with_m, without_m = zip(*marginal_pairs(m), strict=True)
        present, absent = wide[list(with_m)].to_numpy(), wide[list(without_m)].to_numpy()
        per_case = (present - absent).mean(axis=1)  # the mean of each case's 7 pair differences
        low, high = bootstrap_ci(per_case, indices)
        rows.append({"metric": metric, "region": region, "modality": m,
                     "mean_present": present.mean(), "mean_absent": absent.mean(),
                     "mean_diff": per_case.mean(), "ci_low": low, "ci_high": high})
    return rows


def verdict(stats, friedman_results, alpha):
    """The H1 verdict lines, on unrounded numbers, with no human judgement (AC-13)."""
    lines = []
    for g in REGIONS:
        failed = [r["pattern_id"] for r in stats
                  if r["metric"] == "dice" and r["region"] == g and r["pattern_id"] != BASE
                  and not (r["mean_diff"] < 0 and r["p_holm"] < alpha)]
        names = ", ".join(str(p) for p in sorted(failed))
        lines.append(f"H1a {g}: " + (f"not supported (patterns {names})" if failed
                                     else "supported"))
    for g in REGIONS:
        p = next(r["p_holm"] for r in friedman_results
                 if r["metric"] == "dice" and r["region"] == g and r["n_missing"] == 1)
        word = "supported" if p < alpha else "not supported"
        lines.append(f"H1b {g}: {word} (n_missing=1, p_holm={p:.6g})")
    return lines


def cell(column, v):
    """A float is %.6f, a p value %.6g, a missing value an empty field (spec 0004, File formats)."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}" if column in P_COLUMNS else f"{v:.6f}"
    return v


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows({k: cell(k, v) for k, v in r.items()} for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", required=True, type=Path)
    a = ap.parse_args(argv)
    rec = load_finished(a.eval)
    df = load_metrics(a.eval / "metrics.csv")
    if df["case"].nunique() != rec.get("n_cases"):
        raise SystemExit(f"ERROR: metrics.csv has {df['case'].nunique()} cases, "
                         f"expected n_cases {rec.get('n_cases')}")
    cfg = load_config(a.eval / "config.toml")
    if config_hash(cfg) != rec.get("config_hash"):
        raise SystemExit("ERROR: config.toml does not hash to the config_hash in run.json")
    cfg = cfg["eval"]
    indices = bootstrap_indices(df["case"].nunique(), cfg["bootstrap_resamples"], cfg["seed"])
    keys = [(m, g) for m in METRICS for g in REGIONS]
    stats = [r for m, g in keys for r in stats_rows(df, m, g, indices)]
    tests = [r for m, g in keys for r in friedman_rows(df, m, g)]
    marginal = [r for m, g in keys for r in marginal_rows(df, m, g, indices)]
    write_csv(a.eval / "stats.csv", STATS_COLUMNS, stats)
    write_csv(a.eval / "friedman.csv", FRIEDMAN_COLUMNS, tests)
    write_csv(a.eval / "marginal.csv", MARGINAL_COLUMNS, marginal)
    lines = verdict(stats, tests, cfg["alpha"])
    (a.eval / "verdict.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"{df['case'].nunique()} cases -> {len(stats)}, {len(tests)} and {len(marginal)} rows "
          f"in {a.eval}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
