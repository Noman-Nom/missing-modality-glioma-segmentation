"""A compare folder -> seven LaTeX files in writing/tables/ (spec 0006, AC-29).

    uv run python experiments/0007-confirmation/make_table.py --compare <compare dir>
Writes 0007-confirmation-<fold>-summary.tex, -cost.tex, -exposure.tex, -patterns-dice.tex,
-patterns-hd95.tex, -seeds.tex and -volume.tex. No number is typed by hand, and no label is
decided here: H2, H3 and H4 words come from verdict.txt. Reads only the compare folder. Exit 1, leaving
older tables untouched, unless run.json is there with status ok, and every other file is there
with its row count. -seeds.tex and -volume.tex run over a page, so they use the longtable package.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import REGIONS
from common.missing import PATTERNS
from common.runlog import load_run

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "writing" / "tables"
NOTE = {
    "val": ("Development fold, not a test result. Each checkpoint was chosen on this fold at "
            "complete input, so pattern 1 is slightly optimistic."),
    "test": "Test fold, opened once.",
}
METRICS = ("dice", "hd95_mm")
METRIC_LABEL = {"dice": "Dice", "hd95_mm": "HD95 (mm)"}
DECIMALS = {"dice": 3, "hd95_mm": 1}
LEVEL_LABEL = {"incomplete": "Incomplete", "complete": "Complete"}
H2_COMPARISONS = ("E3-E1", "E4-E1", "E4-E3")
H3_COMPARISONS = ("E3-E1", "E4-E1")
SEED_COMPARISONS = (*H2_COMPARISONS, "E3b-E3", "E3b-E1")
PATTERN_COMPARISONS = (*H2_COMPARISONS, "E3b-E3")
ARMS = ("E1", "E3", "E4", "E3b")
SEEDS = 3
FINE = 4  # places for the cost and exposure lines, because the margin is 0.015
STAR = "$^{*}$"
DAGGER = r"$^{\dagger}$"
SUMMARY_ROWS = 87  # 72 H2, 6 H3, 6 H4, 3 descriptive
COST_HEAD = (r"Comparison & Region & Mean & Median & 95\% CI & Min margin & Seeds agree & "
             r"Seeds within & $p$ & Label \\")
EXPOSURE_HEAD = (r"Region & Mean, pattern 1 & Median, pattern 1 & $p$ & 90\% lower, incomplete & "
                 r"Seeds agree & Seeds not worse & H4 & E3b-E1 & 95\% CI \\")
SPLIT_NOTE = (" A line reads split when the averaged test passed but the seeds did not all back "
              "it, so no claim is made.")
H2_LINE = re.compile(r"^H2[abc] (\w+): (supported|not supported|split) \((\S+?),.* "
                     r"seeds_agree=(\d+/\d+)\)$")
H3_LINE = re.compile(r"^H3[ab] (\w+): (\w+) \((\S+?),.* seeds_agree=(\d+/\d+), "
                     r"seeds_within=(\d+/\d+),")
H4_LINE = re.compile(r"^H4 (\w+): (supported|not supported|split) \(.* seeds_agree=(\d+/\d+), "
                     r"seeds_not_worse=(\d+/\d+)\)$")


def finished_run(folder):
    """run.json of a finished comparison, checked before any other file is read."""
    if not (Path(folder) / "run.json").is_file():
        raise SystemExit(f"ERROR: {Path(folder) / 'run.json'} missing")
    rec = load_run(folder)
    if rec.get("status") != "ok":
        raise SystemExit(f"ERROR: compare status is {rec.get('status')!r}, expected 'ok'")
    return rec


def read_rows(path, keys, expected):
    """A CSV as {key tuple: row}. Raises SystemExit if missing or on a wrong row count."""
    if not Path(path).is_file():
        raise SystemExit(f"ERROR: {path} missing")
    with open(path, newline="") as f:
        rows = {tuple(r[k] for k in keys): r for r in csv.DictReader(f)}
    if len(rows) != expected:
        raise SystemExit(f"ERROR: {len(rows)} rows in {Path(path).name}, expected {expected}")
    return rows


def read_compare(folder):
    """The four CSVs, with their row counts checked, and pattern ids as ints."""
    folder = Path(folder)
    n_regions = len(REGIONS)
    summary = read_rows(folder / "summary.csv", ("family", "metric", "region", "level",
                                                 "comparison"), SUMMARY_ROWS)
    patterns = read_rows(folder / "patterns.csv", ("metric", "region", "pattern_id", "comparison"),
                         len(METRICS) * n_regions * len(PATTERNS) * len(PATTERN_COMPARISONS))
    per_seed = read_rows(folder / "per_seed.csv", ("seed", "metric", "region", "level",
                                                   "comparison"),
                         SEEDS * len(METRICS) * n_regions * len(LEVEL_LABEL)
                         * len(SEED_COMPARISONS))
    volume = read_rows(folder / "volume.csv", ("arm", "seed", "pattern_id"),
                       len(ARMS) * SEEDS * len(PATTERNS))
    return (summary, {(m, g, int(p), c): r for (m, g, p, c), r in patterns.items()}, per_seed,
            {(a, int(s), int(p)): r for (a, s, p), r in volume.items()})


def read_verdict(folder):
    """As written by compare_seeds.py, so this file makes no decision: {(comparison, region):
    (word, seeds_agree)} of H2, {(comparison, region): (label, seeds_agree, seeds_within)} of H3,
    and {region: (word, seeds_agree, seeds_not_worse)} of H4."""
    path = Path(folder) / "verdict.txt"
    if not path.is_file():
        raise SystemExit(f"ERROR: {path} missing")
    lines = path.read_text().splitlines()
    h2 = {(m[3], m[1]): (m[2], m[4]) for m in map(H2_LINE.match, lines) if m}
    h3 = {(m[3], m[1]): (m[2], m[4], m[5]) for m in map(H3_LINE.match, lines) if m}
    h4 = {m[1]: (m[2], m[3], m[4]) for m in map(H4_LINE.match, lines) if m}
    want = (len(H2_COMPARISONS) * len(REGIONS), len(H3_COMPARISONS) * len(REGIONS), len(REGIONS))
    if (len(h2), len(h3), len(h4)) != want:
        raise SystemExit(f"ERROR: {len(h2)} H2, {len(h3)} H3 and {len(h4)} H4 lines in verdict.txt, "
                         f"expected {want[0]}, {want[1]} and {want[2]}")
    return h2, h3, h4


def star(row, alpha):
    return STAR if float(row["p_holm"]) < alpha else ""


def interval(row, places):
    return f"[{float(row['ci_low']):.{places}f}, {float(row['ci_high']):.{places}f}]"


def seeds_text(rec):
    *head, last = rec["seeds"]
    return f"seeds {', '.join(str(s) for s in head)} and {last}"


def provenance(rec):
    return [f"% generated by experiments/0007-confirmation/make_table.py from {rec['compare_id']}",
            f"% runs {' '.join(rec['run_ids'].values())}",
            f"% git {rec['git_commit']}. Do not edit."]


def caption(rec, what, note):
    return (f"{NOTE[rec['fold']]} {what}, {rec['fold']} fold, n = {rec['n_cases']}, "
            f"{seeds_text(rec)}.{note}")


def table(rec, what, name, columns, head, rows, note=""):
    """The shared booktabs layout. head and rows are lists of finished LaTeX lines."""
    return "\n".join([
        *provenance(rec),
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption(rec, what, note)}}}",
        rf"\label{{tab:confirmation-{rec['fold']}-{name}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        *head,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        "}",
        r"\end{table}",
    ]) + "\n"


def long_table(rec, what, name, columns, head, rows, note=""):
    """The same layout for a table that runs over a page: the head repeats on each page."""
    return "\n".join([
        *provenance(rec),
        r"{\footnotesize",
        rf"\begin{{longtable}}{{{columns}}}",
        rf"\caption{{{caption(rec, what, note)}}}\label{{tab:confirmation-{rec['fold']}-{name}}}\\",
        r"\toprule",
        *head,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        *head,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        *rows,
        r"\end{longtable}",
        "}",
    ]) + "\n"


def group_head(groups, span, first, sub):
    """The two header lines: one spanning label per group, then the column names."""
    spanning = " & ".join(rf"\multicolumn{{{span}}}{{c}}{{{g}}}" for g in groups)
    blank = " & ".join("" for _ in first)
    return [rf"{blank} & {spanning} \\", " & ".join([*first, *sub * len(groups)]) + r" \\"]


def dagger(h2, metric, region, comparison):
    """A dagger on a Dice cell whose H2 line reads split (AC-32). HD95 has no H2 line."""
    return DAGGER if metric == "dice" and h2[comparison, region][0] == "split" else ""


def summary_table(summary, h2, rec):
    """The H2 primary score, on the seed averaged values: mean_diff with its star and CI."""
    rows = []
    for m in METRICS:
        for g in REGIONS:
            cells = []
            for c in H2_COMPARISONS:
                row = summary["H2", m, g, "incomplete", c]
                diff = f"{float(row['mean_diff']):+.{DECIMALS[m]}f}"
                cells += [f"{diff}{star(row, rec['alpha'])}{dagger(h2, m, g, c)}",
                          interval(row, DECIMALS[m])]
            rows.append(" & ".join([METRIC_LABEL[m], g, *cells]) + r" \\")
    return table(rec, "Paired difference of the mean over the 14 incomplete patterns, each subject "
                 "averaged over the seeds", "summary", "ll" + "rr" * len(H2_COMPARISONS),
                 group_head(H2_COMPARISONS, 2, ["Metric", "Region"], ["Diff", "CI"]), rows,
                 f" Star: the paired test alone, Holm adjusted two sided p below {rec['alpha']:g} "
                 "over the 3 comparisons, in either direction. Dagger: the seeds did not all back "
                 "it, so no claim is made.")


def cost_table(summary, h3, rec):
    """H3: Dice at pattern 1. An inconclusive line is never no cost (AC-14)."""
    rows = []
    for c in H3_COMPARISONS:
        for g in REGIONS:
            r = summary["H3", "dice", g, "complete", c]
            word, agree, within = h3[c, g]
            low, high = float(r["ci_low"]), float(r["ci_high"])
            rows.append(" & ".join([
                c, g, f"{float(r['mean_diff']):+.{FINE}f}", f"{float(r['median_diff']):+.{FINE}f}",
                interval(r, FINE), f"{max(abs(low), abs(high)):.{FINE}f}", agree, within,
                f"{float(r['p_raw']):.3g}", word]) + r" \\")
    return table(rec, "Complete input cost: paired Dice difference at pattern 1", "cost",
                 "llrrrrrrrl",
                 [COST_HEAD], rows,
                 f" Equivalence margin {rec['equiv_margin']:g} Dice. The label comes from the CI "
                 f"and the seeds; the p value is reported beside it.{SPLIT_NOTE}")


def exposure_table(summary, h4, rec):
    """H4 per region, with E3b-E1 beside it as a descriptive column with no verdict (AC-15)."""
    rows = []
    for g in REGIONS:
        c = summary["H4", "dice", g, "complete", "E3b-E3"]
        primary = summary["H4", "dice", g, "incomplete", "E3b-E3"]
        e1 = summary["descriptive", "dice", g, "complete", "E3b-E1"]
        word, agree, not_worse = h4[g]
        rows.append(" & ".join([
            g, f"{float(c['mean_diff']):+.{FINE}f}", f"{float(c['median_diff']):+.{FINE}f}",
            f"{float(c['p_raw']):.3g}", f"{float(primary['ci_low']):+.{FINE}f}", agree, not_worse,
            word,
            f"{float(e1['mean_diff']):+.{FINE}f}", interval(e1, FINE)]) + r" \\")
    return table(rec, "Exposure arm: E3b against E3 on Dice", "exposure", "lrrrrrrlrr",
                 [EXPOSURE_HEAD], rows,
                 f" E3b-E1 is descriptive and has no verdict.{SPLIT_NOTE}")


def pattern_table(patterns, rec, metric, stat, what, name):
    """stat of the four arms per pattern and region. A star on E3 or E4 is its test against E1, a
    star on E3b its test against E3."""
    places = DECIMALS[metric]
    rows = []
    for p, pattern in PATTERNS.items():
        cells = []
        for g in REGIONS:
            e3, e4 = patterns[metric, g, p, "E3-E1"], patterns[metric, g, p, "E4-E1"]
            e3b = patterns[metric, g, p, "E3b-E3"]
            cells += [f"{float(e3[f'{stat}_second']):.{places}f}",
                      f"{float(e3[f'{stat}_first']):.{places}f}{star(e3, rec['alpha'])}",
                      f"{float(e4[f'{stat}_first']):.{places}f}{star(e4, rec['alpha'])}",
                      f"{float(e3b[f'{stat}_first']):.{places}f}{star(e3b, rec['alpha'])}"]
        rows.append(" & ".join([str(p), pattern.name, *cells]) + r" \\")
    return table(rec, what, name, "rl" + "rrrr" * len(REGIONS),
                 group_head(REGIONS, 4, ["ID", "Present"], list(ARMS)), rows,
                 f" Star: Holm adjusted two sided p below {rec['alpha']:g} over the 15 patterns, "
                 "in either direction, for E3 and E4 against E1 and for E3b against E3.")


def seeds_table(per_seed, rec):
    """Each seed on its own: mean and median difference, one row per (metric, level, region,
    comparison). No p value (AC-12)."""
    seeds = rec["seeds"]
    rows = []
    for m in METRICS:
        for level, level_label in LEVEL_LABEL.items():
            for g in REGIONS:
                for c in SEED_COMPARISONS:
                    found = [per_seed[str(s), m, g, level, c] for s in seeds]
                    rows.append(" & ".join([
                        METRIC_LABEL[m], level_label, g, c,
                        *(f"{float(r[k]):+.{DECIMALS[m]}f}" for k in ("mean_diff", "median_diff")
                          for r in found)]) + r" \\")
    return long_table(rec, "Paired difference per seed, with no pooling", "seeds",
                      "llll" + "r" * 2 * len(seeds),
                      group_head(("Mean", "Median"), len(seeds),
                                 ["Metric", "Level", "Region", "Comparison"],
                                 [str(s) for s in seeds]), rows)


def volume_table(volume, rec):
    """Predicted minus true WT volume in mL per arm and pattern, one column per seed: the mean
    difference and its limits of agreement. Descriptive, no test (AC-16)."""
    seeds = rec["seeds"]
    rows = []
    for arm in ARMS:
        for p in PATTERNS:
            cells = []
            for s in seeds:
                r = volume[arm, s, p]
                cells.append(f"{float(r['mean_diff_ml']):+.1f} [{float(r['loa_low_ml']):.1f}, "
                             f"{float(r['loa_high_ml']):.1f}]")
            rows.append(" & ".join([arm, str(p), *cells]) + r" \\")
    return long_table(rec, "Predicted minus true whole tumour volume in mL, mean with limits of "
                      "agreement", "volume", "lr" + "r" * len(seeds),
                      [" & ".join(["Arm", "ID", *(f"Seed {s}" for s in seeds)]) + r" \\"], rows,
                      " Pattern IDs as in the Dice table. Descriptive, no test.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compare", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args(argv)
    rec = finished_run(a.compare)
    summary, patterns, per_seed, volume = read_compare(a.compare)
    h2, h3, h4 = read_verdict(a.compare)
    files = {"summary": summary_table(summary, h2, rec),
             "cost": cost_table(summary, h3, rec),
             "exposure": exposure_table(summary, h4, rec),
             "patterns-dice": pattern_table(patterns, rec, "dice", "mean",
                                            "Mean Dice of each arm under each pattern",
                                            "patterns-dice"),
             "patterns-hd95": pattern_table(patterns, rec, "hd95_mm", "median",
                                            "Median HD95 in mm of each arm under each pattern",
                                            "patterns-hd95"),
             "seeds": seeds_table(per_seed, rec),
             "volume": volume_table(volume, rec)}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, tex in files.items():
        path = a.out_dir / f"0007-confirmation-{rec['fold']}-{name}.tex"
        path.write_text(tex)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
