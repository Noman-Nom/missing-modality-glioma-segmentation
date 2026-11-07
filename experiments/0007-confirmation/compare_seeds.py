"""Twelve eval folders -> the three seed comparison of E1, E3, E4 and E3b (spec 0006).

    uv run python experiments/0007-confirmation/compare_seeds.py --runs \
        experiments/0007-confirmation/runs.toml --fold val
    uv run python experiments/0007-confirmation/compare_seeds.py --runs \
        experiments/0007-confirmation/runs.toml --fold test --preflight
    uv run python experiments/0007-confirmation/compare_seeds.py --runs \
        experiments/0007-confirmation/runs.toml --fold test
Exit 2, printing every failed rule and writing nothing, unless the manifest, the 12 runs and the 12
evals pass AC-10. For each subject, region and pattern, an arm's value is the mean over its 3 seeds,
and every test runs on those means, so n is the number of subjects (AC-11). Writes
<out>/compare-<fold>/ with summary.csv, patterns.csv, per_seed.csv, volume.csv, verdict.txt (18
lines: 9 H2, 6 H3, 3 H4) and run.json, last. All numbers come from metrics.csv of the evals, so it
uses no GPU. A finished folder is refused unless --rerun-reason is given (AC-17).

--fold val needs a committed tree, with no untracked file, and no test eval on disk (AC-10, AC-34).
--fold test also refuses if a sealed file moved since the val compare or the tag, if the commit of
an eval has no passing literature gate, or if that commit is not the one the preflight passed
(AC-28). --preflight runs every guard that needs no test metric (AC-20), and reads no test case, so
it goes before the opening. It writes one file, compare-val/preflight.json, and only on a pass: it
deletes any older one first, so the file always means the last preflight passed. Once a *-test
entry exists in --evals the opening has happened, so it refuses and leaves the file as it is, the
one proof of the commit the opening used (AC-34).

This repeats some of 0005-strategies/compare.py on purpose: experiments never import each other,
and that file stays unchanged (AC-21). A test proves both give the same H2 lines (AC-24).
"""
import argparse
import csv
import json
import shutil
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cache import sha256_file
from common.data import REGIONS
from common.evaluation import (
    PREFLIGHT,
    PREREG_TAG,
    check_rules,
    check_test_gate,
    compare_hash,
    descends_from_tag,
    is_ancestor,
    protocol_hash,
    seal_problems,
    val_file_problems,
    val_record_problems,
)
from common.missing import PATTERNS
from common.runlog import config_hash, git_info, load_config, load_run
from common.stats import bootstrap_ci, bootstrap_indices, holm, paired_wilcoxon
from common.strategies import strategy_name

REPO = Path(__file__).resolve().parents[2]
# protocol_hash() at the tag commit. Repinned there, beside SEALED_HASH; a test keeps them equal.
EXPECTED_PROTOCOL_HASH = "48925eb52e0cbc05ee97ec1715484ad3cff7fc9573d8f3a68d6bf64abfe6745c"
EVAL_CONFIG = Path(__file__).resolve().parents[1] / "0004-missing-eval" / "config.toml"
EQUIV_MARGIN = 0.015  # in Dice, borrowed from Oechsner 2026 (AC-14), a sealed value
ARMS = ("e1", "e3", "e4", "e3b")
ARM_NAMES = {"e1": "E1", "e3": "E3", "e4": "E4", "e3b": "E3b"}
SEEDS = (42, 43, 44)
STRATEGIES = {"e1": None, "e3": "dropout", "e4": "presence", "e3b": "exposure"}
COMPLETE_SHARE = 0.5
METRICS = ("dice", "hd95_mm")
PRIMARY = "incomplete"
COMPLETE = "complete"
VOLUME_REGION = "WT"
LEVELS = {PRIMARY: [p.id for p in PATTERNS.values() if p.missing],
          **{f"missing_{n}": [p.id for p in PATTERNS.values() if len(p.missing) == n]
             for n in (1, 2, 3)}}
SCORES = LEVELS | {COMPLETE: [p.id for p in PATTERNS.values() if not p.missing]}
# (name, first, second): the paired difference is d = value(first) - value(second)
H2_COMPARISONS = (("E3-E1", "e3", "e1"), ("E4-E1", "e4", "e1"), ("E4-E3", "e4", "e3"))
H2_LINES = {"H2a": "E3-E1", "H2b": "E4-E1", "H2c": "E4-E3"}
H3_LINES = {"H3a": ("E3-E1", "e3", "e1"), "H3b": ("E4-E1", "e4", "e1")}
EXPOSURE = ("E3b-E3", "e3b", "e3")
EXPOSURE_E1 = ("E3b-E1", "e3b", "e1")
PATTERN_COMPARISONS = (*H2_COMPARISONS, EXPOSURE)
SEED_COMPARISONS = (*H2_COMPARISONS, EXPOSURE, EXPOSURE_E1)
SAME_RUN_KEYS = ("train_cases", "val_cases", "gpu", "cache_hash")
SAME_EVAL_KEYS = ("fold", "n_cases", "protocol_hash", "config_hash", "splits_sha256", "gpu")
SUMMARY_COLUMNS = ["family", "metric", "region", "level", "comparison", "n", "mean_first",
                   "mean_second", "mean_diff", "median_diff", "ci_low", "ci_high", "ci_level",
                   "n_nonzero", "wilcoxon_stat", "p_raw", "p_holm"]
PATTERN_COLUMNS = ["metric", "region", "pattern_id", "pattern_name", "comparison", "n",
                   "mean_first", "mean_second", "median_first", "median_second", "mean_diff",
                   "median_diff", "ci_low", "ci_high", "n_nonzero", "wilcoxon_stat", "p_raw",
                   "p_holm"]
PER_SEED_COLUMNS = ["seed", "metric", "region", "level", "comparison", "mean_diff", "median_diff"]
VOLUME_COLUMNS = ["arm", "seed", "pattern_id", "pattern_name", "n", "mean_diff_ml", "loa_low_ml",
                  "loa_high_ml"]
RUN_JSON_KEYS = ("compare_id", "status", "fold", "n_cases", "run_ids", "eval_commits",
                 "eval_sha256", "versions_equal", "n_resumes", "seeds", "alpha", "equiv_margin",
                 "compare_hash", "protocol_hash", "splits_sha256", "prereg_tag", "git_commit",
                 "git_dirty", "rerun_reason", "superseded")
P_COLUMNS = {"p_raw", "p_holm"}


def slots():
    """The 12 (arm, seed) pairs, in the order every report uses."""
    return [(arm, seed) for arm in ARMS for seed in SEEDS]


def label(arm, seed):
    return f"{arm} seed{seed}"


def manifest_problems(path, root):
    """Every failed AC-9 and AC-10 rule of runs.toml. Returns (errors, {(arm, seed): run folder}).
    A path must stay under results/, so a run outside the study cannot slip in. A key that is not a
    slot is refused too, because no arm or seed is added after the tag (AC-1)."""
    if not Path(path).is_file():
        return [f"{path} missing"], {}
    try:
        with open(path, "rb") as f:
            manifest = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        return [f"{path} is not valid TOML: {e}"], {}
    tables = {arm: manifest[arm] for arm in ARMS if isinstance(manifest.get(arm), dict)}
    errors = [f"manifest: unknown table [{k}]" for k in manifest if k not in (*ARMS, "replaced")]
    dirs = {}
    for arm in ARMS:
        known = {f"seed{s}" for s in SEEDS}
        errors += [f"manifest: unknown entry {arm}.{k}" for k in tables.get(arm, {})
                   if k not in known]
        for seed in SEEDS:
            value = tables.get(arm, {}).get(f"seed{seed}")
            if not value or not isinstance(value, str):
                errors.append(f"manifest: no entry for {label(arm, seed)}")
            elif Path(value).parts[0] != "results" or ".." in Path(value).parts:
                errors.append(f"manifest: {label(arm, seed)} is not under results/: {value}")
            else:
                dirs[arm, seed] = Path(root) / value
    return errors, dirs


def same_across(items, key, what):
    """One error per item whose value of key differs from the first item's."""
    first_name, first = items[0]
    return [f"{name} {what}: {key} differs from {first_name}" for name, rec in items[1:]
            if rec.get(key) != first.get(key)]


def without(cfg, key):
    return {k: v for k, v in cfg.items() if k != key}


def run_problems(dirs):
    """Every failed AC-10 rule on the 12 run folders. Returns (errors, {slot: run.json}), the
    records empty unless all 12 could be read."""
    missing = [f"{label(*slot)}: {d / name} missing" for slot, d in dirs.items()
               for name in ("run.json", "config.toml") if not (d / name).is_file()]
    if missing:
        return missing, {}
    recs = {slot: load_run(d) for slot, d in dirs.items()}
    cfgs = {slot: load_config(d / "config.toml") for slot, d in dirs.items()}
    errors = []
    seen = {}
    for slot, rec in recs.items():
        arm, seed = slot
        name = label(arm, seed)
        if rec.get("run_id") in seen:
            errors.append(f"{name}: run id repeats {seen[rec.get('run_id')]}")
        seen.setdefault(rec.get("run_id"), name)
        if rec.get("seed") != seed or cfgs[slot].get("seed") != seed:
            errors.append(f"{name}: seed is {rec.get('seed')!r}, config seed is "
                          f"{cfgs[slot].get('seed')!r}, slot says {seed}")
        if rec.get("status") != "ok":
            errors.append(f"{name}: status is {rec.get('status')!r}, not 'ok'")
        if rec.get("quick") is not False:
            errors.append(f"{name}: a quick run")
        if strategy_name(cfgs[slot]) != STRATEGIES[arm]:
            errors.append(f"{name}: strategy is {strategy_name(cfgs[slot])!r}, "
                          f"expected {STRATEGIES[arm]!r}")
        if arm == "e3b" and cfgs[slot].get("strategy", {}).get("complete_share") != COMPLETE_SHARE:
            errors.append(f"{name}: complete_share is not {COMPLETE_SHARE}")
    for arm in ARMS:
        errors += [f"{label(arm, s)}: config differs from {label(arm, SEEDS[0])} outside seed"
                   for s in SEEDS[1:]
                   if without(cfgs[arm, s], "seed") != without(cfgs[arm, SEEDS[0]], "seed")]
    for seed in SEEDS:
        errors += [f"{label(arm, seed)}: config differs from {label('e1', seed)} outside [strategy]"
                   for arm in ARMS[1:]
                   if without(cfgs[arm, seed], "strategy") != without(cfgs["e1", seed], "strategy")]
    items = [(label(*slot), rec) for slot, rec in recs.items()]
    for key in SAME_RUN_KEYS:
        errors += same_across(items, key, "run")
    return errors, recs


def eval_problems(eval_dirs, run_dirs, fold):
    """Every failed AC-10 rule on the 12 eval folders. Returns (errors, {slot: run.json},
    {slot: metrics}), the last two empty unless all 12 could be read."""
    errors = []
    for slot, d in eval_dirs.items():
        try:
            errors += [f"{label(*slot)} eval: {e}" for e in check_rules(d, run_dirs[slot])]
        except OSError as e:
            errors.append(f"{label(*slot)} eval: cannot read {e.filename}")
    if not all((d / name).is_file() for d in eval_dirs.values()
               for name in ("run.json", "metrics.csv")):
        return errors, {}, {}
    recs = {slot: load_run(d) for slot, d in eval_dirs.items()}
    frames = {slot: pd.read_csv(d / "metrics.csv") for slot, d in eval_dirs.items()}
    errors += [f"{label(*slot)} eval: fold is {rec.get('fold')!r}, expected {fold!r}"
               for slot, rec in recs.items() if rec.get("fold") != fold]
    items = [(label(*slot), rec) for slot, rec in recs.items()]
    for key in SAME_EVAL_KEYS:
        errors += same_across(items, key, "eval")
    first = next(iter(frames))
    errors += [f"{label(*slot)} eval: case set differs from {label(*first)}"
               for slot, df in frames.items() if set(df["case"]) != set(frames[first]["case"])]
    errors += [f"{label(*slot)} eval: protocol_hash is not the pinned one"
               for slot, rec in recs.items() if rec.get("protocol_hash") != EXPECTED_PROTOCOL_HASH]
    errors += [f"{label(*slot)} eval: commit {rec.get('git_commit')} does not descend from "
               f"{PREREG_TAG}" for slot, rec in recs.items()
               if not descends_from_tag(rec.get("git_commit"), PREREG_TAG, REPO)]
    for slot, df in frames.items():
        n = sum(int(np.isinf(pd.to_numeric(df[c], errors="coerce")).sum()) for c in METRICS)
        if n:
            errors.append(f"{label(*slot)} eval: {n} infinite Dice or HD95 value(s)")
    versions = [rec.get("versions") for rec in recs.values()]
    if fold == "val":
        errors += [f"{label(*slot)} eval: versions differ from {label(*first)}"
                   for slot, rec in recs.items() if rec.get("versions") != versions[0]]
    elif any(v != versions[0] for v in versions):
        print("WARNING: versions differ between the evals")
    return errors, recs, frames


def averaged(frames):
    """{arm: metrics with one row per (case, region, pattern), the mean over the arm's 3 seeds}
    (AC-11). HD95 keeps the 373.13 mm empty rule, because it is already in metrics.csv."""
    return {arm: pd.concat([frames[arm, s] for s in SEEDS])
            .groupby(["case", "region", "pattern_id"], as_index=False)[list(METRICS)].mean()
            for arm in ARMS}


def wide_tables(means, metric, region):
    """{arm: one row per case (sorted, so all rows pair up), one column per pattern}."""
    return {arm: df[df["region"] == region].pivot(index="case", columns="pattern_id",
                                                  values=metric)
            for arm, df in means.items()}


def primary_score(wide, arm, level):
    """One number per subject: the mean over the level's patterns."""
    return wide[arm][SCORES[level]].mean(axis=1).to_numpy()


def paired(first, second, indices, ci=0.95):
    """The columns of one row that describe the pair of per subject arrays (first, second)."""
    d = first - second
    low, high = bootstrap_ci(d, indices, ci)
    return {"n": len(d), "mean_first": first.mean(), "mean_second": second.mean(),
            "mean_diff": d.mean(), "median_diff": float(pd.Series(d).median()),
            "ci_low": low, "ci_high": high, "ci_level": round(ci * 100),
            "n_nonzero": int((d != 0).sum())}


def h2_rows(means, metric, region, indices):
    """The 12 H2 rows of one (metric, region). Only the primary score is tested, with Holm over the
    3 comparisons (AC-13)."""
    wide = wide_tables(means, metric, region)
    tests = [paired_wilcoxon(primary_score(wide, a, PRIMARY) - primary_score(wide, b, PRIMARY))
             for _, a, b in H2_COMPARISONS]
    adjusted = holm([p for _, p in tests])
    rows = []
    for level in LEVELS:
        for i, (name, a, b) in enumerate(H2_COMPARISONS):
            row = dict.fromkeys(SUMMARY_COLUMNS) | {
                "family": "H2", "metric": metric, "region": region, "level": level,
                "comparison": name,
                **paired(primary_score(wide, a, level), primary_score(wide, b, level), indices)}
            if level == PRIMARY:
                row |= {"wilcoxon_stat": tests[i][0], "p_raw": tests[i][1], "p_holm": adjusted[i]}
            rows.append(row)
    return rows


def scored_row(family, wide, region, level, comparison, indices, ci=0.95, test=True):
    """One Dice row of summary.csv outside H2. p_holm stays blank, and test=False leaves the
    Wilcoxon columns blank too, for a row that reports no test."""
    name, a, b = comparison
    first, second = primary_score(wide, a, level), primary_score(wide, b, level)
    row = dict.fromkeys(SUMMARY_COLUMNS) | {
        "family": family, "metric": "dice", "region": region, "level": level, "comparison": name,
        **paired(first, second, indices, ci)}
    if test:
        row["wilcoxon_stat"], row["p_raw"] = paired_wilcoxon(first - second)
    return row


def h3_rows(means, region, indices):
    """The 2 H3 rows of one region: Dice at pattern 1, one 95% CI (AC-14)."""
    wide = wide_tables(means, "dice", region)
    return [scored_row("H3", wide, region, COMPLETE, c, indices) for c in H3_LINES.values()]


def h4_rows(means, region, indices):
    """The 2 H4 rows of one region: E3b-E3 at pattern 1, then on the primary score, whose 90% CI
    gives the lower bound of AC-15."""
    wide = wide_tables(means, "dice", region)
    return [scored_row("H4", wide, region, COMPLETE, EXPOSURE, indices),
            scored_row("H4", wide, region, PRIMARY, EXPOSURE, indices, ci=0.90)]


def descriptive_row(means, region, indices):
    """E3b-E1 at pattern 1: its 95% CI only, no test and no verdict (AC-15)."""
    return scored_row("descriptive", wide_tables(means, "dice", region), region, COMPLETE,
                      EXPOSURE_E1, indices, test=False)


def summary_rows(means, indices):
    """Every summary.csv row, in the order of Output schemas."""
    return ([r for m in METRICS for g in REGIONS for r in h2_rows(means, m, g, indices)]
            + [r for g in REGIONS for r in h3_rows(means, g, indices)]
            + [r for g in REGIONS for r in h4_rows(means, g, indices)]
            + [descriptive_row(means, g, indices) for g in REGIONS])


def pattern_rows(means, metric, region, indices):
    """The 60 patterns.csv rows of one (metric, region), on the averaged values, with Holm over the
    15 patterns of each comparison. Secondary, so pattern 1 differs from H3 by design."""
    wide = wide_tables(means, metric, region)
    tests = {name: {p: paired_wilcoxon(wide[a][p].to_numpy() - wide[b][p].to_numpy())
                    for p in PATTERNS} for name, a, b in PATTERN_COMPARISONS}
    adjusted = {name: dict(zip(by_pattern, holm([p for _, p in by_pattern.values()]), strict=True))
                for name, by_pattern in tests.items()}
    rows = []
    for p in PATTERNS:
        for name, a, b in PATTERN_COMPARISONS:
            first, second = wide[a][p].to_numpy(), wide[b][p].to_numpy()
            pair = paired(first, second, indices)
            del pair["ci_level"]
            rows.append(dict.fromkeys(PATTERN_COLUMNS) | pair | {
                "metric": metric, "region": region, "pattern_id": p,
                "pattern_name": PATTERNS[p].name, "comparison": name,
                "median_first": float(pd.Series(first).median()),
                "median_second": float(pd.Series(second).median()),
                "wilcoxon_stat": tests[name][p][0], "p_raw": tests[name][p][1],
                "p_holm": adjusted[name][p]})
    return rows


def per_seed_rows(frames):
    """The per_seed.csv rows: each seed on its own, with no p value (AC-12)."""
    rows = []
    for seed in SEEDS:
        one_seed = {arm: frames[arm, seed] for arm in ARMS}
        for metric in METRICS:
            for region in REGIONS:
                wide = wide_tables(one_seed, metric, region)
                for level in (PRIMARY, COMPLETE):
                    for name, a, b in SEED_COMPARISONS:
                        d = primary_score(wide, a, level) - primary_score(wide, b, level)
                        rows.append({"seed": seed, "metric": metric, "region": region,
                                     "level": level, "comparison": name,
                                     "mean_diff": float(d.mean()),
                                     "median_diff": float(pd.Series(d).median())})
    return rows


def volume_rows(frames):
    """The volume.csv rows: predicted minus true WT volume per arm, seed and pattern, computed for
    each seed separately, with limits of agreement. Descriptive, no test (AC-16)."""
    rows = []
    for arm, seed in slots():
        df = frames[arm, seed]
        wt = df[df["region"] == VOLUME_REGION].sort_values("case")
        for p, pattern in PATTERNS.items():
            d = wt[wt["pattern_id"] == p]
            diff = (d["pred_ml"] - d["true_ml"]).to_numpy()
            mean, sd = float(diff.mean()), float(diff.std(ddof=1))
            rows.append({"arm": ARM_NAMES[arm], "seed": seed, "pattern_id": p,
                         "pattern_name": pattern.name, "n": len(diff), "mean_diff_ml": mean,
                         "loa_low_ml": mean - 1.96 * sd, "loa_high_ml": mean + 1.96 * sd})
    return rows


def cost_label(ci_low, ci_high, margin=EQUIV_MARGIN):
    """The label of AC-14, on unrounded bounds, with strict comparisons, in the spec's order."""
    if -margin < ci_low and ci_high < margin:
        return "equivalent"
    if ci_high < 0:
        return "cost"
    if ci_low > 0:
        return "gain"
    return "inconclusive"


def seed_means(per_seed, region, name, level):
    """The unrounded Dice mean_diff of each seed, from the rows per_seed.csv is written from."""
    return [r["mean_diff"] for r in per_seed
            if (r["metric"], r["region"], r["level"], r["comparison"])
            == ("dice", region, level, name)]


def seeds_agree(per_seed, region, name, mean_diff):
    """How many seeds have a Dice mean_diff at pattern 1 of the same sign as the averaged one."""
    return sum(np.sign(v) == np.sign(mean_diff)
               for v in seed_means(per_seed, region, name, COMPLETE))


def seeds_within(per_seed, region, name):
    """How many seeds have a Dice mean_diff at pattern 1 strictly inside the margin (AC-14)."""
    return sum(-EQUIV_MARGIN < v < EQUIV_MARGIN
               for v in seed_means(per_seed, region, name, COMPLETE))


def seeds_above(per_seed, region, name, level, floor):
    """How many seeds have a Dice mean_diff strictly above floor at this level (AC-32)."""
    return sum(v > floor for v in seed_means(per_seed, region, name, level))


def seed_word(word, *counts):
    """AC-32: a claim the averaged test gave stands only if every count covers all the seeds. This
    adds a condition and removes none, so it runs only on a word that is a claim."""
    return "split" if any(c < len(SEEDS) for c in counts) else word


def h3_verdict(summary, per_seed):
    """The six H3 lines (AC-14, AC-18, AC-32). The p value is reported, and never changes the
    label."""
    rows = {(r["region"], r["comparison"]): r for r in summary if r["family"] == "H3"}
    lines = []
    for h, (name, _, _) in H3_LINES.items():
        for g in REGIONS:
            r = rows[g, name]
            low, high = r["ci_low"], r["ci_high"]
            agree = seeds_agree(per_seed, g, name, r["mean_diff"])
            within = seeds_within(per_seed, g, name)
            word = cost_label(low, high)
            if word == "equivalent":
                word = seed_word(word, within)
            elif word in ("cost", "gain"):
                word = seed_word(word, agree)
            lines.append(f"{h} {g}: {word} ({name}, "
                         f"mean_diff={r['mean_diff']:+.6f}, median_diff={r['median_diff']:+.6f}, "
                         f"ci=[{low:+.6f}, {high:+.6f}], "
                         f"min_margin={max(abs(low), abs(high)):.6f}, "
                         f"seeds_agree={agree}/{len(SEEDS)}, seeds_within={within}/{len(SEEDS)}, "
                         f"p_raw={r['p_raw']:.6g})")
    return lines


def h4_verdict(summary, alpha, per_seed):
    """The three H4 lines. Both parts must hold, so no Holm correction is used (AC-15), and then
    all the seeds must back both parts (AC-32)."""
    rows = {(r["region"], r["level"]): r for r in summary if r["family"] == "H4"}
    lines = []
    for g in REGIONS:
        c, primary = rows[g, COMPLETE], rows[g, PRIMARY]
        agree = seeds_above(per_seed, g, EXPOSURE[0], COMPLETE, 0)
        not_worse = seeds_above(per_seed, g, EXPOSURE[0], PRIMARY, -EQUIV_MARGIN)
        gain = c["mean_diff"] > 0 and c["median_diff"] > 0 and c["p_raw"] < alpha
        word = (seed_word("supported", agree, not_worse)
                if gain and primary["ci_low"] > -EQUIV_MARGIN else "not supported")
        lines.append(f"H4 {g}: {word} ({EXPOSURE[0]}, complete_mean_diff={c['mean_diff']:+.6f}, "
                     f"complete_median_diff={c['median_diff']:+.6f}, "
                     f"complete_p={c['p_raw']:.6g}, primary_ci90_low={primary['ci_low']:+.6f}, "
                     f"seeds_agree={agree}/{len(SEEDS)}, "
                     f"seeds_not_worse={not_worse}/{len(SEEDS)})")
    return lines


def h2_verdict(summary, alpha, per_seed):
    """The nine H2 lines, on unrounded numbers, with the rule of spec 0005 (AC-13), then the seed
    rule (AC-32)."""
    primary = {(r["region"], r["comparison"]): r for r in summary
               if r["family"] == "H2" and r["metric"] == "dice" and r["level"] == PRIMARY}
    lines = []
    for h, name in H2_LINES.items():
        for g in REGIONS:
            r = primary[g, name]
            agree = seeds_above(per_seed, g, name, PRIMARY, 0)
            word = (seed_word("supported", agree) if r["mean_diff"] > 0 and r["p_holm"] < alpha
                    else "not supported")
            lines.append(f"{h} {g}: {word} ({name}, mean_diff={r['mean_diff']:+.6f}, "
                         f"p_holm={r['p_holm']:.6g}, seeds_agree={agree}/{len(SEEDS)})")
    return lines


def non_finite(rows):
    """AC-11: every number that goes out must be finite. Returns error strings."""
    return [" ".join(str(x) for x in r.values() if isinstance(x, str)) + f": {k} is {v}"
            for r in rows for k, v in r.items()
            if isinstance(v, (float, np.floating)) and not np.isfinite(v)]


def cell(column, v):
    """A float is %.6f, a p value %.6g, a missing value an empty field (spec 0004, File formats)."""
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}" if column in P_COLUMNS else f"{v:.6f}"
    return v


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows({k: cell(k, v) for k, v in r.items()} for r in rows)


def folder_problems(folder, rerun_reason):
    """AC-17: only a finished folder may be replaced, and only with a reason."""
    if not folder.exists():
        return [f"--rerun-reason replaces a finished folder, and {folder} is not one"
                ] if rerun_reason else []
    if not (folder / "run.json").is_file() or load_run(folder).get("status") != "ok":
        return [f"{folder} exists but is not finished: look at it, then move it by hand"]
    return [] if rerun_reason else [f"{folder} is finished; give --rerun-reason to replace it"]


def opened_folders(evals):
    """Every *-test entry in the evals folder, finished or not, file or folder. Any one of them
    means the opening has happened (AC-34)."""
    return sorted(Path(evals).glob("*-test"))


def val_side_problems(evals):
    """AC-10 and AC-34, for --fold val only. A clean tree lets the record say git_dirty false, and
    a test eval on disk means the opening has happened, so the record must not move."""
    dirty = git_info()[1]
    errors = [] if dirty is False else [
        f"git_dirty is {dirty!r}, must be False: commit first, an untracked file counts too"]
    return errors + [f"{d} exists: the test was opened, so the val compare stays as it is"
                     for d in opened_folders(evals)]


def val_compare_problems(out, run_recs, eval_recs):
    """AC-28 and AC-35 against compare-val/run.json. eval_recs are the test evals, or in the
    preflight the val evals, which must still be the ones the val compare read."""
    folder = out / "compare-val"
    if not (folder / "run.json").is_file():
        return [f"{folder} missing: run compare_seeds.py --fold val first"]
    rec = load_run(folder)
    errors = val_record_problems(rec)
    if rec.get("run_ids") != {slot_key(s): r["run_id"] for s, r in run_recs.items()}:
        errors.append("val compare: names other runs")
    return errors + [f"{label(*slot)} eval: {k} differs from the val compare's"
                     for slot, ev in eval_recs.items() for k in ("protocol_hash", "splits_sha256")
                     if ev.get(k) != rec.get(k)]


def val_files_problems(a, run_recs):
    """AC-35 on --fold test: all 12 val metrics.csv are still the files the val compare read."""
    folder = a.out / "compare-val"
    if not (folder / "run.json").is_file():
        return []  # val_compare_problems has said so
    hashes = load_run(folder).get("eval_sha256", {})
    return [e for slot, r in run_recs.items() for e in val_file_problems(
        a.evals / f"{r['run_id']}-val" / "metrics.csv", hashes.get(slot_key(slot)))]


def preflight_record_problems(out, eval_recs):
    """AC-28: every test eval comes from the one commit that passed the preflight."""
    path = out / "compare-val" / PREFLIGHT
    if not path.is_file():
        return [f"{path} missing: the preflight did not pass"]
    commit = json.loads(path.read_text()).get("git_commit")
    return [f"{label(*slot)} eval: commit {ev.get('git_commit')} is not the preflight's, {commit}"
            for slot, ev in eval_recs.items() if ev.get("git_commit") != commit]


def preflight_problems(a, run_dirs, run_recs):
    """Every guard of the test path that needs no test metric (AC-20): the 12 val evals, the gate
    of evaluate.py for each model, which holds the seal (AC-33), and the val compare, whose commit
    must be reachable from here, or a squash merge would only show on Kaggle (AC-34). Reads no test
    case."""
    commit, dirty = git_info()
    val_dirs = {s: a.evals / f"{r['run_id']}-val" for s, r in run_recs.items()}
    errors, val_recs, _ = eval_problems(val_dirs, run_dirs, "val")
    protocol = protocol_hash()
    if protocol != EXPECTED_PROTOCOL_HASH:
        errors.append("protocol_hash() is not the pinned one")
    config = config_hash({"eval": {**load_config(EVAL_CONFIG)["eval"], "quick": False}})
    for slot, rec in run_recs.items():
        errors += [f"{label(*slot)} gate: {e}" for e in check_test_gate(
            "test", True, False, rec, (commit, dirty), val_dirs[slot], run_dirs[slot], protocol,
            config)]
    errors += val_compare_problems(a.out, run_recs, val_recs)
    if (a.out / "compare-val" / "run.json").is_file():
        val_commit = load_run(a.out / "compare-val").get("git_commit")
        if not is_ancestor(val_commit, commit, REPO):
            errors.append(f"val compare: commit {val_commit} is not an ancestor of {commit}: "
                          "merge with a merge commit, never a squash")
    return errors


def guards(a):
    """Every failed rule of AC-10, AC-17 and AC-28. Returns (errors, run records, eval folders,
    eval records, frames)."""
    if a.preflight and a.fold != "test":
        return ["--preflight is only for --fold test"], {}, {}, {}, {}
    errors, run_dirs = manifest_problems(a.runs, a.root)
    if a.preflight:  # before every early return, so a bad manifest still prints it (AC-34)
        errors += [f"{d} exists: the test was opened already, so {PREFLIGHT} stays as it is"
                   for d in opened_folders(a.evals)]
    if a.fold == "val":
        errors += val_side_problems(a.evals)
    if len(run_dirs) != len(slots()):
        return errors, {}, {}, {}, {}
    more, run_recs = run_problems(run_dirs)
    errors += more
    if not run_recs:
        return errors, {}, {}, {}, {}
    if a.preflight:
        return errors + preflight_problems(a, run_dirs, run_recs), run_recs, {}, {}, {}
    eval_dirs = {slot: a.evals / f"{rec.get('run_id')}-{a.fold}" for slot, rec in run_recs.items()}
    more, eval_recs, frames = eval_problems(eval_dirs, run_dirs, a.fold)
    errors += more
    if a.fold == "test":
        errors += val_compare_problems(a.out, run_recs, eval_recs)
        errors += val_files_problems(a, run_recs)
        errors += preflight_record_problems(a.out, eval_recs)
        for commit in sorted({str(rec.get("git_commit")) for rec in eval_recs.values()}):
            errors += seal_problems(commit)
    errors += folder_problems(a.out / f"compare-{a.fold}", a.rerun_reason)
    return errors, run_recs, eval_dirs, eval_recs, frames


def slot_key(slot):
    return f"{slot[0]}_seed{slot[1]}"


def run_record(a, run_recs, eval_dirs, eval_recs, cfg, superseded):
    """run.json of a finished compare, in the order of Output schemas, plus the two rerun fields
    of AC-17, so a replaced folder keeps its reason."""
    versions = [rec.get("versions") for rec in eval_recs.values()]
    first = eval_recs[ARMS[0], SEEDS[0]]
    commit, dirty = git_info()
    record = {
        "compare_id": f"compare-{a.fold}", "status": "ok", "fold": a.fold,
        "n_cases": first["n_cases"],
        "run_ids": {slot_key(s): rec["run_id"] for s, rec in run_recs.items()},
        "eval_commits": {slot_key(s): rec["git_commit"] for s, rec in eval_recs.items()},
        "eval_sha256": {slot_key(s): sha256_file(d / "metrics.csv") for s, d in eval_dirs.items()},
        "versions_equal": all(v == versions[0] for v in versions),
        "n_resumes": {slot_key(s): len(rec["resumes"]) for s, rec in run_recs.items()},
        "seeds": list(SEEDS), "alpha": cfg["alpha"], "equiv_margin": EQUIV_MARGIN,
        "compare_hash": compare_hash(), "protocol_hash": first["protocol_hash"],
        "splits_sha256": first["splits_sha256"], "prereg_tag": PREREG_TAG, "git_commit": commit,
        "git_dirty": dirty, "rerun_reason": a.rerun_reason, "superseded": superseded}
    assert tuple(record) == RUN_JSON_KEYS
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--fold", required=True, choices=("val", "test"))
    ap.add_argument("--evals", type=Path, default=REPO / "results/0007-confirmation/evals")
    ap.add_argument("--out", type=Path, default=REPO / "results/0007-confirmation")
    ap.add_argument("--root", type=Path, default=REPO, help="the folder that holds results/")
    ap.add_argument("--rerun-reason")
    ap.add_argument("--preflight", action="store_true", help="the checks before the test opening")
    a = ap.parse_args(argv)
    # first, so a refusal leaves no record, unless the opening has made it the only proof (AC-34)
    if a.preflight and a.fold == "test" and not opened_folders(a.evals):
        (a.out / "compare-val" / PREFLIGHT).unlink(missing_ok=True)
    errors, run_recs, eval_dirs, eval_recs, frames = guards(a)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 2
    if a.preflight:
        record = {"status": "ok", "git_commit": git_info()[0],
                  "date": datetime.now(UTC).isoformat(timespec="seconds")}
        (a.out / "compare-val" / PREFLIGHT).write_text(json.dumps(record, indent=2))
        print(f"preflight PASS: nothing read from the test fold, {PREFLIGHT} written")
        return 0

    first = ARMS[0], SEEDS[0]
    cfg = load_config(eval_dirs[first] / "config.toml")["eval"]
    n_cases = eval_recs[first]["n_cases"]
    # one index matrix per fold, with subjects sorted by case (AC-11)
    indices = bootstrap_indices(n_cases, cfg["bootstrap_resamples"], cfg["seed"])
    means = averaged(frames)
    summary = summary_rows(means, indices)
    keys = [(m, g) for m in METRICS for g in REGIONS]
    patterns = [r for m, g in keys for r in pattern_rows(means, m, g, indices)]
    per_seed = per_seed_rows(frames)
    volume = volume_rows(frames)
    bad = non_finite([*summary, *patterns, *per_seed, *volume])
    if bad:
        for e in bad:
            print(f"ERROR: {e}")
        return 2
    lines = [*h2_verdict(summary, cfg["alpha"], per_seed), *h3_verdict(summary, per_seed),
             *h4_verdict(summary, cfg["alpha"], per_seed)]

    # built beside the final folder, so a crash leaves no half folder and keeps a finished one
    final = a.out / f"compare-{a.fold}"
    partial = a.out / f".compare-{a.fold}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    write_csv(partial / "summary.csv", SUMMARY_COLUMNS, summary)
    write_csv(partial / "patterns.csv", PATTERN_COLUMNS, patterns)
    write_csv(partial / "per_seed.csv", PER_SEED_COLUMNS, per_seed)
    write_csv(partial / "volume.csv", VOLUME_COLUMNS, volume)
    (partial / "verdict.txt").write_text("\n".join(lines) + "\n")
    superseded = None
    if final.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        superseded = f"{final.name}.superseded-{stamp}"
        final.rename(final.with_name(superseded))
    record = run_record(a, run_recs, eval_dirs, eval_recs, cfg, superseded)
    (partial / "run.json").write_text(json.dumps(record, indent=2))
    partial.rename(final)
    print("\n".join(lines))
    print(f"{n_cases} subjects, {len(SEEDS)} seeds -> {len(summary)} rows in {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
