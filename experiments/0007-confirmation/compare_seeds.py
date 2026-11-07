"""Twelve eval folders -> the three seed comparison of E1, E3, E4 and E3b (spec 0006).

    uv run python experiments/0007-confirmation/compare_seeds.py --runs \
        experiments/0007-confirmation/runs.toml --fold val
Exit 2, printing every failed rule and writing nothing, unless the manifest, the 12 runs and the 12
evals pass AC-10. For each subject, region and pattern, an arm's value is the mean over its 3 seeds,
and every test runs on those means, so n is the number of subjects (AC-11). Writes
<out>/compare-<fold>/ with summary.csv and verdict.txt. All numbers come from metrics.csv of the
evals, so it uses no GPU.

Built so far: build steps 1 and 2, the H2 lines. The test path is build step 6, so --fold test is
refused until then. H3, H4, per_seed.csv, volume.csv and run.json come in later build steps.

This repeats some of 0005-strategies/compare.py on purpose: experiments never import each other,
and that file stays unchanged (AC-21). A test proves both give the same H2 lines (AC-24).
"""
import argparse
import csv
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import REGIONS
from common.evaluation import PREREG_TAG, check_rules
from common.missing import PATTERNS
from common.runlog import load_config, load_run
from common.stats import bootstrap_ci, bootstrap_indices, holm, paired_wilcoxon
from common.strategies import strategy_name

REPO = Path(__file__).resolve().parents[2]
# protocol_hash() at the tag commit. Repinned there, beside SEALED_HASH; a test keeps them equal.
EXPECTED_PROTOCOL_HASH = "0bc76bfaba23b0d7b35775038b10a28b21650e5da374de4351374dfaa617f642"
ARMS = ("e1", "e3", "e4", "e3b")
SEEDS = (42, 43, 44)
STRATEGIES = {"e1": None, "e3": "dropout", "e4": "presence", "e3b": "exposure"}
COMPLETE_SHARE = 0.5
METRICS = ("dice", "hd95_mm")
PRIMARY = "incomplete"
LEVELS = {PRIMARY: [p.id for p in PATTERNS.values() if p.missing],
          **{f"missing_{n}": [p.id for p in PATTERNS.values() if len(p.missing) == n]
             for n in (1, 2, 3)}}
# (name, first, second): the paired difference is d = value(first) - value(second)
H2_COMPARISONS = (("E3-E1", "e3", "e1"), ("E4-E1", "e4", "e1"), ("E4-E3", "e4", "e3"))
H2_LINES = {"H2a": "E3-E1", "H2b": "E4-E1", "H2c": "E4-E3"}
SAME_RUN_KEYS = ("train_cases", "val_cases", "gpu", "cache_hash")
SAME_EVAL_KEYS = ("fold", "n_cases", "protocol_hash", "config_hash", "splits_sha256", "gpu")
SUMMARY_COLUMNS = ["family", "metric", "region", "level", "comparison", "n", "mean_first",
                   "mean_second", "mean_diff", "median_diff", "ci_low", "ci_high", "ci_level",
                   "n_nonzero", "wilcoxon_stat", "p_raw", "p_holm"]
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


def descends_from_tag(commit):
    """True if commit holds the protocol tag. Unknown tag or unknown commit gives False."""
    r = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                        f"refs/tags/{PREREG_TAG}", str(commit)],
                       capture_output=True, check=False)
    return r.returncode == 0


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
               if not descends_from_tag(rec.get("git_commit"))]
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
    return wide[arm][LEVELS[level]].mean(axis=1).to_numpy()


def paired(first, second, indices, level=0.95):
    """The columns of one row that describe the pair of per subject arrays (first, second)."""
    d = first - second
    low, high = bootstrap_ci(d, indices, level)
    return {"n": len(d), "mean_first": first.mean(), "mean_second": second.mean(),
            "mean_diff": d.mean(), "median_diff": float(pd.Series(d).median()),
            "ci_low": low, "ci_high": high, "ci_level": round(level * 100),
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


def h2_verdict(summary, alpha):
    """The nine H2 lines, on unrounded numbers, with the rule of spec 0005 (AC-13)."""
    primary = {(r["region"], r["comparison"]): r for r in summary
               if r["family"] == "H2" and r["metric"] == "dice" and r["level"] == PRIMARY}
    lines = []
    for h, name in H2_LINES.items():
        for g in REGIONS:
            r = primary[g, name]
            word = "supported" if r["mean_diff"] > 0 and r["p_holm"] < alpha else "not supported"
            lines.append(f"{h} {g}: {word} ({name}, mean_diff={r['mean_diff']:+.6f}, "
                         f"p_holm={r['p_holm']:.6g})")
    return lines


def non_finite(rows):
    """AC-11: every number that goes out must be finite. Returns error strings."""
    return [f"{r['family']} {r['metric']} {r['region']} {r['level']} {r['comparison']}: {k} is {v}"
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


def guards(a):
    """Every failed rule of AC-10. Returns (errors, eval folders, eval records, frames)."""
    if a.fold == "test":
        return ["--fold test is not built yet: it comes with spec 0006, build step 6"], {}, {}, {}
    errors, run_dirs = manifest_problems(a.runs, a.root)
    if len(run_dirs) != len(slots()):
        return errors, {}, {}, {}
    more, run_recs = run_problems(run_dirs)
    errors += more
    if not run_recs:
        return errors, {}, {}, {}
    eval_dirs = {slot: a.evals / f"{rec.get('run_id')}-{a.fold}" for slot, rec in run_recs.items()}
    more, eval_recs, frames = eval_problems(eval_dirs, run_dirs, a.fold)
    errors += more
    final = a.out / f"compare-{a.fold}"
    if final.exists():
        errors.append(f"{final} exists: look at it, then move it by hand")
    return errors, eval_dirs, eval_recs, frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--fold", required=True, choices=("val", "test"))
    ap.add_argument("--evals", type=Path, default=REPO / "results/0007-confirmation/evals")
    ap.add_argument("--out", type=Path, default=REPO / "results/0007-confirmation")
    ap.add_argument("--root", type=Path, default=REPO, help="the folder that holds results/")
    a = ap.parse_args(argv)
    errors, eval_dirs, eval_recs, frames = guards(a)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 2

    first = ARMS[0], SEEDS[0]
    cfg = load_config(eval_dirs[first] / "config.toml")["eval"]
    n_cases = eval_recs[first]["n_cases"]
    # one index matrix per fold, with subjects sorted by case (AC-11)
    indices = bootstrap_indices(n_cases, cfg["bootstrap_resamples"], cfg["seed"])
    means = averaged(frames)
    summary = [r for m in METRICS for g in REGIONS for r in h2_rows(means, m, g, indices)]
    bad = non_finite(summary)
    if bad:
        for e in bad:
            print(f"ERROR: {e}")
        return 2
    lines = h2_verdict(summary, cfg["alpha"])

    # built beside the final folder, so a crash leaves no half folder
    final = a.out / f"compare-{a.fold}"
    partial = a.out / f".compare-{a.fold}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    write_csv(partial / "summary.csv", SUMMARY_COLUMNS, summary)
    (partial / "verdict.txt").write_text("\n".join(lines) + "\n")
    partial.rename(final)
    print("\n".join(lines))
    print(f"{n_cases} subjects, {len(SEEDS)} seeds -> {len(summary)} rows in {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
