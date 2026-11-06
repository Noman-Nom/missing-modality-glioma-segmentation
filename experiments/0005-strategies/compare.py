"""Three eval folders -> the paired comparison of E1, E3 and E4 and the H2 verdict (spec 0005).

    uv run python experiments/0005-strategies/compare.py --e1-run <dir> --e3-run <dir> \
        --e4-run <dir> --fold val
Writes <out>/compare-<fold>/ with summary.csv (72 rows), patterns.csv (270 rows), verdict.txt and
run.json, run.json last. Exit 2, printing every failed rule and writing nothing, unless the three
runs and three evals pass AC-15, and on test AC-16. A finished folder needs --rerun-reason. All
numbers come from metrics.csv of the evals, so it uses no GPU and needs no best.pt. The rules are
the Statistics protocol of the spec, fixed before any test number.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import REGIONS
from common.evaluation import check_rules
from common.missing import PATTERNS
from common.runlog import git_info, load_config, load_run
from common.stats import bootstrap_ci, bootstrap_indices, holm, paired_wilcoxon
from common.strategies import strategy_name

REPO = Path(__file__).resolve().parents[2]
SUMMARY_COLUMNS = ["metric", "region", "level", "comparison", "n", "mean_first", "mean_second",
                   "mean_diff", "median_diff", "ci_low", "ci_high", "n_nonzero", "wilcoxon_stat",
                   "p_raw", "p_holm"]
PATTERN_COLUMNS = ["metric", "region", "pattern_id", "pattern_name", "comparison", "n",
                   "mean_first", "mean_second", "median_first", "median_second", "mean_diff",
                   "median_diff", "ci_low", "ci_high", "n_nonzero", "wilcoxon_stat", "p_raw",
                   "p_holm"]
RUN_JSON_KEYS = ("compare_id", "status", "fold", "n_cases", "e1_run_id", "e3_run_id", "e4_run_id",
                 "e1_eval_config_hash", "protocol_hash", "splits_sha256", "compare_hash",
                 "versions_equal", "n_resumes", "seed", "alpha", "git_commit", "git_dirty",
                 "rerun_reason", "superseded", "wall_time_s")
P_COLUMNS = {"p_raw", "p_holm"}
METRICS = ("dice", "hd95_mm")
MODELS = ("e1", "e3", "e4")
STRATEGIES = {"e1": None, "e3": "dropout", "e4": "presence"}
# (name, first, second): the paired difference is d = value(first) - value(second)
COMPARISONS = (("E3-E1", "e3", "e1"), ("E4-E1", "e4", "e1"), ("E4-E3", "e4", "e3"))
HYPOTHESES = {"H2a": "E3-E1", "H2b": "E4-E1", "H2c": "E4-E3"}
PRIMARY = "incomplete"
LEVELS = {PRIMARY: [p.id for p in PATTERNS.values() if p.missing],
          **{f"missing_{n}": [p.id for p in PATTERNS.values() if len(p.missing) == n]
             for n in (1, 2, 3)}}
SAME_RUN_KEYS = ("train_cases", "val_cases", "seed", "gpu", "cache_hash")
SAME_EVAL_KEYS = ("fold", "n_cases", "protocol_hash", "config_hash", "splits_sha256", "gpu")
# a change to the comparison, or to the path a strategy takes through scoring (AC-16); order matters
COMPARE_HASH_FILES = ("experiments/0005-strategies/compare.py", "experiments/common/strategies.py",
                      "experiments/common/model.py", "experiments/0004-missing-eval/evaluate.py")


def compare_hash():
    """sha256 over the bytes of COMPARE_HASH_FILES, in that order (AC-16)."""
    h = hashlib.sha256()
    for name in COMPARE_HASH_FILES:
        h.update((REPO / name).read_bytes())
    return h.hexdigest()


def run_problems(run_dirs):
    """Every failed AC-15 rule on the three run folders. Returns (errors, records), records empty
    unless all three could be read."""
    missing = [f"{m} run: {d / name} missing" for m, d in run_dirs.items()
               for name in ("run.json", "config.toml") if not (d / name).is_file()]
    if missing:
        return missing, {}
    recs = {m: load_run(d) for m, d in run_dirs.items()}
    cfgs = {m: load_config(d / "config.toml") for m, d in run_dirs.items()}
    errors = []
    for m, rec in recs.items():
        if rec.get("status") != "ok":
            errors.append(f"{m} run: status is {rec.get('status')!r}, not 'ok'")
        if rec.get("quick") is not False:
            errors.append(f"{m} run: a quick run")
        if strategy_name(cfgs[m]) != STRATEGIES[m]:
            errors.append(f"{m} run: strategy is {strategy_name(cfgs[m])!r}, "
                          f"expected {STRATEGIES[m]!r}")
    plain = {m: {k: v for k, v in cfg.items() if k != "strategy"} for m, cfg in cfgs.items()}
    errors += [f"{m} run: config differs from E1 outside [strategy]" for m in MODELS[1:]
               if plain[m] != plain["e1"]]
    errors += [f"{m} run: {k} differs from E1" for m in MODELS[1:] for k in SAME_RUN_KEYS
               if recs[m].get(k) != recs["e1"].get(k)]
    return errors, recs


def eval_problems(eval_dirs, run_dirs, fold):
    """Every failed AC-15 rule on the three eval folders. Returns (errors, records, frames), the
    last two empty unless all three evals could be read."""
    errors = []
    for m, d in eval_dirs.items():
        try:
            errors += [f"{m} eval: {e}" for e in check_rules(d, run_dirs[m])]
        except OSError as e:  # on test check_rules needs the committed splits.csv on this machine
            errors.append(f"{m} eval: cannot read {e.filename}")
    paths = {m: (d / "run.json", d / "metrics.csv") for m, d in eval_dirs.items()}
    if not all(p.is_file() for pair in paths.values() for p in pair):
        return errors, {}, {}
    recs = {m: load_run(d) for m, d in eval_dirs.items()}
    frames = {m: pd.read_csv(d / "metrics.csv") for m, d in eval_dirs.items()}
    errors += [f"{m} eval: fold is {rec.get('fold')!r}, expected {fold!r}"
               for m, rec in recs.items() if rec.get("fold") != fold]
    errors += [f"{m} eval: {k} differs from E1" for m in MODELS[1:] for k in SAME_EVAL_KEYS
               if recs[m].get(k) != recs["e1"].get(k)]
    errors += [f"{m} eval: case set differs from E1" for m in MODELS[1:]
               if set(frames[m]["case"]) != set(frames["e1"]["case"])]
    return errors, recs, frames


def val_compare_problems(out, run_ids, hashed):
    """AC-16: a test comparison needs the finished val comparison of the same runs and code."""
    folder = out / "compare-val"
    if not (folder / "run.json").is_file():
        return [f"{folder / 'run.json'} missing: run compare.py --fold val first"]
    rec = load_run(folder)
    errors = []
    if rec.get("status") != "ok":
        errors.append(f"val compare: status is {rec.get('status')!r}, not 'ok'")
    errors += [f"val compare: {m}_run_id differs" for m in MODELS
               if rec.get(f"{m}_run_id") != run_ids[m]]
    if rec.get("compare_hash") != hashed:
        errors.append("compare_hash differs from the val compare's: "
                      "compare val again with --rerun-reason")
    return errors


def folder_problems(folder, rerun_reason):
    """AC-17: only a finished folder may be replaced, and only with a reason."""
    if not folder.exists():
        return [f"--rerun-reason replaces a finished folder, and {folder} is not one"
                ] if rerun_reason else []
    if not (folder / "run.json").is_file() or load_run(folder).get("status") != "ok":
        return [f"{folder} exists but is not finished: look at it, then move it by hand"]
    return [] if rerun_reason else [f"{folder} is finished; give --rerun-reason to replace it"]


def versions_match(recs):
    """True if every eval record given holds the same versions. A difference never refuses,
    because a test opening cannot be taken back (AC-16)."""
    versions = [rec.get("versions") for rec in recs]
    return all(v == versions[0] for v in versions)


def wide_tables(frames, metric, region):
    """{model: one row per case (sorted, so all rows pair up), one column per pattern}."""
    return {m: df[df["region"] == region].pivot(index="case", columns="pattern_id", values=metric)
            for m, df in frames.items()}


def paired(first, second, indices):
    """The columns of one row that describe the pair of per subject arrays (first, second)."""
    d = first - second
    low, high = bootstrap_ci(d, indices)
    return {"n": len(d), "mean_first": first.mean(), "mean_second": second.mean(),
            "mean_diff": d.mean(), "median_diff": float(pd.Series(d).median()),
            "ci_low": low, "ci_high": high, "n_nonzero": int((d != 0).sum())}


def primary_score(wide, model, level):
    """One number per subject: the mean over the level's patterns."""
    return wide[model][LEVELS[level]].mean(axis=1).to_numpy()


def summary_rows(frames, metric, region, indices):
    """The 12 summary.csv rows of one (metric, region). Only the primary score is tested, with
    Holm over the 3 comparisons."""
    wide = wide_tables(frames, metric, region)
    tests = [paired_wilcoxon(primary_score(wide, a, PRIMARY) - primary_score(wide, b, PRIMARY))
             for _, a, b in COMPARISONS]
    adjusted = holm([p for _, p in tests])
    rows = []
    for level in LEVELS:
        for i, (name, a, b) in enumerate(COMPARISONS):
            row = dict.fromkeys(SUMMARY_COLUMNS) | {
                "metric": metric, "region": region, "level": level, "comparison": name,
                **paired(primary_score(wide, a, level), primary_score(wide, b, level), indices)}
            if level == PRIMARY:
                row |= {"wilcoxon_stat": tests[i][0], "p_raw": tests[i][1], "p_holm": adjusted[i]}
            rows.append(row)
    return rows


def pattern_rows(frames, metric, region, indices):
    """The 45 patterns.csv rows of one (metric, region), with Holm over the 15 patterns of each
    comparison."""
    wide = wide_tables(frames, metric, region)
    diffs = {name: {p: wide[a][p].to_numpy() - wide[b][p].to_numpy() for p in PATTERNS}
             for name, a, b in COMPARISONS}
    tests = {name: {p: paired_wilcoxon(d) for p, d in by_pattern.items()}
             for name, by_pattern in diffs.items()}
    adjusted = {name: dict(zip(by_pattern, holm([p for _, p in by_pattern.values()]), strict=True))
                for name, by_pattern in tests.items()}
    rows = []
    for p in PATTERNS:
        for name, a, b in COMPARISONS:
            first, second = wide[a][p].to_numpy(), wide[b][p].to_numpy()
            rows.append(dict.fromkeys(PATTERN_COLUMNS) | {
                "metric": metric, "region": region, "pattern_id": p,
                "pattern_name": PATTERNS[p].name, "comparison": name,
                **paired(first, second, indices),
                "median_first": float(pd.Series(first).median()),
                "median_second": float(pd.Series(second).median()),
                "wilcoxon_stat": tests[name][p][0], "p_raw": tests[name][p][1],
                "p_holm": adjusted[name][p]})
    return rows


def verdict(summary, alpha):
    """The nine H2 lines, on unrounded numbers, with no human judgement (AC-18)."""
    primary = {(r["region"], r["comparison"]): r for r in summary
               if r["metric"] == "dice" and r["level"] == PRIMARY}
    lines = []
    for h, name in HYPOTHESES.items():
        for g in REGIONS:
            r = primary[g, name]
            word = "supported" if r["mean_diff"] > 0 and r["p_holm"] < alpha else "not supported"
            lines.append(f"{h} {g}: {word} ({name}, mean_diff={r['mean_diff']:+.6f}, "
                         f"p_holm={r['p_holm']:.6g})")
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


def guards(a, run_dirs, out):
    """Every failed rule of AC-15, AC-16 and AC-17. Returns (errors, run records, eval records,
    val eval records, frames)."""
    errors, run_recs = run_problems(run_dirs)
    if not run_recs:
        return errors, {}, {}, {}, {}
    run_ids = {m: rec.get("run_id") for m, rec in run_recs.items()}
    parents = {"e1": a.e1_evals, "e3": a.evals, "e4": a.evals}
    eval_dirs = {m: parents[m] / f"{run_ids[m]}-{a.fold}" for m in MODELS}
    more, eval_recs, frames = eval_problems(eval_dirs, run_dirs, a.fold)
    errors += more
    val_recs = {}
    if a.fold == "test":
        errors += val_compare_problems(out, run_ids, compare_hash())
        val_dirs = {m: parents[m] / f"{run_ids[m]}-val" for m in MODELS}
        val_recs = {m: load_run(d) for m, d in val_dirs.items() if (d / "run.json").is_file()}
    errors += folder_problems(out / f"compare-{a.fold}", a.rerun_reason)
    return errors, run_recs, eval_recs, val_recs, frames


def main(argv=None):
    start = time.time()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e1-run", required=True, type=Path)
    ap.add_argument("--e3-run", required=True, type=Path)
    ap.add_argument("--e4-run", required=True, type=Path)
    ap.add_argument("--fold", required=True, choices=("val", "test"))
    ap.add_argument("--e1-evals", type=Path, default=REPO / "results" / "0004-missing-eval")
    ap.add_argument("--evals", type=Path, default=REPO / "results" / "0005-strategies" / "evals")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "0005-strategies")
    ap.add_argument("--rerun-reason")
    a = ap.parse_args(argv)
    run_dirs = {"e1": a.e1_run, "e3": a.e3_run, "e4": a.e4_run}
    errors, run_recs, eval_recs, val_recs, frames = guards(a, run_dirs, a.out)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 2
    versions_equal = versions_match([*eval_recs.values(), *val_recs.values()])
    if not versions_equal:
        print("WARNING: versions differ between the evals, recorded as versions_equal false")

    e1_folder = a.e1_evals / f"{run_recs['e1']['run_id']}-{a.fold}"
    cfg = load_config(e1_folder / "config.toml")["eval"]
    n_cases = eval_recs["e1"]["n_cases"]
    indices = bootstrap_indices(n_cases, cfg["bootstrap_resamples"], cfg["seed"])
    keys = [(m, g) for m in METRICS for g in REGIONS]
    summary = [r for m, g in keys for r in summary_rows(frames, m, g, indices)]
    patterns = [r for m, g in keys for r in pattern_rows(frames, m, g, indices)]
    lines = verdict(summary, cfg["alpha"])

    # built beside the final folder, so a crash leaves no half folder and keeps a finished one
    final = a.out / f"compare-{a.fold}"
    partial = a.out / f".compare-{a.fold}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    write_csv(partial / "summary.csv", SUMMARY_COLUMNS, summary)
    write_csv(partial / "patterns.csv", PATTERN_COLUMNS, patterns)
    (partial / "verdict.txt").write_text("\n".join(lines) + "\n")
    superseded = None
    if final.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        superseded = f"{final.name}.superseded-{stamp}"
        final.rename(final.with_name(superseded))
    commit, dirty = git_info()
    e1_eval = eval_recs["e1"]
    record = {
        "compare_id": final.name, "status": "ok", "fold": a.fold, "n_cases": n_cases,
        **{f"{m}_run_id": run_recs[m]["run_id"] for m in MODELS},
        "e1_eval_config_hash": e1_eval["config_hash"], "protocol_hash": e1_eval["protocol_hash"],
        "splits_sha256": e1_eval["splits_sha256"], "compare_hash": compare_hash(),
        "versions_equal": versions_equal,
        "n_resumes": {m: len(run_recs[m]["resumes"]) for m in MODELS},
        "seed": cfg["seed"], "alpha": cfg["alpha"], "git_commit": commit, "git_dirty": dirty,
        "rerun_reason": a.rerun_reason, "superseded": superseded,
        "wall_time_s": time.time() - start}
    assert tuple(record) == RUN_JSON_KEYS
    (partial / "run.json").write_text(json.dumps(record, indent=2))
    partial.rename(final)
    print("\n".join(lines))
    print(f"{n_cases} cases -> {len(summary)} and {len(patterns)} rows in {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
