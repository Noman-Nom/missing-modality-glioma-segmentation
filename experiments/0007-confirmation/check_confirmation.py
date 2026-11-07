"""The check of the confirmation study (spec 0006, AC-23).

    uv run python experiments/0007-confirmation/check_confirmation.py --runs \
        experiments/0007-confirmation/runs.toml
Exit 1, printing every failure, unless the 12 slots hold the right runs: 4 arms at 3 seeds with the
guards of compare_seeds.py (AC-1), E3b as e3.toml with only [strategy] changed and no extra input
channel (AC-2), the 3 kept seed 42 runs, and 9 new runs from one clean commit that descends from the
tag (AC-5), each run in its own place (AC-9), the rescore proof of seed 42 (AC-26), all 12 val evals
from one clean commit (AC-7), and uv.lock unchanged since E1 (AC-23). Reads run files, evals and
git, so it uses no GPU and no test case. Each proof of AC-26 prints its own line, so the rescoring
can be read before the 9 trainings, while the empty slots still fail.
"""
import argparse
import csv
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))  # the same experiment, never another one
import compare_seeds as cs
from common.data import REGIONS
from common.evaluation import PREREG_TAG, descends_from_tag
from common.missing import PATTERNS
from common.runlog import load_config, load_run

REPO = Path(__file__).resolve().parents[2]
E3_CONFIG = REPO / "experiments/0005-strategies/e3.toml"
E3B_CONFIG = REPO / "experiments/0007-confirmation/e3b.toml"
LOCK_COMMIT = "5ce3d3b"  # the commit of E1 seed 42 (AC-23)
SEED_42 = 42
NEW_RUNS = "results/0007-confirmation/runs"  # AC-9
# The 3 kept models (AC-5): arm -> (run folder, its val eval from before the volume columns)
KEPT = {
    "e1": ("results/0003-baseline/20261103-023817-f9b3afb6",
           "results/0004-missing-eval/20261103-023817-f9b3afb6-val"),
    "e3": ("results/0005-strategies/runs/20261106-024450-10cfa2cc",
           "results/0005-strategies/evals/20261106-024450-10cfa2cc-val"),
    "e4": ("results/0005-strategies/runs/20261106-235247-8999a39e",
           "results/0005-strategies/evals/20261106-235247-8999a39e-val"),
}
MAX_REPEATS = 2  # AC-6, per slot
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


def is_kept(slot):
    return slot[1] == SEED_42 and slot[0] in KEPT


def location_problems(dirs, root):
    """AC-5 and AC-9: each kept run stays in its own folder, each new run sits in NEW_RUNS."""
    errors = []
    for slot, folder in dirs.items():
        if is_kept(slot) and folder != Path(root) / KEPT[slot[0]][0]:
            errors.append(f"{cs.label(*slot)}: must stay at {KEPT[slot[0]][0]}")
        elif not is_kept(slot) and folder.parent != Path(root) / NEW_RUNS:
            errors.append(f"{cs.label(*slot)}: a new run must sit in {NEW_RUNS}")
    return errors


def replaced_problems(manifest, dirs):
    """AC-6: each entry of [replaced] is `<run id> = "<arm> seed<N>: <reason>"`, its run is out of
    every slot, and a slot has at most MAX_REPEATS."""
    with open(manifest, "rb") as f:
        replaced = tomllib.load(f).get("replaced", {})
    names = {cs.label(*slot) for slot in cs.slots()}
    in_use = {folder.name for folder in dirs.values()}
    errors = [f"[replaced] {run}: it is still in a slot" for run in replaced if run in in_use]
    slots = Counter()
    for run, entry in replaced.items():
        name, _, reason = str(entry).partition(":")
        if name not in names or not reason.strip():
            errors.append(f'[replaced] {run}: write it as "<arm> seed<N>: <reason>"')
        slots[name] += 1
    return errors + [f"{name}: {n} repeats, the limit is {MAX_REPEATS}"
                     for name, n in slots.items() if n > MAX_REPEATS]


def e3b_config_problems(e3b=E3B_CONFIG, e3=E3_CONFIG):
    """AC-2 on the files: e3b.toml is e3.toml with only [strategy] changed."""
    new, old = load_config(e3b), load_config(e3)
    errors = [] if cs.without(new, "strategy") == cs.without(old, "strategy") else [
        f"{e3b} differs from {e3} outside [strategy]"]
    if new.get("strategy") != {"name": "exposure", "complete_share": cs.COMPLETE_SHARE}:
        errors.append(f"{e3b}: [strategy] is not exposure with share {cs.COMPLETE_SHARE}")
    return errors


def channel_problems(recs):
    """AC-2 on the runs: the same number of weights as E1 means 4 input channels, as a presence
    channel would add weights to the first layer."""
    return [f"{cs.label('e3b', s)}: n_params differs from E1, so it is not 4 input channels"
            for s in cs.SEEDS if recs["e3b", s].get("n_params") != recs["e1", s].get("n_params")]


def one_clean_commit(recs, what):
    """Every failure of `all these records come from one commit with a clean tree`, and the
    sorted commits."""
    commits = sorted({str(rec.get("git_commit")) for rec in recs.values()})
    errors = [f"{cs.label(*slot)}: git_dirty is {rec.get('git_dirty')!r}, must be False"
              for slot, rec in recs.items() if rec.get("git_dirty") is not False]
    if len(commits) != 1:
        errors.append(f"{what} come from {len(commits)} commits, not one: {commits}")
    return errors, commits


def commit_problems(recs):
    """AC-5 and AC-23: the 9 new runs come from one clean commit that descends from the tag, with
    uv.lock unchanged since E1. An unknown commit fails too, because git then exits 128."""
    errors, commits = one_clean_commit({s: r for s, r in recs.items() if not is_kept(s)},
                                       f"the {len(cs.slots()) - len(KEPT)} new runs")
    for commit in commits:
        if not descends_from_tag(commit, PREREG_TAG, REPO):
            errors.append(f"commit {commit} does not descend from {PREREG_TAG}")
        lock = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", LOCK_COMMIT, commit, "--",
                               "uv.lock"], capture_output=True, check=False)
        if lock.returncode:
            errors.append(f"uv.lock differs between {LOCK_COMMIT} and {commit}, or git cannot tell")
    return errors


def eval_commit_problems(evals, recs):
    """AC-7: all 12 val evals are scored from one commit."""
    folders = {slot: evals / f"{rec.get('run_id')}-val" for slot, rec in recs.items()}
    errors = [f"{cs.label(*slot)}: {d / 'run.json'} missing" for slot, d in folders.items()
              if not (d / "run.json").is_file()]
    if errors:
        return errors
    return one_clean_commit({s: load_run(d) for s, d in folders.items()}, "the 12 val evals")[0]


def rescore_report(evals, root, splits):
    """AC-26 for each kept model, as {arm: errors}. A missing file is a failure, not a pass."""
    report = {}
    for arm, (run, old) in KEPT.items():
        new, old = evals / f"{Path(run).name}-val", Path(root) / old
        lost = [d / "metrics.csv" for d in (new, old) if not (d / "metrics.csv").is_file()]
        report[arm] = ([f"{path} missing" for path in lost] if lost
                       else rescore_problems(new, old, splits))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--evals", type=Path, default=REPO / "results/0007-confirmation/evals")
    ap.add_argument("--root", type=Path, default=REPO, help="the folder that holds results/")
    ap.add_argument("--splits", type=Path, default=REPO / "data/splits/splits.csv")
    a = ap.parse_args(argv)
    errors, dirs = cs.manifest_problems(a.runs, a.root)
    errors += location_problems(dirs, a.root)
    if dirs:
        errors += replaced_problems(a.runs, dirs)
    errors += e3b_config_problems()
    if len(dirs) == len(cs.slots()):
        more, recs = cs.run_problems(dirs)
        errors += more
        if recs:
            errors += channel_problems(recs) + commit_problems(recs)
            errors += eval_commit_problems(a.evals, recs)
    for arm, found in rescore_report(a.evals, a.root, a.splits).items():
        print(f"AC-26 {arm}: {'FAIL' if found else 'ok'}")
        errors += [f"AC-26 {arm}: {e}" for e in found]
    for e in errors:
        print(f"FAIL: {e}")
    if errors:
        return 1
    print(f"PASS: {len(cs.slots())} models, seed 42 rescored, one clean commit, uv.lock unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
