"""The shared sweep over the 15 missing modality patterns, and the rules around it (spec 0004).

E3 and E4 reuse this module, so every strategy is scored by the same code. One case and pattern
pair gives 3 rows, one per region, and the pattern 1 path is the same as every other. The folder
rule, resume, the test gate and check_rules live here too (AC-5, AC-6, AC-7, AC-9, AC-16).
"""
import csv
import hashlib
import inspect
import json
import math
import os
import statistics
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from common.cache import cache_dir, repo_path, sha256_file
from common.data import (
    MODALITIES,
    REGION_LABELS,
    REGIONS,
    CachedCase,
    cached_eval_transforms,
    cached_items,
    fold_cases,
    label_to_regions,
    subject_of,
)
from common.metrics import region_metrics
from common.missing import PATTERNS, apply_pattern
from common.model import predict
from common.runlog import config_hash, load_config, load_run

REPO = Path(__file__).resolve().parents[2]
# order matters; whole model.py and data.py are left out on purpose, E4 changes the input channels
# in build_model() and E3 may add training transforms, so PROTOCOL_SOURCES seals only what a case
# passes through at scoring time (spec 0004, AC-20)
PROTOCOL_FILES = (
    "experiments/0004-missing-eval/config.toml", "experiments/0004-missing-eval/analyse.py",
    "experiments/common/evaluation.py", "experiments/common/metrics.py",
    "experiments/common/missing.py", "experiments/common/stats.py",
)
PROTOCOL_SOURCES = (predict, cached_items, cached_eval_transforms, CachedCase, label_to_regions)
# what a compare depends on, after PROTOCOL_FILES: no file here may move after the tag (spec 0006,
# AC-28, AC-33). This file holds the list, so the gate on Kaggle can run the seal
SEALED_FILES = (*PROTOCOL_FILES, "experiments/common/strategies.py", "experiments/common/model.py",
                "experiments/0004-missing-eval/evaluate.py",
                "experiments/0007-confirmation/compare_seeds.py", "experiments/common/runlog.py",
                "experiments/common/cache.py", "experiments/common/data.py",
                "experiments/0007-confirmation/rq1_seeds.py")
METRIC_COLUMNS = ["eval_id", "case", "subject", "pattern_id", "pattern_name", "region", "dice",
                  "hd95_mm", "sensitivity", "precision", "gt_empty", "pred_empty", "pred_ml",
                  "true_ml"]
# every key is present in every eval run.json (null where not yet known); spec 0004, Data model
EVAL_RUN_KEYS = (
    "eval_id", "status", "reason", "quick", "fold", "n_cases", "source_run", "source_run_id",
    "source_config_hash", "source_best_iteration", "source_best_sha256", "source_seed",
    "splits_sha256", "protocol_hash", "config_hash", "git_commit", "git_dirty", "seed", "device", "gpu",
    "versions", "rerun_reason", "superseded", "resumes", "predict_seconds", "metric_seconds",
    "wall_time_s", "peak_vram_reserved_mib",
)
# a resume stops if any of these differs from run.json (AC-5); paths are not here, Kaggle moves them
RESUME_KEYS = ("git_commit", "config_hash", "protocol_hash", "source_run_id",
               "source_config_hash", "source_best_iteration", "source_best_sha256", "device", "gpu")
# a test eval must hold the same value as its val eval on each of these (AC-7, AC-16), gpu too
# (spec 0006, AC-36)
VAL_MATCH_KEYS = ("protocol_hash", "config_hash", "source_best_iteration", "source_best_sha256",
                  "splits_sha256", "gpu")
PREREG_TAG = "prereg-v1"  # the current protocol tag, moved only by spec 0006, AC-31
# the human's literature check, one `key: value` line each, as a path inside the repo (AC-19)
LITERATURE_GATE = "docs/literature/test-gate.md"
GATE_FILE = REPO / LITERATURE_GATE
GATE_KEYS = ("status", "date", "papers_read", "cai_status", "citation_search")
MIN_PAPERS_READ = 50
COMPARE_VAL = REPO / "results" / "0007-confirmation" / "compare-val"  # spec 0006, AC-27
PREFLIGHT = "preflight.json"  # written beside the val compare record when the preflight passes
MAX_MEAN_DICE_DIFF = 0.001  # AC-9, per region
MAX_CASE_DICE_DIFF = 0.01  # AC-9, per (case, region)


class GuardError(Exception):
    """A check refused the run before any GPU work (exit 2)."""


def identity(image, pattern_id):
    """The default image_hook: no change."""
    return image


def source_pins(run_dir, rec):
    """The run.json fields that tie an eval to the exact E1 weights and split file (AC-7)."""
    run_dir = Path(run_dir)
    splits = load_config(run_dir / "config.toml")["data"]["splits_csv"]
    return {"source_best_iteration": rec["best_iteration"],
            "source_best_sha256": sha256_file(run_dir / "checkpoints" / "best.pt"),
            "splits_sha256": sha256_file(repo_path(splits))}


def eval_cases(cfg, eval_cfg, rec, fold):
    """The AC-1 case list, sorted. cfg is the E1 config, eval_cfg its [eval] table, rec its run.json.

    Raises GuardError if a chosen case shares a subject with train, or a test case with val.
    """
    if fold == "val":
        cases = sorted(rec["val_cases"])
    elif fold == "test":
        cases = fold_cases(repo_path(cfg["data"]["splits_csv"]), "test",
                           timepoints=eval_cfg["test_timepoints"],
                           exclude_flair_flag=cfg["data"]["exclude_flair_flag"])
    else:
        raise ValueError(f"unknown fold {fold!r}")
    subjects = {subject_of(c) for c in cases}
    others = rec["train_cases"] + (rec["val_cases"] if fold == "test" else [])
    shared = sorted(subjects & {subject_of(c) for c in others})
    if shared:
        raise GuardError(f"{fold} shares {len(shared)} subject(s) with the run's other cases: "
                         f"{shared[:3]}")
    return cases


def protocol_hash():
    """sha256 over the bytes of PROTOCOL_FILES, then the source text of PROTOCOL_SOURCES, then the
    region and modality constants, in that order. A source that cannot be read raises: a skipped
    source would seal less than the record says (AC-20)."""
    h = hashlib.sha256()
    for name in PROTOCOL_FILES:
        h.update((REPO / name).read_bytes())
    for source in PROTOCOL_SOURCES:
        h.update(inspect.getsource(source).encode("utf-8"))
    h.update(repr((MODALITIES, REGIONS, REGION_LABELS)).encode("utf-8"))
    return h.hexdigest()


def compare_hash():
    """sha256 over the bytes of SEALED_FILES, in that order (spec 0006, AC-28)."""
    h = hashlib.sha256()
    for name in SEALED_FILES:
        h.update((REPO / name).read_bytes())
    return h.hexdigest()


def prepare_folder(eval_dir, fields, rerun_reason=None):
    """Apply the one finished run rule to eval_dir (AC-5, AC-6). Raises GuardError on a refusal.

    Returns (old, superseded). old is the run.json to resume, or None for a fresh start.
    superseded is the new name of a finished folder moved aside, or None. Nothing is renamed or
    cut unless the run may go ahead. A reason only replaces a finished folder: an unfinished one
    resumes without it, and an empty folder is a start that died before its run.json.
    """
    eval_dir = Path(eval_dir)
    old = load_run(eval_dir) if (eval_dir / "run.json").is_file() else None
    if old is None and (eval_dir / "metrics.csv").exists():
        raise GuardError(f"{eval_dir} has metrics.csv but no run.json")
    if old is not None and old["status"] == "ok":
        if not rerun_reason:
            raise GuardError(f"{eval_dir} is finished; give --rerun-reason to replace it")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        moved = eval_dir.with_name(f"{eval_dir.name}.superseded-{stamp}")
        eval_dir.rename(moved)
        return None, moved.name
    if rerun_reason:
        raise GuardError(f"--rerun-reason replaces a finished folder, and {eval_dir} is not one")
    if old is not None:
        differ = [f"{k}: run.json {old[k]!r}, now {fields[k]!r}" for k in RESUME_KEYS
                  if old[k] != fields[k]]
        if differ:
            raise GuardError("cannot resume, " + "; ".join(differ))
    return old, None


def _whole_pair(lines):
    """True if 3 complete metrics.csv lines are the 3 regions of one (case, pattern)."""
    rows = [next(csv.reader([b.decode(errors="replace")]), []) for b in lines]
    if any(len(r) != len(METRIC_COLUMNS) for r in rows):
        return False
    return ([r[METRIC_COLUMNS.index("region")] for r in rows] == list(REGIONS)
            and len({(r[1], r[3]) for r in rows}) == 1)


def clean_metrics(path):
    """Cut metrics.csv back to its last whole pair, and return the done (case, pattern) set.

    A kill can leave a truncated line or part of a pair. A row is never edited, only cut.
    """
    path = Path(path)
    if not path.is_file():
        return set()
    data = path.read_bytes()
    header, newline, _ = data.partition(b"\n")
    if newline and header.decode(errors="replace").split(",") != METRIC_COLUMNS:
        raise GuardError(f"{path} does not start with the metrics.csv header")
    keep = len(header) + 1 if newline else 0
    done = set()
    # the piece after the last newline is a truncated line, or nothing
    lines = data[keep:].split(b"\n")[:-1]
    for i in range(0, len(lines) - 2, 3):
        group = lines[i:i + 3]
        if not _whole_pair(group):
            break
        first = next(csv.reader([group[0].decode()]))
        done.add((first[1], int(first[3])))
        keep += sum(len(b) + 1 for b in group)
    with open(path, "r+b") as f:
        f.truncate(keep)
        os.fsync(f.fileno())
    return done


def _nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def volume_ml(mask, pixdim):
    """The volume of a binary mask in mL: voxel count x voxel size in mm^3 / 1000 (spec 0006, AC-7).
    Python float, so metrics.csv keeps every digit."""
    voxels = int(np.asarray(mask).astype(bool).sum())
    return float(voxels * math.prod(float(s) for s in pixdim) / 1000)


def _volume_problems(rows):
    """AC-7 of spec 0006: both volume columns present, not negative, true_ml the same on every row
    of a (case, region). Returns error strings."""
    empty = sum(1 for r in rows if _nan(r.get("pred_ml")) or _nan(r.get("true_ml")))
    if empty:
        return [f"{empty} row(s) with no pred_ml or true_ml"]
    errors = []
    negative = sum(1 for r in rows if float(r["pred_ml"]) < 0 or float(r["true_ml"]) < 0)
    if negative:
        errors.append(f"{negative} row(s) with a negative volume")
    true_ml = {}
    for r in rows:
        true_ml.setdefault((r["case"], r["region"]), set()).add(r["true_ml"])
    moved = sum(1 for values in true_ml.values() if len(values) > 1)
    if moved:
        errors.append(f"true_ml is not constant on {moved} (case, region) pair(s)")
    return errors


def _tie_out(rec, rows, source_run):
    """AC-9: pattern 1 of a val eval against the E1 metrics_val.csv. Returns error strings."""
    path = source_run / "metrics_val.csv"
    if not path.is_file():
        return [f"{path} missing, no tie out"]
    errors = []
    gpu = load_run(source_run)["gpu"]
    if rec["gpu"] != gpu:
        errors.append(f"gpu is {rec['gpu']!r}, the E1 run used {gpu!r}")
    with open(path, newline="") as f:
        e1 = {(r["case"], r["region"]): float(r["dice"]) for r in csv.DictReader(f)}
    e2 = {(r["case"], r["region"]): float(r["dice"]) for r in rows if r["pattern_id"] == "1"}
    if set(e1) != set(e2):
        return errors + [f"case set differs from E1: {len(set(e1) ^ set(e2))} (case, region) keys"]
    for g in REGIONS:
        keys = [k for k in e1 if k[1] == g]
        diff = abs(statistics.fmean(e1[k] for k in keys) - statistics.fmean(e2[k] for k in keys))
        if diff > MAX_MEAN_DICE_DIFF:
            errors.append(f"{g} mean Dice differs from E1 by {diff:.6f}")
    far = [k for k in sorted(e1) if abs(e1[k] - e2[k]) > MAX_CASE_DICE_DIFF]
    if far:
        errors.append(f"{len(far)} case Dice value(s) differ from E1 by over "
                      f"{MAX_CASE_DICE_DIFF}, first {far[0]}")
    return errors


def _test_case_set(eval_dir, cases, source_run):
    """AC-16 on test: the cases scored are the AC-1 list. Returns error strings.

    n_cases comes from the eval's own run.json, so it cannot catch a list cut short consistently.
    """
    cfg = load_config(source_run / "config.toml")
    eval_cfg = load_config(eval_dir / "config.toml")["eval"]
    try:
        want = set(eval_cases(cfg, eval_cfg, load_run(source_run), "test"))
    except GuardError as e:
        return [str(e)]
    if cases != want:
        missing, extra = len(want - cases), len(cases - want)
        return [f"case set differs from the test fold: {missing} missing, {extra} extra"]
    return []


def check_rules(eval_dir, source_run):
    """Every rule a finished eval folder must hold (AC-9, AC-16). Returns error strings.

    Shared by check_eval.py and the test gate, so no pass flag can go stale.
    """
    eval_dir, source_run = Path(eval_dir), Path(source_run)
    if not (eval_dir / "run.json").is_file():
        return [f"{eval_dir}: run.json missing"]
    rec = load_run(eval_dir)
    errors = []
    if rec["status"] != "ok":
        errors.append(f"status is {rec['status']!r}, not 'ok'")
    if rec["quick"] is not False:
        errors.append("a quick run")
    config = eval_dir / "config.toml"
    if not config.is_file():
        return errors + [f"{config} missing"]
    if config_hash(load_config(config)) != rec["config_hash"]:
        errors.append("config.toml does not hash to the config_hash in run.json")
    path = eval_dir / "metrics.csv"
    if not path.is_file():
        return errors + [f"{path} missing"]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != rec["n_cases"] * len(PATTERNS) * len(REGIONS):
        errors.append(f"metrics.csv has {len(rows)} rows, expected "
                      f"{rec['n_cases'] * len(PATTERNS) * len(REGIONS)}")
    cases = {r["case"] for r in rows}
    if len(cases) != rec["n_cases"]:
        errors.append(f"metrics.csv has {len(cases)} cases, expected {rec['n_cases']}")
    have = Counter((r["case"], r["pattern_id"], r["region"]) for r in rows)
    want = {(c, str(p), g) for c in cases for p in PATTERNS for g in REGIONS}
    if set(have) != want or max(have.values(), default=1) > 1:
        errors.append(f"(case, pattern, region) rows: {len(want - set(have))} missing, "
                      f"{sum(1 for v in have.values() if v > 1)} repeated")
    nans = sum(1 for r in rows if _nan(r["dice"]) or _nan(r["hd95_mm"]))
    if nans:
        errors.append(f"{nans} row(s) with no Dice or HD95")
        return errors
    errors += _volume_problems(rows)
    if rec["fold"] == "val":
        errors += _tie_out(rec, rows, source_run)
    else:
        errors += _test_case_set(eval_dir, cases, source_run)
        val_dir = eval_dir.parent / f"{rec['source_run_id']}-val"
        if not (val_dir / "run.json").is_file():
            errors.append(f"{val_dir} missing, no hashes to compare")
        else:
            val = load_run(val_dir)
            errors += [f"{k} differs from the val eval's" for k in VAL_MATCH_KEYS
                       if val[k] != rec[k]]
    return errors


def gate_problems(text):
    """Every failed rule of the literature gate (spec 0006, AC-19), given the text of the gate
    file. A repeated key is refused, so a later line cannot quietly change an earlier one."""
    found = {}
    for line in text.splitlines():
        key, colon, value = line.partition(":")
        if colon and key.strip() in GATE_KEYS:
            found.setdefault(key.strip(), []).append(value.strip())
    errors = [f"literature gate: no {k}" for k in GATE_KEYS if not any(found.get(k, []))]
    errors += [f"literature gate: {k} appears {len(v)} times" for k, v in found.items()
               if len(v) > 1]
    if errors:
        return errors
    if found["status"][0] != "pass":
        errors.append(f"literature gate: status is {found['status'][0]!r}, not 'pass'")
    papers = found["papers_read"][0]
    if not papers.isascii() or not papers.isdigit():
        errors.append(f"literature gate: papers_read is {papers!r}, not a whole number")
    elif int(papers) < MIN_PAPERS_READ:
        errors.append(f"literature gate: papers_read is {papers}, below {MIN_PAPERS_READ}")
    return errors


def is_ancestor(older, commit, repo):
    """True if commit holds older. A name git does not know gives False."""
    r = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", str(older),
                        str(commit)], capture_output=True, check=False)
    return r.returncode == 0


def descends_from_tag(commit, tag, repo):
    """True if commit holds the tag. An unknown tag or an unknown commit gives False."""
    return is_ancestor(f"refs/tags/{tag}", commit, repo)


def seal_problems(commit):
    """AC-28 for one commit: no sealed file differs from the tag, and its literature gate passes.
    Asks git about that commit, so an edit in the tree or a rewrite after the opening changes
    nothing here."""
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", f"refs/tags/{PREREG_TAG}",
                           str(commit), "--", *SEALED_FILES], capture_output=True, check=False)
    errors = [f"commit {commit}: a sealed file differs from {PREREG_TAG}"] if diff.returncode else []
    gate = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{LITERATURE_GATE}"],
                          capture_output=True, encoding="utf-8", check=False)
    if gate.returncode:
        return errors + [f"commit {commit}: it holds no {LITERATURE_GATE}"]
    return errors + [f"commit {commit}: {e}" for e in gate_problems(gate.stdout)]


def val_record_problems(rec):
    """AC-35: the val compare record is a clean, finished val compare under this tag, of the
    sealed files as they are now. rec is its run.json. Returns error strings."""
    errors = []
    if rec.get("status") != "ok":
        errors.append(f"val compare: status is {rec.get('status')!r}, not 'ok'")
    if rec.get("fold") != "val":
        errors.append(f"val compare: fold is {rec.get('fold')!r}, not 'val'")
    if rec.get("prereg_tag") != PREREG_TAG:
        errors.append(f"val compare: prereg_tag is {rec.get('prereg_tag')!r}, not {PREREG_TAG!r}")
    if rec.get("git_dirty") is not False:
        errors.append(f"val compare: git_dirty is {rec.get('git_dirty')!r}, must be False")
    if not descends_from_tag(rec.get("git_commit"), PREREG_TAG, REPO):
        errors.append(f"val compare: commit {rec.get('git_commit')} does not descend from "
                      f"{PREREG_TAG}")
    if rec.get("compare_hash") != compare_hash():
        errors.append("compare_hash differs from the val compare's: a sealed file moved")
    return errors


def val_file_problems(path, want):
    """AC-35: path is a val metrics.csv and want its entry in eval_sha256. A missing file is a
    listed error, never a traceback, because this runs before the val eval is checked."""
    if not want:
        return [f"val compare: no eval_sha256 entry for {path}"]
    if not Path(path).is_file():
        return [f"{path} missing, the val compare read it"]
    return [] if sha256_file(path) == want else [f"{path} changed since the val compare"]


def opening_problems(run_id, commit, val_dir):
    """The rules of the one test opening (spec 0006, AC-27, AC-33, AC-35), as error strings: the
    literature gate passes, the protocol tag is an ancestor of this commit, no sealed file moved,
    and the val compare is whole, lists this run, and read the val metrics.csv now in val_dir.
    Reads no case."""
    errors = (gate_problems(GATE_FILE.read_text(encoding="utf-8")) if GATE_FILE.is_file()
              else [f"{GATE_FILE} missing: the literature gate has not passed"])
    if not descends_from_tag(commit, PREREG_TAG, REPO):
        errors.append(f"commit {commit} does not descend from {PREREG_TAG}")
    errors += seal_problems(commit)
    if not (COMPARE_VAL / "run.json").is_file():
        return errors + [f"{COMPARE_VAL} missing: run compare_seeds.py --fold val first"]
    compare = load_run(COMPARE_VAL)
    errors += val_record_problems(compare)
    slots = {run: slot for slot, run in compare.get("run_ids", {}).items()}
    if run_id not in slots:
        return errors + [f"val compare does not list run {run_id}"]
    return errors + val_file_problems(Path(val_dir) / "metrics.csv",
                                      compare.get("eval_sha256", {}).get(slots[run_id]))


def site_problems(val_dir, commit, gpu):
    """The rules that can only be checked where the sweep runs (spec 0006, AC-34, AC-36): this is
    the commit that passed the preflight, on the GPU of the val eval. evaluate.py calls this for
    --fold test beside check_test_gate. The preflight does not, because its own record does not
    exist yet and the laptop GPU is not the Kaggle one. Every missing file is a listed error."""
    errors = []
    path = COMPARE_VAL / PREFLIGHT
    if not path.is_file():
        errors.append(f"{path} missing: the preflight has not passed, or its record is not here")
    else:
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            errors.append(f"preflight: status is {rec.get('status')!r}, not 'ok'")
        if rec.get("git_commit") != commit:
            errors.append(f"preflight passed on {rec.get('git_commit')}, this is {commit}")
    if not (Path(val_dir) / "run.json").is_file():
        return errors + [f"{val_dir} missing: no val GPU to compare"]
    val_gpu = load_run(val_dir).get("gpu")
    if gpu != val_gpu:
        errors.append(f"gpu is {gpu!r}, the val eval used {val_gpu!r}")
    return errors


def check_test_gate(fold, open_test, quick, rec, git, val_dir, run_dir, protocol, config):
    """Every failed rule of the test gate (AC-7), as error strings. Empty means the gate is open.

    rec is the E1 run.json and git is (commit, dirty) from git_info(). config is the config_hash
    of this run: protocol seals only the default config.toml, so --config needs its own check.
    source_pins() checks that the weights and the split file are still the ones val scored, and
    opening_problems() adds the rules of spec 0006. Every model passes through here, so the
    preflight of compare_seeds.py runs these same rules, and only site_problems() is added on
    Kaggle. It reads no case. It stops accidents; it is not proof of one opening.
    """
    if fold != "test":
        return ["--open-test is only for --fold test"] if open_test else []
    errors = []
    if not open_test:
        errors.append("--open-test not given")
    if quick:
        errors.append("--quick is not allowed on the test fold")
    if git[1] is not False:
        errors.append(f"git_dirty is {git[1]!r}, must be False: commit first")
    if rec["quick"] is not False:
        errors.append("the E1 run is a quick run")
    val_dir = Path(val_dir)
    errors += opening_problems(rec["run_id"], git[0], val_dir)
    if not (val_dir / "run.json").is_file():
        return errors + [f"{val_dir} missing: finish the val eval first"]
    errors += [f"val eval: {e}" for e in check_rules(val_dir, run_dir)]
    val = load_run(val_dir)
    if val["protocol_hash"] != protocol:
        errors.append("protocol_hash differs from the val eval's: rerun val with --rerun-reason")
    if val["config_hash"] != config:
        errors.append("config_hash differs from the val eval's: use the config the val eval used")
    try:
        pins = source_pins(run_dir, rec)
    except FileNotFoundError as e:
        return errors + [f"{e.filename} missing, cannot compare it with the val eval"]
    errors += [f"{k} differs from the val eval's: restore it or rerun val with --rerun-reason"
               for k, v in pins.items() if val[k] != v]
    return errors


def _cell(v):
    """floats keep every digit (repr), a NaN is an empty field."""
    if isinstance(v, float):
        return "" if math.isnan(v) else repr(float(v))
    return v


def metrics_writer(f, eval_id):
    """A sweep writer on the open CSV f: adds eval_id, then flushes and syncs each pair's rows."""
    w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, lineterminator="\n")
    if f.tell() == 0:
        w.writeheader()

    def write(rows):
        for r in rows:
            w.writerow({k: _cell(v) for k, v in r.items()} | {"eval_id": eval_id})
        f.flush()
        os.fsync(f.fileno())

    return write


def sweep(model, cases, cfg, device, writer, patterns=range(1, 16), done=frozenset(),
          image_hook=identity, progress=None):
    """Score each (case, pattern) not in done, cases sorted, patterns ascending.

    image_hook(image, pattern_id) runs after apply_pattern() and returns an image. writer(rows)
    takes the 3 row dicts of one pair. Returns the summed timings, and gives them to
    progress(seconds) after each pair, so a kill loses no time. Raises on a NaN Dice or HD95,
    before that pair reaches the writer.
    """
    transform = cached_eval_transforms(cfg)
    seconds = {"predict_seconds": 0.0, "metric_seconds": 0.0}
    for item in cached_items(cache_dir(cfg), sorted(cases)):
        case = item["case"]
        todo = [p for p in sorted(patterns) if (case, p) not in done]
        if not todo:
            continue
        d = transform(item)
        for p in todo:
            start = time.perf_counter()
            pred = predict(model, image_hook(apply_pattern(d["image"], p), p), cfg, device)
            middle = time.perf_counter()
            rows = []
            for ri, region in enumerate(REGIONS):
                m = region_metrics(pred[ri], d["label"][ri], d["pixdim"], with_hd95=True)
                if math.isnan(m["dice"]) or math.isnan(m["hd95_mm"]):
                    raise ValueError(f"NaN metric in {case} pattern {p} {region}")
                rows.append({"case": case, "subject": subject_of(case), "pattern_id": p,
                             "pattern_name": PATTERNS[p].name, "region": region, **m,
                             "pred_ml": volume_ml(pred[ri], d["pixdim"]),
                             "true_ml": volume_ml(d["label"][ri], d["pixdim"])})
            seconds["predict_seconds"] += middle - start
            seconds["metric_seconds"] += time.perf_counter() - middle
            writer(rows)
            if progress:
                progress(seconds)
    return seconds
