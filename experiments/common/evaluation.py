"""The shared sweep over the 15 missing modality patterns, and the rules around it (spec 0004).

E3 and E4 reuse this module, so every strategy is scored by the same code. One case and pattern
pair gives 3 rows, one per region, and the pattern 1 path is the same as every other. The folder
rule, resume, the test gate and check_rules live here too (AC-5, AC-6, AC-7, AC-9, AC-16).
"""
import csv
import hashlib
import inspect
import math
import os
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

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
METRIC_COLUMNS = ["eval_id", "case", "subject", "pattern_id", "pattern_name", "region", "dice",
                  "hd95_mm", "sensitivity", "precision", "gt_empty", "pred_empty"]
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
# a test eval must hold the same value as its val eval on each of these (AC-7, AC-16)
VAL_MATCH_KEYS = ("protocol_hash", "config_hash", "source_best_iteration", "source_best_sha256",
                  "splits_sha256")
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


def check_test_gate(fold, open_test, quick, rec, git, val_dir, run_dir, protocol, config):
    """Every failed rule of the test gate (AC-7), as error strings. Empty means the gate is open.

    rec is the E1 run.json and git is (commit, dirty) from git_info(). config is the config_hash
    of this run: protocol seals only the default config.toml, so --config needs its own check.
    source_pins() checks that the weights and the split file are still the ones val scored. It
    takes plain values, so E3 and E4 can reuse it. It reads no case. It stops accidents; it is not
    proof of one opening.
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
                             "pattern_name": PATTERNS[p].name, "region": region, **m})
            seconds["predict_seconds"] += middle - start
            seconds["metric_seconds"] += time.perf_counter() - middle
            writer(rows)
            if progress:
                progress(seconds)
    return seconds
