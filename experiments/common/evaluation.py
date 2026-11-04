"""The shared sweep over the 15 missing modality patterns (spec 0004, AC-1, AC-3, AC-4).

E3 and E4 reuse this module, so every strategy is scored by the same code. One case and pattern
pair gives 3 rows, one per region, and the pattern 1 path is the same as every other.
"""
import csv
import math
import os
import time

from common.cache import cache_dir, repo_path
from common.data import REGIONS, cached_eval_transforms, cached_items, fold_cases, subject_of
from common.metrics import region_metrics
from common.missing import PATTERNS, apply_pattern
from common.model import predict

METRIC_COLUMNS = ["eval_id", "case", "subject", "pattern_id", "pattern_name", "region", "dice",
                  "hd95_mm", "sensitivity", "precision", "gt_empty", "pred_empty"]
# every key is present in every eval run.json (null where not yet known); spec 0004, Data model
EVAL_RUN_KEYS = (
    "eval_id", "status", "reason", "quick", "fold", "n_cases", "source_run", "source_run_id",
    "source_config_hash", "source_best_iteration", "source_seed", "splits_sha256",
    "protocol_hash", "config_hash", "git_commit", "git_dirty", "seed", "device", "gpu",
    "versions", "rerun_reason", "superseded", "resumes", "predict_seconds", "metric_seconds",
    "wall_time_s", "peak_vram_reserved_mib",
)


class GuardError(Exception):
    """A check refused the run before any GPU work (exit 2)."""


def identity(image, pattern_id):
    """The default image_hook: no change."""
    return image


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
          image_hook=identity):
    """Score each (case, pattern) not in done, cases sorted, patterns ascending.

    image_hook(image, pattern_id) runs after apply_pattern() and returns an image. writer(rows)
    takes the 3 row dicts of one pair. Returns the summed timings. Raises on a NaN Dice or HD95,
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
    return seconds
