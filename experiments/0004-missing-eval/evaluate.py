"""Score the frozen E1 best.pt on all 15 missing modality patterns (spec 0004, RQ1).

    uv run python experiments/0004-missing-eval/evaluate.py --run <E1 run dir> --fold val
Reads the E1 run read only, and writes <out>/<source run id>-<fold>[-quick]/ holding config.toml,
run.json and metrics.csv (n x 15 x 3 rows). --quick scores the first 2 cases of the fold, all 15
patterns. An unfinished folder resumes with the same command. A finished one is refused unless
--rerun-reason "<text>" is given, which renames it .superseded-<utc time>. --fold test also needs
--open-test and every rule of the test gate (AC-7): nothing of the test fold is read before it.
Exit 1 on a bad source run or a failed sweep; 2 on a guard (subjects, cache, gate, folder, resume).
Test hook: EVAL_STOP_AFTER=<N> stops after N pairs, leaving status running like a killed session.
"""
import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cache import check_cache
from common.evaluation import (
    EVAL_RUN_KEYS,
    GuardError,
    check_test_gate,
    clean_metrics,
    eval_cases,
    metrics_writer,
    prepare_folder,
    protocol_hash,
    source_pins,
    sweep,
)
from common.model import build_model
from common.runlog import (
    RunRecord,
    config_hash,
    dump_config,
    git_info,
    load_config,
    load_run,
    versions,
)
from common.seed import set_seed
from common.training import failed_reason

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]
COUNTERS = ("predict_seconds", "metric_seconds", "wall_time_s")


def progress_recorder(run, device):
    """A sweep progress callback: run.json holds the totals after every pair, so a kill loses
    no time. Timings add up across resumes, and the peak keeps the maximum (AC-5)."""
    base = {k: run.data[k] for k in COUNTERS}
    peak = run.data["peak_vram_reserved_mib"]
    start = time.time()
    stop_after = int(os.environ.get("EVAL_STOP_AFTER", "0"))
    pairs = 0

    def record(seconds):
        nonlocal pairs, peak
        if device.type == "cuda":
            peak = max(peak or 0.0, torch.cuda.max_memory_reserved() / 2**20)
        run.update(predict_seconds=base["predict_seconds"] + seconds["predict_seconds"],
                   metric_seconds=base["metric_seconds"] + seconds["metric_seconds"],
                   wall_time_s=base["wall_time_s"] + time.time() - start,
                   peak_vram_reserved_mib=peak)
        run.write()
        pairs += 1
        if pairs == stop_after:
            raise SystemExit(3)

    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--fold", required=True, choices=["val", "test"])
    ap.add_argument("--out", type=Path, default=REPO / "results" / "0004-missing-eval")
    ap.add_argument("--config", type=Path, default=EXP / "config.toml")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--open-test", action="store_true")
    ap.add_argument("--rerun-reason", default="")
    a = ap.parse_args(argv)

    rec = load_run(a.run)
    commit, dirty = git_info()
    protocol = protocol_hash()
    eval_cfg = {**load_config(a.config)["eval"], "quick": a.quick}
    saved = {"eval": eval_cfg}
    # first, before any case list or .npz (AC-7)
    errors = check_test_gate(a.fold, a.open_test, a.quick, rec, (commit, dirty),
                             a.out / f"{rec['run_id']}-val", a.run, protocol, config_hash(saved))
    if errors:
        print("ERROR: test gate, nothing read:\n" + "\n".join("  " + e for e in errors))
        return 2

    if rec["status"] != "ok":
        print(f"ERROR: run status is {rec['status']!r}, not 'ok'; nothing evaluated")
        return 1
    best = a.run / "checkpoints" / "best.pt"
    if not best.is_file():
        print(f"ERROR: {best} missing")
        return 1
    state = torch.load(best, map_location="cpu")
    if state["config_hash"] != rec["config_hash"]:
        print(f"ERROR: best.pt config hash {state['config_hash']} differs from run.json "
              f"{rec['config_hash']}; nothing evaluated")
        return 1

    cfg = load_config(a.run / "config.toml")
    try:
        cases = eval_cases(cfg, eval_cfg, rec, a.fold)
    except GuardError as e:
        print(f"ERROR: {e}; nothing evaluated")
        return 2
    if a.quick:
        cases = cases[:2]
    errors = check_cache(cfg, cases, allowed_folds=(a.fold,))
    if errors:
        print("ERROR: cache guard, nothing evaluated:\n" + "\n".join("  " + e for e in errors[:20]))
        return 2

    eval_id = f"{rec['run_id']}-{a.fold}" + ("-quick" if a.quick else "")
    eval_dir = a.out / eval_id
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu")
    fields = {
        "eval_id": eval_id, "quick": a.quick, "fold": a.fold, "n_cases": len(cases),
        "source_run": str(a.run), "source_run_id": rec["run_id"],
        "source_config_hash": rec["config_hash"], "source_seed": rec["seed"],
        "protocol_hash": protocol, **source_pins(a.run, rec),
        "config_hash": config_hash(saved), "git_commit": commit, "git_dirty": dirty,
        "seed": eval_cfg["seed"], "device": device.type, "versions": versions(),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}
    reason = a.rerun_reason.strip()
    metrics = eval_dir / "metrics.csv"
    try:
        old, superseded = prepare_folder(eval_dir, fields, reason)
        done = clean_metrics(metrics) if old is not None else set()
    except GuardError as e:
        print(f"ERROR: {e}; nothing evaluated")
        return 2

    eval_dir.mkdir(parents=True, exist_ok=True)
    if old is not None:
        run = RunRecord(
            eval_dir, keys=EVAL_RUN_KEYS, defaults={k: v for k, v in old.items() if k != "status"},
            reason=None, resumes=old["resumes"] + [{
                "utc": datetime.now(UTC).isoformat(timespec="seconds"), "done_pairs": len(done),
                "git_commit": commit, "previous_status": old["status"]}])
        print(f"{eval_id}: resuming, {len(done)} pairs done")
    else:
        dump_config(saved, eval_dir / "config.toml")
        run = RunRecord(
            eval_dir, keys=EVAL_RUN_KEYS,
            defaults={"resumes": [], **dict.fromkeys(COUNTERS, 0.0)},
            rerun_reason=reason or None, superseded=superseded, **fields)
    print(f"{eval_id}: {len(cases)} cases, config {fields['config_hash'][:12]}, git {commit}")

    try:
        set_seed(eval_cfg["seed"])
        model = build_model(cfg).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        with open(metrics, "a", newline="") as f:
            sweep(model, cases, cfg, device, metrics_writer(f, eval_id), done=done,
                  progress=progress_recorder(run, device))
    except Exception as e:  # noqa: BLE001 (any failure must end as status failed, AC-10)
        run.update(status="failed", reason=failed_reason(e))
        run.write()
        print(f"ERROR: sweep failed: {failed_reason(e)}")
        return 1
    run.update(status="ok")
    run.write()
    print(f"{len(cases)} cases x 15 patterns x 3 regions -> {metrics}")
    print(f"predict {run.data['predict_seconds']:.1f} s, "
          f"metrics {run.data['metric_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
