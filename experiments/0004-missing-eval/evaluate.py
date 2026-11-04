"""Score the frozen E1 best.pt on all 15 missing modality patterns (spec 0004, RQ1).

    uv run python experiments/0004-missing-eval/evaluate.py --run <E1 run dir> --fold val
Reads the E1 run read only, and writes <out>/<source run id>-<fold>[-quick]/ holding config.toml,
run.json and metrics.csv (n x 15 x 3 rows). --quick scores the first 2 cases of the fold, all 15
patterns. Exit 1 on a bad source run or a failed sweep; 2 on a guard (subjects, cache, folder).
--fold test is refused for now: the test gate of AC-7 is not built yet.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cache import check_cache, repo_path, sha256_file
from common.evaluation import EVAL_RUN_KEYS, GuardError, eval_cases, metrics_writer, sweep
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--fold", required=True, choices=["val", "test"])
    ap.add_argument("--out", type=Path, default=REPO / "results" / "0004-missing-eval")
    ap.add_argument("--config", type=Path, default=EXP / "config.toml")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)

    if a.fold == "test":  # first, before any case list or .npz (AC-7)
        print("ERROR: the test gate (spec 0004, AC-7) is not built yet; nothing read")
        return 2

    rec = load_run(a.run)
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
    eval_cfg = {**load_config(a.config)["eval"], "quick": a.quick}
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
    if eval_dir.exists():  # the full rule of AC-6 comes with the run record milestone
        print(f"ERROR: {eval_dir} already exists; nothing evaluated")
        return 2
    eval_dir.mkdir(parents=True)
    saved = {"eval": eval_cfg}
    dump_config(saved, eval_dir / "config.toml")

    set_seed(eval_cfg["seed"])
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    commit, dirty = git_info()
    run = RunRecord(
        eval_dir, keys=EVAL_RUN_KEYS, defaults={"resumes": []}, eval_id=eval_id, quick=a.quick,
        fold=a.fold, n_cases=len(cases), source_run=str(a.run), source_run_id=rec["run_id"],
        source_config_hash=rec["config_hash"], source_best_iteration=rec["best_iteration"],
        source_seed=rec["seed"], splits_sha256=sha256_file(repo_path(cfg["data"]["splits_csv"])),
        config_hash=config_hash(saved), git_commit=commit, git_dirty=dirty,
        seed=eval_cfg["seed"], device=device.type,
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None, versions=versions())
    print(f"{eval_id}: {len(cases)} cases, config {run.data['config_hash'][:12]}, git {commit}")

    start = time.time()
    try:
        with open(eval_dir / "metrics.csv", "w", newline="") as f:
            seconds = sweep(model, cases, cfg, device, metrics_writer(f, eval_id))
    except Exception as e:  # noqa: BLE001 (any failure must end as status failed, AC-10)
        run.update(status="failed", reason=failed_reason(e), wall_time_s=time.time() - start)
        run.write()
        print(f"ERROR: sweep failed: {failed_reason(e)}")
        return 1
    run.update(
        status="ok", wall_time_s=time.time() - start, **seconds,
        peak_vram_reserved_mib=(torch.cuda.max_memory_reserved() / 2**20
                                if device.type == "cuda" else None))
    run.write()
    print(f"{len(cases)} cases x 15 patterns x 3 regions -> {eval_dir / 'metrics.csv'}")
    print(f"predict {seconds['predict_seconds']:.1f} s, metrics {seconds['metric_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
