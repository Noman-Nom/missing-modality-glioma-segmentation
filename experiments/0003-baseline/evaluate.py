"""Score best.pt of an E1 run on its val cases, complete input only (spec 0003, AC-13).

    uv run python experiments/0003-baseline/evaluate.py --run <run dir>
Reads the run's config.toml and run.json val_cases (never selects again) and writes
metrics_val.csv there: n_val x 3 rows. The test fold is never read (E2 owns it).
Exit 1 if the run status is not ok, best.pt is missing, or its config hash differs from run.json;
2 if the cache guard fails.
"""
import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.cache import cache_dir, check_cache
from common.data import REGIONS, cached_eval_transforms, cached_items, subject_of
from common.metrics import region_metrics
from common.missing import apply_pattern
from common.model import build_model, predict
from common.runlog import load_config, load_run
from common.seed import set_seed

COLUMNS = ["run_id", "case", "subject", "region", "dice", "hd95_mm", "sensitivity", "precision",
           "gt_empty", "pred_empty"]
PATTERN = 1  # complete input


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    a = ap.parse_args(argv)

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
    cases = rec["val_cases"]
    errors = check_cache(cfg, cases)
    if errors:
        print("ERROR: cache guard, nothing evaluated:\n" + "\n".join("  " + e for e in errors[:20]))
        return 2

    set_seed(cfg["seed"])
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []
    tf = cached_eval_transforms(cfg)
    for item in cached_items(cache_dir(cfg), cases):
        d = tf(item)
        pred = predict(model, apply_pattern(d["image"], PATTERN), cfg, device)
        for ri, region in enumerate(REGIONS):
            m = region_metrics(pred[ri], d["label"][ri], d["pixdim"])
            assert not math.isnan(m["dice"]) and not math.isnan(m["hd95_mm"])
            rows.append({"run_id": rec["run_id"], "case": d["case"],
                         "subject": subject_of(d["case"]), "region": region, **m})

    out = a.run / "metrics_val.csv"
    tmp = out.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if isinstance(v, float) and math.isnan(v) else v
                        for k, v in r.items()})
    tmp.replace(out)
    print(f"{len(cases)} cases x {len(REGIONS)} regions = {len(rows)} rows -> {out}")
    print("mean Dice " + "  ".join(
        f"{g} {statistics.mean(r['dice'] for r in rows if r['region'] == g):.4f}"
        for g in REGIONS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
