"""Evaluate last.pt of a tracer run under all 15 missing modality patterns (spec 0001, AC-7).

    uv run python experiments/0002-tracer-bullet/evaluate.py --run <run dir>
Reads the run's config.toml and run.json val_cases (never selects again). Writes metrics.csv there.
Checkpoint rule (spec 0002): a fixed iteration budget, report the final weights (last.pt), no
checkpoint selection. The val WT Dice in run.json is a learning curve only.
Exit 1 if the run status is not ok, last.pt is missing, or its config hash differs from run.json;
2 if input files are missing.
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import REGIONS, check_inputs, eval_transforms, items, subject_of
from common.metrics import region_metrics
from common.missing import PATTERNS, apply_pattern
from common.model import build_model, predict
from common.runlog import load_config, load_run
from common.seed import set_seed

COLUMNS = ["run_id", "case", "subject", "pattern_id", "pattern", "region", "dice", "hd95_mm",
           "sensitivity", "precision", "gt_empty", "pred_empty"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data-dir", default=os.environ.get("BRATS_TRAIN_DIR"),
                    help="default: BRATS_TRAIN_DIR, else the data_dir recorded in run.json")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    a = ap.parse_args(argv)

    rec = load_run(a.run)
    if rec["status"] != "ok":
        print(f"ERROR: run status is {rec['status']!r}, not 'ok'; nothing evaluated")
        return 1
    last = a.run / "checkpoints" / "last.pt"
    if not last.is_file():
        print(f"ERROR: {last} missing")
        return 1
    state = torch.load(last, map_location="cpu")
    if state["config_hash"] != rec["config_hash"]:
        print(f"ERROR: last.pt config hash {state['config_hash']} differs from run.json "
              f"{rec['config_hash']}; nothing evaluated")
        return 1
    cfg = load_config(a.run / "config.toml")
    data_dir = a.data_dir or rec["data_dir"]
    cases = rec["val_cases"]
    missing = check_inputs(data_dir, cases)
    if missing:
        print(f"ERROR: {len(missing)} input file(s) missing:\n" + "\n".join("  " + m for m in missing))
        return 2

    set_seed(cfg["seed"])
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []
    tf = eval_transforms(cfg)
    for item in items(data_dir, cases):
        d = tf(item)  # crop box from all 4 raw channels, before any pattern
        for pid, pat in PATTERNS.items():
            pred = predict(model, apply_pattern(d["image"], pid), cfg, device)
            for ri, region in enumerate(REGIONS):
                m = region_metrics(pred[ri], d["label"][ri], d["pixdim"])
                assert not math.isnan(m["dice"]) and not math.isnan(m["hd95_mm"])
                rows.append({"run_id": rec["run_id"], "case": d["case"],
                             "subject": subject_of(d["case"]), "pattern_id": pid,
                             "pattern": pat.name, "region": region, **m})

    out = a.run / "metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if isinstance(v, float) and math.isnan(v) else v
                        for k, v in r.items()})
    wt1 = [r["dice"] for r in rows if r["pattern_id"] == 1 and r["region"] == "WT"]
    print(f"{len(cases)} cases x {len(PATTERNS)} patterns x {len(REGIONS)} regions = "
          f"{len(rows)} rows -> {out}")
    print(f"pattern 1 ({PATTERNS[1].name}) mean WT Dice {sum(wt1) / len(wt1):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
