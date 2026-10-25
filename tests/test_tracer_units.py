"""Spec 0001: pure helpers of the tracer scripts. AC-6 loss window, the --quick override, and
the AC-12 table (statistics, both empty and one empty counts per spec 0002 AC-9, refusal on a
wrong row count)."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest
from common.runlog import RUN_KEYS

EXP = Path(__file__).resolve().parents[1] / "experiments" / "0002-tracer-bullet"
COLUMNS = ["run_id", "case", "subject", "pattern_id", "pattern", "region", "dice", "hd95_mm",
           "sensitivity", "precision", "gt_empty", "pred_empty"]


def load(name):
    spec = importlib.util.spec_from_file_location(f"tracer_unit_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train, make_table = load("train"), load("make_table")


# AC-6: W = min(50, iterations // 2)
# covers: AC-6
def test_loss_window_full_run_uses_50():
    losses = [1.0] * 50 + [0.5] * 1900 + [0.2] * 50
    assert train.loss_window(losses) == (50, 1.0, pytest.approx(0.2))


# covers: AC-6
def test_loss_window_short_run_uses_half():
    assert train.loss_window([4.0, 2.0, 1.0, 3.0, 0.0]) == (2, 3.0, 1.5)


@pytest.mark.parametrize("losses", [[], [0.7]])
def test_loss_window_too_short_gives_no_means(losses):
    assert train.loss_window(losses) == (0, None, None)


def test_apply_quick_overrides_and_leaves_the_input_alone():
    cfg = {"seed": 42, "data": {"n_train_subjects": 16, "n_val_subjects": 4, "timepoint": 0},
           "train": {"iterations": 2000, "val_every": 250, "lr": 0.0003}}
    quick = train.apply_quick(cfg)
    assert quick["data"] == {"n_train_subjects": 2, "n_val_subjects": 1, "timepoint": 0}
    assert quick["train"] == {"iterations": 10, "val_every": 5, "lr": 0.0003}
    assert cfg["train"]["iterations"] == 2000 and cfg["data"]["n_train_subjects"] == 16


def row(pid, region, dice, hd, gt_empty=False, pred_empty=False, case="BraTS-GLI-00005-000"):
    return {"run_id": "r", "case": case, "subject": case.split("-")[2], "pattern_id": str(pid),
            "pattern": "x", "region": region, "dice": str(dice), "hd95_mm": str(hd),
            "sensitivity": "", "precision": "", "gt_empty": str(gt_empty),
            "pred_empty": str(pred_empty)}


# covers: AC-12; spec 0002, AC-9
def test_cell_stats_mean_median_and_empty_counts():
    rows = [row(1, "ET", 0.9, 2.0), row(1, "ET", 0.6, 4.0), row(1, "ET", 0.0, 373.13,
                                                              pred_empty=True),
            row(1, "ET", 1.0, 0.0, gt_empty=True, pred_empty=True)]
    # Dice mean 0.625, median 0.75; HD95 mean 94.7825, median 3.0; 1 both empty, 1 one empty
    assert make_table.cell_stats(rows) == ("0.625", "0.750", "94.8", "3.0", "1", "1")


# spec 0002, AC-9: a miss on the truth side counts as one empty, like a miss on the prediction side
def test_cell_stats_counts_an_empty_truth_as_one_empty():
    rows = [row(1, "ET", 0.0, 373.13, gt_empty=True), row(1, "ET", 0.0, 373.13, pred_empty=True)]
    assert make_table.cell_stats(rows)[4:] == ("0", "2")


def write_run(run, val_cases, rows):
    run.mkdir()
    rec = dict.fromkeys(RUN_KEYS)
    rec.update(run_id="r", status="ok", config_hash="abc", git_commit="deadbeef", seed=42,
               val_cases=val_cases)
    (run / "run.json").write_text(json.dumps(rec))
    with open(run / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def full_rows(cases):
    return [row(pid, region, 0.5, 3.0, case=c) for c in cases for pid in range(1, 16)
            for region in ("WT", "TC", "ET")]


# covers: AC-12
def test_table_has_15_pattern_rows_and_states_real_n(tmp_path):
    cases = ["BraTS-GLI-00005-000", "BraTS-GLI-00006-000"]
    write_run(tmp_path / "run", cases, full_rows(cases))
    tex = make_table.make(tmp_path / "run")
    body = [ln for ln in tex.splitlines() if ln.split(" & ")[0].isdigit()]
    assert [ln.split(" & ")[0] for ln in body] == [str(i) for i in range(1, 16)]
    assert body[0].startswith("1 & T1+T1ce+T2+FLAIR & 0.500 & 0.500 & 3.0 & 3.0 & 0 & 0 & 0.500")
    assert len(body[0].split(" & ")) == 2 + 6 * 3
    assert r"\begin{tabular}{rl" + "rrrrrr" * 3 + "}" in tex
    assert r"\multicolumn{6}{c}{WT}" in tex and r"\cmidrule(lr){15-20}" in tex
    assert "Dice & Dice & HD95 & HD95 & Both & One & Dice" in tex
    assert "mean & med. & mean & med. & empty & empty & mean" in tex
    assert "n = 2 val subjects" in tex and "Not a result." in tex
    assert "config abc, git deadbeef, seed 42" in tex


# covers: AC-12
def test_table_refuses_a_wrong_row_count(tmp_path):
    cases = ["BraTS-GLI-00005-000", "BraTS-GLI-00006-000"]
    write_run(tmp_path / "run", cases, full_rows(cases)[:-1])
    with pytest.raises(SystemExit, match="89 rows in metrics.csv, expected 90"):
        make_table.make(tmp_path / "run")


def test_table_refuses_a_missing_metrics_file(tmp_path):
    write_run(tmp_path / "run", ["BraTS-GLI-00005-000"], [])
    (tmp_path / "run" / "metrics.csv").unlink()
    with pytest.raises(SystemExit, match="missing"):
        make_table.make(tmp_path / "run")
