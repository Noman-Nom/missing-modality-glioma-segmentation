"""Spec 0004, thin thread: the case lists and the sweep in common/evaluation.py (AC-1, AC-3,
AC-4). The sweep runs on a small cache with a stub model, no GPU and no real data."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from common import evaluation
from common.cache import write_case
from common.data import MODALITIES, REGIONS, fold_cases
from common.evaluation import (
    METRIC_COLUMNS,
    GuardError,
    clean_metrics,
    eval_cases,
    metrics_writer,
    sweep,
)
from common.missing import PATTERNS
from common.runlog import load_config

REPO = Path(__file__).resolve().parents[1]
E1_CONFIG = REPO / "experiments" / "0003-baseline" / "config.toml"
EVAL_CONFIG = REPO / "experiments" / "0004-missing-eval" / "config.toml"
SPLITS = REPO / "data" / "splits" / "splits.csv"
VAL = ["BraTS-GLI-00005-000", "BraTS-GLI-00006-000"]
TRAIN = ["BraTS-GLI-00001-000", "BraTS-GLI-00001-001"]


class Everywhere(torch.nn.Module):
    """Predicts every region everywhere."""

    def forward(self, x):
        return torch.full((x.shape[0], 3, *x.shape[2:]), 5.0)


@pytest.fixture
def cached(tmp_path):
    """Two synthetic cases in a real cache, and a minimal config for them."""
    cdir = tmp_path / "cache"
    cdir.mkdir()
    cfg = {"data": {"cache_dir": str(cdir)}, "train": {"amp": False},
           "infer": {"roi": [8, 8, 8], "sw_batch_size": 1, "overlap": 0.5, "threshold": 0.5}}
    rng = np.random.default_rng(0)
    label = np.zeros((8, 8, 8), np.uint8)
    label[2:5, 2:5, 2:5] = 2
    label[3, 3, 3] = 3
    for c in VAL:
        write_case(cdir, c, rng.normal(size=(4, 8, 8, 8)).astype(np.float32), label,
                   (1.0, 1.0, 1.0), ((0, 8), (0, 8), (0, 8)))
    return cfg, cdir


def test_eval_config_is_the_spec_table():
    assert load_config(EVAL_CONFIG) == {"eval": {
        "seed": 42, "test_timepoints": [0], "bootstrap_resamples": 10000, "alpha": 0.05}}


# covers: AC-1. Case IDs only: splits.csv holds no image.
def test_eval_cases_on_the_committed_splits():
    cfg = load_config(E1_CONFIG)
    rec = {"train_cases": fold_cases(SPLITS, "train", [0, 1], True),
           "val_cases": fold_cases(SPLITS, "val", [0], True)}
    eval_cfg = load_config(EVAL_CONFIG)["eval"]
    val = eval_cases(cfg, eval_cfg, rec, "val")
    test = eval_cases(cfg, eval_cfg, rec, "test")
    assert (len(val), len(test)) == (114, 228)
    assert test == sorted(test)
    assert len({c[:15] for c in test}) == 228  # one case per subject
    assert all(c.endswith("-000") for c in test)


# covers: AC-1
def test_val_cases_come_from_the_run_record_sorted():
    rec = {"train_cases": TRAIN, "val_cases": VAL[::-1]}
    assert eval_cases({}, {}, rec, "val") == VAL


# covers: AC-1
def test_val_case_sharing_a_subject_with_train_is_refused():
    rec = {"train_cases": TRAIN + ["BraTS-GLI-00006-001"], "val_cases": VAL}
    with pytest.raises(GuardError, match="00006"):
        eval_cases({}, {}, rec, "val")


# covers: AC-1
@pytest.mark.parametrize("field", ["train_cases", "val_cases"])
def test_test_case_sharing_a_subject_with_train_or_val_is_refused(tmp_path, field):
    splits = tmp_path / "splits.csv"
    splits.write_text("case,timepoint,fold,flair_mask_flag\nBraTS-GLI-00009-000,0,test,False\n")
    cfg = {"data": {"splits_csv": str(splits), "exclude_flair_flag": True}}
    rec = {"train_cases": TRAIN, "val_cases": VAL}
    rec[field] = rec[field] + ["BraTS-GLI-00009-001"]
    with pytest.raises(GuardError, match="00009"):
        eval_cases(cfg, {"test_timepoints": [0]}, rec, "test")


# covers: AC-3
def test_sweep_order_rows_and_values(cached):
    cfg, _ = cached
    got = []
    sweep(Everywhere(), VAL[::-1], cfg, torch.device("cpu"), got.append)
    assert [(r[0]["case"], r[0]["pattern_id"]) for r in got] == [
        (c, p) for c in VAL for p in range(1, 16)]
    assert all([x["region"] for x in rows] == list(REGIONS) for rows in got)
    wt, tc = got[0][:2]  # WT is the 27 voxel cube, TC only its 1 label 3 voxel; all 512 predicted
    assert wt["pattern_name"] == PATTERNS[1].name and wt["subject"] == "00005"
    assert wt["dice"] == pytest.approx(2 * 27 / (512 + 27))
    assert tc["dice"] == pytest.approx(2 * 1 / (512 + 1))
    assert wt["sensitivity"] == 1.0 and wt["precision"] == pytest.approx(27 / 512)
    assert wt["gt_empty"] is False and wt["pred_empty"] is False


# covers: AC-3
def test_sweep_raises_on_nan_before_the_pair_is_written(cached, monkeypatch):
    cfg, _ = cached
    real = evaluation.region_metrics
    calls = []

    def nan_on_fourth_pair(*a, **k):
        calls.append(1)
        return real(*a, **k) | ({"hd95_mm": math.nan} if len(calls) == 10 else {})

    monkeypatch.setattr(evaluation, "region_metrics", nan_on_fourth_pair)
    got = []
    with pytest.raises(ValueError, match="NaN metric in .* pattern 4 WT"):
        sweep(Everywhere(), VAL, cfg, torch.device("cpu"), got.append)
    assert len(got) == 3


# covers: AC-4
def test_image_hook_runs_after_apply_pattern(cached):
    cfg, _ = cached
    seen = {}

    def hook(image, pattern_id):
        seen[pattern_id] = np.array([np.abs(image[i]).sum() > 0 for i in range(len(MODALITIES))])
        return image

    sweep(Everywhere(), VAL[:1], cfg, torch.device("cpu"), lambda rows: None, image_hook=hook)
    assert sorted(seen) == list(range(1, 16))
    for p, kept in seen.items():
        assert kept.tolist() == [m in PATTERNS[p].present for m in MODALITIES], p


# covers: AC-4
def test_image_hook_output_is_what_the_model_sees(cached):
    cfg, _ = cached
    sizes = []

    class Spy(Everywhere):
        def forward(self, x):
            sizes.append(float(x.max()))
            return super().forward(x)

    sweep(Spy(), VAL[:1], cfg, torch.device("cpu"), lambda rows: None, patterns=[1],
          image_hook=lambda image, p: image * 0 + 7)
    assert sizes and set(sizes) == {7.0}


# covers: AC-3. Only pairs not in done are scored, and a finished case is not opened.
def test_sweep_skips_done_pairs(cached):
    cfg, cdir = cached
    (cdir / f"{VAL[0]}.npz").unlink()
    done = {(VAL[0], p) for p in range(1, 16)} | {(VAL[1], 1)}
    got = []
    sweep(Everywhere(), VAL, cfg, torch.device("cpu"), got.append, done=done)
    assert [r[0]["pattern_id"] for r in got] == list(range(2, 16))


# covers: AC-3. repr keeps every digit, NaN is an empty field, lines end in \n only.
def test_metrics_writer_format(tmp_path):
    path = tmp_path / "metrics.csv"
    row = {"case": "c", "subject": "s", "pattern_id": 1, "pattern_name": "n", "region": "WT",
           "dice": 1 / 3, "hd95_mm": np.float64(2.5), "sensitivity": math.nan, "precision": 0.5,
           "gt_empty": True, "pred_empty": False}
    with open(path, "w", newline="") as f:
        metrics_writer(f, "ev")([row])
    with open(path, newline="") as f:
        (got,) = list(csv.DictReader(f))
    assert list(got) == METRIC_COLUMNS
    assert got["eval_id"] == "ev" and float(got["dice"]) == 1 / 3 and got["hd95_mm"] == "2.5"
    assert got["sensitivity"] == "" and got["gt_empty"] == "True"
    assert b"\r" not in path.read_bytes()


# covers: AC-5
def write_pairs(path, pairs, extra=b""):
    with open(path, "w", newline="") as f:
        write = metrics_writer(f, "e")
        for case, p in pairs:
            write([{"case": case, "subject": "s", "pattern_id": p, "pattern_name": "n",
                    "region": g, "dice": 0.5, "hd95_mm": 1.0, "sensitivity": 0.5, "precision": 0.5,
                    "gt_empty": False, "pred_empty": False} for g in REGIONS])
    with open(path, "ab") as f:
        f.write(extra)
    return path.read_bytes()


def test_clean_metrics_keeps_a_clean_file(tmp_path):
    path = tmp_path / "m.csv"
    before = write_pairs(path, [("a", 1), ("a", 2), ("b", 1)])
    assert clean_metrics(path) == {("a", 1), ("a", 2), ("b", 1)}
    assert path.read_bytes() == before


def test_clean_metrics_cuts_a_truncated_line(tmp_path):
    path = tmp_path / "m.csv"
    before = write_pairs(path, [("a", 1)])
    write_pairs(path, [("a", 1)], extra=b"e,a,s,2,n,W")
    assert clean_metrics(path) == {("a", 1)}
    assert path.read_bytes() == before


@pytest.mark.parametrize("rows", [1, 2])
def test_clean_metrics_cuts_part_of_a_pair(tmp_path, rows):
    path = tmp_path / "m.csv"
    before = write_pairs(path, [("a", 1)])
    whole = write_pairs(path, [("a", 1), ("a", 2)])
    path.write_bytes(b"".join(whole.splitlines(keepends=True)[:4 + rows]))
    assert clean_metrics(path) == {("a", 1)}
    assert path.read_bytes() == before


def test_clean_metrics_handles_no_rows(tmp_path):
    path = tmp_path / "m.csv"
    assert clean_metrics(path) == set() and not path.exists()
    header = write_pairs(path, [])
    assert clean_metrics(path) == set() and path.read_bytes() == header
    path.write_bytes(header[:9])  # killed inside the header: the writer starts it again
    assert clean_metrics(path) == set() and path.read_bytes() == b""


def test_clean_metrics_refuses_a_foreign_file(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(GuardError, match="header"):
        clean_metrics(path)
    assert path.read_text() == "a,b\n1,2\n"


def test_sweep_gives_progress_after_each_written_pair(cached):
    cfg, _ = cached
    calls = []
    sweep(Everywhere(), VAL[:1], cfg, torch.device("cpu"),
          lambda rows: calls.append("write"), patterns=[1, 2],
          progress=lambda s: calls.append(s["predict_seconds"] > 0))
    assert calls == ["write", True, "write", True]
