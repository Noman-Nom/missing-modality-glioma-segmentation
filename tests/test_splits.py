"""Spec 0002, AC-11: the subject level split, with no data and no results/ folder."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "data" / "splits" / "splits.csv"
# N_LABELLED_CASES in experiments/0001-splits/check_splits.py
N_LABELLED_CASES = 1251


def load_make_splits():
    path = REPO / "experiments" / "0001-splits" / "make_splits.py"
    spec = importlib.util.spec_from_file_location("make_splits", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make = load_make_splits().make


def toy_cases(path):
    """6 subjects with 2 timepoints each, in the cases.csv format of the audit."""
    toy = pd.DataFrame({
        "split": "train", "case": [f"BraTS-GLI-0000{i}-00{t}" for i in range(6) for t in (0, 1)],
        "subject": [f"0000{i}" for i in range(6) for _ in (0, 1)],
        "timepoint": [t for _ in range(6) for t in (0, 1)],
        "ml_WT": np.arange(12) * 10.0 + 1, "ml_TC": 1.0, "ml_ET": 1.0,
        "ET_empty": False, "TC_empty": False, "brain_dice_t1n_vs_t2f": 1.0, "problems": ""})
    toy.to_csv(path, index=False)
    return path


# covers: AC-11 (a)
def test_toy_keeps_subjects_together_and_is_deterministic(tmp_path):
    cases = toy_cases(tmp_path / "cases.csv")
    a, b = make(cases), make(cases)
    assert a.equals(b)
    assert len(a) == 12 and (a.groupby("subject").fold.nunique() == 1).all()


# covers: AC-11 (b)
def test_committed_split_has_no_leakage():
    s = pd.read_csv(SPLITS, dtype={"subject": str})
    assert s.case.is_unique and len(s) == N_LABELLED_CASES
    assert (s.groupby("subject").fold.nunique() == 1).all(), "a subject is in two folds"
    sets = {f: set(s[s.fold == f].subject) for f in ("train", "val", "test")}
    assert all(sets.values())
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
