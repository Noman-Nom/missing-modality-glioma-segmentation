"""Spec 0002, AC-11: the subject level split, with no data and no results/ folder."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def load_check_splits():
    path = REPO / "experiments" / "0001-splits" / "check_splits.py"
    spec = importlib.util.spec_from_file_location("check_splits", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_splits = load_check_splits()


def toy_split():
    """4 cases in the splits.csv format, subject IDs with leading zeros."""
    return pd.DataFrame({
        "case": [f"BraTS-GLI-0000{i}-000" for i in range(4)],
        "subject": [f"0000{i}" for i in range(4)],
        "fold": ["train", "train", "val", "test"]})


@pytest.fixture
def committed(tmp_path, monkeypatch):
    """A toy committed split in tmp_path, standing in for data/splits/splits.csv."""
    path = tmp_path / "committed.csv"
    toy_split().to_csv(path, index=False)
    monkeypatch.setattr(check_splits, "SPLITS", str(path))
    return path


def test_n_labelled_cases_matches_the_number_in_this_file():
    assert check_splits.N_LABELLED_CASES == N_LABELLED_CASES


# covers: AC-2 (--compare passes on the same cases and folds, whatever the row order)
def test_compare_accepts_the_same_split_in_another_row_order(committed, tmp_path, capsys):
    new = tmp_path / "new.csv"
    toy_split().iloc[::-1].to_csv(new, index=False)
    check_splits.main(["--compare", str(new)])
    assert "4 cases, same fold per case" in capsys.readouterr().out


# covers: AC-2 (a case fewer fails, with both counts)
def test_compare_rejects_a_missing_case(committed, tmp_path):
    new = tmp_path / "new.csv"
    toy_split().iloc[:3].to_csv(new, index=False)
    with pytest.raises(AssertionError, match="3 regenerated vs 4 committed"):
        check_splits.compare(new)


# covers: AC-2 (one case in another fold fails, naming the case)
def test_compare_rejects_a_case_in_another_fold(committed, tmp_path):
    new = tmp_path / "new.csv"
    s = toy_split()
    s.loc[3, "fold"] = "val"
    s.to_csv(new, index=False)
    with pytest.raises(AssertionError, match=r"1 cases in a different fold, first: \['BraTS-GLI-00003-000'\]"):
        check_splits.compare(new)


# covers: AC-2 (the committed split is never written)
def test_compare_leaves_the_committed_file_untouched(committed, tmp_path):
    before = committed.read_bytes()
    new = tmp_path / "new.csv"
    toy_split().assign(fold="test").to_csv(new, index=False)
    with pytest.raises(AssertionError):
        check_splits.compare(new)
    assert committed.read_bytes() == before


def test_read_split_keeps_leading_zeros_in_subject(committed):
    assert check_splits.read_split(committed).subject.tolist() == ["00000", "00001", "00002", "00003"]


# covers: AC-10 (N_LABELLED_CASES guard): a split one case short fails with its length
def test_check_fails_with_the_length_when_a_case_is_missing(tmp_path, monkeypatch):
    short = tmp_path / "short.csv"
    pd.read_csv(SPLITS, dtype={"subject": str}).iloc[:-1].to_csv(short, index=False)
    monkeypatch.setattr(check_splits, "SPLITS", str(short))
    monkeypatch.setattr(check_splits, "make", lambda _: check_splits.read_split(short))
    with pytest.raises(AssertionError, match=f"^{N_LABELLED_CASES - 1}$"):
        check_splits.main([])
