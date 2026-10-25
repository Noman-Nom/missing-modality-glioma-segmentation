"""Checks for the split: no subject in two folds, all cases covered once, same seed -> same file,
strata roughly balanced. The toy case with a known answer lives in tests/test_splits.py.

    uv run python experiments/0001-splits/check_splits.py
    uv run python experiments/0001-splits/check_splits.py --compare <tmp>/splits.csv
--compare checks a regenerated split against the committed one (spec 0002, AC-2): same cases,
same fold per case. It never writes data/splits/splits.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from make_splits import make

CASES = "results/0000-data-audit/cases.csv"
SPLITS = "data/splits/splits.csv"
# labelled BraTS 2023 GLI training cases, recorded in data/DATA.md (Splits)
N_LABELLED_CASES = 1251


def read_split(path):
    return pd.read_csv(path, dtype={"subject": str}).sort_values("case").reset_index(drop=True)


def compare(path):
    """Regenerated split at `path` vs the committed one: identical cases and fold per case."""
    new, saved = read_split(path), read_split(SPLITS)
    assert new.case.tolist() == saved.case.tolist(), \
        f"case lists differ: {len(new)} regenerated vs {len(saved)} committed"
    diff = new[new.fold != saved.fold]
    assert diff.empty, f"{len(diff)} cases in a different fold, first: {diff.case.tolist()[:5]}"
    print(f"OK {path} matches {SPLITS}: {len(new)} cases, same fold per case")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compare", help="regenerated splits CSV to compare with the committed one")
    a = ap.parse_args(argv)
    if a.compare:
        compare(a.compare)
        return

    a = make(CASES)
    b = make(CASES)
    assert a.equals(b), "not deterministic"
    saved = read_split(SPLITS)
    assert saved.shape == a.shape and (saved.case == a.case).all() and (saved.fold == a.fold).all(), \
        "data/splits/splits.csv is stale; rerun make_splits.py"
    assert a.case.is_unique and len(a) == N_LABELLED_CASES, len(a)
    assert (a.groupby("subject").fold.nunique() == 1).all(), "LEAKAGE: subject in two folds"
    sets = {f: set(a[a.fold == f].subject) for f in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    n = a.subject.nunique()
    for f, frac in (("train", .7), ("val", .1), ("test", .2)):
        assert abs(len(sets[f]) / n - frac) < 0.02, (f, len(sets[f]) / n)
    med = a.groupby("fold").ml_WT.median()
    assert med.max() - med.min() < 15, med.to_dict()
    print("OK", {f: len(v) for f, v in sets.items()}, "subjects; WT median", med.round(1).to_dict())


if __name__ == "__main__":
    main()
