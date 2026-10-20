"""Checks for the split: no subject in two folds, all cases covered once, same seed -> same file,
strata roughly balanced, tiny toy case with a known answer."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from make_splits import make

CASES = "results/0000-data-audit/cases.csv"
SPLITS = "data/splits/splits.csv"


def main():
    # toy: 2 subjects with 2 cases each must stay together
    toy = pd.DataFrame({
        "split": "train", "case": [f"BraTS-GLI-0000{i}-00{t}" for i in range(6) for t in (0, 1)],
        "subject": [f"0000{i}" for i in range(6) for _ in (0, 1)],
        "timepoint": [t for _ in range(6) for t in (0, 1)],
        "ml_WT": np.arange(12) * 10.0 + 1, "ml_TC": 1.0, "ml_ET": 1.0,
        "ET_empty": False, "TC_empty": False, "brain_dice_t1n_vs_t2f": 1.0, "problems": ""})
    import tempfile, os
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "c.csv"); toy.to_csv(p, index=False)
        s = make(p)
        assert (s.groupby("subject").fold.nunique() == 1).all(), "toy: subject split across folds"

    a = make(CASES)
    b = make(CASES)
    assert a.equals(b), "not deterministic"
    saved = pd.read_csv(SPLITS, dtype={"subject": str})
    assert saved.drop(columns=[]).shape == a.shape and (saved.case == a.case).all() and (saved.fold == a.fold).all(), \
        "data/splits/splits.csv is stale; rerun make_splits.py"
    assert a.case.is_unique and len(a) == 1251, len(a)
    assert (a.groupby("subject").fold.nunique() == 1).all(), "LEAKAGE: subject in two folds"
    sets = {f: set(a[a.fold == f].subject) for f in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    n = a.subject.nunique()
    for f, frac in (("train", .7), ("val", .1), ("test", .2)):
        assert abs(len(sets[f]) / n - frac) < 0.02, (f, len(sets[f]) / n)
    med = a.groupby("fold").ml_WT.median()
    assert med.max() - med.min() < 15, med.to_dict()
    print("OK", {f: len(v) for f, v in sets.items()}, "subjects; WT median", med.round(1).to_dict())


main()
