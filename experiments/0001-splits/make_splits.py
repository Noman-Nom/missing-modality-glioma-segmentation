"""E1 fixed subject-level split of the 1251 labelled BraTS 2023 GLI training cases.

Reads results/0000-data-audit/cases.csv. Writes data/splits/splits.csv (IDs only, safe to commit).
Fractions (subjects): train 70%, val 10%, test 20%. Seed 42.
Strata: WT-volume tertile of the subject (mean over its cases) x "any case without ET".
All cases of one subject go to the same split (subject = the first number in the case ID).
The 219 official validation cases have no labels and are NOT used here.
"""
import argparse
import numpy as np
import pandas as pd

SEED = 42
FRAC = {"train": 0.70, "val": 0.10, "test": 0.20}
# T2-FLAIR brain mask disagrees with T1n (Dice < 0.9) in the audit; flagged, not dropped
FLAIR_MASK_FLAG = 0.9


def make(cases_csv):
    d = pd.read_csv(cases_csv)
    d = d[d.split == "train"].copy()
    d["subject"] = d["subject"].astype(str).str.zfill(5)
    g = d.groupby("subject").agg(ml_WT=("ml_WT", "mean"), et_empty=("ET_empty", "any")).reset_index()
    g["tertile"] = pd.qcut(g.ml_WT, 3, labels=False)
    g["stratum"] = g.tertile.astype(str) + "_" + g.et_empty.astype(int).astype(str)
    rng = np.random.default_rng(SEED)
    assign = {}
    for _, grp in g.groupby("stratum"):
        subs = np.array(sorted(grp.subject))
        rng.shuffle(subs)
        n = len(subs)
        n_val = int(round(n * FRAC["val"]))
        n_test = int(round(n * FRAC["test"]))
        for i, s in enumerate(subs):
            assign[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    d["fold"] = d.subject.map(assign)
    d["wt_tertile"] = d.subject.map(g.set_index("subject").tertile)
    d["flair_mask_flag"] = d.brain_dice_t1n_vs_t2f < FLAIR_MASK_FLAG
    d["neg_values_flag"] = d.problems.fillna("").str.contains("negative")
    cols = ["case", "subject", "timepoint", "fold", "wt_tertile", "ml_WT", "ml_TC", "ml_ET",
            "ET_empty", "TC_empty", "flair_mask_flag", "neg_values_flag"]
    return d[cols].sort_values("case").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="results/0000-data-audit/cases.csv")
    ap.add_argument("--out", default="data/splits/splits.csv")
    a = ap.parse_args()
    s = make(a.cases)
    s.to_csv(a.out, index=False)
    print(f"seed={SEED} wrote {a.out}")
    print(s.groupby("fold").agg(cases=("case", "size"), subjects=("subject", "nunique"),
                                WT_median=("ml_WT", "median"), ET_empty=("ET_empty", "sum")).round(1))
