"""Turn results/0000-data-audit/cases.csv into summary.md and a few plain-text tables."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/0000-data-audit")
df = pd.read_csv(out / "cases.csv")
L = []
w = L.append


def dist(s):
    s = s.dropna()
    q = s.quantile([0, .05, .25, .5, .75, .95, 1]).round(2).tolist()
    return f"n={len(s)} mean={s.mean():.2f} sd={s.std():.2f} min/5/25/50/75/95/max={q}"


for split, g in df.groupby("split"):
    w(f"\n## {split}: {len(g)} cases, {g.subject.nunique()} subjects")
    w(f"- timepoints per case id: {g.timepoint.value_counts().sort_index().to_dict()}")
    spc = g.groupby("subject").size()
    w(f"- cases per subject: {spc.value_counts().sort_index().to_dict()}")
    w(f"- shapes: {g['shape'].value_counts().to_dict()}")
    w(f"- spacing: {g['spacing'].value_counts().to_dict()}")
    w(f"- orientation: {g['orient'].value_counts().to_dict()}")
    for m in ["t1n", "t1c", "t2w", "t2f"]:
        w(f"- {m} dtype: {g[f'{m}_dtype'].value_counts().to_dict()}")
    bad = g[g.problems.fillna("") != ""]
    w(f"- cases with problems: {len(bad)}")
    for _, r in bad.head(30).iterrows():
        w(f"  - {r.case}: {r.problems}")
    w("- brain voxels (nonzero) per modality: " + "; ".join(
        f"{m} median={g[f'{m}_brain_vox'].median():.0f}" for m in ["t1n", "t1c", "t2w", "t2f"]))
    w(f"- brain mask Dice t1n vs others (min): " + ", ".join(
        f"{m}={g[f'brain_dice_t1n_vs_{m}'].min():.3f}" for m in ["t1c", "t2w", "t2f"]))
    w("- intensity (per modality, brain voxels): median of p99 / sd of p99 across cases / min-max of case means")
    for m in ["t1n", "t1c", "t2w", "t2f"]:
        w(f"  - {m}: p99 median={g[f'{m}_p99'].median():.0f}, p99 sd={g[f'{m}_p99'].std():.0f}, "
          f"p99 range=[{g[f'{m}_p99'].min():.0f}, {g[f'{m}_p99'].max():.0f}], "
          f"mean range=[{g[f'{m}_mean'].min():.0f}, {g[f'{m}_mean'].max():.0f}]")
    if split == "train":
        w(f"- seg label sets: {g.seg_labels.value_counts().to_dict()}")
        for k in ["WT", "TC", "ET"]:
            w(f"- tumour volume {k} (mL): {dist(g[f'ml_{k}'])}")
        w(f"- empty ET: {int(g.ET_empty.sum())} ({g.ET_empty.mean()*100:.1f}%), "
          f"empty TC: {int(g.TC_empty.sum())}, empty WT: {int(g.WT_empty.sum())}")
        w(f"- ET voxels < 100: {(g.vox_ET.between(1, 99)).sum()}  < 500: {(g.vox_ET.between(1, 499)).sum()}")
        tot = g[["vox_NCR", "vox_ED", "vox_ET"]].sum()
        w(f"- label share of tumour voxels: NCR {tot.vox_NCR/tot.sum():.3f}, ED {tot.vox_ED/tot.sum():.3f}, ET {tot.vox_ET/tot.sum():.3f}")
        brain = g[["t1n_brain_vox"]].median().iloc[0]
        w(f"- WT as share of median brain volume: {g.ml_WT.median()*1000/brain*100:.1f}% (median case)")
        # volume tertiles for stratified splitting
        w(f"- WT volume tertile edges (mL): {g.ml_WT.quantile([1/3, 2/3]).round(1).tolist()}")
        # subjects with >1 timepoint
        multi = g.groupby("subject").size()
        w(f"- subjects with >1 case: {(multi > 1).sum()} (they MUST stay in one split)")

tr, va = df[df.split == "train"], df[df.split == "val"]
ov = set(tr.subject) & set(va.subject)
w(f"\n## overlap train vs val\n- shared subject ids: {len(ov)} {sorted(ov)[:10]}")
w(f"- shared case ids: {len(set(tr.case) & set(va.case))}")
(out / "summary.md").write_text("# E0 data audit summary\n" + "\n".join(L) + "\n")
print("\n".join(L))
