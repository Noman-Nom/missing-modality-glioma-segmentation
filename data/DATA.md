# Datasets

| Field | MSD Task01_BrainTumour | BraTS 2023 GLI (training) | BraTS 2023 GLI (validation) |
|---|---|---|---|
| Source | https://medicaldecathlon.com/dataaws/ | https://www.synapse.org/Synapse:syn51514105 | same Synapse folder |
| Licence / terms | to confirm | to confirm | to confirm |
| Version / date downloaded | not recorded | download in progress/unknown | download in progress/unknown |
| Local path | not recorded (`MSD_DATA_DIR` in `.env`) | `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` in repo root (git-ignored) | `ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/` in repo root (git-ignored) |
| Hash (sha256) | not computed | not computed | not computed |
| Cases | to count | 1251 cases, 1133 subjects (audited 2026-10-20) | 219 cases, 199 subjects (audited 2026-10-20) |
| Modality order | to verify from files (do not assume) | by file name suffix, never by position: `-t1n`, `-t1c`, `-t2w`, `-t2f` | same |
| Label values | to verify from spec and files | 0 background, 1 NCR, 2 ED, 3 ET (values 0-3 only, confirmed in all 1251 files) | no masks |
| Masks present | training only expected; verify | yes, 1251 of 1251 (`-seg.nii.gz`) | **no** (0 of 219) |
| Overlap / provenance | may share cases with BraTS | includes older BraTS cases (see `BraTS2023_2017_GLI_Mapping.xlsx`) | to check |
| Ethics approval | unknown, ask supervisor | unknown | unknown |

Also downloaded: `BraTS2023_2017_GLI_Mapping.xlsx` (maps 2023 IDs to 2017 IDs; use it for the overlap check).

Raw data is never committed. Only the audit manifest and split files (subject IDs) are committed.

## Audit results (BraTS 2023 GLI, experiment 0000, run 2026-10-20)

Source: `experiments/0000-data-audit/`, output `results/0000-data-audit/cases.csv` (not committed).

- **Geometry.** All 1470 volumes are 240x240x155, 1 mm isotropic, LPS orientation. No missing, unreadable or NaN files. Seg files carry only a qform (`sform_code` 0); image files carry an sform.
- **Data type.** About 15% of image files are float32, the rest int16. Load as float32.
- **Intensities are not comparable between cases.** The 99th percentile per volume has median 570-1470 but reaches 1e5-3.5e6 in 15-20 cases per modality. Normalise per volume inside the brain mask with percentile clipping. 10 training and 2 validation cases have negative values.
- **Brain masks.** T1n, T1c, T2w agree (Dice >= 0.935). T2-FLAIR differs from T1n in 4 cases (Dice < 0.9): `BraTS-GLI-01435-000` (0.66), `01416-000` (0.80), `01432-000` (0.82), `01431-000` (0.88). Not yet viewed by eye; flagged in `data/splits/splits.csv` (`flair_mask_flag`), not excluded.
- **Subjects.** 118 training and 20 validation subjects have two cases (timepoints 000 and 001). Split by subject.
- **Label sets in training.** `0,1,2,3`: 1180; `0,2,3`: 37; `0,1,2`: 27; `0,2`: 6; `0,1,3`: 1. Empty ET: 33 cases (2.6%); empty TC: 6; empty WT: 0. Dice for empty regions needs one fixed rule in `common/metrics.py`.
- **Tumour size (mL).** WT median 89 (range 2.8-362); TC median 28; ET median 17. 23 cases have fewer than 500 ET voxels.
- **Overlap.** Training and validation share no subject IDs and no case IDs. Overlap with MSD Task01 not checked (MSD not downloaded; `BraTS2023_2017_GLI_Mapping.xlsx` not found in the repo).
- **Quirk.** `BraTS-GLI-00011-000` and `BraTS-GLI-00014-000` contain a stray folder named `...-t1n.nii` / `...-t2w.nii` next to the normal `.nii.gz`. Load only `*.nii.gz`.
- **Not done.** sha256, download date, licence and ethics status are still to be filled in by the human.

## Splits (experiment 0001)

`data/splits/splits.csv` (committed, IDs only). Seed 42, subject level, stratified by subject WT-volume tertile x any-empty-ET. The 219 official validation cases have no labels and are not used for scores.

| fold | subjects | cases | WT median (mL) | empty-ET cases |
|---|---|---|---|---|
| train | 791 | 867 | 87.8 | 22 |
| val | 114 | 129 | 97.1 | 4 |
| test | 228 | 255 | 90.4 | 7 |

Regenerate: `python3 experiments/0001-splits/make_splits.py`. Check: `python3 experiments/0001-splits/check_splits.py`.
