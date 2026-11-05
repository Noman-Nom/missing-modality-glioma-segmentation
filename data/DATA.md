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

## Private copy on Kaggle (for training, ADR 0002)

- **Where.** Private Kaggle dataset `muhammadnomanned/brats-gli-train`, created 2026-11-01. Never make it public.
- **What.** One file, `brats-gli-train.tar.bin`, 13,394,974,720 bytes: an uncompressed tar of `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` (1251 case folders, 6255 `.nii.gz` files). The notebook unpacks it with `tar -xf` (`experiments/0003-baseline/KAGGLE.md`).
- **Why a tar with an odd name.** Kaggle unpacks `.zip` files and every `.nii.gz` inside them on upload. It writes plain `.nii` files, and where the gzip header carries an old file name it makes a folder, for example `BraTS-GLI-00011-000-t1c.nii/00000055_brain_t1ce.nii`. Our loader reads only `*.nii.gz`, so it would find nothing. With the two stray folders present, processing failed outright (tested 2026-11-01 with 3 cases: with the strays it failed, without them it passed). A `.tar.bin` is stored as one unopened file (tested with 1 case: same byte count on Kaggle).
- **Left out.** The two stray folders `BraTS-GLI-00011-000/BraTS-GLI-00011-000-t1n.nii/` and `BraTS-GLI-00014-000/BraTS-GLI-00014-000-t2w.nii/`. They hold one raw `.nii` each and are not read by any script. The official zip from Synapse does not contain them (7507 entries, against 7511 in the local folder), so they were most likely made by an earlier local unpack.
- **Terms.** The BraTS 2023 terms above are still "to confirm". Confirm that they allow a private copy on Kaggle before the thesis is submitted.

## E1 model on Kaggle (for E2, ADR 0002)

- **Where.** Private Kaggle dataset `muhammadnomanned/brats-e1-run`. Uploaded, and `isPrivate` true in its Kaggle metadata (checked 2026-11-05). The steps are in `experiments/0004-missing-eval/KAGGLE.md`. Never make it public.
- **What.** One file, `e1-run.tar.bin`, 51,599,360 bytes: an uncompressed tar of `config.toml`, `run.json`, `metrics_val.csv` and `checkpoints/best.pt` from run `20261103-023817-f9b3afb6`. SHA256 of `best.pt`: `975f56a148af15c01714f397aa4f248f518adda11d55e0de8f8155517d521a58`.
- **Why a tar.** It keeps the `checkpoints/` folder, and stops Kaggle from touching the files on upload, the same reason as the BraTS tar above.
- **What it holds.** Model weights and case IDs with their scores, no images. The weights were trained on BraTS data, so the BraTS terms still to confirm above apply here too.

## Splits (experiment 0001)

`data/splits/splits.csv` (committed, IDs only). It holds all 1251 labelled training cases (`N_LABELLED_CASES` in `check_splits.py`). Seed 42, subject level, stratified by subject WT-volume tertile x any-empty-ET. The 219 official validation cases have no labels and are not used for scores.

| fold | subjects | cases | WT median (mL) | empty-ET cases |
|---|---|---|---|---|
| train | 791 | 867 | 87.8 | 22 |
| val | 114 | 129 | 97.1 | 4 |
| test | 228 | 255 | 90.4 | 7 |

Regenerate: `uv run python experiments/0001-splits/make_splits.py` (needs `results/0000-data-audit/cases.csv`; if it is missing, rerun the audit first). Check: `uv run python experiments/0001-splits/check_splits.py`. Reproduce check (spec 0002, AC-2), never overwrites the committed file: `uv run python experiments/0001-splits/make_splits.py --out <tmp>/splits.csv`, then `uv run python experiments/0001-splits/check_splits.py --compare <tmp>/splits.csv`.

Reproduced under the locked environment on 2026-10-25 (human run, pandas 3.1.0 from `uv run`): `check_splits.py --compare` printed `OK ... 1251 cases, same fold per case`. SHA256 of the committed `data/splits/splits.csv`: `c804b7543cf0ebda587a6da753fd6445b7222f9218efdee36253e4c176b282dc`.
