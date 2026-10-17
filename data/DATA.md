# Datasets

| Field | MSD Task01_BrainTumour | BraTS 2023 GLI (training) | BraTS 2023 GLI (validation) |
|---|---|---|---|
| Source | https://medicaldecathlon.com/dataaws/ | https://www.synapse.org/Synapse:syn51514105 | same Synapse folder |
| Licence / terms | to confirm | to confirm | to confirm |
| Version / date downloaded | not recorded | download in progress/unknown | download in progress/unknown |
| Local path | not recorded (`MSD_DATA_DIR` in `.env`) | not recorded (`BRATS_DATA_DIR` in `.env`) | not recorded |
| Hash (sha256) | not computed | not computed | not computed |
| Cases | to count | to count | to count |
| Modality order | to verify from files (do not assume) | to verify | to verify |
| Label values | to verify from spec and files | to verify | to verify |
| Masks present | training only expected; verify | expected; verify | **assume no public masks** until verified |
| Overlap / provenance | may share cases with BraTS | includes older BraTS cases (see `BraTS2023_2017_GLI_Mapping.xlsx`) | to check |
| Ethics approval | unknown, ask supervisor | unknown | unknown |

Also downloaded: `BraTS2023_2017_GLI_Mapping.xlsx` (maps 2023 IDs to 2017 IDs; use it for the overlap check).

Raw data is never committed. Only the audit manifest and split files (subject IDs) are committed.
