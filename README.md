# Segmenting Gliomas When Sequences Are Missing: A Controlled Comparison of Training Strategies for Incomplete Multimodal MRI

Master's thesis project. University, program, student and supervisor: not recorded yet.

We build a reproducible brain-tumor segmentation pipeline (3D U-Net, BraTS 2023 GLI, MSD Task01) under a 4 GB GPU budget. We measure how performance drops when MRI modalities (T1, T1ce, T2, FLAIR) are missing and compare strategies to stay robust (modality dropout, presence indicators). Label efficiency and self-supervised pretraining are optional extensions. This is not a clinical tool.

## Status
Repo bootstrapped. Data audit, experiments and writing not started. See `context/progress-tracker.md`.

## Setup
    git config core.hooksPath .githooks     # enable the pre-commit hook
    cp .env.example .env                    # then fill in local data paths

Python environment: created in milestone 2.

## Reproduce results
Not available yet. When experiments exist, one command per experiment will be listed here and in `experiments/AGENTS.md`.

## Repo map
- `context/` overview, progress tracker, original draft
- `docs/` scope, specs, ADRs, literature notes, reviews
- `data/` dataset notes only; MRI is never committed
- `experiments/` experiment code (not started)
- `writing/` thesis text and `refs.bib` (not started)
- `app/` reserved, no demo planned
- `AGENTS.md` rules for AI tools; `.claude/` workflow skills
