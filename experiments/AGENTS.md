# experiments/

## Overview
Code for all experiments: data audit, splits, training, missing-modality evaluation. Shared code is in `common/`; each experiment is an `NNNN-name/` folder.

## Stack
Python 3, PyTorch, MONAI, NiBabel, NumPy, SciPy, pandas. 4 GB GPU: patch training, batch size 1, mixed precision. Versions are pinned in `uv.lock` at the repo root (ADR 0001).

## Key files
- `common/data.py` loaders, per-dataset label and channel mapping; the one `BRATS_ID_RE` and `ALLOWED_LABELS` (import them, never redefine)
- `common/missing.py` missing-modality patterns and how a missing channel is represented
- `common/metrics.py` Dice, sensitivity, precision, HD95 (empty-region rules defined once)
- `common/seed.py` one function that sets all seeds
- `common/model.py` the compact 3D U-Net and sliding window prediction
- `common/runlog.py` config loading, config hash, git and version info, `run.json`
- `NNNN-name/config.toml` one config per experiment

## Commands
Run from the repo root. Each script's docstring holds its exact run line.
- Tracer (0002): `uv run python experiments/0002-tracer-bullet/train.py [--quick]`, then `evaluate.py --run <dir>`, `make_table.py --run <dir>`, `check_tracer.py <dir>`.
- Checks: `check_audit.py`, `check_splits.py`, `check_tracer.py` in each experiment folder.
- Split reproduce check: `make_splits.py --out <tmp>/splits.csv`, then `check_splits.py --compare <tmp>/splits.csv`. Never regenerate over `data/splits/splits.csv`.
- Tests import `common` because `pyproject.toml` puts `experiments/` on the pytest path.

## Conventions
- One config file per experiment; seeds set in one function.
- Outputs go only to `results/<experiment>/`. Generated LaTeX tables go to `writing/tables/`.
- Each run prints the config hash and the git commit.
- Notebooks may explore, but results come from scripts.
- Scripts run nothing on import: work goes in `main()` under `if __name__ == "__main__":` (tested in `tests/test_experiment_scripts.py`).
- Missing-modality patterns are defined before running and evaluated on the same subjects for every method.

## Import direction
Experiments import from `common/`, never from each other.

## Naming
`NNNN-short-name`, snake_case modules, region names exactly WT, TC, ET, modality names T1, T1ce, T2, FLAIR.

Folder numbers are build order. E numbers (E0 to E7, defined in `context/project-overview.md`) are thesis labels. This table is the one mapping (spec 0002):

| E number | Thesis label | Folder |
|---|---|---|
| E0 | Data inspection | `0000-data-audit` |
| E1 to E7 | Baseline, missing modality evaluation, strategies, extensions | folder added here when built |
| none | Infrastructure: subject level split | `0001-splits` |
| none | Infrastructure: tracer bullet | `0002-tracer-bullet` |

## Gotchas
- Label values and channel order must come from the audit, not memory. MSD and BraTS differ.
- GPU nondeterminism: record settings used for determinism checks.

## Agent skills / related specs
See `docs/specs/thesis/` and `.claude/skills/`.
