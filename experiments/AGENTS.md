# experiments/

## Overview
Code for all experiments: data audit, splits, training, missing-modality evaluation. Nothing here yet (milestone 2 starts it).

## Stack
Python 3, PyTorch, MONAI, NiBabel, NumPy, SciPy. 4 GB GPU: patch training, batch size 1, mixed precision. Versions are pinned in the lockfile once it exists.

## Key files (planned)
- `common/data.py` loaders, per-dataset label and channel mapping
- `common/missing.py` missing-modality patterns and how a missing channel is represented
- `common/metrics.py` Dice, sensitivity, precision, HD95 (empty-region rules defined once)
- `common/seed.py` one function that sets all seeds
- `NNNN-name/config.yaml` one config per experiment

## Commands
Not created yet. Each experiment will document: run, check, regenerate table.

## Conventions
- One config file per experiment; seeds set in one function.
- Outputs go only to `results/<experiment>/`.
- Each run prints the config hash and the git commit.
- Notebooks may explore, but results come from scripts.
- Missing-modality patterns are defined before running and evaluated on the same subjects for every method.

## Import direction
Experiments import from `common/`, never from each other.

## Naming
`NNNN-short-name` (E0 to E7 map to the numbers), snake_case modules, region names exactly WT, TC, ET, modality names T1, T1ce, T2, FLAIR.

## Gotchas
- Label values and channel order must come from the audit, not memory. MSD and BraTS differ.
- GPU nondeterminism: record settings used for determinism checks.

## Agent skills / related specs
See `docs/specs/thesis/` and `.claude/skills/`.
