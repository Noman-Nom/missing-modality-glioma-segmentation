# What's built

**Last updated:** 2026-10-21, milestone 2 step 2 (environment smoke test)
**Last full check:** only git and hook checks. Not run: CI, any experiment (none exist), citation verification.

| # | Feature | Status | What it gives you |
|---|---|---|---|
| A | Repo and AI setup | done | Rules, skills, docs layout, pre-commit hook |
| B | Proposal | built, not supervisor-reviewed | `writing/proposal/proposal.pdf`, 7 pages |
| C | Data audit (E0) | done, run on all 1470 cases | `experiments/0000-data-audit/` writes `results/0000-data-audit/cases.csv` and `summary.md` |
| D | Subject-level split (E1) | done, check passes | `data/splits/splits.csv`, `experiments/0001-splits/` |
| E | Python environment and smoke test | done, 3 fast + 1 GPU test pass | `pyproject.toml`, `uv.lock`, `tests/test_env_smoke.py` |

## A. Repo and AI setup
Any AI chat reads `AGENTS.md` and starts with the same rules. Enable the hook with `git config core.hooksPath .githooks`.

## B. Proposal
Build: `cd writing/proposal && latexmk -xelatex proposal.tex`. Citations are unverified.

## C. Data audit (E0)
Run: `python3 experiments/0000-data-audit/audit.py --train <train dir> --val <val dir>` then `python3 experiments/0000-data-audit/summarize.py`. Check: `python3 experiments/0000-data-audit/check_audit.py <train dir>`. Uses numpy only (own NIfTI reader). Full run done 2026-10-20.

## D. Splits (E1)
`python3 experiments/0001-splits/make_splits.py` writes `data/splits/splits.csv` (train 791 / val 114 / test 228 subjects, seed 42). Check: `python3 experiments/0001-splits/check_splits.py` (no subject in two folds, deterministic, toy case).

## E. Environment and smoke test (ADR 0001)
Resolved versions (2026-10-20): Python 3.12.3, torch 2.14.1+cu130 (arch list has `sm_75`), monai 1.6.1, nibabel 5.4.2, scipy 1.18.1 (added 2026-10-21), pytest 9.1.1, ruff 0.17.1, uv 0.13.2.
Tests: `uv run pytest tests/test_env_smoke.py -v` (fast) and `uv run pytest tests/test_env_smoke.py -m slow -v -s` (GPU, human runs it). Default runs skip `slow`.
Run on 2026-10-21 (human, Quadro T2000 Max-Q):
- AC-3 passed: MONAI 3D U-Net (602,594 params), 96^3 patch, batch 1, fp16 autocast, forward and backward. Peak allocated 144 MiB, reserved 198 MiB, limit 4096 MiB.
- AC-4 Dice toy case (0.75) passed. NiBabel loading of `BraTS-GLI-00000-000` (4 modalities + seg, labels in {0,1,2,3}) passed.
- AC-4 HD95 toy case (3.0) failed first: MONAI needs `scipy` (`scipy.ndimage.binary_erosion`). Human ran `uv add scipy` (scipy 1.18.1, as ADR 0001 allows); re-run: 3 passed, 1 deselected (GPU test passed separately).
- Caveat: the 144 MiB figure is one tiny step with random data, not a training-memory estimate.
- Note: the 2026-10-21 run created `.venv` inside the repo because `UV_PROJECT_ENVIRONMENT` was not set in that shell. It is git-ignored; see ADR 0001 for the intended location.
