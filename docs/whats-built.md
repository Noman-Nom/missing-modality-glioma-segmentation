# What's built

**Last updated:** 2026-10-17, milestone A and proposal draft
**Last full check:** only git and hook checks. Not run: CI, any experiment (none exist), citation verification.

| # | Feature | Status | What it gives you |
|---|---|---|---|
| A | Repo and AI setup | done | Rules, skills, docs layout, pre-commit hook |
| B | Proposal | built, not supervisor-reviewed | `writing/proposal/proposal.pdf`, 7 pages |
| C | Data audit (E0) | done, run on all 1470 cases | `experiments/0000-data-audit/` writes `results/0000-data-audit/cases.csv` and `summary.md` |
| D | Subject-level split (E1) | done, check passes | `data/splits/splits.csv`, `experiments/0001-splits/` |

## A. Repo and AI setup
Any AI chat reads `AGENTS.md` and starts with the same rules. Enable the hook with `git config core.hooksPath .githooks`.

## B. Proposal
Build: `cd writing/proposal && latexmk -xelatex proposal.tex`. Citations are unverified.

## C. Data audit (E0)
Run: `python3 experiments/0000-data-audit/audit.py --train <train dir> --val <val dir>` then `python3 experiments/0000-data-audit/summarize.py`. Check: `python3 experiments/0000-data-audit/check_audit.py <train dir>`. Uses numpy only (own NIfTI reader). Full run done 2026-10-20.

## D. Splits (E1)
`python3 experiments/0001-splits/make_splits.py` writes `data/splits/splits.csv` (train 791 / val 114 / test 228 subjects, seed 42). Check: `python3 experiments/0001-splits/check_splits.py` (no subject in two folds, deterministic, toy case).
