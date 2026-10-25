# Review, feat/tracer-bullet, 2026-10-25

**Reviewed by**: Sonnet 5.5 (author on unknown model)
**Scope**: about 40 code, test and config files, branch vs main (b900a2d) plus untracked tests
**Verdict**: Changes requested

## Summary
Adds the shared `experiments/common/` code, the data audit and split experiments, and the tracer bullet pipeline (train, evaluate, table, check) with fast tests (65 pass in about 5 s). The scientific core is correct: label to region mapping, channel order by suffix, normalisation, missing channel zeroing, the empty region rule, HD95 spacing, sliding window and evaluation on validation cases only all check out. The headline problem is an undeclared dependency (pandas) that breaks the audit and split scripts in the locked environment, plus 39 ruff errors.

## Major
### 🟠 pandas is imported but not declared or locked, `experiments/0001-splits/make_splits.py:11`
**Problem**: `make_splits.py`, `check_splits.py` (line 11 and 60), `summarize.py:6` and `audit.py:169` import pandas. `pyproject.toml` lists only monai, nibabel, scipy, torch, and `uv.lock` has no pandas entry. `uv run python -c "import pandas"` fails with ModuleNotFoundError (verified).
**Why it matters**: `data/splits/splits.csv` is the root of every later experiment and cannot be regenerated or checked in the pinned environment. This breaks the reproducibility rule and ADR 0001. It also means the split code has no test that can run.
**Suggested fix**: Either add pandas to dependencies with a lock update (needs an ADR line, per "no new infrastructure"), or rewrite the four scripts with the csv module and numpy. Then run `check_splits.py` in the locked env.

### 🟠 Lint gate fails: ruff reports 39 errors, `pyproject.toml:21`
**Problem**: `uv run ruff check .` finds 39 errors (8 auto fixable), including in `tests/test_tracer_units.py` and the experiment scripts. AGENTS.md lists ruff as a command.
**Why it matters**: CI that runs lint will fail, and the "lint" convention is unmet.
**Suggested fix**: Fix or scope the ruff config (for example per-file ignores for the audit scripts), then re-run.

## Minor
### 🟡 Split scripts are not tested and run at import, `experiments/0001-splits/check_splits.py:100`
`main()` is called at module level with no `__main__` guard, hard codes `len(a) == 1251`, and reads `results/0000-data-audit/cases.csv`, which is gitignored. A fresh clone cannot run it. pytest has no split test beyond `test_select_subjects_real_splits`. Add a guard and a toy leakage test under `tests/`.

### 🟡 Unexpected label values are silently ignored, `experiments/common/data.py:25`
`label_to_regions` uses `np.isin`, so a stray label (for example legacy 4) is dropped from WT/TC/ET without a warning, and `load_case` casts to int16 without checking. Assert that labels are a subset of {0,1,2,3} in `Preprocess`. The audit checks this, but training does not.

### 🟡 Unexpected exceptions leave `run.json` at status running, `experiments/0002-tracer-bullet/train.py:189`
Only OOM, non finite loss and Ctrl+C are caught. Any other error (bad file, shape error) propagates and the `finally` writes the record with status `running`. `check_tracer` handles that case, but a `failed` status with the reason would be cleaner.

### 🟡 "Empty" column counts correct both-empty cases, `experiments/0002-tracer-bullet/make_table.py:176`
`gt_empty or pred_empty` includes rows where both are empty (Dice 1 by rule). The spec says "rows scored by the empty rule", so this is consistent, but the column can read as "failures". Consider splitting both-empty from one-sided empty before the real tables.

### 🟡 `best.pt` is selected but evaluation uses `last.pt`, `experiments/0002-tracer-bullet/evaluate.py:258`
Matches AC-7 ("final checkpoint"), but the four val subjects serve for checkpoint selection (train.py:126) and for reporting. Fine for a tracer labelled "Not a result"; for real experiments, report on the test fold with the protocol fixed first.

### 🟡 `subject_of` assumes the BraTS ID format, `experiments/common/data.py:50`
`case.split("-")[2]` breaks silently for other IDs (MSD names, malformed strings). Validate with the regex used in `audit.py:22`.

## Nits
- ⚪ `experiments/0002-tracer-bullet/train.py:167`, run_id has one second resolution; two runs with the same config in one second collide on `mkdir`.
- ⚪ `experiments/common/data.py:29`, "first N subjects by sorted ID" is a deterministic but non random subset; fine for the tracer, note it when sizing real runs.
- ⚪ `experiments/common/model.py:257`, stray double blank line.
- ⚪ `experiments/0002-tracer-bullet/evaluate.py:273`, the checkpoint `config_hash` is stored but never compared with the run config.

## Strengths
- Region mapping (WT 1|2|3, TC 1|3, ET 3), suffix based channel order, crop on raw stack with an asserted tumour voxel count, and the empty region rule with MONAI HD95 only when both sides are non empty are all correct and tested with toy known answers (Dice 0.75, HD95 3.0, spacing).
- Evaluation reads `val_cases` from `run.json` rather than reselecting, never touches the test fold, and `select_subjects` asserts no test case and no subject overlap.
- Run records (config hash, git state, versions, determinism state, atomic writes) and failure tests (inf loss, missing file, Ctrl+C) are thorough.

## Test coverage
65 fast tests pass (`uv run pytest -q`). Covered: data mapping, normalisation, crop, metrics, missing patterns, seed determinism, runlog, model, train failure paths, table and check. Not covered: `make_splits.py`/`check_splits.py` (no test, needs pandas), `audit.py`, `train` on label sanity, the GPU path (slow, human run).
