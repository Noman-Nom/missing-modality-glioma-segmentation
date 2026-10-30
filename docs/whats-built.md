# What's built

**Last updated:** 2026-10-25, tracer review fixes built (spec 0002) on top of the tracer bullet (spec 0001)
**Last full check:** only git and hook checks. Not run: CI, any experiment (none exist), citation verification.

| # | Feature | Status | What it gives you |
|---|---|---|---|
| A | Repo and AI setup | done | Rules, skills, docs layout, pre-commit hook |
| B | Proposal | built, not supervisor-reviewed | `writing/proposal/proposal.pdf`, 7 pages |
| C | Data audit (E0) | done, run on all 1470 cases | `experiments/0000-data-audit/` writes `results/0000-data-audit/cases.csv` and `summary.md` |
| D | Subject-level split (infrastructure, no E number) | done, check passes, leakage test in `tests/test_splits.py` | `data/splits/splits.csv`, `experiments/0001-splits/` |
| E | Python environment and smoke test | done, 3 fast + 1 GPU test pass | `pyproject.toml`, `uv.lock`, `tests/test_env_smoke.py` |
| F | Tracer bullet (spec 0001) | built, full run passes `check_tracer.py`; not yet verified or tested by /check, /test | `experiments/common/`, `experiments/0002-tracer-bullet/`, `writing/tracer/tracer.tex` |
| G | Tracer review fixes (spec 0002) | built, verified by /check verify and tested by /test (2026-10-25) | pandas locked, `ruff` clean, `last.pt` only, strict labels and IDs, `tests/test_splits.py`, `tests/test_experiment_scripts.py` |

## A. Repo and AI setup
Any AI chat reads `AGENTS.md` and starts with the same rules. Enable the hook with `git config core.hooksPath .githooks`.

## B. Proposal
Build: `cd writing/proposal && latexmk -xelatex proposal.tex`. Citations are unverified.

## C. Data audit (E0)
Run: `uv run python experiments/0000-data-audit/audit.py --train <train dir> --val <val dir>` then `uv run python experiments/0000-data-audit/summarize.py`. Check: `uv run python experiments/0000-data-audit/check_audit.py <train dir>`. Own numpy NIfTI reader; case IDs parsed with the shared `BRATS_ID_RE` from `common/data.py`. Full run done 2026-10-20.

## D. Splits (infrastructure, no E number)
`uv run python experiments/0001-splits/make_splits.py` writes `data/splits/splits.csv` (train 791 / val 114 / test 228 subjects, seed 42). Check: `uv run python experiments/0001-splits/check_splits.py` (no subject in two folds, deterministic). `--compare <csv>` checks a regenerated split against the committed one without overwriting it. Fast test with no data: `tests/test_splits.py` (toy case, committed split has no leakage). E numbers to folders: the table in `experiments/AGENTS.md`.

## E. Environment and smoke test (ADR 0001)
Resolved versions (2026-10-20): Python 3.12.3, torch 2.14.1+cu130 (arch list has `sm_75`), monai 1.6.1, nibabel 5.4.2, scipy 1.18.1 (added 2026-10-21), pytest 9.1.1, ruff 0.17.1, uv 0.13.2.
Tests: `uv run pytest tests/test_env_smoke.py -v` (fast) and `uv run pytest tests/test_env_smoke.py -m slow -v -s` (GPU, human runs it). Default runs skip `slow`.
Run on 2026-10-21 (human, Quadro T2000 Max-Q):
- AC-3 passed: MONAI 3D U-Net (602,594 params), 96^3 patch, batch 1, fp16 autocast, forward and backward. Peak allocated 144 MiB, reserved 198 MiB, limit 4096 MiB.
- AC-4 Dice toy case (0.75) passed. NiBabel loading of `BraTS-GLI-00000-000` (4 modalities + seg, labels in {0,1,2,3}) passed.
- AC-4 HD95 toy case (3.0) failed first: MONAI needs `scipy` (`scipy.ndimage.binary_erosion`). Human ran `uv add scipy` (scipy 1.18.1, as ADR 0001 allows); re-run: 3 passed, 1 deselected (GPU test passed separately).
- Caveat: the 144 MiB figure is one tiny step with random data, not a training-memory estimate.
- Note: the 2026-10-21 run created `.venv` inside the repo because `UV_PROJECT_ENVIRONMENT` was not set in that shell. It is git-ignored; see ADR 0001 for the intended location.

## F. Tracer bullet (spec 0001)
Amended by spec 0002 (section G): only `last.pt` is saved and reported, `run.json` has no `best_iteration`, the table has both empty and one empty counts.
Shared code in `experiments/common/`: `seed.py` (`set_seed`), `runlog.py` (config hash, git, versions, `run.json`), `data.py` (channel order T1, T1ce, T2, FLAIR by file suffix; WT/TC/ET; subject selection; crop and normalise), `missing.py` (15 patterns), `metrics.py` (empty region rule, HD95 373.13 mm), `model.py` (U-Net and sliding window prediction, shared by validation and evaluation).
Commands (repo root, `BRATS_TRAIN_DIR` set): `train.py [--quick]`, `evaluate.py --run <dir>`, `make_table.py --run <dir>`, `check_tracer.py <dir>`, all in `experiments/0002-tracer-bullet/`; then `cd writing/tracer && latexmk -xelatex tracer.tex`.
Fast tests: `uv run pytest -q` gives 30 passed, 2 slow deselected (2026-10-25), including a CPU thread test on synthetic NIfTI cases.
Quick run 2026-10-25 (AI, GPU, `--quick`, run `20261024-235630-31b12b4f`, dirty tree): status ok, 10 iterations, 20.2 s, peak reserved 368 MiB, 45 metrics rows, table and `tracer.pdf` build, `check_tracer.py` prints `FAIL: quick run` as designed. These numbers are wiring checks, not results.
Full run 2026-10-25 (human, Quadro T2000 Max-Q), run `20261025-000309-f4bbd0b9`, seed 42:
- AC-5: 2,000 iterations, wall 385.3 s, peak reserved 484.0 MiB (allocated 209.6 MiB) of 4096.
- AC-6: mean loss first 50 iterations 1.531, last 50 0.360. Pattern 1 (complete input) mean WT Dice 0.9032 on the 4 val subjects (00017, 00056, 00061, 00070), threshold 0.3.
- Val WT Dice during training: 0.772 (500), 0.796 (1000), 0.834 (1500), 0.903 (2000); best is the last. GradScaler skipped 5 steps.
- AC-7, AC-12, AC-13: 180 metrics rows, table built, `tracer.pdf` builds (n = 4), `check_tracer.py` PASS. Slow tests: 2 passed.
- Warnings logged: 2 (a fork with threads warning from the DataLoader, a MONAI deprecation); no determinism warnings.
- Caveat: `run.json` records git `49f1111` with a dirty tree, because the code was committed afterwards as `7111e2c`. The next real experiment should run from a clean commit.
- These are tracer numbers on the first 4 val subjects by ID, not results; never quote them.

## G. Tracer review fixes (spec 0002)
Fixes from the fresh model review (`docs/reviews/2026-10-25-feat-tracer-bullet.md`).
- Environment: `pandas` 3.1.0 locked (ADR 0001). The audit and split scripts run with `uv run`.
- Split reproduce check: `make_splits.py --out <tmp>/splits.csv`, then `check_splits.py --compare <tmp>/splits.csv`. Human run 2026-10-25 under pandas 3.1.0: 1251 cases, same fold per case, committed file unchanged; SHA256 recorded in `data/DATA.md`. `tracer.pdf` builds with 0 overfull warnings.
- Lint: `uv run ruff check .` passes. `EXE002` is ignored because of the NTFS executable bit (ADR 0001).
- Checkpoint rule: a fixed iteration budget, report the final weights (`last.pt`), no selection. `evaluate.py` exits 1 when the `last.pt` config hash differs from `run.json`. `check_tracer.py` still accepts old runs that hold `best_iteration` (`LEGACY_KEYS`); the full run `20261025-000309-f4bbd0b9` still prints PASS.
- Fail loudly: `load_case()` raises `ValueError` on label values outside 0 to 3 or non integer values; `subject_of()` raises on anything that is not `BraTS-GLI-NNNNN-NNN` (`BRATS_ID_RE`, also used by `audit.py`); any crash in training writes status `failed` with the exception type, then re raises.
- Run ids: a taken run folder gets a suffix `-1`, `-2`, and `run_id` equals the folder name.
- Table: per region Dice mean and median, HD95 mean and median, Both empty (Dice 1), One empty (Dice 0); headers on two rows so the table fits the landscape page.
- No script under `experiments/` runs code on import (`check_splits.py`, `check_audit.py`, `summarize.py` now have `main()`).
- Fast tests: `uv run pytest -q` gives 84 passed, 2 slow deselected (2026-10-25).

## H. E1 complete input baseline (spec 0003), in progress
Milestones 1 to 3 of 4 built (thin thread, thicken, robustness and checks, build plan items 1 to 6, 2026-10-27 to 2026-10-31, branch `feat/e1-baseline`). Only quick runs on real data so far; the human runs (milestone 4) are next.
- Shared code in `experiments/common/`: `data.py` adds `fold_cases()` (sorted case IDs of one fold, timepoint and FLAIR flag filters) and `CachedCase` with `cached_train_transforms()` and `cached_eval_transforms()`. `cache.py` is new: `PREPROCESS_VERSION = 1`, `preprocess_hash()`, `preprocess_case()`, `write_case()` (float16 `.npz` through a temp file), `manifest.csv` and `cache.json` writers, and `check_cache()` (the hash, missing case and test fold guard). `training.py` is new: `train()` with `batch_hook(batch, rng)` (its own generator, seed + 1), `validate(model, val_cases, patterns, ...)` (Dice only), the selection score, `best.pt` (strict improvement) and `last.pt`, and the E1 `run.json` key list. `metrics.region_metrics()` takes `with_hd95=False` so val skips HD95. `runlog.RunRecord` takes optional `keys` and `defaults` (tracer behaviour unchanged).
- Scripts in `experiments/0003-baseline/`: `config.toml`, `preprocess.py [--folds] [--quick]`, `train.py [--quick] [--seed N] [--device cpu]`, `evaluate.py --run <dir>` (`metrics_val.csv`, pattern 1 only), `make_table.py --run <dir>` (`writing/tables/0003-baseline-val.tex`).
- Thicken (milestone 2, AC-5, AC-7, AC-8): `training.lr_factor()` with `LambdaLR` (linear warmup over `warmup` steps, cosine to 0 at the last step; the logged `lr` is the one applied); `training.check_config()` refuses `iterations` not a multiple of `val_every` (`train.py` exit 2); `val_log.csv` (one row per iteration, pattern, case, region); the fixed write order at each val point (`val_log.csv`, `best.pt` if improved, `last.pt`, `run.json`, each through a temp file and rename). `data.py` adds `RandBrainIntensityd` (per channel scale and shift on nonzero voxels only, draws only from its own `self.R`) and three `RandFlipd` (one per axis, the same flip for `image`, `label`, `wt`) after the crop in `cached_train_transforms()`. The shared `setup` fixture and `only_run()` moved to `tests/conftest.py`.
- Robustness and checks (milestone 3, AC-2, AC-6, AC-9, AC-11, AC-12, AC-15, AC-16): `training.Trainer` holds the model, optimiser, schedule, scaler and loader, and its `step()` is one iteration (shared by `train()` and `memory_test.py`). `last.pt` holds the full state (optimiser, scheduler, scaler, best score and iteration, `scaler_skipped_steps`, hook generator, torch, CUDA, NumPy and Python RNG states). `train.py --resume <run dir>` (only `--device` with it) rolls `train_log.csv`, `val_log.csv` and `val_history` back to the `last.pt` iteration, deletes a `best.pt` written after it, reseeds a fresh loader with `seed * 100003 + iteration`, and adds one `resumes` entry; it refuses (exit 2) status `ok`, a live `train.pid`, an edited `config.toml`, a changed cache hash, or no `last.pt`. Failures keep `failed_iteration` and leave `last.pt` usable; peaks, wall time and warnings carry across segments. `preprocess.py --verify` (sha256 of every file, float32 recompute of the first 3 train cases) and a free disk check (35 MB per case to write, exit 2). `memory_test.py` (steps, val on the 2 largest val cases, 20 more steps; `memory_test.json`; a CPU run never passes). `check_baseline.py <run dir>` (structure, then the full run thresholds; a quick run gets `FAIL: quick run`). Known gap: if a crash lands between the `best.pt` and `last.pt` writes, the resume deletes that `best.pt`, and the older best weights are gone; the run then needs a later improvement or `evaluate.py` refuses it (flagged for `/architect`).
- Quick run on real data 2026-10-27 (human, GPU, `--quick`, run `20261027-005949-420300a3`, commit `1dcecc5`, dirty tree from the unrelated proposal edits): `preprocess.py --quick` cached 3 cases (2 train, 1 val, 0.09 GB); `train.py --quick` status ok, 10 iterations, 7.6 s, peak reserved 450 MiB; `evaluate.py` wrote 3 rows, no NaN; the table builds with the AC-14 caption. Wiring checks at 64^3 and 10 steps, not results.
- Quick run after milestone 2 (AI, GPU, `--quick`, into a scratch folder, run `20261027-010520-420300a3`, commit `793b015`, dirty tree): status ok, 10 iterations, 8.0 s, peak reserved 450 MiB, lr 0.0005, 0.001, then cosine to 0.0 at step 10, `val_log.csv` 6 rows, no warnings. Wiring checks, not results.
- Fast tests: `uv run pytest -q` gives 156 passed, 2 slow deselected (2026-10-31); `tests/test_baseline_thread.py` has 12 (including a CPU thread test on synthetic cases), `tests/test_baseline_thicken.py` adds 7, `tests/test_baseline_robust.py` 11 (failures, resume, verify, disk), `tests/test_baseline_checks.py` 17 (memory test wiring, check script, determinism with 0 and 2 workers).
