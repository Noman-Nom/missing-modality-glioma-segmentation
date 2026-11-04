# Review, feat/e2-missing-eval, 2026-11-04

**Reviewed by**: Claude Sonnet 5.5 (reviewer; author model not stated)
**Scope**: 23 files, branch vs main (merge base 351b054, diff to HEAD)
**Verdict**: Changes requested

## Summary
Adds E2: a shared 15 pattern sweep (`common/evaluation.py`), a statistics module (`common/stats.py`), and the scripts `evaluate.py`, `analyse.py`, `make_table.py`, `check_eval.py`, plus spec, ADR, Kaggle and records updates. The code follows spec 0004 closely. Imputation is zero after normalisation via the existing `apply_pattern`, metrics reuse the single empty region rule, the statistics are as specified (Wilcoxon on matched per subject differences, Holm per (metric, region), Friedman inside equal size groups, shared bootstrap matrix), and the analysis guards and table generation read the produced files. `uv run pytest -q` gives 407 passed and ruff is clean. One major issue (resume does not guard the GPU, which the AC-9 tie out relies on) and several minors. No blockers, and no invented or fabricated numbers found.

## Major
### 🟠 Resume does not guard or refresh `device` and `gpu`, `experiments/common/evaluation.py:42`
**Problem**: `RESUME_KEYS` has `git_commit, config_hash, protocol_hash, source_run_id, source_config_hash, source_best_iteration`. `device`, `gpu`, `seed` and `splits_sha256` are not compared, and on resume `evaluate.py:150-154` rebuilds the record from the old fields, so `gpu` and `device` keep the first session's values.
**Why it matters**: AC-9 requires `gpu` in the eval `run.json` to equal E1's, and `check_rules` (`evaluation.py:168`) uses that field. A session that dies on a T4 and is resumed on a P100 or on CPU (`--device cpu`, or a new Kaggle notebook with the wrong accelerator, an easy mistake per `KAGGLE.md`) will mix rows from two devices under a `run.json` that still says `Tesla T4`. The tie out would then pass on stale metadata, and nondeterminism between GPUs would be hidden from the thesis numbers.
**Suggested fix**: Add `device`, `gpu` (and `splits_sha256`, `seed`) to `RESUME_KEYS` so a different device is exit 2, or record each session's gpu in the `resumes` entry and make `check_rules` fail if any entry differs from E1's. Add a test using the existing mismatch parametrisation (`test_resume_mismatch_is_refused_before_any_cut`).

## Minor
### 🟡 Test fold has no expected case set or count check, `experiments/common/evaluation.py:206-211`
`check_rules` takes `n_cases` from the eval's own `run.json`, so a test eval is never compared with 228 or with `fold_cases(...)`. A truncated case list that was written consistently would pass. Spec AC-16 asks for "every case present"; for test, recompute the case list (or at least compare with the val eval's n times 2 per the spec's own assumption) and report a difference.

### 🟡 Stale device/protocol coverage gap: `predict()` is outside `protocol_hash`, `experiments/common/evaluation.py:26-30`
`model.py` (sliding window, overlap, fp16 autocast, threshold use) and `data.py`/`evaluate.py` are not hashed. This is a deliberate spec decision (E4 changes channels), but it means a change to `predict()` after the val run still passes the gate (`check_test_gate`), because the AC-9 tie out only runs on val at val time. Consider also storing the `git_commit` of the val eval and failing the gate if `common/model.py`, `common/data.py` differ between that commit and HEAD, or hashing `predict()` input settings in the E1 `[infer]` block. At least note it in the Consequences.

### 🟡 Table stars ignore direction, `experiments/0004-missing-eval/make_table.py:58-60`
A star is printed whenever `p_holm < alpha`, including a pattern whose `mean_diff` is positive (Wilcoxon is two sided). H1a reads the sign, the table does not. A starred positive Dice difference would read as "significant drop". Add the sign condition for Dice, or state in the caption what the star means.

### 🟡 `make_table.py` does not check that the CSVs belong to a finished run, `experiments/0004-missing-eval/make_table.py:201-205`
It reads `run.json` for captions but never checks `status == "ok"`, `quick`, or that `n` in `stats.csv` equals `n_cases`. After a rerun or resume, an old `stats.csv` could be turned into a table with a new run's provenance header and caption. Spec only asks for row counts, but "table actually reads the produced results" is stronger with a status and `n` check.

### 🟡 Asymptotic Wilcoxon with few non zero differences, `experiments/common/stats.py:332`
`method="asymptotic"` is specified, but when `n_nonzero` is small (identical predictions for a pattern in many cases, e.g. ET with FLAIR missing) the normal approximation is poor. Stats rows carry `n_nonzero`, so the data to flag it exist; consider noting in the thesis text or a warning column. Not a protocol change if only reported.

### 🟡 `old` truthiness decides resume vs fresh, `experiments/0004-missing-eval/evaluate.py:143,149`
`if old` works because a loaded run.json is a non empty dict, but `prepare_folder` returns `None` for fresh. Use `is not None` to avoid a silent fresh start on a future change.

## Nits
- ⚪ `experiments/0004-missing-eval/make_table.py:72` a tiny negative mean prints `-0.000`; format after rounding.
- ⚪ `experiments/0004-missing-eval/evaluate.py:102-110` `torch.load(best, map_location="cpu")` without `weights_only` is fine for own file, but state the choice (the file hash is pinned in KAGGLE.md, good).
- ⚪ `experiments/common/evaluation.py:118` `_whole_pair` decodes with `errors="replace"`; a corrupt line could be accepted as a valid pair if it still splits into 12 fields. Unlikely after fsync, but the row value is not validated.

## Strengths
- Test gate runs before any case list or `.npz` access, reruns `check_rules` on the val eval instead of trusting a flag, treats `git_info() == (None, None)` as dirty, and prints every failed rule.
- Resume is careful: truncated lines and partial pairs are cut, never edited; timings add up per pair; a NaN stops before the pair is written; the byte equal resume property is tested for several cut points.
- `analyse.py` guards in spec order (run.json, status, row set, case count) and writes nothing on refusal; verdict is computed on unrounded numbers with a fixed template; Holm, Friedman tie case and all zero Wilcoxon are handled and tested on known answers.
- Unit of pairing is the subject, one shared bootstrap index matrix pairs every row, and the marginal effect is built from exact one modality matched pairs.
- AC-9 tie out compares per (case, region) as well as the mean, so a compensating drift cannot hide.

## Test coverage
Strong. 407 fast tests pass, ruff clean. `tests/test_missing_eval.py` covers sweep rows, source and cache guards, NaN to failed, determinism, every analysis refusal, verdict rules, table layout and star vs alpha, folder rule, resume (cut variants, mismatch, failed resume), tie out, and each test gate condition. `test_common_evaluation.py` and `test_common_stats.py` cover the shared code (including the hook order). Gaps: no test for a resume on a different `device`/`gpu` (see the major), no test that a test eval with a wrong case set fails `check_rules`, and no test that a starred pattern with positive `mean_diff` is handled. Slow and GPU runs were not executed (by instruction).
