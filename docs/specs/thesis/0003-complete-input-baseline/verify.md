# 0003. Complete input baseline (E1): verify

How `/check verify` proves each acceptance criterion in [index.md](index.md). The AI may run the fast checks itself. The human runs the slow checks and pastes the last two lines of each.

All commands run from the repo root with `UV_PROJECT_ENVIRONMENT` and `UV_LINK_MODE=copy` set (ADR 0001) and `BRATS_TRAIN_DIR` pointing at the BraTS training folder.

## Fast (AI may run, no GPU, no real data)

    uv run pytest -q
    uv run ruff check .

Expected: all pass, `slow` tests deselected, ruff clean.

| AC | Test that proves it |
|---|---|
| AC-1 | `tests/test_common_data.py::test_fold_cases_*` (on the committed `splits.csv`: 863 train, 114 val, no `test` row, sorted, same on a second call, flagged excluded, timepoint filter) |
| AC-2 | `tests/test_common_cache.py::test_write_case_roundtrip` (synthetic case: float16 within `rtol=1e-3, atol=1e-3`, label equal, spacing and crop box), `::test_manifest_and_cache_json_keys`, `::test_rerun_skips_existing`, `::test_verify_detects_changed_file`, `::test_low_disk_exit_2` |
| AC-3 | `tests/test_common_cache.py::test_hash_mismatch_exit_2`, `::test_missing_case_exit_2`, `::test_preprocess_refuses_other_hash` |
| AC-4 | `tests/test_baseline.py::test_no_test_case_loaded` (mini `splits.csv` with a `test` row: not cached, and a forced test case raises) |
| AC-5 | `tests/test_common_data.py::test_flip_keeps_alignment`, `::test_brain_intensity_keeps_background_zero` (per channel mask), `::test_brain_intensity_uses_self_R` (same `set_random_state` seed, same output; global NumPy seed changes nothing), `::test_eval_transforms_no_augmentation`, `::test_cached_case_keys_and_dtypes` |
| AC-7 | `tests/test_common_training.py::test_lr_warmup_cosine` (step 1 is `lr / warmup`, step `warmup` is `lr`, step `iterations` is 0), `::test_iterations_not_multiple_exit_2`, `::test_scaler_off_on_cpu` |
| AC-8 | `tests/test_common_training.py::test_selection_score_and_tie` (fixed fake Dice values: score = mean of region means at pattern 1, a tie keeps the earlier one), `::test_validate_patterns_column` (`patterns=[1, 15]` gives both `pattern_id` values in `val_log.csv`) |
| AC-10 | `tests/test_baseline.py::test_run_json_keys` (exact key set), `::test_seed_override_changes_hash` |
| AC-11 | `tests/test_baseline.py::test_resume_restores_state` (weights, optimiser, scheduler lr, scaler and iteration equal `last.pt`; log rows 1 to 10 with no gap or duplicate), `::test_resume_refusals` (status `ok`, live pid, edited config, other cache hash), `::test_resume_drops_later_best` (`best.pt` at 10, `last.pt` at 5: `best.pt` deleted, best from `last.pt`) |
| AC-12 | `tests/test_baseline.py::test_inject_inf_failed` (`BASELINE_INJECT_INF=7`: status `failed`, `failed_iteration` 7, exit 1, `last.pt` at 5; the resume runs with the variable unset) |
| AC-13 | `tests/test_baseline.py::test_evaluate_rows` (1 × 3 rows, no NaN), `::test_evaluate_refuses_not_ok` |
| AC-14 | `tests/test_baseline.py::test_table_caption`, `::test_table_empty_counts` (toy rows give `Both empty` 1 and `One empty` 1) |
| AC-15 | `tests/test_baseline.py::test_quick_twice_same_loss` (CPU, `loss` and `lr` within 1e-6, with `num_workers` 0 and 2) |
| AC-17 | `tests/test_common_training.py::test_hook_rng_independent` (a hook that draws from `rng` leaves the patch and flip stream of the same seed unchanged) |
| AC-16 (part) | `tests/test_baseline.py::test_check_rejects_quick` |
| AC-17 | `tests/test_experiment_scripts.py` covers the new scripts (nothing runs on import); the tracer tests still pass |

The baseline tests build synthetic NIfTI cases and a mini `splits.csv` in `tmp_path` and run the scripts with `--quick --device cpu`.

## Quick thread on real data (AI may run once, a few minutes, estimated)

    uv run python experiments/0003-baseline/preprocess.py --quick
    uv run python experiments/0003-baseline/train.py --quick --device cpu
    uv run python experiments/0003-baseline/evaluate.py --run <run dir printed above> --device cpu
    uv run python experiments/0003-baseline/make_table.py --run <run dir>
    uv run python experiments/0003-baseline/check_baseline.py <run dir>

Expected: status `ok` and `"quick": true`, `metrics_val.csv` has 3 rows, the table caption says n = 1, and `check_baseline.py` prints `FAIL: quick run`. Proves the wiring of AC-2, AC-8, AC-13, AC-14, AC-16 on real files (structure only).

## Slow (human runs)

Run in this order and paste the last two lines of each:

    uv run python experiments/0003-baseline/preprocess.py
    uv run python experiments/0003-baseline/preprocess.py --verify
    uv run python experiments/0003-baseline/memory_test.py
    uv run python experiments/0003-baseline/train.py
    uv run python experiments/0003-baseline/evaluate.py --run <run dir>
    uv run python experiments/0003-baseline/make_table.py --run <run dir>
    uv run python experiments/0003-baseline/check_baseline.py <run dir>
    uv run pytest -m slow -q

| Step | Expected | AC |
|---|---|---|
| `preprocess.py` | 996 cases written, about 30 GB (estimated), `cache.json` folds `train,val` | AC-2, AC-4 |
| `preprocess.py --verify` | every sha256 matches, 3 recomputed cases within tolerance | AC-2 |
| `memory_test.py` | `PASS`, peak reserved below 3,584 MiB over 200 steps, val on the 2 largest val cases and 20 more steps; s/iter, data wait and projected hours printed. If `FAIL`, or projected above 10 h, stop and go back to `/architect` | AC-6 |
| `train.py` | status `ok`, 40,000 iterations, 10 val points (time unknown until the memory test) | AC-7, AC-8, AC-10 |
| `evaluate.py` | 342 rows, mean Dice per region printed | AC-13 |
| `make_table.py` | `writing/tables/0003-baseline-val.tex` written | AC-14 |
| `check_baseline.py` | `PASS` | AC-9, AC-16 |

If the run crashes, `train.py --resume <run dir>` continues it, and `check_baseline.py` still has to print `PASS` afterwards (AC-11).
