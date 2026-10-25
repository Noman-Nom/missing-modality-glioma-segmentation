# 0001. Tracer bullet: verify

How `/check verify` proves each acceptance criterion in [index.md](index.md). Fast checks the AI may run itself. Slow checks are run by the human, who pastes the last two lines.

All commands run from the repo root with `UV_PROJECT_ENVIRONMENT` set (ADR 0001) and `BRATS_TRAIN_DIR` pointing at the BraTS training folder.

## Fast (AI may run, no GPU, no data)

    uv run pytest -q

Expected: all pass, `slow` tests deselected.

| AC | Test that proves it |
|---|---|
| AC-1 | `tests/test_common_data.py::test_select_subjects_*` (counts 16 and 4, no `test` fold, no overlap, deterministic, flagged and timepoint 1 excluded) |
| AC-2 | `tests/test_common_data.py::test_label_to_regions`, `::test_channel_order_by_suffix` |
| AC-3 | `tests/test_common_data.py::test_normalise_*` (clip, z-score over nonzero, background 0, constant channel), `::test_crop_keeps_label_count` |
| AC-4 | `tests/test_common_missing.py` (15 patterns, exact table from index.md, zeros only in missing channels) |
| AC-8 | `tests/test_common_metrics.py` (four empty cases, Dice 0.75, HD95 3.0, 373.13) |
| AC-10 | `tests/test_common_seed.py::test_same_seed_same_loss` |
| AC-11 (part) | `tests/test_tracer_failures.py` (synthetic NIfTI cases and mini `splits.csv` in `tmp_path`, `--device cpu`, 32^3 test config; `TRACER_INJECT_INF=1` gives status `failed`, exit 1; a deleted file exits 2 before training) |
| AC-13 (part) | `tests/test_tracer_failures.py::test_check_rejects_quick` (a fake `quick: true` run folder gives `FAIL: quick run`, exit 1) |

## Quick thread (AI may run once, under about 2 minutes)

    uv run python experiments/0002-tracer-bullet/train.py --quick
    uv run python experiments/0002-tracer-bullet/evaluate.py --run <run dir printed above>
    uv run python experiments/0002-tracer-bullet/make_table.py --run <run dir>
    cd writing/tracer && latexmk -xelatex tracer.tex && cd -

Expected: `run.json` status `ok` and `"quick": true`; `metrics.csv` has 1 × 15 × 3 = 45 rows; the table caption says n = 1; `tracer.pdf` builds; `check_tracer.py` on this run prints `FAIL: quick run`. Proves the wiring for AC-7, AC-9, AC-12 (structure only).

## Slow (human runs, about 20 to 30 minutes, estimated)

    uv run python experiments/0002-tracer-bullet/train.py
    uv run python experiments/0002-tracer-bullet/evaluate.py --run <run dir>
    uv run python experiments/0002-tracer-bullet/make_table.py --run <run dir>
    uv run python experiments/0002-tracer-bullet/check_tracer.py <run dir>
    cd writing/tracer && latexmk -xelatex tracer.tex && cd -
    uv run pytest -m slow -q

| AC | What to see |
|---|---|
| AC-5 | `run.json`: `iterations` 2000, `peak_vram_reserved_mib` below 4096 (`peak_vram_allocated_mib` also logged) |
| AC-6 | `check_tracer.py` PASS (`loss_end_mean` below `loss_start_mean`, mean `dice` of `metrics.csv` rows pattern 1 WT above 0.3) |
| AC-7 | `metrics.csv` 180 rows, no NaN Dice or HD95 (checked by `check_tracer.py`) |
| AC-9 | `run.json` has every key in the *run.json keys* table of index.md; run folder has `config.toml` |
| AC-12 | `writing/tables/0002-tracer-bullet.tex` exists, `tracer.pdf` shows 15 rows with mean, median and empty rule count per region, caption n = 4 |
| AC-13 | `check_tracer.py` prints PASS and exits 0 on the full run |
| AC-14 | `docs/whats-built.md` and `context/progress-tracker.md` list the run id, peak VRAM, wall time, pattern 1 WT Dice |

Paste back: the last two lines of `train.py`, `evaluate.py` and `check_tracer.py`.

If AC-6 fails (Dice at or below 0.3), that is a finding, not something to tune away on the test fold: report it, then debug on the val subjects only.
