# 0004. Missing modality evaluation (E2): verify

How `/check verify` proves each acceptance criterion in [index.md](index.md). The AI may run the fast checks. The human runs the slow ones and pastes the last two lines of each.

All commands run from the repo root with `UV_PROJECT_ENVIRONMENT` and `UV_LINK_MODE=copy` set (ADR 0001).

## Fast (AI may run, no GPU, no real data)

    uv run pytest -q
    uv run ruff check .

Expected: all pass, `slow` deselected, ruff clean. Tests live in `tests/test_common_stats.py`, `tests/test_common_evaluation.py` and `tests/test_missing_eval.py`.

| AC | Proof |
|---|---|
| AC-1 | `eval_cases` on the committed `splits.csv` gives 228 test cases, unique subjects, sorted; a shared subject stops with exit 2 |
| AC-2 | Non `ok` run, missing `best.pt`, hash mismatch, and a test case refused by `allowed_folds` |
| AC-3 | 2 × 15 × 3 rows, exact columns, a forced NaN sets `failed` |
| AC-4 | A recording `image_hook` sees the channels already zeroed |
| AC-5 | `EVAL_STOP_AFTER` and a truncated last line, then a resume equal to an uninterrupted run; each mismatch refused |
| AC-6 | Finished folder refused; with a reason the old folder is renamed and logged |
| AC-7 | One test per refused condition (including `git_dirty` of `None` and a changed `protocol_hash`), each proving no `.npz` was opened; one passing path with `git_info` patched |
| AC-9, AC-16 | Toy eval folders, one rule broken at a time: GPU differs, mean off by 0.002, one case off by 0.02, each refused; the projected hours printed on val |
| AC-16 | A finished synthetic test eval with `metrics.csv` and `n_cases` both cut to one case: `check_rules` reports `case set differs from the test fold`, and a test subject shared with train is reported, not raised (`test_gate_open_scores_the_synthetic_test_fold_once`) |
| AC-5 | A resume with a different `device` or `gpu` stops with exit 2 and cuts nothing (`test_resume_mismatch_is_refused_before_any_cut`) |
| AC-14 | `make_table.py` refuses a missing `run.json`, a status that is not `ok`, and an `n` that differs from `n_cases`, each exit 1 with any older `.tex` byte equal; the Dice and HD95 captions carry the star line with the config `alpha` |
| AC-20 | `protocol_hash` changes when any entry of `PROTOCOL_SOURCES` or `REGION_LABELS` changes, and stays when `build_model` changes; the hashed text of `predict` contains `sliding_window_inference` |
| AC-10 | Exact `run.json` key set; an injected exception gives `failed` and a `reason` |
| AC-11 | Toy cases with known answers, including fully tied Friedman rows giving statistic 0, p 1, W 0 |
| AC-12 | Row counts (90, 18, 24), a missing row refused, two runs byte identical; a toy case where `marginal.csv` gives a known matched pair answer; refused with exit 1 and no output file: `run.json` missing, each status that is not `ok`, `status` or `n_cases` absent or null, a sweep stopped between two cases, and fewer cases than `n_cases` under status `ok`; a refusal leaves an older output file byte equal; a quick run with status `ok` accepted |
| AC-13 | Toy CSVs reaching each `H1a` and `H1b` line, compared exactly with the template |
| AC-14 | Captions, header comment, val note present; a wrong row count refused; a test reads `e2-preview.tex` and finds the four inputs; a star appears or goes when `alpha` changes in the eval folder's `config.toml` |
| AC-15 | Two quick CPU runs in two `--out` folders, byte equal |

Script rule: `tests/test_experiment_scripts.py` covers the new scripts (nothing runs on import).

## Slow (the human runs these, no AI polling)

**AC-14 compile.** Build the stub, after a quick run has written the four `.tex` files:

    cd writing/tracer && latexmk -xelatex e2-preview.tex

**AC-18 real val run**, on Kaggle by `experiments/0004-missing-eval/KAGGLE.md`:

    uv run python experiments/0003-baseline/preprocess.py --folds val
    uv run python experiments/0004-missing-eval/evaluate.py --run <E1 run> --fold val --out <out>
    uv run python experiments/0004-missing-eval/analyse.py --eval <eval dir>
    uv run python experiments/0004-missing-eval/make_table.py --eval <eval dir> --out-dir <out>
    uv run python experiments/0004-missing-eval/check_eval.py <eval dir>

Expected: `check_eval.py` passes, including the AC-9 tie out, with `predict_seconds` and `metric_seconds` in `run.json`, and prints the projected test hours.

## By reading

| AC | Check |
|---|---|
| AC-8 | No real test case in any cache, and no `-test` eval folder in `results/0004-missing-eval/` |
| AC-17 | `KAGGLE.md`, ADR 0002, `data/DATA.md` and `experiments/AGENTS.md` updated |
| AC-19 | Scope rows reworded |
