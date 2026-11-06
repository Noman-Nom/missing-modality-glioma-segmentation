# 0005. Modality dropout (E3) and presence indicator (E4): verify

How `/check verify` proves each acceptance criterion in [index.md](index.md). The AI may run the fast checks. The human runs the slow ones and pastes the last two lines of each.

All commands run from the repo root with `UV_PROJECT_ENVIRONMENT` and `UV_LINK_MODE=copy` set (ADR 0001).

## Fast (AI may run, no GPU, no real data)

    uv run pytest -q
    uv run ruff check .

Expected: all pass, `slow` deselected, ruff clean. Tests live in `tests/test_common_strategies.py`, `tests/test_strategy_runs.py` and `tests/test_strategy_compare.py`.

| AC | Proof |
|---|---|
| AC-1 | `e3.toml` and `e4.toml` minus `[strategy]` equal the E1 config, dict against dict |
| AC-2 | Same seed gives the same draws for both strategies; 15,000 draws hit every value from 1 to 15 and none outside; a quick run's first patches are the same with and without a drawing hook |
| AC-3 | On a batched `(B, 4, X, Y, Z)` `MetaTensor`, the channels of the drawn pattern are exactly 0, the others equal the input, `label` and `wt` unchanged |
| AC-4 | Pattern 6 gives presence 0, 0, 1, 1; shape `(8, X, Y, Z)`, float32; NumPy and torch equal; `presence_hook` uses the pattern the sample drew |
| AC-5 | A padded input gives logits equal to the bit to constant presence; no `[strategy]` gives the same state dict keys and shapes as today; a saved 4 channel `best.pt` shape loads |
| AC-6 | `name = "x"` exits 2 on a new run and on `--resume` |
| AC-7 | Quick E1 `loss` column equal on CPU between the commit before this feature and the current one (run once at build time, result in `docs/whats-built.md`); E1 `config_hash` still starts `f9b3afb6`; a quick E4 run passes a val point; an E4 run resumes to `ok` |
| AC-8 | `val_patterns` default and `selection_score` unchanged; a quick E3 `run.json` has one pattern in `val_log.csv` |
| AC-9 | A quick E4 run gets a `metrics_val.csv` with n × 3 rows; a quick E1 `metrics_val.csv` is byte equal between the commit before this feature and the current one (run once at build time) |
| AC-10 | Quick sweep on E3 and E4; a recording hook on E4 sees 8 channels at `predict()` |
| AC-11 | `protocol_hash()` equals `146635cf4521a239c4e8cba877528b4c688a6410cd3b817df671db4f1f57825b` |
| AC-12 | Toy run folders, one field changed at a time, each exit 1; `versions` differing warns and passes; pattern counts add up to `iterations` × `batch_size`; the number of resumes is printed |
| AC-15 | One test per guard, each exit 2 and no folder written |
| AC-16 | Missing val compare, other run ids, a stale `compare_hash`, each exit 2; differing `versions` sets `versions_equal` false and still writes; each of the four files changes the hash |
| AC-17 | 72 and 270 rows, exact columns and order; a toy case with a known `mean_diff`, Holm and CI; all zero `d` gives p 1; two runs byte identical; a finished folder refused, then accepted with a reason, old folder kept |
| AC-18 | Toy CSVs reaching supported and not supported for a positive and a negative `mean_diff`, compared exactly with the template |
| AC-19 | Captions and header comment present; val and test wording; each refusal leaves older `.tex` byte equal; the stub inputs the three files |
| AC-22 | `test_gate_opens_on_the_joint_tar_layout` in `tests/test_missing_eval.py`: only the four run files of the tar, with the val eval moved to another folder, open the gate, and the test eval passes `check_rules` |
| AC-23 | The folder set in `tests/test_experiment_scripts.py` holds `0005-strategies`; `common.strategies`, `common.model` and `common.training` each import first in a fresh process |

## Slow (the human runs these, no AI polling)

**AC-19 compile.** After a toy comparison has written the three `.tex` files:

    cd writing/tracer && latexmk -xelatex e3e4-preview.tex

**AC-13 real runs**, on Kaggle by `experiments/0005-strategies/KAGGLE.md`, once per strategy:

    uv run python experiments/0003-baseline/train.py --config experiments/0005-strategies/<e3 or e4>.toml --out <runs>
    uv run python experiments/0003-baseline/evaluate.py --run <run>
    uv run python experiments/0004-missing-eval/evaluate.py --run <run> --fold val --out <evals>
    uv run python experiments/0003-baseline/check_baseline.py <run>
    uv run python experiments/0005-strategies/check_strategy.py <run> --e1 <E1 run>
    uv run python experiments/0004-missing-eval/check_eval.py <eval dir>

Expected: each check prints `PASS`.

**AC-15 to AC-18 real val comparison**, on the laptop, no GPU:

    uv run python experiments/0005-strategies/compare.py --e1-run <R1> --e3-run <R3> --e4-run <R4> --fold val
    uv run python experiments/0005-strategies/make_table.py --compare results/0005-strategies/compare-val

Expected: 9 H2 lines in `verdict.txt`, three tables, captions saying development fold.

**AC-20, AC-21 joint opening**, on Kaggle by `KAGGLE.md`, with E1, E3 and E4 in turn:

    uv run python experiments/0003-baseline/preprocess.py --folds test
    uv run python experiments/0004-missing-eval/evaluate.py --run <run> --fold test --open-test --out <evals>
    uv run python experiments/0004-missing-eval/check_eval.py <test eval dir>

Then, on the E1 test eval only, `analyse.py` and `make_table.py`, and then:

    uv run python experiments/0005-strategies/compare.py --e1-run <R1> --e3-run <R3> --e4-run <R4> --fold test
    uv run python experiments/0005-strategies/make_table.py --compare results/0005-strategies/compare-test

Expected: every check passes, and the opening is recorded by hand.

## By reading

| AC | Check |
|---|---|
| AC-14 | The tracker shows one run per strategy, or a logged reason for any repeat |
| AC-20 | Commit order: the val compare is recorded before the test evals start; the test evals name the pinned commit |
| AC-21 | The tracker records date, commit and the three eval ids |
| AC-22 | `KAGGLE.md` has the `R1`, `R3`, `R4` variables and the *Joint test layout* paths |
| AC-22 | In `KAGGLE.md`, every notebook cell that scores or checks runs through `sh()`, not a plain `!` line. Run once in a real executed notebook, a failed command must stop the notebook before the next cell (done 2026-11-06 with nbconvert, not on Kaggle) |
| AC-22 | The step 6 tar lists `runs/`, `evals-e1/` and `evals/` as in *Joint test layout*, with no `last.pt` or `train_log.csv`, and the upload folder holds only the tar |
| AC-22 | `KAGGLE.md`, ADR 0002, `data/DATA.md`, `experiments/AGENTS.md` and the dated notes in specs 0003 and 0004 updated |
