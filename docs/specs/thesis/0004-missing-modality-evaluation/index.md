# 0004. Missing modality evaluation (E2): score the frozen E1 model on all 15 patterns

**Date**: 2026-11-04
**Status**: In Progress

## Summary

E2 measures how far the E1 model falls when MRI sequences are missing (RQ1). It scores all 15 modality patterns on the 114 val subjects now, and uses those numbers to design E3 and E4. The 228 test subjects stay sealed: this spec builds and proves the test path on synthetic cases, and the real test sweep happens once, for E1, E3 and E4 together, under feature 7. This document is also the written protocol, so the statistics below cannot change after numbers are seen.

## Requirements

**User stories**:
- As the researcher, I want every modality pattern scored on the same subjects so that the drop in RQ1 is a paired, fair measurement.
- As the researcher, I want the test fold sealed behind a code gate so that it cannot be opened by accident, and every opening is logged.
- As the researcher, I want one shared sweep and one statistics module so that E3 and E4 are scored by exactly the same code.

**Acceptance criteria**:
- **AC-1 Case lists.** With `--fold val`, cases are `val_cases` from the E1 `run.json` (114). With `--fold test`, cases are `fold_cases(splits_csv, "test", timepoints=[eval].test_timepoints, exclude_flair_flag=[data].exclude_flair_flag)` (228, one per subject). Before any GPU work, `evaluate.py` stops with exit 2 if a chosen case shares a subject with the run's `train_cases`, or a test case shares a subject with its `val_cases`.
- **AC-2 Source guards.** `evaluate.py` applies the guards of `0003-baseline/evaluate.py`: exit 1 if the E1 run status is not `ok`, `best.pt` is missing, or its config hash differs from `run.json`; exit 2 if `check_cache(cfg, cases, allowed_folds=(fold,))` fails. Model, `[infer]` and `[data]` settings come only from the E1 run's `config.toml`. The E1 run folder is read only.
- **AC-3 Sweep.** For each case in sorted order and each pattern ID 1 to 15 ascending, the sweep applies `apply_pattern()`, then `image_hook`, predicts with `common.model.predict()` at the fixed threshold 0.5, and scores WT, TC and ET with `region_metrics(..., with_hd95=True)`. `metrics.csv` holds exactly n × 15 × 3 rows (5,130 for val, 10,260 for test), with the columns in *Data model*. Any NaN in `dice` or `hd95_mm` sets status `failed` and exit 1.
- **AC-4 Shared code.** The sweep, the case lists and the test gate live in `common/evaluation.py`, with `image_hook` defaulting to no change, so E3 and E4 reuse them. A test proves the hook runs after `apply_pattern()`.
- **AC-5 Resume.** After each case and pattern pair, its 3 rows are flushed and synced to disk. The same command then resumes: it drops any truncated line and any incomplete pair, skips finished pairs, and appends one entry to `resumes` in `run.json` (`utc`, `done_pairs`, `git_commit`, `previous_status`). `predict_seconds`, `metric_seconds` and `wall_time_s` add up across resumes, and `peak_vram_reserved_mib` keeps the maximum. A resume stops with exit 2 if `git_commit`, `config_hash`, `protocol_hash` or the source run differ from `run.json`. The resumed `metrics.csv` equals an uninterrupted one, byte for byte, on the same machine.
- **AC-6 One folder, one finished run.** The eval folder is `<out>/<source run id>-<fold>/` (with `-quick` added for quick runs). If that folder has status `ok`, `evaluate.py` stops with exit 2 unless `--rerun-reason "<text>"` is given. With a reason, the old folder is renamed `<name>.superseded-<utc time>`, and the new `run.json` records `rerun_reason` and `superseded`.
- **AC-7 Test gate.** The gate runs first, before the case list is built and before any `.npz` is opened. `--fold test` stops with exit 2, printing every failed rule, unless all hold: `--open-test` is given, `--quick` is not, `git_dirty` is exactly false (`None` from `git_info()` counts as dirty), the E1 run has `quick` false, a `<source run id>-val` eval folder in the same `--out` has status `ok` and passes `check_rules()` at that moment (the gate reruns the rules, so no pass flag can go stale), and the current `protocol_hash` equals that val eval's. On a mismatch the fix is to rerun val with `--rerun-reason`. `--open-test` with `--fold val` is also exit 2. Rule AC-6 then limits test to one finished run per E1 run, unless a reason is logged. This gate stops accidents. It is not proof, because deleting a folder resets it, so each real opening is also recorded by hand in `context/progress-tracker.md`.
- **AC-8 No real test run in this feature.** Feature 6 proves the test path only on synthetic cases. `preprocess.py --folds test` is not run on real data, and no real test number exists when feature 6 closes. The real sweep is a task of feature 7.
- **AC-9 Tie out to E1.** On val, `check_rules()` compares pattern 1 with the E1 `metrics_val.csv`: `gpu` in the eval `run.json` equals `gpu` in the E1 `run.json`, the case set is equal, mean Dice per region differs by at most 0.001, and each (case, region) Dice by at most 0.01. A failure means the cache or the inference drifted, and no E2 number is trusted.
- **AC-10 Run record.** `run.json` holds the keys in *Data model*, including `protocol_hash`, `predict_seconds` and `metric_seconds`, so the real HD95 cost is measured. Status moves as in *State transitions*. Any exception sets `failed` with a `reason`, and exit 1.
- **AC-11 Statistics module.** `common/stats.py` provides `paired_wilcoxon`, `holm`, `friedman`, `bootstrap_indices` and `bootstrap_ci` with the exact rules in *Statistics protocol*, each tested on a toy case with a known answer, including `holm` on a textbook example and all zero differences.
- **AC-12 Analysis.** `analyse.py --eval <dir>` reads only `metrics.csv`, `config.toml` and `run.json`. It runs four checks in this order, and stops at the first that fails with `SystemExit("ERROR: ...")`, which is exit 1: `run.json` is present; its `status` is `ok`; no (case, pattern, region) row is missing; the number of cases in `metrics.csv` equals `n_cases` in `run.json`. A `status` or `n_cases` that is absent or null counts as a failed check. A corrupt `run.json` may stay a plain traceback, which also exits 1. A refusal writes no output file and leaves any older one untouched. The `status` check is needed because a sweep that stops between two cases leaves every remaining row whole. A quick run with status `ok` is accepted, because AC-15 needs it. Otherwise it writes `stats.csv` (90 rows), `friedman.csv` (18 rows) and `marginal.csv` (24 rows). Two runs give byte identical files. No GPU is used.
- **AC-13 Verdict.** `analyse.py` prints and stores in `verdict.txt` the H1 result under the decision rule in *Statistics protocol*, using exactly the *Verdict lines* template, with no human judgement.
- **AC-14 Tables and heatmap.** `make_table.py --eval <dir>` writes four LaTeX files to `writing/tables/`: `0004-missing-eval-<fold>-dice.tex`, `-hd95.tex`, `-marginal.tex` and `-heatmap.tex` (PGFPlots), laid out as in *Table layouts*. They read only the analysis CSVs, `run.json`, and `alpha` from `config.toml` in the eval folder (the same value `analyse.py` uses for the verdict). Each carries a header comment with run id, config hash, git commit and seed, and a caption with n, seed and fold. A val caption says: `Development fold, not a test result. The checkpoint was chosen on this fold at complete input, so pattern 1 is slightly optimistic.` A test caption says: `Test fold, opened once.` All four compile under XeLaTeX through the committed stub `writing/tracer/e2-preview.tex`, with no new package install.
- **AC-15 Determinism.** Two `--quick` runs with the same seed, in two different `--out` folders (AC-6 refuses a second run in one), give byte equal `metrics.csv` and analysis files on CPU. `evaluate.py` calls `set_seed([eval].seed)` before loading the model.
- **AC-16 Check.** `check_eval.py <eval dir> [--run <E1 run>]` is a thin wrapper around `common.evaluation.check_rules(eval_dir, source_run)`. `--run` defaults to `source_run` in `run.json`, and is needed when the eval folder moved between machines. It exits 1 on any failed rule: status `ok`, `quick` false, row count, every case and pattern present, no NaN, AC-9 on val, and on test a `protocol_hash` equal to the val eval's. It prints the reason for each failure. On val it also prints the projected test time: `(predict_seconds + metric_seconds) * 2 / 3600` hours, because test has exactly twice the cases.
- **AC-17 Kaggle and records.** `experiments/0004-missing-eval/KAGGLE.md` gives the full val run and the session death steps. It pins the exact git commit at clone, so a resume in a new session matches `git_commit`. It also covers the rebuild of the val cache, and copying the val eval folder into the `--out` folder for a later test run, because `/kaggle/input` is read only. Any copy of an eval folder keeps `run.json`, because `analyse.py` refuses a folder without it. ADR 0002 records that E2 also runs on the T4, and that `best.pt` sits in a private Kaggle dataset. `data/DATA.md` lists that dataset. `experiments/AGENTS.md` maps E2 to `0004-missing-eval`.
- **AC-18 The real val run.** The human runs the val sweep on the Kaggle T4 for seed 42. `check_eval.py` passes, and the four tables compile. Timings go in `run.json`, not guessed.
- **AC-19 Scope reword.** Feature 6's done line says val scored and test path proven. Feature 7's says it owns the single joint test opening of E1, E3 and E4.

## Decision

**Chosen option**: Option 2: val first, one sealed joint test opening, shared sweep and statistics in `common/`.

E2 scores the frozen E1 `best.pt` on all 15 patterns of val on the Kaggle T4, reports them as development numbers, and leaves a gated, unused test path for the joint opening under feature 7.

**Implementation skills**: `pytorch-patterns` (`affaan-m/ecc`, `.claude/skills/pytorch-patterns/`) · `python-testing-patterns` (`wshobson/agents`, `.claude/skills/python-testing-patterns/`) · `scipy-best-practices` (`mindrally/skills`, `.claude/skills/scipy-best-practices/`)

## Rationale

Reasoning and options: see [rationale.md](rationale.md).

## Feature design

### Statistics protocol (fixed before any test number)

The unit is the subject. Each subject has one case (timepoint 000). The per case paired difference is `d = value(pattern p) - value(pattern 1)`. For Dice a negative `d` is worse; for HD95 a positive `d` is worse.

| Item | Rule |
|---|---|
| Primary metric | Dice, WT, TC and ET separately |
| Secondary metric | HD95, including the 373.13 mm empty rule, same tests, own families |
| Patterns tested | 14 incomplete patterns (2 to 15), each against pattern 1 |
| Test | `scipy.stats.wilcoxon(d, zero_method="wilcox", correction=False, alternative="two-sided", method="asymptotic")`. All `d` zero gives statistic 0, p 1 |
| Multiple testing | Holm over the 14 patterns, one family per (metric, region). Alpha 0.05 |
| Which modality matters | `scipy.stats.friedmanchisquare` per (metric, region) within each group of equal size: `n_missing` 1 (4 patterns), 2 (6) and 3 (4), so the test compares which modalities are missing, not how many. Holm over the 3 groups of a (metric, region). Kendall W = statistic / (n × (k - 1)), with k patterns in the group (an effect size from 0 to 1). All rows fully tied would give NaN in SciPy, so this case is coded as statistic 0, p 1, W 0 |
| Marginal effect | For modality m, take each of the 7 non empty subsets S of the other three modalities, and form the matched pair `value(S + m) - value(S)`. Each case's value is the mean of its 7 pair differences, so each pair differs by exactly one modality. `S + m` is pattern 1 for one pair. Descriptive, with a bootstrap CI, no test |
| Confidence interval | 95% percentile bootstrap of the mean over subjects, 10,000 resamples. One integer index matrix of shape (resamples, n) from `numpy.random.default_rng([eval].seed)` is reused for every row, so all rows are paired. `np.percentile` at 2.5 and 97.5, linear. No CI on medians |
| Means versus medians | Dice leads with the mean. HD95 leads with the median, because the empty rule can swamp a mean. Both are stored, with empty counts |
| Cases kept | Every case, including 7 empty ET and 1 empty TC in test, and the one test case with `neg_values_flag`. No exclusions after the fact |

**H1 decision rule.**
- First half, per region: `supported` if all 14 patterns have Dice `mean_diff < 0` and Holm p below 0.05; `not supported` otherwise, naming the patterns that fail. The test is two sided, so the direction comes from the sign of `mean_diff`.
- Second half, per region: `supported` if the Dice Friedman Holm p in the `n_missing` 1 group is below 0.05. That group is the literal question, which modality is missing. Groups 2 and 3 are reported only.

**Verdict lines.** One line each, regions in order WT, TC, ET, first all `H1a`, then all `H1b`:

```text
H1a WT: supported
H1a TC: not supported (patterns 7, 12)
H1b WT: supported (n_missing=1, p_holm=1.2e-05)
H1b ET: not supported (n_missing=1, p_holm=0.031)
```

These lines show the format only, and hold no results. Pattern IDs ascend, comma separated. `p_holm` uses `%.6g`.

A statistically clear drop is not the same as a large one, so the thesis always reports the size next to the p value.

### Data model sketch

Nothing is committed: `results/*` and `writing/tables/` are ignored by git.

| Entity | Key | Fields | Rows |
|---|---|---|---|
| Eval folder `results/0004-missing-eval/<source run id>-<fold>/` | source run + fold | `config.toml`, `run.json`, `metrics.csv`, `stats.csv`, `friedman.csv`, `marginal.csv`, `verdict.txt` | 1 per E1 run and fold |
| `metrics.csv` | case, pattern_id, region | `eval_id, case, subject, pattern_id, pattern_name, region, dice, hd95_mm, sensitivity, precision, gt_empty, pred_empty`. `sensitivity` and `precision` may be empty under the rule of spec 0001 | n × 15 × 3 |
| `stats.csv` | metric, region, pattern_id | `metric` (dice or hd95_mm), `region, pattern_id, pattern_name, n, mean, median, mean_diff, ci_low, ci_high, n_nonzero, wilcoxon_stat, p_raw, p_holm, n_pred_empty`. `mean` and `median` describe the values. `mean_diff` is the mean of `d`, and `ci_low` and `ci_high` bound it. `n_nonzero` counts non zero `d`. Pattern 1 leaves the 7 fields from `mean_diff` to `p_holm` empty, and keeps `n_pred_empty` | 90 |
| `friedman.csv` | metric, region, n_missing | `metric, region, n_missing, n, statistic, p_raw, p_holm, kendall_w` | 2 × 3 × 3 = 18 |
| `marginal.csv` | metric, region, modality | `metric, region, modality, mean_present, mean_absent, mean_diff, ci_low, ci_high`. `mean_present` is over the 7 `S + m` patterns, `mean_absent` over the 7 `S` patterns, `mean_diff` the mean of the pair differences, and the CI bounds it | 24 |

`run.json` keys: `eval_id, status, reason, quick, fold, n_cases, source_run, source_run_id, source_config_hash, source_best_iteration, source_seed, splits_sha256, protocol_hash, config_hash, git_commit, git_dirty, seed, device, gpu, versions, rerun_reason, superseded, resumes, predict_seconds, metric_seconds, wall_time_s, peak_vram_reserved_mib`.

`protocol_hash` is the sha256 over the bytes, in this order, of: `experiments/0004-missing-eval/config.toml`, `experiments/0004-missing-eval/analyse.py`, `experiments/common/evaluation.py`, `metrics.py`, `missing.py`, `stats.py`. `model.py` is left out on purpose, because E4 will change the input channels.

**File formats.** All CSVs use `lineterminator="\n"`. Floats are written `%.6f`, p values `%.6g`. A missing value is an empty field. `metrics.csv` writes floats with `repr()` so nothing is rounded.

`experiments/0004-missing-eval/config.toml` (the eval folder holds a copy of this `[eval]` section with one added key, `quick`, and `config_hash` is `config_hash()` of that; E1 settings stay with the source run):

```toml
[eval]
seed = 42
test_timepoints = [0]
bootstrap_resamples = 10000
alpha = 0.05
```

### Table layouts

All tables use `booktabs` inside `\resizebox{\linewidth}{!}{...}`, with Dice at 3 decimals and HD95 at 1.

| File | Rows | Columns |
|---|---|---|
| `-dice.tex` | 15 patterns | ID, present modalities, then per region: mean, `mean_diff`, CI as `[low, high]`, with a star if `p_holm` is below alpha |
| `-hd95.tex` | 15 patterns | ID, present modalities, then per region: median, mean, `n_pred_empty`, with a star if `p_holm` is below alpha |
| `-marginal.tex` | 4 modalities × 2 metrics | metric, modality, then per region: `mean_diff`, CI |
| `-heatmap.tex` | 15 × 3 cells | PGFPlots `matrix plot*` of Dice `mean_diff`, patterns on the y axis, regions on the x axis, values printed in each cell, greyscale so it survives printing, scale from the lowest value to 0 |

`writing/tracer/e2-preview.tex` is a small committed stub. It loads `fontspec`, `booktabs`, `graphicx` and `pgfplots` (`compat=1.18`) and inputs the four files.

### State transitions

Eval status: `running → ok`, `running → failed`, `running or failed → running` (resume), `ok → renamed superseded` (only through `--rerun-reason`). A killed Kaggle session leaves `running`, which resumes the same way.

### API surface (scripts, run from the repo root)

| Command | Key inputs | Key outputs | Gate | Key errors |
|---|---|---|---|---|
| `experiments/0004-missing-eval/evaluate.py` | `--run` (req), `--fold val or test` (req), `--out` (default `results/0004-missing-eval`), `--config`, `--device auto or cpu`, `--quick`, `--open-test`, `--rerun-reason` | eval folder, `metrics.csv`, `run.json` | AC-7 for test | 1 source or run failure; 2 guard, gate, finished folder, resume mismatch |
| `.../analyse.py` | `--eval` (req) | `stats.csv`, `friedman.csv`, `marginal.csv`, `verdict.txt` | none | 1 `run.json` missing, status not `ok`, a missing row, or a case count that differs from `n_cases` |
| `.../make_table.py` | `--eval` (req), `--out-dir` (default `writing/tables`) | four `.tex` files | none | 1 analysis CSV missing or wrong row count |
| `.../check_eval.py` | eval dir (positional), `--run` | pass or the failures | none | 1 any rule failed |

`common/evaluation.py`:
- `eval_cases(cfg, eval_cfg, rec, fold)`, returns the AC-1 list after the subject checks.
- `check_test_gate(fold, open_test, quick, rec, git, val_dir, run_dir, protocol)`, returns error strings. It takes plain values, not E2's parsed arguments, so feature 7 can reuse it for E3 and E4.
- `check_rules(eval_dir, source_run)`, returns error strings, shared by `check_eval.py` and the gate.
- `sweep(model, cases, cfg, device, writer, patterns=range(1, 16), done=frozenset(), image_hook=identity)`, returns `{"predict_seconds", "metric_seconds"}` and raises on NaN. `writer(rows)` takes the 3 row dicts of one pair and owns the flush and sync. `done` is a set of `(case, pattern_id)`. `image_hook(image, pattern_id)` returns an image.

`run.json` uses the run record class in `common/runlog.py` with explicit `keys` and `defaults` (otherwise the tracer defaults get added). A resume follows `common/training.py` at the `RunRecord(...)` resume call: `defaults` are the old fields without `status`.

`common/stats.py`:
- `paired_wilcoxon(d)`, returns `(stat, p)`.
- `holm(p)`, returns adjusted values, monotone and capped at 1.
- `friedman(matrix)`, takes n × k, returns `(stat, p, w)`.
- `bootstrap_indices(n, resamples, seed)`, returns an integer array (resamples, n).
- `bootstrap_ci(values, indices, level=0.95)`, returns `(low, high)` of the mean.

Holm is about 6 lines here. SciPy has no Holm function, and `statsmodels` is not in `uv.lock`.

`--quick` sets `quick = true` before hashing, and uses the first 2 cases of the fold, with the device as given and an output folder ending in `-quick`. All 15 patterns still run. `EVAL_STOP_AFTER=N` is a test only variable that stops after N pairs, in the style of `BASELINE_INJECT_INF`.

### Value sourcing

| Action | Value | Source |
|---|---|---|
| evaluate | model weights | `<run>/checkpoints/best.pt` |
| evaluate | model, infer, data settings, cache dir | `<run>/config.toml` |
| evaluate | val case list | `run.json` `val_cases` |
| evaluate | test case list | `fold_cases()` on the config's `splits_csv`, `[eval].test_timepoints`, `[data].exclude_flair_flag` |
| evaluate | missing channel value | `apply_pattern()`: zero, the brain mean after normalisation (spec 0001) |
| evaluate | threshold 0.5 | `[infer].threshold` of the E1 config, never tuned |
| evaluate | HD95 empty value | `HD95_EMPTY_MM` in `common/metrics.py` |
| evaluate | `subject` | `subject_of(case)` |
| evaluate | `pattern_name` | `PATTERNS[id].name` |
| evaluate | `git_commit`, `git_dirty` | `git_info()` |
| evaluate | `splits_sha256` | sha256 of the `splits_csv` file |
| evaluate | `eval_id` | the eval folder name |
| evaluate | `seed`, `source_seed` | `[eval].seed` (used by `set_seed`); the E1 `run.json` `seed` (information only) |
| evaluate | `device`, `gpu` | `torch.device` as chosen; `torch.cuda.get_device_name()`, else null |
| evaluate | `peak_vram_reserved_mib` | `torch.cuda.max_memory_reserved()`, else null |
| evaluate | `wall_time_s` | `time.time()` around the sweep, added across resumes |
| evaluate | `reason` | the exception text on failure |
| analyse | bootstrap seed, resamples, alpha | `[eval]` in the eval folder's `config.toml` |
| analyse | `status`, `n_cases` for the guards | `run.json` in the eval folder |
| analyse | `d` | `metrics.csv`, pattern p minus pattern 1, matched on case and region |
| analyse | `n_pred_empty` | `metrics.csv` `pred_empty` count |
| check | E1 Dice for the tie out | `<run>/metrics_val.csv`, from `--run`, else `source_run` in `run.json` |
| make_table | captions | `run.json` plus `n` in `stats.csv` |
| make_table | `alpha` for the star | `[eval]` in the eval folder's `config.toml` |

### Key invariants

- Every pattern scores the same cases. `analyse.py` refuses gaps, whole missing cases included, and any eval that is not status `ok`.
- No script reads a test case without passing `check_test_gate()`.
- Pattern 1 goes through the same path as all others, so complete input is not special cased.
- A row in `metrics.csv` depends only on the case, the pattern and the weights, so resume cannot change a value.
- E2 tunes nothing: no threshold, no postprocessing, no pattern selection.
- A test run starts only if the current `protocol_hash` equals the val eval's, checked before any GPU work. If the protocol must change, even by formatting, val is rerun with `--rerun-reason`, so any change leaves a trace.
- E2 keeps the `[preprocess]` settings of E1 unchanged, so the rebuilt cache has the same preprocess hash.
- `0003-baseline/evaluate.py` stays unchanged and still reads no test case.

### Security model

Single researcher, no accounts. Outputs hold numbers and case IDs only, never images. The Kaggle dataset holding `best.pt` stays private. BraTS data stays under its terms in `data/DATA.md`.

### Configuration required

- `BRATS_TRAIN_DIR`: the existing variable, used by `preprocess.py`.
- A new private Kaggle dataset with the E1 run folder: `config.toml`, `run.json`, `metrics_val.csv`, `checkpoints/best.pt`. Upload before the val run.

### Critical test scenarios

- Happy path: synthetic E1 run, `--quick` on val, then `analyse.py`, `make_table.py`: 2 × 15 × 3 rows and all files, verifies **AC-3**, **AC-12**, **AC-14**.
- Died sweep: a hand built toy folder (not `EVAL_STOP_AFTER`, which stops between cases only at a multiple of 15) stopped between two cases, so every present case is whole, is refused with exit 1 and no output file while status is `running`; a `metrics.csv` with fewer cases than `n_cases` is refused under status `ok`, verifies **AC-12**.
- Resume: `EVAL_STOP_AFTER=7` plus a truncated last line, then rerun: equal to an uninterrupted run, verifies **AC-5**.
- Test gate: each missing condition refused with exit 2 and no test case loaded, then the same call passes on a synthetic non quick run with a clean `git_info`, verifies **AC-7**.
- Second run: a finished folder refused, then accepted with a reason, and the old folder kept, verifies **AC-6**.
- Leak: a test case sharing a subject with a train case stops with exit 2, verifies **AC-1**.
- Statistics: `holm` on a hand worked example, all zero `d`, a Friedman matrix with a known result, a bootstrap CI on a constant, verifies **AC-11**.

## Build plan

Tracer Bullet: one thin path first, then thicken.

1. Thin thread: `config.toml`, `common/evaluation.py` (`eval_cases`, `sweep`), `evaluate.py` with AC-2 guards, `paired_wilcoxon` and `holm`, `analyse.py` for Dice only, the Dice table, one quick run on synthetic cases, satisfies **AC-1**, **AC-2**, **AC-3**, **AC-4**, **AC-15**
2. Run record and folder rule: `run.json` keys, status, timings, `protocol_hash`, `--rerun-reason`, satisfies **AC-6**, **AC-10**
3. Resume: flush and sync per pair, truncation cleanup, mismatch refusal, `EVAL_STOP_AFTER`, satisfies **AC-5**
4. Test gate: `check_test_gate`, `--open-test`, synthetic test path, satisfies **AC-7**, **AC-8**
5. Full statistics: HD95, Friedman, marginal, bootstrap, `verdict.txt`, satisfies **AC-11**, **AC-12**, **AC-13**
6. Remaining tables and heatmap, XeLaTeX compile, satisfies **AC-14**
7. `check_eval.py` with the tie out, satisfies **AC-9**, **AC-16**
8. Analysis guards: `analyse.py` reads `run.json`, runs the AC-12 checks in order, and lines 4 and 5 of its docstring say so. The `make_table.py` docstring names `run.json` and `alpha` from `config.toml`. `write_toy_metrics` also writes a `run.json` and takes `status` and `n_cases`, so its callers stay short, and `test_analyse_differences_are_matched_on_case` gets a `run.json` by hand, because it builds its own folder. A test proves the star follows `alpha` in the eval folder's `config.toml` (no code change for that). Build this before step 10, because `analyse.py` is in `protocol_hash`, satisfies **AC-12**, **AC-14**
9. `KAGGLE.md`, ADR 0002, `data/DATA.md`, `experiments/AGENTS.md`, scope rewording, `docs/whats-built.md` (including the sentence that says `analyse.py` reads only `metrics.csv` and `config.toml`), tracker, satisfies **AC-17**, **AC-19**
10. Human: upload the E1 run, run the val sweep, analyse, tables, check, satisfies **AC-18**

## Consequences

**Positive**:
- The design of E3 and E4 can use real drop numbers without touching test.
- E1, E3 and E4 share one sweep, one statistics module and one sealed opening, so RQ2 stays paired.
- The real HD95 cost gets measured on val before test is spent.

**Negative / tradeoffs**:
- The RQ1 thesis numbers stay val only until feature 7, so the RQ1 chapter stays provisional for now.
- Val numbers are slightly optimistic at pattern 1, because the checkpoint was chosen there.
- A `git_dirty` rule means the test sweep must run from a clean commit.
- A byte change to any protocol file costs a val rerun before test.
- The test gate stops accidents, but cannot prove a single opening, so that record also depends on your honesty and the tracker.
- Kaggle now also holds model weights, and uses more GPU quota.
- Holm per region is less strict than one family of 42 tests.
- One seed means the results describe variation between subjects, not between trained models.
- `analyse.py` cannot give a partial reading of a running sweep. That is on purpose: a verdict on part of a fold is not a result.

**Neutral**:
- Proposal wording grows slightly: Friedman is new, and HD95 is named secondary.
- Feature 11 keeps error analysis, and no longer owns the core tests.

## Follow-up

- [ ] Feature 7 spec: state that E3 and E4 are chosen on val only, run the joint test opening with `--open-test`, decide selection through `val_patterns`, decide which script runs E3 and E4 (each needs its own `-val` folder), and plan around the test time projected by `check_eval.py`.
- [ ] Record the Kaggle quota and session limit in ADR 0002 (already an open item there). E1's docs use 9 h as a working cap.
- [ ] Update the proposal text to name Friedman and the secondary role of HD95. The proposal has uncommitted edits, so this is left to you.
- [ ] Add verified literature notes for Wilcoxon, Holm, Friedman and the bootstrap before the thesis cites them. Until then write `[CITE?]`.
- [ ] Seeds 43 and 44, if GPU time allows: rerun E2 per seed. Combining seeds needs its own decision.
