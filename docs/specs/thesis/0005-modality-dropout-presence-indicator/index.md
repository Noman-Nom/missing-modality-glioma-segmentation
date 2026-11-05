# 0005. Modality dropout (E3) and presence indicator (E4): two retrained models, one paired comparison, one sealed test opening

**Date**: 2026-11-05
**Status**: In Progress

## Summary

E3 and E4 retrain the E1 model with the same data, budget and seed, and change one thing each. E3 hides a random set of MRI sequences on every training step (modality dropout). E4 does the same and also tells the network which sequences are present, through four extra input channels (presence indicators). Both are scored by the E2 sweep on all 15 patterns, and a new paired comparison against E1 answers RQ2. This feature also runs the single joint opening of the test fold for E1, E3 and E4.

## Requirements

**User stories**:
- As the researcher, I want E3 and E4 to differ from E1 only in the strategy, so that any change in Dice comes from the strategy and nothing else.
- As the researcher, I want one fixed paired comparison, written before any test number exists, so that RQ2 has one honest answer.
- As the researcher, I want one joint test opening for all three models, so that no model is scored on test alone or chosen after seeing test.

**Acceptance criteria**:

*Strategies*

- **AC-1 Configs differ only in the strategy.** `experiments/0005-strategies/e3.toml` and `e4.toml` are full copies of `experiments/0003-baseline/config.toml` plus one section: `[strategy]` with `name = "dropout"` and `name = "presence"`. A test loads each one, removes `[strategy]`, and asserts it equals the E1 config, comment free dict against dict. So iterations (40,000), val cadence, patch, lr, grad clip, `amp = false`, cache dir, `[preprocess]` and seed 42 all match E1.
- **AC-2 The draw.** `draw_patterns(rng, n)` in `common/strategies.py` returns `rng.integers(1, len(PATTERNS) + 1, size=n)`, so every sample gets one of the 15 patterns with equal chance, complete input included. This is the only sampling rule, with no parameter in the config. It is drawn once per batch from `Trainer.hook_rng`, seed + 1, so the patches a seed gives do not change (spec 0003, AC-17). E3 and E4 at one seed get the same pattern sequence. A test proves both.
- **AC-3 E3 hook.** `dropout_hook(batch, rng)` returns a new batch whose `image` has, for each sample, the channels of its drawn pattern set to exactly 0, the same zero E2 uses. `batch["image"]` is `(B, 4, X, Y, Z)` on the training device, so the hook loops over samples and calls `apply_pattern(batch["image"][i], pattern_i)`, because `apply_pattern()` on the whole batch with `channel_axis=0` raises `IndexError`, and `missing.py` is sealed. Tests use that batched shape, as a MONAI `MetaTensor` as the real loader gives it. `label` and `wt` are unchanged. It runs after augmentation (`Trainer.step` already calls it there), so no later step can change a zero.
- **AC-4 E4 hook.** `append_presence(image, pattern_id)` takes `(4, X, Y, Z)` NumPy or torch and returns `(8, X, Y, Z)` float32 of the same kind: the 4 image channels, then 4 presence channels in `MODALITIES` order, each all 1.0 if that modality is present in the pattern and all 0.0 if not. `presence_hook(batch, rng)` is `dropout_hook` followed by `append_presence` per sample, with the pattern that sample drew. The image hook at scoring time is `presence_image_hook(image, pattern_id)`, which calls `append_presence`. A presence channel is never random, so E4 trains with dropout: without it all four channels would be 1 on every step and the network would learn nothing from them.
- **AC-5 E4 model.** `build_model(cfg)` builds a `UNet` with 8 input channels wrapped in `PresenceUNet` when `strategy_name(cfg) == "presence"`, and the same 4 channel `UNet` as today otherwise. All other settings match E1. `PresenceUNet.forward(x)` replaces the 4 presence channels of `x` by their maximum over the spatial axes of that patch, spread back to the full patch, then calls the `UNet`. Reason: `predict()` pads with 0 when a case is smaller than the window, and a 0 there would read as missing. With this step a padded patch gives the same logits as the same patch with constant presence channels, and a test proves it to the bit. An E1 run has no `[strategy]`, so `build_model` gives the same parameters as today and the E1 `best.pt` still loads.
- **AC-6 Strategy lookup.** `strategy_name(cfg)` returns `cfg.get("strategy", {}).get("name")`. `strategy_hooks(cfg)` returns `{"batch_hook": ..., "image_hook": ...}`: for no strategy, the no change pair; for `dropout`, `dropout_hook` and the no change image hook; for `presence`, `presence_hook` and `presence_image_hook`. `check_config()` in `common/training.py` adds a problem string for any other name, so `train.py` exits 2 before training, on start and on `--resume`.
- **AC-7 One training path.** E3 and E4 run through `experiments/0003-baseline/train.py --config <e3 or e4 toml> --out results/0005-strategies/runs`. `main()` passes `**strategy_hooks(cfg)` to `train()` on both the new run path and the `--resume` path. `train()` takes `image_hook` with the no change default and hands it to `validate()`, which applies it after `apply_pattern()`, as `sweep()` does. E1 behaviour is unchanged: at build time a quick E1 run on CPU from the commit before this feature and one after it give equal `loss` columns in `train_log.csv`, and a test proves `config.toml` of E1 still hashes to `f9b3afb6`.
- **AC-8 Selection.** `best.pt` is chosen exactly as in E1: mean of the WT, TC and ET val Dice at pattern 1, tie keeps the earlier point, `val_patterns` stays `(1,)`. `selection_score` is not changed.
- **AC-9 Tie out file.** `experiments/0003-baseline/evaluate.py` applies `strategy_hooks(cfg)["image_hook"]` after `apply_pattern()` and is run on each E3 and E4 run, on the T4, so each has the `metrics_val.csv` that `_tie_out()` needs. For E1 the hook changes nothing: at build time, `metrics_val.csv` of a quick E1 run is byte equal between the commit before this feature and the one after it. It still never reads the test fold.
- **AC-10 One sweep.** `experiments/0004-missing-eval/evaluate.py` passes `image_hook=strategy_hooks(cfg)["image_hook"]` to `sweep()`, with `cfg` from the source run's `config.toml`. E3 and E4 evals use `--out results/0005-strategies/evals`. E1 keeps `results/0004-missing-eval`. All three use the default `experiments/0004-missing-eval/config.toml`.
- **AC-11 The seal holds.** No file or source named in `PROTOCOL_FILES` or `PROTOCOL_SOURCES` changes, nor `MODALITIES`, `REGIONS` or `REGION_LABELS`. A test asserts `protocol_hash()` equals `146635cf4521a239c4e8cba877528b4c688a6410cd3b817df671db4f1f57825b`, the value in the finished E1 val eval, so the E1 val eval stays valid and is not rerun.

*Checks and runs*

- **AC-12 Fairness check.** `experiments/0005-strategies/check_strategy.py <run dir> --e1 <E1 run dir>` exits 1 and prints each failure unless: the run has status `ok` and `quick` false; `strategy_name` is `dropout` or `presence`; its `config.toml` equals the E1 run's after removing `[strategy]`; `train_cases`, `val_cases`, `cache_hash`, `seed`, `gpu` and `iterations` equal the E1 run's; `versions` equal, printed as a warning if not, not a failure. It prints how many of the 40,000 draws fell on each pattern, recomputed from `default_rng(seed + 1)` with `draw_patterns(rng, batch_size)` once per iteration, and it prints `len(resumes)`. A resume is allowed, but the loader is reseeded on resume (spec 0003, AC-11), so that run's later patches no longer match the other models'. The draws still match, because `hook_rng` is restored from `last.pt`. `check_baseline.py` is also run on each E3 and E4 run, unchanged, including its `BEST_SCORE_MIN` of 0.60 at complete input. If that fails, stop and debug. Do not lower it.
- **AC-13 Kaggle runs.** The human runs E3 and E4 for seed 42 on the Kaggle T4, then `evaluate.py` (E1 script, AC-9), then the val sweep (AC-10), then `check_baseline.py`, `check_strategy.py` and `check_eval.py`, from one pinned commit. `run.json` holds the real timings. No number is projected in place of a measurement.
- **AC-14 One training per strategy.** E3 and E4 are each trained once at seed 42. A run may be repeated only if it failed or a check in AC-12 failed, with the reason in `context/progress-tracker.md`. Nothing about either strategy changes after its val numbers are seen. Val decides nothing here: the comparison is fixed by this document.

*Comparison*

- **AC-15 Comparison guards.** `experiments/0005-strategies/compare.py --e1-run R1 --e3-run R3 --e4-run R4 --fold val|test` stops with exit 2, writing nothing, printing every failed rule, unless: each run has status `ok` and `quick` false; its strategy is none, `dropout` and `presence` in that order; the configs differ only in `[strategy]`; `train_cases`, `val_cases`, `seed`, `gpu` and `cache_hash` are equal across the three runs; `check_rules(eval_dir, run_dir)` is empty for each eval; `fold`, `n_cases`, `protocol_hash`, `config_hash`, `splits_sha256`, `gpu` and the case set in `metrics.csv` are equal across the three evals. Eval folders are `<e1-evals>/<run id>-<fold>` and `<evals>/<run id>-<fold>`, with `--e1-evals` defaulting to `results/0004-missing-eval` and `--evals` to `results/0005-strategies/evals`. The run folders and `data/splits/splits.csv` must be on the machine that runs it (see *Joint test layout*).
- **AC-16 Comparison on test.** With `--fold test` it also needs `<out>/compare-val/run.json` with status `ok`, the same three run ids, and the same `compare_hash`. `compare_hash` is the sha256 over the bytes, in this order, of `experiments/0005-strategies/compare.py`, `experiments/common/strategies.py`, `experiments/common/model.py` and `experiments/0004-missing-eval/evaluate.py`. So a change to the comparison or to the path a strategy takes through scoring after val is refused, unless val is compared again with `--rerun-reason`. `versions` are compared between each test eval and its val eval, and with all three evals: a difference does not refuse, it sets `versions_equal` to false and prints a warning, because a test opening cannot be taken back.
- **AC-17 Comparison outputs.** Written to `results/0005-strategies/compare-<fold>/`, under the rules in *Statistics protocol*: `summary.csv` (72 rows), `patterns.csv` (270 rows), `verdict.txt`, `run.json`. `run.json` is written last. A finished folder is refused with exit 2 unless `--rerun-reason "<text>"` is given, then it is renamed `<name>.superseded-<utc time>` and the new `run.json` records `rerun_reason` and `superseded`. Two runs give byte identical CSVs and verdict. No GPU is used.
- **AC-18 Verdict.** `verdict.txt` holds the nine H2 lines in the *Verdict lines* template, reached with no human judgement.
- **AC-19 Tables.** `experiments/0005-strategies/make_table.py --compare <dir>` writes to `writing/tables/`: `0005-strategies-<fold>-summary.tex`, `-dice.tex` and `-hd95.tex`, laid out as in *Table layouts*. They read only `summary.csv`, `patterns.csv` and `run.json` of the compare folder, each with a header comment naming the three run ids, git commit and seed, and a caption with n, seed and fold. Val captions start `Development fold, not a test result.` Test captions start `Test fold, opened once.` Before writing it refuses with exit 1, leaving older tables untouched, if `run.json` is missing, status is not `ok`, or a CSV is missing or has the wrong row count. All three compile under XeLaTeX through a new committed stub `writing/tracer/e3e4-preview.tex`.

*Joint test opening*

- **AC-20 Order.** Each of these happens before the next starts: E3 and E4 val evals pass `check_eval.py`; `compare.py --fold val` passes; the val compare folder is copied into the repo and the tree is committed; that commit is pinned; `preprocess.py --folds test` runs on the T4; then `evaluate.py --fold test --open-test` runs for E1, E3 and E4 from that commit. This order is a written step in `KAGGLE.md`, and `compare.py --fold test` enforces what it can (AC-16). A resume across Kaggle sessions is normal (spec 0004, AC-5).
- **AC-21 After the opening.** On test the human runs `check_eval.py` for each of the three, `analyse.py` and `make_table.py` on the E1 test eval only (the RQ1 numbers), then `compare.py --fold test` and `make_table.py --compare`. The opening is recorded by hand in `context/progress-tracker.md` with date, commit and the three eval ids. No threshold, pattern, region, metric, or test changes after that. A number that disappoints stays in the thesis.

*Records*

- **AC-22 Records.** `experiments/0005-strategies/KAGGLE.md` gives every human step, the session death steps and the commit pin. A new private Kaggle dataset `muhammadnomanned/brats-joint-test` holds one `joint-test.tar.bin`, listed in `data/DATA.md`, laid out as in *Joint test layout*. `KAGGLE.md` starts with three shell variables, `R1`, `R3` and `R4`, holding the run ids. `train.py` prints each id as `run dir <path>` and the human fills them in, because the id holds a time stamp and cannot be known in advance. ADR 0002 gets a line for E3, E4 and that dataset. `experiments/AGENTS.md` maps E3 and E4 to `0005-strategies`, lists `common/strategies.py`, and states that E3 and E4 run the E1 and E2 scripts as commands, never by import. Spec 0003 AC-17 and spec 0004 *Key invariants* get a dated note where this spec changes their wording.
- **AC-23 Experiment check.** `0005-strategies` is added to the folder set at `tests/test_experiment_scripts.py:24`, which today would pass without it, so its scripts are proven to run nothing on import, and `uv run pytest -q` and `uv run ruff check .` pass.

## Decision

**Chosen option**: Option 2: dropout and presence as config driven hooks on the one E1 training path and the one E2 sweep, with a new paired comparison.

Uniform draw over all 15 patterns, 40,000 iterations, seed 42, selection at complete input exactly as E1, E4 built as E3 plus presence channels, and a primary test on the per subject mean over the 14 incomplete patterns.

**Implementation skills**: `pytorch-patterns` (`affaan-m/ecc`, `.claude/skills/pytorch-patterns/`) · `python-testing-patterns` (`wshobson/agents`, `.claude/skills/python-testing-patterns/`) · `scipy-best-practices` (`mindrally/skills`, `.claude/skills/scipy-best-practices/`)

## Rationale

Reasoning and options: see [rationale.md](rationale.md).

## Feature design

### Strategies at a glance

| | E1 (zero fill) | E3 (modality dropout) | E4 (presence indicator) |
|---|---|---|---|
| Config | `0003-baseline/config.toml` | `0005-strategies/e3.toml` | `0005-strategies/e4.toml` |
| Training input | 4 channels, always complete | 4 channels, random pattern each step | 8 channels, random pattern each step |
| Scoring input, pattern p | missing channels 0 | missing channels 0 | missing channels 0, plus 4 presence channels of p |
| Network input channels | 4 | 4 | 8 |
| Retrained here | no | yes | yes |

Example: on one step of E4, draw 6 (T2+FLAIR). T1 and T1ce are zeroed. The 8 channels are T1, T1ce, T2, FLAIR, then 0, 0, 1, 1 as constant maps.

### Statistics protocol (fixed before any test number)

The unit is the subject, one case each, matched on case and region. Statistics come from `common/stats.py` unchanged. `alpha`, `seed` and `bootstrap_resamples` come from the E1 eval folder's `config.toml`.

| Item | Rule |
|---|---|
| Comparisons | `E3-E1`, `E4-E1`, `E4-E3`. Paired difference `d = value(first) - value(second)` for one subject. For Dice positive is better, for HD95 negative is better |
| Primary score | For each subject and region, the mean over the 14 incomplete patterns (2 to 15), so one number per subject. Pattern 1 is left out, because it is not a missing case |
| Primary metric | Dice, WT, TC and ET separately |
| Secondary metric | HD95, including the 373.13 mm empty rule, same tests, own families. Its primary score is pulled by empty predictions on weak patterns, so read it as how often the model finds nothing, not as a pure boundary measure, and use the per pattern medians for boundaries |
| Primary test | `paired_wilcoxon(d)` on the primary score. `holm` over the 3 comparisons, one family per (metric, region). Alpha 0.05 |
| Per pattern test | `paired_wilcoxon(d)` for each of the 15 patterns, `holm` over the 15, one family per (metric, region, comparison). Secondary, it shows where a gain sits |
| Complete input cost | The pattern 1 row of `patterns.csv`. Reported with its size and CI. A non significant result is never written as no cost, because that would need an equivalence test this study does not run |
| By number missing | Same score over `n_missing` 1 (4 patterns), 2 (6) and 3 (4). Descriptive, CI, no test |
| Confidence interval | 95% percentile bootstrap of the mean of `d` over subjects, `bootstrap_indices(n, resamples, seed)` reused for every row, `bootstrap_ci` |
| Cases kept | Every case. No exclusions after the fact |

**H2 decision rule.** For each comparison and region: `supported` if the Dice primary `mean_diff > 0` and `p_holm < alpha`, otherwise `not supported`. H2a is `E3-E1`, H2b is `E4-E1`, H2c is `E4-E3`. The direction is the sign of the mean, even though the test is on ranks. If `mean_diff` and `median_diff` have opposite signs, the thesis says so beside the number. A strategy is called best only if a supported line says so. If H2c is not supported, the thesis says the indicator added no shown benefit over dropout alone.

**Verdict lines.** One line each, regions in order WT, TC, ET, first all H2a, then H2b, then H2c. `mean_diff` uses `%+.6f`, `p_holm` uses `%.6g`. These lines show the format only, and hold no results:

```text
H2a WT: supported (E3-E1, mean_diff=+0.123456, p_holm=1.2e-05)
H2c ET: not supported (E4-E3, mean_diff=-0.001234, p_holm=0.42)
```

### Data model sketch

Nothing is committed: `results/*` and `writing/tables/` are ignored by git.

| Entity | Key | Fields | Rows |
|---|---|---|---|
| Run folder `results/0005-strategies/runs/<run id>/` | run id | same as E1, spec 0003. `RUN_KEYS` unchanged. `metrics_val.csv` from AC-9 | 1 per strategy and seed |
| Eval folder `results/0005-strategies/evals/<run id>-<fold>/` | run id + fold | same as E2, spec 0004 | 1 per run and fold |
| Compare folder `results/0005-strategies/compare-<fold>/` | fold | `summary.csv`, `patterns.csv`, `verdict.txt`, `run.json` | 1 per fold |
| `summary.csv` | metric, region, level, comparison | `metric, region, level, comparison, n, mean_first, mean_second, mean_diff, median_diff, ci_low, ci_high, n_nonzero, wilcoxon_stat, p_raw, p_holm`. `level` is `incomplete`, `missing_1`, `missing_2` or `missing_3`. The last 3 fields are filled only for `incomplete` | 2 × 3 × 4 × 3 = 72 |
| `patterns.csv` | metric, region, pattern_id, comparison | `metric, region, pattern_id, pattern_name, comparison, n, mean_first, mean_second, median_first, median_second, mean_diff, median_diff, ci_low, ci_high, n_nonzero, wilcoxon_stat, p_raw, p_holm` | 2 × 3 × 15 × 3 = 270 |

Compare `run.json` keys: `compare_id, status, fold, n_cases, e1_run_id, e3_run_id, e4_run_id, e1_eval_config_hash, protocol_hash, splits_sha256, compare_hash, versions_equal, n_resumes, seed, alpha, git_commit, git_dirty, rerun_reason, superseded, wall_time_s`.

Row order in each CSV: `metric` dice then hd95_mm, region WT, TC, ET, level or pattern ascending, comparison `E3-E1`, `E4-E1`, `E4-E3`. File formats as spec 0004: `lineterminator="\n"`, floats `%.6f`, p values `%.6g`, empty field for a missing value.

### Table layouts

All tables use `booktabs` inside `\resizebox{\linewidth}{!}{...}`, Dice at 3 decimals, HD95 at 1.

| File | Rows | Columns |
|---|---|---|
| `-summary.tex` | Dice and HD95 × 3 regions, level `incomplete` | metric, region, then per comparison: `mean_diff`, CI, star if `p_holm` below alpha |
| `-dice.tex` | 15 patterns | ID, present modalities, then per region mean Dice of E1, E3, E4, with a star on E3 and E4 if `p_holm` of its comparison with E1 is below alpha |
| `-hd95.tex` | 15 patterns | same as Dice, with the median |

The star caption line reads as in spec 0004, and says the test is two sided, so the direction comes from `mean_diff`.

### Joint test layout

`/kaggle/input` is read only, and the gate reads `<out>/<run id>-val`, so the notebook unpacks the tar into `/kaggle/working/joint/`:

```text
joint/runs/<run id>/            3 run folders: config.toml, run.json, metrics_val.csv, checkpoints/best.pt
joint/evals-e1/<R1>-val/        E1 val eval, --out for the E1 test run
joint/evals/<R3>-val/           E3 val eval, --out for the E3 test run
joint/evals/<R4>-val/           E4 val eval, --out for the E4 test run
```

So E1 runs with `--run joint/runs/$R1 --out joint/evals-e1`, and E3 and E4 with `--out joint/evals`. The tar keeps `best.pt` byte equal, and the gate proves it through `source_best_sha256`. Session death follows `0004-missing-eval/KAGGLE.md`: the output folder is saved and restored into `/kaggle/working/joint/`, so a resume finds its partial `metrics.csv`.

After the sweeps the human copies back to the laptop: E1's test eval to `results/0004-missing-eval/`, and E3's and E4's to `results/0005-strategies/evals/`. `compare.py` then runs on the laptop with `--e1-run results/0003-baseline/$R1 --e3-run results/0005-strategies/runs/$R3 --e4-run results/0005-strategies/runs/$R4`. It needs no `best.pt`, but `check_rules` reads each run's `run.json`, `config.toml` and `metrics_val.csv`, and on test re-derives the case list from the committed `data/splits/splits.csv`, so the laptop must be at a commit with that same file.

### State transitions

Runs and evals follow specs 0003 and 0004. A compare folder has one state: it exists with status `ok`, or it does not exist. `ok → renamed superseded` only through `--rerun-reason`.

The feature moves through these steps, each needing the one before: E3 and E4 trained → val evals → val compare → commit pinned → test cache → three test evals → test compare → done.

### API surface (scripts, run from the repo root)

| Command | Key inputs | Key outputs | Gate | Key errors |
|---|---|---|---|---|
| `experiments/0003-baseline/train.py` (changed) | `--config`, `--out`, `--quick`, `--resume`, as before | run folder | none | 2 unknown strategy (AC-6), as before otherwise |
| `experiments/0003-baseline/evaluate.py` (changed) | `--run` | `metrics_val.csv` | none | as before |
| `experiments/0004-missing-eval/evaluate.py` (changed) | as before | eval folder | AC-7 of spec 0004 for test | as before |
| `experiments/0005-strategies/check_strategy.py` | run dir (positional), `--e1` (req) | PASS or failures, pattern counts | none | 1 any rule failed |
| `experiments/0005-strategies/compare.py` | `--e1-run`, `--e3-run`, `--e4-run`, `--fold` (req), `--e1-evals`, `--evals`, `--out` (default `results/0005-strategies`), `--rerun-reason` | compare folder | AC-15, AC-16 | 2 any guard |
| `experiments/0005-strategies/make_table.py` | `--compare` (req), `--out-dir` (default `writing/tables`) | three `.tex` files | none | 1 AC-19 refusals |

`common/strategies.py` (imports `common.data` and `common.missing` only, so `common/model.py` and `common/training.py` may import it, and a test imports each of the three first in a fresh process to prove no cycle): `draw_patterns`, `append_presence`, `dropout_hook`, `presence_hook`, `presence_image_hook`, `keep_batch` and `keep_image` (the no change pair), `strategy_name`, `strategy_hooks`, `strategy_problems` (the config check, named apart from the `check_strategy.py` script).

`common/model.py`: `PresenceUNet`, and `build_model` as in AC-5. `predict()` is not touched.

`common/training.py`: `validate(..., image_hook=keep_image)`, `_val_point`, `_loop` and `train` pass `image_hook` through, `check_config` calls `strategy_problems`. `selection_score`, `RUN_KEYS`, `Trainer` and the loss do not change. The module docstring's word `unchanged` is corrected.

`memory_test.py` and `gpu_bench.py` are not run for E3 and E4. Their batches skip the hooks, and the 4 channel model size is known. `check_baseline.py` reads the real peak from `run.json`.

### Value sourcing

| Action | Value | Source |
|---|---|---|
| train | strategy | `[strategy].name` in `e3.toml` or `e4.toml`, copied into the run's `config.toml` |
| train | pattern of a sample | `draw_patterns(hook_rng, B)` |
| train | `hook_rng` state on resume | `last.pt`, already saved (spec 0003) |
| train | patches, flips, intensity | as E1, seed 42, loader generator |
| train | selection score | `selection_score` at pattern 1 |
| train | `n_params` | `Trainer`, as E1, so E4's few extra weights are on record |
| evaluate | presence channels of pattern p | `PATTERNS[p].present`, in `MODALITIES` order |
| evaluate | model shape | strategy in the source run's `config.toml` |
| check_strategy | E1 values | `--e1` run's `run.json` and `config.toml` |
| check_strategy | pattern counts | `draw_patterns(default_rng(seed + 1), batch_size)` repeated `iterations` times, `batch_size` from the run's `[train]` |
| check_strategy | number of resumes | `len(rec["resumes"])` of the run |
| compare | `n_resumes` | `{e1, e3, e4: len(resumes)}` from the three runs |
| compare | all cell values | `metrics.csv` of the three eval folders, Dice and `hd95_mm` |
| compare | `alpha`, seed, resamples | `[eval]` in the E1 eval folder's `config.toml` |
| compare | `pattern_name` | `PATTERNS[id].name` |
| compare | `n_missing` | `len(PATTERNS[id].missing)` |
| compare | `compare_hash` | sha256 of the four files in AC-16 |
| compare | `git_commit`, `git_dirty` | `git_info()` |
| compare | `versions_equal` | `versions` in the eval `run.json` files |
| compare | `wall_time_s` | `time.time()` around the script |
| make_table | run ids, git, seed, n, fold | the compare folder's `run.json` |
| make_table | `alpha` for the star | the compare folder's `run.json` key `alpha`, copied there by `compare.py`, so there is one value |

### Key invariants

- E1, E3 and E4 share subjects, splits, iterations, selection rule, sweep, metrics and statistics, and patches at a given seed up to a run's first resume. The strategy is the only intended difference. Model init does not move patches, because the loader has its own generator.
- E1 is never retrained or rescored on val for this feature.
- Every sealed file stays byte equal (AC-11).
- A missing channel is the same exact zero in training, in val and in the sweep.
- E3 and E4 are scored through the same `predict()`, `apply_pattern()` and `sweep()` as E1.
- An E4 presence channel always tells the truth about the pattern: no noise, no dropout on the presence channels.
- `common/strategies.py` never reads a case, so it cannot touch the test fold.
- No script reads a test case without passing `check_test_gate()`.
- No experiment imports another experiment. E3 and E4 reuse the E1 and E2 scripts as commands only.

### Security model

Single researcher, no accounts. Outputs hold numbers and case IDs only. The new Kaggle dataset stays private, with the same terms as the E1 dataset in `data/DATA.md`. It holds model weights and metrics, no MRI volumes.

### Configuration required

- No new environment variable. `BRATS_TRAIN_DIR` is used by `preprocess.py` as before.
- The private Kaggle dataset `muhammadnomanned/brats-joint-test`, uploaded before the joint opening.

### Critical test scenarios

- Happy path E3 and E4: `train.py --quick --config e3.toml`, then `evaluate.py` of E1, then the sweep `--quick`, on synthetic cases on CPU, for both strategies, verifies **AC-3**, **AC-4**, **AC-7**, **AC-9**, **AC-10**.
- Config fairness: dict equality against the E1 config, verifies **AC-1**.
- Draws: same seed gives the same sequence for E3 and E4; a large draw hits all 15 and never 0 or 16; the patch sequence does not change when the hook draws, verifies **AC-2**.
- Presence values: pattern 6 gives 0, 0, 1, 1 in `MODALITIES` order; NumPy and torch give equal values, verifies **AC-4**.
- Padding: an image smaller than the window through `predict()` matches a forward pass on the same input with constant presence channels, for a present and for a missing channel (a missing one stays 0), verifies **AC-5**.
- E1 untouched: the E1 `best.pt` shape loads, a quick E1 run keeps its losses, verifies **AC-5**, **AC-7**.
- Resume: an E4 run stopped by `BASELINE_INJECT_INF`, then `--resume`, finishes `ok` with presence at val, proving both paths pass the hooks, verifies **AC-7**.
- Unknown strategy: `name = "x"` exits 2 on a new run and on resume, verifies **AC-6**.
- Seal: `protocol_hash()` equals the pinned value, verifies **AC-11**.
- Fairness check: each changed field refused, one at a time, on toy run folders, verifies **AC-12**.
- Comparison: toy eval folders with known differences give known `mean_diff`, Holm and verdict lines, including a tie (all zero `d`), row counts, byte equal reruns, verifies **AC-17**, **AC-18**.
- Guards: swapped strategy order, a different case set, a quick eval, a stale `compare_hash`, a missing val compare, each exit 2 and no output, verifies **AC-15**, **AC-16**.
- Tables: each refusal leaves older `.tex` byte equal, verifies **AC-19**.

## Build plan

Tracer Bullet: one thin path first, then thicken.

1. Thin thread for E3: `common/strategies.py` with `draw_patterns`, `dropout_hook`, the no change pair, `strategy_name`, `strategy_hooks`, `strategy_problems`; `e3.toml`; `train.py` passes the hooks; quick run on synthetic cases, through `evaluate.py` and the sweep unchanged, satisfies **AC-1**, **AC-2**, **AC-3**, **AC-6**, **AC-8**
2. E4 through the same path: `append_presence`, `presence_hook`, `presence_image_hook`, `PresenceUNet`, `build_model`, `image_hook` through `train()` and `validate()`, the two `evaluate.py` edits, `e4.toml`, the resume test, satisfies **AC-4**, **AC-5**, **AC-7**, **AC-9**, **AC-10**
3. Seal test with the pinned hash, run before anything else is merged, satisfies **AC-11**
4. Thin comparison: `compare.py` with Dice primary only, `verdict.txt`, on toy eval folders, satisfies **AC-18**
5. Full comparison: HD95, `summary.csv`, `patterns.csv`, bootstrap, all guards, `compare_hash`, folder rule, satisfies **AC-15**, **AC-16**, **AC-17**
6. `make_table.py` and the preview stub, satisfies **AC-19**
7. `check_strategy.py`, satisfies **AC-12**
8. Records: `KAGGLE.md`, ADR 0002, `data/DATA.md`, `experiments/AGENTS.md`, the dated notes in specs 0003 and 0004, `docs/whats-built.md`, tracker, satisfies **AC-14**, **AC-20**, **AC-21**, **AC-22**, **AC-23**
9. Human: E3 and E4 on the T4, E1 script evaluate, val sweeps, all checks, val compare, satisfies **AC-13**
10. Human: the joint test opening and everything after it, satisfies **AC-20**, **AC-21**

## Consequences

**Positive**:
- One training path and one scoring path for all three models, so a difference cannot come from a second copy of the code.
- The sealed E2 val eval stays valid, so no rerun.
- E3 and E4 share the pattern sequence always, and the patches when neither run resumed, so `E4-E3` measures the presence channels almost alone. `n_resumes` in the compare record shows whether that held.
- RQ2 gets one primary answer per region, not hundreds of tests.

**Negative / tradeoffs**:
- GPU time, projected from the measured runs: about 8.2 h per training notebook, 1.1 h per val sweep, and 2.1 h per test sweep. In all about 25 h on the T4, over at least three sessions. The Kaggle quota is still not recorded, so check it first.
- Only 1 in 15 steps of E3 and E4 sees complete input, so both may lose Dice at pattern 1. That cost is reported as a result. It is not tuned away.
- Selecting at complete input may not pick the checkpoint best for missing patterns. E1 shows the gap is small late in training (0.0012 between the last two val points), but E3 and E4 are unmeasured.
- A uniform draw spends as many steps on weak patterns such as T1 alone as on strong ones.
- One seed means the results describe variation between subjects, not between trained models.
- `0003-baseline/evaluate.py` and `train.py` now serve three experiments, which bends the plain reading of one folder per experiment (see rationale).
- A joint opening costs one sitting of attention, and a mistake on test cannot be taken back.
- The seal holds only by not touching those files, so a bug in one found now means an E1 val rerun, with a logged reason.
- Four files outside `protocol_hash` are caught by `compare_hash` only at the comparison, after the sweep has run, so the guard stops a bad comparison, not a bad sweep.

**Neutral**:
- E4 has a few more weights than E3, recorded in `n_params`.
- Feature 11 keeps error analysis. The paired tests for RQ2 live here.

## Follow-up

- [ ] Before the E3 run: record the Kaggle weekly quota and session limit in ADR 0002 (open since E1). This feature needs about 25 h.
- [ ] Seeds 43 and 44, for E1, E3 and E4 together, if GPU time allows. Combining seeds needs its own decision.
- [ ] Add verified literature notes before the thesis cites modality dropout or the uniform draw. `neverova2016` is unverified, so until then write `[CITE?]`.
- [ ] Update the proposal text to name the primary score and H2, and to say that E4 includes dropout. The proposal has uncommitted edits, so this is left to you.
- [ ] Feature 0 open item: `exclude_flair_flag` stays as E1 ran it. If it changes, every model must be redone together, or not at all.
