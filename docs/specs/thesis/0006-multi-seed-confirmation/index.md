# 0006. Confirmation study: three seeds, a complete input cost test, an exposure arm, one sealed test opening

**Date**: 2026-11-07
**Status**: In Progress

## Summary

A manuscript by Cai et al. 2026 already compares zero fill, modality dropout and presence channels on BraTS, at low resolution, so the study can no longer claim to be the first. This spec changes the study into a full resolution confirmation, and adds a direct test of a question the papers disagree on: does dropout cost accuracy when all four sequences are present? It adds seeds 43 and 44 for E1, E3 and E4, one new arm (E3b, half of training on complete input), a cost test with an equivalence margin, a whole tumour volume measure, a public git tag that fixes the protocol, and a literature check before the test is opened. In total it needs about 114 h of Kaggle time (plan on about 5 weeks of quota), and the test fold stays sealed until all 12 models are done.

Updated 2026-11-07, before the tag, from the review of `feat/confirmation-study`: a claim now needs all 3 seeds behind it, the Kaggle gate runs the seal check and ties the commit to the preflight, the val compare record is checked in full, and a fresh test sweep checks its GPU. See AC-32 to AC-37.

Updated again 2026-11-07, before the tag: `--preflight` no longer deletes `preflight.json` once a test eval exists, and step 7 brings the record back from the Kaggle Output. See AC-34.

## Requirements

**User stories**:
- As the researcher, I want three seeds for every arm, so that a result does not rest on one training run.
- As the researcher, I want a test that can show a cost, no meaningful cost, or no clear answer, so that the complete input question gets a real answer.
- As the researcher, I want the protocol fixed in public before the new runs, so that a reviewer can see the rules came first.
- As the researcher, I want a literature check before the test opening, so that a closer paper does not surprise me when it is too late to adapt.

**Acceptance criteria**:

*Arms and runs*

- **AC-1 Arms.** Four arms, each at seeds 42, 43 and 44: E1 (zero fill), E3 (dropout), E4 (dropout plus presence), and E3b (exposure). That is 12 models. No arm or seed is added after `prereg-v1` is tagged.
- **AC-2 E3b draw.** `draw_exposure_patterns(rng, n, complete_share)` in `common/strategies.py` calls `rng.random(n)` once, then `rng.integers(2, 16, size=n)` once. A sample gets pattern 1 if its first value is below `complete_share`, otherwise its second value, so each of the 14 incomplete patterns has equal chance. `_drop` takes the draw function as an argument, with `draw_patterns` as the default. `exposure_hook` uses it, `complete_share` is bound in `strategy_hooks(cfg)` with `functools.partial`, and the image hook is `keep_image`. `experiments/0007-confirmation/e3b.toml` is `experiments/0005-strategies/e3.toml` with only `[strategy]` changed to `name = "exposure"` and `complete_share = 0.5`. E3b has 4 input channels and no presence channels.
- **AC-3 Old draws unchanged.** `draw_patterns`, `dropout_hook` and `presence_hook` give identical output for the same `rng` state before and after this change, proven by a test with fixed expected values. So seed 42 of E1, E3 and E4 stay valid.
- **AC-4 Strategy guard.** `STRATEGIES` holds `"exposure"`. `strategy_problems` refuses `complete_share` outside the open range 0 to 1, a missing `complete_share` for `exposure`, and any `complete_share` on another strategy.
- **AC-5 Seed 42 runs kept.** The seed 42 runs of E1 (`20261103-023817-f9b3afb6`), E3 (`20261106-024450-10cfa2cc`) and E4 (`20261106-235247-8999a39e`) are kept, and `check_confirmation.py` holds these three ids as constants. The 9 new runs all come from one clean commit that descends from `prereg-v1`. Training order: E1, then E3 and E4, then E3b last. If quota runs short, E3b is dropped as a whole through the amendment path of AC-31, before any E3b val number is seen.
- **AC-6 Repeat rule.** A run is repeated only if it failed, or a named check failed (AC-20), with the reason recorded in `context/progress-tracker.md`. A score that fails `BEST_SCORE_MIN` in `check_baseline.py` counts as a failed check. Any other low score is never a reason. A slot gets at most 2 repeats. A replaced run id stays in `runs.toml` under `[replaced]` with its reason. A slot that still fails stops the study until the amendment path of AC-31 is used. Every finished seed stays in the analysis.

*Volume*

- **AC-7 Volume columns.** `METRIC_COLUMNS` gains `pred_ml` and `true_ml`, appended at the end, computed in `sweep` from the voxel count times the product of `d["pixdim"]`, divided by 1000, cast to Python `float`. All 12 val evals are scored from one commit into `results/0007-confirmation/evals/`, so no `--rerun-reason` is needed and the old eval folders stay as proof. `check_rules` gains: both columns present, not negative, and `true_ml` constant per (case, region). `SEALED_HASH` in `tests/test_common_strategies.py` is repinned in the tag commit, and a dated note in spec 0005 amends its AC-11 and its rule that E1 is never rescored. `check_eval.py` passes on these folders.

*Protocol*

- **AC-8 Public tag.** `prereg-v1` is an annotated git tag on the commit holding this spec and all code of build steps 1 to 8. The human pushes it, and publishes a GitHub Release on it for a server time stamp, before any new training starts. The tag object hash goes in `context/progress-tracker.md`. The tag message states exactly what had been seen: E1 val numbers, E1 selection score 0.8818, E3 selection score 0.8699, whether any E3 or E4 val eval was read, and that the test fold had not been opened. `0005-strategies/compare.py --fold val` is not run before the tag. If it already ran, the tag message says so.
- **AC-9 Run manifest.** `experiments/0007-confirmation/runs.toml` lists, for each of the 12 (arm, seed) pairs, the run folder as a path under `results/`. Seed 42 runs stay in their own folders, and new runs go to `results/0007-confirmation/runs/`. The human fills it in, because run ids hold a time stamp.

*Comparison*

- **AC-10 Guards.** `compare_seeds.py --fold val|test` stops with exit 2, writes nothing, and prints every failed rule, unless: the manifest has all 12 entries with no run id repeated; each run's seed equals its slot; each run has status `ok` and `quick` false; each arm has the expected strategy, and E3b has `complete_share == 0.5`; within an arm the configs differ only in `seed`; at the same seed the configs differ only in `[strategy]`; `train_cases`, `val_cases`, `gpu` and `cache_hash` are equal across all 12; `check_rules` is empty for each eval; `fold`, `n_cases`, `protocol_hash`, `config_hash`, `splits_sha256`, `gpu` and the case set are equal across all 12 evals; `protocol_hash` equals the expected value pinned in `compare_seeds.py`; every eval `git_commit` descends from the tag; no Dice or HD95 is infinite; and on `--fold val` only, the dirty value of `git_info()` is exactly `False`, so `None` refuses, an untracked file counts as dirt, and the message says so. `--fold test` records `git_dirty` and does not refuse it, because `compare_hash` already refuses a moved sealed file, and a stray file must not block the final compare. Library `versions` must match across evals on val, and on test a mismatch prints a warning and is recorded in `run.json`.
- **AC-11 Seed averaging.** For each subject, region and pattern, the value of an arm is the mean over its 3 seeds. HD95 includes the 373.13 mm empty rule in that mean. Every test runs on these averaged values, so `n` is the number of subjects, never subjects times seeds. Any non finite value in an averaged or derived number exits 2. Every interval uses `bootstrap_indices(n_subjects, resamples, seed)` once per fold, with subjects sorted by case, and `resamples` and `seed` come from the eval folder config, as in spec 0005.
- **AC-12 Per seed view.** `per_seed.csv` gives, for each seed separately, `mean_diff` and `median_diff` for `E3-E1`, `E4-E1`, `E4-E3`, `E3b-E3` and `E3b-E1`, for Dice and HD95, per region, at `level` `incomplete` and `complete`. It holds no p value. The thesis shows it beside each main result.
- **AC-13 H2 lines.** The nine H2 lines of spec 0005 are kept, with the same rule, on the averaged values, and the seed rule of AC-32 then applies. The rule of spec 0005 alone can no longer give `supported`.
- **AC-14 H3 cost test.** For `E3-E1` and `E4-E1`, on Dice at pattern 1, per region, one 95% `bootstrap_ci` of the mean of `d` decides. Margin is 0.015 Dice. Each line gets one of four labels, checked in this order, with strict comparisons on unrounded values:
  - `equivalent` if both CI bounds lie inside minus 0.015 and plus 0.015. This matches two one sided tests at 2.5% each.
  - `cost` if `ci_high < 0`.
  - `gain` if `ci_low > 0`.
  - `inconclusive` otherwise. The thesis never writes this one as no cost.

  `paired_wilcoxon(d)` and `median_diff` are reported beside the label, and do not change it. Each line also shows `min_margin`, which is `max(|ci_low|, |ci_high|)`, the smallest margin at which equivalence would hold, `seeds_agree`, the count of the 3 seeds whose `mean_diff` has the same sign as the averaged one, and `seeds_within`, the count of the 3 seeds whose `mean_diff` lies strictly inside minus 0.015 and plus 0.015. The seed rule of AC-32 then applies to the label.
- **AC-15 H4 exposure test.** One line per region, for `E3b-E3` on Dice. A line reads `supported` if at pattern 1 `mean_diff > 0`, `median_diff > 0` and raw `p < alpha`, and the lower 90% CI bound of the primary `d` is above minus 0.015. Otherwise `not supported`. Both parts must hold, so no Holm correction is used. The primary p is descriptive. The seed rule of AC-32 then applies. `E3b-E1` at pattern 1 is a descriptive row with its 95% CI and no verdict. H4 never changes an H2 or H3 line.
- **AC-16 Volume report.** `volume.csv` gives, for WT, per arm, per seed and per pattern, computed for each seed separately, the mean of predicted minus true volume in mL, and limits of agreement (the mean plus and minus 1.96 standard deviations of the difference over subjects, with `ddof=1`). Rows are ordered by E1, E3, E4, E3b, then seed, then pattern. Descriptive, no test, no verdict line.
- **AC-17 Outputs.** Written to `results/0007-confirmation/compare-<fold>/`: `summary.csv`, `patterns.csv`, `per_seed.csv`, `volume.csv`, `verdict.txt` and `run.json`, with `run.json` written last, in the schemas of *Output schemas*. `preflight.json` (AC-34) is the one other file in `compare-val/`. Two runs give byte identical CSVs and verdict. A finished folder is refused unless `--rerun-reason "<text>"` is given.
- **AC-18 Verdict.** `verdict.txt` holds the 9 H2 lines, then 6 H3 lines (`H3a` is `E3-E1`, `H3b` is `E4-E1`, regions WT, TC, ET), then 3 H4 lines, reached with no human judgement. Templates, format only, no results:

```text
H2a WT: not supported (E3-E1, mean_diff=+0.000000, p_holm=0, seeds_agree=0/3)
H3a WT: equivalent (E3-E1, mean_diff=-0.000000, median_diff=-0.000000, ci=[-0.000000, +0.000000], min_margin=0.000000, seeds_agree=0/3, seeds_within=0/3, p_raw=0)
H4 WT: not supported (E3b-E3, complete_mean_diff=+0.000000, complete_median_diff=+0.000000, complete_p=0, primary_ci90_low=-0.000000, seeds_agree=0/3, seeds_not_worse=0/3)
```

*Test opening*

- **AC-19 Literature gate.** `docs/literature/test-gate.md` holds one `key: value` line each for `status`, `date`, `papers_read`, `cai_status` and `citation_search`. The gate passes only if `status` is `pass` and `papers_read` is at least 50, and all five are present. The human writes this file. If the gate finds a closer paper, the human routes the change through `/architect` and AC-31 before any test number exists.
- **AC-20 Order.** Each step happens before the next starts: all 12 runs pass `check_baseline.py`, and E3, E4 and E3b also pass `check_strategy.py` against the E1 run of the same seed; all 12 val evals pass `check_eval.py`; `check_confirmation.py` passes; the tree is committed with `runs.toml` filled in; `compare_seeds.py --fold val` passes on that clean tree; the gate of AC-19 passes; the tree is committed and pinned; `compare_seeds.py --fold test --preflight` passes, which runs every guard that needs no test metric and writes `preflight.json`; the upload of AC-25 happens; `preprocess.py --folds test` runs; `evaluate.py --fold test --open-test` runs for all 12 from that commit; `compare_seeds.py --fold test` runs.
- **AC-21 Spec 0005 test path retired.** The joint test opening of spec 0005 (its AC-20 and AC-21) does not run. `0005-strategies/compare.py` stays in the repo, unchanged, for val only. A dated note in spec 0005 points here, and a test proves `experiments/0005-strategies/KAGGLE.md` gives no step that opens test. The RQ1 steps of its AC-21 move to AC-30.
- **AC-22 After the opening.** No threshold, pattern, region, metric, margin or test changes. A number that disappoints stays in the thesis. The opening is recorded in `context/progress-tracker.md` with date, commit and the 12 eval ids.

*Records and checks*

- **AC-23 Experiment check.** `experiments/0007-confirmation/check_confirmation.py` exits 1 and prints each failure unless AC-1, AC-2, AC-5, AC-9 and AC-26 hold for the manifest, and `git diff --quiet 5ce3d3b <pinned commit> -- uv.lock` holds. `0007-confirmation` joins the folder set in `tests/test_experiment_scripts.py`. `uv run pytest -q` and `uv run ruff check .` pass.
- **AC-24 Toy truth.** On a toy table with a known answer, tests prove: the 4 labels of AC-14 each appear on the right input, including a CI touching the margin exactly; a case where the mean is below 0 but most subjects gained; the unit is the subject (AC-11); a 95% CI is used for H3 and a 90% bound for H4; and three identical seeds give the same H2 lines as `0005-strategies/compare.py`, once the `seeds_agree` field is removed from the new line, on a toy built from values exact in binary (steps of 0.25), so that averaging cannot move a strict comparison. `0005-strategies/compare.py` stays unchanged. The seed cases of AC-32 are proven the same way.
- **AC-25 Records.** `experiments/0007-confirmation/KAGGLE.md` gives every human step, with `--out` for each command, and ADR 0002 gets a line for these runs, the quota used and the session limit. `experiments/AGENTS.md` maps E3b to `0007-confirmation`. A private Kaggle dataset `muhammadnomanned/brats-confirmation-test` holds one `confirmation-test.tar.bin` with the 12 run folders, the 12 val eval folders, `compare-val/run.json` and `compare-val/preflight.json`, and is listed in `data/DATA.md`. It is built after the preflight passes. Step 6 cell 2 of `KAGGLE.md` copies both `compare-val` files into the repo folder.
- **AC-26 Rescore proof.** For seed 42 of E1, E3 and E4, each new val eval matches its old one on all 15 patterns: Dice within 0.0001 and HD95 within 0.01 mm, per row. `true_ml` for WT is within 0.1 mL of `ml_WT` in `data/splits/splits.csv`. A failure stops the study until explained.
- **AC-27 The opening is gated.** `check_test_gate`, which `evaluate.py` calls before reading any case, also refuses unless: the gate of AC-19 passes; `PREREG_TAG` is an ancestor of `HEAD`; and `results/0007-confirmation/compare-val/run.json` has status `ok` and lists this run id. It also runs the rules of AC-33 and AC-35. `evaluate.py` adds the rules of AC-34 and AC-36 before reading any case. So the guard fires before the test is read, not after.
- **AC-28 Sealed files.** `SEALED_FILES` lives in `common/evaluation.py` (AC-33) and is `PROTOCOL_FILES` plus `common/strategies.py`, `common/model.py`, `0004-missing-eval/evaluate.py`, `0007-confirmation/compare_seeds.py`, `common/runlog.py`, `common/cache.py`, `common/data.py` and `0007-confirmation/rq1_seeds.py`. `compare_hash` is the sha256 over their bytes in that order, and `run.json` stores it. `compare_seeds.py --fold test` refuses if `compare_hash`, `protocol_hash` or `splits_sha256` differs from the val compare, if `git diff --quiet <tag> <commit> -- SEALED_FILES` fails for any eval commit, if `git show <eval commit>:docs/literature/test-gate.md` lacks `status: pass`, or if a test eval `git_commit` differs from `git_commit` in `compare-val/preflight.json`.
- **AC-29 Tables.** A new `experiments/0007-confirmation/make_table.py --compare <dir>` writes to `writing/tables/`: `0007-confirmation-<fold>-summary.tex`, `-cost.tex`, `-exposure.tex`, `-patterns-dice.tex`, `-patterns-hd95.tex`, `-seeds.tex` and `-volume.tex`. They read only the compare folder. Captions give n, the 3 seeds and the fold, and start `Development fold, not a test result.` or `Test fold, opened once.` It refuses with exit 1, leaving older tables untouched, if `run.json` is not `ok`. Verdict words come from `verdict.txt` only. `make_table.py` gains a parser for H2 lines, `H3_LINE` and `H4_LINE` accept `split`, and `read_verdict` also returns the counts. `-cost.tex` shows `seeds_agree` and `seeds_within`, and `-exposure.tex` shows `seeds_agree` and `seeds_not_worse`. In `-summary.tex`, a Dice primary cell whose H2 line reads `split` gets a dagger beside its star, and the caption says the star is the paired test alone and the dagger means the seeds did not all back it, so no claim is made.
- **AC-30 RQ1 with three seeds.** After the opening, `0004-missing-eval/analyse.py` and its `make_table.py` run on each of the 3 E1 test evals, and all 3 are reported. An H1 claim is made only if all 3 seeds agree. Otherwise the thesis reports the split, and the claim is not made.
- **AC-31 Amendment path.** `PREREG_TAG` in `common/evaluation.py` names the current tag. Before the test opening, a change after the tag needs a dated reason in this spec, a new tag `prereg-v2` (then `v3`), and a pushed GitHub Release. After the opening, no amendment exists. Nothing in this path may depend on a test number.

*Review fixes, added 2026-11-07, before the tag. No multi seed number existed when these were written.*

- **AC-32 Seed rule.** A claim needs the averaged test to pass and all 3 seeds to back it. The averaged rule runs first and is unchanged. If it gives no claim (`not supported`, `inconclusive`), the line keeps that word. If it gives a claim and the seed check below fails, the line reads `split`, and the thesis reports the split and makes no claim, as in AC-30. `split` is never written as no effect, no cost or a weak effect. All values are unrounded `mean_diff` from the Dice rows of `per_seed.csv`, comparisons are strict, and each line prints the counts, also when they change nothing.

  | Line | Averaged word | Seed check, all must be 3/3 | Level |
  |---|---|---|---|
  | H2 | `supported` | `seeds_agree`: seed `mean_diff > 0` | `incomplete` |
  | H3 | `cost` or `gain` | `seeds_agree`: same sign as the averaged `mean_diff` | `complete` |
  | H3 | `equivalent` | `seeds_within`: seed `mean_diff` inside minus 0.015 and plus 0.015 | `complete` |
  | H4 | `supported` | `seeds_agree`: seed `mean_diff > 0` at `complete`, and `seeds_not_worse`: seed `mean_diff` above minus 0.015 at `incomplete` | both |

  Each count has one meaning, whatever the word on the line. On H2 and H4, `seeds_agree` counts seeds with `mean_diff > 0`. On H3, it counts seeds with the same sign as the averaged `mean_diff`. `seeds_within` appears on H3 only, and `seeds_not_worse` on H4 only. H2 has one condition, so it prints one count.

  The check adds a condition and never removes one, so it needs no correction and `cost_label` does not change. `h2_verdict` and `h4_verdict` gain `per_seed`. There are still 18 lines.
- **AC-33 The seal runs at the opening.** `SEALED_FILES`, `compare_hash()` and `seal_problems()` move to `common/evaluation.py`, and `compare_seeds.py` imports them. `opening_problems` calls `seal_problems(commit)`, and refuses if `compare_hash()` differs from `compare_hash` in the val compare record. So a commit that moved a sealed file is refused before any case is read, and `preflight_problems` drops its own call to `seal_problems`, because the gate now holds it.
- **AC-34 Preflight record.** When `--preflight` passes, and only then, it writes `<out>/compare-val/preflight.json` with `status` (`ok`), `git_commit` and `date`. Every `--preflight` first deletes any older `preflight.json`, so on a refusal no record is left, and the file always means the last preflight passed. There is one exception: if any entry matching `*-test` exists in `--evals`, finished or not, file or folder, the opening has happened, so `--preflight` leaves `preflight.json` exactly as it found it and refuses, with one line per entry: `<entry> exists: the test was opened already, so preflight.json stays as it is`. `main` deletes the record only if `--fold` is `test` and no such entry exists, so a wrong `--fold val --preflight` touches nothing. The record is the only proof of the commit, and after the opening no new preflight can pass, so a lost record could never come back. One small function, `opened_folders(evals)` in `compare_seeds.py`, gives this `*-test` list to `main`, `guards` and `val_side_problems`, so all three agree. `guards` adds the refusal for `--preflight` right after `manifest_problems`, ahead of every early return, so it prints even if the manifest is bad. This replaces the per run id check at the end of `preflight_problems`, so a stray `*-test` entry cannot pass the preflight and rewrite the record. The laptop cannot see test folders that are still on Kaggle, so `KAGGLE.md` step 7 copies `confirmation/compare-val/preflight.json` from the last **Output** over the laptop copy, after the evals and before `--fold test`, with no comparison. That copy is the record the Kaggle gate read, so it wins. Every batch carries it forward, so every **Output** holds the same record. The preflight also refuses unless the `git_commit` of the val compare record is an ancestor of HEAD, so a squash merge is found on the laptop, not on Kaggle. A rerun of the val compare moves the record away with the old folder, so the preflight must pass again. `compare_seeds.py --fold val` refuses if any `*-test` eval folder exists, so the record cannot be lost after the opening. A new function `site_problems(val_dir, commit, gpu)` in `common/evaluation.py` holds the rules that can only be checked where the sweep runs. It reads `COMPARE_VAL / "preflight.json"`, the same repo path the gate uses for the val compare, whatever `--out` says. It refuses unless that file exists, has status `ok`, and its `git_commit` equals the `commit` of `git_info()`. `evaluate.py` calls it for `--fold test`, right after `check_test_gate`, and prints both lists under `test gate, nothing read`. The preflight never calls it, because its own record does not exist yet and the laptop GPU is not the Kaggle one. `site_problems` runs even if `check_test_gate` returned early, so a missing `preflight.json` or val `run.json` is a listed error, never a traceback.
- **AC-35 The val compare record is checked in full.** A new `val_record_problems(rec)` in `common/evaluation.py` refuses unless `status` is `ok`, `fold` is `val`, `prereg_tag` equals `PREREG_TAG`, `git_dirty` is `False`, `git_commit` descends from the tag, and `compare_hash` equals `compare_hash()`. `opening_problems` and `val_compare_problems` both call it. `run.json` gains `eval_sha256`, keyed as `run_ids` are, holding the sha256 of each `metrics.csv` the compare read. `opening_problems(run_id, commit, val_dir)` inverts `run_ids` to find the slot of this run, and refuses if `metrics.csv` in `val_dir` does not match that entry, or the entry or the file is missing. A missing file is a listed error, checked before hashing, because this runs before the `val_dir` check of `check_test_gate`. `--fold test` checks all 12 val files again, at `a.evals/<run_id>-val`.
- **AC-36 GPU at the opening.** `evaluate.py` works out `device` and `gpu` before the gate. `site_problems` reads `gpu` from `load_run(val_dir)`, and refuses unless the current `gpu` equals it. `--device cpu` gives `gpu = None`, so a P100 or a CPU sweep stops before any case is read. `gpu` joins `VAL_MATCH_KEYS`, so `check_rules`, and through it `compare_seeds.py --fold test`, also refuse a finished test eval on another GPU.
- **AC-37 Pins.** All of AC-32 to AC-36 land before the tag. `common/evaluation.py` is a protocol file, so `protocol_hash()` moves, and `EXPECTED_PROTOCOL_HASH` and `SEALED_HASH` are repinned in the same commit. No `model.py`, `strategies.py`, `metrics.py`, `stats.py`, `missing.py`, training code or scoring function changes, so seed 42 stays valid and AC-3 and AC-26 stand. A test proves `predict`, `sweep` and `region_metrics` give the same output, or a diff of the change shows they were not touched.

## Decision

**Chosen option**: Option 2: build the confirmation alongside spec 0005, in a new experiment folder.

Keep every checked seed 42 run, add 9 new models from one pinned commit, compare all 12 with a new script, and open the test only once, for all 12, after a literature gate.

Settled with the human: new spec 0006, the claim, 3 seeds, decide now with a recheck, one 95% interval decides the cost label with the paired test beside it, RQ1 claims need all 3 E1 seeds to agree, average per subject, E3b as a secondary arm on the E3 base, BraTS Africa in its own spec, no published network arm, WT volume as a secondary measure, a public git tag, keep seed 42, repeat only on failure.

Settled with the human on 2026-11-07, from the review: a new word `split` when the seeds do not all back a claim, a preflight record that the Kaggle gate checks, and a wider seal over `common/runlog.py`, `common/cache.py`, `common/data.py` and `rq1_seeds.py`. The cross check by a second model raised nine further fixes, and the human accepted all of them.

Settled by this spec (pick, why, runner up):
- **Equivalence method**: a 95% bootstrap CI inside the margin, which matches two one sided tests at 2.5%, one interval per contrast. It reuses `bootstrap_ci`, so no new library. Runner up: a t based TOST, which assumes a shape that Dice differences often lack.
- **Margin**: 0.015 Dice, on the mean difference, borrowed from Öchsner 2026, who applied 1.5 points to medians. Runner up: a margin taken from inter rater spread, which has no agreed number.
- **Name of the new arm**: E3b, because E5 is the reduced label study.
- **H4 uses no Holm**: both conditions must hold, so it is an intersection union test, and a correction would let the unused primary p change the line. Runner up: Holm over 2, which the cross check showed works backwards.
- **`equivalent` uses `seeds_within`, not signs**: with a true zero effect, the 3 signs disagree 3 times out of 4, so a sign rule would punish the very result it tests. Runner up: the sign rule of RQ1.
- **Where the Kaggle only rules live**: a separate `site_problems`, so `check_test_gate` stays the one gate the preflight shares. Runner up: a flag on `check_test_gate`, which would hide two different gates in one function.
- **Val compare runs on a clean tree**: `runs.toml` is committed first (AC-20), so the record can honestly say `git_dirty` false, and a dirty tree is refused at the compare, minutes of work, not at the preflight. Runner up: skip the `git_dirty` check, which leaves the record unproven.
- **Clean tree on val only**: on test, `compare_hash` already catches a moved sealed file, so a stray untracked file should not block the final compare. Runner up: clean tree on both folds.
- **What is hashed**: `metrics.csv` only, because every number in the compare comes from it. Runner up: also the eval `run.json`, which adds a file that a rerun changes anyway.
- **New script, not a changed one**: `compare_seeds.py` in a new folder, as a new experiment is a new folder. Runner up: extend `0005-strategies/compare.py`, which would change a reviewed file and its `compare_hash`.

**Implementation skills**: `pytorch-patterns` (`affaan-m/ecc`, `.claude/skills/pytorch-patterns/`) · `scipy-best-practices` (`mindrally/skills`, `.claude/skills/scipy-best-practices/`) · `python-testing-patterns` (`wshobson/agents`, `.claude/skills/python-testing-patterns/`)

## Which fixes touch sealed files

Every fix touches at least one sealed file, so all of them land before the tag. After the tag each would need the amendment path of AC-31. Moving `protocol_hash()` needs a repin. Moving only `compare_hash()` needs none, because it is stored at the val compare, not pinned.

| Fix | Sealed files changed | Moves | Not sealed |
|---|---|---|---|
| Seed rule (AC-32) | `0007-confirmation/compare_seeds.py` | `compare_hash` | `0007-confirmation/make_table.py`, tests |
| Seal at the opening (AC-33) | `common/evaluation.py`, `compare_seeds.py` | `protocol_hash`, `compare_hash` | `0007-confirmation/KAGGLE.md`, tests |
| Preflight record (AC-34) | `common/evaluation.py`, `compare_seeds.py`, `0004-missing-eval/evaluate.py` | `protocol_hash`, `compare_hash` | `KAGGLE.md`, tests |
| Preflight record kept after the opening (AC-34) | `0007-confirmation/compare_seeds.py` | `compare_hash` | `KAGGLE.md`, tests |
| Val compare record (AC-35) | `common/evaluation.py`, `compare_seeds.py` | `protocol_hash`, `compare_hash` | `KAGGLE.md` (commit before step 4 compare), tests |
| GPU at the opening (AC-36) | `common/evaluation.py`, `0004-missing-eval/evaluate.py` | `protocol_hash`, `compare_hash` | tests |
| Wider seal (AC-28) | `common/evaluation.py`, which now holds `SEALED_FILES` | `protocol_hash`, `compare_hash` | none. `runlog.py`, `cache.py`, `data.py` and `rq1_seeds.py` join the list, and their code does not change |

## Feature design

**Data model sketch**: nothing new is committed except the toml, the gate file and scripts. `results/*` and `writing/tables/` stay ignored by git.

| Item | Fields | Notes |
|---|---|---|
| `runs.toml` | `[e1]`, `[e3]`, `[e4]`, `[e3b]`, each with `seed42`, `seed43`, `seed44` as paths under `results/`, plus `[replaced]` | one slot per model, 12 in all |
| `metrics.csv` (changed) | existing columns plus `pred_ml`, `true_ml` | one row per (case, pattern, region) |
| `summary.csv` | `family`, `metric`, `region`, `level`, `comparison`, `n`, `mean_first`, `mean_second`, `mean_diff`, `median_diff`, `ci_low`, `ci_high`, `ci_level`, `n_nonzero`, `wilcoxon_stat`, `p_raw`, `p_holm` | `family` is `H2`, `H3`, `H4` or `descriptive` |
| `per_seed.csv` | `seed`, `metric`, `region`, `level`, `comparison`, `mean_diff`, `median_diff` | no p values |
| `volume.csv` | `arm`, `seed`, `pattern_id`, `pattern_name`, `n`, `mean_diff_ml`, `loa_low_ml`, `loa_high_ml` | WT only, 180 rows |
| `test-gate.md` | `status`, `date`, `papers_read`, `cai_status`, `citation_search` | written by the human |
| `compare-val/preflight.json` | `status`, `git_commit`, `date` | written on a preflight pass, git ignored, goes up in the tar |

**Output schemas** (row order: family, metric, region, level, comparison):
- `summary.csv`: `H2` keeps every row of spec 0005 on the averaged values, including HD95 and `missing_1` to `missing_3`. `H3` adds 6 Dice rows at `level` `complete`, with `ci_level` 95. `H4` adds 3 Dice rows at `complete` and 3 at `incomplete`, with `ci_level` 90 on the incomplete rows. `descriptive` adds 3 `E3b-E1` Dice rows at `complete`. `p_holm` is blank where this spec uses none.
- `patterns.csv`: the columns of spec 0005, for `E3-E1`, `E4-E1`, `E4-E3` and `E3b-E3`, for Dice and HD95, per region and pattern, 95% CI, `holm` over the 15 patterns per (metric, region, comparison). It is secondary, and its pattern 1 p values differ from H3 by design.
- `run.json`: `compare_id`, `status`, `fold`, `n_cases`, the 12 run ids, `eval_commits`, `eval_sha256`, `versions_equal`, `n_resumes` per run, `seeds`, `alpha`, `equiv_margin`, `compare_hash`, `protocol_hash`, `splits_sha256`, `prereg_tag`, `git_commit`, `git_dirty`.

**State transitions**: `prereg-v1` tagged → 9 new runs → 12 val evals → val compare → literature gate → preflight → test opening → test compare. No step goes back after the test opening.

**Interface surface** (commands, no web API):

| Command | Key inputs | Key outputs | Key errors |
|---|---|---|---|
| `train.py --seed N` with `e3.toml`, `e4.toml`, `e3b.toml` or the E1 config | existing | a run folder | existing guards |
| `evaluate.py --run R --fold val --out results/0007-confirmation/evals` | existing | a val eval with volume columns | existing gate |
| `compare_seeds.py --runs runs.toml --fold val\|test` | manifest, `--evals results/0007-confirmation/evals`, `--preflight`, `--rerun-reason` | the outputs of AC-17 | exit 2 on a failed guard |
| `check_confirmation.py --runs runs.toml` | manifest | PASS, or each failure | exit 1 |
| `0007-confirmation/make_table.py --compare <dir>` | a compare folder | tables in `writing/tables/` | exit 1 if `run.json` is not `ok` |

**Value sourcing**:

| Action | Value | Source |
|---|---|---|
| E3b draw | `complete_share` | `[strategy]` in `e3b.toml` |
| Volume | voxel size | `d["pixdim"]`, already passed to `region_metrics` |
| Averaging | run ids per arm and seed | `runs.toml` |
| Cost test | margin 0.015 | constant `EQUIV_MARGIN` in `compare_seeds.py`, a sealed file |
| Guards | expected `protocol_hash` | constant in `compare_seeds.py`, set at the tag commit |
| Guards | tag name | `PREREG_TAG` in `common/evaluation.py` |
| Tests | `alpha`, `bootstrap_resamples`, bootstrap seed | the eval folder `config.toml`, all equal by AC-10 |
| Literature gate | pass or fail | `docs/literature/test-gate.md`, written by the human |
| Seed rule | seed `mean_diff` per level | the in memory rows that `per_seed.csv` is written from, unrounded |
| Seed rule | margin for `seeds_within` and `seeds_not_worse` | `EQUIV_MARGIN` |
| Commit tie | preflighted commit | `COMPARE_VAL / "preflight.json"`, from `git_info()` at the preflight |
| Commit tie | Kaggle commit | `git_info()` in `evaluate.py` |
| GPU check | current GPU | `torch.cuda.get_device_name()` when the device is `cuda`, else `None` |
| GPU check | expected GPU | `gpu` in `load_run(val_dir)` for this run |
| Val record | eval hashes | `sha256_file` of each val `metrics.csv` at the val compare |

**Key invariants**:
- The arms differ only in `[strategy]`, and within an arm only in `seed`.
- The test fold is read only after AC-20 holds, enforced before reading by AC-27.
- One subject counts once in each test.
- The margin, the labels and the families never change after the tag, except through AC-31, and never after the opening. AC-28 enforces this.
- No claim rests on the averaged test alone: it needs all 3 seeds (AC-32).
- All 12 test evals come from the one commit that passed the preflight, on the GPU of the val evals (AC-34, AC-36).

**Security model**: no MRI, mask or derived patient image goes in git. Run ids, hashes and summary CSVs hold no patient image. Kaggle datasets stay private. The `prereg-v1` tag exposes code and protocol only.

**Critical test scenarios**:
- Happy path: a toy manifest of 12 folders gives the 18 verdict lines, twice, byte identical, verifies **AC-13**, **AC-14**, **AC-15**, **AC-17**, **AC-18**.
- Unit: a toy with 3 seeds gives `n` equal to the subjects, verifies **AC-11**.
- Label: toy differences give `equivalent`, `cost`, `gain` and `inconclusive`, verifies **AC-14**, **AC-24**.
- Failure: a missing entry, a changed config, a quick run or a failed status stops with exit 2 and writes nothing, verifies **AC-10**.
- Gate: `evaluate.py --fold test --open-test` with no `status: pass`, without the tag as an ancestor, or with a run missing from the val compare, is refused before any case is read, verifies **AC-19**, **AC-27**.
- Tamper: a changed sealed file, or a changed margin, between val and test is refused, verifies **AC-28**.
- Rescore: a changed score on seed 42 fails, verifies **AC-26**.
- Seed split: a passing averaged test with one seed against gives `split` on H2, H3 cost and gain, and H4. Three seeds of mixed sign, all inside the margin, stay `equivalent`. One seed outside the margin gives `split`. A line that was already `not supported` or `inconclusive` keeps its word, verifies **AC-32**.
- Seal on Kaggle: a commit with a moved sealed file is refused by `evaluate.py --fold test --open-test` before any case is read, verifies **AC-33**.
- Commit tie: no `preflight.json`, a failed preflight (no file), or a record naming another commit is refused before any case is read, and a test eval from another commit fails `--fold test`, verifies **AC-34**.
- Record after the opening: a passed preflight, then a `*-test` folder on disk, then `--preflight`: exit 2, `preflight.json` is byte for byte the same, and `--fold test` still passes. A stray `*-test` entry outside the 12 run ids, or a bad manifest, gives the same result. `--fold val --preflight` leaves the record alone. With no `*-test` entry a refusal on `--fold test` still deletes the record, verifies **AC-34**.
- Val record: `fold` test, another `prereg_tag`, `git_dirty` true, a missing `eval_sha256`, or a val `metrics.csv` changed or missing after the compare is refused by the gate and the preflight, verifies **AC-35**.
- Reachability: a val compare commit that is not an ancestor of HEAD is refused by the preflight, and a val compare with a `*-test` folder on disk is refused, verifies **AC-34**.
- GPU: a fresh test sweep on another GPU or on CPU is refused before any case is read, and a finished test eval with another `gpu` fails `check_rules`, verifies **AC-36**.
- Old draws: fixed expected values match, verifies **AC-3**.

## Build plan

Tracer Bullet: one thin thread first, from the new arm through to a verdict on toy folders, then thicken, then the human runs.

1. Thin thread: `draw_exposure_patterns`, the `exposure` strategy, `e3b.toml`, a shared `expected_draw_counts(cfg, n)` used by `check_strategy.py`, with the seal test of old draws and a `--quick` run of E3b, satisfies **AC-2**, **AC-3**, **AC-4**
2. Thin thread: `compare_seeds.py` guards and averaging on a toy manifest, with H2 lines only, satisfies **AC-9**, **AC-10**, **AC-11**, **AC-13**
3. Add H3 and H4, `per_seed.csv` and the verdict, satisfies **AC-12**, **AC-14**, **AC-15**, **AC-18**, **AC-24**
4. Add the volume columns, `volume.csv`, the repinned hash and rescoring on toy data, satisfies **AC-7**, **AC-16**, **AC-26**
5. Add outputs, rerun rules and `0007-confirmation/make_table.py`, satisfies **AC-17**, **AC-29**
6. Add the test path: the gate inside `check_test_gate`, `--preflight`, sealed files, the amendment path, the retirement of the spec 0005 path, the RQ1 steps, satisfies **AC-19**, **AC-20**, **AC-21**, **AC-22**, **AC-27**, **AC-28**, **AC-30**, **AC-31**
7. Add `check_confirmation.py`, `KAGGLE.md` and records, satisfies **AC-1**, **AC-5**, **AC-6**, **AC-23**, **AC-25**
8. Add the review fixes: the seed rule and its tables first, then move the seal into the gate, then the preflight record, the val record and the GPU check, then the wider seal, then `KAGGLE.md` and the `compare_seeds.py` docstring (commit before the step 4 compare, `git status --short` prints nothing at all, the two `compare-val` files in the upload and cell 2, the line that says the preflight is the same gate, every line that says the preflight writes nothing), then repin the two hashes last, satisfies **AC-10**, **AC-13** to **AC-15**, **AC-18**, **AC-20**, **AC-24**, **AC-25**, **AC-27** to **AC-29**, **AC-32** to **AC-37**
9. Keep the record after the opening: add `opened_folders`, use it in `main`, `guards` and `val_side_problems`, drop the per run id check from `preflight_problems`, flip `test_the_preflight_refuses_after_the_opening` (it now pins the old deletion, through the last line of `refused_preflight`, so the record must exist and be unchanged), update the `compare_seeds.py` docstring, then add the copy to `KAGGLE.md` step 7 and the exception to its preflight paragraph, satisfies **AC-34**
10. The human tags and pushes `prereg-v1` and publishes the Release, satisfies **AC-8**
11. The human runs the 9 new trainings, E3b last, then all 12 val evals, `check_confirmation.py`, then `compare_seeds.py --fold val`, satisfies **AC-1**, **AC-5**, **AC-7**, **AC-10**, **AC-26**
12. The human completes the review and the gate, runs the preflight, opens the test, and runs the RQ1 steps, satisfies **AC-19**, **AC-20**, **AC-22**, **AC-27**, **AC-30**

## Kaggle budget

Projection, from E1 and E3 timings, not a measurement. Whole notebooks fit about 3 a week, so plan about 5 weeks, which leaves room for one failure.

| Step | Each | Count | Total |
|---|---|---|---|
| E1, E3, E4 at seeds 43, 44: train, val, test | about 11.6 h | 6 | about 70 h |
| E3b at seeds 42, 43, 44: train, val, test | about 11.6 h | 3 | about 35 h |
| Seed 42 val rescoring | about 1.1 h | 3 | about 3 h |
| Seed 42 test sweep | about 2.1 h | 3 | about 6 h |
| **Total** | | | **about 114 h, plan about 5 weeks at 30 h a week** |

## Migration plan

**Strategy**: strangler. New path beside spec 0005, then spec 0005's test path is retired.
**Phases**:
1. Build `0007-confirmation`. In `0005-strategies`, the one allowed edit is that `check_strategy.py` calls the shared draw count.
2. Rescore seed 42 val with the volume columns.
3. Retire the spec 0005 test path, before any test read.
**Rollback**: until the test is opened, drop E3b or volume through AC-31, and the seed runs still stand. After the opening there is no rollback.
**Risks**: A forgotten repin after the review fixes fails the pin test, so it is caught before the tag. Kaggle library drift between the seed 42 commits and the new commit. AC-10 compares eval versions, AC-23 compares `uv.lock`, and the thesis reports the commits. A session death uses the resume steps, but a resume breaks patch matching with E1, so it is disclosed per run.

## Consequences

**Positive**:
- Results rest on 3 seeds with their spread shown.
- The study gets a real answer on the complete input cost, or says clearly that it has none.
- The public tag beats a local preregistration.

**Negative / tradeoffs**:
- About 114 h of quota and about 5 weeks of calendar time.
- The first thesis results move later, and the test opening now waits on 50 papers.
- The H2 result is no longer a first finding. A benefit of E3 over E1 is a replication.
- The bootstrap CI ignores training variation between seeds, so `per_seed.csv` must be read beside it.
- The margin is borrowed, and it was applied to medians in its source, so a reviewer may question it. `min_margin` lets a reader apply another one.
- E3 and E1 selection scores on val (a gap of 0.0119, one seed, selection biased) were seen before this spec, and they sit just inside the margin. The margin was taken from Öchsner, but this cannot prove it was uninfluenced, so the tag says so.
- `seeds_agree` shows spread between 3 models, but no interval includes it. AC-32 uses it as a veto, not as evidence: with no true effect, all 3 signs still agree 1 time in 4.
- A real but small effect can read `split`, and the thesis then makes no claim on that line. This lowers power on purpose.
- The word `split` adds a fifth H3 outcome and a third H2 and H4 outcome, which the tables and the thesis must explain.
- The preflight now writes one file, and the step 4 val compare needs one more commit.
- After the opening `preflight.json` means the commit the opening used, not the last preflight. A refused preflight leaves it in place on purpose.
- A preflight run on the laptop during step 6 cannot see the opening, so it can still delete or rewrite the laptop copy. Step 7 repairs this from the Kaggle **Output**, by a documented copy, not by code.
- `runlog.py`, `cache.py` and `data.py` are shared helpers, and they are now sealed, so an edit to them after the tag needs the amendment path.

**Neutral**:
- E3b adds one new arm name, and a fourth arm to every table.
- BraTS Africa stays planned under scope feature 10, with its own spec and ADR.

## Follow-up

- [ ] Open review finding not settled here, it needs no decision: finding 4 (time the setup from the first Kaggle notebook, before step 6). Findings 1, 3, 8 and 9 are fixed.
- [x] Add dated notes to spec 0005: this spec replaces its test path (AC-21), and amends its AC-11 and its rule that E1 is never rescored (AC-7).
- [ ] Add scope feature for this spec, and move the joint test opening from feature 7 to it.
- [ ] The human checks the publication status of Cai et al. on the journal sites and Google Scholar.
- [ ] Update `writing/proposal/proposal.tex` and the RQ wording in `context/project-overview.md` for the new claim, then add Cai et al., Öchsner et al. and others to `writing/refs.bib` with verification.
- [ ] `/architect` cross dataset evaluation, with its ADR, for BraTS Africa (scope feature 10). Its independence from BraTS 2023 GLI must be shown in `data/DATA.md`.
- [x] Record the Kaggle session limit in ADR 0002 before the tag, because the 12 test sweeps must span sessions.
- [x] The human gives the E4 run id, and records that its checks passed. It goes in AC-5 and `check_confirmation.py`.
- [x] Copy the E4 output to the laptop. Only the E3 output folder exists there today.

## Rationale

Reasoning and options: see [rationale.md](rationale.md).
