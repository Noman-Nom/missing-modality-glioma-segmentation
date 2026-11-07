# Verify: confirmation study · spec 0006 · updated 2026-11-07
_Steps derived from spec 0006 acceptance criteria. `/check verify` runs these; `/test` locks the durable ones. This file covers milestones 1 to 4. The human runs add to it._

## Commands
- [ ] `uv run pytest -q tests/test_common_strategies.py tests/test_confirmation_compare.py tests/test_strategy_runs.py tests/test_experiment_scripts.py` → all pass → AC-2, AC-3, AC-4, AC-9, AC-10, AC-11, AC-13
- [ ] `uv run ruff check .` → clean → AC-23
- [ ] `uv run python experiments/0003-baseline/train.py --config experiments/0007-confirmation/e3b.toml --quick --out <tmp>` → status ok, 4 input channels → AC-2
- [ ] `uv run python experiments/0005-strategies/check_strategy.py <E3 run> --e1 results/0003-baseline/20261103-023817-f9b3afb6` → PASS, same draw counts per pattern as before this change → AC-3
- [ ] `uv run python experiments/0007-confirmation/compare_seeds.py --runs experiments/0007-confirmation/runs.toml --fold val` on the real manifest today → exit 2, naming every empty slot, writes nothing → AC-9, AC-10
- [ ] `uv run python experiments/0007-confirmation/compare_seeds.py --runs experiments/0007-confirmation/runs.toml --fold test` on the real manifest today → exit 2, naming every empty slot, writes nothing → AC-20

## Value sourcing
- [ ] `complete_share`: change it in a copy of `e3b.toml` to 0.25 → share of complete draws in `expected_draw_counts` near a quarter, and `compare_seeds.py` refuses that run → AC-2, AC-10
- [ ] Run ids per arm and seed: point `e3.seed43` at the wrong seed's folder in a toy manifest → exit 2, `slot says 43` → AC-9, AC-10
- [ ] `alpha`, `bootstrap_resamples`, bootstrap seed: change them in the eval folder `config.toml` of a toy → the indices follow, and differing configs across evals are refused → AC-10, AC-11
- [ ] Expected `protocol_hash`: edit a sealed file → the pin test fails, and an eval with another hash is refused → AC-10
- [ ] Tag name: `PREREG_TAG` set to a tag that does not exist → every eval refused as not descending → AC-10

## Acceptance-criteria coverage
- AC-2 by the exposure tests, `e3b.toml` test, quick run · AC-3 by the three fixed value tests · AC-4 by the `complete_share` tests
- AC-9 by the real manifest test and the path guards · AC-10 by the guard table, tag and protocol tests · AC-11 by the averaging, unit and non finite tests
- AC-13 by the known answer and verdict tests · AC-24, H2 part only, by the three identical seeds test
- Not covered yet: every other AC, in later milestones.

## Milestone 2 (added 2026-11-07)
Toy folders only, until the human runs. The full `uv run pytest -q` is for the human: it was killed here before it finished.
- [ ] `uv run pytest -q tests/test_confirmation_compare.py tests/test_common_evaluation.py tests/test_common_strategies.py tests/test_strategy_compare.py tests/test_strategy_runs.py tests/test_missing_eval.py tests/test_experiment_scripts.py` → all pass → AC-7, AC-12, AC-14 to AC-18, AC-24, AC-26, AC-29
- [ ] `uv run pytest -q` → all pass → AC-23
- [ ] In `compare_seeds.py`, a CI of exactly minus 0.015 to 0.01 → `inconclusive`, never `equivalent` → AC-14
- [ ] On the 12 real val evals, `compare_seeds.py --fold val` → 18 verdict lines, `summary.csv` 87 rows, `patterns.csv` 360, `per_seed.csv` 180, `volume.csv` 180, `run.json` last → AC-17, AC-18
- [ ] Run it again with `--rerun-reason "<text>"`, into a copy → CSVs and verdict byte identical, the old folder kept as `.superseded-<stamp>` → AC-17
- [ ] `0007-confirmation/make_table.py --compare <dir>` → 7 files whose captions start with the fold note and give n and the 3 seeds; they build in XeLaTeX with `longtable` in the preamble → AC-29
- [ ] Set `status` in a copy of `run.json` to anything but `ok` → exit 1, older tables unchanged → AC-29

## Value sourcing, milestone 2
- [ ] Voxel size: run `sweep` on a case with spacing that is not 1 mm → `pred_ml` and `true_ml` scale with the product of `d["pixdim"]` → AC-7
- [ ] Margin: `EQUIV_MARGIN` in `compare_seeds.py`, recorded as `equiv_margin` in `run.json`, a sealed file → AC-14, AC-28
- [ ] Bootstrap settings and `alpha`: from the eval folder config, as a test already proves → AC-11
- [ ] `true_ml` for WT on the 3 seed 42 evals, within 0.1 mL of `ml_WT`, and scores unchanged, through `rescore_problems` → AC-26

## Milestone 3 (added 2026-11-07)
Toy folders and a toy git repo, until the human runs. No test case exists in any of these steps.
- [ ] `uv run pytest -q` → 811 or more pass → AC-19 to AC-21, AC-27, AC-28, AC-30, AC-31
- [ ] `uv run ruff check .` → clean → AC-23
- [ ] `compare_seeds.py --runs experiments/0007-confirmation/runs.toml --fold test --preflight` on the real manifest today → exit 2, writes nothing → AC-20
- [ ] `compare_seeds.py --runs experiments/0007-confirmation/runs.toml --fold val --preflight` → exit 2, `only for --fold test` → AC-20
- [ ] `evaluate.py --run results/0003-baseline/20261103-023817-f9b3afb6 --fold test --open-test` today → exit 2, prints `test gate, nothing read`, names the missing gate file, the tag and the val compare, and leaves no `*-test` folder → AC-19, AC-27
- [ ] Remove `opening_problems` from `check_test_gate` in a copy → gate tests fail. Exclude `test_the_pinned_protocol_hash_is_the_seal`, or that test trips first and proves nothing → AC-27
- [ ] Remove the `git diff` check from `seal_problems` in a copy → `test_an_eval_commit_with_a_moved_sealed_file_is_refused` fails → AC-28
- [ ] `grep -- '--open-test\|--fold test\|--folds test' experiments/0005-strategies/KAGGLE.md` → no match → AC-21
- [ ] `git diff db5d56f -- experiments/0005-strategies/compare.py` → empty → AC-21
- [ ] Read the dated notes in spec 0005, at its AC-11, AC-20, AC-21 and the E1 rescoring line → each points to spec 0006 → AC-7, AC-21
- [ ] Real opening, human only: all 12 test evals, then `rq1_seeds.py` → 6 lines, each `agree` or `split` → AC-22, AC-30

## Value sourcing, milestone 3
- [ ] Literature gate: in a copy of `test-gate.md`, try `papers_read: 49`, `status: Pass`, a missing `cai_status` and a second `status` line → each refused by `gate_problems`, at the preflight and at `evaluate.py`; at 50 papers it passes → AC-19
- [ ] Gate at the commit: an eval commit whose own `test-gate.md` says fail is refused, even if the tree says pass, because `seal_problems` uses `git show` → AC-28
- [ ] Tag name: `PREREG_TAG` set to `prereg-v2` with evals on the older commit → refused until they are scored again → AC-31
- [ ] Val compare: `run_ids` without this run, or a status that is not `ok` → the gate refuses before any case is read → AC-27
- [ ] Margin: a changed `EQUIV_MARGIN` changes `compare_hash`, so `--fold test` refuses → AC-14, AC-28
- [ ] H1 agreement: 3 seeds with `not supported` and different pattern lists → `agree`; 2 against 1 → `split, claim not made` → AC-30

## Milestone 4 (added 2026-11-07)
Toy folders and a toy git repo, until the human runs. No test case exists in any of these steps.
- [ ] `uv run pytest -q` → all pass, 811 plus the new check and records tests → AC-23
- [ ] `uv run ruff check .` → clean → AC-23
- [ ] `check_confirmation.py --runs experiments/0007-confirmation/runs.toml` today → exit 1, three `AC-26 <arm>: FAIL` lines, nine `no entry` lines, three `metrics.csv missing` lines → AC-1, AC-9, AC-26
- [ ] `grep -c 'brats-joint-test' data/DATA.md docs/architecture/decisions/0002-kaggle-t4-training.md` → 1 in `data/DATA.md` (the line saying it was replaced), 0 in the ADR → AC-25
- [ ] Read `0007-confirmation/KAGGLE.md` from top to bottom as the person who will run it → every step is there, in the order of AC-20, nothing assumes this chat → AC-20, AC-25
- [ ] After step 2 of `KAGGLE.md`, on the real rescored evals → three `AC-26 <arm>: ok` lines → AC-7, AC-26
- [ ] After step 4, on all 12 → `PASS` → AC-1, AC-2, AC-5, AC-9, AC-23
- [ ] Human: `git tag -n99 prereg-v1` holds every fact of AC-8 plus the E4 score 0.8711, and the Release shows on GitHub → AC-8

## Value sourcing, milestone 4
- [ ] Kept run ids: `KEPT` in `check_confirmation.py` against the `run.json` of each folder, and an entry pointing elsewhere → `must stay at` → AC-5
- [ ] Pinned commit: read from the 9 new `run.json` files, never typed in. Two commits, or a dirty run → fails. A commit before the tag → fails → AC-5
- [ ] Lock commit: `LOCK_COMMIT` is `5ce3d3b`; a changed `uv.lock` or an unknown commit → fails → AC-23
- [ ] 4 input channels: `n_params` of each E3b run against the E1 of its seed. The real values are 12,870,506 for E1 and E3 and 12,877,418 for E4, so the rule tells them apart → AC-2
- [ ] Session limit: the batch `SIZE` in step 6 comes from ADR 0002, which does not hold it yet → AC-25

## Not covered yet
AC-8 needs the human tag. AC-22 needs the real opening. AC-1, AC-5 and AC-6 are proven on toys only until the 9 runs exist.
