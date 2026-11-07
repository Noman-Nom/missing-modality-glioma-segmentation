# Verify: confirmation study · spec 0006 · updated 2026-11-07
_Steps derived from spec 0006 acceptance criteria. `/check verify` runs these; `/test` locks the durable ones. This file covers milestone 1 (build steps 1 and 2). Later milestones add to it._

## Commands
- [ ] `uv run pytest -q tests/test_common_strategies.py tests/test_confirmation_compare.py tests/test_strategy_runs.py tests/test_experiment_scripts.py` → all pass → AC-2, AC-3, AC-4, AC-9, AC-10, AC-11, AC-13
- [ ] `uv run ruff check .` → clean → AC-23
- [ ] `uv run python experiments/0003-baseline/train.py --config experiments/0007-confirmation/e3b.toml --quick --out <tmp>` → status ok, 4 input channels → AC-2
- [ ] `uv run python experiments/0005-strategies/check_strategy.py <E3 run> --e1 results/0003-baseline/20261103-023817-f9b3afb6` → PASS, same draw counts per pattern as before this change → AC-3
- [ ] `uv run python experiments/0007-confirmation/compare_seeds.py --runs experiments/0007-confirmation/runs.toml --fold val` on the real manifest today → exit 2, naming every empty slot, writes nothing → AC-9, AC-10
- [ ] `uv run python experiments/0007-confirmation/compare_seeds.py --runs experiments/0007-confirmation/runs.toml --fold test` → exit 2, says build step 6 → AC-20

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
