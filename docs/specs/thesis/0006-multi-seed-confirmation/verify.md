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
