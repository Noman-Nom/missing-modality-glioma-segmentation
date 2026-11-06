# Review, feat/e3-e4-strategies, 2026-11-06

**Reviewed by**: Sonnet 5.5 fresh reviewer (author on a different session)
**Scope**: 29 files, branch vs main (merge base 6dfcf0c; writing/proposal and uv.lock excluded)
**Verdict**: Approve with nits

## Summary
Adds E3 (modality dropout) and E4 (dropout plus presence channels) as hooks on the shared E1 training and sweep path, a strict three-way paired comparison (`compare.py`), a fairness check, table generator, Kaggle runbook and spec 0005. The shared-path edits are small and keep E1 behaviour (no `[strategy]` gives the same `UNet`, identity hooks, same `protocol_hash` inputs). I found no blockers or majors. Fast gates run by me: `ruff check .` clean, 317 tests pass across the strategy, compare, runs and missing-eval test files. The remaining items are rigour and robustness points.

## Minor
### 🟡 Holm family does not cover the nine reported hypotheses, `experiments/0005-strategies/compare.py:176-178`
**Problem**: `holm` runs over the 3 comparisons within one (metric, region). The verdict then prints nine H2 lines (3 hypotheses x WT/TC/ET) each marked "supported" on its own `p_holm`, and HD95 forms further families.
**Why it matters**: three regions are correlated but still three claims; a reviewer at Q1 level may ask for control across regions. It is fixed in the spec before any test number, so it is a design choice, not a bug.
**Suggested fix**: state in the thesis that the family is per (metric, region) and that regions are separate claims, or add a pre-test amendment to the spec for a 9-wide family. Do not change it after the test is opened.

### 🟡 Selection at complete input only, for arms trained on all 15 patterns, `experiments/common/training.py:499` (spec 0005 AC-8)
**Problem**: `best.pt` for E3/E4 is picked on pattern 1 although only 1/15 of their training draws are complete input. Spec 0005 records this risk at its line 267.
**Why it matters**: the bias runs against E3/E4, so it cannot inflate a benefit claim, but it can understate them and weakens the "same selection rule" argument as the best fair rule.
**Suggested fix**: keep as is (it is pre-registered), but report the val Dice trajectory at all 15 patterns for E3/E4 if `val_patterns` logs allow, and name this as a limitation in the chapter.

### 🟡 Test gate is advisory and checks little about the val compare, `experiments/0005-strategies/compare.py:118-132`
**Problem**: `val_compare_problems` checks status, run ids and `compare_hash` only. It does not check the val folder's `fold` is `val`, its `protocol_hash` or `splits_sha256` equal the test evals', or that `git_dirty` was false. `compare_hash` omits `common/training.py`, `stats.py` and `missing.py` (the last two are covered indirectly through `protocol_hash` on the eval side only). A hand-made `compare-val/run.json` passes.
**Why it matters**: the gate stops accidents, as the docstrings say, but "val before test" is only partly enforced.
**Suggested fix**: also compare `fold`, `protocol_hash`, `splits_sha256` of the val record with the test evals, and refuse a test compare when its own `git_dirty` is true.

### 🟡 No explicit NaN or empty-input guard before the stats, `experiments/0005-strategies/compare.py:158-165, 172-188`
**Problem**: `paired_wilcoxon` returns NaN for a NaN `d` (`d.any()` is True for NaN), `holm` then propagates NaN, and `verdict` prints "not supported" because `NaN < alpha` is False. `bootstrap_ci` fails on `n_cases == 0`. The sweep asserts no NaN, and `check_rules` probably catches blanks, but compare relies on that implicitly.
**Why it matters**: a corrupt `metrics.csv` would silently read as "not supported" instead of stopping.
**Suggested fix**: in `eval_problems`, fail on any NaN in dice/hd95 or on a pivot with missing (case, pattern) cells; fail on zero cases.

### 🟡 `check_strategy.py` draw counts are recomputed, not observed, `experiments/0005-strategies/check_strategy.py:51-57`
**Problem**: the printed per-pattern counts come from re-running `default_rng(seed+1)`, not from the run. A run that did not apply the hook (or crashed on resume with a bad `hook_rng`) passes if the config says `dropout`.
**Why it matters**: the check proves intent, not what training did. The test `tests/test_strategy_runs.py` covers the hook path, but a real Kaggle run has no evidence.
**Suggested fix**: log the drawn pattern IDs (or counts) in `train_log.csv`/`run.json` and compare in the check; or say plainly in the script's output that counts are expected, not observed.

## Nits
- ⚪ `experiments/0005-strategies/make_table.py:70-71`, a star is shown for any `p_holm < alpha` regardless of sign; the caption explains it, but a worse-than-E1 cell will also be starred.
- ⚪ `experiments/0005-strategies/compare.py:306-324`, between `final.rename(superseded)` and `partial.rename(final)` a crash leaves no `compare-<fold>`, and the next run then rejects `--rerun-reason`. Rare; document the manual fix.
- ⚪ `experiments/common/training.py:3-4`, module docstring line is long and mixes two sentences; wrap it.
- ⚪ `experiments/0005-strategies/compare.py:46-48`, `STRATEGIES` duplicates the name list in `common/strategies.py`; derive it to keep one source.

## Strengths
- E1 path preserved: no `[strategy]` gives the same `UNet`, `keep_batch`/`keep_image` draw nothing, `hook_rng` is separate from loader RNG and restored on resume, so E3 and E4 see identical pattern draws.
- Only-difference rule is enforced in code twice: config equality outside `[strategy]` and equal cases/seed/cache/gpu in both `check_strategy.py` and `compare.py`.
- Zero fill and mask agree: `apply_pattern` is used in training (`_drop`), `validate`, and `sweep`; presence flags come from the same `PATTERNS` table; `PresenceUNet` amax makes sliding-window zero padding harmless.
- Seal untouched: `predict`, `evaluation.py` and the other protocol files are not edited, so the E1 val eval and `protocol_hash` stay valid; `compare_hash` adds a seal on the strategy-specific scoring path.
- Compare is deterministic and pairs by sorted case; atomic folder write, run.json last, rerun needs a reason; Holm and Wilcoxon handle all-zero differences.
- Docs, spec, verify, tracker, whats-built, DATA.md and ADR 0002 updated in the same change.

## Test coverage
Good. `test_common_strategies.py` (hooks, presence channels, PresenceUNet padding), `test_strategy_runs.py` (E1 unchanged, hook path in training and eval), `test_strategy_compare.py` (733 lines: guards, known-answer Holm/verdict, byte-equal reruns, tables) and `test_missing_eval.py` additions all pass. Gaps are the NaN/empty cases and the gate-record checks noted above.
