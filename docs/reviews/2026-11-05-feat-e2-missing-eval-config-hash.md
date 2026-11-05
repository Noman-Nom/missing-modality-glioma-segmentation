# Review, E2 missing modality evaluation (focus: config_hash fix 32f4f0f), 2026-11-05

**Reviewed by**: Claude Sonnet 5.5 (author on Claude Opus 5.5)
**Scope**: 24 files in `351b054..32f4f0f` (merged as bcfa863), branch vs base. Uncommitted proposal edits ignored as asked.
**Verdict**: Approve with nits

## Summary

Commit 32f4f0f makes the test gate and `check_rules` compare `config_hash` with the val eval, and makes `check_rules` stop returning early on an empty `metrics.csv`. I traced both and the logic is correct. I also removed the fix in a scratch copy (outside the repo) and ran the new tests: 5 of them fail without the fix, so they do test it. The fast gates pass on the repo (`ruff` clean, 455 passed). Nothing here must change before the Kaggle val run (AC-18). The findings below are small, and the only ones worth a thought before the run are the two that touch sealed files, because they are free now and cost a second val run later.

## What the two hashes cover

- `config_hash` (`runlog.py:33`) is sha256 of `json.dumps({"eval": <[eval] table>, "quick": bool}, sort_keys=True)`. `config.toml` has only an `[eval]` table (`seed`, `test_timepoints`, `bootstrap_resamples`, `alpha`), so the hash covers every key in the file, whichever `--config` path is used. It is built from the parsed values, so comments, line endings, key order and the file path do not matter.
- `protocol_hash` (`evaluation.py:93`) covers the bytes of the default `config.toml`, `analyse.py`, `evaluation.py`, `metrics.py`, `missing.py`, `stats.py`, the source text of `predict`, `cached_items`, `cached_eval_transforms`, `CachedCase`, `label_to_regions`, and the modality and region constants.
- Before 32f4f0f, `--config other.toml` slipped past `protocol_hash` (it seals only the default file). Now `config_hash` closes that. Between them, every setting that reaches the analysis is covered.

What still passes both hashes is listed in the findings below. None of it is a flaw in the fix.

## Does the gate really refuse?

I traced `evaluate.py:89-99` and `evaluation.py:271-299`.

- Order: `load_run(a.run)`, then `protocol_hash()`, then the config hash is built from `--config`, then `check_test_gate`. Only `run.json`, the config and the val folder are read. No case list, no splits file, no `.npz` is touched before the gate returns. Good.
- The config hash is computed on every call, including resume, so there is no path that skips the check. A resume of a test folder runs the gate again, and `config_hash` is also in `RESUME_KEYS`.
- `fold != "test"` returns early (line 279), which is right: the val fold has nothing to compare with. A val run under another `--config` is allowed, and test must then use that same config. This is the "config val used" rule, and `test_gate_follows_the_config_val_used` pins it.
- None equals None cannot happen here: `config` is always a 64 character string, so a val record with a null hash is refused. In `check_rules` both sides come from records `evaluate.py` wrote with a real hash. An old `run.json` with no `config_hash` key would give a `KeyError`, not a refusal. That cannot occur for this repo, because `RunRecord` writes every `EVAL_RUN_KEYS` key, and no val eval exists yet.
- Stability: ints, floats (`0.05`), a list of ints and sorted keys give identical JSON on every machine on Python 3.12. No path is inside the hash. Kaggle clones the repo with git (`KAGGLE.md:49`), so `git_info()` works there and the dirty check sees real values. I see no source of a false refuse on the Kaggle run.

## Empty eval refusal

`check_rules` no longer returns early on an empty `metrics.csv` (`evaluation.py:253-256`, the early return now sits only under `if nans`). With zero rows and `n_cases` cut to 0, the row count check passes, so the refusal comes from the case set check: `_tie_out` on val (E1 has cases, E2 has none) and `_test_case_set` on test. Both are covered by tests. It cannot fire on a legal run: a legal run has `n_cases` above 0 and all rows present. I checked that no legal row yields a non numeric Dice or HD95 (empty regions score 0.0 and `HD95_EMPTY_MM`), so the NaN check does not fire wrongly either.

## Major

None.

## Minor

### 🟡 The eval folder's `config.toml` is trusted, not checked against `config_hash`, `experiments/0004-missing-eval/analyse.py:166`
**Problem**: `analyse.py` reads `alpha`, `seed` and `bootstrap_resamples` from `<eval>/config.toml`. Nothing checks that this copy still hashes to the `config_hash` in `run.json`. A hand edit of that file changes the verdict and both hashes stay quiet. `check_rules` (`evaluation.py:223`) does not look either.
**Why it matters**: This is the one path where a result changing setting passes both hashes. It needs a manual edit, so it is an accident risk only, but the project rule says every number comes from a logged config.
**Suggested fix**: In `check_rules`, add one line: the hash of `load_config(eval_dir / "config.toml")["eval"]` must equal `rec["config_hash"]`. `evaluation.py` is sealed, so doing it later costs a second val run. Doing it now is free. If you skip it, say so in the spec as a known limit.

### 🟡 `evaluate.py` and `common/data.py`, `cache.py`, `model.py` are outside both hashes, `experiments/common/evaluation.py:39-44`
**Problem**: The test fold list comes from `fold_cases` in `data.py`, and `check_rules` recomputes it with the same function. The model is built in `build_model` and `model.eval()` and `set_seed` are called in `evaluate.py:164-168`. An edit to any of these between val and test would pass `protocol_hash` and `config_hash`.
**Why it matters**: The file comment at line 36 says this is deliberate (E3 and E4 change the model and data code). The risk is real but small. For `evaluate.py` alone, sealing it is cheap, since it is the file that holds the val and test flow. The cost is that any later edit to it, even a docstring, needs a val rerun.
**Suggested fix**: Decide now. Either add `evaluate.py` to `PROTOCOL_FILES`, or record in the spec that it is unsealed on purpose. Both are free before AC-18 and costly after.

### 🟡 The gate does not compare `splits_sha256` or the checkpoint between val and test, `experiments/common/evaluation.py:294-298`
**Problem**: `splits_sha256` and `source_best_iteration` are recorded in the val `run.json` but the gate never compares them with the current run. `best.pt` has no hash at all, only the `config_hash` stored inside it (`evaluate.py:109`).
**Why it matters**: Swapping `best.pt` or `splits.csv` after val would not be noticed. `check_rules` on val ties E1 Dice to within 0.001 for the val fold, which catches a swapped checkpoint only if the gate is rerun on the new weights, and the gate does run it, so a swapped checkpoint is mostly caught by the tie out. A changed `splits.csv` is not.
**Suggested fix**: Add `splits_sha256` and `source_best_iteration` to the keys compared in the test branch of `check_rules` and in `check_test_gate`. Can wait only if you accept a second val run, since this file is sealed. It fits in the same edit as the first minor.

### 🟡 Spec 0004 does not list the `config_hash` rule yet, `docs/specs/thesis/0004-missing-modality-evaluation/index.md` (AC-7, AC-16) and `verify.md`
Already flagged by the author. Add the rule and the matching checks in `verify.md`. Docs only, so it can wait until after the Kaggle run, but do it before the thesis chapter quotes AC-7.

## Nits

- ⚪ `experiments/common/evaluation.py:263`, the message says "no protocol_hash to compare" but the folder is now needed for both hashes. Say "no hashes to compare".
- ⚪ `experiments/common/evaluation.py:298`, "use the config val used" reads oddly. "use the config the val eval used" is clearer.
- ⚪ `tests/test_missing_eval.py` has a comment line for `test_analyse_accepts_a_quick_run_with_status_ok` that now sits above a different test (`test_analyse_reads_only_its_three_files` was inserted between them). Move the comment back so each test has its own.
- ⚪ `data/DATA.md` still says the E1 dataset is not uploaded. Already known.

## Do the new tests fail without the fix?

I copied the tree at 32f4f0f into the scratchpad, reversed the `experiments/` part of the commit, and ran the new tests there. These failed, as they should: `test_gate_refuses_a_config_outside_the_seal`, `test_gate_follows_the_config_val_used`, `test_gate_open_scores_the_synthetic_test_fold_once` (the `config_hash` section), `test_rules_refuse_an_eval_that_scored_no_case`, `test_rules_refuse_a_test_eval_that_scored_no_case`. Two passed with and without the fix, and they should: `test_gate_opens_with_a_copy_of_the_config_val_used` (guards against a false refuse) and `test_check_eval_prints_no_projection_on_test`. The seal test uses a `gate` fixture that fails on any test read, so it proves "nothing read" and not only the error text.

## Did the previous review's fixes hold?

Yes. `device` and `gpu` are in `RESUME_KEYS` (`evaluation.py:56`), the test fold case set check exists (`_test_case_set`), `predict` and the other case path functions are in `PROTOCOL_SOURCES`, and the resume test with moved paths passes. The full fast suite is green.

## Strengths

- The fix is small and the right shape: it compares content, not paths, so a copied config still opens (tested), and a different config is refused before any test data is read (tested).
- The tests are written as mutation checks: each states the failure it guards and each fails when the fix is removed.
- The Kaggle resume is designed for moved paths and tested for byte equal output.

## Test coverage

Strong. The new gate rule has a refuse test, an allow test, a "follow the val config" test and an after the fact `check_rules` test. The empty eval rule is tested on both folds. Gap: no test for a val `run.json` with a missing `config_hash` key (would be a `KeyError`, not a clean refusal). That is a nit for this repo, since no such file can exist.

## Answer for the Kaggle run (AC-18)

The Kaggle val run can go ahead on bcfa863. Nothing in 32f4f0f is wrong, and no false refuse is expected. Before you start, think once about the first two minors, because `evaluation.py` and `PROTOCOL_FILES` are sealed and changing them after the val sweep costs a second val run. If you accept the limits they describe, run as is.
