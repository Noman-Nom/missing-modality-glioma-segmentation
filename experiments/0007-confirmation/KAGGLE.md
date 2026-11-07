# The confirmation study: the test path

This file holds the steps of spec 0006 from the val compare to the end. You run all of them by hand. The training steps, the 12 val evals, the upload of `brats-confirmation-test` and the notebook cells come with build step 7, and they go in this file too. Until then, this file is not enough to run the study.

**Nothing below is a result yet. No test number exists, and the test fold has not been opened.**

```bash
RUNS=experiments/0007-confirmation/runs.toml
```

## The order (AC-20)

Each step happens before the next one starts:

1. All 12 runs pass `check_baseline.py`. E3, E4 and E3b also pass `check_strategy.py` against the E1 run of the same seed.
2. All 12 val evals pass `check_eval.py`.
3. `check_confirmation.py` passes.
4. `compare_seeds.py --fold val` passes.
5. You write `docs/literature/test-gate.md`, and the gate passes.
6. You commit the tree, and pin that commit.
7. `compare_seeds.py --fold test --preflight` passes.
8. On Kaggle, `preprocess.py --folds test` runs.
9. On Kaggle, `evaluate.py --fold test --open-test` runs for all 12, from that commit.
10. `compare_seeds.py --fold test` runs.

Steps 1 to 7 never read a test case. **Step 8 does, and it cannot be taken back.**

## The literature gate (AC-19)

You write this file by hand once you have read at least 50 papers. It needs all five lines, each once:

```text
status: pass
date: 2027-01-15
papers_read: 50
cai_status: <what you found about its peer review>
citation_search: <where you searched, and when>
```

The date and the count above show the format only. If the check finds a closer paper, we recommend writing `status: fail` and taking the change through `/architect` and the amendment path below, before any test number exists.

## The preflight (laptop, no GPU)

Commit the tree first, with `compare-val` still in `results/` on the laptop, then run:

```bash
uv run python experiments/0007-confirmation/compare_seeds.py --runs $RUNS --fold test --preflight
```

It should end with `preflight PASS`. It runs the same gate that `evaluate.py` runs for each of the 12 models, and writes nothing. If it refuses, nothing has been read. We recommend bringing the message to the main chat and changing nothing by hand.

## What the Kaggle notebook must have

Two things the gate reads, and a shallow clone would miss:

- **A clone with tags.** The gate asks git whether `prereg-v1` is an ancestor of the commit, so never use `--depth 1` here, unlike the older notebooks. The commit also needs to be on GitHub, with the tag pushed.
- **The val compare in the repo folder.** Unpack `compare-val/run.json` to `results/0007-confirmation/compare-val/run.json` inside the clone. `results/` is ignored by git, so the tree stays clean.

## After the opening (laptop, no GPU)

Copy the 12 test evals to `results/0007-confirmation/evals/`, keeping every file. Then the RQ2 comparison:

```bash
uv run python experiments/0007-confirmation/compare_seeds.py --runs $RUNS --fold test
uv run python experiments/0007-confirmation/make_table.py --compare results/0007-confirmation/compare-test
```

The RQ1 numbers (AC-30), for each of the 3 E1 test evals. `make_table.py` uses the same file names for every eval of a fold, so each seed gets its own folder, or the last one would overwrite the others:

```bash
E=results/0007-confirmation/evals
for SEED in 42 43 44; do
  R=<the E1 run id of this seed, from runs.toml>
  uv run python experiments/0004-missing-eval/check_eval.py $E/$R-test --run <its run folder>
  uv run python experiments/0004-missing-eval/analyse.py --eval $E/$R-test
  uv run python experiments/0004-missing-eval/make_table.py --eval $E/$R-test --out-dir writing/tables/e1-seed$SEED
done
uv run python experiments/0007-confirmation/rq1_seeds.py --eval $E/<id 42>-test --eval $E/<id 43>-test --eval $E/<id 44>-test
```

All 3 seeds go in the thesis. An H1 claim is made only on a line that says `agree`. On a line that says `split`, the thesis reports the split and makes no claim.

Then record the opening in `context/progress-tracker.md`: the date, the commit and the 12 eval ids (AC-22).

**No threshold, pattern, region, metric, margin or test changes after this. A number that disappoints stays in the thesis.**

## The amendment path (AC-31)

Before the test opening, a change after the tag needs all of this, in order:

1. A dated reason in spec 0006.
2. `PREREG_TAG` in `common/evaluation.py` set to the next tag, `prereg-v2`, then `v3`. This changes `protocol_hash()`, so repin `EXPECTED_PROTOCOL_HASH` and `SEALED_HASH` in the same commit.
3. The new annotated tag on that commit, pushed, with a GitHub Release on it.
4. All 12 val evals scored again from that commit, and the val compare run again, because every eval must descend from the current tag.

After the opening, no amendment exists. Nothing in this path may depend on a test number.
