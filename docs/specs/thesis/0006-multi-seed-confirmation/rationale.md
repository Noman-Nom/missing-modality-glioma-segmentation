# 0006. Rationale

## Context

> ⚠️ Premise note: this design rests on 10 papers out of 387 candidates, and on a manuscript whose publication status is not confirmed. A closer paper could change the claim. The literature gate (AC-19) is there for this reason. Seeds are worth running either way, so waiting would only lose quota.

> ⚠️ Premise note: this topic holds several decisions (claim, seeds, cost test, exposure arm, volume, protocol). They are one spec because they change one thing, the single test opening, and they must be fixed together. The independent dataset is split out into its own spec.

The review found Cai et al. 2026: a compact 3D U-Net on BraTS 2021, all 15 patterns, a score over the 14 incomplete patterns, zero fill, dropout, and dropout with presence channels, with a paired subject bootstrap (`docs/literature/review-notes.md`, note 3.1). So the claim of a first comparison is gone (basis: review notes, section 2). Their limits are clear: 96×96×64, up to 8 epochs, one seed for the presence comparison, and test reuse.

Papers disagree on one question. Cai find a large cost on complete input. Öchsner et al. show equivalence and Chrysochoou et al. a tiny difference, but those two drop only one or two sequences. Spec 0005 reports this number, but it cannot conclude no cost, because it runs no equivalence test (basis: spec 0005, *Statistics protocol*).

Spec 0005 also fixes one training per strategy at seed 42 (its AC-14), and its comparison assumes one model per arm. The test fold is still sealed, so the design may still change, and any change must come before that opening (basis: `AGENTS.md`, the rule that the evaluation protocol is fixed before test results are seen).

The forces are: 30 h of Kaggle quota a week, a journal submission before June 2027, one person, seed 42 runs that passed their checks, and the rule that a new experiment is a new folder (basis: ADR 0002, `AGENTS.md`).

## Options considered

### Option 1: Fix in place, extend spec 0005

Edit `0005-strategies/compare.py` to accept several seeds, and add the new tests there.

**Pros**:
- One comparison script.
- Less code overall.

**Cons**:
- It changes a reviewed file, and its `compare_hash`.
- It mixes finished work and new plans, in a spec still In Progress.
- It breaks the rule that a new experiment is a new folder.

### Option 2: Build alongside, in a new folder (chosen)

A new folder and script for all 12 models, while spec 0005 stays as built for val.

**Pros**:
- Checked runs and reviewed code stay untouched.
- Small, reversible steps, as the strangler pattern advises.

**Cons**:
- Some logic overlaps with `compare.py`.
- Two comparison scripts exist in the repo, one of them retired for test.

### Option 3: Redo everything from one commit

Retrain all 12 models from one new commit.

**Pros**:
- The cleanest story, with no mixed commits.

**Cons**:
- About 25 h more quota.
- It throws away runs that passed all their checks, for no change in the model code path, which AC-3 proves.

## Rationale

Option 2 fits the forces best. The value of a seed 42 run comes from its checks, and those hold if the path is unchanged, so proving that with a test (AC-3) is far cheaper than repeating it (basis: strangler pattern). A new folder also follows the repo rule (basis: `AGENTS.md`).

Averaging per subject keeps one subject as one unit. Pooling every seed as if it were more patients would inflate n and give false confidence (basis: pseudoreplication, a known statistics error). The price is that the interval leaves out seed variation, so `per_seed.csv` shows it instead.

The human chose an equivalence test because the claim needs three possible answers. After the cross check, the human chose one 95% interval to decide each label. It avoids a clash where direction comes from the mean and significance from ranks, and it matches two one sided tests at 2.5% each (basis: TOST). The margin comes from Öchsner et al. The E1 and E3 selection scores on val had already been seen (0.8818 and 0.8699, a gap of 0.0119), and that gap sits just inside the margin, so independence cannot be proven. The tag states this, and each line prints the smallest margin at which equivalence would hold.

E3b gets its own lines so that a secondary arm cannot weaken H2 or H3. Its verdict needs two conditions together, so it uses no correction (basis: intersection union test). It builds on E3 because one change from E3 keeps the comparison clean, and Cai saw no shown benefit from presence channels.

No published network arm: this study asks about training strategies, and a new network would break the rule that arms differ in one thing (basis: `AGENTS.md`, fair comparison). A public git tag with a GitHub Release gives an outside time stamp with no new service, so no ADR is needed. A tag alone is weaker, because its date is set by the client and a tag can be moved. An outside registry would be stronger, and the human can still add one later through an ADR.

### Review fixes, 2026-11-07

The review of `feat/confirmation-study` found no blocker, but four gaps in the rules (`docs/reviews/2026-11-07-feat-confirmation-study.md`, findings 2, 5, 6, 7). All four were settled before the tag. No E3 or E4 val eval had been read, and only seed 42 exists, so no multi seed number could have shaped these rules.

**Seed rule.** Every test runs on subject values averaged over seeds, so n is the number of subjects, and the interval sees no variation between models (basis: AC-11). So one seed can carry a line. RQ1 already answers this with agree or split, and the human chose the same answer for H2, H3 and H4. A plain copy does not work for `equivalent`: with a true zero effect, seed signs are coin flips and agree only 1 time in 4, so a sign rule would make equivalence nearly unreachable. Equivalence claims closeness, so its seed check asks each seed to be close (basis: the logic of TOST). The no worse part of H4 uses the same idea, one sided. That count is `seeds_not_worse`, so a one sided check does not carry the name of a two sided one. The check only adds a condition, so it can only lower false claims (basis: intersection union test). Options weighed: report only, which leaves a `supported` line on one seed; falling back to the old words, which hides that the averaged test passed; a bootstrap over seeds, which means little with 3 seeds.

**Commit tie.** The preflight and the Kaggle gate were not the same gate, and the only link between them was a human pasting 40 characters. The chosen record makes the link a check in code, at the one step that cannot be undone. Options weighed: printing the commit and comparing by hand, which stays a human step. The new test compare check also leaves proof afterwards that all 12 evals used the preflighted commit.

**Keeping the preflight record after the opening.** AC-34 said a test eval on disk means the record must not move, and it guarded `--fold val` for that reason. But `--preflight` deleted the record as its first act, before any guard, so it broke the same rule. A preflight run after step 7 (say, to check something) would refuse, as it should, and the record would be gone. Then `--fold test` refuses with `the preflight did not pass`, and no new preflight can pass, so the final compare is blocked for good. The test `test_the_preflight_refuses_after_the_opening` pinned this by accident. Fixing it in place was chosen. Options weighed: never deleting the record, which is simpler, but a stale pass would then survive a later refusal on the same commit and the file would no longer mean the last preflight passed. Deleting only after the guards ran, which gives the same behaviour with more code. Dropping the record check from `--fold test`, which loses the proof that all 12 evals used the preflighted commit.

The check on disk cannot see test folders that are still on Kaggle, between steps 6 and 7. A dirty laptop tree in that window makes a refused preflight delete the laptop copy, and a passing one on a new commit rewrites it. Step 7 now copies the record from the **Output**, because that copy is what the Kaggle gate read, so it is the honest one. Runner up: a marker file written at the opening, which the laptop would not see either.

**Val compare record.** This record is the only link between the val compare and the opening. Asking for `git_dirty` false has a side effect found while tracing: `runs.toml` is filled in at step 4 and was only committed at step 5, so the normal path would have failed. The order of AC-20 now commits first.

**GPU.** The opening is irreversible, so the check belongs before the read. The GPU name was only known after the gate, so `evaluate.py` works it out earlier.

**Wider seal.** The cross check found that the gate leans on code nothing seals: the dirty check in `runlog.py`, the hash in `cache.py`, and `fold_cases` in `data.py`, which defines the test case set. `rq1_seeds.py` decides RQ1 claims and was also outside the seal. The human chose to add all four. Adding files moves only `compare_hash`, which is stored and not pinned.

## References

**Project sources**:
- `docs/literature/review-notes.md`, sections 2, 3 and 5, for Cai et al. 2026, Öchsner et al. 2026, Chrysochoou et al. 2026, Chen et al. 2026. These are unverified, and none is in `writing/refs.bib` yet.
- Spec 0005, *Statistics protocol*, AC-14, AC-16, AC-20, AC-21.
- Spec 0004, the sealed test path.
- ADR 0002, Kaggle T4 and quota.
- `AGENTS.md`, rules on reproducibility, fair comparison, new experiments and the fixed protocol.

**Practices & standards**:
- Strangler pattern.
- TOST (two one sided tests) for equivalence.
- Holm correction.
- Intersection union test.
- Avoiding pseudoreplication.
- Preregistration.
