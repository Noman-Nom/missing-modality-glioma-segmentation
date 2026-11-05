# 0005. Modality dropout (E3) and presence indicator (E4): rationale

The decision record behind [index.md](index.md). `/develop` skips this file.

The engineer asked for every recommended option to be taken with no interview, so each call below is the recommendation, made on 2026-11-05 and open to change. No test number exists at this date.

## Context

> ⚠️ Premise note: "Presence indicator" on its own is not a strategy that can be tested against zero fill. A model trained on complete input would see all four presence channels at 1 on every step, so it learns nothing from them, and they would become an untrained signal at scoring time. The right framing is that E4 is modality dropout plus presence channels, and the indicator's own value is read from `E4-E3`, not from `E4-E1`.

RQ2 asks which practical strategy works best under comparable conditions. E2 gave the reason to ask. On val, the E1 model keeps WT Dice at 0.916 with complete input, but falls to 0.003 with T1ce alone and 0.012 with T1 alone, and ET falls to 0.019 with T1 alone. These are development numbers. The model has never had to work without a sequence, so it has only one way to fail.

Five forces shape the problem.

**Fairness.** `AGENTS.md` requires the same subjects, splits, metrics and seeds, and a paired comparison before any benefit is claimed. Any second copy of the training or scoring code is a place where a difference could creep in unseen.

**The seal.** E2 sealed the protocol. `protocol_hash` covers `common/evaluation.py`, `metrics.py`, `missing.py`, `stats.py`, `analyse.py`, the eval config, and `predict()` with four other sources. A byte change to any of them closes the test gate until E1's val eval is rerun, which costs 1.06 h and leaves a mark on the record. The current tree still hashes to the finished eval's value (`146635cf...`, checked 2026-11-05).

**What the earlier specs assumed, and what the code does.** Reading the code found four gaps. Spec 0003 says E3 and E4 reuse the loop unchanged and that selection over patterns needs no change to `common/`. But `_val_point` always calls `selection_score(rows)` at pattern 1, whatever `val_patterns` holds. And `validate()` has no image hook, so an 8 channel model would crash at the first val point. `check_rules` needs a `metrics_val.csv` in each source run, and only the E1 script writes that. Spec 0004 lets `build_model` change, but `build_model` reads no config for channels yet.

**Padding.** `predict()` is sealed and pads with zero when a case is smaller than the window. The laptop's manifest holds only 3 of the 996 cached cases, so nobody can say from here that all crops reach 128 on every axis, and the test fold's shapes must not be inspected for this. For E1 and E3 a zero in padding is plain background. For a presence channel a zero means missing.

**Cost.** Measured on the T4: 0.60 s per training step, 7.4 h for E1's training and val, about 8.2 h for the whole notebook, 256 s per val point at one pattern, and 3,800 s for the 15 pattern sweep on 114 cases (`predict_seconds` plus `metric_seconds`). The weekly quota is still not recorded.

One thing is not a force here. `amp` stays false and the loss stays Dice plus BCE, because both are settled in spec 0003.

## Options considered

### Option 1: separate training and scoring scripts for E3 and E4

A new `train.py` and `evaluate.py` in the new folder, sharing only `common/`.

**Pros**:
- Follows the folder rule to the letter.
- E1 and E2 scripts stay untouched.

**Cons**:
- About 230 lines copied (guards, resume, gate, run record), so two paths must stay equal by hand.
- `resume_problems`, `run_cases` and the rest sit in E1's script, and experiments may not import each other, so they would first move into `common/`.
- The evaluate guards cannot move into `common/evaluation.py`, because it is sealed.

### Option 2: config driven hooks on the one path, with a new comparison (chosen)

A `[strategy]` section selects hooks from a new `common/strategies.py`. `train.py` and both `evaluate.py` scripts read it from the run's config and pass the hooks in. Only configs, `compare.py`, `check_strategy.py` and `make_table.py` are new.

**Pros**:
- One path for all three, so the only code difference is the hook.
- About 10 changed lines across the existing scripts, none of them sealed.
- A resume or a later sweep reads the strategy from the run folder, so it cannot pick the wrong one.

**Cons**:
- Scripts in the E1 and E2 folders now serve E3 and E4.
- `common/model.py` gains a strategy branch.
- Touches `train.py` and `training.py`, which the E1 tests guard.

### Option 3: move all CLI code into `common/`, with thin scripts in every folder

**Pros**:
- Cleanest final layout, and honours the folder rule fully.

**Cons**:
- The largest refactor of tested code, right before a seal sensitive run.
- `evaluate.py`'s guards would need a new module anyway.
- Nothing about the science gets better.

### Option 4: E4 only through a conditioning layer (FiLM)

Feed the presence vector to a small layer that scales features inside the network (feature wise modulation). Input stays 4 channels.

**Pros**:
- Stronger conditioning, deep in the network.
- No padding question at all.

**Cons**:
- `predict()` is sealed and calls `model(x)` with one tensor, so the vector would need a side channel into the model.
- More new code and more parameters, against a compact model.
- Tests a larger change, so `E4-E3` measures more than the presence signal.

## Rationale

**Option 2** wins on the fairness force: the claim that only the strategy differs is true by construction, not by review. The folder rule (`A new experiment is a new folder`, and experiments share code only through `common/`) is met in the way that matters here: the new folder holds what is new, and no experiment imports another. Running the E1 and E2 scripts as commands is a reading of that rule, so `experiments/AGENTS.md` states it openly. Option 3 is the honest alternative if you read the rule more strictly, at a real cost before a sealed run.

**E4 is E3 plus presence, so E3 is its control.** From the premise note. It also gives the sharpest test in the study: same draws at seed 42, and the same patches unless a run resumed (a resume reseeds the loader), so `E4-E3` isolates the channels.

**Uniform over all 15 patterns.** Two independent coin flips per modality at 0.5, redrawing the empty set, give the same thing, so this is the plainest form of the idea. It matches how results are scored, since every pattern counts equally. Above all it has no parameter, so there is nothing to tune on val and nothing to defend. The runner up was half the steps complete, half uniform over the 14 incomplete, which would likely protect complete input better, but adds a number whose only honest source is a tuning run E3 does not have budget for. The cost, 1 complete step in 15, is reported as a finding.

**Selection at complete input, same as E1.** Spec 0003's user story asks for the same selection rule, and spec 0004 left this open. Selecting over all 15 patterns would add 15 × 256 s at 10 val points, about 10.7 h per run, and it would need a change to `_val_point`. The runner up was a panel of 5 patterns (1 plus the 4 single modality ones), about 3.6 h more per run. E1's late curve is flat under the cosine schedule (0.8818 at 36,000, 0.8806 at 40,000), so selection is not where the difference will come from, though that is one model's curve, not proof. The risk of this choice works against the strategies, so a benefit that survives it is conservative.

**Presence as 4 input channels.** It is the path spec 0004 prepared (`image_hook`, and `predict()` already takes any channel count), and it is the smallest change that gives the network the signal. Option 4 is the runner up.

**`PresenceUNet`, not a guard.** Padding cannot be ruled out, so the design must hold if it happens. A guard would need test fold shapes before the opening. Taking the patch maximum is three lines, costs nothing, and is safe because a window always overlaps real voxels: padding happens only where the image is smaller than the window on that axis. The alternatives each fail on one side: flipping the coding (1 for missing) only moves the error to the missing patterns, and training with padded presence would need a change to the sealed transforms.

**Tie out through the E1 script.** `check_rules` is sealed and needs `metrics_val.csv`. The runner up was deriving that file from `val_log.csv` with no GPU, which is also an independent path, but gives no HD95 and skips the file format `check_baseline.py` reads. The E1 `evaluate.py` takes one changed line and about 5 minutes per run, and then `check_baseline.py` works on E3 and E4 unchanged, which is a stronger result. This changes spec 0004's invariant that `0003-baseline/evaluate.py` stays unchanged, so a dated note goes there. The reason for that invariant, no read of test, still holds.

**Primary score over the 14 incomplete patterns.** RQ2 is about missing input, so the primary number excludes pattern 1. One score per subject and region gives 9 tests with a single clear reading. Testing every pattern as primary would give 270 tests, and an answer only a table could carry. Those stay as secondary, to show where a gain sits. Equal weight on the 14 patterns is a choice: it counts T1 alone the same as a single missing sequence. Reporting by `n_missing` shows what that hides. Holm per (metric, region) keeps the family rule of spec 0004. Three comparisons are tested because each one answers something different: does dropout help, does the pair help, does the indicator add anything.

**Winner's curse guarded by order, not by a split.** Nothing is selected between E3 and E4 on val, so the test fold reports all three and no model is dropped. That is why the val comparison is a dry run of the code, and may decide nothing.

**`compare_hash`.** `compare.py` and `strategies.py` are new and outside `protocol_hash`, and they cannot join it without editing the sealed `evaluation.py`. So the comparison carries its own hash. It closes, for RQ2, the gap spec 0004 accepted for `evaluate.py`. It is only checked at the end, so it stops a changed comparison, not a changed sweep. The commit pin in `KAGGLE.md` covers the rest.

**A versions mismatch warns, it does not refuse.** A refusal at that point would throw away a test opening for a cause that cannot be undone. A logged flag lets a human judge it, and the thesis can state it.
