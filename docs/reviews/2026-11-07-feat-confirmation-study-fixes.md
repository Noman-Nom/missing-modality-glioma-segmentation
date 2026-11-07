# Review, feat/confirmation-study (fixes commit 9baacd4), 2026-11-07

**Reviewed by**: Claude Sonnet 5.5 (fresh context; the fixes were written in another session)
**Scope**: 54 files on the branch, branch vs main. Main focus is commit `9baacd4`, 17 files, the code that had no review. The two later commits only touch docs and the proposal and were skimmed.
**Verdict**: Approve with nits

## Summary

The commit does what the spec says. The seed rule (AC-32), the seal inside the Kaggle gate (AC-33), the `preflight.json` record (AC-34), the full val compare record (AC-35), the GPU check (AC-36) and the repin (AC-37) are all in the code and match the spec text line by line. I found no blocker and no major. The fixes for findings 1, 3, 8 and 9 of the last review work. The new code adds only small gaps, listed below. I did not rerun the suite (you said 941 passed and ruff is clean) and I ran no training, GPU work or Kaggle step.

Every finding below says whether the file is sealed and what fixing it costs in repins. A short rule first. `compare_hash()` is computed from the files on disk, so changing any sealed file before the tag needs no repin by hand. It only means the val compare has to be run after the change. `EXPECTED_PROTOCOL_HASH` (in `compare_seeds.py`) and `SEALED_HASH` (in `tests/test_common_strategies.py`) must be repinned only if a file in `PROTOCOL_FILES` changes, which includes `experiments/common/evaluation.py`.

## Did the fixes fix findings 1 to 9?

- **Finding 1 (untracked files)**: fixed in the docs (`KAGGLE.md:56`, `:349`, the step list) and in code. `val_side_problems` (`compare_seeds.py:530`) now tells you that an untracked file counts. See minor finding 4 for the side effect.
- **Finding 2 (Kaggle gate has no seal)**: fixed. `opening_problems` calls `seal_problems` and `val_record_problems` (`evaluation.py`), and `site_problems` ties the commit to `preflight.json`. The `KAGGLE.md` line now says the preflight is the same gate plus two rules that only Kaggle can check, which is true.
- **Finding 3 (silent `!` lines)**: fixed. Cell 1 uses `sh()` for install, sync, `mkdir` and `tar`, asserts 40 characters and `rev-parse HEAD == COMMIT`, and raises unless the count is 1251. `os` and `glob` are imported at the top of the cell and `BRATS_TRAIN_DIR` is set with `%env` before it is read, so the new lines do not break. The failure path of `sh()` has still never run on Kaggle (unchanged from the last review).
- **Finding 4 (session time)**: still open by design. It waits for the real setup time.
- **Finding 5 (val compare record trusted by name)**: fixed by AC-35. Every rule I looked for is there: fold, tag, dirty flag, ancestry, `compare_hash`, and a sha256 of each val `metrics.csv`.
- **Finding 6 (GPU)**: fixed. `gpu` joined `VAL_MATCH_KEYS` and `site_problems` compares the live GPU with the val eval.
- **Finding 7 (seed agreement ignored)**: fixed. The `split` word is applied only to a line that was already a claim. I checked `h2_verdict`, `h3_verdict` and `h4_verdict` against the AC-32 table. Strictness, levels and which count guards which word all match. `cost_label` is untouched.
- **Finding 8 (`rq1_seeds.py` and E1)**: fixed. See nit 2 for one small edge.
- **Finding 9 (false sentence)**: fixed by rewording (`KAGGLE.md:380`). The test volumes are still cached before the gate runs. See minor finding 5.

I also checked the record kept after the opening. `main` deletes `preflight.json` only for `--fold test` with no `*-test` entry (`compare_seeds.py:668`). `guards` adds one line per `*-test` entry before any early return. A wrong `--fold val --preflight` returns before touching the file. Step 7 copies the Kaggle copy back over the laptop copy. I found no path where the record is lost by accident.

## Blockers

None found.

## Major

None found.

## Minor

### 🟡 1. `site_problems` can crash on a damaged record and reads a fixed path, `experiments/common/evaluation.py:444-458`, `:80`
**Problem**: `json.loads(path.read_text())` at line 454 has no guard, so a half written or hand edited `preflight.json` gives a traceback instead of a listed error. `compare_seeds.py:570` has the same pattern. Also `COMPARE_VAL` is fixed to the repo path, while `compare_seeds.py` takes `--out`, so a non default `--out` makes the preflight write the record where the Kaggle gate does not look. The second point was nit 5 of the last review and is still there.
**Why it matters**: It fails closed, before any case is read, so there is no harm. It only costs a confusing message at the one moment you do not want one.
**Suggested fix**: Catch `json.JSONDecodeError` and `OSError` and add a listed error. Optionally have the preflight refuse a non default `--out`.
**Sealed?**: yes, `evaluation.py` is a protocol file and a sealed file. **Repin?**: yes, `protocol_hash()` moves, so `EXPECTED_PROTOCOL_HASH` and `SEALED_HASH` must be repinned in the same commit. I think this is worth doing only if you are already touching `evaluation.py` for something else. `compare_seeds.py:570` alone would not need a repin.

### 🟡 2. `rq1_seeds.py` does not tie the three evals to the seal or the preflight commit, `experiments/0007-confirmation/rq1_seeds.py:42-62`
**Problem**: The fix for finding 8 checks the arm (E1, from `runs.toml`), the fold and that the protocol hash is the same across the three evals. It does not check that the shared hash equals the pinned one, that `git_commit` is the preflight commit, or that `verdict.txt` was made from this eval by `analyse.py`.
**Why it matters**: The three real evals will pass `compare_seeds.py --fold test` first, which does check these. So this only matters if someone runs `rq1_seeds.py` on a set that `compare_seeds.py` never saw. Low risk.
**Suggested fix**: Optional. Reject a hash other than `protocol_hash()` and a commit that differs across the three evals.
**Sealed?**: yes (`SEALED_FILES` lists it). **Repin?**: no. Only `compare_hash` moves, which is computed live, as long as you change it before the tag.

### 🟡 3. The record protection cannot see test evals that sit only on Kaggle, `experiments/0007-confirmation/compare_seeds.py:524-527`, `:668`
**Problem**: `opened_folders` looks in `--evals` on the laptop. During step 6 the test folders are on Kaggle, so a preflight rerun on the laptop between batches is allowed and rewrites the laptop copy. The docs say so, and step 7 restores the right copy from the Output. If you forget that copy, `preflight_record_problems` refuses the compare, so it fails closed.
**Why it matters**: A documented and closed gap, not a hole. I mention it so you read the step 7 copy as required, not optional.
**Suggested fix**: None needed in code. Maybe add "do not run the preflight again after batch 1" to the step 6 intro in `KAGGLE.md`.
**Sealed?**: `KAGGLE.md` is not sealed. No repin.

### 🟡 4. A stray file from another chat now also blocks the val compare, `experiments/0007-confirmation/compare_seeds.py:530-537`
**Problem**: `val_side_problems` refuses `--fold val` on any untracked file (`git status --porcelain`). AGENTS.md says parallel chats share one checkout, and the last review found five such files. The tree is clean right now, but `bf35445` and `90713a5` show the literature work lands in the same checkout.
**Why it matters**: The message names the cause ("an untracked file counts too"), so the fix is quick. It is the right trade, because the record must say `git_dirty: false`. I list it so it is not a surprise at step 5.
**Suggested fix**: Before step 4 and step 7 of the list, run `git status --short` and commit or move anything that shows.
**Sealed?**: no change needed.

### 🟡 5. Test volumes are cached before the gate runs, `experiments/0007-confirmation/KAGGLE.md:380`, `:413`
**Problem**: Finding 9 was fixed by rewording, and the new sentence is true. The order is unchanged: cell 3 (`preprocess.py --folds test`) reads and caches the test volumes, then cell 4 runs the gate. A refusal after cell 3 leaves a test cache on Kaggle, with no prediction or score.
**Why it matters**: No leak, since no model has seen a case. It is only the "nothing was read" promise that is weaker than it sounds, and the text no longer claims it.
**Suggested fix**: Optional. Run the gate part of the preflight in cell 2, before cell 3, so a refusal costs nothing.
**Sealed?**: no (`KAGGLE.md`). No repin.

## Nits

- ⚪ `experiments/0007-confirmation/rq1_seeds.py:54`, if an eval lacks both `source_run_id` and `source_seed`, `None == e1_ids.get(None)` passes this check. The seed list check a few lines below still refuses it, so it fails closed. Sealed, no repin.
- ⚪ `experiments/common/evaluation.py:50-55`, nothing automated proves that `predict`, `sweep` and `region_metrics` are unchanged (AC-37 allows "a diff shows they were not touched"). The diff of `9baacd4` shows they were not, and `protocol_hash()` seals their source. No action.
- ⚪ `experiments/0007-confirmation/compare_seeds.py:524`, `opened_folders` matches `*-test` only. A `-test-quick` folder does not count, but `check_test_gate` refuses `--quick` on the test fold anyway.

## Strengths

- The seed rule is a small, well fenced change: it adds a condition and removes none, runs only on a word that was already a claim, and prints its counts on every line. It matches the AC-32 table in every cell I checked.
- The seal and the val compare record moved into the shared gate, so the preflight and Kaggle cannot drift apart. `opening_problems` returns a listed error for every missing file before hashing it, in the order the gate needs.
- The preflight record logic is careful: the delete happens first, only for `--fold test` and only with no opened entry. `--fold val --preflight` touches nothing. Docs, spec, verify list and tests all describe the same behaviour.
- `KAGGLE.md` cell 1 now fails at once on a bad commit, a failed install or a wrong case count, and the new lines use names that are defined (`os`, `glob`, `BRATS_TRAIN_DIR`).
- `make_table.py` still decides nothing. The regexes read the new `split` word and the new counts from `verdict.txt`, and a dagger marks a split Dice cell.

## Test coverage

Configured. `tests/test_confirmation_compare.py` (+526 lines) and `tests/test_missing_eval.py` (+142 lines) cover the new rules: AC-33 through the real `evaluate.py` gate, AC-35 each rule alone and a record with nothing in it, the val files check, AC-36 on fresh and finished sweeps, and the record after the opening. I read the headers and the AC mapping, not every case. What cannot be covered before the tag: the real `prereg-v1` tag, a real Kaggle session, and the `sh()` failure path. `site_problems` with a damaged JSON record (minor 1) has no test, which matches that it is unguarded.
