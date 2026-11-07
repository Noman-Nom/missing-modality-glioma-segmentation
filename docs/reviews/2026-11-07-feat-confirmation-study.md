# Review, feat/confirmation-study, 2026-11-07

**Reviewed by**: Claude Sonnet 5.5 (author on a different model session; same family, fresh context)
**Scope**: 49 files, branch vs main (merge base 6dfcf0c), plus the parallel literature files left alone
**Verdict**: Approve with nits

## Summary

The branch adds the E3b arm, the volume columns, the test gate inside `check_test_gate`, the `--preflight`, the sealed files, `compare_seeds.py`, `check_confirmation.py`, `rq1_seeds.py` and the Kaggle steps for spec 0006. The code is careful and fails closed in almost every place I traced. I found no blocker. The two questions you asked both come out fine in the main path, with small gaps that are listed below. The most likely real trouble is an untracked file in the checkout making the preflight refuse (finding 1 under Minor).

What I ran: `uv run pytest -q` gave **845 passed, 2 deselected, 94 warnings in 210 s**. `uv run ruff check .` gave **All checks passed**. I did not run slow tests, training, GPU work or any Kaggle step. Nothing here proves real behaviour on Kaggle. The tag `prereg-v1` does not exist yet (`git tag -l` is empty), so the tag based guards were only ever tested on fixtures.

## Question 1: the preflight, val evals at the tag commit, and test-gate.md

**Answer: the preflight handles this window correctly. It does not wrongly fail, and no test run slips through that window. There is one narrow gap, described at the end of this section.**

The real order of events, with the code that runs at each point:

1. Tag `prereg-v1` on commit T. Nothing yet reads a gate.
2. Val evals are scored on Kaggle at T (`KAGGLE.md:130-164`, `:203-220`). For the val fold, `check_test_gate` returns an empty list at once (`experiments/common/evaluation.py:384-385`). So a missing `test-gate.md` cannot stop a val sweep.
3. On the laptop, `check_confirmation.py` and `compare_seeds.py --fold val` run (`KAGGLE.md:289-290`). I traced `guards()` for the val fold (`compare_seeds.py:545-567`). It runs `eval_problems`, which checks the tag ancestry of each eval commit with `descends_from_tag`. T itself counts as a descendant, because `git merge-base --is-ancestor` is true for the same commit (`evaluation.py:349-353`). No code in this path reads `test-gate.md`. `check_confirmation.py` also never reads it (`check_confirmation.py:154-212`). So val evals at T pass, as they should.
4. The human writes `test-gate.md`, commits it, and pins commit P (a descendant of T).
5. The preflight runs on P (`compare_seeds.py:522-542`). It does these things in order:
   - It checks the 12 val evals (`:529`). The val evals carry commit T, not P. The check only asks "descends from the tag", not "equals the current commit", so they pass.
   - It runs the real `check_test_gate` for each of the 12 models (`:534-537`). That call includes `opening_problems` (`evaluation.py:395`), which needs `test-gate.md` on disk, the tag as an ancestor of P, and `compare-val/run.json` with status ok that lists the run.
   - It runs `val_compare_problems` and `seal_problems(P)`. The seal reads the gate with `git show P:docs/literature/test-gate.md` (`compare_seeds.py:496`), so the gate must be committed, not only on disk.
   - It refuses if any `-test` eval folder exists (`:540-542`).
6. Test evals run on Kaggle from P. `evaluate.py` calls the same `check_test_gate` before it reads a case (`experiments/0004-missing-eval/evaluate.py:97-101`).

Checks of the awkward cases:

- Gate file written but not committed: `git_info()` uses `git status --porcelain`, which counts untracked files (`experiments/common/runlog.py:67`). The gate then fails with `git_dirty is True` (`evaluation.py:391-392`). Fails closed.
- Gate present but `status: fail` or fewer than 50 papers: refused by `gate_problems` (`evaluation.py:327-346`). The key checks refuse a missing or repeated line.
- Running `evaluate.py --fold test --open-test` before the gate exists: refused with "the literature gate has not passed" (`evaluation.py:360-361`). This happens before any case is read.
- Preflight run with the val compare missing or not ok: refused (`evaluation.py:364-368`, `compare_seeds.py:507-512`).

**The narrow gap.** The preflight and the Kaggle gate are not the same gate, although `KAGGLE.md:333` says "It runs the same gate that `evaluate.py` runs". The preflight adds `seal_problems(P)` (`compare_seeds.py:539`). `evaluate.py` does not run the seal. So on Kaggle a commit whose `SEALED_FILES` differ from the tag would still open the test, and only `compare_seeds.py --fold test` would notice, after the test numbers exist (`compare_seeds.py:564-565`). Nothing checks that the `COMMIT` pasted into the Kaggle cell is the commit the preflight ran on. This matches spec AC-28 as written (the seal is checked at compare time), so it is not a code bug, but it is weaker than the doc line suggests. See Minor finding 2.

## Question 2: the plain `!` lines in cell 1 of `experiments/0007-confirmation/KAGGLE.md`

**Answer: a failed plain `!` line does not stop the cell. The cell goes on silently. The cell as a whole fails closed later in most cases, but not all, and it can waste hours before it does. It is the same pattern as the proven 0005 cell, so I judge the behaviour known, not new.**

Line by line (`KAGGLE.md:92-120`), based on how IPython and Jupyter run `!` lines:

- **Does a failed `!` stop the cell?** No. A `!cmd` line calls the shell and ignores a non zero exit. The author says the same at `KAGGLE.md:90`. This is how IPython works. I did not run Kaggle, so this is from the IPython contract, not from a test here.
- **Does `!cd` persist?** The cell has no `!cd`. It uses `%cd /tmp/repo` (`:111`), which changes the kernel's own directory and persists into later cells and into later `!` and `sh()` calls. A `!cd` would not persist, because each `!` runs in its own shell. So this is right as written. In step 6 cell 2 the relative lines `!mkdir -p results/...` and `!cp ... results/...` (`:384-385`) rely on that `%cd`, which holds as long as the kernel is not restarted.
- **Do variables and quoting expand?** Yes. `{DATA_TAR}` in `!tar -xf {DATA_TAR} -C /tmp/brats` (`:118`) is expanded by IPython from the notebook namespace. In `sh()`, the f-string is expanded first, then IPython expands the string once more. No command here holds a leftover `{` or `$`, so nothing breaks. `%env` lines set `os.environ`, which `uv` sees.
- **Fail open?** Line by line:
  - `git clone` and `git checkout` use `subprocess.run(..., check=True)` (`:109-110`). A bad clone or a placeholder commit raises and the cell stops. This is fail closed. One hole: `checkout` also accepts a branch name, a tag name or a short hash. The doc says "full 40 character commit" but the code does not enforce it.
  - `!git log --oneline -1` (`:112`) only prints. It asserts nothing.
  - `!pip install -q uv` and `!uv sync --frozen` (`:113-114`) can fail silently. If `uv` is missing, the next `sh("uv run ...")` in cell 3 exits 127 and raises, so it fails closed, but only after the data tar was unpacked. If `uv sync --frozen` fails but `uv` exists, `uv run` in cell 3 syncs again without `--frozen`. That may touch `uv.lock`, which would make the tree dirty. The dirty flag is then caught: `check_confirmation.py` requires `git_dirty` false (`check_confirmation.py:141-145`) and the test gate does too. So it fails closed, but late, after a long training.
  - `!nvidia-smi` (`:115`) only prints. The GPU is checked later by the eval tie out (`evaluation.py:235-237`) and by `SAME_RUN_KEYS` in `compare_seeds.py:84`.
  - `!tar -xf` (`:118`) can fail partway (for example a full `/tmp`) and the cell still ends with exit status ok. `!ls ... | wc -l` always succeeds, so the **1251** count is a manual check (`KAGGLE.md:122`). The version is run with "Save & Run All" and the browser closed (`:87`), so nobody reads that number. The later `preprocess.py --verify` should catch missing data, but I did not read that script's checks, so I mark that as uncertain.
  - `glob(...)[0]` (`:116`) raises IndexError if the dataset is not attached. Fails closed.
- **Comparison with 0005** (`experiments/0005-strategies/KAGGLE.md:40-69`): the same `sh()`, the same `finished()`, the same clone and checkout, the same plain `!` lines. 0007 adds `import shutil` and `COMMIT` only. The E3 and E4 runs passed on `db5d56f` with this pattern, so the success path is proven. The failure path of `sh()` (that `_exit_code` is set and read right) has no evidence in the repo that it ever fired. I mark that as uncertain, though IPython sets `_exit_code` in `system_piped`, as far as I know.

See Minor finding 3 for the fix.

## Blockers

None found.

## Major

None found.

## Minor

### 🟡 1. Untracked files make the preflight refuse, and the docs say "tracked", `KAGGLE.md:52`, `KAGGLE.md:321`, `experiments/common/runlog.py:67`

**Problem**: The pin step says "`git status --short` shows nothing tracked as changed". But `git_info()` marks the tree dirty for any line of `git status --porcelain`, and that includes untracked files (`??`). `check_test_gate` then fails with `git_dirty is True` (`evaluation.py:391-392`). Right now the checkout holds five untracked files from the parallel literature chat (`candidates.csv`, `review-notes.md`, `search-log.csv`, `latexmkrc`, and the `Brain_MRI_...md` file). AGENTS.md says parallel chats share one checkout, so this will probably happen again.
**Why it matters**: The preflight will refuse for a reason the doc told the human not to worry about, and the message says "commit first", which does not point at an untracked file. It fails closed, so no harm, but it costs time at the worst moment. The same holds on the tag step at `:52`.
**Suggested fix**: Change both doc lines to "`git status --short` prints nothing at all (untracked files count)". Optionally make the gate message say which kind of dirt it saw.

### 🟡 2. The Kaggle opening gate does not run the seal check, `experiments/common/evaluation.py:395`, `experiments/0007-confirmation/compare_seeds.py:539`, `experiments/0007-confirmation/KAGGLE.md:333`

**Problem**: `KAGGLE.md:333` says the preflight "runs the same gate that `evaluate.py` runs". It runs that gate plus `seal_problems` and `val_compare_problems` with the compare hash. On Kaggle, `check_test_gate` checks the tag ancestry, the gate file, and that `compare-val/run.json` is ok and names the run. It does not check `compare_hash`, nor that sealed files equal the tag. `protocol_hash` only covers `PROTOCOL_FILES` and the listed source functions, not `strategies.py`, `model.py`, `evaluate.py` or `compare_seeds.py`. Nothing checks that the Kaggle `COMMIT` equals the commit that passed the preflight.
**Why it matters**: A mistyped or later `COMMIT` that changes a sealed file still opens the test. The sealed file check at `compare_seeds.py:564-565` then refuses, but only after the test numbers exist, which is what the protocol wants to avoid. The risk is an accident, not a cheat, and it is small because the same 40 characters are pasted. It matches spec AC-28 as worded.
**Suggested fix**: Either reword `KAGGLE.md:333` to say the preflight is stricter, and add a line to compare the cell 1 `COMMIT` with `git rev-parse HEAD` printed by the preflight; or call the seal check from `check_test_gate` too. If you change a sealed file, repin `EXPECTED_PROTOCOL_HASH` and `SEALED_HASH` before the tag.

### 🟡 3. Cell 1 can run on the wrong or half built environment, `experiments/0007-confirmation/KAGGLE.md:112-119`

**Problem**: See Question 2. `!pip install`, `!uv sync --frozen` and `!tar` fail silently. `COMMIT` is not checked to be a 40 character hash and the 1251 count is not asserted in code.
**Why it matters**: Most failures are caught later, but each notebook version costs about 9.5 h of a weekly 30 h quota. A silent early failure that is only caught in cell 3, or as a dirty flag after training, can waste a lot.
**Suggested fix**: Use `sh()` for `pip install`, `uv sync --frozen`, `mkdir` and `tar`. After the checkout, assert `len(COMMIT) == 40` and that `git rev-parse HEAD` equals `COMMIT`. Replace the manual count with a check that raises unless the count is 1251. Use `sh()` for the step 6 `mkdir` and `cp` lines too (`:380-385`). Keep plain `!` only for print only lines.

### 🟡 4. A 12 hour session leaves little room, and a killed version may lose its Output, `experiments/0007-confirmation/KAGGLE.md:364`, `docs/architecture/decisions/0002-kaggle-t4-training.md:65`

**Problem**: Measured numbers from the repo: E1, E3 and E4 train in 7.38, 7.64 and 7.52 h (`wall_time_s` in their `run.json`), and a 114 case val sweep takes about 3800 s (predict plus metric seconds). So a step 3 notebook is about 7.6 h train, 1.1 h val sweep, plus cache and setup, which fits the 9.5 h estimate and leaves about 2.5 h. A step 6 batch is 4 sweeps at about 2.1 h (228 cases, scaled from val; a projection) which is 8.4 h, and the ADR leaves 3.6 h for unpacking the full BraTS tar, `uv sync` and the test cache. I could not find a measured time for that setup.
**Why it matters**: If a version reaches the limit, `KAGGLE.md:364` itself says it "may lose its Output". Then a step 6 batch restarts from zero, and a step 3 notebook loses up to 9.5 h of training (`:237`). The resume paths for both exist and look right, but they depend on the Output surviving.
**Suggested fix**: Time the setup of the first step 2 notebook (cache build and unpack) and write the real number in the ADR before step 6. If setup plus 4 sweeps is above about 10 h, use `SIZE = 3` (4 batches). Note that this is a projection, not a measurement, so I rate it uncertain.

### 🟡 5. The val compare record is trusted by name only, `experiments/0007-confirmation/compare_seeds.py:503-519`, `experiments/common/evaluation.py:364-371`

**Problem**: `compare-val/run.json` is checked for status ok, run ids, `compare_hash`, and `protocol_hash` and `splits_sha256` of the evals. It is not checked for `fold == "val"`, `prereg_tag`, `git_dirty`, or that the val evals on disk are the ones it read (there is no hash of the eval files in it). The Kaggle gate checks even less (status and run id only).
**Why it matters**: A val eval rescored after the val compare, or a compare made on a dirty tree, would still pass the preflight. Low chance, but this record is the only link between the val compare and the opening.
**Suggested fix**: Add `fold == "val"`, `prereg_tag == PREREG_TAG` and `git_dirty is False` checks. Optionally store a sha256 of each `metrics.csv` in the compare record and check it in the preflight.

### 🟡 6. GPU is not checked on the test fold, `experiments/common/evaluation.py:309-321`, `experiments/0007-confirmation/KAGGLE.md:85`

**Problem**: `KAGGLE.md:85` says "the checks refuse a different GPU". That holds for val (`_tie_out`, `evaluation.py:235-237`) and for resumes. For a fresh test sweep, `check_rules` has no GPU check, and `check_test_gate` has none. `compare_seeds.py` only requires that the 12 test evals share one `gpu` value (`SAME_EVAL_KEYS`, `:85`) and does not compare it with the val GPU.
**Why it matters**: A test sweep on a P100 would pass all checks if all 12 used it. This is the one irreversible opening.
**Suggested fix**: In the gate or in `check_rules` for test, compare the current GPU name with the val eval's `gpu`, or add the val GPU to `SAME_EVAL_KEYS` across folds.

### 🟡 7. H2, H3 and H4 verdicts ignore seed agreement, `experiments/0007-confirmation/compare_seeds.py:403-445`

**Problem**: Every test runs on the 3 seed mean per subject (`averaged`, `:235-240`), so n is the number of subjects and the bootstrap and Wilcoxon only see subject variation. `h3_verdict` prints `seeds_agree` but `cost_label` ignores it (`:385-393`). `h2_verdict` and `h4_verdict` print no seed information (`:420-445`), though `per_seed.csv` has it. Only RQ1 has an agree or split rule (`rq1_seeds.py:68-77`).
**Why it matters**: A line can say "supported" or "gain" when one seed drives it. With 3 seeds, the model to model variation is the main risk to a claim of a strategy benefit (AGENTS.md: claim a benefit only if a paired comparison supports it). The design follows the spec, so this is a gap in the spec's rule, not a coding error.
**Suggested fix**: Before the tag, decide and write in the spec how the thesis may word an H2 or H4 line when `per_seed.csv` shows a split. Adding `seeds_agree=k/3` to the H2 and H4 lines is cheap. Note that H4 uses an uncorrected alpha over 3 regions and H2 corrects only within each (metric, region); this is as specified, but say so in the thesis.

### 🟡 8. `rq1_seeds.py` does not check the evals are E1 runs, `experiments/0007-confirmation/rq1_seeds.py:44-61`

**Problem**: It checks fold test, status ok, and that the `source_seed` values are 42, 43, 44. It does not check that each eval is of an E1 model, or that the three share a protocol hash or commit.
**Why it matters**: Passing the E3 eval of seed 43 by mistake would give an "agree" line for RQ1 built on mixed arms.
**Suggested fix**: Read each eval's source config (or the run) and require `strategy_name` to be None, and require one `protocol_hash` across the three.

### 🟡 9. KAGGLE.md says nothing is read if the gate refuses, but the test cache is built first, `experiments/0007-confirmation/KAGGLE.md:362`, `:395`

**Problem**: Step 6 cell 3 runs `preprocess.py --folds test` (reads and caches the test volumes) before cell 4 runs the gate inside `evaluate.py`. So a refusing gate leaves test volumes read and cached, although line 362 says "If the gate refuses, nothing has been read". Line 43 states the truth (step 8 reads cases).
**Why it matters**: A wording issue, but this is the sealing promise. No prediction or metric exists at that point, so I do not call it a leak.
**Suggested fix**: Reword line 362 to "no prediction or score has been made". Optionally run the preflight's gate part on Kaggle in cell 2, before the preprocess, which would make the sentence true.

## Nits

- ⚪ `experiments/0007-confirmation/KAGGLE.md:106`, `finished()` opens the file without closing it. Same text as in 0005.
- ⚪ `experiments/0007-confirmation/KAGGLE.md:19-24` and `:196`, `RUNS` means the manifest path on the laptop and the output folder in the Kaggle cell 2. Harmless but easy to confuse. A different name would help.
- ⚪ `experiments/0007-confirmation/compare_seeds.py:201-203`, `eval_problems` catches only `OSError` around `check_rules`. A `KeyError` from a damaged `run.json` would be a traceback, not a listed error. Fails closed.
- ⚪ `experiments/0007-confirmation/compare_seeds.py:309`, the H4 primary row also computes a Wilcoxon test that is never used. Not wrong, but it is one more number that could be misread in `summary.csv`.
- ⚪ `experiments/common/evaluation.py:71`, `COMPARE_VAL` is fixed to the repo path while `compare_seeds.py` has `--out`. Run with a non default `--out`, the preflight and the gate would look in different places.

## Strengths

- The test gate is shared: `check_test_gate` is called by `evaluate.py` and by the preflight, so the preflight cannot drift from the real gate. The gate runs before any case is read (`evaluate.py:97-101`).
- Most guards fail closed with a clear message: unknown tag, unknown commit, unreadable git, a repeated gate key, a non numeric `papers_read`, a test eval already on disk, a quick run, a dirty tree.
- `compare_seeds.py` writes into a `.partial` folder and renames at the end, refuses to replace a finished folder without `--rerun-reason`, and refuses non finite numbers (`:633-653`, `:479-486`, `:448-452`). Pairing is by sorted subject on one shared bootstrap index matrix, so all comparisons use the same resamples (`:618`).
- Seed handling is strict: each arm's configs must differ only in seed, each seed's arms only in `[strategy]`, and the train and val cases, GPU and cache hash must match across all 12 (`:181-191`). Resumes are counted and recorded (`:586`).
- **The finished E2, E3 and E4 runs are not changed by this branch.** I checked `git diff db5d56f HEAD` for `experiments/common`, `0003-baseline`, `0004-missing-eval` and `0005-strategies`. Between the E3/E4 runs and now, only `evaluation.py` (volume columns, gate, seal constants) and `strategies.py` (added E3b, `expected_draw_counts`) changed, and `training.py` and `model.py` did not change at all. The E3 and E4 draw functions and hooks are the same code with the same RNG calls (`strategies.py`: `draw_patterns` unchanged, `_drop` gets a default argument). Config hashes come from the config content (`strategy_problems` only adds new rejections for configs with `complete_share`, which E3 and E4 do not have), so they do not move. What does move is `protocol_hash()`, because `evaluation.py` is a sealed protocol file. Old evals carry the old hash, and that is why seed 42 is rescored in step 2 and tied out by AC-26. The rescore tolerance is not too tight: the existing E1, E3 and E4 val evals match their `metrics_val.csv` at pattern 1 with a maximum Dice difference of exactly 0.0 on the same T4 (I computed this from the files in `results/`).
- The E3b config differs from `e3.toml` by two lines, `name` and `complete_share` (I diffed them), and a check enforces it (`check_confirmation.py:121-128`).

## Test coverage

Configured. The fast suite is large (about 3,000 new test lines in this branch: `test_confirmation_compare.py`, `test_strategy_compare.py`, `test_common_strategies.py`, `test_strategy_runs.py`, `test_missing_eval.py`, `test_fetch_papers.py`) and it passes here: 845 passed, 2 deselected. A test keeps `EXPECTED_PROTOCOL_HASH` equal to the sealed hash (`tests/test_common_strategies.py:195`). I did not read all of the test files. What is not covered, and cannot be before the tag: the real `prereg-v1` tag on a real repository, a real Kaggle session, the IPython `_exit_code` failure path in `sh()`, and the upload subset (only `config.toml`, `run.json`, `metrics_val.csv` and `best.pt` per run, `KAGGLE.md:345-346`) being enough for the Kaggle gate. The preflight runs on the full laptop run folders and does not exercise that subset. If a file is missing there, the gate refuses before any case is read (`evaluation.py:406-408`), so it fails closed, but it would cost a Kaggle session to find out. I rate that uncertain.

Not reviewed in depth: `make_table.py` (both), `fetch_papers.py`, the writing files, and the spec 0003 to 0005 text beyond what the questions needed.
