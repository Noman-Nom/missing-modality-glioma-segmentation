# Running the confirmation study on Kaggle

This file holds every human step of spec 0006, from the public tag to the end. You run all of them. Training and scoring run on the Kaggle Tesla T4, and the comparisons run on the laptop with no GPU. The steps depend on each other, so we recommend doing them in order.

**Nothing below is a result yet. No test number exists, and the test fold has not been opened.**

| Step | Where | What | Planned GPU time |
|---|---|---|---|
| 1 | Laptop | The tag and the Release | none |
| 2 | Kaggle | Score seed 42 of E1, E3 and E4 again | about 3 h |
| 3 | Kaggle | Train and score 9 new models, one notebook each | about 9.5 h each, about 85 h |
| 4 | Laptop | Copy back, checks, val compare | none |
| 5 | Laptop | Literature gate, pin the commit, preflight, upload | none |
| 6 | Kaggle | The one test opening, in 3 notebook versions | about 25 h |
| 7 | Laptop | After the opening | none |

In all, about 114 h. These times are projections from E1 and E3, not measurements. Whole notebooks fit about 3 a week, so plan on about 5 weeks. The real timings land in each `run.json`.

```bash
RUNS=experiments/0007-confirmation/runs.toml
R1="20261103-023817-f9b3afb6"
R3="20261106-024450-10cfa2cc"
R4="20261106-235247-8999a39e"
```

These are the 3 kept seed 42 runs (AC-5). Every other run id holds a time stamp, so you fill it in as each run finishes.

## The order (AC-20)

Each step happens before the next one starts:

1. All 12 runs pass `check_baseline.py`. E3, E4 and E3b also pass `check_strategy.py` against the E1 run of the same seed.
2. All 12 val evals pass `check_eval.py`.
3. `check_confirmation.py` passes.
4. You commit the tree, with `runs.toml` filled in.
5. `compare_seeds.py --fold val` passes, on that clean tree.
6. You write `docs/literature/test-gate.md`, and the gate passes.
7. You commit the tree, and pin that commit.
8. `compare_seeds.py --fold test --preflight` passes, and writes `preflight.json`.
9. You upload the study files.
10. On Kaggle, `preprocess.py --folds test` runs.
11. On Kaggle, `evaluate.py --fold test --open-test` runs for all 12, from that commit.
12. `compare_seeds.py --fold test` runs.

Steps 1 to 9 never read a test case. **Step 10 does, and it cannot be taken back.** The numbers in this list are the 12 steps of AC-20, inside steps 3 to 6 of this file.

## Step 1: the tag and the Release (laptop, no GPU)

**This comes before any new training (AC-8).** Do not run `0005-strategies/compare.py --fold val` first, or the tag message must say so.

Before you tag, we recommend these four things:

- `/check verify` and `/check review`, because a bug found after training costs up to 114 h of GPU time to redo.
- `uv run pytest -q` and `uv run ruff check .` pass.
- The Kaggle quota and the session limit are in ADR 0002. Step 6 depends on the session limit.
- `git status --short` prints nothing at all. Untracked files count, so a stray file from another chat stops the checks too.

Merge to `main` with a merge commit, never a squash. A squash can leave `db5d56f` unreachable, and the seed 42 records need it. Then write the message to a file. Keep only what is true. This template holds what the records say on 2026-11-07, so check each line against what you really saw, above all the E3 and E4 line:

```text
prereg-v1: the protocol of spec 0006, fixed before any new training.

Seen before this tag:
- E1 val numbers, on all 15 patterns, and the E1 selection score 0.8818.
- E3 selection score 0.8699 and E4 selection score 0.8711.
- E3 val eval: one case read on 2026-11-07 for a tie out during verification, BraTS-GLI-00017-000. Shown: the differences from a fresh score, and its predicted WT volume at complete input. No E3 Dice and no aggregate.
- E4 val eval read: no.
- 0005-strategies/compare.py --fold val run: no.
- The test fold has not been opened.
```

Then tag, push and publish:

```bash
git tag -a prereg-v1 -F <the message file>
git push origin main prereg-v1
gh release create prereg-v1 --verify-tag --title prereg-v1 --notes-file <the message file>
git rev-parse prereg-v1            # the tag object: write it in context/progress-tracker.md
git rev-parse 'prereg-v1^{commit}' # the COMMIT of steps 2 and 3
```

The second hash matters. An annotated tag has its own hash, and the notebooks need the commit it points to.

**Steps 2 and 3 all use this one `COMMIT` (AC-5, AC-7).** Do not move it, even for a change that looks harmless. A change after the tag takes the amendment path at the end of this file.

## The notebook settings (steps 2, 3 and 6)

- **Accelerator** set to **GPU T4 x2**, not P100, and **Internet** on. Never use the laptop for a step marked Kaggle, because the checks refuse a different GPU.
- **Add Input**, then **Your Datasets**, then **brats-gli-train**. Each step below names what else it needs.
- Put each cell below in its own notebook cell, in order. Click **Save Version**, choose **Save & Run All (Commit)**, and close the browser if you like.
- When a version finishes, open it, go to the **Logs**, and paste into the main chat the last 2 lines of each command, plus the `nvidia-smi` line and the `git log` line. Then download the **Output**.

**Cell 1, the same in every notebook: code at one exact commit.** `sh()` runs a command and stops the notebook if it fails. A plain `!` line stops nothing, so we keep `!` only for lines that just print. This is a full clone, never `--depth 1`, because the test gate asks git about the tag.

```python
COMMIT = "paste the full 40 character commit here"
%env UV_PROJECT_ENVIRONMENT=/tmp/venv
%env UV_LINK_MODE=copy
%env BRATS_TRAIN_DIR=/tmp/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
import glob, json, os, shutil, subprocess

def sh(cmd):
    get_ipython().system(cmd)
    if get_ipython().user_ns["_exit_code"] != 0:
        raise RuntimeError(f"stopped, this command failed: {cmd}")

def finished(folder):
    path = f"{folder}/run.json"
    return os.path.exists(path) and json.load(open(path))["status"] == "ok"

URL = "https://github.com/Noman-Nom/missing-modality-glioma-segmentation.git"
subprocess.run(["git", "clone", "-q", URL, "/tmp/repo"], check=True)
subprocess.run(["git", "-C", "/tmp/repo", "checkout", "-q", COMMIT], check=True)
HEAD = subprocess.run(["git", "-C", "/tmp/repo", "rev-parse", "HEAD"], check=True,
                      capture_output=True, text=True).stdout.strip()
assert len(COMMIT) == 40, f"COMMIT must be the full 40 characters: {COMMIT}"
assert HEAD == COMMIT, f"checked out {HEAD}, not {COMMIT}"
%cd /tmp/repo
!git log --oneline -1
sh("pip install -q uv")
sh("uv sync --frozen")
!nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
DATA_TAR = glob.glob("/kaggle/input/**/brats-gli-train.tar.bin", recursive=True)[0]
sh("mkdir -p /tmp/brats")
sh(f"tar -xf {DATA_TAR} -C /tmp/brats")
CASES = len(glob.glob(os.environ["BRATS_TRAIN_DIR"] + "/*"))
print(CASES)
if CASES != 1251:
    raise RuntimeError(f"stopped, expected 1251 cases, found {CASES}")
```

The cell stops unless the count is **1251**.

## Step 2: score seed 42 again (one notebook, about 3 h)

The three kept models get their volume columns here (AC-7). We put this before the 9 trainings on purpose: it is the cheapest way to find a problem in the new scoring code. If a score moved, you find out after 3 h, not after 85 h.

**Add Input**, beyond the data: **brats-e1-run**, and the **Output** of the two seed 42 notebooks, `muhammadnomanned/notebooke097442f0d` (E3) and `muhammadnomanned/notebookefdb279372` (E4).

**Cell 2: the three models**

`/kaggle/input` is read only, so the runs are copied to `/tmp`.

```python
R1, R3, R4 = "20261103-023817-f9b3afb6", "20261106-024450-10cfa2cc", "20261106-235247-8999a39e"
SOURCE = "/tmp/seed42"
EVALS = "/kaggle/working/results/0007-confirmation/evals"
RUN_TAR = glob.glob("/kaggle/input/**/e1-run.tar.bin", recursive=True)[0]
sh(f"mkdir -p {SOURCE}")
sh(f"tar -xf {RUN_TAR} -C {SOURCE}")
for R in (R3, R4):
    shutil.copytree(glob.glob(f"/kaggle/input/**/runs/{R}/", recursive=True)[0], f"{SOURCE}/{R}")
!ls {SOURCE}
```

You should see the three run ids.

**Cell 3: the cache and its check**

```python
sh("uv run python experiments/0003-baseline/preprocess.py")
sh("uv run python experiments/0003-baseline/preprocess.py --verify")
```

The last line must read `verify ok: 996 files, 0 problem(s)`. The cache takes about 35 GB of `/tmp`.

**Cell 4: the three sweeps and their checks**

```python
for R in (R1, R3, R4):
    if not finished(f"{EVALS}/{R}-val"):
        sh(f"uv run python experiments/0004-missing-eval/evaluate.py --run {SOURCE}/{R} --fold val --out {EVALS}")
    sh(f"uv run python experiments/0004-missing-eval/check_eval.py {EVALS}/{R}-val")
```

From the **Output**, copy the three `<run id>-val` folders into `results/0007-confirmation/evals/` on the laptop, keeping every file. Then run the rescore proof (AC-26):

```bash
uv run python experiments/0007-confirmation/check_confirmation.py --runs $RUNS
```

It exits 1 at this point, because 9 slots are still empty. Read only its first 3 lines, which should say `AC-26 e1: ok`, `AC-26 e3: ok` and `AC-26 e4: ok`. **If one says `FAIL`, the study stops until the difference is explained.** We recommend bringing it to the main chat, and starting no training.

## Step 3: train and score 9 new models (one notebook each)

Run them in this order (AC-5). E3b goes last, so if quota runs short it can be dropped as a whole, through the amendment path, before any E3b val number is seen.

| Order | `NAME` | `SEED` |
|---|---|---|
| 1, 2 | `e1` | 43, 44 |
| 3, 4 | `e3` | 43, 44 |
| 5, 6 | `e4` | 43, 44 |
| 7, 8, 9 | `e3b` | 42, 43, 44 |

**Add Input**: only the data. E3, E4 and E3b get their fairness check on the laptop in step 4, because it needs the E1 run of the same seed.

**Cell 2: which model**

```python
NAME, SEED = "e1", 43  # one line of the table above
CONFIGS = {"e1": "experiments/0003-baseline/config.toml",
           "e3": "experiments/0005-strategies/e3.toml",
           "e4": "experiments/0005-strategies/e4.toml",
           "e3b": "experiments/0007-confirmation/e3b.toml"}
OUT = "/kaggle/working/results/0007-confirmation"
RUNS, EVALS = f"{OUT}/runs", f"{OUT}/evals"
```

**Cell 3: the cache and its check**

The same as cell 3 of step 2, with the same `verify ok: 996 files` line.

**Cell 4: training**

```python
sh(f"uv run python experiments/0003-baseline/train.py --config {CONFIGS[NAME]} --seed {SEED} --out {RUNS}")
```

**Cell 5: scoring and checks**

```python
RUN = sorted(glob.glob(f"{RUNS}/2*/"))[-1].rstrip("/")
EVAL = f"{EVALS}/{os.path.basename(RUN)}-val"
print(RUN)
sh(f"uv run python experiments/0003-baseline/evaluate.py --run {RUN}")
if not finished(EVAL):
    sh(f"uv run python experiments/0004-missing-eval/evaluate.py --run {RUN} --fold val --out {EVALS}")
sh(f"uv run python experiments/0003-baseline/check_baseline.py {RUN}")
sh(f"uv run python experiments/0004-missing-eval/check_eval.py {EVAL}")
```

The first command is the E1 script. It gives the run its `metrics_val.csv`, which `check_eval.py` compares with pattern 1 of the sweep. Both checks must end in `PASS`.

If a check fails, we recommend stopping there, and bringing the failure to the main chat before you start the next model.

**The repeat rule (AC-6).** Repeat a run only if it failed, or a named check failed, and that includes a score under `BEST_SCORE_MIN` in `check_baseline.py`. Any other low score is never a reason. Write the reason in `context/progress-tracker.md`, and keep the old run id in `runs.toml`:

```toml
[replaced]
20261110-010203-abcd1234 = "e3 seed43: check_baseline failed on <what>"
```

A slot gets at most 2 repeats. A slot that still fails stops the study until the amendment path is used. Every finished seed stays in the analysis.

### After a session dies

Every finished training step and swept pair is on disk, so nothing finished is lost if the output survives. If a dead version shows no **Output**, the run starts again from cell 1, with its reason in the tracker.

1. Make a new notebook (or a new version) with the same settings and inputs. Also **Add Input**: the dead version's **Output**.
2. Keep the **same `NAME`, `SEED` and `COMMIT`**. Run cells 1 to 3 again. The cache lives in `/tmp` and is gone, and rebuilding it gives the same hash.
3. **Do not run cell 4**, because it would start a second run. Put this cell in its place:

```python
OLD = sorted(glob.glob("/kaggle/input/**/results/0007-confirmation/", recursive=True))[-1]
shutil.copytree(OLD, OUT)
RUN = sorted(glob.glob(f"{RUNS}/2*/"))[-1].rstrip("/")
if os.path.exists(f"{RUN}/train.pid"):
    os.remove(f"{RUN}/train.pid")  # a process number from the old machine can match a live one here
if not finished(RUN):
    sh(f"uv run python experiments/0003-baseline/train.py --resume {RUN}")
```

4. Run cell 5 as it is. It repeats the E1 script, which only writes `metrics_val.csv` again, and it resumes a sweep that stopped partway.

A resume is allowed, but it reseeds the loader, so that run's later patches no longer match the other models'. The pattern draws still match. `compare_seeds.py` records the count of resumes per run, so a reader can see it.

## Step 4: copy back, checks, val compare (laptop, no GPU)

From each **Output**, copy `results/0007-confirmation/runs/<run id>/` and `results/0007-confirmation/evals/<run id>-val/` to the same place on the laptop, keeping every file, `run.json` included. Fill in the 9 empty lines of `runs.toml`.

Then, for seed 43 and again for seed 44, set `E1` to that seed's E1 run folder and check the three other arms against it:

```bash
E1=<the E1 run folder of this seed, from runs.toml>
for RUN in <the E3, E4 and E3b run folders of this seed>; do
  uv run python experiments/0005-strategies/check_strategy.py $RUN --e1 $E1
done
```

For seed 42 only E3b is new:

```bash
uv run python experiments/0005-strategies/check_strategy.py results/0007-confirmation/runs/<E3b seed 42> --e1 results/0003-baseline/$R1
```

`check_strategy.py` reads the two run files only. `check_baseline.py` is not repeated, because it reads the cache, which stayed on Kaggle.

Then all 12 val evals and the study check:

```bash
uv run python - <<'EOF' > results/0007-confirmation/runs.txt
import tomllib
manifest = tomllib.load(open("experiments/0007-confirmation/runs.toml", "rb"))
print("\n".join(manifest[a][f"seed{s}"] for a in ("e1", "e3", "e4", "e3b") for s in (42, 43, 44)))
EOF
for RUN in $(cat results/0007-confirmation/runs.txt); do
  uv run python experiments/0004-missing-eval/check_eval.py results/0007-confirmation/evals/$(basename $RUN)-val --run $RUN
done
uv run python experiments/0007-confirmation/check_confirmation.py --runs $RUNS
```

Each check should end in `PASS`.

**Commit before the val compare (AC-10).** The compare refuses a dirty tree, and the test gate later refuses a compare that was run on one. So commit `runs.toml` first, and check that `git status --short` prints nothing at all. `results/` and `writing/tables/` are git ignored, so they stay out of it. Then:

```bash
uv run python experiments/0007-confirmation/compare_seeds.py --runs $RUNS --fold val
uv run python experiments/0007-confirmation/make_table.py --compare results/0007-confirmation/compare-val
```

The compare should write 18 verdict lines. These are development numbers, and the captions say so. They choose nothing, because the comparison is fixed by the spec. A line can read `split`: the averaged test passed, but the 3 seeds did not all back it, so no claim is made on that line (AC-32).

## Step 5: the gate, the pin, the preflight, the upload (laptop, no GPU)

**Stop here for a moment.** Anything you want to add to the study has to go through the amendment path before step 6.

### The literature gate (AC-19)

You write this file by hand once you have read at least 50 papers. It needs all five lines, each once:

```text
status: pass
date: 2027-01-15
papers_read: 50
cai_status: <what you found about its peer review>
citation_search: <where you searched, and when>
```

The date and the count above show the format only. If the check finds a closer paper, we recommend writing `status: fail` and taking the change through `/architect` and the amendment path below, before any test number exists.

### The pin

Before the preflight:

1. Record the val run in `context/progress-tracker.md`: the run ids, the check results and the real timings.
2. Commit the tree, with `runs.toml` filled in and `docs/literature/test-gate.md` in it. `results/` is git ignored, so `compare-val` stays on the laptop.
3. Merge to `main` with a merge commit, never a squash, push, and run `git rev-parse HEAD`. That full commit is the `COMMIT` of step 6. A squash makes the commit of the val compare unreachable, and the preflight refuses that.
4. Check `git status --short` prints nothing at all. The test gate refuses a dirty tree, and untracked files count.

After this point, do not change a sealed file. The sealed files are `SEALED_FILES` in `common/evaluation.py`.

### The preflight

With the tree committed, and `compare-val` still in `results/` on the laptop, run:

```bash
uv run python experiments/0007-confirmation/compare_seeds.py --runs $RUNS --fold test --preflight
```

It should end with `preflight PASS`. It runs the same gate that `evaluate.py` runs for each of the 12 models, the seal included (AC-33). On Kaggle, `evaluate.py` adds two rules that can only be checked there: the commit must be the one the preflight passed, and the GPU must be the one of the val evals (AC-34, AC-36).

On a pass it writes one file, `results/0007-confirmation/compare-val/preflight.json`, which holds the commit. Every run first deletes the old file, so the file always means the last preflight passed. There is one exception: once any `*-test` entry sits in `results/0007-confirmation/evals/`, the opening has happened, so the preflight refuses and the file stays as it is. It is the only proof of the commit the opening used, and no new preflight could pass to replace it. A rerun of the val compare moves it away, so the preflight must pass again. If it refuses, no test case has been read. We recommend bringing the message to the main chat and changing nothing by hand.

### The upload (once)

This comes after the preflight passes, because the upload holds its record. The test notebook needs the 12 run folders, the 12 val evals, the val compare record and the preflight record (AC-25). They go up as one uncompressed tar with a name Kaggle does not recognise, the same way as the BraTS data. The staging folder sits outside the upload folder, because `kaggle datasets create` uploads what it finds there.

```bash
STAGE=../kaggle-confirmation-stage
mkdir -p ../kaggle-confirmation-test $STAGE/evals $STAGE/compare-val
for RUN in $(cat results/0007-confirmation/runs.txt); do
  R=$(basename $RUN)
  mkdir -p $STAGE/runs/$R/checkpoints
  cp $RUN/config.toml $RUN/run.json $RUN/metrics_val.csv $STAGE/runs/$R/
  cp $RUN/checkpoints/best.pt $STAGE/runs/$R/checkpoints/
  cp -r results/0007-confirmation/evals/$R-val $STAGE/evals/
done
cp results/0007-confirmation/compare-val/run.json results/0007-confirmation/compare-val/preflight.json $STAGE/compare-val/
ls $STAGE/runs | wc -l    # must be 12
ls $STAGE/evals | wc -l   # must be 12
tar -cf ../kaggle-confirmation-test/confirmation-test.tar.bin -C $STAGE runs evals compare-val
stat -c %s ../kaggle-confirmation-test/confirmation-test.tar.bin  # write this number in data/DATA.md
kaggle datasets init -p ../kaggle-confirmation-test               # then set title and id (muhammadnomanned/brats-confirmation-test)
kaggle datasets create -p ../kaggle-confirmation-test             # private unless you pass --public; never do that
```

Then check the dataset says private, and write that in `data/DATA.md`. That edit is a docs change after the pin, so we recommend making it after the opening, or the `COMMIT` of step 6 would no longer be the tip.

## Step 6: the one test opening (3 notebook versions)

This is the one opening of the test fold. We recommend reading the whole step first, and checking that steps 1 to 5 all passed. **Batch 1 reads test cases, and that cannot be taken back.** Cell 3 caches the test volumes, and cell 4 then runs the gate before the sweep reads a case. If the gate refuses, no prediction or score has been made. We recommend bringing the message to the main chat, and changing nothing by hand.

The 12 sweeps take about 25 h, so they cannot fit one session. Each notebook version runs `SIZE` models: batch 1 runs the first 4, batch 2 the next 4, batch 3 the last 4. Set `SIZE` from the session limit in ADR 0002, because a version that hits the limit may lose its **Output**.

**Add Input**: **brats-gli-train** and **brats-confirmation-test**. For batch 2 and later, also the **Output** of the version before it, and only that one.

Cell 1 is the cell of the notebook settings, with the `COMMIT` of step 5.

**Cell 2: the study files**

`/kaggle/input` is read only, and the sweep writes beside the val eval, so the files go into `/kaggle/working/confirmation/`. The gate reads the val compare record and the preflight record from the repo folder, so both are copied there. `results/` is git ignored, so the tree stays clean.

```python
BATCH, SIZE = 1, 4
WORK = "/kaggle/working/confirmation"
OLD = glob.glob("/kaggle/input/**/confirmation/runs", recursive=True)
if not OLD:
    TEST_TAR = glob.glob("/kaggle/input/**/confirmation-test.tar.bin", recursive=True)[0]
    sh(f"mkdir -p {WORK}")
    sh(f"tar -xf {TEST_TAR} -C {WORK}")
else:
    shutil.copytree(os.path.dirname(OLD[0]), WORK)
sh("mkdir -p results/0007-confirmation/compare-val")
sh(f"cp {WORK}/compare-val/run.json {WORK}/compare-val/preflight.json results/0007-confirmation/compare-val/")
IDS = list(json.load(open(f"{WORK}/compare-val/run.json"))["run_ids"].values())
print(len(IDS), len(os.listdir(f"{WORK}/runs")))
```

Both numbers must be **12**.

**Cell 3: the test cache and its check (about 7.5 GB)**

```python
sh("uv run python experiments/0003-baseline/preprocess.py --folds test")
sh("uv run python experiments/0003-baseline/preprocess.py --verify")
```

The last line must read `verify ok: 255 files, 0 problem(s)`. The cache holds all 255 test cases, and the sweep scores 228 of them, one per subject.

**Cell 4: this batch of sweeps and their checks**

```python
for R in IDS[(BATCH - 1) * SIZE : BATCH * SIZE]:
    EVAL = f"{WORK}/evals/{R}-test"
    if not finished(EVAL):
        sh(f"uv run python experiments/0004-missing-eval/evaluate.py --run {WORK}/runs/{R} --fold test --open-test --out {WORK}/evals")
    sh(f"uv run python experiments/0004-missing-eval/check_eval.py {EVAL}")
```

Keep the **same `COMMIT`** in every batch. Run the next batch until `IDS` is used up.

### After a session dies

A resume across sessions is normal (spec 0004, AC-5). Run the same batch again with the dead version's **Output** as an input. Cell 2 sees that **Output** and restores it, so the partial `metrics.csv` comes back. The same `evaluate.py` command then cuts `metrics.csv` back to its last whole pair and skips the finished pairs.

A resume stops with exit 2 if the commit, the config, `protocol_hash`, the source run, the `device` or the `gpu` differs. If that happens because the session got another accelerator, set it back to the T4 and run again. Never delete a folder, and never start a test sweep over without a logged reason.

## Step 7: after the opening (laptop, no GPU)

From the last version's **Output**, copy the 12 `confirmation/evals/<run id>-test/` folders to `results/0007-confirmation/evals/`, keeping every file.

Then copy `confirmation/compare-val/preflight.json` from the same **Output** over `results/0007-confirmation/compare-val/preflight.json`, with no comparison. That copy is the record the Kaggle gate read, so it wins. The laptop cannot see the test folders while they sit on Kaggle, so a preflight run there during step 6 could have changed its copy (AC-34).

Then the RQ2 comparison:

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

All 3 seeds go in the thesis. An H1 claim is made only on a line that says `agree`. On a line that says `split`, the thesis reports the split and makes no claim. `rq1_seeds.py` reads `runs.toml`, so an eval of another arm is refused.

The same word on the RQ2 lines means the same thing (AC-32). `split` is never written as no effect, no cost or a weak effect.

Then record the opening in `context/progress-tracker.md`: the date, the commit and the 12 eval ids (AC-22).

**No threshold, pattern, region, metric, margin or test changes after this. A number that disappoints stays in the thesis.**

## The amendment path (AC-31)

Before the test opening, a change after the tag needs all of this, in order:

1. A dated reason in spec 0006.
2. `PREREG_TAG` in `common/evaluation.py` set to the next tag, `prereg-v2`, then `v3`. This changes `protocol_hash()`, so repin `EXPECTED_PROTOCOL_HASH` and `SEALED_HASH` in the same commit.
3. The new annotated tag on that commit, pushed, with a GitHub Release on it.
4. All 12 val evals scored again from that commit, and the val compare run again, because every eval must descend from the current tag.

After the opening, no amendment exists. Nothing in this path may depend on a test number.
