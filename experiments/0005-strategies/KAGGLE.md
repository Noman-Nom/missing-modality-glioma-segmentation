```bash
R1="20261103-023817-f9b3afb6"
R3="fill in after step 1"
R4="fill in after step 2"
```

These three lines hold the run ids of E1, E3 and E4. `train.py` prints each id in its `run dir <path>` line. The id holds a time stamp, so nobody can know R3 and R4 in advance. Fill them in here as the runs finish. The lines work as they are in a shell and in a notebook cell.

# Running E3 and E4 on Kaggle, and the joint test opening

This file takes E3 and E4 from training through the one opening of the test fold (ADR 0002, spec 0005). All training and scoring runs on the Kaggle Tesla T4. The comparisons run on the laptop and use no GPU.

The steps depend on each other, so we recommend doing them in order:

| Step | Where | What | Planned GPU time |
|---|---|---|---|
| 1 | Kaggle | Train and score E3 | about 9.3 h |
| 2 | Kaggle | Train and score E4 | about 9.3 h |
| 3 | Laptop | Copy back and check again | none |
| 4 | Laptop | Val comparison | none |
| 5 | Laptop | Pin the commit | none |
| 6 | Laptop | Upload the joint tar | none |
| 7 | Kaggle | The test opening, E1, E3 and E4 | about 6.3 h |
| 8 | Laptop | Tables, verdict, the record | none |

In all, about 25 h. These times come from E1 and E2, which is the only measuring done so far: about 8.2 h for a whole training notebook, 1.1 h for a val sweep, and 2.1 h for a test sweep. E3 and E4 are not measured yet. E4 has a few more weights, so its time may differ a little. The real timings land in each `run.json`.

**Steps 1 to 6 never read a test case. Step 7 does, and it cannot be taken back.**

## Before you start

- Check your GPU quota on the Kaggle account page, and how long one session may run. Step 1 needs one unbroken session of about 9.3 h, or the resume steps below. Spec 0005 asks for these two numbers to go into ADR 0002 before the E3 run.
- You need the notebook settings from `experiments/0003-baseline/KAGGLE.md`: **Accelerator** set to **GPU T4 x2**, not P100, and **Internet** on. Do not use the laptop for any step marked Kaggle, because the checks refuse a different GPU.
- Click **Add Input**, then **Your Datasets**, and add **brats-gli-train** and **brats-e1-run**.
- Get the commit you want to run onto GitHub. A merge to `main` first is the clean choice. We recommend `/check verify` and `/check review` first, because a bug found after training costs about 19 h of GPU time to redo.

**One training per strategy (AC-14).** E3 and E4 are each trained once, at seed 42. You may repeat a run only if it failed, or if `check_strategy.py` or `check_baseline.py` refused it, and the reason goes in `context/progress-tracker.md`. Nothing about either strategy changes after you have seen its val numbers. If `check_baseline.py` fails on its best score at complete input, stop and debug. Do not lower the 0.60.

## Steps 1 and 2: train and score one strategy (one Save & Run All each)

Make one notebook with `NAME = "e3"`. When that has passed, make a second one with `NAME = "e4"` and the **same `COMMIT`**. Both strategies must run from one commit (AC-13).

Put each block in its own cell, in this order. Then click **Save Version**, choose **Save & Run All (Commit)**, and close the browser if you like.

**Cell 1: code at one exact commit, and the environment**

`sh()` runs a command and stops the notebook if it fails. A plain `!` line does not stop anything, so a later cell would run on a failed step.

```python
NAME = "e3"  # "e4" in the second notebook
COMMIT = "paste the full 40 character commit here"
%env UV_PROJECT_ENVIRONMENT=/tmp/venv
%env UV_LINK_MODE=copy
%env BRATS_TRAIN_DIR=/tmp/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
import glob, json, os, subprocess

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
%cd /tmp/repo
!git log --oneline -1
!pip install -q uv
!uv sync --frozen
!nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

**Cell 2: unpack the data and the E1 run**

```python
R1 = "20261103-023817-f9b3afb6"
E1 = f"/tmp/e1-run/{R1}"
OUT = "/kaggle/working/results/0005-strategies"
RUNS, EVALS = f"{OUT}/runs", f"{OUT}/evals"
DATA_TAR = glob.glob("/kaggle/input/**/brats-gli-train.tar.bin", recursive=True)[0]
RUN_TAR = glob.glob("/kaggle/input/**/e1-run.tar.bin", recursive=True)[0]
!mkdir -p /tmp/brats /tmp/e1-run
!tar -xf {DATA_TAR} -C /tmp/brats
!tar -xf {RUN_TAR} -C /tmp/e1-run
!ls /tmp/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData | wc -l
```

The count must be **1251**.

**Cell 3: the cache and its check**

```python
sh("uv run python experiments/0003-baseline/preprocess.py")
sh("uv run python experiments/0003-baseline/preprocess.py --verify")
```

The last line must read `verify ok: 996 files, 0 problem(s)`, the same 996 files E1 used. The cache takes about 35 GB of `/tmp`. `memory_test.py` is not run here. Its batches skip the strategy hooks, so it would not measure this run, and `check_baseline.py` reads the real peak from `run.json`.

**Cell 4: training**

```python
sh(f"uv run python experiments/0003-baseline/train.py --config experiments/0005-strategies/{NAME}.toml --out {RUNS}")
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
sh(f"uv run python experiments/0005-strategies/check_strategy.py {RUN} --e1 {E1}")
sh(f"uv run python experiments/0004-missing-eval/check_eval.py {EVAL}")
```

The first command is the E1 script. It gives the run its `metrics_val.csv`, which `check_eval.py` compares with pattern 1 of the sweep. `check_strategy.py` prints how the draws fell on the 15 patterns, and how many times the run resumed. Each of the three checks must end in `PASS`.

**When the version finishes:** open it, go to the **Logs**, and paste back into the main chat the last 2 lines of each command, plus the `nvidia-smi` line and the `git log` line. Then download the **Output**, and fill in `R3` or `R4` at the top of this file.

If a check fails, stop there, and bring the failure to the main chat. Do not start the other strategy until you know why.

### After a session dies

Every finished training step and swept pair is on disk, so nothing finished is lost if the output survives. If a dead version shows no **Output**, the run starts again from cell 1, with its reason in the tracker.

1. Make a new notebook (or a new version) with the same settings and inputs. Also **Add Input**: the dead version's **Output**.
2. Keep the **same `NAME` and `COMMIT`**. Run cells 1 to 3 again. The cache lives in `/tmp` and is gone, and rebuilding it gives the same hash.
3. **Do not run cell 4**, because it would start a second run. Put this cell in its place:

```python
import shutil
OLD = sorted(glob.glob("/kaggle/input/**/results/0005-strategies/", recursive=True))[-1]
shutil.copytree(OLD, OUT)
RUN = sorted(glob.glob(f"{RUNS}/2*/"))[-1].rstrip("/")
if os.path.exists(f"{RUN}/train.pid"):
    os.remove(f"{RUN}/train.pid")  # a process number from the old machine can match a live one here
if not finished(RUN):
    sh(f"uv run python experiments/0003-baseline/train.py --resume {RUN}")
```

4. Run cell 5 as it is. It repeats the E1 script, which only writes `metrics_val.csv` again. It then resumes a sweep that stopped partway, or skips one that finished.

A resume is allowed. It reseeds the loader, so the run's later patches no longer match the other models'. The pattern draws still match, because `hook_rng` is restored from `last.pt`. `check_strategy.py` prints the number of resumes, and `compare.py` records it, so a reader can see it. `train.py --resume` and the sweep each stop with exit 2 if something they pin has changed. Spec 0003 AC-11 and spec 0004 AC-5 list what they pin.

Known gap (spec 0003, flagged for `/architect`): if a crash lands between the `best.pt` and `last.pt` writes, the resume deletes that `best.pt`, and the older best weights are gone.

## Step 3: copy back and check again (laptop)

From each notebook's **Output**, copy `results/0005-strategies/runs/<run id>/` into `results/0005-strategies/runs/`, and `results/0005-strategies/evals/<run id>-val/` into `results/0005-strategies/evals/`. Keep every file, `run.json` included.

Then, with `R1`, `R3` and `R4` set in your shell:

```bash
for R in $R3 $R4; do
  uv run python experiments/0005-strategies/check_strategy.py results/0005-strategies/runs/$R --e1 results/0003-baseline/$R1
  uv run python experiments/0004-missing-eval/check_eval.py results/0005-strategies/evals/$R-val --run results/0005-strategies/runs/$R
done
```

`--run` is needed here because the folder moved between machines. `check_baseline.py` is not repeated, because it reads the cache, which stayed on Kaggle.

## Step 4: the val comparison (laptop, no GPU)

```bash
uv run python experiments/0005-strategies/compare.py --e1-run results/0003-baseline/$R1 --e3-run results/0005-strategies/runs/$R3 --e4-run results/0005-strategies/runs/$R4 --fold val
uv run python experiments/0005-strategies/make_table.py --compare results/0005-strategies/compare-val
cd writing/tracer && latexmk -xelatex e3e4-preview.tex
```

These are development numbers. The captions say so, and the choice of method does not depend on them, because the comparison is fixed by the spec. `results/0005-strategies/compare-val/` must stay on the laptop: `compare.py --fold test` looks for it in step 8.

## Step 5: pin the commit (laptop)

**Stop here for a moment.** Anything you want to add to the study has to be decided before step 7, and that includes more seeds, another dataset and another method.

When you are ready:

1. Record the val run in `context/progress-tracker.md`: the run ids, the check results and the real timings.
2. Commit the tree, with `R3` and `R4` filled in at the top of this file. `results/` is git ignored, so this commit holds the code and the records, and the compare folder stays in `results/` on the laptop.
3. Merge to `main`, push, and run `git rev-parse HEAD`. That full commit is the `COMMIT` of step 7.
4. Check `git status --short` shows nothing tracked as changed. The test gate refuses a dirty tree.

After this point, do not change a file named in `PROTOCOL_FILES` or `PROTOCOL_SOURCES`, or any of the four files in `compare_hash`. The test gate and `compare.py --fold test` would refuse, and the way out is a logged rerun of val.

## Step 6: upload the joint tar (laptop, once)

The test notebook needs three run folders and three val evals, because the gate reads `<out>/<run id>-val`. They go up as one uncompressed tar with a name Kaggle does not recognise, the same way as the BraTS data. The staging folder sits outside the upload folder, because `kaggle datasets create` uploads what it finds there.

```bash
cd /media/noman/E0948A169489EF7A/qarar-research
STAGE=../kaggle-joint-stage
mkdir -p ../kaggle-joint-test $STAGE/evals-e1 $STAGE/evals
for RUN in results/0003-baseline/$R1 results/0005-strategies/runs/$R3 results/0005-strategies/runs/$R4; do
  KEEP=$STAGE/runs/$(basename $RUN)
  mkdir -p $KEEP/checkpoints
  cp $RUN/config.toml $RUN/run.json $RUN/metrics_val.csv $KEEP/
  cp $RUN/checkpoints/best.pt $KEEP/checkpoints/
done
cp -r results/0004-missing-eval/$R1-val $STAGE/evals-e1/
cp -r results/0005-strategies/evals/$R3-val results/0005-strategies/evals/$R4-val $STAGE/evals/
tar -cf ../kaggle-joint-test/joint-test.tar.bin -C $STAGE runs evals-e1 evals
stat -c %s ../kaggle-joint-test/joint-test.tar.bin   # write this number in data/DATA.md
kaggle datasets init -p ../kaggle-joint-test         # then set title and id (muhammadnomanned/brats-joint-test)
kaggle datasets create -p ../kaggle-joint-test       # private unless you pass --public; never do that
```

## Step 7: the test opening (one Save & Run All)

This is the one opening of the test fold. Before you click, we recommend reading the whole step, and checking that steps 1 to 6 all passed. If the gate refuses, nothing has been read. Bring the message to the main chat, and change nothing by hand.

Settings as before: **GPU T4 x2**, **Internet** on. **Add Input**: **brats-gli-train** and **brats-joint-test**.

**Cell 1: code at the pinned commit**

Use cell 1 from step 1, with the `COMMIT` of step 5, and leave out the `NAME` line. Then add the three lines from the top of this file, in a cell of their own.

**Cell 2: unpack the data and the joint tar**

`/kaggle/input` is read only, and the sweep writes beside the val eval, so the tar goes into `/kaggle/working/joint/`:

```python
JOINT = "/kaggle/working/joint"
DATA_TAR = glob.glob("/kaggle/input/**/brats-gli-train.tar.bin", recursive=True)[0]
JOINT_TAR = glob.glob("/kaggle/input/**/joint-test.tar.bin", recursive=True)[0]
!mkdir -p /tmp/brats {JOINT}
!tar -xf {DATA_TAR} -C /tmp/brats
!tar -xf {JOINT_TAR} -C {JOINT}
!ls /tmp/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData | wc -l
!ls {JOINT}/runs {JOINT}/evals-e1 {JOINT}/evals
```

The count must be **1251**, and you should see the three run ids and three val evals.

**Cell 3: the test cache and its check (about 7.5 GB)**

```python
sh("uv run python experiments/0003-baseline/preprocess.py --folds test")
sh("uv run python experiments/0003-baseline/preprocess.py --verify")
```

The last line must read `verify ok: 255 files, 0 problem(s)`. The cache holds all 255 test cases, and the sweep scores 228 of them, one per subject. The settings are E1's, so the preprocess hash stays the same.

**Cell 4: the three sweeps, E1 first**

```python
OUTS = {R1: f"{JOINT}/evals-e1", R3: f"{JOINT}/evals", R4: f"{JOINT}/evals"}
for R, OUT in OUTS.items():
    if not finished(f"{OUT}/{R}-test"):
        sh(f"uv run python experiments/0004-missing-eval/evaluate.py --run {JOINT}/runs/{R} --fold test --open-test --out {OUT}")
```

**Cell 5: the checks**

```python
for R, OUT in OUTS.items():
    sh(f"uv run python experiments/0004-missing-eval/check_eval.py {OUT}/{R}-test")
```

**When the version finishes:** paste back into the main chat the last 2 lines of each command, plus the `nvidia-smi` line and the `git log` line. Then download the **Output**.

### After a session dies

A resume across sessions is normal (spec 0004, AC-5). The same `evaluate.py` command resumes: it cuts `metrics.csv` back to its last whole pair, skips the finished pairs, and adds an entry to `resumes` in `run.json`.

1. Make a new notebook (or a new version) with the same settings and inputs. Also **Add Input**: the dead version's **Output**.
2. Keep the **same `COMMIT`**. Run cell 1.
3. In cell 2, replace the `!tar -xf {JOINT_TAR} -C {JOINT}` line and the `mkdir` of `{JOINT}` with this, so the partial `metrics.csv` comes back:

```python
import shutil
OLD = sorted(glob.glob("/kaggle/input/**/joint/", recursive=True))[-1]
shutil.copytree(OLD, JOINT)
```

4. Run cells 3 to 5 as they are. Cell 4 skips a sweep that finished.

A resume stops with exit 2 if the commit, the config, `protocol_hash`, the source run, the `device` or the `gpu` differs. If that happens because the session got another accelerator, set it back to the T4 and run again. Never delete a folder, and never start a test sweep over without a logged reason.

## Step 8: after the opening (laptop, no GPU)

Copy E1's test eval to `results/0004-missing-eval/`, and E3's and E4's to `results/0005-strategies/evals/`, keeping every file. The laptop must be at a commit with the same `data/splits/splits.csv`.

The RQ1 numbers, from the E1 test eval only:

```bash
uv run python experiments/0004-missing-eval/check_eval.py results/0004-missing-eval/$R1-test --run results/0003-baseline/$R1
uv run python experiments/0004-missing-eval/analyse.py --eval results/0004-missing-eval/$R1-test
uv run python experiments/0004-missing-eval/make_table.py --eval results/0004-missing-eval/$R1-test
```

The RQ2 comparison:

```bash
for R in $R3 $R4; do
  uv run python experiments/0004-missing-eval/check_eval.py results/0005-strategies/evals/$R-test --run results/0005-strategies/runs/$R
done
uv run python experiments/0005-strategies/compare.py --e1-run results/0003-baseline/$R1 --e3-run results/0005-strategies/runs/$R3 --e4-run results/0005-strategies/runs/$R4 --fold test
uv run python experiments/0005-strategies/make_table.py --compare results/0005-strategies/compare-test
```

`compare.py` refuses if `compare-val` is missing, names other runs, or has a different `compare_hash`. If `versions` differ, it warns and writes anyway, with `versions_equal` false, because a test opening cannot be taken back.

Then record the opening by hand in `context/progress-tracker.md`: the date, the commit and the three eval ids.

**No threshold, pattern, region, metric or test changes after this. A number that disappoints stays in the thesis.**
