```bash
R1="20261103-023817-f9b3afb6"
R3="20261106-024450-10cfa2cc"
R4="fill in after step 2"
```

These three lines hold the run ids of E1, E3 and E4. `train.py` prints each id in its `run dir <path>` line. The id holds a time stamp, so nobody can know R3 and R4 in advance. Fill them in here as the runs finish. The lines work as they are in a shell and in a notebook cell.

# Running E3 and E4 on Kaggle

This file takes E3 and E4 from training through the val comparison (ADR 0002, spec 0005). All training and scoring runs on the Kaggle Tesla T4. The comparisons run on the laptop and use no GPU.

The steps depend on each other, so we recommend doing them in order:

| Step | Where | What | Planned GPU time |
|---|---|---|---|
| 1 | Kaggle | Train and score E3 | about 9.3 h |
| 2 | Kaggle | Train and score E4 | about 9.3 h |
| 3 | Laptop | Copy back and check again | none |
| 4 | Laptop | Val comparison | none |

In all, about 19 h. These times come from E1 and E2, which is the only measuring done so far: about 8.2 h for a whole training notebook, 1.1 h for a val sweep, and 2.1 h for a test sweep. E3 and E4 are not measured yet. E4 has a few more weights, so its time may differ a little. The real timings land in each `run.json`.

**No step here reads a test case.**

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

Spec 0006 changes this step in two ways. Do not run it before `prereg-v1` is tagged, or the tag message must say so (AC-8). And `compare.py` now refuses eval folders scored before the volume columns were added (AC-7).

```bash
uv run python experiments/0005-strategies/compare.py --e1-run results/0003-baseline/$R1 --e3-run results/0005-strategies/runs/$R3 --e4-run results/0005-strategies/runs/$R4 --fold val
uv run python experiments/0005-strategies/make_table.py --compare results/0005-strategies/compare-val
cd writing/tracer && latexmk -xelatex e3e4-preview.tex
```

These are development numbers. The captions say so, and the choice of method does not depend on them, because the comparison is fixed by the spec. `compare.py` stays for val only (spec 0006, AC-21).

## The test path of this file is retired

Spec 0006 (AC-21) replaced steps 5 to 8, which were the joint test opening of E1, E3 and E4. The test fold is opened once, for all 12 models of the confirmation study, and those steps live in `experiments/0007-confirmation/KAGGLE.md`. This file opens no test.
