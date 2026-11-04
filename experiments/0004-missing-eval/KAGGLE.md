# Running E2 on Kaggle

E2 scores the frozen E1 model on all 15 modality patterns on the Kaggle Tesla T4 (ADR 0002, spec 0004). This file covers the val sweep only. The test fold stays sealed until feature 7.

The sweep must run on the T4, because the check compares pattern 1 with E1's val Dice and refuses another GPU name. E1 ran on `Tesla T4`.

Every number below is a real measurement or a count. Timing is the one rough part, see step 2.

## Before you start

- You need the Kaggle notebook settings from `experiments/0003-baseline/KAGGLE.md`: **Accelerator** set to **GPU T4 x2**, not P100, and **Internet** on.
- Click **Add Input**, then **Your Datasets**, and add **brats-gli-train** and **brats-e1-run** (step 1).
- Get the commit you want to run onto GitHub. Everything the sweep needs must be pushed. A merge to `main` first is the clean choice.
- We recommend finishing `/check review` first. `analyse.py` and the shared metric files are in `protocol_hash`, so any later change to them costs a second val run before test can open.

## Step 1: upload the E1 run (once)

The notebook needs four files of the E1 run: `config.toml`, `run.json`, `metrics_val.csv` and `checkpoints/best.pt`. They go up as one uncompressed tar with a name Kaggle does not recognise, the same way as the BraTS data. A tar keeps the `checkpoints/` folder and stops Kaggle from touching the files on upload (see `data/DATA.md`).

On the laptop:

```bash
cd /media/noman/E0948A169489EF7A/qarar-research
mkdir -p ../kaggle-e1-run
ID=20261103-023817-f9b3afb6
tar -cf ../kaggle-e1-run/e1-run.tar.bin -C results/0003-baseline \
  $ID/config.toml $ID/run.json $ID/metrics_val.csv $ID/checkpoints/best.pt
stat -c %s ../kaggle-e1-run/e1-run.tar.bin    # 51599360
kaggle datasets init -p ../kaggle-e1-run      # then set title and id (muhammadnomanned/brats-e1-run)
kaggle datasets create -p ../kaggle-e1-run    # private unless you pass --public; never do that
```

## Step 2: the val run (one Save & Run All)

Put each block in its own cell, in this order. Then click **Save Version**, choose **Save & Run All (Commit)**, and close the browser if you like.

How long it takes is not measured on the T4 yet. A throwaway 2 case run on the laptop GPU took 126 s for 30 pairs (2026-11-04, not a result), which is about 2 h for all 1,710 pairs at laptop speed. The real timings land in `run.json`.

**Cell 1: code at one exact commit, and the environment**

Paste the full commit into `COMMIT`. A pinned commit is what lets a later session resume: `evaluate.py` refuses a resume when `git_commit` differs, and a plain clone of `main` could land on a newer one.

```python
COMMIT = "paste the full 40 character commit here"
%env UV_PROJECT_ENVIRONMENT=/tmp/venv
%env UV_LINK_MODE=copy
%env BRATS_TRAIN_DIR=/tmp/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
import subprocess
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
import glob, hashlib
ID = "20261103-023817-f9b3afb6"
E1 = f"/tmp/e1-run/{ID}"
OUT = "/kaggle/working/results/0004-missing-eval"
EVAL = f"{OUT}/{ID}-val"
BEST = "975f56a148af15c01714f397aa4f248f518adda11d55e0de8f8155517d521a58"
DATA_TAR = glob.glob("/kaggle/input/**/brats-gli-train.tar.bin", recursive=True)[0]
RUN_TAR = glob.glob("/kaggle/input/**/e1-run.tar.bin", recursive=True)[0]
!mkdir -p /tmp/brats /tmp/e1-run
!tar -xf {DATA_TAR} -C /tmp/brats
!tar -xf {RUN_TAR} -C /tmp/e1-run
!ls /tmp/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData | wc -l
assert hashlib.sha256(open(f"{E1}/checkpoints/best.pt", "rb").read()).hexdigest() == BEST
```

The count must be **1251**. The last line stops the notebook if `best.pt` is not the file E1 finished with.

**Cell 3: rebuild the val cache and verify it (val only, about 3.8 GB)**

```python
!uv run python experiments/0003-baseline/preprocess.py --folds val
!uv run python experiments/0003-baseline/preprocess.py --verify
```

The last line must read `verify ok: 129 files, 0 problem(s)`. The cache holds 129 val cases, one for every timepoint of the fold, and the sweep scores 114 of them, one per subject. `--verify` recomputes only train cases, so here it checks file hashes alone. The real check that this cache matches E1's is the tie out in cell 5. The cache lives in `/tmp` and is gone after the session. The E1 settings are unchanged, so a rebuild gives the same preprocess hash.

**Cell 4: the sweep**

```python
!uv run python experiments/0004-missing-eval/evaluate.py --run {E1} --fold val --out {OUT}
```

**Cell 5: analysis, tables and check**

```python
!uv run python experiments/0004-missing-eval/analyse.py --eval {EVAL}
!uv run python experiments/0004-missing-eval/make_table.py --eval {EVAL} --out-dir /kaggle/working/tables
!uv run python experiments/0004-missing-eval/check_eval.py {EVAL}
```

`analyse.py` refuses a sweep that did not finish, so it never gives a verdict on part of the fold. `check_eval.py` prints `PASS` and the projected test time.

**When the version finishes:** open it, go to the **Logs**, and paste back into the main chat the last 2 lines of each command, plus the `nvidia-smi` line and the `git log` line. Then download the **Output**.

Copy the whole eval folder into `results/0004-missing-eval/` in the repo. Any copy must keep `run.json`, because `analyse.py` refuses a folder without it. Then, on the laptop, build the thesis tables from that copy and check it, passing `--run` because the folder moved between machines:

```bash
ID=20261103-023817-f9b3afb6
uv run python experiments/0004-missing-eval/check_eval.py results/0004-missing-eval/$ID-val --run results/0003-baseline/$ID
uv run python experiments/0004-missing-eval/make_table.py --eval results/0004-missing-eval/$ID-val
cd writing/tracer && latexmk -xelatex e2-preview.tex
```

## After a session dies

Every finished pair is flushed and synced to `metrics.csv`, and `run.json` holds the running totals, so nothing finished is lost if the output survives. We have not seen a real E2 session die yet. If a dead version shows no **Output**, the sweep starts again from step 2.

1. Make a new notebook (or a new version) with the same settings and inputs. Also **Add Input**: the dead version's **Output**.
2. Keep the **same `COMMIT`** in cell 1. Run cells 1 to 3 again.
3. Add this cell before cell 4. `/kaggle/input` is read only, so the folder has to be copied out first:

```python
import os, shutil
OLD = sorted(glob.glob(f"/kaggle/input/**/results/0004-missing-eval/{ID}-val/", recursive=True))[-1]
os.makedirs(OUT, exist_ok=True)
shutil.copytree(OLD, EVAL)
```

4. Run cells 4 and 5 as they are. The same `evaluate.py` command resumes: it cuts `metrics.csv` back to its last whole pair, skips the finished pairs, and adds an entry to `resumes` in `run.json`.

A resume stops with exit 2 if the commit, the config, `protocol_hash`, the source run, the `device` or the `gpu` differs, so one `metrics.csv` never mixes two machines. A finished folder is refused too, and needs `--rerun-reason "<text>"` to be replaced.

If a resume stops because the session got another accelerator, you have two ways out. You could set the accelerator back to the T4 and run again, which keeps every finished pair. Or you could move the folder aside by hand and start clean, because `--rerun-reason` works only on a finished folder. The same goes for a failed or running folder from an older commit. Never delete a folder.

## Later: the test run (feature 7, do not run now)

The test fold opens once, for E1, E3 and E4 together, and feature 7 sets those steps. One thing is fixed here. The test gate looks for `<out>/<run id>-val`, and `/kaggle/input` is read only. So copy the finished val eval folder into the `--out` folder first, the same way as the cell above, keeping `run.json`.
