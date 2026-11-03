# Running E1 on Kaggle

E1 trains on a free Kaggle Tesla T4 (ADR 0002). The GPU check is step 1, the data is in place (step 2), and step 3 is the full run as one **Save & Run All**.

## Before you start

- Verify your Kaggle account with a phone number. Without that, the GPU options stay hidden.
- The notebook clones `main` from GitHub, so everything the run needs must be merged there.
- In a new notebook, open **Settings** and set **Accelerator** to **GPU T4 x2**. Do not pick P100: the locked PyTorch is built for CUDA 13, which has no Pascal kernels. Also turn **Internet** on, which clone and `uv sync` need.
- Click **Add Input**, then **Your Datasets**, and add **brats-gli-train**.

## Step 1: GPU check (done 2026-10-31, repeat only if Kaggle changes the GPU or driver)

```python
%env UV_PROJECT_ENVIRONMENT=/tmp/venv
%env UV_LINK_MODE=copy
!git clone --depth 1 --branch main https://github.com/Noman-Nom/missing-modality-glioma-segmentation.git /tmp/repo
%cd /tmp/repo
!git log --oneline -1
```

```python
!pip install -q uv
!uv sync --frozen
```

```python
!uv run python experiments/0003-baseline/gpu_bench.py
```

The code and the environment live in `/tmp`, so they do not fill `/kaggle/working`, which is the folder Kaggle saves as output. `uv sync --frozen` installs exactly what `uv.lock` pins.

Result on 2026-10-31: Tesla T4 (sm_75), 2 visible, driver 580.178.04, torch 2.14.1+cu130, 0.201 s/iter (2.2 h for 40,000 steps, fp16, before E1 moved to float32), peak reserved 898 MiB. Laptop: 0.764 s/iter (8.5 h). A `FAIL: ...` line means CUDA 13 does not work there: stop and decide in the main chat.

## Step 2: data (done 2026-11-01)

The private dataset `muhammadnomanned/brats-gli-train` holds one file, `brats-gli-train.tar.bin` (13,394,974,720 bytes): the training folder packed as a plain tar. Kaggle unpacks `.zip` and `.nii.gz` files on upload and renames some of them, which our loader cannot read, so the folder is uploaded under a name Kaggle leaves alone. The two stray `.nii` folders (cases 00011 and 00014) are left out. See `data/DATA.md`.

To rebuild it (only if the dataset is ever lost), on the laptop:

```bash
cd /media/noman/E0948A169489EF7A/qarar-research
tar -cf ../kaggle-brats-tar/brats-gli-train.tar.bin \
  --exclude='*/BraTS-GLI-00011-000-t1n.nii' --exclude='*/BraTS-GLI-00014-000-t2w.nii' \
  ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
kaggle datasets create -p ../kaggle-brats-tar    # dataset-metadata.json sits next to the tar
```

## Step 3: the E1 run (one Save & Run All)

Put each block in its own cell, in this order. Then click **Save Version**, choose **Save & Run All (Commit)**, and close the browser if you like: the run keeps going.

**Cell 1: code and environment**

```python
%env UV_PROJECT_ENVIRONMENT=/tmp/venv
%env UV_LINK_MODE=copy
%env BRATS_TRAIN_DIR=/tmp/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
!git clone --depth 1 --branch main https://github.com/Noman-Nom/missing-modality-glioma-segmentation.git /tmp/repo
%cd /tmp/repo
!git log --oneline -1
!pip install -q uv
!uv sync --frozen
!nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

**Cell 2: unpack the data (1 to 2 minutes)**

```python
import glob
TAR = glob.glob("/kaggle/input/**/brats-gli-train.tar.bin", recursive=True)[0]
print(TAR)
!mkdir -p /tmp/brats && tar -xf {TAR} -C /tmp/brats
!ls /tmp/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData | wc -l
!df -h /tmp /kaggle/working
```

The count must be **1251**. The search finds the file wherever Kaggle mounts the dataset.

**Cell 3: cache and its checks**

```python
!uv run python experiments/0003-baseline/preprocess.py
!uv run python experiments/0003-baseline/preprocess.py --verify
!uv run python experiments/0003-baseline/memory_test.py --out /kaggle/working/results/0003-baseline
```

The cache (about 35 GB) goes to `/tmp/repo/data/processed/0003-baseline/`. `preprocess.py` stops with exit 2 if the disk is too small; then stop and decide in the main chat.

**Cell 4: training, only after a memory test PASS (time measured by the memory test, at most 9 h)**

```python
import json, subprocess
MEM = "/kaggle/working/results/0003-baseline/memory_test.json"
HASH = ("import sys; sys.path.insert(0, 'experiments'); "
        "from common.runlog import config_hash, load_config; "
        "print(config_hash(load_config('experiments/0003-baseline/config.toml')))")
now = subprocess.run(["uv", "run", "python", "-c", HASH], capture_output=True, text=True,
                     check=True).stdout.strip()
try:
    mem = json.load(open(MEM))
except FileNotFoundError:
    mem = None
if mem is None or mem["pass"] is not True or mem["config_hash"] != now:
    print("NOT TRAINING: no memory test PASS for this config:",
          mem and {k: mem[k] for k in ("pass", "projected_hours", "config_hash")})
    raise SystemExit(1)  # stops Save & Run All here, so cell 5 never runs
!uv run python experiments/0003-baseline/train.py --out /kaggle/working/results/0003-baseline
```

Since spec 0003 step 9, E1 trains in float32 (`amp = false`), and `memory_test.py` fails when the projection is above 9 hours. The gate also checks the config hash, so a `memory_test.json` from another config never counts. A full run's config equals `config.toml` as committed (no `--quick`, no `--seed`), so its hash is the one computed here.

**Cell 5: evaluation, table and check**

```python
RUN = sorted(glob.glob("/kaggle/working/results/0003-baseline/2*/"))[-1]
print(RUN)
!uv run python experiments/0003-baseline/evaluate.py --run {RUN}
!uv run python experiments/0003-baseline/make_table.py --run {RUN} --out /kaggle/working/0003-baseline-val.tex
!uv run python experiments/0003-baseline/check_baseline.py {RUN}
```

**When the version finishes:** open it, go to the **Logs**, and paste back into the main chat the last 2 lines of each command: `preprocess`, `preprocess --verify`, `memory_test`, `train`, `evaluate`, `make_table`, `check_baseline`, plus the `nvidia-smi` line. Then download the **Output** (`results/0003-baseline/<run_id>/` and the table) and copy the run folder into `results/0003-baseline/` in the repo. Rebuild the table there with `make_table.py --run results/0003-baseline/<run_id>`, so the thesis table comes from the repo copy.

## After a session dies

1. Make a new notebook (or a new version) with the same settings and inputs. Also **Add Input**: the dead version's **Output**.
2. Run cells 1 to 3 again (the cache lives in `/tmp` and is gone; rebuilding it gives the same hash).
3. Copy the run folder back, delete its `train.pid`, and resume:

```python
import glob, shutil, os
OLD = sorted(glob.glob("/kaggle/input/**/results/0003-baseline/2*/", recursive=True))[-1]
RUN = "/kaggle/working/results/0003-baseline/" + os.path.basename(OLD.rstrip("/"))
shutil.copytree(OLD, RUN)
if os.path.exists(RUN + "/train.pid"):
    os.remove(RUN + "/train.pid")  # a process number from the old machine can match a live one here
!uv run python experiments/0003-baseline/train.py --resume {RUN}
```

Then run cell 5. Resume refuses (exit 2) if the config or the cache hash changed, if the run already finished, or if there is no `last.pt`.

Known gap (spec 0003, flagged for `/architect`): if a crash lands between the `best.pt` and `last.pt` writes, the resume deletes that `best.pt`, and the older best weights are gone.
