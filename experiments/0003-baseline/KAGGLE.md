# Running E1 on Kaggle

You can run E1 on a free Kaggle GPU instead of the laptop. Step 1 checks the GPU and its speed with random data, so no BraTS data leaves your machine yet. Steps 2 and 3 are a draft, to be confirmed after step 1.

## Before you start

- Verify your Kaggle account with a phone number. Without that, the GPU options stay hidden.
- Make sure the branch you want is pushed to GitHub. The notebook clones the public repo.
- In a new notebook, open **Settings** and set **Accelerator** to **GPU T4 x2**. Do not pick P100: the locked PyTorch is built for CUDA 13, which has no Pascal kernels. Also turn **Internet** on, which clone and `uv sync` need.

## Step 1: GPU check (about 10 minutes of quota, mostly the install)

Paste each block into its own cell and run them in order.

```python
%env UV_PROJECT_ENVIRONMENT=/tmp/venv
%env UV_LINK_MODE=copy
!git clone --depth 1 --branch feat/e1-baseline https://github.com/Noman-Nom/missing-modality-glioma-segmentation.git /tmp/repo
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

The code and the environment live in `/tmp`, so they do not fill `/kaggle/working`, which is the folder Kaggle saves as output. `uv sync --frozen` installs exactly what `uv.lock` pins. If the notebook's Python is not 3.12, uv downloads 3.12 by itself.

**Paste back the whole output of the last cell** (about 10 lines). What it tells you:

| Line | Look for |
|---|---|
| `torch ... built for CUDA 13.0, driver ...` | the driver Kaggle runs |
| `GPU Tesla T4 (sm_75), 2 visible` | the right card; 2 means one seed per GPU is possible later |
| `FAIL: ...` | CUDA 13 does not work there; stop and decide in the main chat |
| `... s/iter -> ... h` | compare with the laptop: 0.764 s/iter, 8.5 h (training only) |

## Step 2: data (draft)

1. Upload `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` (13 GB, 1,251 case folders) as a **private** Kaggle dataset. Keep it private.
2. In the training notebook, use **Add Input** to attach it. It appears under `/kaggle/input/<dataset name>/`.

## Step 3: training session (draft)

```python
%env BRATS_TRAIN_DIR=/kaggle/input/<dataset name>/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
!uv run python experiments/0003-baseline/preprocess.py
!uv run python experiments/0003-baseline/memory_test.py --out /kaggle/working/results/0003-baseline
!uv run python experiments/0003-baseline/train.py --out /kaggle/working/results/0003-baseline
```

Run it as **Save Version, Save & Run All**, so it keeps going after you close the browser. The cache, about 35 GB as estimated by the disk check, goes to `/tmp/repo/data/processed/`. `preprocess.py` stops with exit 2 if the disk is too small.

**After a session dies:** attach that version's output as an input, copy the run folder back into `/kaggle/working/results/0003-baseline/`, **delete its `train.pid`** (a process number from the old machine can match a live process on the new one, and `--resume` would then wrongly refuse), rebuild the cache, then run:

```python
!uv run python experiments/0003-baseline/train.py --resume /kaggle/working/results/0003-baseline/<run_id>
```
