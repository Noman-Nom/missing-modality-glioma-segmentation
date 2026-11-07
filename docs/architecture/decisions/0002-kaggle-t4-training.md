# 0002. Train E1, E3 and E4 on a Kaggle Tesla T4

**Date**: 2026-11-02
**Status**: Accepted (decided by the human 2026-10-31 after the GPU benchmark; AC-3 to AC-5 open until the first E1 run on Kaggle)

## Summary

The long training runs move from the laptop (Quadro T2000 Max-Q) to a free Kaggle notebook with a Tesla T4. The training step is 3.8 times faster there (0.201 against 0.764 s/iter), and the losses match the laptop. E1, E3 and E4 all train on the same Kaggle T4 setup, so their timings and conditions stay comparable. The laptop stays the machine for development, fast tests and quick runs.

## Context

AGENTS.md says everything must fit the laptop unless an ADR says otherwise. Spec 0003 (E1) trains a 3D U-Net at 128^3 for 40,000 iterations. E3 and E4 are the same size or larger. On the laptop that is 8.5 h of training steps alone per run, before validation and data loading, and the laptop is also the only machine for everything else.

`experiments/0003-baseline/gpu_bench.py` runs the real `Trainer.step()` on random batches of the spec shape (4 x 128^3), so only the data is fake. Results, 2026-10-31:

| | Laptop | Kaggle |
|---|---|---|
| GPU | Quadro T2000 Max-Q (sm_75) | Tesla T4 (sm_75), 2 visible |
| Driver, torch | 580.178.04, 2.14.1+cu130 | 580.178.04, 2.14.1+cu130 |
| s/iter (training step only) | 0.764 | 0.201 |
| 40,000 iterations | 8.5 h | 2.2 h |
| Peak reserved VRAM | 898 MiB | 898 MiB |
| Loss at steps 10 to 30 | reference | same to 4 decimals |

These are benchmark numbers, not results.

## Requirements

**Acceptance criteria**:
- **AC-1**: The environment on Kaggle is the one `uv.lock` pins (`uv sync --frozen`), the same as the laptop. Done 2026-10-31 (benchmark).
- **AC-2**: The training step on the T4 gives the same losses as the laptop on the same seed. Done 2026-10-31 (steps 10 to 30, 4 decimals).
- **AC-3**: BraTS data reaches Kaggle unchanged: the files the notebook reads are byte-identical `.nii.gz` files, in a private dataset.
- **AC-4**: A full E1 run on Kaggle passes `preprocess.py --verify`, `memory_test.py` and `check_baseline.py`.
- **AC-5**: Every run records the hardware it ran on, so a reader can see that E1, E3 and E4 shared it. `run.json` already holds `gpu` (device name) and `versions` (torch, CUDA, cuDNN); the driver version is not recorded yet.

## Options considered

### Option 1: keep training on the laptop
**Pros**: no new service; the data never leaves the machine.
**Cons**: 8.5 h of steps per run, plus validation; one crash or sleep loses a session; the laptop is blocked for days across E1, E3, E4 and their seeds.

### Option 2: Kaggle notebook, Tesla T4 (chosen)
**Pros**: 3.8 times faster; free; same driver, CUDA build and lockfile; 2 GPUs visible, so a second seed could run on the other GPU later.
**Cons**: a GPU quota per week and a time limit per session (current numbers on the account's quota page, to record here); the data must be uploaded; a session can die and needs the resume steps; it is an outside service holding medical data.

### Option 3: Kaggle P100
**Cons**: the locked torch is built for CUDA 13, which has no Pascal (sm_60) kernels. Not usable without changing ADR 0001.

### Option 4: paid cloud GPU
**Cons**: paid services are out unless an ADR justifies the cost; Option 2 is fast enough.

## Decision

**Chosen option**: Option 2, Kaggle notebook with a Tesla T4 (accelerator "GPU T4 x2").

Rules:
- E1, E3 and E4 train on the Kaggle T4. Do not mix laptop and Kaggle runs inside one comparison.
- The notebook clones `main` and runs `uv sync --frozen`. The steps live in `experiments/0003-baseline/KAGGLE.md`.
- Data: a **private** Kaggle dataset, `muhammadnomanned/brats-gli-train`, holding one file, `brats-gli-train.tar.bin` (an uncompressed tar, 13,394,974,720 bytes). Kaggle unpacks `.zip` and `.nii.gz` files on upload and renames some of them, so the folder is packed as a tar with a name Kaggle does not recognise, and the notebook unpacks it. Details in `data/DATA.md`.
- Results go to `/kaggle/working/` (saved as the notebook's output) and are copied back into `results/` in the repo checkout by the human.
- The laptop stays the machine for development, `uv run pytest -q`, slow tests and `--quick` runs.
- E2 also runs on the T4 (added 2026-11-04, spec 0004). Its check compares pattern 1 with E1's val Dice, so it has to use the GPU E1 used. The steps live in `experiments/0004-missing-eval/KAGGLE.md`. The notebook pins one exact commit instead of cloning the tip of `main`, because a resume refuses a different `git_commit`.
- The E1 model weights sit in a **private** Kaggle dataset, `muhammadnomanned/brats-e1-run`, holding one file, `e1-run.tar.bin`. It holds `best.pt` and the three run files the sweep reads. Details in `data/DATA.md`.
- E3 and E4 train and score on the T4 (added 2026-11-06, spec 0005), through the E1 and E2 scripts, from one pinned commit. The steps live in `experiments/0005-strategies/KAGGLE.md`.
- The confirmation study trains and scores on the T4 too (added 2026-11-07, spec 0006): 9 new models, 3 seed 42 val rescores and 12 test sweeps, from pinned commits. The steps live in `experiments/0007-confirmation/KAGGLE.md`. Planned quota: about 114 h, about 5 weeks. The quota page showed GPU 30 h available of 30 h on 2026-11-07; the page text does not state the period, and the human reported it as weekly on 2026-11-06. Used so far: 0 h of the 114 h. Session limit: 12 hours for CPU and GPU notebook sessions (9 hours for TPU), read from Kaggle's notebook docs, Technical Specifications, on 2026-11-07, by the human. The docs say "at time of writing". The 12 test sweeps (about 25 h) must span sessions, so `SIZE = 4` in step 6: 4 sweeps at about 2.1 h is about 8.4 h, which leaves about 3.6 h for setup and the test cache. These times are projections. If a batch runs close to the limit, a smaller `SIZE` is the safe change. This replaces the joint test opening of spec 0005.
- The test inputs sit in a **private** Kaggle dataset, `muhammadnomanned/brats-confirmation-test`, holding one file, `confirmation-test.tar.bin`. It holds the 12 run folders, the 12 val evals and `compare-val/run.json`. It does not exist yet: the human uploads it after the val compare. Details in `data/DATA.md`.

## Rationale

The benchmark shows the same environment, the same losses and the same memory, with steps 3.8 times faster. The memory use (898 MiB) is far below the T4's 16 GB, so the spec settings stay as written. Keeping all three experiments on one hardware setup is what the fair comparison rule needs: same subjects, splits, metrics, seeds, and now the same machine.

## Consequences

**Positive**:
- A full E1 run takes hours, not most of a day, and does not block the laptop.
- Seeds for E1, E3 and E4 become practical within the weekly quota.

**Negative / tradeoffs**:
- A private copy of BraTS data sits on Kaggle. The BraTS 2023 terms are still "to confirm" in `data/DATA.md`; confirm that they allow this before the thesis is submitted (ask the supervisor if unclear).
- Sessions can die. The resume steps in KAGGLE.md must be followed, including deleting a stale `train.pid`.
- Kaggle now also holds model weights trained on BraTS data, and E2 uses more GPU quota.
- Runs are not bit-identical across GPUs. Only the 30-step loss match is shown; no cross-machine claim beyond that.

**Neutral**:
- The 2 GPUs are not used together; one run uses one GPU.

## Follow-up

- [x] Record the Kaggle session limit in this ADR, read from Kaggle, not assumed. Done 2026-11-07: 12 hours for GPU sessions, see the confirmation study line above. The GPU quota is recorded there too.
- [ ] Add the driver version to `run.json` (AC-5) through a spec change; until then the KAGGLE.md GPU check cell prints it for the notes.
- [ ] First E1 run on Kaggle passes the checks (AC-3, AC-4); then mark this ADR fully done. AC-4 passed 2026-11-03: run `20261103-023817-f9b3afb6`, in float32 (spec 0003 step 9), 0.600 s/iter on real data, 7.4 h of training and val, about 8.2 h for the whole notebook. This replaces the fp16 benchmark's 2.2 h as the planning number. AC-3 is still to confirm.
- [ ] Confirm the BraTS 2023 terms allow a private copy on Kaggle; record it in `data/DATA.md`.

## References

- `experiments/0003-baseline/gpu_bench.py`, `experiments/0003-baseline/KAGGLE.md`, `experiments/0004-missing-eval/KAGGLE.md`, `experiments/0005-strategies/KAGGLE.md`
- ADR 0001 (environment, CUDA 13 build)
- `docs/specs/thesis/0003-complete-input-baseline/`, `docs/specs/thesis/0004-missing-modality-evaluation/`, `docs/specs/thesis/0005-modality-dropout-presence-indicator/`
