# 0003. Complete input baseline (E1): rationale

The decision record behind [index.md](index.md). `/develop` builds from `index.md` and skips this file.

## Context

The tracer (spec 0001) proved the thread from raw BraTS files to a LaTeX table on 16 train and 4 val subjects, with a 0.6M parameter U-Net, 96^3 patches and 2,000 iterations (peak reserved 484 MiB, 385 s). That model was a wiring check, not a baseline. RQ1 (how much performance drops when modalities are missing) and RQ2 (which strategy copes best) both measure against one reference model, so that model has to be credible, trained on the whole train fold, and chosen without touching the test fold.

Four forces shape the problem. First, the hardware: a Quadro T2000 with 4 GB VRAM and a laptop with 15 GB RAM (about 3 GB free when checked on 2026-10-27). The 867 train fold cases cropped at float32 come to about 52 GB (estimated), so the tracer's full RAM cache cannot work. Second, fairness: E3 (modality dropout) and E4 (presence indicator) must use the same data, budget, loop and selection rule, or the RQ2 comparison is not paired. Third, the protocol rule in `AGENTS.md`: the evaluation protocol is fixed before any test result is seen, and model selection uses val only. Fourth, time: each strategy and seed costs a long GPU run on a laptop that can sleep, overheat or crash.

Spec 0001 left five choices open for E1: both timepoints or one, augmentation, caching, iteration budget and checkpoint selection. If they stay undecided, `/develop` would have to invent them, and E3 and E4 would later inherit those guesses as if they were reasoned.

## Options considered

### Option 1: Shared loop in `common/`, float16 disk cache by its own script, wider U-Net at 128^3

A `preprocess.py` writes each case once as float16 `.npz`, guarded by a preprocess hash. A new `common/training.py` holds the loop with a `batch_hook` for E3 and E4. The model is MONAI `UNet` (32 to 320 channels, 4 downsamplings, 2 res units) on 128^3 patches. Training runs 40,000 iterations with warmup and cosine lr, val on all 114 val subjects every 4,000 iterations, selection by mean region Dice, and logged resume.

**Pros**:
- One cache and one loop for E1 to E4, so the comparison is paired by construction.
- Preprocessing outlives runs as a documented script, which the `AGENTS.md` rule asks for.
- Uses about 30 GB of the 134 GB free disk, with constant RAM.
- Resume protects multi hour runs, and every resume is recorded.

**Cons**:
- More shared code to build and review now (cache, loop, resume) before any number exists.
- float16 storage adds small rounding the tracer did not have.
- Resume weakens the determinism claim for runs that crashed.
- The memory fit and run time are unmeasured estimates until the memory test runs.

### Option 2: Copy the tracer, swap in MONAI `PersistentDataset`, keep everything in the E1 folder

Copy `experiments/0002-tracer-bullet/train.py` into `0003-baseline/`, replace `CacheDataset` with `PersistentDataset`, raise the iteration count and the model size, and keep fail loud with no resume.

**Pros**:
- Least new code, and it is fast to get a first number.
- `PersistentDataset` is built into MONAI and needs no cache script.

**Cons**:
- E3 and E4 would copy the loop again or move it to `common/` later. The first breaks the sharing rule, and the second means reworking a verified loop under deadline.
- The first pass over the data happens inside the timed run, float32 pickles take about 52 GB, and invalidation relies on `hash_transform`.
- A crash late in a multi hour run loses the whole run.

### Option 3: A stronger established architecture (MONAI `SegResNet`, or nnU-Net style training)

Use a BraTS proven architecture or the nnU-Net recipe (poly lr, heavy augmentation, deep supervision) for a higher baseline.

**Pros**:
- Likely higher absolute Dice, closer to published BraTS numbers.
- Recipes that reviewers recognise.

**Cons**:
- No longer the "compact 3D U-Net" the proposal and thesis title describe, so the framing would need to change.
- nnU-Net is a new framework, and the rules require an ADR first. Its memory needs and run times on 4 GB are uncertain.
- The study is about relative degradation and strategy differences, and a higher absolute baseline does not answer RQ1 or RQ2 better.

## Rationale

Option 1 wins because fairness is the thesis's main contribution, and fairness has to be built in, not added later. A loop shared through `common/` with one hook point means E3 and E4 differ from E1 only in what the hook does to the batch, which is exactly the controlled comparison RQ2 asks for. Option 2 saves a day now, but it guarantees a refactor of the most load bearing code right before the strategy runs, when a subtle change would quietly break the pairing.

The 15 GB RAM limit rules out the tracer's caching, and the `AGENTS.md` rule that anything outliving one run gets its own script points to a preprocessing step. float16 halves the disk use and the read time, and its rounding (about 5e-4 relative) is far below the noise between training runs. The preprocess hash makes a stale cache impossible to use by accident.

The wider U-Net at 128^3 keeps the thesis framing (a compact 3D U-Net) while using the 4 GB card for real. The tracer used about 12% of it. Option 3's possible gain in absolute Dice does not serve the research questions, and it costs an ADR and a changed framing. The memory test is the gate that turns the estimate into a measurement before hours are spent.

Val only scoring, with the test fold uncached, is the plainest way to satisfy "protocol fixed before test results are seen": E1 cannot leak what it never reads. Selecting on the mean of WT, TC and ET Dice, rather than WT alone, keeps the harder regions in the choice. Resume was chosen because a crash in hour five of a laptop run is likely over 9 to 15 runs, and the weaker determinism claim is stated honestly in the spec.

## Choices from the design conversation

Each was put to the engineer with a recommended pick. All recommended picks were accepted.

| Choice | Picked | Runner up |
|---|---|---|
| What E1 scores | val only, test untouched | val plus complete input test |
| Train timepoints | both (867 cases before the flag) | timepoint 000 only (791) |
| Flagged FLAIR cases | exclude through the config flag until viewed | keep |
| Seeds | seed in config with `--seed`, run 42 now | three seeds now |
| Model and patch | wider UNet, 128^3, batch 1 | wider UNet, 96^3, batch 2 |
| Augmentation | flips plus brain only intensity scale and shift | flips only |
| Budget | 40,000 iterations | 20,000 |
| LR schedule | warmup 500, cosine to 0 | poly 0.9 |
| Caching | own script, float16 `.npz` on disk | `PersistentDataset` |
| Val cadence | all 114 every 4,000 | all 114 every 2,000 |
| Selection | `best.pt` by mean of WT, TC, ET Dice | `best.pt` by WT Dice |
| Resume | from `last.pt`, logged and guarded | fail loud, no resume |
| Code layout | loop in `common/training.py`, thin E1 folder | all in the E1 folder |
| Memory test | 200 steps + 1 val case, below 3,584 MiB | same, below 4,096 MiB |
| Postprocessing | none | remove small ET components |
| Cache folds | train and val now, test in E2 | all three now |
| Cache file | `.npz`, float16 image, raw uint8 label | `.npz` with region masks |
| Val log | per case rows | means only |
| Checkpoints | `best.pt` weights + `last.pt` full state | keep every val point |
| E1 output | val metrics + LaTeX table | val CSV only |
| Fast check | quick run twice + synthetic cases | unit tests only |
| Failures | stop, `failed`, resumable | skip the bad step |
| Resume guard | refuse on config hash, warn on commit | refuse on either |

Decided while writing (recommended, not asked):

| Choice | Picked | Why | Runner up |
|---|---|---|---|
| `.npz` compression | uncompressed | z-scored float16 compresses poorly, and decoding would load the CPU workers on every step | `savez_compressed` |
| Workers | 4 | enough to keep a 128^3 step fed from disk on 16 cores without using much RAM | 2 (tracer) |
| Inference roi | 128^3 | matches the training patch | 96^3 |
| Val metric during training | Dice only | HD95 on 114 cases at every val point costs time and does not enter the selection | Dice and HD95 |
| Warmup start | lr 0 | the simplest linear warmup | lr 1e-5 |
| Resume loader seed | `seed * 100003 + iteration` | deterministic and distinct per resume point | reuse the original seed |
| Sanity floor | `best_score > 0.60` | a loose floor fixed before any number, which catches a broken run without tuning to a target | no floor |
| Cache dir name | `data/processed/0003-baseline/` | confirmed in the data model review | a dataset named folder |
