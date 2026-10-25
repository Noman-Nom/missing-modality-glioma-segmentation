# 0001. Tracer bullet: rationale

Decision record for [index.md](index.md). `/develop` builds from `index.md` and skips this file.

## Context

The thesis asks how glioma segmentation degrades when MRI sequences are missing (RQ1) and which simple strategy keeps it reliable (RQ2). Every answer depends on a long pipeline: reading BraTS NIfTI files, mapping four modalities and three labels correctly, normalising intensities that the audit shows are wildly different between cases (99th percentiles from about 570 up to 3.5 million), cropping to fit a 4 GB GPU, training, sliding window inference, scoring WT, TC and ET under 15 missing patterns, and moving those numbers into LaTeX without hand copying. A silent error anywhere (a swapped channel, the wrong label set, a metric that skips empty regions) would invalidate every later table.

What exists today: a data audit (1470 cases clean), a subject level split (791 / 114 / 228 subjects), and a working environment (ADR 0001) whose smoke test showed a MONAI 3D U-Net forward and backward pass at 96^3 uses 144 MiB peak. Nothing yet trains, evaluates or writes a table. `experiments/common/` is empty. ADR 0001 deferred the determinism settings to this spec.

Forces: the project follows Tracer Bullet (one thin thread through every layer, then thicken). The AI must not run long jobs, so the full run has to be short enough for the human to rerun freely, and a quicker mode is needed for the AI's own checks. Several rules here outlive the tracer: the channel order, the label to region map, the normalisation, the missing channel fill, the 15 pattern IDs, and the empty region rule. The repo rules require the missing pattern protocol to be fixed before any test fold result exists, and require every number to come from a script with a logged config and seed. The hardware is a Quadro T2000 with 4096 MiB and about 5 GB of free system RAM.

If this is not decided now, each later experiment would reinvent these pieces slightly differently, and the paired comparisons RQ2 needs would compare pipelines rather than strategies.

## Options considered

### Option 1: thin full thread, data to thesis table, all 15 patterns

Train a small model on 16 subjects for 2,000 iterations, evaluate 4 val subjects under all 15 zero fill patterns, write a long metrics CSV and a run record, and generate a LaTeX table that a stub document builds.

**Pros**:
- Exercises every layer, including writing, so the "no manual copy" rule is real from day one.
- Freezes the pattern list and empty region rule in tested code before any test fold result.
- Cheap enough (about 20 minutes) to rerun after each change.

**Cons**:
- More code than a CSV only thread (a table script and a stub LaTeX file).
- 4 val subjects give noisy numbers that could be mistaken for results.

### Option 2: data to metrics CSV only

The same thread but stopping at `metrics.csv`; the writing hook arrives with feature 12.

**Pros**:
- Less to build now.

**Cons**:
- The writing layer stays untested until the end, which is exactly where tracer bullets are meant to remove risk.
- The table format gets decided under deadline pressure.

### Option 3: a larger run that doubles as the baseline (64 train / 16 val)

Make the tracer a first real baseline.

**Pros**:
- Gives a first estimate of time and memory at a more realistic scale.

**Cons**:
- 1 to 2 hours per run makes the thread slow to iterate on.
- Blurs into feature 5, so baseline choices (augmentation, caching, timepoints, budget) would be made before the plumbing is proven.

## Rationale

Option 1 matches the Tracer Bullet approach: the riskiest parts of this thesis are not model quality but correctness of the plumbing (channel and label mapping, empty region scoring, pattern definitions) and traceability of numbers into the text. Option 1 tests all of that at the smallest size where learning is still visible. Option 2 leaves the writing layer as the last untested piece. Option 3 is the baseline itself and belongs to feature 5, once the thread is known to work.

The sub decisions, each with its reason:

- **Sigmoid regions over softmax classes.** The thesis reports WT, TC and ET, so training on those regions targets what is scored. The label map is written by hand ({1,2,3}, {1,3}, {3}) because MONAI's built in BraTS transform assumes the older {1, 2, 4} labels; BraTS 2023 uses 3 for ET (audit, all 1251 files). Runner up: 4 class softmax, which is valid but optimises a different target.
- **Clip then z-score over nonzero voxels, zero fill.** The audit found extreme outlier intensities, so clipping before z-score is necessary. After z-score, zero is the brain mean, a neutral fill that the thesis can state in one sentence. Min max scaling would make zero mean "darkest tissue", a stronger and less neutral signal.
- **Crop on all 4 modalities, before the pattern.** Keeps all 15 patterns on identical grids so paired per case comparison is exact. The cost (using information a real missing scenario lacks) affects only the background crop, not the tumour, and is stated in Consequences.
- **96^3 patches, 1:1 tumour sampling, no augmentation.** 96^3 is the size the smoke test measured. Tumour biased sampling is needed because ET is small (23 cases under 500 voxels). The sampler uses a separate 1 channel WT key, because MONAI treats channel 0 of a multichannel label as background (the cross check found this). Augmentation is a baseline concern; adding it here would hide whether the bare thread learns.
- **Cache the fixed preprocessing (changed after the cross check).** You first chose on the fly loading. The cross check estimated, without measuring, that decompressing 5 NIfTI files and computing percentiles every step would push 2,000 iterations past 30 minutes. With `CacheDataset`, the fixed steps run once per case (about 1 GB for 16 cases), only the random patch crop runs per step, and determinism gets easier. The cost is that the tracer no longer measures real loading speed at scale; feature 5 has to deal with that.
- **Mean and median in the table, plus the empty rule count.** With n = 4, one 373.13 mm penalty can dominate a mean HD95. Showing the median and how many rows the empty rule scored makes that visible.
- **Smoke test U-Net, DiceLoss plus BCE, AdamW.** Reusing the measured model avoids a new memory unknown. Dice plus BCE is the usual pairing for sigmoid region outputs: Dice handles class imbalance, BCE gives stable early gradients.
- **Evaluate `last.pt`, not `best.pt`.** Selecting and reporting on the same 4 subjects would bias the tracer numbers upward. Keeping `best.pt` saved costs nothing and tests the saving path for feature 5.
- **Timepoint 000 only.** One case per subject keeps paired tests over independent units. Averaging per subject is a valid runner up but adds code with no tracer benefit.
- **BraTS empty region rule.** It defines a number for every case, so no region silently drops out of a mean, and it matches the convention in BraTS reporting so later comparisons are possible. The cost is a large HD95 penalty (373.13 mm) that can dominate means; reporting medians later can address that.
- **Seeded, deterministic cuDNN, `warn_only`.** Strict mode may raise on some 3D backward ops and block training; seeds only gives no reproducibility claim at all. `warn_only` plus logging which ops warned is honest and keeps the run going.
- **Fail loud, no resume.** A 20 minute run is cheap to restart; resume logic is untested risk. Skipping bad batches would hide bugs.
- **TOML config.** Python's `tomllib` reads it with no new dependency, so no ADR is needed. YAML would need PyYAML and an ADR first.
- **Sources only references.** No web fetch was needed: every choice rests on project files or well known practice. Nothing is cited into `refs.bib`.

## References

**Project sources**:
- `AGENTS.md` (Tracer Bullet, reproducibility, fixed protocol, fair comparison, never poll long runs)
- `docs/architecture/decisions/0001-python-environment.md` (stack, memory plan, deferred determinism)
- `data/DATA.md` (label values, suffixes, intensity outliers, empty ET counts, stray folders, split sizes)
- `tests/test_env_smoke.py` (U-Net config, 144 MiB peak at 96^3, Dice and HD95 toy cases)
- `docs/architecture/system-overview.md` (pipeline order)
- `context/project-draft.md` sections 6 and 7 (preprocessing, metrics, empty region note)

**Practices & standards**:
- BraTS region convention (WT, TC, ET as overlapping regions) and its empty region scoring
- Clip and z-score intensity normalisation inside the brain for multi site MRI
- Dice plus cross entropy loss for imbalanced segmentation
- PyTorch reproducibility guidance (seeds, cuDNN flags, deterministic algorithms, cuBLAS workspace)
- Tracer bullet development (thin end to end thread, then thicken)
