# 0001. Tracer bullet: one thin end to end run from BraTS files to a thesis table

**Date**: 2026-10-25
**Status**: Accepted

## Summary

We build one small but complete run that touches every layer of the thesis: it reads BraTS cases, preprocesses them, trains a compact 3D U-Net for 2,000 steps on 16 subjects, evaluates 4 validation subjects under all 15 missing modality patterns, writes a metrics file, and turns that file into a LaTeX table that a stub document builds. The numbers it produces are not results; the point is that every piece exists, is tested, and connects. The shared rules it fixes here (channel order, label to region mapping, normalisation, missing channel fill, the empty region rule, the run record) are the ones every later experiment reuses, so they are decided once, now.

## Requirements

**User stories**:
- As the student, I want one command chain that goes from raw BraTS files to a table in a LaTeX document, so that every later experiment only thickens a pipeline that already works.
- As the supervisor or examiner, I want every number in that table to trace back to a script, a config hash, a git commit and a seed, so that I can trust and rerun it.

**Acceptance criteria**:
- **AC-1**: Subsets are chosen by code from `data/splits/splits.csv`: 16 train subjects from fold `train` and 4 val subjects from fold `val`, each the first N subjects by sorted subject ID among rows with `timepoint == 0` and `flair_mask_flag == False`. No subject comes from fold `test`, no subject is in both subsets, and two calls return the same lists.
- **AC-2**: Channels are built in the fixed order T1, T1ce, T2, FLAIR from the BraTS file suffixes `-t1n`, `-t1c`, `-t2w`, `-t2f` (matched by name, never by position). Labels map to three region masks: WT = {1, 2, 3}, TC = {1, 3}, ET = {3}. A fast test checks both on toy inputs.
- **AC-3**: Preprocessing, per case: first crop the raw 4 channel stack to the bounding box of voxels `!= 0` in any modality (margin 0), the same box for image and label; the count of nonzero label voxels is unchanged by the crop (asserted). Then, per modality, compute the brain mask as raw voxels `!= 0` (before clipping), clip to the 0.5th and 99.5th percentile over that mask, and z-score over that mask; voxels outside the mask are set to 0. A channel with no nonzero voxels, or std below 1e-8, stays all 0 and logs a warning.
- **AC-4**: `experiments/common/missing.py` defines exactly 15 non empty modality subsets in one fixed order with stable IDs, complete input first. A missing channel is set to all zeros after normalisation (zero equals the brain mean after z-score). The crop box does not depend on the pattern.
- **AC-5**: Training runs 2,000 iterations (batch 1, one 96^3 patch per step, fp16 autocast) to completion on the Quadro T2000. Peak allocated and peak reserved VRAM are logged, and peak reserved is below 4096 MiB.
- **AC-6**: Learning signal: in `train_log.csv`, mean `loss` over the last W iterations is below the mean over the first W, where W = min(50, iterations // 2); and in `metrics.csv`, mean `dice` over rows with `pattern_id == 1` and `region == "WT"` (the final checkpoint, complete input, 4 val subjects) is above 0.3.
- **AC-7**: Evaluation of the final checkpoint writes `metrics.csv` with exactly n_val × 15 × 3 rows (case × pattern × region; 180 for the full run, n_val read from `run.json`), each with Dice, HD95 in mm, sensitivity and precision computed under the AC-8 rule. No Dice or HD95 value is NaN.
- **AC-8**: `experiments/common/metrics.py` holds the one empty region rule (BraTS convention): truth and prediction both empty gives Dice 1 and HD95 0; exactly one empty gives Dice 0 and HD95 373.13 mm; sensitivity is NaN when truth is empty; precision is NaN when prediction is empty. Fast tests cover all four cases plus the known Dice 0.75 and HD95 3.0 toy cases.
- **AC-9**: Every run writes `run.json` with exactly the keys in the *run.json keys* table below (status, failure reason, config hash, git commit and dirty flag, seed, package versions, GPU, determinism settings and warnings, peak VRAM, wall time, iterations, loss means, val WT Dice history, subject lists, skipped scaler steps), and copies the effective config to `config.toml` in the run folder.
- **AC-10**: `experiments/common/seed.py` has one `set_seed()` that seeds Python, NumPy and torch, calls `monai.utils.set_determinism(seed)`, sets the determinism flags, and returns a seeded `torch.Generator` for the DataLoader. A fast CPU test runs 3 training steps of a tiny U-Net twice with the same seed and gets bitwise equal losses, then restores `use_deterministic_algorithms(False)`.
- **AC-11**: A non finite loss (checked with `torch.isfinite(loss)` before backward) or a CUDA out of memory error stops the run with a nonzero exit and a `run.json` whose status is `failed` with the reason. Steps that GradScaler skips for inf gradients are normal, not failures; their count is logged. A missing input file for any selected case is reported before training starts. Ctrl+C writes status `interrupted`.
- **AC-12**: `make_table.py` reads only the run's `metrics.csv` and `run.json` and writes `writing/tables/0002-tracer-bullet.tex`: 15 rows (one per pattern), and per region (WT, TC, ET) mean Dice, median Dice, mean HD95, median HD95, and the count of rows scored by the empty rule (`gt_empty` or `pred_empty`); the caption states the real n_val and that it is a tracer bullet, not a result. `writing/tracer/tracer.tex` inputs that table and builds with `latexmk -xelatex`.
- **AC-13**: `check_tracer.py <run dir>` validates a run folder (status ok, n_val × 15 × 3 rows, expected columns and the 15 pattern IDs, AC-5 and AC-6 thresholds) and exits 0 on pass, 1 on fail with the reason. On a run with `quick: true` it runs the structure checks, then always prints `FAIL: quick run` and exits 1.
- **AC-14**: `uv run pytest` (fast tests) passes with no data and no GPU. `docs/whats-built.md` and `context/progress-tracker.md` are updated in the same change.

## Decision

**Chosen option**: Option 1: a thin full thread, data to thesis table, with all 15 patterns evaluated by zero fill.

The tracer trains on complete input only and evaluates zero fill under every pattern, using the shared modules in `experiments/common/` that every later experiment reuses.

Choices fixed by this spec (each with its runner up; reasoning in [rationale.md](rationale.md)):

| Area | Choice | Runner up |
|---|---|---|
| Output | 3 sigmoid region channels WT, TC, ET, threshold 0.5, no hierarchy enforced | 4 class softmax |
| Normalise | clip 0.5/99.5 pct over nonzero, z-score over nonzero, background 0 | min max to [0, 1] |
| Missing fill | zeros after normalisation | learned or mean token (later strategies) |
| Crop | bounding box of raw voxels `!= 0` in any modality, margin 0, before normalising and before any pattern | crop per available modality |
| Patches | `SpatialPadd` to 96^3 (value 0), then `RandCropByPosNegLabeld` with `label_key="wt"` (a separate 1 channel WT mask, not the 3 channel label, because MONAI treats channel 0 of a multichannel label as background), `image_key="image"`, `image_threshold=0`, pos 1, neg 1, `num_samples=1` | 128^3 |
| Augmentation | none in the tracer | random flips |
| Loading | `monai.data.CacheDataset` (`cache_rate=1.0`) for the fixed steps (load, cast, crop, normalise, regions, WT key, pad); only the random patch crop runs per step. `monai.data.DataLoader`, 2 workers, `shuffle=True`, seeded `generator`, wrapped in an endless cycling iterator (2000 / 16 = 125 passes) | on the fly, no cache (slower, about 30 min estimated) |
| Model | MONAI `UNet`, 3D, in 4, out 3, channels (16, 32, 64, 128), strides (2, 2, 2), `num_res_units=1` (the smoke test model, 602,594 params) | MONAI `SegResNet` |
| Loss | MONAI `DiceLoss(sigmoid=True)` (defaults: `batch=False`, `smooth_nr=smooth_dr=1e-5`) + `torch.nn.BCEWithLogitsLoss`, equal weight, loss computed in float32 | `DiceFocalLoss` |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-5, no scheduler, `torch.amp.GradScaler` | SGD with poly decay |
| Length | 2,000 iterations, val every 500 (at 500, 1000, 1500, 2000): complete input, same `[infer]` settings and `region_metrics()` as evaluation, mean WT Dice | 20 epochs |
| Checkpoints | save `last.pt` and `best.pt` (highest val WT Dice; a tie keeps the earlier one); **evaluate `last.pt` only** | evaluate `best.pt` |
| Inference | MONAI `sliding_window_inference`, roi 96^3, `sw_batch_size=1`, overlap 0.5, gaussian mode, padding constant 0; input and output buffers float32, only the model call runs under fp16 autocast, logits cast to float32 before sigmoid | overlap 0.25 |
| HD95 | `monai.metrics.compute_hausdorff_distance(pred, gt, percentile=95, directed=False, spacing=pixdim)` on binary masks, only when both are non empty; on the cropped grid, which gives the same value as the full grid because no tumour lies outside the crop | own SciPy code |
| Evaluation cases | timepoint 000 only | average per subject |
| Determinism | seeded, `monai.utils.set_determinism`, `cudnn.benchmark=False`, `cudnn.deterministic=True`, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; warnings captured with `warnings.simplefilter("always")` in the main process. Claim: seeded with logged nondeterministic ops, not bitwise GPU reproducibility | strict (error on nondeterministic ops) |
| Failures | fail loud, no resume | resume from checkpoint |
| Config | `config.toml` read with stdlib `tomllib` | YAML (needs PyYAML and an ADR) |
| Seed | 42 (same as the split) | none |

No new dependency is added. Everything uses what ADR 0001 installed (torch, MONAI, NiBabel, SciPy, NumPy, pytest).

**Implementation skills**: none (no community skills are installed for this stack).

## Rationale

Reasoning and options: see [rationale.md](rationale.md).

## Feature design

**Code layout** (new files; experiments import only from `common/`):

| Path | Holds |
|---|---|
| `experiments/common/__init__.py` | empty, makes `common` a package |
| `experiments/common/seed.py` | `set_seed(seed)`, the determinism flags |
| `experiments/common/data.py` | `MODALITIES`, `SUFFIX`, `label_to_regions()`, `select_subjects()`, `case_paths()`, `check_inputs()`, `normalise()`, `crop_box()`, the MONAI transform lists |
| `experiments/common/missing.py` | `PATTERNS`, `apply_pattern()` |
| `experiments/common/metrics.py` | `region_metrics()` with the empty region rule, `HD95_EMPTY_MM = 373.13` |
| `experiments/common/runlog.py` | `load_config()`, `config_hash()`, `git_info()`, `versions()`, `RunRecord` (writes `run.json`) |
| `experiments/0002-tracer-bullet/config.toml` | the one config |
| `experiments/0002-tracer-bullet/train.py` | training, writes the run folder |
| `experiments/0002-tracer-bullet/evaluate.py` | 15 pattern evaluation, writes `metrics.csv` |
| `experiments/0002-tracer-bullet/make_table.py` | `metrics.csv` to LaTeX table |
| `experiments/0002-tracer-bullet/check_tracer.py` | the experiment check (AC-13) |
| `tests/test_common_*.py` | fast tests (AC-2, 3, 4, 8, 10, and subset selection AC-1) |
| `writing/tracer/tracer.tex` | stub XeLaTeX document (Times New Roman) that `\input`s the table |

`pyproject.toml` gets `pythonpath = ["experiments"]` under `[tool.pytest.ini_options]` so tests import `common.data`, and a minimal `[tool.ruff]` block (`line-length = 100`, `target-version = "py312"`) so CI's `ruff check .` has a config. `.gitignore` gets `writing/tables/` (generated output, rebuilt from `results/`). CSV files are read and written with the stdlib `csv` module (pandas is not installed): `splits.csv` booleans are the strings `"True"`/`"False"`, and `subject` stays a string (leading zeros).

**Data model sketch** (files, not a database):

`config.toml` (all fields required):

```toml
seed = 42
[data]
splits_csv = "data/splits/splits.csv"
n_train_subjects = 16
n_val_subjects = 4
timepoint = 0
exclude_flair_flag = true
[preprocess]
clip_low_pct = 0.5
clip_high_pct = 99.5
cache_rate = 1.0
[train]
iterations = 2000
val_every = 500
patch = [96, 96, 96]
pos = 1
neg = 1
batch_size = 1
num_workers = 2
lr = 1e-3
weight_decay = 1e-5
amp = true
[model]
channels = [16, 32, 64, 128]
strides = [2, 2, 2]
num_res_units = 1
[infer]
roi = [96, 96, 96]
sw_batch_size = 1
overlap = 0.5
threshold = 0.5
```

`--quick` overrides (applied before hashing): `n_train_subjects = 2`, `n_val_subjects = 1`, `iterations = 10`, `val_every = 5`.

Run folder `results/0002-tracer-bullet/<run_id>/` (git ignored), where `run_id = YYYYMMDD-HHMMSS-<first 8 of config hash>` in UTC:

| File | Content |
|---|---|
| `config.toml` | the effective config (after `--quick` overrides), the one `evaluate.py` reads |
| `run.json` | the keys below; `status` in {`running`, `ok`, `failed`, `interrupted`} |
| `train_log.csv` | `iteration, loss, dice_loss, bce_loss, lr, seconds` one row per iteration; plus `val_wt_dice` on val rows |
| `metrics.csv` | long format, columns below |
| `checkpoints/last.pt`, `checkpoints/best.pt` | model state dict, iteration, config hash |

**run.json keys** (all present on every run; `null` where not yet known):

| Key | Type | Meaning |
|---|---|---|
| `run_id` | str | as above |
| `status` | str | `running`, `ok`, `failed`, `interrupted` |
| `reason` | str or null | failure reason |
| `quick` | bool | `--quick` used |
| `config_hash` | str | sha256 hex of the effective config |
| `git_commit` | str | `git rev-parse HEAD` |
| `git_dirty` | bool | `git status --porcelain` non empty |
| `seed` | int | from config |
| `data_dir` | str | resolved data path (recorded, not hashed) |
| `device` | str | `cuda` or `cpu` |
| `gpu` | str or null | `torch.cuda.get_device_name()` |
| `versions` | object | keys `python`, `torch`, `cuda`, `cudnn`, `monai`, `nibabel`, `numpy`, `scipy` |
| `determinism` | object | keys `cudnn_benchmark`, `cudnn_deterministic`, `deterministic_algorithms`, `warn_only`, `cublas_workspace_config` |
| `nondeterminism_warnings` | list of str | unique warning messages |
| `peak_vram_allocated_mib` | float or null | `torch.cuda.max_memory_allocated()` / 2^20 |
| `peak_vram_reserved_mib` | float or null | `torch.cuda.max_memory_reserved()` / 2^20 |
| `wall_time_s` | float | training wall time |
| `iterations` | int | iterations completed |
| `scaler_skipped_steps` | int | steps GradScaler skipped |
| `loss_window` | int | W from AC-6 |
| `loss_start_mean` | float | mean loss of the first W iterations |
| `loss_end_mean` | float | mean loss of the last W iterations |
| `val_wt_dice` | list of [int, float] | (iteration, mean WT Dice) at each val point |
| `best_iteration` | int or null | iteration of `best.pt` |
| `train_cases` | list of str | case IDs |
| `val_cases` | list of str | case IDs (`evaluate.py` uses this list, never selects again) |

`metrics.csv` columns: `run_id` (str), `case` (str, e.g. `BraTS-GLI-00002-000`), `subject` (str, 5 digit), `pattern_id` (int 1 to 15), `pattern` (str, e.g. `T1+T1ce+T2+FLAIR`), `region` (WT, TC or ET), `dice` (float), `hd95_mm` (float), `sensitivity` (float or empty), `precision` (float or empty), `gt_empty` (bool), `pred_empty` (bool). Unique key: (`run_id`, `case`, `pattern_id`, `region`).

**Pattern list** (fixed order; bit order T1 = 1, T1ce = 2, T2 = 4, FLAIR = 8). IDs are positions 1 to 15 in this list, sorted by number of present modalities descending, then by bitmask descending:


| ID | Present | Missing | Bitmask |
|---|---|---|---|
| 1 | T1+T1ce+T2+FLAIR | none | 15 |
| 2 | T1ce+T2+FLAIR | T1 | 14 |
| 3 | T1+T2+FLAIR | T1ce | 13 |
| 4 | T1+T1ce+FLAIR | T2 | 11 |
| 5 | T1+T1ce+T2 | FLAIR | 7 |
| 6 | T2+FLAIR | T1, T1ce | 12 |
| 7 | T1ce+FLAIR | T1, T2 | 10 |
| 8 | T1+FLAIR | T1ce, T2 | 9 |
| 9 | T1ce+T2 | T1, FLAIR | 6 |
| 10 | T1+T2 | T1ce, FLAIR | 5 |
| 11 | T1+T1ce | T2, FLAIR | 3 |
| 12 | FLAIR | T1, T1ce, T2 | 8 |
| 13 | T2 | T1, T1ce, FLAIR | 4 |
| 14 | T1ce | T1, T2, FLAIR | 2 |
| 15 | T1 | T1ce, T2, FLAIR | 1 |

The list is generated by code and a test pins this exact table, so an ID never changes meaning.

**State transitions** (`run.json` status): `running` → `ok` | `failed` | `interrupted`. Written as `running` at start, rewritten once at exit. A run folder with status `running` and no live process is a crashed run; `check_tracer.py` fails it.

**API surface** (command line; all run with `uv run`, from the repo root):

| Command | Key inputs | Key outputs | Key errors |
|---|---|---|---|
| `python experiments/0002-tracer-bullet/train.py` | `--config` (path, default the folder's `config.toml`), `--data-dir` (path, default env `BRATS_TRAIN_DIR`), `--splits` (path, default `[data] splits_csv`, so tests can pass a mini file), `--out` (default `results/0002-tracer-bullet`), `--quick` (flag, overrides above), `--device` (`auto` or `cpu`, default `auto`) | run folder with `config.toml`, `run.json`, `train_log.csv`, checkpoints; prints run dir, config hash, git commit | exit 2: no data dir or missing case files (before training); exit 1: OOM or non finite loss (status `failed`); exit 130: Ctrl+C |
| `python experiments/0002-tracer-bullet/evaluate.py` | `--run` (run dir, req), `--data-dir` (as above), `--device` (as above); reads the run's `config.toml` and `run.json` `val_cases` | `metrics.csv` in the run dir; prints mean WT Dice for pattern 1 | exit 1: run status not `ok`, or `last.pt` missing |
| `python experiments/0002-tracer-bullet/make_table.py` | `--run` (run dir, req) | `writing/tables/0002-tracer-bullet.tex` | exit 1: `metrics.csv` missing or row count not n_val × 15 × 3 (n_val = length of `run.json` `val_cases`) |
| `python experiments/0002-tracer-bullet/check_tracer.py` | run dir (positional) | `PASS` or `FAIL: <reason>` | exit 1 on any failed check |
| `cd writing/tracer && latexmk -xelatex tracer.tex` | the generated table | `tracer.pdf` | LaTeX error if the table is missing |

`--quick` runs (the AI's fast gate) are marked `"quick": true` in `run.json`. `check_tracer.py` runs the structure checks on them and then always prints `FAIL: quick run` and exits 1, so a quick run can never pass as the real one.

Test hook: when the env var `TRACER_INJECT_INF=1` is set, `train.py` replaces the input of iteration 1 with `inf`, so the AC-11 test can force a non finite loss. Unset, it does nothing.

**Table layout** (`writing/tables/0002-tracer-bullet.tex`): a `table` with `booktabs` rules, `\caption{Tracer bullet, n = <n_val> val subjects, final checkpoint. Not a result.}` and `\label{tab:tracer}`. Columns: pattern ID, present modalities (the pattern string), then for each region WT, TC, ET: Dice mean, Dice median, HD95 mean, HD95 median, empty rule count. Dice printed with 3 decimals, HD95 with 1. `writing/tracer/tracer.tex` is an `article` with `fontspec` (Times New Roman), `booktabs`, `geometry` landscape, and `\input{../tables/0002-tracer-bullet.tex}`.

**Value sourcing**:

| Action | Value | Source |
|---|---|---|
| select subjects | subject IDs, case IDs | `splits.csv` columns `subject`, `case`, `fold`, `timepoint`, `flair_mask_flag`; N from `config.toml [data]` |
| load case | file paths | `--data-dir` / `BRATS_TRAIN_DIR` + `<case>/<case>-{t1n,t1c,t2w,t2f,seg}.nii.gz` |
| load case | voxel spacing for HD95 | NIfTI header `pixdim` of the seg file (audit says 1 mm; read, not assumed) |
| normalise | clip bounds | percentiles from config, computed per volume over nonzero voxels |
| normalise | brain mask | raw voxels `!= 0`, computed before clipping |
| normalise | empty or constant channel (no nonzero voxels, or std below 1e-8) | rule here: channel stays 0 and a warning logged |
| crop | crop box | raw 4 channel stack, any channel `!= 0`, margin 0, before normalising |
| patch | tumour centre candidates | the 1 channel `wt` key (label > 0) |
| missing | which channels are zero | `PATTERNS[pattern_id]` in `common/missing.py` |
| train | seed | `config.toml seed` |
| train | start / end loss | mean of `loss` over the first and last W iterations in `train_log.csv`, W = min(50, iterations // 2) |
| train | peak VRAM | `torch.cuda.max_memory_allocated()` and `max_memory_reserved()` at exit, in MiB |
| train | best checkpoint | val WT Dice (complete input, mean over val subjects) at each `val_every`; tie keeps the earlier |
| evaluate | which cases, which config | `run.json` `val_cases` and the run folder's `config.toml` |
| evaluate | prediction | sigmoid of `last.pt` logits (float32), threshold from the run's `config.toml [infer]` |
| check | AC-6 Dice | `metrics.csv` rows `pattern_id == 1`, `region == "WT"`, mean of `dice` |
| check | AC-6 loss | `run.json` `loss_start_mean`, `loss_end_mean` (recomputed from `train_log.csv` and compared) |
| check | AC-5 | `run.json` `iterations == 2000`, `peak_vram_reserved_mib < 4096` |
| evaluate | Dice, HD95, sensitivity, precision | `common/metrics.py region_metrics()` on the cropped grid |
| evaluate | HD95 for one empty side | `HD95_EMPTY_MM = 373.13` (volume diagonal, sqrt(240² + 240² + 155²)) |
| run record | config hash | sha256 of the effective config (after `--quick` overrides) serialised as sorted key JSON; `data_dir` is not part of it |
| run record | git commit, dirty | `git rev-parse HEAD`, `git status --porcelain` non empty |
| run record | versions, GPU | `sys.version`, `torch.__version__`, `torch.version.cuda`, `torch.backends.cudnn.version()`, package `__version__`, `torch.cuda.get_device_name()` |
| run record | nondeterminism warnings | `warnings.catch_warnings(record=True)` with `simplefilter("always")` around training in the main process, unique messages |
| run record | skipped scaler steps | count of iterations where `scaler.get_scale()` dropped after `scaler.update()` |
| table | mean and median Dice and HD95, empty rule count | over the n_val cases in `metrics.csv` per (`pattern_id`, `region`); count = rows with `gt_empty` or `pred_empty`; printed with 3 and 1 decimals (display formatting only); n_val from `run.json` |

**Key invariants**:
- No `test` fold subject is ever read by any tracer script (asserted in `select_subjects()` and tested).
- The crop box is computed once per case from all 4 modalities before any pattern is applied, so all 15 patterns are scored on the same grid and the same voxels.
- Label voxel count before and after crop is equal (assert per case).
- Missing channels are applied after normalisation; normalisation statistics never see the fill value.
- Every `metrics.csv` row has a matching `run.json` with status `ok` and the same `run_id`.
- The table script reads only `metrics.csv`; no number is typed by hand.

**Security model**: Medical images stay local. Raw cases are read only from the data dir; nothing is written there. Run folders, checkpoints and generated tables are git ignored (`results/*`, `*.pt`, `writing/tables/`). `metrics.csv` holds public BraTS case IDs and scores only, no voxel data. Data use terms are still to confirm (`data/DATA.md`); the tracer does not change that status.

**Configuration required**:
- `BRATS_TRAIN_DIR`: path to `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` (the folder that holds the case folders). Read when `--data-dir` is not given. Put it in `.env` (git ignored) or the shell.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`: set by `set_seed()` before the first CUDA call (needed for deterministic cuBLAS).
- `UV_PROJECT_ENVIRONMENT`, `UV_LINK_MODE`: as ADR 0001.

**Edge cases and failure handling**:
- Missing or unreadable file for a selected case: `check_inputs()` lists every missing file and exits 2 before the model is built.
- Stray folders named `...-t1n.nii` (cases 00011, 00014): only `*.nii.gz` files are matched.
- float32 and int16 images: always cast to float32 on load.
- Negative intensities (10 train cases): nonzero means `!= 0`, so negatives are included in the statistics.
- Val case with empty ET (33 train cases have one; the 4 val subjects may): handled by the AC-8 rule, not skipped.
- CUDA OOM: caught as `torch.cuda.OutOfMemoryError`, `run.json` status `failed`, reason and peak VRAM recorded, exit 1.
- Non finite loss (checked every iteration with `torch.isfinite(loss)` before backward): status `failed`, iteration recorded, exit 1. Not skipped. Inf gradients that GradScaler skips (normal in early fp16 steps) are counted, not treated as failures.
- Cropped brain smaller than 96 in some axis: `SpatialPadd` pads to 96^3 with 0 before the patch crop; sliding window inference pads the same way.
- RAM: 16 cached cropped cases are about 1 GB (estimated, not measured; roughly 60 MB each); the DataLoader uses its default `prefetch_factor=2`.
- Ctrl+C: status `interrupted`, `last.pt` not overwritten mid write (save to temp file, then rename).
- Prediction empty for a region: scored by the AC-8 rule (Dice 0 or 1, HD95 373.13 or 0).

**Critical test scenarios**:
- Happy path: the human runs train, evaluate, make table and the LaTeX build; `check_tracer.py` prints PASS, verifies **AC-5**, **AC-6**, **AC-7**, **AC-12**, **AC-13**
- Data correctness: toy label array `[0,1,2,3]` gives WT `[0,1,1,1]`, TC `[0,1,0,1]`, ET `[0,0,0,1]`; a fake case folder with shuffled file order still gives channels in T1, T1ce, T2, FLAIR order, verifies **AC-2**
- No leakage: subsets contain no `test` subject, no overlap, same on second call, flagged and timepoint 1 rows excluded, verifies **AC-1**
- Empty rule: four toy cases (both empty, truth empty, prediction empty, neither) give the AC-8 values, verifies **AC-8**
- Determinism: two 3 step CPU runs of a tiny U-Net on random 32^3 inputs with seed 42 give equal losses, verifies **AC-10**
- Failure case: with `TRACER_INJECT_INF=1`, `--device cpu`, a test config (patch 32^3, 2 iterations) and tiny synthetic NIfTI cases plus a mini `splits.csv` written into `tmp_path`, the run ends with status `failed` and exit 1; the same setup with one file deleted exits 2 before training, verifies **AC-11**
- Quick run rejected: `check_tracer.py` on a `quick: true` run prints `FAIL: quick run`, exits 1, verifies **AC-13**

## Build plan

Tracer Bullet order: first get a thin thread through every layer with `--quick`, then thicken each layer with its tests and checks. The AI may run fast tests and one `--quick` run (under about 2 minutes); the human runs the full 2,000 iteration run.

1. [x] Add `pythonpath = ["experiments"]` to pytest config and the `[tool.ruff]` block, create `experiments/common/__init__.py` and `seed.py` with `set_seed()`, satisfies **AC-10**
2. [x] `common/runlog.py`: `load_config()`, `config_hash()`, `git_info()`, `versions()`, `RunRecord` writing `run.json` with status transitions, satisfies **AC-9**
3. [x] `common/data.py`: modality and suffix map, `label_to_regions()`, `select_subjects()`, `case_paths()`, `check_inputs()`, satisfies **AC-1**, **AC-2**
4. [x] `common/data.py`: `normalise()`, `crop_box()` and the train and eval MONAI transform lists (load, cast, crop on raw, normalise, regions, `wt` key, pad, then the random pos neg patch as the only random step), satisfies **AC-3**
5. [x] `common/missing.py`: `PATTERNS` and `apply_pattern()`, satisfies **AC-4**
6. [x] `common/metrics.py`: `region_metrics()` with the empty region rule, using MONAI's HD95 only when both sides are non empty, satisfies **AC-8**
7. [x] `0002-tracer-bullet/config.toml` and `train.py` (CacheDataset, seeded DataLoader with endless cycling, model, loss, AMP, val every N, checkpoints, `--quick`, `--device`, config copy into the run folder), satisfies **AC-5**, **AC-6**, **AC-9**
8. [x] `evaluate.py` (sliding window, 15 patterns, `metrics.csv`), satisfies **AC-7**
9. [x] `make_table.py`, `writing/tracer/tracer.tex`, `.gitignore` line for `writing/tables/`, satisfies **AC-12**
10. [x] Thread check: AI runs `train.py --quick`, `evaluate.py`, `make_table.py` and the LaTeX build once end to end; fix whatever breaks, satisfies **AC-7**, **AC-12**
11. [x] Thicken failures: OOM, non finite loss, Ctrl+C, missing files, atomic checkpoint save, with fast tests in `tests/test_tracer_failures.py` (forced non finite loss, missing file), satisfies **AC-11**
12. [x] `check_tracer.py`, satisfies **AC-13**
13. [x] Fast tests in `tests/test_common_data.py`, `test_common_missing.py`, `test_common_metrics.py`, `test_common_seed.py`, plus a `slow` test that runs one real case through the eval transforms and checks shapes, satisfies **AC-1**, **AC-2**, **AC-3**, **AC-4**, **AC-8**, **AC-10**, **AC-14**
14. [x] Human runs the full chain (commands in [verify.md](verify.md)) and pastes the last two lines of each; record the run id, peak VRAM, wall time and pattern 1 WT Dice in `docs/whats-built.md` and `context/progress-tracker.md`, satisfies **AC-5**, **AC-6**, **AC-14**

## Consequences

**Positive**:
- Every later experiment (E1 baseline, E2 patterns, E3 dropout, E4 presence indicator) starts from tested shared code and only changes config and training logic.
- The pattern list, empty region rule and normalisation are fixed in code before any test fold number exists, which the "protocol fixed before test results" rule needs.
- The writing layer is proven: tables come from scripts from the first day.

**Negative / tradeoffs**:
- The first N subjects by sorted ID are not a representative sample; tracer Dice says nothing about model quality and must never be quoted as a result.
- The 4 val subjects are used both to pick `best.pt` and to report tracer numbers. We evaluate `last.pt` to keep the reported numbers free of that selection, but it means the "best" checkpoint is unused here.
- No augmentation: less robust than the real baseline will be. Caching all of the train fold (867 cases) will not fit in RAM, so feature 5 has to choose between partial caching and a precompute step.
- Cropping on all 4 modalities uses information a real missing scenario would not have. This is a simulation choice, documented; it keeps every pattern on the same grid.
- `warn_only` determinism means GPU runs may still differ slightly; the logged warnings say which ops cause it.

**Neutral**:
- `experiments/AGENTS.md` says `config.yaml`; this spec uses `config.toml`. `/sync` should update that line.
- `pyproject.toml` and `.gitignore` each get one line.
- Folder number 0002 follows 0000 (audit) and 0001 (splits).

## Follow-up

- [ ] Label mismatch: `docs/whats-built.md` calls the splits "E1", but `context/project-overview.md` uses E1 for the complete input baseline. Pick one mapping of E numbers to folders and fix the other file.
- [ ] `/scope` reconcile: rows 0 (inventory) and 3 (subject split) look done or partly done per `docs/whats-built.md`; row 4 (label and channel mapping tests) is partly delivered by this spec.
- [ ] `/sync` after build: `experiments/AGENTS.md` (TOML config, `common/` files now exist, commands), AGENTS.md Commands section.
- [ ] Feature 5 decides: training on both timepoints, augmentation, caching, iteration budget, and checkpoint selection on the full val fold.
- [ ] Decide whether generated tables in `writing/tables/` are committed once real thesis results exist (git ignored for now).
