# 0003. Complete input baseline (E1): a wider 3D U-Net trained on the full train fold, chosen on val

**Date**: 2026-10-27
**Status**: In Progress

## Summary

E1 trains the reference model that every missing modality strategy is compared against. It trains a wider version of the tracer's 3D U-Net on all 863 usable train fold cases for 40,000 steps, checks it on the 114 val subjects every 4,000 steps, and keeps the checkpoint with the best mean Dice over the three tumour regions. The test fold is never read, so the E2 protocol can be fixed before anyone sees a test number. Preprocessing runs once into a cache on disk (float16, half precision numbers), and the training loop lives in `experiments/common/` so E3 and E4 reuse it unchanged.

## Requirements

**User stories**:
- As the researcher, I want one complete input baseline trained on the full train fold so that E2 can measure how it degrades under each missing modality pattern (RQ1).
- As the researcher, I want E3 and E4 to share the exact same loop, data, budget and selection rule so that any strategy comparison is fair and paired (RQ2).
- As the researcher, I want a long laptop run to survive a crash and say exactly what happened, so that hours of GPU time are not lost and nothing is hidden.

**Acceptance criteria**:
- **AC-1**: `common.data.fold_cases(splits_csv, fold, timepoints, exclude_flair_flag)` returns the sorted case IDs of one fold. With the committed `data/splits/splits.csv` and the E1 config it returns 863 train cases (timepoints 000 and 001, the 4 flagged FLAIR cases removed) and 114 val cases (timepoint 000 only). It never returns a `test` row unless asked for fold `test`, and two calls give the same list.
- **AC-2**: `preprocess.py` writes one uncompressed `<case>.npz` per case to the cache dir, holding `image` (4×D×H×W float16, `MODALITIES` order, cropped and normalised exactly as spec 0001 AC-3), `label` (D×H×W uint8, raw values 0 to 3), `spacing` (3 float32, mm) and `crop_box` (3×2 int). It also writes `manifest.csv` and `cache.json` (the columns and keys in *Feature design*). For a case, the cached image matches the float32 preprocessing within `rtol=1e-3, atol=1e-3`, and the cached label equals the cropped raw label exactly. It caches every case of folds `train,val` by default (996 cases, with no timepoint or flag filter), and `test` only when asked with `--folds test`. Existing files with a matching hash are skipped, so a rerun only adds missing cases. Before writing, it checks free disk space against 35 MB per case still to write (estimated) and exits 2 if there is not enough. `manifest.csv` and `cache.json` are rewritten whole through a temp file and rename, with `n_cases` and `folds` recomputed from the manifest. `preprocess.py --verify` rechecks the sha256 of every manifest file, recomputes the first 3 sorted train cases in float32 against the same tolerance, and exits 1 on any mismatch.
- **AC-3**: `train.py`, `memory_test.py` and `evaluate.py` stop with exit 2 before any GPU work if `cache.json` is missing, its `preprocess_hash` differs from the hash of the config, or any needed case file is missing. `preprocess.py` stops with exit 2 if the cache dir already holds a `cache.json` with a different hash. Nothing is ever deleted automatically.
- **AC-4**: E1 reads no test fold case. Every case that `train.py`, `memory_test.py` and `evaluate.py` load is asserted to be in fold `train` or `val` before loading. The E1 cache manifest has no `test` row after the E1 steps, and `run.json` lists every train and val case used.
- **AC-5**: Training augmentation, in this order after the random patch crop: `RandFlipd` on each of the 3 spatial axes (p 0.5 each, the same flip for `image`, `label`, `wt`), then `RandBrainIntensityd`, which per channel c multiplies by `1 + s` and adds `o`, with `s` and `o` drawn uniformly in [−0.1, 0.1] per channel (p 1.0), only on voxels where `image[c] != 0` before the step (a per channel mask, the same one `normalise()` builds per modality; padding from `SpatialPadd` is 0 and so counts as background). Background voxels stay exactly 0. Brain voxels that happen to be exactly 0 are left untouched, which is negligible. `RandBrainIntensityd` subclasses MONAI `RandomizableTransform` and `MapTransform` and draws only from `self.R`, never from global NumPy or torch state, so MONAI's `DataLoader` `worker_init_fn` seeds it per worker. Val and evaluation get no augmentation.
- **AC-6**: The model is MONAI `UNet` (3D, in 4, out 3) with channels (32, 64, 128, 256, 320), strides (2, 2, 2, 2), `num_res_units=2`, trained on 128^3 patches at batch 1. `memory_test.py` runs, in one process with the full config and the same `common/training.py` code paths, 200 training steps on real cached train cases, then a val phase on the 2 largest val cases (by cropped voxel count from `manifest.csv`), then 20 more training steps. It passes when peak reserved VRAM over the whole process is below 3,584 MiB, and writes `memory_test.json` (peaks, mean s/iter over steps 51 to 200, mean data loading wait per step, the val cases and their mean seconds, and the projected hours for the full run).
- **AC-7**: Training runs `iterations = 40000` with AdamW (lr 1e-3, weight decay 1e-5), fp16 autocast with `GradScaler`, and Dice plus BCE loss as in spec 0001. Iteration i (counted from 1) uses lr `lr * i / warmup` for i ≤ `warmup`, and otherwise `lr * 0.5 * (1 + cos(pi * (i - warmup) / (iterations - warmup)))`, so the last step runs at lr 0 (built with `LambdaLR`; the logged `lr` is the one applied at step i). `GradScaler` and autocast are enabled only when `amp` is true and the device is CUDA, so CPU runs use plain float32. A config where `iterations` is not a multiple of `val_every` is rejected at load (exit 2).
- **AC-8**: Every `val_every = 4000` iterations (10 val points, the last at 40,000), `validate(model, val_cases, patterns)` predicts all 114 val cases with `common.model.predict()` and the `[infer]` settings for each pattern ID in `patterns` (applied with `apply_pattern()`; E1 passes `[1]`, complete input), and `region_metrics()` Dice is computed per case, pattern and region. `torch.cuda.empty_cache()` runs before and after each val phase. One row per (iteration, pattern_id, case, region) goes to `val_log.csv`. The E1 selection score is the mean over WT, TC and ET of the mean Dice over the val cases at pattern 1. `best.pt` is written when the score is strictly higher than every earlier one (a tie keeps the earlier one). `run.json` holds `best_iteration`, `best_score` and the full `val_history`. At each val point the writes happen in this order, each through a temp file and rename: `val_log.csv` rows, then `best.pt` (if improved), then `last.pt`, then `run.json`.
- **AC-9**: Learning signal sanity floor, set before any E1 number is seen: mean `loss` over the last 50 iterations is below the mean over the first 50, and `best_score` is above 0.60. If it fails, the run is investigated, not retuned until it passes.
- **AC-10**: Every run writes `run.json` with exactly the keys in the *run.json keys* table, copies the effective config to the run folder as `config.toml`, and prints the run dir, config hash, seed and git commit at start. In `train_log.csv`, `seconds` is the wall time of that one step (data wait included). In `run.json`, `wall_time_s` sums all segments including redone iterations, and the peaks (training and val together), wall time and `scaler_skipped_steps` are written at every val point and at exit. The seed comes from `seed` in the config, and `--seed N` overrides it before hashing, so each seed gets its own config hash and run folder.
- **AC-11**: At each val point, `last.pt` is overwritten with the full training state (model, optimiser, scheduler, scaler, iteration, best score and best iteration so far, `scaler_skipped_steps`, the hook generator state, and the torch, CUDA, NumPy and Python RNG states). `train.py --resume <run_dir>` restores that state exactly, rewrites `train_log.csv` and `val_log.csv` without the rows after the restored iteration, and drops the `val_history` entries after it. If `best.pt` holds an iteration after the restored one, it is deleted, and the best score and iteration from `last.pt` apply. The data loader and its iterator are built fresh, with the generator seeded from `seed * 100003 + iteration`, and training continues from the next iteration. Resume is refused with exit 2 if status is `ok`, if status is `running` and the process in `train.pid` is alive, if the run's `config.toml` hash differs from `run.json` `config_hash`, or if the cache hash differs from `run.json` `cache_hash`. A git commit different from the original prints a warning and is recorded. Each resume adds one entry to `run.json` `resumes`.
- **AC-12**: A non finite loss (checked every step before backward) or a CUDA out of memory error sets status `failed` with the reason and the iteration, and exits 1. Ctrl+C sets `interrupted` and exits 130. In each case the last `last.pt` stays usable for `--resume`. A non finite loss is never skipped. Steps that GradScaler skips for inf gradients (detected when `scaler.get_scale()` is lower after `scaler.update()` than before) are counted in `scaler_skipped_steps`, which carries across resumes through `last.pt`, and are not treated as failures.
- **AC-13**: `evaluate.py --run <dir>` scores `best.pt` on the run's `val_cases` at pattern 1 (complete input) only and writes `metrics_val.csv` with exactly n_val × 3 rows (342 for the full run), with Dice, HD95 in mm, sensitivity and precision under the spec 0001 AC-8 empty region rule. No Dice or HD95 value is NaN. It refuses (exit 1) a run whose status is not `ok` or that has no `best.pt`.
- **AC-14**: `make_table.py --run <dir>` writes `writing/tables/0003-baseline-val.tex`: one row per region WT, TC, ET with Dice mean and median (3 decimals), HD95 mean and median (1 decimal), and two counts as in spec 0002 AC-9: `Both empty` (scored Dice 1, HD95 0) and `One empty` (scored Dice 0, HD95 373.13). The HD95 mean and median include those rule values, as in the tracer. The caption reads `Complete input baseline, validation fold (development), n = <n_val>, seed <seed>, checkpoint at iteration <best_iteration>. Not a test result.`
- **AC-15**: Determinism: two `--quick` runs with the same seed on CPU give `loss` and `lr` columns in `train_log.csv` that match within 1e-6 at every iteration, and so do two such runs with `num_workers = 2`. A `--seed 43` quick run records seed 43 and a different config hash.
- **AC-16**: `check_baseline.py <run_dir>` prints `PASS` or `FAIL: <reason>` and exits 1 on any failure. It checks status `ok`, `iterations == 40000`, `peak_vram_reserved_mib < 3584`, 10 val points in `val_history`, `best_iteration` among the val points, AC-9, `metrics_val.csv` row count and no NaN, 863 train and 114 val cases with none from fold `test`, and the cache hash. A quick run always gets `FAIL: quick run` after the structure checks.
- **AC-17**: The training loop, val, checkpointing, resume and failure handling live in `experiments/common/training.py` with a `batch_hook(batch, rng)` called after augmentation and before the forward pass. `batch` is the collated dict already on the device (keys `case`, `image`, `label`, `wt`). `rng` is a dedicated `numpy.random.Generator` seeded with `seed + 1`, separate from the loader and transform streams, so a hook that draws from it never changes which patches and flips a seed gives. E1 passes the identity hook, which draws nothing. `experiments/0002-tracer-bullet/` is not changed, and its tests still pass.
- **AC-18**: `experiments/AGENTS.md` maps E1 to `0003-baseline`. `docs/whats-built.md` and `context/progress-tracker.md` are updated in the same change.

## Decision

**Chosen option**: Option 1: shared training loop in `common/`, float16 disk cache built by its own script, wider U-Net at 128^3.

E1 is a thin experiment folder, `experiments/0003-baseline/`, on top of a reusable loop and cache in `experiments/common/`, trained once per seed on the full train fold and selected on val only.

**Implementation skills**: `pytorch-patterns` (`affaan-m/ecc`, `.claude/skills/pytorch-patterns/`) · `python-testing-patterns` (`wshobson/agents`, `.claude/skills/python-testing-patterns/`)

**Settled choices** (each was asked or recommended in the design conversation; the runner up is in [rationale.md](rationale.md)):

| Area | Choice |
|---|---|
| What E1 scores | val only, complete input. Test fold untouched (E2 owns it) |
| Train cases | all train fold cases, both timepoints, `exclude_flair_flag = true` (863) |
| Val cases | val fold, timepoint 000 (114) |
| Seeds | `seed = 42` in config, `--seed` override; run 42 now, 43 and 44 later if time allows |
| Model | MONAI `UNet`, channels (32, 64, 128, 256, 320), strides (2, 2, 2, 2), `num_res_units=2` |
| Patch | 128^3, batch 1, `RandCropByPosNegLabeld` on `wt`, pos 1, neg 1 (as spec 0001) |
| Loss | `DiceLoss(sigmoid=True)` + `BCEWithLogitsLoss`, equal weight, float32 (as spec 0001) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-5, `GradScaler` |
| LR schedule | linear warmup over 500 iterations from 0, then cosine to 0 at 40,000 (`LambdaLR`, stepped once per iteration) |
| Budget | 40,000 iterations (about 46 passes over 863 cases) |
| Augmentation | 3 axis flips, then brain only intensity scale and shift (AC-5) |
| Cache | `preprocess.py` writes uncompressed float16 `.npz` to `data/processed/0003-baseline/` |
| Loading | `monai.data.Dataset` over the cache, `monai.data.DataLoader`, 4 workers, `shuffle=True`, seeded generator, endless cycling iterator |
| Val | all 114 every 4,000 iterations, complete input, Dice only (HD95 only in `evaluate.py`) |
| Selection | `best.pt` by mean of WT, TC, ET mean val Dice, a tie keeps the earlier one |
| Postprocessing | none, sigmoid > 0.5 |
| Inference | spec 0001 settings, roi raised to 128^3 to match the patch |
| Resume | from `last.pt` at the last val point, logged, guarded (AC-11) |
| Failures | fail loud, status `failed`, resumable (AC-12) |
| Memory test | 200 steps + 1 val case, pass below 3,584 MiB |
| Determinism | spec 0001 settings unchanged. Claim: seeded and logged, not bitwise identical on GPU, and a resumed run is not identical to an unbroken one |

## Rationale

See [rationale.md](rationale.md).

## Feature design

**Data model sketch** (confirmed in the design conversation; all files are git ignored):

| Artifact | Key | Holds | Relationship |
|---|---|---|---|
| `data/processed/0003-baseline/<case>.npz` | case | `image` float16 4×D×H×W, `label` uint8 D×H×W raw, `spacing` float32 (3,), `crop_box` int (3, 2) | 1:1 with a `splits.csv` case, 1:1 with a manifest row |
| `data/processed/0003-baseline/manifest.csv` | case (unique) | `case, subject, fold, timepoint, shape_d, shape_h, shape_w, sha256, preprocess_hash` | N:1 to `cache.json` |
| `data/processed/0003-baseline/cache.json` | one per cache dir | `preprocess_hash`, `preprocess_version`, `preprocess` (the hashed settings), `git_commit`, `folds`, `n_cases`, `splits_sha256`, `created_utc`, `updated_utc` | read by every E1 script (AC-3) |
| `results/0003-baseline/<run_id>/run.json` | run_id | the keys below | 1 run : 1 seed : 1 cache |
| `.../config.toml` | run_id | effective config after `--quick` and `--seed` | 1:1 run |
| `.../train_log.csv` | (run, iteration) | `iteration, loss, dice_loss, bce_loss, lr, seconds`; on val rows also `val_wt, val_tc, val_et, val_score` | N:1 run |
| `.../val_log.csv` | (run, iteration, pattern_id, case, region) | `iteration, pattern_id, case, region, dice, gt_empty, pred_empty` | N:1 run, val cases only |
| `.../checkpoints/best.pt` | run | `model` state dict, `iteration`, `config_hash`, `score` | 1:1 run |
| `.../checkpoints/last.pt` | run | `best.pt` fields plus `optimizer`, `scheduler`, `scaler`, `best_score`, `best_iteration`, `scaler_skipped_steps`, `hook_rng`, `rng` (torch, cuda, numpy, python) | 1:1 run, overwritten |
| `.../train.pid` | run | process id while training runs, deleted at exit | 1:1 run |
| `.../metrics_val.csv` | (run, case, region) | `run_id, case, subject, region, dice, hd95_mm, sensitivity, precision, gt_empty, pred_empty` | N:1 run |
| `results/0003-baseline/memory_test.json` | latest | `peak_vram_allocated_mib`, `peak_vram_reserved_mib`, `seconds_per_iteration`, `data_wait_s_per_step`, `val_cases`, `val_seconds`, `projected_hours`, `config_hash`, `pass` | 1 per config |
| `writing/tables/0003-baseline-val.tex` | run | the AC-14 table | 1:1 run |

`run_id = YYYYMMDD-HHMMSS-<first 8 of config hash>` in UTC, with the collision suffix from spec 0002.

**Hashes**: `config_hash` is `common.runlog.config_hash()` of the whole effective config. `preprocess_hash` is the same function over `{"preprocess": cfg["preprocess"], "version": PREPROCESS_VERSION}`, where `PREPROCESS_VERSION = 1` is a constant in `common/cache.py`, raised by hand whenever the preprocessing code changes its output. Training settings never invalidate the cache.

**run.json keys** (all present on every run, `null` where not yet known):

| Key | Type | Meaning |
|---|---|---|
| `run_id`, `status`, `reason`, `quick`, `config_hash`, `git_commit`, `git_dirty`, `seed`, `data_dir`, `device`, `gpu`, `versions`, `determinism`, `nondeterminism_warnings`, `scaler_skipped_steps`, `loss_window`, `loss_start_mean`, `loss_end_mean`, `train_cases`, `val_cases` | as spec 0001 | unchanged meaning |
| `peak_vram_allocated_mib`, `peak_vram_reserved_mib` | float or null | maximum over the original segment and every resume |
| `wall_time_s` | float | training wall time summed over segments |
| `iterations` | int | last completed iteration |
| `n_params` | int | trainable parameters |
| `cache_dir` | str | resolved cache path |
| `cache_hash` | str | `preprocess_hash` the run was started with |
| `seconds_per_iteration` | float or null | mean training s/iter excluding val, over all segments |
| `val_history` | list of objects | per val point: `iteration`, `wt`, `tc`, `et`, `score`, `seconds` |
| `best_iteration` | int or null | iteration of `best.pt` |
| `best_score` | float or null | its selection score |
| `failed_iteration` | int or null | iteration where a failure happened |
| `resumes` | list of objects | per resume: `from_iteration`, `utc`, `git_commit`, `git_dirty`, `previous_status` |

**State transitions** (`run.json` status): `running` → `ok` | `failed` | `interrupted`. `failed` | `interrupted` | `running` (dead process) → `running` through `--resume`. `ok` is final. A folder with status `running` and no live `train.pid` process is a crashed run. `check_baseline.py` fails it, and `--resume` may continue it.

**Config** (`experiments/0003-baseline/config.toml`):

```toml
seed = 42
[data]
splits_csv = "data/splits/splits.csv"
cache_dir = "data/processed/0003-baseline"
train_timepoints = [0, 1]
val_timepoints = [0]
exclude_flair_flag = true
[preprocess]
clip_low_pct = 0.5
clip_high_pct = 99.5
[train]
iterations = 40000
val_every = 4000
warmup = 500
patch = [128, 128, 128]
pos = 1
neg = 1
batch_size = 1
num_workers = 4
lr = 1e-3
weight_decay = 1e-5
amp = true
flip_p = 0.5
intensity_scale = 0.1
intensity_shift = 0.1
[model]
channels = [32, 64, 128, 256, 320]
strides = [2, 2, 2, 2]
num_res_units = 2
[infer]
roi = [128, 128, 128]
sw_batch_size = 1
overlap = 0.5
threshold = 0.5
[memory_test]
steps = 200
limit_mib = 3584
```

`--quick` overrides (applied before hashing): first 2 train and first 1 val case of the sorted lists, `iterations = 10`, `val_every = 5`, `warmup = 2`, `patch = roi = [64, 64, 64]`, `num_workers = 0`.

**API surface** (command line, run with `uv run` from the repo root; every script keeps work in `main()`):

| Command | Key inputs | Key outputs | Key errors |
|---|---|---|---|
| `python experiments/0003-baseline/preprocess.py` | `--config`, `--data-dir` (default env `BRATS_TRAIN_DIR`), `--splits`, `--folds` (default `train,val`), `--quick` (only the quick cases: the first 2 train and first 1 val of the filtered sorted `fold_cases()` lists), `--verify` (check only, writes nothing) | `.npz` files, `manifest.csv`, `cache.json`; prints cases written and skipped, and GB on disk | exit 2: no data dir, missing case file, cache dir holds a different hash, not enough free disk; exit 1: `--verify` mismatch |
| `python experiments/0003-baseline/memory_test.py` | `--config`, `--device` | `results/0003-baseline/memory_test.json`; prints `PASS` or `FAIL`, the peak, s/iter, projected hours | exit 2: AC-3 cache guard; exit 1: peak at or above `limit_mib`, or OOM |
| `python experiments/0003-baseline/train.py` | `--config`, `--seed`, `--out` (default `results/0003-baseline`), `--quick`, `--device` (`auto` or `cpu`), `--resume <run_dir>` (excludes the others except `--device`) | run folder (data model above) | exit 2: cache guard, bad config, resume refused; exit 1: non finite loss or OOM; exit 130: Ctrl+C |
| `python experiments/0003-baseline/evaluate.py` | `--run` (req), `--device` | `metrics_val.csv`; prints mean Dice per region | exit 1: status not `ok` or no `best.pt`; exit 2: cache guard |
| `python experiments/0003-baseline/make_table.py` | `--run` (req) | `writing/tables/0003-baseline-val.tex` | exit 1: `metrics_val.csv` missing or wrong row count |
| `python experiments/0003-baseline/check_baseline.py` | run dir (positional) | `PASS` or `FAIL: <reason>` | exit 1 on any failed check |

Test hook (as in the tracer): `BASELINE_INJECT_INF=<iteration>` makes `train.py` feed `inf` at that iteration, so the AC-12 test can force a non finite loss. Unset, it does nothing.

**Shared code** (new or changed in `experiments/common/`):

| File | Adds |
|---|---|
| `data.py` | `fold_cases()`; `CachedCase` transform (reads a `.npz` and returns `case` str, `image` float32 (4, D, H, W), `label` float32 (3, D, H, W) from `label_to_regions()`, `wt` uint8 (1, D, H, W) from raw label > 0, and `pixdim` float32 (3,) from `spacing`, the same key names as the tracer's `Preprocess`); `RandBrainIntensityd`; `cached_train_transforms(cfg)` and `cached_eval_transforms(cfg)`. The existing tracer transforms stay as they are |
| `cache.py` | `PREPROCESS_VERSION`, `preprocess_hash(cfg)`, `write_case()`, `read_manifest()`, `check_cache(cfg, cases)` (the AC-3 guard) |
| `training.py` | `train(cfg, run_dir, train_cases, val_cases, batch_hook=identity, val_patterns=(1,), resume=False)`, the lr lambda, `validate(model, val_cases, patterns)`, `save_state()`, `load_state()`, failure handling and the `run.json` updates |
| `runlog.py` | no change in behaviour; the E1 key list lives in `training.py` |

**Value sourcing**:

| Action | Value produced | Source |
|---|---|---|
| fold_cases | train and val case lists | `splits.csv` columns `case, fold, timepoint, flair_mask_flag`; filters from `[data]` |
| preprocess | crop box, normalised image | spec 0001 `crop_box()` and `normalise()`, settings from `[preprocess]` |
| preprocess | `spacing` | NIfTI header `pixdim` via `load_case()` |
| preprocess | `preprocess_hash` | `config_hash()` of `[preprocess]` + `PREPROCESS_VERSION` |
| preprocess | `splits_sha256` | sha256 of the `--splits` file |
| preprocess | per file `sha256` | sha256 of the written `.npz` |
| CachedCase | WT, TC, ET masks | `label_to_regions()` on the cached raw label |
| train | lr per step | `[train] lr, warmup, iterations` through the warmup and cosine lambda |
| train | loss start and end means | `train_log.csv` `loss`, W = 50 (min(50, iterations // 2) in quick) |
| train | selection score | `val_log.csv` Dice: mean over cases per region, then mean of the 3 regions |
| train | `best_iteration`, `best_score` | the val point with the highest score, a tie keeps the earlier one |
| train | resume loader seed | `seed * 100003 + restored iteration` |
| train | peak VRAM | `torch.cuda.max_memory_reserved()` and `max_memory_allocated()`, reset per segment, maximum kept |
| train | `seconds_per_iteration` | wall time of training steps only (val excluded) / steps |
| memory_test | 2 largest val cases | `manifest.csv` shape product among the `fold_cases()` val list for the config, ties by sorted case ID |
| memory_test | `projected_hours` | `(seconds_per_iteration * iterations + val_seconds * n_val * iterations / val_every) / 3600`, where `val_seconds` is the mean over the 2 largest cases, so the val part is an overestimate |
| train | hook generator | `numpy.random.default_rng(seed + 1)`, state in `last.pt` |
| evaluate | metrics | `region_metrics()` with the cached `spacing` |
| make_table | `best_iteration`, `seed`, n_val | `run.json` |
| check | expected 863 and 114 | `fold_cases()` on the committed `splits.csv` with the run's config, compared with `run.json` |

**Key invariants**:
- No `test` fold case is loaded by any E1 script (asserted per case before loading).
- The cache only serves a config whose `preprocess_hash` matches `cache.json`.
- Image and label of a training sample get the same spatial transforms; intensity steps never make a background voxel nonzero.
- Val and evaluation use timepoint 000 only and the same `[infer]` settings.
- `best.pt` changes only when the score strictly improves.
- After a resume, `train_log.csv` has exactly one row per iteration from 1 to the current iteration, with no gap and no duplicate.

**Security model**: The cache holds derived patient images, so it lives under `data/processed/`, which is git ignored (and `*.npz` and `*.pt` are blocked by `.gitignore` and the pre commit hook). Raw data under the data dir is only read. `metrics_val.csv`, `val_log.csv` and the table hold public BraTS case IDs and scores only. The data use terms in `data/DATA.md` are still to be confirmed, and E1 does not change that status. No secrets are involved.

**Configuration required**:
- `BRATS_TRAIN_DIR`: existing, the BraTS 2023 GLI training folder (read by `preprocess.py` only).
- `CUBLAS_WORKSPACE_CONFIG`: existing, set by `common.seed` as in spec 0001.
- `BASELINE_INJECT_INF`: test hook only (AC-12).

**Edge cases and failures**:
- Disk fills while caching: the `.npz` is written to a temp name and renamed, so a half file never counts. The manifest row is added only after the rename, and a rerun picks up where it stopped.
- A cropped case smaller than 128 on some axis: `SpatialPadd` to the patch with 0, and sliding window pads the same way (as spec 0001).
- Patch centres: `RandCropByPosNegLabeld` keeps `image_threshold=0` from the tracer, so negative centres fall where any channel is above 0. No train case has an empty WT (`data/DATA.md`), so positive centres always exist.
- Slow reads: the repo drive is NTFS, so cache reads may limit speed. `memory_test.json` reports `data_wait_s_per_step`, and `cache_dir` is a config value if the cache has to move to another drive.
- A crash between val point writes: the fixed write order and the resume rule for `best.pt` (AC-8, AC-11) keep `best.pt`, `last.pt` and `run.json` consistent.
- RAM: 4 workers, each holding about one 30 MB case plus prefetch (estimated, not measured). The cache is not held in RAM.
- Disk: `preprocess.py` caches every case of the requested folds, with no timepoint or flag filter, so later experiments can filter differently: 996 cases for `train,val` (867 + 129), about 30 GB (estimated from the tracer's crop sizes at half precision, not measured). 134 GB was free on 2026-10-25. E1 then reads 977 of them (863 + 114).
- Laptop sleep or power loss: status stays `running`. `--resume` continues from the last val point, and at most 4,000 iterations are redone.
- OOM during val on a large case: same path as a training OOM (AC-12), so the memory test checks the largest val case first.
- An empty region in a val case: the spec 0001 AC-8 rule (both empty is Dice 1) is used in the selection score unchanged.

**Critical test scenarios**:
- Happy path: synthetic NIfTI cases in `tmp_path`, `preprocess.py --quick`, then `train.py --quick --device cpu`, `evaluate.py`, `make_table.py` give status `ok`, a 1 × 3 row `metrics_val.csv` and the table, verifies **AC-1**, **AC-2**, **AC-7**, **AC-8**, **AC-10**, **AC-13**, **AC-14**.
- Determinism: two quick runs, same seed, same `loss` column, verifies **AC-15**.
- Failure case: `BASELINE_INJECT_INF=7` gives `failed` and exit 1. Then, with the variable unset, `--resume` restores iteration 5 and finishes `ok` with 10 log rows, verifies **AC-11**, **AC-12**.
- Interrupted writes: a fake run with `best.pt` at iteration 10 and `last.pt` at 5 resumes with `best.pt` deleted and the best score from `last.pt`, verifies **AC-11**.
- Guard: a changed `clip_high_pct` gives exit 2 before training; a resume after editing the run's `config.toml` gives exit 2, verifies **AC-3**, **AC-11**.
- Leakage: a mini `splits.csv` with a `test` row is never cached or loaded, verifies **AC-4**.
- Augmentation: a fixed seed flip keeps image and label voxel aligned, and background stays 0 after the intensity step, verifies **AC-5**.

## Build plan

Ordered by Tracer Bullet: a thin quick thread through cache, loop, val, evaluation and table first, then thicken.

1. Thin thread: `fold_cases()`, `common/cache.py`, `CachedCase`, `preprocess.py`, a minimal `common/training.py` (no augmentation, no resume, constant lr), `train.py`, `evaluate.py`, `make_table.py`, `config.toml`, and the happy path test on synthetic cases, satisfies **AC-1**, **AC-2**, **AC-3**, **AC-4**, **AC-10**, **AC-13**, **AC-14**, **AC-17**
2. Thicken training: warmup and cosine lr, CPU scaler off, `validate()` with `patterns` every `val_every` into `val_log.csv`, selection score and `best.pt`, the fixed write order, the config multiple check, satisfies **AC-7**, **AC-8**, **AC-10**
3. Augmentation: `RandFlipd` on 3 axes, `RandBrainIntensityd`, alignment and background tests, satisfies **AC-5**
4. Robustness: `last.pt` full state, the hook generator, `--resume` with its guards and the `best.pt` rule, `train.pid`, failure statuses, the scaler skip counter, the inject hook and the failure tests, `preprocess.py --verify` and the free space check, satisfies **AC-2**, **AC-11**, **AC-12**, **AC-17**
5. Checks: `memory_test.py`, `check_baseline.py`, the determinism and seed tests, satisfies **AC-6**, **AC-9**, **AC-15**, **AC-16**
6. Docs: E1 row in `experiments/AGENTS.md`, `docs/whats-built.md`, `context/progress-tracker.md`, satisfies **AC-18**
7. Human runs, in order (the AI prints the commands and does not wait): `preprocess.py` (train, val), `preprocess.py --verify`, `memory_test.py`, then if it passes `train.py`, `evaluate.py`, `make_table.py`, `check_baseline.py`, and the human pastes the last two lines of each, satisfies **AC-2**, **AC-6**, **AC-7**, **AC-8**, **AC-9**, **AC-13**, **AC-16**. If the memory test fails or projects more than 10 hours, stop and return to `/architect`; do not shrink the model by hand.

## Consequences

**Positive**:
- E2, E3 and E4 inherit one cache, one loop and one selection rule, so the RQ2 comparison is paired by construction.
- The test fold stays unseen until the E2 protocol is written.
- A crash costs at most one val interval, and every resume is on the record.

**Negative / tradeoffs**:
- The 40,000 iteration budget and the memory fit are estimates until `memory_test.py` runs. The time could be well above the guessed 4 to 6 hours.
- About 30 GB of derived patient images sit on disk under `data/processed/`.
- float16 caching adds a small rounding (relative error about 5e-4) that the tracer did not have. That's within normal numerical noise, but it's a real difference between E1 and the tracer.
- Resume makes a run seeded and logged, not identical to an unbroken run. Two runs of the same seed with different crash histories may differ slightly.
- One seed is only a weak basis for variance. Seed claims wait for seeds 43 and 44.
- Selection on complete input favours complete input. E3 and E4 must decide whether they select the same way, and that is a fairness decision for their spec.

**Neutral**:
- `common/training.py` becomes the most load bearing shared module and needs careful review before E3.
- The tracer keeps its own loop, so there are two loops in the repo until a later cleanup.

## Follow-up

- [ ] E2 spec: run `preprocess.py --folds test`, evaluate `best.pt` on all 15 patterns, fix the protocol before scoring.
- [ ] E3 and E4 spec: decide whether their checkpoint selection stays at complete input (this spec's rule) or averages over patterns through `val_patterns`, and record it before training. No change to `common/` is needed for either choice.
- [ ] Seeds 43 and 44 after E2 if GPU time allows.
- [ ] Feature 0: view the 4 flagged FLAIR cases. If they are fine, set `exclude_flair_flag = false` and record the new config hash.
- [ ] `select_subjects()` in `common/data.py` notes "revisit when sizing E1". E1 uses `fold_cases()` instead, and `select_subjects()` stays for the tracer only.
