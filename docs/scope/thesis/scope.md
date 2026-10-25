# Scope: Segmenting Gliomas When Sequences Are Missing: A Controlled Comparison of Training Strategies for Incomplete Multimodal MRI

Train a compact 3D U-Net on BraTS 2023 GLI, measure the drop under all missing-modality patterns, compare simple robustness strategies, and optionally study label efficiency.

**Build approach:** Tracer Bullet.
**Workflow:** Beta (after `/develop`: `/check verify`, then `/test`). Features that turn into thesis claims carry `· GA` and also get a fresh model `/check review` and `/document`.

_These are recommendations to keep your build orderly, not requirements. Skip anything that does not fit. You decide when a feature is `done`._

## At a glance

| #  | Feature                                              | Phase      | Status  |
| -- | ---------------------------------------------------- | ---------- | ------- |
| A  | Repo, environment, AI setup                          | Foundation | done    |
| 0  | Dataset inventory and visualisation (E0)             | Slice 1    | in-progress |
| 1  | Literature map and verified citations                | Slice 1    | planned |
| 2  | Tracer bullet: tiny end to end                       | Slice 1    | done    |
| 13 | Tracer review fixes                                  | Slice 2    | in-progress |
| 3  | Subject-level split and leakage test                 | Slice 2    | in-progress |
| 4  | Preprocessing, label and channel mapping tests       | Slice 2    | in-progress |
| 5  | Compact 3D U-Net baseline, GPU memory test (E1)      | Slice 3    | planned |
| 6  | Missing-modality evaluation, 15 patterns (E2)        | Slice 3    | planned |
| 7  | Modality dropout (E3) and presence indicator (E4)    | Slice 4    | planned |
| 8  | Reduced-label study (E5), optional                   | Slice 5    | planned |
| 9  | Self-supervised pilot (E6), optional                 | Slice 5    | planned |
| 10 | Cross-dataset evaluation (E7), only if independent   | Slice 5    | planned |
| 11 | Error analysis and statistics                        | Slice 6    | planned |
| 12 | Thesis writing and reproducibility package           | Slice 6    | planned |

Priority if time is short: 0 to 7 and 11, 12. Features 8 to 10 are extensions.

## Foundation

### A. Repo, environment, AI setup (done)
Rules, skills, docs layout, pre-commit hook, uv environment (ADR 0001).

## Slice 1: Data, literature, tracer

### 0. Dataset inventory and visualisation (E0) (in-progress)
Know exactly what is in BraTS 2023 GLI (and MSD Task01) before any model sees it, and use it only under its terms.
Done when: the audit runs on every case, `data/DATA.md` records counts, hashes, licence and the odd cases, and the MSD overlap with BraTS is known.
Code in `experiments/0000-data-audit/`.

- [x] Audit run on all 1470 BraTS cases, findings in `data/DATA.md`
- [ ] Human fills sha256, download date, licence and ethics status in `data/DATA.md`
- [ ] Look at the 4 FLAIR cases with a low brain mask Dice and decide keep or flag
- [ ] Audit MSD Task01 and check its case overlap with BraTS: /develop dataset inventory

### 1. Literature map and verified citations (planned)
Map the missing modality literature and confirm the gap the thesis fills. Every citation verified against its source.
Done when: each `refs.bib` entry has a note in `docs/literature/` and is marked verified in `docs/literature/index.md`, and the gap is stated with sources.

- [ ] Build it: /develop literature map

### 2. Tracer bullet: tiny end to end (done)

One thin run from raw BraTS files to a LaTeX table, through data, model, evaluation and writing. Spec: [0001](../../specs/thesis/0001-tracer-bullet/index.md).
Code in `experiments/common/`, `experiments/0002-tracer-bullet/`, `writing/tracer/`.
Done when: `check_tracer.py` passes on the human's full run and `tracer.pdf` builds from the generated table.

- [x] Design it (spec)
- [x] Build it: /develop tracer bullet
  - [x] Shared code in `experiments/common/` (seed, run log, data mapping, preprocessing, missing patterns, metrics) with fast tests (AC-1, AC-2, AC-3, AC-4, AC-8, AC-9, AC-10)
  - [x] Thin thread: `train.py`, `evaluate.py`, `make_table.py`, `writing/tracer/tracer.tex`, one `--quick` run end to end (AC-7, AC-12)
  - [x] Thicken: failure handling, `check_tracer.py`, failure tests (AC-11, AC-13, AC-14)
  - [x] Human full run, numbers recorded in docs (AC-5, AC-6, AC-14)
- [x] Verify it: /check verify tracer bullet
- [x] Test it: /test tracer bullet

## Slice 2: Clean foundations for E1

### 13. Tracer review fixes (in-progress)
Fix what the fresh model review found before every later experiment builds on `experiments/common/`. Findings: `docs/reviews/2026-10-25-feat-tracer-bullet.md`. Spec: [0002](../../specs/thesis/0002-tracer-review-fixes/index.md). Also closes feature 3.
Done when: the audit and split scripts run in the locked environment, `uv run ruff check .` is clean, `evaluate.py` and `train.py` agree on which checkpoint counts, unknown label values raise an error, and one E number to folder mapping is used in every doc.
Code in `experiments/common/`, `experiments/0000-data-audit/`, `experiments/0001-splits/`, `experiments/0002-tracer-bullet/`, `tests/test_splits.py`.

- [x] Design it (spec): /architect tracer review fixes
- [x] Build it: /develop tracer review fixes
  - [x] Locked env and splits: pandas, import guards, `--compare`, `tests/test_splits.py`, human split check (AC-1, AC-2, AC-10, AC-11)
  - [x] Lint clean: `EXE002` ignore, fix the rest (AC-3)
  - [x] Strict shared code: label check, BraTS ID regex, model and docstring cleanups (AC-6, AC-7, AC-14)
  - [x] Tracer pipeline: `last.pt` only, legacy key in `check_tracer`, config hash check, `failed` status, run id suffix, split empty columns (AC-4, AC-4b, AC-5, AC-8, AC-9, AC-12)
  - [x] Docs: E number table, spec 0001 amendment, whats-built and progress tracker (AC-13, AC-15)
- [ ] Verify it: /check verify tracer review fixes
- [ ] Test it: /test tracer review fixes

### 3. Subject-level split and leakage test (in-progress)
The fixed subject level split already exists (seed 42, 791 / 114 / 228 subjects). What is left is a real pytest check, not only a script.
Done when: a fast pytest test proves no subject is in two folds and every case appears once, without needing the gitignored `results/` folder.
Code in `experiments/0001-splits/`, `data/splits/splits.csv`.

- [x] Split written and `check_splits.py` passes
- [x] Leakage test in `tests/`: built under feature 13 (spec [0002](../../specs/thesis/0002-tracer-review-fixes/index.md), AC-11)

### 4. Preprocessing, label and channel mapping tests (in-progress)
BraTS 2023 GLI mapping, channel order and normalisation are tested (spec 0001). MSD Task01 has its own mapping and needs its own tests.
Done when: both datasets have an explicit label and channel mapping, each checked by a test on a real case.
Code in `experiments/common/data.py`.

- [x] BraTS 2023 GLI label, channel and normalisation tests (spec 0001)
- [ ] MSD Task01 mapping and tests: /develop label and channel mapping

## Slice 3: Baseline and the missing modality drop

### 5. Compact 3D U-Net baseline, GPU memory test (E1) · needs a decision (planned)
The complete input baseline every strategy is compared against, trained on the full train fold within 4 GB.
Done when: the baseline trains on the train fold, picks its checkpoint on val only, fits in 4 GB with the peak logged, and its check passes. Open choices from spec 0001: both timepoints or one, augmentation, caching, iteration budget, checkpoint selection.

- [ ] Design it (spec): /architect compact 3D U-Net baseline

### 6. Missing-modality evaluation, 15 patterns (E2) · needs a decision · GA (planned)
Measure how the baseline degrades under every one of the 15 non empty modality subsets (RQ1).
Done when: every pattern is scored per region (Dice, HD95) on the same subjects, the protocol is fixed before any test fold number is seen, and the table comes from a script.

- [ ] Design it (spec): /architect missing modality evaluation

## Slice 4: Strategies

### 7. Modality dropout (E3) and presence indicator (E4) · needs a decision · GA (planned)
Compare simple robustness strategies against zero fill, on the same subjects, splits, metrics and seeds (RQ2).
Done when: each strategy is trained and evaluated on all 15 patterns, and any claimed benefit rests on a paired comparison.

- [ ] Design it (spec): /architect modality dropout and presence indicator

## Slice 5: Extensions (optional)

### 8. Reduced-label study (E5), optional · needs a decision (planned)
Train with fewer labelled subjects and see how fast performance falls (RQ3).
Done when: fixed label fractions drawn from the train fold only, each trained and scored on the same evaluation set.

- [ ] Design it (spec): /architect reduced label study

### 9. Self-supervised pilot (E6), optional · needs a decision (planned)
A small pilot: does self supervised pretraining help at low label fractions (RQ3)?
Done when: a pretrained and a scratch model are compared at the same label fractions within the 4 GB budget.

- [ ] Design it (spec): /architect self supervised pilot

### 10. Cross-dataset evaluation (E7), only if independent · needs a decision (planned)
Test transfer to a second dataset, only if its independence from the training data is shown in `data/DATA.md` (RQ4).
Done when: independence is documented, or the feature is dropped with the reason recorded.

- [ ] Design it (spec): /architect cross dataset evaluation

## Slice 6: Analysis and thesis

### 11. Error analysis and statistics · needs a decision · GA (planned)
Paired statistics and error analysis behind every comparison in the thesis.
Done when: each claimed difference has a paired test with its method stated, and failure cases are shown per pattern and region.

- [ ] Design it (spec): /architect error analysis and statistics

### 12. Thesis writing and reproducibility package · needs a decision (planned)
The thesis chapters, each mapped to RQs, plus a package that rebuilds every number from scripts. Open: university template, and whether `writing/tables/` is committed once real results exist.
Done when: the thesis builds from generated tables and figures, and a clean checkout reproduces them.

- [ ] Design it (spec): /architect thesis writing and reproducibility
