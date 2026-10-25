# Scope: Segmenting Gliomas When Sequences Are Missing: A Controlled Comparison of Training Strategies for Incomplete Multimodal MRI

Train a compact 3D U-Net on BraTS 2023 GLI, measure the drop under all missing-modality patterns, compare simple robustness strategies, and optionally study label efficiency.

**Build approach:** Tracer Bullet.
**Status:** coarse plan. Run `/scope` to refine before building.

## At a glance

| #  | Feature                                              | Phase      | Status  |
| -- | ---------------------------------------------------- | ---------- | ------- |
| A  | Repo, environment, AI setup                          | Foundation | done    |
| 0  | Dataset inventory and visualisation (E0)             | Slice 1    | planned |
| 1  | Literature map and verified citations                | Slice 1    | planned |
| 2  | Tracer bullet: tiny end to end                       | Slice 1    | done    |
| 3  | Subject-level split and leakage test                 | Slice 2    | planned |
| 4  | Preprocessing, label and channel mapping tests       | Slice 2    | planned |
| 5  | Compact 3D U-Net baseline, GPU memory test (E1)      | Slice 3    | planned |
| 6  | Missing-modality evaluation, 15 patterns (E2)        | Slice 3    | planned |
| 7  | Modality dropout (E3) and presence indicator (E4)    | Slice 4    | planned |
| 8  | Reduced-label study (E5), optional                   | Slice 5    | planned |
| 9  | Self-supervised pilot (E6), optional                 | Slice 5    | planned |
| 10 | Cross-dataset evaluation (E7), only if independent   | Slice 5    | planned |
| 11 | Error analysis and statistics                        | Slice 6    | planned |
| 12 | Thesis writing and reproducibility package           | Slice 6    | planned |

Priority if time is short: 0 to 7 and 11, 12. Features 8 to 10 are extensions.

## Features

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
