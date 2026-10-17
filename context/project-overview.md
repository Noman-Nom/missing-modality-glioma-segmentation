# Project overview

## Problem
Multimodal brain MRI (T1, T1ce, T2, FLAIR) gives complementary information, but in practice some sequences are missing. A model trained only on complete cases can fail when a sequence is absent. Zero-filling a missing channel is a weak fix. We ask how bad the drop is and which practical strategy keeps the model reliable.

## Research questions
- **RQ1** How much does segmentation performance (Dice, HD95 per WT/TC/ET) change when one or more modalities are missing?
- **RQ2** Which practical missing-modality strategy performs best under comparable conditions (zero-fill, modality dropout, presence indicators)?
- **RQ3** Can self-supervised pretraining or reduced-label training help when fewer masks are available? (optional extension)
- **RQ4** Do results transfer to a different dataset, if a compatible and independent one exists? (optional; may be dropped if independence cannot be shown)

Hypotheses (to test, not assumed): performance varies across missing patterns; modality dropout or presence indicators improve robustness; self-supervised pretraining helps at low label fractions; performance drops across datasets.

## Contribution
Empirical study: controlled, subject-level, paired comparison of missing-modality strategies on a 4 GB GPU budget. Publication is a goal, not a guarantee. Novelty is unconfirmed until the literature review.

## Experiments
E0 data inspection, E1 complete-input baseline, E2 missing-modality evaluation (all 15 non-empty subsets), E3 modality dropout, E4 presence indicator (core). E5 reduced labels, E6 self-supervised, E7 cross-dataset (extensions). Full detail in `project-draft.md`.

## Setup answers (Step 0)

| Item | Answer |
|---|---|
| Project title | Segmenting Gliomas When Sequences Are Missing: A Controlled Comparison of Training Strategies for Incomplete Multimodal MRI (earlier working title: Robust and Label-Efficient Brain MRI Segmentation Under Missing Modalities) |
| Type | Master's thesis / research project |
| University, program, supervisor | NUST MISIS, College of IT & Computer Science, Dept. of Computer-aided Design and Engineering Design; Data Science 09.04.01; supervisor Kalitin Denis Vladimirovich |
| Student / roll no. | Qarar Hussain / [to be provided] |
| One-sentence goal | Build a reproducible brain-tumor segmentation system and study how to keep it reliable when MRI modalities are missing |
| Research questions | RQ1 to RQ4 above |
| Main language and stack | Python, PyTorch, MONAI, NiBabel |
| Writing format | LaTeX XeLaTeX + biblatex, Option B (modern two-tone journal, burgundy), British spelling; university template and page limit unknown, ask |
| Hardware limits | Dell Precision 5550, Quadro T2000 Max-Q, 4 GB VRAM, Linux (RAM to confirm) |
| Datasets | MSD Task01_BrainTumour; BraTS 2023 GLI (training, validation, 2017 mapping). Licences: to confirm |
| Sensitive data? | Yes: medical MRI. Never in git |
| Web demo or app planned? | No |
| Deadlines, template, page limit | unknown, ask supervisor (draft assumes 16 weeks) |
