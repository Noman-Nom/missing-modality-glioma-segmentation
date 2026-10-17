# Master's Computer Vision Research Project

## Robust and Label-Efficient Brain MRI Segmentation Under Missing Modalities

**Project type:** Master's thesis / research project\
**Domain:** Medical image analysis, computer vision, deep learning\
**Primary task:** Brain-tumor segmentation from multimodal MRI\
**Working title:** *Robust and Label-Efficient Brain MRI Segmentation
Under Missing Modalities Using Self-Supervised Learning*

> This document is a research plan, not a claim that experiments have
> succeeded. Verify dataset contents, labels, permissions, provenance,
> and all reported metrics before drawing conclusions.

## 1. Executive summary

Brain MRI examinations can contain several sequences, commonly T1,
contrast-enhanced T1 (T1ce/T1Gd), T2, and FLAIR. These sequences provide
complementary information. In practice, one or more sequences may be
unavailable. A model trained only on complete examinations may perform
poorly when a sequence is missing.

We will build and evaluate a segmentation pipeline that predicts
brain-tumor regions from multimodal MRI and systematically tests its
robustness when modalities are removed. We will first establish a
reproducible baseline, compare practical missing-modality strategies,
and then investigate whether self-supervised or label-efficient learning
helps when annotated training data are limited.

The first goal is a rigorous experimental study, not a clinical product
or a large foundation model trained from scratch.

### Main expected contribution

A controlled comparison of (1) complete-modality segmentation, (2) the
same model under missing-modality conditions, (3) at least one method
designed to handle missing inputs, and (4) label-efficient learning if
time and compute allow. A genuine independent-dataset evaluation is
desirable only if data provenance and labels support it.

Publication is a goal, not a guarantee. Novelty must be established
through literature review and experiments.

## 2. Problem statement and motivation

Many segmentation models assume every expected MRI sequence is present
at inference time. Missing scans can occur because of acquisition
protocols, patient factors, time, cost, or data-transfer problems.
Replacing a missing scan with zeros or training only on complete cases
can lead to unreliable predictions.

**Research problem:** How can a brain-tumor segmentation model remain
reliable when one or more MRI modalities are missing, and can learning
from unlabeled MRI reduce dependence on expert segmentation labels?

The project will not diagnose patients, replace radiologists, or claim
clinical safety from benchmark metrics alone.

## 3. Aim, objectives, questions

### Aim

Develop a reproducible brain MRI tumor-segmentation pipeline and
evaluate methods for handling missing modalities and limited annotation.

### Objectives

1.  Understand MRI modalities, labels, file formats, and dataset
    conditions.
2.  Build a tested baseline segmentation model.
3.  Evaluate performance with complete inputs.
4.  Simulate predefined missing-modality conditions.
5.  Compare missing-modality strategies fairly.
6.  Investigate label efficiency with self-supervised or semi-supervised
    learning if feasible.
7.  Evaluate generalization carefully, distinguishing internal testing
    from true external validation.
8.  Document code, experiments, limitations, and reproducibility.

### Research questions

-   **RQ1:** How much does segmentation performance change when one or
    more modalities are missing?
-   **RQ2:** Which practical missing-modality strategy performs best
    under comparable conditions?
-   **RQ3:** Can self-supervised pretraining or reduced-label training
    improve results when fewer segmentation masks are available?
-   **RQ4:** Do results transfer to a different dataset, if a compatible
    and sufficiently independent dataset is available?

### Hypotheses to test

-   Performance will vary across missing-modality patterns.
-   Modality dropout or explicit modality-presence indicators may
    improve robustness.
-   Self-supervised pretraining may help in low-label settings.
-   Performance may decline on different datasets due to acquisition and
    population differences.

These are hypotheses, not assumed results.

## 4. Dataset plan

### Dataset A: Medical Segmentation Decathlon --- Task01_BrainTumour

Official source: https://medicaldecathlon.com/dataaws/

Purpose: - Initial pipeline development and visualization. - Baseline
segmentation. - Controlled missing-modality experiments.

Verify the exact release, case count, modality order, label definitions,
archive integrity, and license/usage terms using the downloaded files
and official documentation.

### Dataset B: BraTS 2023 GLI Challenge

Challenge information:
https://www.synapse.org/Synapse:syn51156910/wiki/622351\
Training-data folder: https://www.synapse.org/Synapse:syn51514105

Files currently being downloaded: -
`ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip` -
`ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip` -
`BraTS2023_2017_GLI_Mapping.xlsx`

Purpose depends on the exact release and its rules. **Do not assume
challenge validation labels are publicly available.** Check the
documentation and extracted contents first.

### Critical overlap warning

Medical Segmentation Decathlon Task01_BrainTumour and BraTS releases may
share source cases or related challenge data. Do not automatically treat
them as independent datasets. Check provenance, subject identifiers
where permitted, and documentation. If independence cannot be
established, describe the experiment as cross-release/cross-dataset
evaluation with a limitation, not definitive external validation.

### Dataset inventory checklist

Record for each dataset: - Official name, version, source URL, download
date, and usage terms. - Archive size and checksum if practical. -
Number of cases and subject IDs. - Available modalities and their
order. - Image shapes, voxel spacing, orientation, and file format. -
Segmentation label values and their meanings. - Whether ground-truth
masks are present. - Missing/corrupted files and possible duplicates. -
Required dataset citations and any overlap with other data.

Do not begin training until a few images and masks have been loaded and
alignment confirmed.

## 5. MRI and labels

### Simple explanation: MRI and modalities

**MRI** stands for **Magnetic Resonance Imaging**. It is a medical scan that lets doctors see inside the human body, including the brain, without surgery.

**MRI modalities** are different types of MRI scans of the same body part. Each type shows different details.

In our brain-tumor project:
- **T1:** Shows brain anatomy.
- **T1ce:** Highlights areas after contrast injection.
- **T2:** Shows fluid and swelling clearly.
- **FLAIR:** Helps highlight abnormalities, such as tumors and surrounding swelling.

Think of these four modalities as four different views of the same brain, each providing different information.

Common modalities: - **T1:** anatomical structure. - **T1ce/T1Gd:**
contrast-enhanced T1, often highlighting enhancing tumor regions. -
**T2:** sensitive to fluid-related changes and edema. - **FLAIR:**
suppresses much of the free-fluid signal and helps show lesions and
edema.

Confirm the actual modalities and their channel order for each release;
do not infer channel order from filenames alone.

BraTS-style annotations often support derived regions such as: - **Whole
Tumor (WT)** - **Tumor Core (TC)** - **Enhancing Tumor (ET)**

Label IDs and mappings can vary by release. Implement and document the
mapping from the dataset specification. Never assume numeric IDs mean
the same thing across datasets.

MRI files are often NIfTI (`.nii` or `.nii.gz`). Use NiBabel or
SimpleITK to inspect volumes and metadata. An RGB image-classification
pipeline is not sufficient for volumetric MRI.

## 6. Methodology

### Workflow

1.  Verify data access and archive integrity.
2.  Inventory subjects, modalities, masks, and metadata.
3.  Visualize each modality with segmentation overlays.
4.  Create subject-level train/validation/test splits.
5.  Preprocess images and masks consistently.
6.  Train a baseline segmentation model.
7.  Evaluate complete-modality performance.
8.  Test predefined missing-modality patterns.
9.  Train/evaluate robustness strategies.
10. Run label-efficiency experiments if feasible.
11. Analyze errors and generalization.
12. Save code, configurations, logs, and results.

### Preprocessing

-   Load MRI and masks with a medical-imaging library.
-   Check orientation, spacing, shapes, and image-mask alignment.
-   Apply consistent spatial transformations to images and masks.
-   Normalize image intensities using a documented method.
-   Crop to foreground or use patches to reduce memory use.
-   Use nearest-neighbor interpolation for discrete masks.
-   Use appropriate continuous interpolation for MRI if resampling is
    needed.
-   Apply augmentation to training cases only; spatial transforms must
    be shared by images and masks.
-   Keep the process reproducible and configurable.

Avoid generic RGB resizing or arbitrary slice-based processing unless it
is a deliberate, documented experiment.

### Data splitting

-   Split by **subject**, never by slice or patch.
-   Keep test subjects untouched until final evaluation.
-   Fix random seeds and save subject IDs for each split.
-   Use validation data for model selection and early stopping.
-   Respect official challenge splits and rules where applicable.
-   Do not repeatedly tune based on test results.

### Baseline model

Start with a compact **3D U-Net** or established MONAI implementation.
The first goal is a correct baseline, not a complex new architecture.

Potential loss: Dice loss plus cross-entropy, with class imbalance
handled explicitly. Tune patch size, batch size, learning rate,
optimizer, augmentation, training duration, and checkpoint selection
using validation data. Record all settings.

### Missing-modality conditions

For four modalities, test: - All modalities present. - Each individual
modality missing. - Selected two-modality combinations. -
Single-modality input, if useful and feasible.

Define every pattern in advance. Clearly document how missing channels
are represented and which modalities are available. Compare methods on
the same subjects and metrics.

### Strategies to compare

1.  **Complete-modality baseline:** trained and evaluated with all
    modalities.
2.  **Zero/channel replacement:** replace a missing channel with a
    documented neutral value; use as a baseline only.
3.  **Modality dropout:** randomly omit modalities during training.
4.  **Presence indicators:** explicitly provide the model with
    information about which modalities are present.
5.  **Modality-specific encoders/fusion:** explore only if simpler
    experiments justify the complexity.

### Label efficiency and self-supervised learning

This is a later phase, after the baseline and missing-modality
experiments work.

-   Train supervised baselines with 100%, 50%, 25%, and 10% of training
    masks.
-   Keep the test set fixed.
-   If resources permit, pretrain an encoder on permitted unlabeled MRI
    with a suitable self-supervised objective.
-   Fine-tune on the same labeled subsets and compare against training
    from scratch.
-   Use fixed seeds or multiple runs where feasible.

Do not claim self-supervised learning helped unless a fair comparison
supports it. Do not train a large foundation model from scratch on a 4
GB GPU.

## 7. Evaluation

### Primary metrics

-   **Dice Similarity Coefficient (DSC):** overlap between prediction
    and reference.
-   **HD95:** 95th-percentile Hausdorff distance for boundary errors.
-   **Sensitivity/recall:** proportion of reference tumor voxels
    recovered.
-   **Precision:** proportion of predicted tumor voxels that are
    correct.

For BraTS-style labels, report Dice and HD95 for WT, TC, and ET where
supported. State the units and how empty regions are handled. Voxel
accuracy alone is not appropriate as the primary metric because
background can dominate.

### Results tables

Report: 1. Complete-modality baseline. 2. Each missing-modality
condition. 3. Each robustness strategy. 4. Each labeled-data fraction.
5. Cross-dataset results only when data compatibility and provenance
justify them.

Include per-region scores, number of subjects, and variability where
possible.

### Error analysis

Inspect false positives, missed tumor regions, small/difficult lesions,
unusual anatomy, and performance changes by missingness pattern.
Visualize predictions alongside MRI and ground truth. Do not show only
favorable examples.

Use paired comparisons on the same subjects. Report confidence intervals
or variation across subjects/runs where feasible. Avoid overstating tiny
numerical differences.

## 8. Compute plan: local 4 GB GPU

Full-volume 3D training may not fit on the user's 4 GB GPU. Begin
with: - Small 3D patches and batch size 1. - Mixed precision if
supported and stable. - Gradient accumulation if needed. - Compact model
widths. - Limited caching/workers if RAM is constrained. - 2D or 2.5D as
a fallback, clearly labelled. - Cloud or university GPU for heavier
experiments.

Run one forward/backward pass first to estimate memory. Save checkpoints
and logs; do not load all volumes into GPU memory.

## 9. Experiment matrix

  -----------------------------------------------------------------------
  ID                      Experiment              Purpose
  ----------------------- ----------------------- -----------------------
  E0                      Inspect data and        Verify data correctness
                          visualize masks         

  E1                      Complete-input baseline Establish reference
                                                  performance

  E2                      Missing-modality        Measure baseline
                          evaluation              fragility

  E3                      Modality-dropout        Test robustness
                          training                improvement

  E4                      Modality-presence       Test explicit
                          indicator               missingness information

  E5                      Reduced-label           Measure annotation
                          supervised training     dependence

  E6                      Self-supervised         Test low-label benefits
                          pretraining, if         
                          feasible                

  E7                      Cross-dataset           Assess generalization
                          evaluation, if valid    
  -----------------------------------------------------------------------

E0--E4 are the core. E5 is recommended if time permits. E6 and E7 depend
on compute, labels, compatibility, and timeline.

## 10. Suggested repository structure

``` text
brain-mri-missing-modality/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── modality_dropout.yaml
│   └── low_label.yaml
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   └── 02_visualize_masks.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   └── unet.py
│   ├── train.py
│   ├── evaluate.py
│   ├── missing_modalities.py
│   └── metrics.py
├── tests/
│   ├── test_data_shapes.py
│   └── test_label_values.py
├── scripts/
├── results/
│   ├── tables/
│   └── figures/
└── docs/
    ├── data_card.md
    └── experiment_log.md
```

Do not upload restricted datasets or raw medical images to a public
GitHub repository. Provide instructions and citations instead.

For every run, save code version, configuration, seed, subject split
IDs, checkpoint, training/validation curves, metrics, per-subject
results, hardware/software versions, and notes.

## 11. Software stack

-   Python and PyTorch.
-   MONAI for medical-image transforms, networks, and metrics.
-   NiBabel or SimpleITK for NIfTI inspection.
-   NumPy and pandas for data handling.
-   Matplotlib for plots.
-   Jupyter for exploration and scripts/configurations for reproducible
    experiments.
-   Git for version control.

Pin working package versions after testing the GPU-enabled PyTorch
setup.

## 12. Suggested 16-week timeline

  -----------------------------------------------------------------------
  Weeks                   Work                    Deliverable
  ----------------------- ----------------------- -----------------------
  1--2                    Literature review,      Literature matrix and
                          dataset                 data card
                          terms/provenance        

  3--4                    Loader, inspection,     Tested data pipeline
                          visualization           

  5--7                    Baseline model          Checkpoint and baseline
                                                  metrics

  8--9                    Missing-modality        Missingness table
                          evaluation              

  10--11                  Modality dropout and    Robustness comparison
                          presence indicator      

  12--13                  Low-label study;        Label-efficiency
                          self-supervised pilot   results
                          if feasible             

  14                      Cross-dataset           Generalization analysis
                          evaluation if justified 

  15                      Error analysis and      Final experiment
                          reproducibility checks  package

  16                      Thesis/report and       Written report and
                          presentation            slides
  -----------------------------------------------------------------------

If time is limited, prioritize data integrity, baseline, and
missing-modality experiments over self-supervised pretraining.

## 13. Literature review plan

Read in this order: 1. MRI modalities and BraTS label definitions. 2. 3D
U-Net and established medical segmentation baselines. 3. Recent BraTS
challenge methods and evaluation protocols. 4. Missing-modality
segmentation methods. 5. Modality dropout and modality-aware fusion. 6.
Self-supervised medical-image learning. 7. Domain generalization and
multi-site robustness. 8. Relevant publications from target PhD labs
such as Tsinghua BIRTH Lab.

Maintain a table with citation/year, dataset, number of subjects,
modalities, method, missing-modality assumptions, metrics, baselines,
limitations, and potential research gap. Verify claims in the full paper
and official dataset documentation.

## 14. Risks and mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  GPU out of memory                   Small patches, batch size 1, mixed
                                      precision, compact model, or remote
                                      compute

  Corrupt/incomplete downloads        Verify archive and test extraction

  Different label conventions         Explicit mappings per release

  Dataset overlap                     Check provenance; avoid false
                                      external-validation claims

  No public validation masks          Follow challenge rules; use labeled
                                      internal validation for model
                                      selection

  Data leakage                        Split by subject and audit
                                      duplicates

  Too many experiments                Prioritize E0--E4

  Self-supervised training too        Make it optional
  expensive                           

  Insufficient novelty                Be honest; establish a gap from
                                      literature before claiming novelty

  Clinical overinterpretation         Present as research, not a
                                      diagnostic tool
  -----------------------------------------------------------------------

## 15. Ethics and responsible use

-   Use data only under its access terms.
-   Cite datasets and follow challenge rules.
-   Do not attempt to identify participants.
-   Do not publish restricted medical images.
-   Do not use the model for clinical decisions.
-   Report limitations and negative results.
-   Do not fabricate metrics, provenance, or novelty.
-   Seek institutional guidance if private hospital data or
    human-subjects research is introduced.

## 16. Expected deliverables

1.  Verified dataset inventory and data card.
2.  Tested preprocessing and loading pipeline.
3.  Baseline segmentation model.
4.  Complete-modality benchmark.
5.  Missing-modality robustness comparison.
6.  Error-analysis figures and tables.
7.  Reduced-label experiment if feasible.
8.  Self-supervised experiment if feasible.
9.  Reproducible code, configurations, logs, and documentation.
10. Master's thesis/report and presentation.
11. Future PhD research direction based on observed limitations.

Publication may be a goal, but completion alone does not guarantee a
paper. Publication depends on novelty, rigor, appropriate comparisons,
evidence, and relevance.

## 17. Connection to future PhD work

The project develops skills in medical image formats, 3D segmentation,
multimodal learning, missing-data robustness, label-efficient learning,
evaluation, and reproducible research.

Possible PhD extensions include stronger self-supervised/foundation
representations for brain MRI, missing-modality synthesis,
cross-hospital generalization, and longitudinal or
neurodegenerative-disease imaging. Choose the extension based on what
the Master's results reveal; do not assume a tumor-segmentation method
automatically transfers to dementia research.

## 18. Immediate next steps

After downloads finish: 1. Extract archives into separate directories
and keep the ZIPs until extraction is verified. 2. Inspect directory
structure and count subjects. 3. Load a few MRI volumes and segmentation
masks. 4. Confirm modalities, channel order, shapes, label values, and
validation-label availability. 5. Review dataset provenance and possible
overlap. 6. Test a loader on one subject and verify image-mask
alignment. 7. Begin baseline training only after these checks pass.

## 19. Project definition in one sentence

**We will build a reproducible brain MRI tumor-segmentation system and
experimentally study how to keep it reliable when one or more MRI
modalities are missing, with a secondary investigation into reducing
dependence on expert-labelled data.**

The essential project is data validation, baseline segmentation,
missing-modality experiments, evaluation, and error analysis.
Self-supervised learning and cross-dataset evaluation are extensions
when data, compute, and time permit.


## 20. International research labs and universities

These groups are relevant to the project's themes. Their research topics can change, so check each official page and recent publications before applying or contacting a supervisor.

### 1. Tsinghua University — BIRTH Lab, China
- **Research fit:** Brain MRI, multimodal imaging, deep learning, brain disease research, and medical-image analysis.
- **Why it matters to our project:** The lab has published work specifically addressing brain-tumor segmentation when MRI modalities are missing, including the AdaMM framework.
- **Official lab:** https://www.birthlab.bme.tsinghua.edu.cn/
- **Research publications:** https://www.birthlab.bme.tsinghua.edu.cn/research/
- **Related paper announcement:** https://bme.tsinghua.edu.cn/en/info/1051/1227.htm

### 2. Yonsei University — Medical Imaging and Computer Vision (MICV), South Korea
- **Research fit:** Medical image analysis, representation learning, domain adaptation/generalization, and neuroimaging-related machine learning.
- **Why it matters:** Useful for studying how models can generalize across datasets and imaging conditions, which connects to our robustness experiments.
- **Official lab/research page:** https://micv.yonsei.ac.kr/research
- **Publications:** https://micv.yonsei.ac.kr/publication

### 3. University of Cambridge — AI for Neuroimaging of Ageing and Dementia, United Kingdom
- **Research fit:** AI for multimodal neuroimaging, dementia subtypes, disease progression, and interpretable models.
- **Why it matters:** This is a potential future PhD direction after the Master's project, especially if we later move from tumor segmentation toward brain ageing and neurodegenerative disease research.
- **Official project page:** https://www.medschl.cam.ac.uk/ai-neuroimaging-ageing-and-dementia

### How these labs relate to our project
- **Closest match to the current thesis:** Tsinghua BIRTH Lab, because its research includes missing-modality brain-tumor segmentation.
- **Useful for robustness and generalization:** Yonsei MICV.
- **Potential future direction in dementia/neuroimaging:** University of Cambridge project.

This list is a starting point, not a ranking of universities or a guarantee of admission. When preparing PhD applications, read each lab's recent papers, identify a specific research question, and check current openings, supervisor availability, funding, and eligibility.
