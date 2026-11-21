# Literature review notes

The shared memory of the review (scope feature 1, Q1 plan). PDFs stay in `~/literature-pdfs`, outside git. Searches are logged in `search-log.csv`, and every paper found is in `candidates.csv`.

Status words used here:
- **read**: the full text was opened and read for this note.
- **not read**: only the title, abstract or a citation was seen. Nothing from such a paper counts as a finding.

## 1. Search rules (fixed 2026-11-07, before the first logged search)

### The question
First question: has anyone already done a fair, subject level, paired comparison of zero fill, modality dropout and presence indicators on BraTS?

Terms, as used in this review:
- **Zero fill**: a missing modality is replaced by zeros, and training saw complete input only.
- **Modality dropout**: whole modalities are removed at random during training.
- **Presence indicator**: the network is told which modalities are present, through input channels or conditioning.
- **Fair**: the arms share the backbone, subjects, split, metrics and seeds, and differ only in the strategy.
- **Subject level, paired**: the same subjects are scored under each arm, and the test is on per subject differences.

### Where we search
- OpenAlex through `fetch_papers.py`, each search logged (ADR 0003).
- Web search, because OpenAlex can miss papers. Those finds are searched again by title, so they enter the log.
- Citation chasing from the papers read: their reference lists, and papers citing them.

### Limits
- Published or preprinted from 2016 onwards. The start year covers HeMIS and ModDrop. Older foundational papers enter only through citation chasing.
- English language.

### Fixed queries
1. `missing modality brain tumor segmentation`
2. `incomplete multimodal MRI glioma segmentation`
3. `modality dropout brain tumor segmentation`
4. `missing MRI sequences segmentation comparison of strategies`
5. `zero filling missing modality segmentation`
6. `modality presence indicator missing MRI segmentation`
7. `robustness missing MRI sequences deep learning segmentation`
8. `benchmark evaluation missing modality BraTS`

Queries added later go in section 4 with the reason, and are logged the same way.

### In or out
- **In**: the paper trains on, or evaluates, multimodal brain tumour MRI, and tests with at least one modality missing.
- **Out**: the task is not segmentation, or the data is not brain tumour MRI. The exception is a paper whose main subject is one of the three strategies. Such a paper is kept as background (ModDrop is the example).
- **Out**: image synthesis with no segmentation result.

### Closeness to our gap
One point for each item a paper meets. Scored from the abstract to choose, then scored again from the full text.

| Item | Meets it when |
|---|---|
| A. Data | It uses BraTS |
| B. Patterns | It tests many missing patterns, ideally all 15 |
| C. Arms | At least two of the three strategies are separate compared arms |
| D. Fair | The arms share the backbone, split and seeds |
| E. Paired | It tests per subject paired differences, not only means |
| F. Reporting | It reports per region results (WT, TC, ET), with Dice and a distance measure |

The 10 papers for deep reading are the highest scores. Ties go to the paper that compares strategies rather than proposing a new network. Search rank and citation count do not choose.

### How the first question will be answered
- **Done already**: at least one paper read in full meets A to E for all three strategies.
- **Partly done**: papers read in full meet part of A to E, or compare only two strategies.
- **Not found**: no paper read so far meets them. This does not prove that none exists, so the wording stays "not found in this search".

Until the 50+ paper review ends, any answer here is provisional.

### Each paper note records
Citation, status, data and split, backbone, arms, missing patterns tested, metrics, statistics, main result with its numbers, closeness score, and what it means for our gap and for the RQs.

## 2. First answer (provisional, 2026-11-07, 10 papers read)

**Has anyone already done a fair, subject level, paired comparison of zero fill, modality dropout and presence indicators on BraTS?**

**Yes, by the rule in section 1: "done already". But only at low resolution, in a manuscript we could not confirm as peer reviewed.** One paper meets A to E for all three strategies: Cai et al. 2026 (note 3.1). Nothing else read comes close on all three.

So the thesis and any journal paper cannot say "nobody has compared these". Cai et al. must be cited and answered.

What keeps the work alive is that Cai et al. describe their own study as a reduced resolution feasibility study, and they name full resolution confirmation and multi seed schedules as the next step. See section 5.

Limits of this answer:
- 10 of 387 candidates were read. It stays provisional until the 50+ paper review ends.
- The status of Cai et al. is unclear. The Zenodo record says "submitted to Journal of Imaging", while the manuscript in the archive is marked as an MDPI Bioengineering submission draft. Two web searches found no published version. That does not prove there is none, so the human should check the journal sites and Google Scholar.

### Scores after reading in full

Items are from section 1. D counts when backbone and split are shared. No paper here except Cai et al. states matching seeds across all arms, so D is generous.

| Note | Paper | A Data | B Patterns | C Arms | D Fair | E Paired | F Report | Score | Strategies compared |
|---|---|---|---|---|---|---|---|---|---|
| 3.1 | Cai 2026 | yes | 15 | yes | yes | yes | yes | 6 | zero fill, dropout, dropout + presence |
| 3.2 | Öchsner 2026 | yes | 1 | yes | yes | yes | yes | 5 | zero fill, FLAIR only dropout |
| 3.3 | Chrysochoou 2026 | yes (BraTS-PEDs) | 3 | yes | yes | yes* | no | 4 | zero fill, dropout, synthesis, copy |
| 3.4 | Diana-Albelda 2026 (CAPS) | yes | 4 | yes | yes | no | yes | 4 | zero fill, dropout, synthesis |
| 3.5 | Lau 2019 | yes | 8** | yes | yes | no | no | 4 | zero fill, dropout, URN |
| 3.6 | Liu 2022 (ModDrop++) | no (MS) | 15 | yes | yes | no | no | 3 | dropout, dropout + presence code |
| 3.7 | Ruffle 2023 | yes | 15 | no | no | no | no | 2 | none: one model per subset |
| 3.8 | Chen 2026 (LASSNet) | yes | 15 | no | no | no | no | 2 | none: new network |
| 3.9 | Pemberton 2023 | yes | no sweep | no | no | no | yes | 2 | none: dropout in every model |
| 3.10 | Grøvik 2021 | no (metastases) | 1 | no | no | no | no | 0 | dropout against a subset model |

| 3.11 | Havaei 2016 (HeMIS) | yes | 15 | no | no | no | no | 2 | new network against mean fill and MLP imputation |
| 3.12 | Dorent 2019 (U-HVED) | yes | 15 | no | yes | yes*** | no | 4 | new network against one model per subset |
| 3.13 | Zhang 2022 (mmFormer) | yes | 15 | no | no | no | no | 2 | new network |
| 3.14 | Liu 2023 (M3AE) | yes | 15 | no | yes**** | yes | yes | 5 | new method; ablation swaps zero, mean and learned fill |
| 3.15 | Sun 2024 ("Unveiling") | yes | 15 | no | no | no | no | 2 | new method, one model per subset |
| 3.16 | Giang 2026 (CMA-mmFormer) | yes | 6 | not usable | no | no | yes | 3 | unreliable, see the note |
| 3.17 | Azad 2022 (SMU-Net) | yes | 15 | no | unclear | no | no | 2 | new network, baselines of unstated source |
| 3.18 | Wang 2021 (ACN) | yes | 15 | no | yes | no | yes | 4 | dedicated models against rerun catch-all models |
| 3.19 | Wang 2023 (ShaSpec) | yes | 15 | no | no | no | no | 2 | new network, baselines copied from other test sets |
| 3.20 | Shi 2024 (PASSION) | yes | not stated | no† | yes | no | yes | 3 | baseline, + ModDrop, + PMR, + PASSION, under imbalanced missing training data |
| 3.21 | Wang 2023 (ProtoKD) | yes | 4 | no | yes | yes‡ | no | 3 | distillation against single modality models |

\* Per patient differences with Bonferroni adjusted p values, but the main text names the test only for risk scores. Table S1 was not opened.
\** Read from Figure 2 in extracted text, whole tumour only. Check against the PDF.
\*\*\* Wilcoxon, p < 0.05, against U-HeMIS only. The paper does not test U-HVED against the subset models, and does not say the test is paired.
\*\*\*\* The shared backbone holds for the ablation table only, which is on the validation split.
† Dropout is a separate arm, but the baseline trains on incomplete data, so it is not our zero fill arm. Only one of our three strategies is present.
‡ Paired t-test, but the table does not say what each star is compared with.

### What the 10 papers agree on
- **Zero fill on a complete input model fails when a key sequence is missing.** CAPS: mean Dice 63.71 against 83.45 with dropout, over the 4 single missing cases. Öchsner: WT Dice 60.4% against 92.6% without FLAIR. Cai: Macro-14 0.3367 against 0.5249.
- **Dropout helps.** Every paper with both arms (3.1 to 3.5) reports this. This part of RQ2 is known, so a benefit of E3 over E1 would be a replication, not a new finding.

### Where they disagree: the open question
**Does dropout cost accuracy on complete input?**
- Cai: yes, large. −0.1090 Dice (95% CI −0.1197 to −0.0985) with uniform draws, and 0.6610 against 0.5822 when complete input exposure rose to 50% (one seed).
- Öchsner: no. Equivalent within ±1.5 points at r = 0.35 (Holm corrected TOST p < 0.001).
- Chrysochoou: almost none. At most 0.006 per patient.
- Lau: dropout was better on complete input, with no test.

These are not like for like. Öchsner and Chrysochoou drop only one or two sequences, on full nnU-Net training. Cai drops any subset, but trains for at most 8 epochs at 96×96×64. Nobody read so far tests uniform all subset dropout at full resolution and full budget, on the same subjects, with a paired test. Our E1, E3 and E4 do exactly that. **This is the strongest gap candidate so far.**

### Presence indicators
Evidence is thin, and it is the weakest part of the literature:
- Cai: paired Macro-14 difference from removing the channels −0.0029 (95% CI −0.0086 to 0.0031, p = 0.3304). One seed, on a reused test set.
- ModDrop++: a presence code on the first layer beat plain dropout on 15 of 15 UMCL rows (for example complete input 0.633 to 0.663), but only 30 patients split 4:1, in MS, with no test.

### Update after 16 papers (2026-11-07, provisional)

1. **Dropout is not one thing.** HeMIS favours dropping zero or one modality. U-HVED gives the complete input 1 draw in 4. Cai and our E3 give it 1 in 15. So comparisons of "dropout" across papers mix different exposures. This strengthens the case for reporting exposure, and for the E3b arm.
2. **The complete input cost is not settled.** Cai finds a large cost at low budget (0.6912 to 0.5822). U-HVED finds a small mixed one against Single (+1.3, -2.0, -1.0, untested). M3AE reaches challenge level on full input, with no matched full only control. No paper so far tests this at full resolution, with the same backbone, split and seeds, with one strategy changed at a time. This is still our strongest gap candidate.
3. **The fill value can matter.** M3AE: 74.4 with zero fill, 45.2 with the mean image, under min-max scaling.
4. **Quality of evidence is uneven.** Two of the six new papers have internal number errors (3.15, 3.16). Check tables before citing a number. Copied baselines are common (3.13, 3.15, 3.16), and these are not paired comparisons.
5. **Statistics stay our strength.** Of the 16, only Cai, Öchsner, Chrysochoou, M3AE and U-HVED report a paired style test, and only Chrysochoou adjusts for multiple tests.

### Update after 21 papers (2026-11-10, provisional)

The five new papers (3.17 to 3.21) are all new methods, not strategy comparisons. None changes the first answer in section 2.

1. **Copied baselines from other test sets are common, even at CVPR.** ShaSpec scores itself on the online validation set and copies baselines scored on other splits. SMU-Net does not say where its baseline numbers come from. Same subject comparison is still uncommon, which supports our framing.
2. **One new piece of dropout evidence goes the other way, in a different setting.** In PASSION, ModDrop on mmFormer lowered mean Dice in 5 of 6 settings (−1.38 to +0.66) and raised HD in all 6, when the training data itself had missing modalities. Our training data is complete, so this does not contradict 3.1 to 3.5. One run, no test.
3. **Post-processing can look like a method gain.** ShaSpec's connected component step alone added up to 2.5 Dice on ET. Our arms use no post-processing, and that must stay the same for all arms.
4. **Dedicated models are a strong reference we do not include.** In ProtoKD, a model trained on one modality beat U-HVED by 9 to 24 points. ACN trains 15 dedicated models. This is a limit of RQ2 to state, not a new arm.
5. **Statistics**: of the 21, only ProtoKD adds a paired test (a t-test, with unclear comparisons). Still no multiplicity correction outside Chrysochoou.
6. **Number errors again.** SMU-Net's Table 1 and Figure 5 disagree on the order of SMU-Net and ACN at full input (85.2 against 84.0 in the table, 83.8 against 85.1 in the figure). That makes three of 21 papers with internal number errors (3.15, 3.16, 3.17).

### Update after 31 papers (2026-11-21, provisional)

The ten new papers (3.22 to 3.31) are nine in scope and one background paper (ModDrop, 3.24). None changes the first answer in section 2, and none narrows the gap in section 5.

1. **Still no test of the complete input cost.** No new paper trains the same backbone with and without all subset dropout and compares them at complete input. The only with and without comparison is ModDrop, in gesture recognition: no cost (96.77 against 96.81 accuracy, one run, validation set), but with about 66% complete input exposure against 6.7% in E3 (my arithmetic).
2. **Dropout is not one thing, again.** Robust-MSeg drops each code with probability 0.5 (complete input 1 in 16, near E3). ModDrop drops at 0.1 (about 2 in 3 complete). Shen and Gao drop exactly one modality and also train on complete input in every step. RFNet, DC-Seg and AdaMM do not give the rate. So "dropout" in a related work sentence must name its exposure.
3. **Collapse without key inputs repeats.** A channel stacked ViT trained on complete input with missing input filled by background fails without FLAIR (WT 5.29, 3.30). LCR-Net's single input collapse (3.29) looks the same, but its training is not described, so it is not evidence.
4. **Same split is not paired.** RFNet's 219, 50, 100 split is the most reused one (3.14, 3.20, 3.25, 3.26), but later papers copy RFNet's numbers rather than score the same subjects under each arm. Copied baselines again in 3.22, 3.23, 3.26, 3.29.
5. **No real benchmark yet.** AdaMM (3.31) calls itself a systematic evaluation of six method families, but it has no validation split, no seeds, no paired test, no plain zero fill or dropout arm, and pattern columns that look mislabelled for several baselines. Section 6 still needs a real benchmark.
6. **Number problems: now 7 of 31** (3.15, 3.16, 3.17, 3.23 claim against table, 3.25, 3.27, 3.31). Check every number before citing it.
7. **Statistics**: none of the ten reports a paired test. The paired style list stays Cai, Öchsner, Chrysochoou, M3AE, U-HVED and ProtoKD, with multiplicity correction only in Chrysochoou.

## 3. Paper notes

Bibliographic details below are copied from the papers. None is in `writing/refs.bib` yet, so none may be cited in a chapter until it is added and verified.

### 3.1 Cai, Tan, Zhu, Lin, Xu 2026. "Availability-Robust Training Can Improve Missing-Modality Brain Tumor Segmentation"
- **Status**: read. Manuscript v1.3.0 and the pilot preregistration, from Zenodo `10.5281/zenodo.22805033` (CC BY 4.0). Not confirmed as peer reviewed. Copy in `~/literature-pdfs/W7213346182-archive/a.zip`.
- **Data and split**: BraTS 2021, 1,251 subjects: 750 train, 123 development, 191 pilot test, 187 final test. External: 146 BraTS-Africa subjects.
- **Backbone**: compact 3D U-Net, about 5.6 M parameters, whole volume resized to 96×96×64, AdamW, up to 8 epochs, batch 4, no augmentation, about 30 minutes per system.
- **Arms**: full only (zero fill at test), availability robust (uniform over 15 codes, zero fill, 4 presence channels), the same without presence channels, and complete exposure 0.25, 0.50, 0.75.
- **Patterns and metrics**: all 15. Macro-14, Worst-15 and full input, on the mean of WT, TC and ET Dice. HD95 as a sensitivity analysis.
- **Statistics**: subject cluster paired bootstrap, 10,000 resamples. No multiplicity correction.
- **Result**: Macro-14 0.5249 against 0.3367 (+0.1882, 95% CI 0.1772 to 0.1996). Full input 0.5822 against 0.6912. Four seeds: 0.5129 ± 0.0193 against 0.3671 ± 0.0298. BraTS-Africa: 0.4774 against 0.2950.
- **Weak points**: the authors state that absolute values are not state of the art. The presence ablation and exposure arms have one seed and reuse the test set. The "preregistration" is local, not in an outside registry. Their control keeps presence channels, so it is not a pure 4 channel zero fill model.
- **A warning for us**: their first external run loaded T2 and FLAIR in swapped order, because BraTS 2023+ names them `-t2w` and `-t2f`. This supports our rule that channel order is checked per dataset.
- **RQs**: RQ1, RQ2, RQ4. Their primary score is almost the same as ours.

### 3.2 Öchsner et al. 2026, Frontiers in Neurology. Doi `10.3389/fneur.2026.1889198`
- **Status**: read. Supplementary tables not opened.
- **Data and split**: BraTS 2021 minus UPenn (848), 5 fold. Test on withheld UPenn-GBM (403).
- **Backbone**: 3D full resolution nnU-Net, 1,000 epochs per fold, ensemble.
- **Arms**: FLAIR only dropout r = 0, 0.35, 0.5, a dedicated three sequence model, and TumorSynth. No presence indicator.
- **Patterns and metrics**: FLAIR present or zeroed. Dice and HD95 for WT, TC, ET, and volume bias.
- **Statistics**: TOST with Holm for equivalence, patient level paired bootstrap with 2,000 resamples. A strong model for how to report.
- **Result**: complete input 94.8% against 95.0%. Without FLAIR 93.4% against 81.0%. WT volume bias 0.83 mL against −45.6 mL. A three sequence model matched dropout at 93.8%.
- **Meaning**: the value of dropout is one model for both cases, not better accuracy. They call UPenn "independent" but say it belongs to the BraTS data family, which matches our overlap rule.
- **RQs**: RQ1, RQ2.

### 3.3 Chrysochoou et al. 2026, npj Precision Oncology. Doi `10.1038/s41698-025-01269-x`
- **Status**: read, as the unedited article in press. Tables S1 to S9 not opened.
- **Data and split**: paediatric, CBTN and BraTS-PEDs, 340 train, 85 validation, 290 test. External PNOC, 43 patients, 157 timepoints.
- **Backbone**: nnU-Net, 1,000 epochs, same hyperparameters across arms.
- **Arms**: zero fill baseline, dropout of FLAIR and T1 (p = 0.4, chosen on validation), ResViT synthesis, copy substitution. No presence indicator.
- **Patterns and metrics**: FLAIR, T1, or both missing. Dice for WT, ET and NET, not the BraTS regions.
- **Result**: dropout stayed within 0.04 Dice of complete input. It gained up to 0.12 over the next best and up to 0.57 over the others.
- **Meaning**: closest large clinical design, but children's tumours and only 3 missing patterns. Its authors say other patterns were not evaluated.
- **RQs**: RQ1, RQ2.

### 3.4 Diana-Albelda et al. 2026 (CAPS), arXiv `2610.28179`
- **Status**: read. Preprint.
- **Data and split**: BraTS 2023 adult glioma, 1,251 training cases, scored on the official 219 case validation set through Synapse.
- **Backbone**: nnU-Net ResEnc M for segmentation.
- **Arms**: Table 1 gives zero imputation, dropout training and L1 synthesis as baselines. The paper's aim is synthesis, not this comparison.
- **Patterns and metrics**: the 4 single missing cases. ENH, TC, WT, Dice and HD95.
- **Result**: over the 4 cases, mean Dice was 63.71 zero fill, 83.45 dropout, 80.84 L1 synthesis. Their full method reached 82.38, still under dropout. Complete input reference 86.65.
- **Statistics**: no paired test on these baselines. Paired tests appear only for radiologist scores.
- **Meaning**: the closest numbers to ours by dataset family, for two arms only.
- **RQs**: RQ1, RQ2.

### 3.5 Lau, Adler, Sjölund 2019, arXiv `1908.06683`
- **Status**: read. Supplement not opened.
- **Data and split**: BraTS 2018, 285 training, scored on the unlabelled validation set by the online system.
- **Backbone**: 2D U-Net, 50 epochs.
- **Arms**: baseline, baseline + dropout (truncated geometric draw, θ = 0.5), URN + dropout, pretrained URN.
- **Result**: figure only. The baseline "fails miserably" when a modality is missing. Dropout offset much of the loss and helped on complete input too. No test.
- **Meaning**: an early clear demonstration in our exact sense, without statistics.
- **RQs**: RQ2.

### 3.6 Liu et al. 2022 (ModDrop++), MICCAI. Arxiv `2203.04959`
- **Status**: read. Supplement not opened.
- **Data and split**: MS lesions, not tumours. UMCL 30 patients, ISBI 5 subjects, split 4:1.
- **Backbone**: 2.5D dense network.
- **Arms**: ModDrop, ModDrop + dynamic head fed a binary presence code, ModDrop++ (adds co-training), one model per pattern.
- **Patterns and metrics**: all 15. Dice and an overall score. No test.
- **Result**: UMCL complete input 0.633, 0.663, 0.704, with 0.690 for the dedicated model.
- **Meaning**: background only. It suggests where presence is fed may matter: ours are input channels, theirs alter the filters.
- **RQs**: RQ2.

### 3.7 Ruffle et al. 2023, Brain Communications. Doi `10.1093/braincomms/fcad118`
- **Status**: read, full text from Europe PMC (`PMC10144694`). The publisher PDF failed. Supplementary tables not opened.
- **Data and split**: BraTS 2021, 1,251 subjects, 5 fold. External: 50 patients.
- **Backbone**: nnU-Net, 1,000 epochs, about 3.5 days per model.
- **Arms**: not strategies. One model per sequence subset, 30 models.
- **Result**: WT Dice 0.907 (single sequence) to 0.945 (all four). ET without contrast 0.756 to 0.790.
- **Meaning**: an estimate of what a dedicated model can reach for each pattern, useful as a ceiling for RQ1. It reports cross validation numbers and no paired test.
- **RQs**: RQ1.

### 3.8 Chen et al. 2026 (LASSNet), arXiv `2609.06733`
- **Status**: read. Preprint.
- **Data and split**: BraTS2019 234/33/68 and BraTS2023 875/125/251.
- **Backbone**: new availability conditioned fusion network, 128³ crops, 1,000 to 1,680 epochs, uniform draw over 15 masks.
- **Result**: BraTS2023 mean over 15 patterns: 90.5 WT, 85.5 TC, 73.6 ET, 83.2 overall. It did not beat IM-Fuse overall.
- **Statistics**: the Wilcoxon test is across the 15 configurations, not across subjects. So it is "paired" in a different sense from ours.
- **Meaning**: a current reference point for BraTS 2023. Its mean includes complete input, so it is not directly comparable to a 14 pattern score. The ablation is one run, and the authors say repeats are needed.
- **RQs**: RQ2, as context.

### 3.9 Pemberton et al. 2023, Scientific Reports. Doi `10.1038/s41598-023-44794-0`
- **Status**: read.
- **Data and split**: BraTS 2021 + PICTURE, 12 hospitals, with real missing sequences. External 227.
- **Arms**: nn-Unet, nvNet and DeepMedic, all with sparsified training (T1, T2, FLAIR zeroed at 20%). No arm without it.
- **Result**: nn-Unet best. No significant gap between complete and incomplete groups.
- **Weak point for our question**: the comparison is between different patients, with an independent t-test, so it is not paired. T1c was always required.
- **Meaning**: compares networks, not strategies. Useful as evidence about real world missingness.
- **RQs**: RQ1, as context.

### 3.10 Grøvik et al. 2021, npj Digital Medicine. Doi `10.1038/s41746-021-00398-4`
- **Status**: read.
- **Data and split**: brain metastases, 100 train, 10 validation, 55 test, on a different hospital.
- **Arms**: input dropout against a DeepLab V3 trained on the test subset.
- **Result**: Dice 0.795 against 0.774 (p = 0.017). Rank sum test, not a paired test. Only one missing pattern.
- **Meaning**: background on dropout. Scores 0, so it was a weak choice for the top 10.
- **RQs**: RQ2, as context.

### 3.11 Havaei, Guizard, Chapados, Bengio 2016 (HeMIS), MICCAI. Arxiv `1607.05194`
- **Status**: read in full (arXiv copy).
- **Data and split**: BraTS 2015 (220 high grade, 54 low grade), 70/10/20. Also two MS cohorts.
- **Backbone**: 2D. One small convolutional stream per modality, then the mean and variance over the modalities present.
- **Arms**: HeMIS, mean filling, and MLP imputation (28 separate MLPs). All means are zero after normalisation, so mean filling here is zero filling.
- **Dropout rule**: warm up with all modalities, then drop at random, "ensuring a higher probability of dropping zero or one modality only". So it is not uniform over the 15 patterns. No probabilities are given.
- **Patterns and metrics**: all 15. Dice only, for Complete, Core, Enhancing. No statistics, wins are counted per pattern.
- **Result**: complete input Dice 83.15, 72.5, 75.37 against mean fill 82.43, 71.46, 72.08. FLAIR alone, whole tumour: 71.26 against 72.30, so zero fill wins this one pattern. T1c alone: 33.46 against 23.11. T2 alone: 58.48 against 2.70.
- **For us**: this is the origin of two ideas we compare (train with dropped modalities, and zero fill as the control). The zero fill failure on single inputs depends on the pattern, not uniformly, which supports reporting all 15. RQ1, RQ2.

### 3.12 Dorent, Joutard, Modat, Ourselin, Vercauteren 2019 (U-HVED), MICCAI. Arxiv `1907.11150`
- **Status**: read in full (arXiv copy).
- **Data and split**: BraTS 2018 training set, 285 subjects, 3 folds, 70/10/20 each.
- **Backbone**: 3D U-Net style, one encoder per modality, product of Gaussians at each level, decoders for images and segmentation. Input 112³ patches, batch 1, Adam, up to 60,000 iterations with early stopping, flips and small rotations.
- **Arms**: HeMIS, U-HeMIS, U-HVED, and Single (one U-Net per subset, 15 per fold, 45 in all, inputs concatenated).
- **Dropout rule**: draw the number of modalities uniformly from 1 to 4, then a subset of that size. By my arithmetic the complete input is drawn 1 in 4, each single 1 in 16, each pair 1 in 24, each triple 1 in 16. This is not uniform over 15, where the complete input would be 1 in 15.
- **Patterns and metrics**: all 15. Dice only. Wilcoxon, p < 0.05, U-HVED against U-HeMIS, no multiplicity correction.
- **Result**: means over 15 patterns, U-HVED against Single: 80.1 and 81.6, 64.0 and 66.2, 50.0 and 52.7. Complete input: 88.8 and 87.5, 76.4 and 78.4, 71.7 and 72.7. So at complete input the shared model is 1.3 higher on whole tumour, but 2.0 lower on core and 1.0 lower on enhancing, with 25% complete input exposure and no test of that difference. T1 alone, whole tumour: 52.4 against 72.7, so "similar to Single" holds for the means, not for every pattern.
- **For us**: the closest early evidence on the complete input cost, and the only one here with a subset specific control. Its exposure is far higher than our E3. RQ2. Supports the exposure arm E3b.

### 3.13 Zhang et al. 2022 (mmFormer), MICCAI. Arxiv `2206.02425`
- **Status**: read in full (arXiv copy).
- **Data and split**: BraTS 2018, 285 subjects, the split of ACN. U-HeMIS, U-HVED and ACN numbers are copied from ACN, not rerun.
- **Backbone**: one convolutional encoder and intra-modal transformer per modality, one inter-modal transformer, one decoder. 106 M parameters, 128³, batch 1, 1000 epochs, about 25 hours, 17 GB on each of 4 V100 GPUs.
- **Arms**: mmFormer against the copied baselines, plus ablations. Modality dropout is built in through a Bernoulli indicator; the missing token is a zero vector.
- **Patterns and metrics**: all 15. Dice only. The text says "significantly" but names no test.
- **Result**: means 59.85, 72.97, 82.94 (ET, TC, WT). Complete input 77.61, 85.78, 89.64. The headline 19.07 is Dice points on ET with one modality, against the two baselines.
- **For us**: it cannot run on our hardware budget as it stands. Its table is a poor reference for us because the baselines are not rerun. Do not compare our Dice with it directly: split, budget and preprocessing differ.

### 3.14 Liu, Wei, Lu, Sun, Wang, Zheng 2023 (M3AE), AAAI. Doi `10.1609/aaai.v37i2.25253` (from OpenAlex)
- **Status**: read in full, except the supplement.
- **Data and split**: BraTS 2018 (199:29:57) and BraTS 2020 (219:50:100), the splits of RFNet.
- **Backbone**: one 3D U-Net style encoder and decoder, the only one here close to our model shape. 128³, batch 2 volumes, Adam, 600 pretraining plus 300 tuning epochs, clip then min-max scale, flips, intensity shift and scale.
- **Arms**: M3AE against HVED, ACN, SMU-Net, RFNet, rerun on the same split (RFNet from the authors' checkpoints). Ablation on the validation split.
- **Patterns and metrics**: all 15. Dice and HD95. Wilcoxon signed rank, p < 0.05, against M3AE only, no multiplicity correction.
- **Result**: means on BraTS 2018 85.8, 77.4, 59.9 (WT, TC, ET), best of the five, though not significantly above RFNet on WT. Full input on the official validation set 90.5, 86.1, 81.0, near the single model challenge entry at 90.4, 86.0, 81.5. HVED there: 88.2, 76.5, 70.9.
- **Fill ablation (key)**: over 15 patterns, mean Dice: learned substitute 75.9, zero fill 74.4, training mean image 45.2. Replacing the fill with the mean image cost about 30 points, so what fills a missing channel can matter a great deal. This was under min-max scaling, where zero is not the mean intensity. In HeMIS zero is the mean. Ours is like HeMIS: `common/data.py` z-scores each channel over the brain mask, so zero is the brain mean. State this when we report.
- **For us**: a counterpoint to the complete input cost: a model trained with dropped modalities reached challenge level on full input. But there is no full only model with the same backbone and budget here, and the model also has pretraining and distillation. It does not settle the question. It is also a candidate for the "one published method" idea, but it has about 900 epochs of training, so it needs a cost check. RQ2.

### 3.15 Sun, Li, Wang, Cheng, Zhou, Li 2024 ("Unveiling Incomplete Modality"), arXiv `2406.08634`
- **Status**: read in full. Preprint in a journal template. Publication not confirmed.
- **Correction to section 6**: it is not a benchmark. It is a new method. Do not use it as one.
- **Data and split**: BraTS 2018 and 2020, RFNet configuration. Baselines are copied from their papers.
- **Backbone**: Swin UNETR style, 61.98 M parameters, four GPUs, one model per pattern (15 models), with masked pretraining and Hölder divergence distillation.
- **Patterns and metrics**: all 15. Dice only. No statistics.
- **Result**: means on BraTS 2018 87.6, 78.0, 65.8. On T2 alone, Hölder gave a mean of 71.1 against 70.4 for KL.
- **Problems found**: the BraTS 2020 table has no average for their own method, and its mmFormer averages (76.1, 66.5, 42.8) are the D2-Net averages from the BraTS 2018 table. Training length is stated both as 800 total epochs and as 800 plus 300. The loss is L1 in the text and MSE in the algorithm.
- **For us**: little. Do not use its numbers without checking.

### 3.16 Giang 2026 (CMA-mmFormer), IOSR-JCE 28(3), 55 to 64. Doi `10.9790/0661-2803025564`
- **Status**: read in full. Journal peer review quality not assessed.
- **Why I checked it**: section 6 listed it under presence indicators. Its idea does touch them: it adds an availability mask and a reliability score to the fusion step.
- **What it claims**: on BraTS 2018, better Dice, HD95 and calibration than mmFormer, and it reports zero fill (78.6, 68.2, 61.5) and dropout (81.4, 72.5, 62.3) lines.
- **Do not use its numbers.** Problems found by reading:
  - The abstract and the results disagree. Abstract, mmFormer to CMA, WT: 84.20 to 87.10. Results: 89.64 to 90.15. The same holds for TC and ET.
  - The ablation table shows the mean Dice falling from 85.7 to 72.8 and 72.1, but the text says falls of 3.10 and 3.60.
  - The methods say the number of epochs "will be reported according to the completed experiments", so it was left unfilled.
  - The complete input baseline is copied from the mmFormer paper, though the paper claims a controlled head to head comparison.
  - Its other mmFormer lines do not match that paper. My arithmetic from mmFormer Table 1: FLAIR only 62.2 (here 67.5), T2 only 62.8 (here 69.1), missing T1ce 69.0 (here 77.2), FLAIR plus T2 68.3 (here 75.9). The source of its numbers is not given.
  - The zero fill and dropout arms are not described anywhere.
- **For us**: the idea of separating a missing modality from a degraded one is interesting, and is outside our scope. Its numbers are not evidence.

### 3.17 Azad, Khosravi, Merhof 2022 (SMU-Net), arXiv `2204.02961`
- **Status**: read in full (arXiv v1, marked "Under Review for MIDL 2022"). Publication not confirmed. OpenAlex gives an RWTH repository DOI `10.18154/rwth-2022-07632`.
- **Data and split**: BraTS 2018, 285 subjects, "pre-processing and data division follow" U-HVED. Input 160×192×128.
- **Backbone**: two parallel 3D U-Nets (full input path and missing input path), co-trained, plus a style discriminator. Adam, lr 1e-4, batch 1, 250 epochs, one RTX 3090. Only the missing input path is used at inference.
- **Arms**: SMU-Net against U-HeMIS, HVED and ACN. The paper says "identical settings for all four", but does not say whether the baselines were rerun. Its ACN numbers differ from the ACN paper (3.18), for example T2 only WT 85.4 here against 85.55 there, so they are not a straight copy. Their source is not given.
- **Training per pattern**: not stated. The design (a missing input path for a subset M′) suggests one model per pattern, like ACN.
- **Patterns and metrics**: all 15. Dice only. No test, no seeds.
- **Result**: mean Dice 85.9, 77.9, 61.8 (WT, TC, ET), against ACN 85.3, 76.8, 60.7. Single modality average (Table 3) 69.4 against 68.5 for ACN and 57.3 for a single modality U-Net.
- **Internal errors found** (my arithmetic from Table 1):
  - Full input, mean of the three regions: Table 1 gives SMU-Net 85.2 and ACN 84.0. Figure 5 gives 83.8 and 85.1, the opposite order.
  - U-HeMIS on T1 only: 34.9 from Table 1, 16.7 in Figure 5 and Table 3. On T2 only: 51.0 against 36.0.
  - U-HeMIS TC mean row says 59.7. The 15 rows give 60.95.
  - SMU-Net on T1 only: 63.6 in Figure 5, 63.3 in Table 3.
  - The ablation (Table 2) is run on FLAIR only, and no validation set is described, so module choices may have been made on the reported data.
- **For us**: little. One more knowledge distillation method with small gains over ACN and no test. Do not use its numbers without checking.

### 3.18 Wang, Zhang, Liu, Lin, Tian, Zhong, Shi, Fan, He 2021 (ACN), MICCAI. Doi `10.1007/978-3-030-87234-2_39` (from OpenAlex)
- **Status**: read in full (arXiv `2106.14591` v2).
- **Data and split**: BraTS 2018, 285 subjects, random 2:1 split into training and validation. Results are on that validation part. No separate test set.
- **Backbone**: the 3D U-Net of the BraTS 2018 winner, two paths (full input and missing input), two discriminators, a mutual information module. Patch 160×192×128, batch 1, Adam, lr 1e-4 with poly decay, 300 epochs, one V100.
- **Arms**: ACN against U-HeMIS and U-HVED ("U-Net is applied as the benchmark to all methods", so they appear to be rerun on the same backbone and split), and against KD-Net on the four single modality cases.
- **Training per pattern**: yes. "A series of independent yet related models are trained dedicated to each missing situation", so 15 dedicated models. The authors name this training cost.
- **Patterns and metrics**: all 15. Dice and HD95 per region. "Significantly" is used, but no test is named. One run, no seeds.
- **Result**: mean Dice ET 61.21, TC 77.62, WT 85.92 (U-HeMIS 46.10, 62.57, 74.05; U-HVED 46.76, 64.84, 79.16). Mean HD95 7.37, 8.13, 7.62. Without T1ce, U-HeMIS and U-HVED reach only about 10 to 30 on ET; ACN about 42 to 47. Table 1 averages check out, except WT 85.92 printed against 86.00 from the rows.
- **Copying downstream**: ShaSpec (3.19) copies its U-HeMIS, U-HVED, KD-Net and ACN numbers from this paper to the decimal.
- **For us**: background. Dedicated models are a different family from our three strategies. But its table shows how badly a catch-all model can fail on ET without T1ce, which matches what we expect for E1 (RQ1).

### 3.19 Wang, Chen, Ma, Avery, Hull, Carneiro 2023 (ShaSpec), CVPR. Doi `10.1109/cvpr52729.2023.01524` (from OpenAlex)
- **Status**: read in full (arXiv `2307.14126` v2).
- **Data and split**: BraTS 2018. Trained on all 285 training subjects "without model selection", 180,000 iterations. Scored on the 66 case official validation set through the online system.
- **Backbone**: 3D U-Net with one shared encoder and one specific encoder per modality, fused at the bottleneck. SGD with Nesterov momentum 0.99, lr 1e-2, cosine decay. A missing modality's feature is the mean of the shared features of the present ones, so this is feature level filling, not zero fill.
- **Arms**: ShaSpec, with and without a post-processing step (connected components, small region removal, called ShaSpec*). Non-dedicated training drops modalities at random; the rate is not given.
- **Baselines are copied, from other test sets.** U-HeMIS and U-HVED match ACN Table 1 to the decimal (for example 11.78, 23.80 on ET with FLAIR only), and mmFormer matches the mmFormer paper (complete input 77.61, 85.78, 89.64). Those were scored on ACN's internal hold out (one third of the 285) and on mmFormer's own split. ShaSpec was scored on the online validation set. So the table compares different subjects.
- **Patterns and metrics**: all 15. Dice only. No test, no seeds.
- **Result**: mean Dice ET 61.58, TC 77.45, WT 85.92; ShaSpec* 63.08, 77.88, 86.26. The post-processing alone moved ET with T1 plus T1c from 75.76 to 78.26.
- **For us**: two cautions. First, a "+2.5 Dice on ET" claim can come from post-processing, so we must keep post-processing identical across arms (we use none). Second, it is a widely cited CVPR paper whose main table compares numbers from different test sets. Fair comparison on the same subjects is not the norm even in strong venues, which supports our framing (RQ2).

### 3.20 Shi, Shang, Sun, Yu, Yang, Yan 2024 (PASSION), ACM MM. Doi `10.1145/3664647.3681543` (from OpenAlex)
- **Status**: read in full (main paper, no supplement).
- **Question**: training data where modalities are missing at different rates (imbalanced), and missing inputs are never seen in training. The authors call this imperfect data training (IDT), against perfect data training (PDT), where all training cases are complete.
- **Data and split**: BraTS 2020, 369 subjects, 219 train, 50 validation, 100 test. Also MyoPS 2020 (cardiac, 25 cases, 2D), out of our scope. Resampled to 1 mm³, cropped, z-scored per volume.
- **Backbone**: mmFormer for the main table; RFNet and M2FTrans for a plug-in test. AdamW, lr 2e-4, batch 1, 300 epochs, RTX 3090. Missing inputs are dropped and their encoded features set to zero vectors.
- **Arms (same backbone and split)**: baseline, +ModDrop, +PMR (a modality balancing method), +PASSION, under six missing rate settings. Rates for (T1 and T1c, FLAIR, T2) are permutations of 0.2, 0.5, 0.8.
- **Patterns and metrics**: Dice and HD per region. The text does not say clearly which test patterns the table averages (probably all 15, as in mmFormer; not confirmed). No test, no seeds.
- **Result for dropout** (my arithmetic from Table 1, BraTS mean Dice): ModDrop minus baseline is −0.46, +0.66, −1.38, −0.33, −0.12 and −0.47 over the six settings. Mean HD got worse in all six, for example 14.35 to 16.03 mm and 17.10 to 22.72 mm. The authors: ModDrop "even brings negative effects". PASSION gains about 1.3 to 2.9 Dice over the baseline and lowers HD by about 4 to 9 mm. The PDT baseline (complete training data) is 70.58.
- **Note**: the baseline is not our zero fill arm, because its training data already has missing cases. And the mmFormer backbone already has built-in modality dropout.
- **For us**: the only paper so far where ModDrop is a separate arm on the same backbone and split and does not help. The setting differs from ours (our training data is complete), so it does not contradict 3.1 to 3.5. It is a reason to report HD95 next to Dice: in PASSION, dropout hurt the distance measure more than Dice (RQ1, RQ2). One run, no test, so it is weak evidence.

### 3.21 Wang, Yan, Zhang, Wei, Li, Li 2023 (ProtoKD), ICASSP. Doi `10.1109/icassp49357.2023.10095014` (from OpenAlex)
- **Status**: read in full (arXiv `2303.09830` v2, 4 pages).
- **Data and split**: BraTS 2018, 285 subjects, 70% train, 10% validation, 20% test. Model selection on validation (highest Dice). The only paper in this batch with a separate validation and test split.
- **Backbone**: VNet. Teacher trained on all four modalities for 1,000 epochs, batch 4, Adam. Then a student per single modality, 1,000 more epochs, with pixel and prototype distillation.
- **Arms**: Unimodal (a VNet trained on one modality only), U-HVED, KD-Net, PMKL (rerun on the same VNet from public code), ProtoKD. Whether U-HVED and KD-Net were rerun is not stated; the U-HVED row has only one decimal of precision, which suggests a copy.
- **Patterns and metrics**: the 4 single modality cases only. Dice only. Paired t-test, p ≤ 0.05. The table marks some baselines with a star too, and does not say what each star is compared with. One run, no seeds.
- **Result** (mean of WT, TC, ET): Teacher, full input, 80.93. ProtoKD against Unimodal: T1 63.07 against 58.77, T2 65.82 against 64.91, T1ce 77.39 against 73.69, FLAIR 64.17 against 62.07. U-HVED: 34.43, 55.27, 64.87, 52.43.
- **For us**: a simple model trained on one modality beats the catch-all U-HVED by 9 to 24 points here, and the distillation method adds only 0.9 to 4.3. A dedicated model is a strong reference that our three arms do not include. Worth one sentence in the discussion as a limit of RQ2, not a new arm.

### 3.22 Chen, Dou, Jin, Chen, Qin, Heng 2019 (Robust-MSeg, called RobustSeg by later papers), MICCAI. Doi `10.1007/978-3-030-32248-9_50` (from candidates.csv)
- **Status**: read in full (arXiv `2002.09708` v1).
- **Data and split**: BraTS 2015, 274 training subjects. Full input scored on the 110 case online test set. Missing patterns scored on "the same data split of BRATS training set as in" HeMIS.
- **Backbone**: 3D U-Net with one content encoder and one appearance encoder per modality, reconstruction decoders, and a learned gate that fuses the content codes at each level. 80³ patches, batch 1, Adam, lr 1e-4 with poly decay. Z-score inside the brain.
- **Dropout rule**: each modality's content code is set to zero with probability 0.5, independently, in latent space, not at the input. The paper draws this "in each downsampling stage" and does not say whether one draw is shared across stages. If it is one draw per iteration, by my arithmetic the complete input is drawn 1 in 16, close to E3's 1 in 15, and all four codes are dropped together 1 in 16, a case the paper does not explain.
- **Arms**: the method against HeMIS and HeMIS's imputation MLP. Those numbers are "directly referenced" from the HeMIS paper, not rerun. I checked them against HeMIS (3.11): the complete input, T2, T1c and FLAIR rows match to the decimal.
- **Patterns and metrics**: all 15. Dice only for missing patterns. No test, no seeds.
- **Result**: means over 15 patterns 84.45, 69.19, 57.33 (whole, core, enhancing), against HeMIS 68.22, 54.07, 43.86. Complete input 89.07, 81.19, 79.13. All nine column means in Table 2 check out (my arithmetic).
- **Closeness score (full text)**: 2 (A, B).
- **Gap (section 5)**: does not narrow it. Dropout is part of the method and there is no model trained without it, so the complete input cost is not tested. It adds one more exposure level to the "dropout is not one thing" thread.

### 3.23 Hu, Maillard, Zhang, Ciceri, La Barbera, Bloch, Gori 2020 (KD-Net), MICCAI. Doi `10.1007/978-3-030-59710-8_75` (from candidates.csv, row `W3090336986`)
- **Status**: read in full, through the arXiv copy (`2106.09564` v1, row `W3169291543`). The venue copy has no open link.
- **Data and split**: BraTS 2018, 285 subjects, three fold cross validation. Central crop 128³, then subsampled to 64³ "due to memory and time constraint". Random flips.
- **Backbone**: an nnU-Net style encoder and decoder for both teacher and student. The teacher sees all four modalities, the student only T1ce. Adam, lr 1e-4, 500 epochs, lr times 0.2 when the validation loss stalls for 50 epochs. Which data the validation loss uses is not described.
- **Arms**: a T1ce only baseline, the teacher, KD-Net. U-HVED and HeMIS are copied from the U-HVED paper (another split). Their T1c only numbers (65.5, 66.7, 62.4 for ET, TC, WT) match the U-HVED T1c only row as reprinted in 3.27 Table 6.
- **Patterns and metrics**: T1ce only. Dice, mean ± SD (over folds, I think; not stated). No test.
- **Result**: KD-Net 71.67, 81.45, 76.98 (ET, TC, WT), baseline 68.1, 80.28, 77.06, teacher with all four 69.47, 80.77, 88.48.
- **Claim against table**: the abstract says the student "always outperforms the baseline mono-modal network". On WT it does not: 76.98 against 77.06.
- **Closeness score (full text)**: 1 (A).
- **Gap (section 5)**: does not narrow it. One point for the RQ2 limit: the four modality teacher beats the T1ce only model by only 1.4 on ET and 0.5 on TC. A dedicated model is a strong reference, as in 3.21.

### 3.24 Neverova, Wolf, Taylor, Nebout 2016 (ModDrop), IEEE TPAMI. Doi `10.1109/tpami.2015.2461544` (from candidates.csv)
- **Status**: read in full (arXiv `1501.00102` v2). **Background**, not in scope by section 1: gesture recognition, not brain MRI. Kept because its main subject is one of our strategies (the example named in section 1). Published 2015 online; found by a title search with `--from-year 2014`, a departure from the 2016 limit that section 1 allows through citation chasing.
- **Data**: ChaLearn 2014 Looking at People, 13,858 gesture instances, depth, intensity, mocap, plus audio. Also MNIST cut into four quarters as four "modalities".
- **Method**: each modality path is pretrained alone, the shared layers start as an average of the paths, then fusion is trained while each modality is set to zero with probability q. Here 20% input dropout plus ModDrop at q = 0.1, "optimized on the validation set". The paper derives that the ModDrop loss is about the complete input loss plus a cross-modality term.
- **Result (validation set, one run, no test)**: complete input, accuracy 96.77 without ModDrop against 96.81 with it, Jaccard 0.876 against 0.880. MNIST complete input error 1.02 against 1.02. With mocap missing: 38.41 against 92.82 accuracy.
- **Exposure, my arithmetic**: q = 0.1 on four paths gives a complete training sample 0.9⁴ = 66% of the time, against 1 in 15 (6.7%) in our E3.
- **Closeness score (full text)**: 2 (C, D), but out of domain.
- **Gap (section 5)**: the only with and without dropout comparison at complete input read so far, and it shows no cost. It does not narrow our gap: different task, about ten times more complete input exposure, pretrained paths, one run. A 0.04 point difference in one run is not evidence of a gain, so the authors' "further improvement" must not be repeated. It is the origin of the word and worth one sentence in chapter 2.

### 3.25 Ding, Yu, Yang 2021 (RFNet), ICCV. Doi `10.1109/iccv48922.2021.00394` (from candidates.csv)
- **Status**: read in full, main paper only (no supplement). OpenAlex has no open link (`no_open_copy` in candidates.csv). Read from the CVF open access copy, saved as `~/literature-pdfs/W3203934147.pdf`. I typed the CVF address from the usual pattern and checked the title inside the PDF.
- **Data and split**: BraTS 2020, 369 subjects, split at random into 219, 50, 100. BraTS 2018 199, 29, 57, plus three folds with the U-HVED split lists. BraTS 2015 242, 12, 20.
- **Backbone**: 3D U-Net, four encoders, a region aware fusion module, and a shared decoder that segments each modality alone as a regulariser. 80³ crops, rotations, intensity shifts and flips, 300 epochs, batch 2, Adam, lr 2e-4 with poly decay.
- **Missing inputs in training**: missing modal features are set to zero. The drop rate is not in the main text.
- **Post-processing**: fewer than 500 enhancing voxels are relabelled as non-enhancing. In the ablation this alone moves the mean ET Dice from 59.05 to 61.47. The paper does not say whether the rerun baselines got the same step.
- **Arms**: HeMIS, U-HVED and RobustSeg (3.22) rerun from the authors' code on the BraTS 2020 split, so same split, different networks. On BraTS 2015 and 2018 most baseline numbers are copied, for example U-HVED 80.1, 64.0, 50.0 from 3.12.
- **Patterns and metrics**: all 15. Dice only. No test, no seeds.
- **Result (BraTS 2020)**: means 86.98, 78.23, 61.47 (whole, core, enhancing), RobustSeg 84.17, 73.45, 55.49. Complete input 91.11, 85.21, 78.00.
- **Internal error (my arithmetic)**: the HeMIS core mean is printed 65.45. Its 15 rows give 62.45. The other 11 column means check out.
- **Closeness score (full text)**: 2 (A, B).
- **Gap (section 5)**: does not narrow it. Its 219, 50, 100 split is reused by M3AE (3.14), PASSION (3.20) and DC-Seg (3.26), which then copy RFNet's numbers. Same split is not paired: no paper scores the same subjects under each arm with a paired test.

### 3.26 Li, Li, Mao, Ding, Huang 2025 (DC-Seg), MICCAI. Doi `10.1007/978-3-032-04984-1_14` (from candidates.csv)
- **Status**: read in full (arXiv `2505.11921` v1).
- **Data and split**: BraTS 2020, the split of RFNet. Also a private white matter hyperintensity set (80 scans), out of scope. 112³ patches, flips, crops, intensity shifts, Adam, lr 2e-4, 500 epochs, batch 2.
- **Backbone**: anatomy and modality encoders per modality, two contrastive losses, reconstruction, RFNet's fusion and its shared decoder regulariser. Missing modalities are zeroed in latent space with a Bernoulli indicator. The rate is not given.
- **Arms**: RobustSeg, RFNet, mmFormer and M3AE, "directly reference the results". Checked: the RobustSeg and RFNet columns equal RFNet's Table 1, row by row.
- **Patterns and metrics**: all 15. Dice only. No test, no seeds.
- **Result**: means 87.54, 79.63, 65.00 against RFNet 86.98, 78.23, 61.47. Complete input 90.95, 86.46, 81.52. Column means check out to rounding. RFNet's ET includes post-processing; DC-Seg's post-processing is not stated.
- **Closeness score (full text)**: 2 (A, B).
- **Gap (section 5)**: does not narrow it. One more paper with copied baselines.

### 3.27 Azad, Khosravi, Dehghanmanshadi, Cohen-Adad, Merhof 2022, review, arXiv. Doi `10.48550/arxiv.2203.06217` (from candidates.csv)
- **Status**: read in full (arXiv `2203.06217` v1). A narrative review, not a benchmark: no new runs.
- **Content**: five groups of methods (synthesis, shared latent space, knowledge distillation, mutual information, GANs), datasets, metrics, tables of published numbers. Modality dropout and zero fill are not treated as strategies of their own. One cited line: in van Tulder and de Bruijne 2015, a synthesised sequence gave the same result as zeros (classification, not segmentation; not read by us).
- **Tables**: Table 6 is SMU-Net's Table 1 (3.17, same first author), with the same U-HeMIS core mean of 59.7 that its rows do not give.
- **Internal errors (my arithmetic)**: Table 7 says it shows the mean of whole, core and enhancing for each single modality.
  - The HeMIS BraTS 2015 row (4.67, 49.93, 20.31, 5.57 for T1, T1c, T2, FLAIR) is the enhancing value alone. The three region means from HeMIS are 18.44, 42.65, 39.66, 38.09.
  - The ACN row (81.0, 82.7, 86.1, 86.41) does not match the means of ACN's own rows in Table 6 (63.8, 80.37, 64.63, 65.3).
  - The RobustSeg, HVED and SMU-Net rows check out.
- **Closeness score**: not scored (review).
- **Gap (section 5)**: does not narrow it. It confirms that in 2022 the field framed the problem by method family, not by training strategy compared on equal terms.

### 3.28 Shen, Gao 2019, IPMI. Doi `10.1007/978-3-030-20351-1_32` (from candidates.csv)
- **Status**: read in full (arXiv `1904.07290` v1).
- **Data and split**: "BraTS17", given as 262 glioblastoma and 199 lower grade scans. That does not match the BraTS 2017 training set (285: 210 and 75, as in 3.27 Table 2); not explained. 2D patches, scaled to −1 to 1, cropped to 200×186, "randomly split the extracted 3D volume data" 80/20. Whether the split is by subject is not clear. Dice is computed per 2D slice.
- **Backbone**: 2D U-Net, one encoder per modality, fusion in the decoder. A missing modality's encoder output is set to zero.
- **Dropout rule**: at each step exactly one modality is dropped, chosen uniformly, and the loss is the full input loss plus the drop one loss plus an adversarial loss on the bottleneck. So every step also trains on complete input.
- **Arms**: the method against the same network without the adversarial loss (both trained with dropout).
- **Patterns and metrics**: 5 (complete input and the four drop one patterns). Dice only. No test, no seeds.
- **Result**: complete input 0.894, 0.790, 0.653 (WT, TC, ET) against 0.875, 0.693, 0.554 without the adversarial loss. FLAIR missing, WT 0.616.
- **Closeness score (full text)**: 1 (A).
- **Gap (section 5)**: does not narrow it, since there is no model without dropout. But its design is one answer to a complete input cost: a complete input loss in every step. Worth one sentence in the discussion (RQ2), not a new arm.

### 3.29 Zhou, Canu, Vera, Ruan 2021 (LCR-Net), IEEE TIP. Doi `10.1109/tip.2021.3070752` (from candidates.csv)
- **Status**: read in full (arXiv `2104.06231` v2).
- **Data and split**: BraTS 2018 (285) and 2019 (335), split at random 80/20. Cropped and resized to 128³, N4, intensity normalisation. Keras, Adam, lr 5e-4, early stopping on a validation loss whose data is not described.
- **Backbone**: one encoder per modality, a "correlation model" that writes each modality's features as a weighted sum of the others, attention fusion, reconstruction decoders.
- **Missing inputs in training**: not stated. No dropout is described.
- **Arms**: with and without the correlation model, and HeMIS, whose BraTS 2018 numbers are "taken from" another paper on another split.
- **Patterns and metrics**: all 15, Dice only. HD and sensitivity at complete input only. No test, no seeds.
- **Reporting conflict**: the text says "all the results are obtained by online evaluation platform", but the ablation tables are captioned "training set", and the missing pattern tables have no stated source. So which subjects were scored is unclear.
- **Result**: complete input 88.2, 78.6, 69.4 (BraTS 2018 ablation table). Single inputs collapse: T1 only WT 5.6, TC 6.3, ET 5.3; T1c only WT 33.6.
- **Closeness score (full text)**: 2 (A, B).
- **Gap (section 5)**: does not narrow it. The collapse on single inputs looks like a model trained on complete input only, but the paper does not say so, so it is not evidence for the zero fill arm.

### 3.30 Erdur, Beischl, Scholz, Pan, Wiestler, Rueckert, Peeken 2025 (MultiMAE for brain MRI), LNCS. Doi `10.1007/978-3-032-09513-8_55` (from candidates.csv)
- **Status**: read in full (arXiv `2509.11442` v2).
- **Data and split**: BraTS 2021 glioma with UPENN-GBM, UCSF-PDGM, LUMIERE, Rembrandt, EGD, TCGA and BraTS 2023 metastasis, "patient overlaps between BraTS'21 and other sources are removed". Glioma segmentation 2,676, 536, 358, plus 252 in-house external cases.
- **Backbone**: ViT-B/16 pretrained as a multimodal masked autoencoder for 1,200 epochs (75% of tokens masked, split over modalities by a Dirichlet draw, so a whole modality can be masked), then a UNETR decoder finetuned for 100 epochs. Finetuning sees complete input only.
- **Arms**: this model against a ViT with the four modalities stacked as channels, MAE pretrained. In the ViT a missing modality is "filled with background intensities". The normalisation is not stated, so the fill value is not known (M3AE, 3.14, shows the fill value can matter).
- **Patterns and metrics**: 5 (complete input and the four drop one patterns). Dice only. No test, one run.
- **Result (frozen encoder, internal test)**: complete input WT 82.78 against 80.40. FLAIR missing, WT 16.69 against 5.29. T1 missing, WT 79.03 against 47.85.
- **Closeness score (full text)**: 1 (A).
- **Gap (section 5)**: does not narrow it: different architectures, so not a controlled comparison, and absolute Dice is low. It is another case where a channel stacked model trained on complete input fails without FLAIR (RQ1).

### 3.31 Zhu, Chen, Chen, Jiang, Zhou, Wang, Qin, Wang, Tian 2026 (AdaMM, "No modality left behind"), Medical Image Analysis 112, 104108. Doi `10.1016/j.media.2026.104108` (from candidates.csv)
- **Status**: read (arXiv `2509.15017` v2, the author accepted manuscript). All text and the BraTS 2024 Dice table read closely; the HD95, IoU, sensitivity and BraTS 2018 tables only skimmed.
- **Data and split**: BraTS 2024 (1,350), BraTS 2018 (285) and a metastasis set (200), each split 80/20 into train and test. **No validation set.** Crop to 160×192×168, flips, rotation, crops. 400 or 1,000 epochs, batch 1, Adam.
- **Backbone**: a 3D U-Net teacher on complete input and a student with one adapter per pattern, a graph module, distillation and a lesion presence head. Training draws the 15 patterns at random, missing inputs zero filled. The drop distribution is not given.
- **Arms**: 16 published methods in six families, on the same split. The paper calls this "a systematic evaluation of six categories". It does not say whether each baseline was retrained, with what settings, or with which channel order.
- **Patterns and metrics**: all 15. Dice, HD95, IoU, sensitivity per region. No test, no seeds. The "Std." column is the spread over the 15 patterns, not over runs.
- **Result (BraTS 2024, means over 15)**: AdaMM 83.90, 64.49, 63.54 (WT, TC, ET). Complete input 88.69, 78.58, 79.32.
- **Problems found (my reading and arithmetic)**:
  - The pattern columns of several baselines look mislabelled. Not confirmed.
    - ACN, SMUNet and MMCformer: ET is high wherever T1 is present and low wherever it is not, as if T1 and T1c were swapped. ACN T1 alone 59.51 against T1c alone 21.23; MMCformer FLAIR+T1+T2 62.16 against FLAIR+T1c+T2 28.65.
    - RA-HVED: ET follows T2 instead, as if T1c and T2 were swapped. T2 alone 53.54 against T1c alone 15.79; T1+T2 49.97 against T1+T1c 18.06.
    - RMBTS and XLSTM-HVED fit no single swap: RMBTS FLAIR+T1c+T2 (no T1) 59.68, XLSTM-HVED T1+T1c+T2 17.03 against FLAIR+T1+T2 52.96.
  - The ablation "w/o GARM" ET row has 40.89 for T1 plus T1c, where its neighbours are about 76. The printed mean of 60.29 fits a value near 77.3, so the cell is likely a typo.
  - LCKD WT mean printed 80.15, rows give 80.35. Some Std. values use the population formula, some the sample one.
  - The threshold τ and the loss weights are chosen by tables on the 80/20 test split, since there is no validation split.
- **Closeness score (full text)**: 3 (A, B, F).
- **Gap (section 5)**: does not narrow it, and it is **not the benchmark section 6 asks for**. It compares published methods, not training strategies on one backbone. It has no plain zero fill or plain dropout arm, no validation split, no seeds and no paired test.

## 4. Record of this search

- 16 logged searches: the 8 fixed queries, then 8 title searches for web finds and for known papers. 387 candidates.
- Added title searches, reason: web search found them or a read paper pointed to them: incomplete imaging data, modality redundancy, targeted FLAIR dropout, ModDrop++, synthetic replacement of missing sequences, BraSyn, HeMIS, CAPS.
- Web searches in this round: 5.
- Second round (2026-11-07): 9 more title searches for the section 6 list, so 25 logged searches and 651 candidates. No new web searches. 13 papers requested, 11 downloaded, 6 read. The 6 were taken from the section 6 list, not by a formal abstract score. RFNet, Qiu 2025 and the journal version of HeMIS have no open copy through the tool. The titles IM-Fuse and Neverova ModDrop did not surface. Two reviews failed to download (`W4410086815`, `W4407980026`).

- Third round (2026-11-21): 10 more logged searches, so 35 logged searches and 698 candidates. Two log lines are not useful: "Modality balancing incomplete multimodal brain tumor segmentation" is logged twice (a retry), and the bare query "IM-Fuse" (74,250 hits) is too broad, although it did surface IM-Fuse (`W4414351039`). The ModDrop search used `--from-year 2014`.
- OpenAlex answered HTTP 429 (too many requests) on many tries, likely the shared free daily budget in ADR 0003. These title searches failed and are **not** in the log: M2FTrans (three tries), "PMR prototypical modal rebalance", the RFNet title, and "Feng modality dropout multiple sclerosis". A free OpenAlex key (ADR 0003, set by the human through an environment variable) would fix this.
- arXiv was searched by title through its public API, by hand, not logged in `search-log.csv` because the tool logs OpenAlex only: M2FTrans, IM-Fuse and "Does Adding a Modality Really" have no arXiv match; RFNet no match; SMIL is `2103.05677`; PMR is `2211.07089`; "ModDrop" with "sclerosis" finds only ModDrop++; "benchmark missing segmentation" finds `2503.18445`, which is RGB-D scene segmentation, out of scope.
- 15 papers requested through the tool: 9 downloaded, 4 `not_pdf` (SMIL `W3176404283`, IM-Fuse `W4414351039`, `W4415016743`, `W7125253548`), 2 `failed` (`W7213990777`, `W4360614210`). RFNet was read from the CVF open access site, outside the tool (note 3.25); its row still says `no_open_copy`. KD-Net was read through its arXiv row (`W3169291543`), its DOI taken from the venue row (`W3090336986`).
- Web searches in this round: 0.
- 10 read: 3.22 to 3.31.

**Departures in this round, stated plainly:**
- The ten were picked from the section 6 list and by title for closeness to the gap, not by a formal abstract score.
- SMIL and PMR have open arXiv copies but are general multimodal methods, not brain tumour MRI, so section 1 puts them out. They were not read.
- ModDrop (3.24) is counted in the 31 as background, the same way ModDrop++ (3.6, multiple sclerosis) was counted before. Without it, 30 papers are in scope.

**Departures from section 1, stated plainly (rounds one and two):**
- Not all 387 were screened. Titles of the first 322 were filtered by keyword, which left 113, and I read those titles. A paper with an unusual title could be missed.
- The 10 were chosen from titles, abstracts and web results, not a formal abstract score across all candidates. After reading, Grøvik scored 0 and Pemberton, Ruffle and LASSNet scored 2. So "the 10 closest" is not strictly true. Better candidates may be in the next list.
- `candidates.csv` shows `not_pdf` for Cai and `failed` for Ruffle, because the script could not fetch them. Both were read through the routes in their notes.

## 5. What is left of our gap (not decided)

These are candidates for `/architect`. Nothing here changes the protocol, and any addition must be decided before the test opening.

| Point | Cai et al. | Ours |
|---|---|---|
| Data | BraTS 2021 | BraTS 2023 GLI |
| Input | whole volume at 96×96×64 | 128³ |
| Budget | up to 8 epochs, about 1,500 steps (my arithmetic) | 40,000 iterations |
| Model | about 5.6 M parameters | about 12.9 M parameters |
| Augmentation | none | flips and intensity |
| Zero fill arm | 8 channels, presence kept | plain 4 channel model |
| Regions in primary test | mean of WT, TC, ET | WT, TC, ET separately |
| Test | Wilcoxon not used, no multiplicity correction | paired Wilcoxon, Holm, bootstrap CI |
| Test set use | sealed for the first comparison, then reused | one joint opening for all arms |
| Seeds | 4 for the main pair, 1 for presence | 1 now (seed 42) |

What this means:
1. **The main claim must change.** From "first comparison" to "a full resolution, full budget, sealed test confirmation of the comparison, and a test of whether the complete input cost survives".
2. **Seeds matter more now.** On seeds we are weaker than Cai today. Seeds 43 and 44 move from nice to have towards needed.
3. **An exposure arm is a possible addition.** Cai's 50% complete input result is one seed and internal only. A new arm costs about 8 h of training per seed, plus sweeps.
4. **An independent dataset has precedent.** BraTS-Africa was used by Cai. Our overlap rule still applies: it is not "external" until independence is shown in `data/DATA.md`.

Checked 2026-11-21 against papers 22 to 31: none of them narrows this gap (see "Update after 31 papers" in section 2). The one with and without dropout comparison at complete input (ModDrop, 3.24) is out of domain, one run, and at about ten times our complete input exposure.

## 6. Next

31 of at least 50 papers read, 30 in scope plus ModDrop as background (papers 22 to 31 read 2026-11-21). Read next:
- Downloaded, not read: IM-Fuse (`W4414351039`), fetched 2026-11-21 from the MICCAI 2025 open access site (`papers.miccai.org/miccai-2025/paper/0747_paper.pdf`), title checked in the PDF, saved as `~/literature-pdfs/W4414351039.pdf` with a text copy. Its candidates.csv row still says `not_pdf`.
- **For the human (no open copy found)**:
  - Qiu et al. 2025, "Does Adding a Modality Really Make Positive Impacts in Incomplete Multi-Modal Brain Tumor Segmentation?", TMI, `W4406137843`, doi `10.1109/tmi.2025.3526818`. Not on arXiv by title; a web search (2026-11-21) found only abstract pages, no open copy. The closest title to our question so far.
  - PMKL, "Learning With Privileged Multimodal Knowledge for Unimodal Segmentation", TMI, `W3206034503`, doi `10.1109/tmi.2021.3119385`. A web search (2026-11-21) found only the code repository and abstract pages, no open copy.
  - M2FTrans: not in `candidates.csv` (the title search failed with 429) and no arXiv match. AdaMM's reference list gives it as Shi, Yu, Cheng, Yang, Cheng, Yan, "M2FTrans: Modality-masked fusion transformer for incomplete multi-modality brain tumor segmentation", IEEE Journal of Biomedical and Health Informatics 28 (2023) 379 to 390 (not checked; no DOI taken from it). Retry the title search first; ask only if it still has no open copy.
  - Feng (ModDrop in MS): not found by a guessed title. The human should give the full title or authors.
- Out of scope, open on arXiv, read only if a strategy paper relies on them: SMIL (`2103.05677`), PMR (`2211.07089`).
- Open and in scope, not read yet (picked by title, score them from the abstract first): MedMAP / "Mind the Gap" (`W4413360068`, `W4406262216`), Multimodal MAE pretraining for missing modalities (`W4410081345`), MGML (`W7117236435`), CCSD (`W7160394731`), Hölder divergence (`W4416895193`), "Brain Tumor Segmentation for Multi-Modal MRI with Missing Information" (`W4381326261`), Scalable Swin (`W4391469551`), integrated glioma framework (`W4415466521`).
- Reviews: Zhou 2023 survey (`W4313340471`, closed), and the two failed downloads (`W4410086815`, `W4407980026`).
- Forward citations from Cai et al. and Öchsner et al. The tool has no citation search yet, so this needs a small addition or a web route.
- A real benchmark paper is still needed. Neither 3.15 nor 3.31 is one.
- OpenAlex limited this round with 429 errors. A free key (ADR 0003) would let the failed title searches run.
