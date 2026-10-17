# Literature index

**Status: no paper has been read or verified yet.** The 19 entries in `writing/refs.bib` were written from the AI's memory. Treat every one as "to verify" until a note exists here and the `researcher` agent has checked DOI/arXiv, authors, year, venue and the specific claim used.

## To verify

| Key | Used for | Note | Verified |
|---|---|---|---|
| menze2015 | BraTS benchmark, labels | none | unverified |
| bakas2017 | BraTS TCGA labels | none | unverified |
| bakas2018 | BraTS review | none | unverified |
| baid2021 | BraTS 2021 (cited for BraTS releases) | none | unverified |
| antonelli2022 | Medical Segmentation Decathlon | none | unverified |
| ronneberger2015 | U-Net | none | unverified |
| cicek2016 | 3D U-Net | none | unverified |
| isensee2021 | nnU-Net baseline reference | none | unverified |
| havaei2016 | HeMIS, any-subset segmentation | none | unverified |
| dorent2019 | Joint completion and segmentation | none | unverified (check authors, venue) |
| chen2019 | Disentanglement and gated fusion | none | unverified (check authors, venue) |
| ding2021 | RFNet, incomplete modalities | none | unverified (check authors, venue) |
| zhang2022 | mmFormer | none | unverified (check author list) |
| neverova2016 | ModDrop, modality dropout | none | unverified |
| zhou2021 | Models Genesis, self-supervised | none | unverified |
| tang2022 | Self-supervised Swin, 3D | none | unverified |
| taha2015 | Segmentation metrics | none | unverified |
| maierhein2024 | Metrics Reloaded | none | unverified |
| cardoso2022 | MONAI | none | unverified |

Also check: the official BraTS 2023 GLI citation list and data-use terms. The proposal cites the 2015, 2017, 2018 and 2021 papers; the official 2023 list may differ. The draft's claim about the Tsinghua BIRTH Lab AdaMM paper is **not cited** because it could not be confirmed.

## Gap
Working claim, **not confirmed**: many incomplete-modality methods propose new architectures, while fewer studies compare simple training strategies (zero-fill, modality dropout, presence indicators) on the same subject-level splits, all 15 modality subsets and paired statistics under a small-GPU budget. The review must test this against real papers and rewrite the claim if prior work already covers it.
