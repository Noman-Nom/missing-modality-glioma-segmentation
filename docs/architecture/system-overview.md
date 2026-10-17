# System overview

## Pipeline in words
Case (T1, T1ce, T2, FLAIR + labels, channel order mapped per dataset) -> audit and fixed subject split -> normalise and crop -> missing-modality simulation (pattern applied to channels) -> train compact 3D U-Net on random patches (zero-fill / modality dropout / presence indicator) -> sliding-window inference -> evaluation per pattern (Dice, HD95, sensitivity, precision for WT, TC, ET) -> results files -> thesis tables.

## Threats to validity (running list)
- MSD and BraTS overlap; cross-dataset results may not be independent.
- Label and channel order differ between releases; a wrong mapping invalidates everything.
- Compact model on small patches may segment poorly, which limits robustness conclusions.
- No public masks for BraTS validation; internal splits only.
- Multiple seeds depend on compute budget.
- Tuning only on validation data; test set used once.
