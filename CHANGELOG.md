# Changelog

Format: Keep a Changelog. One entry per milestone.

## [Unreleased]

### Added
- Repository bootstrap: AGENTS.md system, workflow skills, docs structure, pre-commit hook, CI file.
- E2 missing modality evaluation: the frozen E1 model is scored on all 15 modality patterns for WT, TC and ET, with Dice, HD95, paired statistics, four thesis tables and a run check in `check_eval.py` (see spec 0004).
- Sealed test path for E2: the test fold stays closed behind a code gate until the single joint opening under feature 7. The gate is proven on synthetic cases only, so no test number exists yet (see spec 0004).
- E2 val results from the Kaggle T4 run at `e0c6de1` (114 subjects, seed 42, `check_eval.py` PASS). Mean Dice with all four sequences: WT 0.916, TC 0.885, ET 0.845. Without T1ce: TC 0.422, ET 0.101. Without FLAIR: WT 0.598. These are val numbers from one trained model, not test results.
