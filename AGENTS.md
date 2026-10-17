# Segmenting Gliomas When Sequences Are Missing: A Controlled Comparison of Training Strategies for Incomplete Multimodal MRI

Master's thesis (NUST MISIS, Data Science 09.04.01; supervisor Kalitin Denis Vladimirovich; student Qarar Hussain). We train a compact 3D U-Net on BraTS 2023 GLI (and MSD Task01 for development) and measure how tumor segmentation degrades when MRI modalities are missing (RQ1 to RQ4 in `context/project-overview.md`). We compare missing-modality strategies fairly, and test label efficiency if time allows. Contribution: a controlled, reproducible, subject-level empirical study. Not a clinical tool.

## Stack

- **Language / runtime**: Python 3 with PyTorch and MONAI (versions pinned when the environment is created in milestone 2)
- **Environment**: uv with a committed lockfile (planned; not created yet)
- **Writing**: LaTeX, XeLaTeX with Times New Roman, biblatex + biber, British spelling (university template unknown, ask); citations in `writing/refs.bib`
- **Web demo**: not planned. If added, it gets its own nested AGENTS.md in `app/`.
- **Hardware**: Dell Precision 5550, Quadro T2000 Max-Q, 4 GB VRAM, Linux. Everything must fit this unless an ADR says otherwise.

## Build approach

Tracer Bullet: one thin thread through every layer (data, model, evaluation, writing), working end to end, then thicken it (see `docs/scope/thesis/scope.md`).

## Commands

    cd writing/proposal && latexmk -xelatex proposal.tex   # build the proposal PDF
    latexmk -C                                             # clean build files (same folder)
    # Experiments, tests, lint: not created yet (milestone 2 adds them here)

## Specs and decisions

Specs live in `docs/specs/thesis/NNNN-title/` (`index.md` is the build spec, `verify.md` its checks, `rationale.md` the reasoning). Read the relevant ADR in `docs/architecture/decisions/` before choosing a dataset, method, metric, split or tool. Anything that must outlive one run (a long training job, a download) is a documented step with its own script, never something done by hand in a notebook.

## Rules

- **Reproducibility.** Every number, table and figure in the thesis is produced by a script in `experiments/` with a fixed seed and a logged config. No manual copy of results.
- **Raw data is read only.** `data/raw/` is never edited. Cleaning and splitting are code.
- **Medical data.** No MRI volumes, masks, or derived patient images in git. Use data only under its terms, recorded in `data/DATA.md`.
- **Dataset overlap.** MSD Task01 and BraTS releases may share cases. Never call a cross-dataset test "external validation" unless independence is shown in `data/DATA.md`.
- **Labels and channel order are per dataset.** Explicit mapping from the dataset spec, checked by a test. Never assume IDs or modality order match across datasets.
- **Never invent citations.** Cite a source only if a note exists in `docs/literature/` and the entry is in `writing/refs.bib`. If unsure write `[CITE?]`. All entries are **unverified** until marked verified in `docs/literature/index.md`.
- **Never fabricate or round up results.** If a run did not happen or failed, say so.
- **No data leakage.** Splits are fixed by code (subject level, never slice or patch) and checked by a test.
- **The evaluation protocol is fixed before test results are seen.** Missing-modality patterns are defined in advance. Thresholds and model selection use validation data only.
- **Fair comparison.** Strategies are compared on the same subjects, splits, metrics and seeds. Claim a benefit only if a paired comparison supports it.
- **No new infrastructure, framework, dataset or paid service without an ADR first.** Propose, do not install.
- **A new experiment is a new folder** under `experiments/`. Experiments share code only through `experiments/common/`.
- **Every experiment ships a check** (shapes, determinism with the same seed, no leakage, a metric on a toy case with a known answer).
- **Each claim in a chapter maps to a research question.** Flag text that answers none.
- Never commit secrets, API keys, personal data or large raw datasets.

## Working rules for the AI

- **Never poll a long command.** Do not run a full training or long test suite and wait. Run fast gates only. For anything slow, print the command and ask the human to run it and paste the last two lines.
- **Read narrowly.** Prefer `grep`, `sed -n` and `tail`. Never print a log longer than about 20 lines.
- **One milestone per chat.** Finish, report, stop.
- **Announce before building.** Name every file or chapter you will create before you create it.
- **Explain with examples.** State the rule, then show it on real names, files and numbers.
- **Functionality first.** Right logic and result first, polish later.
- **Keep `docs/whats-built.md` and `context/progress-tracker.md` current** in the same change as the work.
- **Parallel chats share one checkout.** Always `git add` explicit paths, never `git add .`.
- When a fact, number or source is uncertain, say so and ask. Never guess.
- Plain language: short sentences, no jargon without a one line explanation.

## Git

- integration: on
- branch prefix: `feat/` (also `exp/`, `write/`, `fix/`, `docs/`, `chore/`)
- commit: per milestone, Conventional Commits (see CONTRIBUTING.md)
- The human pushes and opens pull requests. The AI prints the commands and a paste-ready PR summary and does not run them.

## Tooling

- **Installed**: git, latexmk. A pre-commit hook in `.githooks/` blocks secrets, MRI data and large files (enable with `git config core.hooksPath .githooks`; already set in this clone).
- **Planned**: uv, ruff, pytest. CI in `.github/workflows/ci.yml` builds writing and runs lint and tests when they exist.
- Testing convention: pytest, fast tests in `tests/`, slow tests marked `slow` and run by the human.

## Agent skills

Workflow skills live in `.claude/skills/` (nine: scope, audit, architect, develop, check, test, document, sync, debug). They are vendored from the thesis repo; do not hand edit them. See `.claude/README.md`.

## Context files

- [writing/AGENTS.md](writing/AGENTS.md) (writing conventions)
- [experiments/AGENTS.md](experiments/AGENTS.md) (experiment code conventions)
- [context/project-overview.md](context/project-overview.md) (problem, RQs, setup answers)
- [context/progress-tracker.md](context/progress-tracker.md) (done, in progress, next)
- [context/project-draft.md](context/project-draft.md) (the original A-to-Z draft, kept for reference)
- [data/DATA.md](data/DATA.md) (datasets: source, licence, hash)
