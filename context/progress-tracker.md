# Progress Tracker
**Update this file in the same change that moves the work, not afterwards.**
**Current phase:** Proposal drafted, waiting on supervisor and missing facts
**Last updated:** 2026-10-25

## Done
### A. Repo and AI setup (2026-10-17)
- Repo, folder tree, AGENTS.md system, nine workflow skills, hook, CI file, docs skeleton.
- Not run: CI (needs a push), the skills (copied, untested here).

### Proposal draft, Option B journal style (2026-10-17)
- `writing/proposal/proposal.tex`, 7 pages (title block with side bar, 6 body pages incl. references), builds with XeLaTeX, no warnings.
- `writing/refs.bib` 19 entries, **all unverified**; `docs/literature/index.md` lists them.
- Open marker: roll number [to be provided]. Dataset counts, licences, ethics and schedule are stated as plans to be recorded by the data audit and data card.

## In progress
- Milestone 0 (data audit): full run done 2026-10-20 (1470 cases clean, findings in `data/DATA.md`). Split written (`data/splits/splits.csv`, check passes). Open: sha256, licence, view the 4 odd FLAIR cases, MSD overlap.
- Milestone 2, step 1: ADR `docs/architecture/decisions/0001-python-environment.md` written (Proposed). Human installed uv 0.13.2, torch 2.14.1+cu130, monai 1.6.1, nibabel 5.4.2; GPU check passed (sm_75 present). Step 2 done 2026-10-21: `tests/test_env_smoke.py` passes (human ran it). AC-3 peak 144 MiB of 4096; Dice, HD95 and NiBabel pass after `uv add scipy` (1.18.1). ADR 0001 AC-1..AC-5 met except the `uv sync --frozen` rebuild check on a clean checkout. Next: commit, then CI and AGENTS.md lines via `/sync`.

- Milestone 2, step 3: tracer bullet built (commit `7111e2c`, branch `feat/tracer-bullet`) and full run `20261025-000309-f4bbd0b9` passed `check_tracer.py` on 2026-10-25: 2,000 iterations, 385 s, peak reserved 484 MiB, pattern 1 WT Dice 0.903 (tracer number, not a result). Next: `/check verify tracer bullet`, `/sync`, then commit the docs and open a PR.

- Tracer review fixes (spec 0002, scope feature 13, also closes feature 3) built 2026-10-25 on `feat/tracer-bullet`: pandas 3.1.0 locked, `ruff` clean, `last.pt` only, strict labels and IDs, `failed` on any crash, run id suffix, two empty counts in the table, E number table in `experiments/AGENTS.md`, `tests/test_splits.py`. `uv run pytest -q` 84 passed. Human split reproduce check passed 2026-10-25 (SHA256 and pandas 3.1.0 in `data/DATA.md`); `tracer.pdf` builds with 0 overfull warnings. `/check verify` passed 2026-10-25; `/test` added 19 tests (split compare and case count guard, no script runs on import, shared ID pattern in the audit, label error names the case folder): `uv run pytest -q` 103 passed. Marked done 2026-10-25 (spec 0002 Accepted). Next: `/sync`, then `/architect E1 complete-input baseline`.

## Next
- Supervisor reads the proposal; fill roll number.
- Milestone 0: extract archives, inventory subjects, load a case from each dataset, confirm channel order, label values and alignment, record in `data/DATA.md`.
- Milestone 1: literature map (verify citations, confirm the gap).
- Milestone 2: tracer bullet (environment, tiny end to end run).

## Blocked / waiting on supervisor
- Thesis template, page limit, deadlines; roll number.
- BraTS and MSD data-use terms; ethics approval if needed.
