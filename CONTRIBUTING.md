# Contributing

## Before you start
Read `context/project-overview.md`, `AGENTS.md` and `context/progress-tracker.md`.

## Setup
    git config core.hooksPath .githooks     # enable the pre-commit hook
    # Python environment: created in milestone 2

## Branches
`main` is the last good state and is never committed to directly (the bootstrap commit is the one exception). Names: `feat/<scope>-<desc>`, `exp/<name>`, `write/<chapter>`, `fix/...`, `chore/...`, `docs/...`. One branch per unit of work.

## Commits
Conventional Commits: `<type>(<scope>): <imperative summary>`.
- Types: `feat fix refactor docs test chore perf build ci`
- Scopes: `exp`, `data`, `model`, `eval`, `lit`, `writing`, `ci`
- Lower case, imperative, under 72 characters, no trailing full stop. The body explains why.
- Examples: `feat(exp): add modality dropout training`, `docs(lit): add note on missing-modality methods`, `fix(data): stop test subjects leaking into train`.

## Pull request checklist
- [ ] Spec exists and is linked; `verify.md` ticked
- [ ] Tests pass
- [ ] `docs/whats-built.md` and `context/progress-tracker.md` updated
- [ ] No secrets, no MRI or raw data committed
- [ ] Every new citation has a literature note and is marked verified or listed "to verify"

## Secrets
`.env` is ignored. `.env.example` lists names only.
