# What's built

**Last updated:** 2026-10-17, milestone A and proposal draft
**Last full check:** only git and hook checks. Not run: CI, any experiment (none exist), citation verification.

| # | Feature | Status | What it gives you |
|---|---|---|---|
| A | Repo and AI setup | done | Rules, skills, docs layout, pre-commit hook |
| B | Proposal | built, not supervisor-reviewed | `writing/proposal/proposal.pdf`, 7 pages |

## A. Repo and AI setup
Any AI chat reads `AGENTS.md` and starts with the same rules. Enable the hook with `git config core.hooksPath .githooks`.

## B. Proposal
Build: `cd writing/proposal && latexmk -xelatex proposal.tex`. Citations are unverified.
