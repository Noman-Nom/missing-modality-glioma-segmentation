# .claude/

## Contents
- `skills/` nine workflow skills: scope, audit, architect, develop, check, test, document, sync, debug
- `agents/` `scout.md` (cheap read-only code scanner) and `researcher.md` (read-only web lookup; here it verifies citations)
- `SKILLS-LICENSE` MIT licence of the upstream skills

## Provenance
Copied from the thesis repo at `/media/noman/E0948A169489EF7A/Anit research/thesis/.claude/`, itself copied from `/media/noman/E0948A169489EF7A/school-saas/.claude/`, which vendored them from `jsmastery-pro/skills` (MIT, commit `43b69e4`, version 2.0.0 as recorded there). Copied on 2026-10-17.

## Rules
Never hand edit files in `skills/`. Project rules go in `AGENTS.md`.

## Updating
Clone upstream, run `diff -r` against this folder, review, then copy.
