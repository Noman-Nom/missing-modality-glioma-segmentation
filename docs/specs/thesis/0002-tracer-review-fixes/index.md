# 0002. Tracer review fixes: lock pandas, clean lint, one checkpoint rule, strict IDs and labels

**Date**: 2026-10-25
**Status**: In Progress

## Summary

A fresh model review of the tracer bullet ([review](../../../reviews/2026-10-25-feat-tracer-bullet.md)) found that the split scripts cannot run in the locked environment (pandas is missing), the lint gate fails, and a few shared helpers fail silently on bad input. This spec fixes all of it before E1 builds on `experiments/common/`. It adds pandas to the locked environment, makes ruff clean, settles that only the final checkpoint (`last.pt`) counts, makes bad labels and bad case IDs raise an error, and writes down one rule for how E numbers map to folders. It also closes scope feature 3 (the subject leakage test), because the same scripts are touched.

## Requirements

**User stories**:
- As the student, I want the split and audit scripts to run with `uv run` in the locked environment, so that `splits.csv` can be regenerated and checked by anyone with the data.
- As the student, I want shared helpers to fail loudly on bad labels or IDs, so that a data problem never turns into a silently wrong Dice.
- As the examiner, I want one stated rule for which checkpoint is reported and how E numbers map to folders, so that every table and doc agrees.

**Acceptance criteria**:
- **AC-1**: `pandas` is in `[project] dependencies` in `pyproject.toml` and in `uv.lock`; `uv run python -c "import pandas"` succeeds. ADR 0001 (`docs/architecture/decisions/0001-python-environment.md`) records pandas with today's date and the reason (the audit and split scripts need it) in its direct dependencies line (line 72), the same way scipy is recorded, and its stale "`scipy` was not installed" sentence (line 149) is corrected.
- **AC-2**: The input is `results/0000-data-audit/cases.csv` (gitignored); if it is missing, the human first reruns the audit (`uv run python experiments/0000-data-audit/audit.py`, command printed by the AI). The human then runs `uv run python experiments/0001-splits/make_splits.py --out <tmp>/splits.csv` and `uv run python experiments/0001-splits/check_splits.py --compare <tmp>/splits.csv`. `--compare` reads both files with `subject` as text, sorts by `case`, and passes only when the `case` lists and the `fold` per case are identical. The committed `data/splits/splits.csv` is never overwritten by this check. On a match, `data/DATA.md` records the SHA256 of the committed file and that it was reproduced under the locked environment (date, pandas version from `uv run`), and its Regenerate and Check commands use `uv run`. On a mismatch, the build stops and reports it (a `/debug` item); nothing is overwritten.
- **AC-3**: `pyproject.toml` has `[tool.ruff.lint] ignore = ["EXE002"]` with a one line comment naming the NTFS executable bit (ADR 0001). `uv run ruff check .` exits 0. The 13 other errors are fixed in code, not ignored, except one `# noqa: BLE001` on the `except Exception` in `audit.py` with a reason comment (the audit must record any unreadable file and continue).
- **AC-4**: `train.py` saves only `checkpoints/last.pt` (no `best.pt`) and `run.json` has no `best_iteration` key (removed from `RUN_KEYS` in `experiments/common/runlog.py`). The val WT Dice history stays in `run.json` as a learning curve. `evaluate.py` evaluates `last.pt` only, and its docstring states the rule: a fixed iteration budget, report the final weights, no checkpoint selection.
- **AC-4b**: `check_tracer.py` accepts a run whose `run.json` keys equal `RUN_KEYS`, or equal `RUN_KEYS` plus the one named legacy key `best_iteration` (constant `LEGACY_KEYS = {"best_iteration"}`), so tracer runs made before this change still pass. Any other missing or extra key still fails. A fast test covers a legacy record (passes) and one with an unknown extra key (fails).
- **AC-5**: `evaluate.py` compares the `config_hash` stored in `last.pt` with `config_hash` in `run.json` and exits 1 with a message naming both hashes when they differ, before any evaluation.
- **AC-6**: `load_case()` raises `ValueError` before the int16 cast when the label volume holds a value outside `ALLOWED_LABELS = {0, 1, 2, 3}` or a non integer value. The integer check is by value (`label == np.round(label)` everywhere), not by dtype, because a scaled NIfTI loads integer labels as float. The message names the case and the sorted bad values. A fast test covers a stray 4, a 1.5, and a float volume holding only 0.0 to 3.0 (passes).
- **AC-7**: One BraTS ID pattern, `BRATS_ID_RE = re.compile(r"^BraTS-GLI-(\d{5})-(\d{3})$")`, lives in `experiments/common/data.py`. `subject_of()` uses it and raises `ValueError` naming the string when it does not match. `audit.py` imports it instead of defining its own. A fast test covers a valid ID, an MSD style name (`BRATS_001`) and a malformed string.
- **AC-8**: In `train.py`, inside the existing `try` that wraps training, an `except Exception` clause placed after the out of memory and non finite loss clauses (and before `finally`) sets status `failed` with reason `"<ExceptionType>: <first line of message>"` (just the type when the message is empty), then raises the exception again (traceback shown, nonzero exit). The `finally` writes the record, and the closing status line is printed inside `finally`, so it appears on this path too. Ctrl+C keeps its current status and exit code. Errors before the `try` (input check, run folder, config dump) happen before `run.json` exists and stay plain errors, as in spec 0001. A fast test injects such an error and checks the status and reason.
- **AC-9**: `make_table.py` replaces the single empty rule count with two counts per region, headed `Both empty` (`gt_empty` and `pred_empty` both `"True"`, scored Dice 1) and `One empty` (exactly one `"True"`, scored Dice 0), compared as CSV strings as today. The table has 15 rows and, per region, Dice mean, Dice median, HD95 mean, HD95 median, Both empty, One empty: the column spec becomes `rl` + `rrrrrr` per region, `\multicolumn{6}`, and `\cmidrule` spans `3+6i` to `8+6i`. Its test checks both counts on a toy `metrics.csv`. The human rebuilds `writing/tracer/tracer.pdf` with `latexmk -xelatex`, and it builds without overfull warnings on the table.
- **AC-10**: No script in `experiments/` runs code on import: `check_splits.py` and `check_audit.py` call `main()` under `if __name__ == "__main__":`. `check_splits.py` takes the expected case count from a named constant `N_LABELLED_CASES = 1251` with a comment pointing to `data/DATA.md`.
- **AC-11**: Fast tests in `tests/test_splits.py` (closes scope feature 3), no data and no `results/` folder needed. `make_splits.py` is loaded by file path with `importlib` (its folder name starts with a digit). (a) The 6 subject toy now inside `check_splits.py` moves into this test: written to `tmp_path` as `cases.csv` and passed to `make()`, it keeps every subject in one fold and gives an equal frame on two calls; (b) the committed `data/splits/splits.csv`, read with `subject` as text, has 1251 unique cases (the number written in the test, with a comment pointing to `N_LABELLED_CASES`), every subject in exactly one fold, and disjoint train, val and test subject sets.
- **AC-12**: Two `train.py` runs with the same config started in the same second get different run folders: on a name collision the run id gets a suffix `-1`, `-2`, and so on (the parent folder is created once with `parents=True, exist_ok=True`, then the run folder with `mkdir(exist_ok=False)` in a loop). The suffix is part of `run_id` itself, so `run_id == run_dir.name` in `run.json`, `metrics.csv` and the table. A fast test creates the colliding folder first and checks the suffix and that `run_id` matches the folder name.
- **AC-13**: One E number rule is used everywhere. `experiments/AGENTS.md` (line 35) states that folder numbers are build order and E numbers (E0 to E7) are thesis labels, and holds the one mapping table (E0 → `0000-data-audit`; E1 to E7 → folder added when built; `0001-splits` and `0002-tracer-bullet` are infrastructure with no E number). `docs/whats-built.md` (lines 11 and 24) and the `make_splits.py` docstring no longer call the split "E1". `grep -rn "map to the numbers" experiments docs context` finds nothing.
- **AC-14**: Cleanups: the stray double blank line in `experiments/common/model.py` is gone, and `select_subjects()` has a docstring line saying that "first N by sorted ID" is deterministic but not random, to revisit when sizing E1.
- **AC-15**: `uv run pytest -q` passes with no data and no GPU. Spec 0001 carries an "Amended by 0002" note (AC-4, AC-9, AC-12 of 0001 change), and `docs/whats-built.md` and `context/progress-tracker.md` are updated in the same change.

## Decision

**Chosen option**: Option 1 + Option 2 + Option 3 + Option 4 as above, plus strict label and ID checks, a `failed` status for any crash, a split empty count, import safe scripts, split leakage tests and collision safe run ids.

Fix the tracer in place (no new modules beyond `tests/test_splits.py`): lock pandas, ignore only `EXE002`, report only `last.pt`, raise on bad labels and IDs, and record one E number mapping in `experiments/AGENTS.md`.

**Implementation skills**: `pytorch-patterns` (`affaan-m/ecc`, `.claude/skills/pytorch-patterns/`) · `python-testing-patterns` (`wshobson/agents`, `.claude/skills/python-testing-patterns/`) · `modern-python` (`trailofbits/skills`, `.claude/skills/modern-python/`)

## Rationale

Reasoning, context and options: see [rationale.md](rationale.md). Checks: [verify.md](verify.md).

## Feature design

**Data model sketch** (files, no database):
- `pyproject.toml`: `dependencies` gains `pandas>=<version uv resolves>`; new `[tool.ruff.lint] ignore = ["EXE002"]`.
- `run.json`: key `best_iteration` removed; all other keys of spec 0001 unchanged; status values unchanged (`running`, `ok`, `failed`, `interrupted`).
- `checkpoints/`: only `last.pt` (state dict, iteration, config hash).
- `writing/tables/0002-tracer-bullet.tex`: per region 6 columns (Dice mean, Dice median, HD95 mean, HD95 median, both empty, one empty) instead of 5.
- `experiments/common/data.py`: new constants `ALLOWED_LABELS = frozenset({0, 1, 2, 3})` and `BRATS_ID_RE`.

**State transitions** (`run.json` status in `train.py`): `running` → `ok` (finished) · `running` → `failed` (OOM, non finite loss, **any other exception** (new)) · `running` → `interrupted` (Ctrl+C). No path leaves `running` as the final status.

**Interface surface**:
| Entry point | Inputs | Outputs | Errors |
|---|---|---|---|
| `make_splits.py` | `--cases` (default `results/0000-data-audit/cases.csv`), `--out` (default `data/splits/splits.csv`) | splits CSV | missing input file |
| `check_splits.py` | optional `--compare <csv>` (regenerated split to compare with the committed one) | `OK ...` line, exit 0 | AssertionError, exit 1 |
| `check_tracer.py <run dir>` | run folder | exit 0 on pass | exit 1 on key mismatch other than `LEGACY_KEYS` |
| `load_case(paths)` | case paths | image, label, pixdim | `ValueError` on labels outside `ALLOWED_LABELS` or non integer labels |
| `subject_of(case)` | case ID string | 5 digit subject string | `ValueError` on no match with `BRATS_ID_RE` |
| `train.py` | config, `--out`, `--quick`, `--device` | run folder with `last.pt` only | any error writes `failed`, raises again |
| `evaluate.py <run dir>` | run folder | `metrics.csv` | exit 1 on config hash mismatch, status not ok, or missing `last.pt`; exit 2 on missing inputs |
| `make_table.py <run dir>` | `metrics.csv`, `run.json` | `.tex` table | as before |

**Value sourcing**:
| Action | Value | Source |
|---|---|---|
| `load_case` check | allowed label set | `ALLOWED_LABELS` in `common/data.py`, from the BraTS label spec in spec 0001 (AC-2 of 0001) |
| `load_case` error | case name | the case folder name from `paths` |
| `subject_of` | subject ID | group 1 of `BRATS_ID_RE` |
| `check_splits` | expected case count | `N_LABELLED_CASES = 1251`, documented in `data/DATA.md` |
| AC-2 check | regenerated split | `make_splits.py --cases results/0000-data-audit/cases.csv --out <tmp>/splits.csv` (audit rerun first if the input is missing) |
| `check_tracer` | allowed old keys | `LEGACY_KEYS = {"best_iteration"}` in `check_tracer.py` |
| AC-2 check | committed split hash | `sha256sum data/splits/splits.csv`, written into `data/DATA.md` |
| AC-2 check | pandas version | `uv run python -c "import pandas; print(pandas.__version__)"` |
| `evaluate` hash check | expected and stored hash | `run.json["config_hash"]` and `torch.load(last.pt)["config_hash"]` |
| failed reason | exception type and message | `type(e).__name__` and `str(e).splitlines()[0]` (empty message gives just the type) |
| table counts | both empty, one empty | `gt_empty` and `pred_empty` columns of `metrics.csv` |
| run id suffix | collision counter | first free `-N` found by `mkdir(exist_ok=False)` |
| E number mapping | which folder is which E | the one table in `experiments/AGENTS.md`; E labels from `context/project-overview.md` |

**Key invariants**:
- The committed `data/splits/splits.csv` is never overwritten by any check.
- No label value outside `ALLOWED_LABELS` reaches `label_to_regions()`.
- Every finished `train.py` process leaves `run.json` with a final status other than `running` (except a hard kill, which `check_tracer` already treats as not ok).
- `evaluate.py` never scores a checkpoint whose config hash differs from its run.
- Importing any module under `experiments/` runs no checks and reads no files.

**Security model**: no change. Only case IDs and fold labels are committed; no MRI data enters git.

**Critical test scenarios**:
- Stray label 4 and label 1.5 raise `ValueError` naming the case, verifies **AC-6**.
- `subject_of("BRATS_001")` and a malformed string raise; a valid ID returns its 5 digit subject, verifies **AC-7**.
- An injected `RuntimeError` in training gives `run.json` status `failed`, reason `RuntimeError: ...`, nonzero exit, verifies **AC-8**.
- Toy `metrics.csv` with one both empty row and one one sided row gives counts 1 and 1, verifies **AC-9**.
- A `last.pt` with a different config hash makes `evaluate.py` exit 1 before scoring, verifies **AC-5**.
- Toy split keeps subjects together; committed split has no subject in two folds, verifies **AC-11**.
- Pre created run folder forces a `-1` suffix, verifies **AC-12**.
- `uv run ruff check .` exits 0, verifies **AC-3**.

## Build plan

Tracer Bullet here means: first make the thread runnable again end to end in the locked environment (deps, lint, splits), then harden the shared code it runs through, then align the docs.

1. `uv add pandas`; record it in ADR 0001; switch `data/DATA.md` commands to `uv run`, satisfies **AC-1**
2. Add `__main__` guards to `check_splits.py` and `check_audit.py`; add `N_LABELLED_CASES` and the `--compare` option to `check_splits.py`, satisfies **AC-10**, **AC-2**
3. Add `tests/test_splits.py` (toy moved from `check_splits.py`, committed split test), satisfies **AC-11**
4. Print for the human: rerun the audit if `cases.csv` is missing, regenerate to a temp path, run `check_splits.py --compare`; on a match record SHA256 and pandas version in `data/DATA.md`, satisfies **AC-2**
5. Add the `EXE002` ignore; fix the 13 remaining ruff errors by hand; `noqa` with reason in `audit.py`, satisfies **AC-3**
6. `ALLOWED_LABELS` check in `load_case()` plus tests, satisfies **AC-6**
7. `BRATS_ID_RE` in `common/data.py`, used by `subject_of()` and `audit.py`, plus tests, satisfies **AC-7**
8. `train.py`: drop `best.pt` and `best_iteration` (also in `common/runlog.py`, `tests/test_common_runlog.py` and `tests/test_tracer_failures.py:82`), catch all to `failed` then raise again, run id suffix, plus tests, satisfies **AC-4**, **AC-8**, **AC-12**
8b. `check_tracer.py`: `LEGACY_KEYS` tolerance plus tests, satisfies **AC-4b**
9. `evaluate.py`: config hash check and the checkpoint rule docstring, plus test, satisfies **AC-4**, **AC-5**
10. `make_table.py`: both empty and one empty columns, update its test, rebuild `tracer.pdf`, satisfies **AC-9**
11. Cleanups in `model.py` and the `select_subjects()` docstring, satisfies **AC-14**
12. E number table in `experiments/AGENTS.md`; fix `docs/whats-built.md` and the `make_splits.py` docstring, satisfies **AC-13**
13. "Amended by 0002" note in spec 0001; update `docs/whats-built.md`, `context/progress-tracker.md`; `uv run pytest -q` and `uv run ruff check .` green, satisfies **AC-15**

## Consequences

**Positive**:
- The split can be regenerated and checked under `uv run`, and a fast test guards it with no data.
- Bad labels, bad IDs and crashed runs are visible instead of silent.
- One checkpoint rule and one E number table remove two sources of mismatch between code and docs.

**Negative / tradeoffs**:
- pandas (and its dependencies) join the locked environment, which the GPU training path never uses.
- With `EXE002` off, a real missing shebang would go unnoticed (no script here relies on one).
- Reporting `last.pt` means a run that overfits late reports its worse final weights; the val Dice history in `run.json` shows when that happens.
- The tracer table gets wider (20 columns); it still fits the landscape page, but the E2 table may need a different layout.
- Existing tracer run folders still contain `best.pt` and `best_iteration`; they are left as they are (old records), and `check_tracer.py` carries one named legacy key (AC-4b) so they still pass. That exception is a small permanent wart in the checker.

**Neutral**:
- Spec 0001's AC-9 (run.json keys), AC-12 (table columns) and checkpoint row change; 0001 stays `Accepted` with an amendment note.
- Scope feature 3 closes with this feature.

## Follow-up

- [ ] If AC-2 finds a split mismatch, open `/debug` on `make_splits.py` before any E1 work.
- [ ] MSD Task01 needs its own ID parser and label set (scope feature 4); `subject_of()` stays BraTS only until then.
- [ ] Decide the E2 table layout (`both empty` and `one empty` may move to a separate table) when E2 is designed.
- [ ] pandas Agent Skill discovery declined (2026-10-25); `/sync` may record the decline in `AGENTS.md`.
- [ ] `/sync` adds pandas to the Stack line in root `AGENTS.md`.
