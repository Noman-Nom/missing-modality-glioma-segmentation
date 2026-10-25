# 0002. Tracer review fixes: rationale

Decision record for [index.md](index.md). `/develop` builds from `index.md` and skips this file.

## Context

The tracer bullet (spec [0001](../0001-tracer-bullet/index.md)) is done and verified, and every later experiment will reuse its shared code in `experiments/common/`. The review rated the scientific core correct but asked for changes in two major areas and several minor ones.

The major problems block reproducibility. `make_splits.py`, `check_splits.py`, `summarize.py` and `audit.py` import pandas, but `pyproject.toml` and `uv.lock` do not contain it, so `data/splits/splits.csv` (the root of every later result) cannot be regenerated or checked with `uv run`. Separately, `uv run ruff check .` reports 39 errors. 26 of them are `EXE002` ("file is executable but has no shebang"). They come from the repo living on an NTFS drive (`fuseblk`), where every file shows as `rwxrwxrwx` even though git stores them as `100644`. ADR 0001 already records that the executable bit is unreliable there. The other 13 errors are real but small.

The minor problems are silent failures in shared code: a stray label value (for example a legacy 4) is dropped from every region without a warning; `subject_of()` takes the third dash separated field of any string, so a non BraTS ID gives a wrong subject without an error; an unexpected exception leaves `run.json` at status `running`; and the table's "empty" count mixes correct cases (both empty, Dice 1) with misses (one side empty, Dice 0). There are also two inconsistencies in the docs. `train.py` saves `best.pt` but `evaluate.py` reports `last.pt`. And `experiments/AGENTS.md` says folder numbers equal E numbers, while `0001-splits` is not E1 (E1 is the complete input baseline) and `docs/whats-built.md` labels the splits "E1".

If nothing changes, E1 would start on a split nobody can regenerate, a red lint gate, and helpers that can quietly score the wrong thing.

## Options considered

The spec bundles several small decisions. Each was weighed with the engineer; the main alternatives are below.

### Option 1: pandas, add and lock it (chosen) vs rewrite with csv and numpy

Adding pandas keeps the split logic unchanged, so the regenerated split can be compared with the committed one directly, and later result aggregation (E2 tables, paired tests) will want it anyway. **Con**: one more locked dependency (about 60 MB) for a training stack that never imports it. The rewrite needs no new dependency. **Con**: five scripts change, and you would have to prove that a hand written stratified split reproduces the committed `splits.csv`, which is the riskiest file in the repo. A dev only dependency was also considered. **Con**: the audit and split scripts are part of the reproducible pipeline, not tooling, so they belong in the main dependencies.

### Option 2: ruff, ignore EXE002 and fix the rest (chosen) vs an explicit rule set vs per file ignores

Ignoring one rule fixes the root cause (the file system) with the smallest change. **Con**: if the repo ever moves to a Linux file system, a real missing shebang would go unflagged (low cost: no script here relies on being executed directly). Pinning `lint.select` gives full control. **Con**: you then choose and maintain a rule list. Per file ignores on old scripts. **Con**: `EXE002` would still fire on every new file.

### Option 3: checkpoint, last.pt counts (chosen) vs best.pt vs both behind a flag

A fixed iteration budget reporting the final weights uses no data for selection, so the reported set stays clean, and the run folder gets simpler. **Con**: if training overfits late, the final weights are worse than the best ones. `best.pt` is common practice. **Con**: with few val subjects selection is noisy, and real experiments would then have to report on the test fold only, a protocol change. A flag keeps both. **Con**: two code paths and a choice per experiment, which invites quiet inconsistency between tables.

### Option 4: E numbers, decouple with a mapping table (chosen) vs renaming folders

Decoupling keeps every existing path (results, run records, spec links) valid. **Con**: readers must look up the table once. Renaming folders so the number equals the E number. **Con**: splits and tracer have no E number, and renames break `results/` paths and spec 0001's links.

## Rationale

The forces are reproducibility and fail loudly. The project rules say every number traces to a script with a fixed seed in the locked environment, and that splits are fixed by code and checked by a test. Today the split cannot be rerun under `uv run`, and no pytest test covers it, so both rules are broken at the root. Adding pandas is the only fix that leaves the split code untouched, which is what lets AC-2 prove the committed split is reproducible instead of producing a new one. The committed file wins on a mismatch because the tracer run and every later result rest on it.

The lint fix follows the same logic: the errors that matter (unused imports, unsorted imports, a blind except) are fixed in code, and only the one rule that measures the file system rather than the code is switched off, with ADR 0001 as the recorded reason. For the checkpoint, using the same 4 val subjects to select a checkpoint and to report it is a small leak. With a fixed budget, the reported number depends only on the config and the seed, which fits the fair comparison rule (same budget for every strategy). Label and ID checks raise rather than assert, because asserts vanish under `python -O` and these checks guard what the thesis measures.

RECOMMEND items settled here: (1) `N_LABELLED_CASES = 1251` stays a named constant rather than being read from the audit output, because the audit output is gitignored and the count is a dataset fact recorded in `data/DATA.md`; runner up, derive it from `results/0000-data-audit/cases.csv`. (2) Run id collisions get a counter suffix, not microseconds, because a counter is guaranteed unique and stays readable; runner up, microsecond timestamps. (3) The `audit.py` blind except stays, with a `noqa` and a reason, because the audit's job is to log any unreadable file and continue; runner up, list the expected exception types (gzip, OS, nibabel errors), which risks crashing the audit on one the list missed. (4) The split comparison checks the `case` and `fold` columns, not raw bytes, because float formatting may differ between pandas versions while the assignment is what matters; the SHA256 of the committed file is still recorded.
