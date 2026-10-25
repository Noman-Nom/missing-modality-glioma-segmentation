# Verify: tracer review fixes · spec 0002 · updated 2026-10-25

_Steps derived from spec 0002 acceptance criteria. `/check verify` runs these; `/test` locks the durable ones._

How `/check verify` proves each acceptance criterion in [index.md](index.md). Fast checks the AI may run itself. Steps marked **(human)** are run by the human, who pastes the last two lines. All commands run from the repo root with `UV_PROJECT_ENVIRONMENT` set (ADR 0001).

## Commands

- [x] `uv run python -c "import pandas; print(pandas.__version__)"` → prints `3.1.0`; `grep -n pandas pyproject.toml uv.lock | head -3` shows it in `[project] dependencies` and the lock → AC-1
- [x] `sed -n 72p docs/architecture/decisions/0001-python-environment.md` names pandas, 2026-10-25 and the reason; line 149 no longer says "scipy was not installed" → AC-1
- [x] **(human)** if `results/0000-data-audit/cases.csv` is missing, rerun the audit first; then `uv run python experiments/0001-splits/make_splits.py --out /tmp/splits.csv` and `uv run python experiments/0001-splits/check_splits.py --compare /tmp/splits.csv` → `OK ... matches data/splits/splits.csv: 1251 cases, same fold per case`; `git diff --quiet data/splits/splits.csv` exits 0 (committed file untouched) → AC-2
- [x] On a match: `sha256sum data/splits/splits.csv` and the pandas version are recorded in `data/DATA.md` (Splits), whose Regenerate and Check commands use `uv run`. On a mismatch: stop, open `/debug`, overwrite nothing → AC-2
- [x] `uv run ruff check .` → `All checks passed!`; `grep -n -A2 "tool.ruff.lint" pyproject.toml` shows only `EXE002` with the NTFS comment; `grep -rn "noqa" experiments` shows only the one `BLE001` in `audit.py` with a reason → AC-3
- [x] `grep -rn "best" experiments/0002-tracer-bullet/train.py experiments/common/runlog.py` finds no `best.pt` or `best_iteration`; the evaluate docstring states the fixed budget, final weights rule → AC-4
- [x] `uv run python experiments/0002-tracer-bullet/check_tracer.py results/0002-tracer-bullet/20261025-000309-f4bbd0b9` → `PASS` (an old run holding `best_iteration` still passes) → AC-4b
- [x] `uv run pytest -q tests/test_tracer_failures.py -k "legacy or key_differences"` → passes (legacy key accepted, unknown extra or missing key rejected) → AC-4b
- [x] `uv run pytest -q tests/test_tracer_failures.py -k hash_mismatch` → passes; exit 1 names both hashes before any scoring, no `metrics.csv` → AC-5
- [x] `uv run pytest -q tests/test_common_data.py -k "load_case"` → stray 4 and 1.5 raise `ValueError` naming the case; a float volume of 0.0 to 3.0 loads → AC-6
- [x] `uv run pytest -q tests/test_common_data.py -k subject_of`; `grep -rn "ID_RE = " experiments` finds only `BRATS_ID_RE` in `common/data.py` → AC-7
- [x] `uv run pytest -q tests/test_tracer_failures.py -k "unexpected_error or failed_reason"` → status `failed`, reason `RuntimeError: <first line>`, status line printed, error raised again → AC-8
- [x] `uv run pytest -q tests/test_tracer_units.py -k "cell_stats or 15_pattern"` → both empty and one empty counts right, 6 columns per region, `\multicolumn{6}`, `\cmidrule(lr){15-20}` → AC-9
- [x] **(human)** `uv run python experiments/0002-tracer-bullet/make_table.py --run results/0002-tracer-bullet/20261025-000309-f4bbd0b9`, then `cd writing/tracer && latexmk -xelatex tracer.tex` and `grep -c Overfull tracer.log` → `0` → AC-9
- [x] Import every script by path, each under its own name: `for f in sorted(Path("experiments").glob("0*/*.py")): spec = importlib.util.spec_from_file_location(f"x_{f.parent.name}_{f.stem}".replace("-", "_"), f); spec.loader.exec_module(importlib.util.module_from_spec(spec))` → all 9 import with no check output and no file read (no script runs on import); `grep -n N_LABELLED_CASES experiments/0001-splits/check_splits.py` → AC-10
- [x] `uv run pytest -q tests/test_splits.py` with `results/` renamed away → 2 passed → AC-11
- [x] `uv run pytest -q tests/test_tracer_failures.py -k "collision or suffix"` → second run in the same second gets `-1`, `run_id == run_dir.name` → AC-12
- [x] `grep -rn "map to the numbers" experiments docs context | grep -v 0002-tracer-review-fixes` finds nothing; the E number table is in `experiments/AGENTS.md`; `docs/whats-built.md` and `make_splits.py` no longer call the split E1 → AC-13
- [x] `experiments/common/model.py` has no double blank line inside; `select_subjects()` docstring says deterministic but not random → AC-14
- [x] `uv run pytest -q` → all pass, slow deselected; spec 0001 carries `Amended by 0002`; `docs/whats-built.md` (section G) and `context/progress-tracker.md` updated → AC-15

## Value sourcing

- [x] Allowed label set: change one voxel of a synthetic seg to 4 → `ValueError` naming `ALLOWED_LABELS` order `[0, 1, 2, 3]` → `ALLOWED_LABELS`
- [ ] Case name in the label error: the folder name of `paths["seg"]`, not the file name → `load_case` error
- [x] Subject ID: `subject_of("BraTS-GLI-01234-001")` → `"01234"` (group 1, leading zeros kept) → `BRATS_ID_RE`
- [ ] Expected case count: `check_splits.py` fails with the length printed if the regenerated split has a case fewer → `N_LABELLED_CASES`
- [x] Regenerated split: produced by `make_splits.py --cases results/0000-data-audit/cases.csv --out <tmp>`, never the committed path → AC-2 input
- [x] Committed split hash: `sha256sum data/splits/splits.csv` matches the value written into `data/DATA.md` → AC-2 record
- [x] pandas version: the version in `data/DATA.md` equals `uv run python -c "import pandas; print(pandas.__version__)"` → AC-2 record
- [x] Legacy keys: only `best_iteration` is tolerated → `LEGACY_KEYS`
- [x] Config hashes: edit `run.json` `config_hash` of a copied run → `evaluate.py` exits 1 naming both → `run.json` vs `last.pt`
- [x] Failed reason: an exception with an empty message gives just the type; a multi line message keeps the first line only → `failed_reason`
- [x] Table counts: a row with `gt_empty=True`, `pred_empty=False` counts as one empty, not both empty → `gt_empty` and `pred_empty`
- [ ] Run id suffix: with `<stamp>` and `<stamp>-1` taken, the next run is `<stamp>-2` → first free `-N`
- [x] E number mapping: every E label in the `experiments/AGENTS.md` table matches `context/project-overview.md` line 18 → E number table

## Acceptance-criteria coverage

- AC-1 steps 1, 2 · AC-2 steps 3, 4 (human) · AC-3 step 5 · AC-4 step 6 · AC-4b steps 7, 8 · AC-5 step 9 · AC-6 step 10 · AC-7 step 11 · AC-8 step 12 · AC-9 steps 13, 14 (human) · AC-10 step 15 · AC-11 step 16 · AC-12 step 17 · AC-13 step 18 · AC-14 step 19 · AC-15 step 20
