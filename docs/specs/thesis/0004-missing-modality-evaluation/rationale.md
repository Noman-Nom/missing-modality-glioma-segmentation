# 0004. Missing modality evaluation (E2): rationale

## Context

> ⚠️ Premise note: the first E2 reading comes from val, the fold that chose the E1 checkpoint at complete input. Pattern 1 on val is therefore slightly optimistic, and the drop may look a little larger than on test. This is acceptable for design work, but val numbers must never be reported as the RQ1 answer. The table captions say so.

E1 is finished and frozen: run `20261103-023817-f9b3afb6`, seed 42, best at iteration 36,000. RQ1 needs its score on all 15 non empty modality subsets. The proposal fixes paired Wilcoxon tests, Holm correction over subsets and a 95% bootstrap with 10,000 resamples, but leaves the families, sides and the second half of H1 open. These cannot honestly be decided after numbers exist.

Forces:
- **The test fold is the only clean evidence.** 228 subjects, never loaded. It can be spent once. E3 and E4 are not designed yet, so a test reading now could steer them.
- **Cost.** E1 val took 256 s for 114 cases, 2.24 s per case on the T4. 15 patterns on 228 cases is 3,420 predictions, about 2.1 h, plus HD95 time that has not been measured. Val is about 1.1 h. A session can die, so several hours of work must survive.
- **Fairness.** RQ2 is only valid if E3 and E4 go through identical scoring code.
- **State of the data.** The Kaggle cache was in `/tmp` and is gone, and the local cache holds 3 quick cases, so preprocessing must rerun.
- **Empty regions.** Under the rule of spec 0001, an empty prediction scores 373.13 mm. Missing T1ce may empty ET often (a guess, not measured), which could swamp HD95 means.

## Options considered

### Option 1: test now, as the scope first read
Run val, freeze, and score test straight away.
- **Pros**: RQ1 closes now, and feature 6 finishes as first worded.
- **Cons**: E3 and E4 are designed with test numbers known. It is a small leak, but one the thesis would have to disclose.

### Option 2: val first, one sealed joint test opening (chosen)
Val informs E3 and E4. Test opens once, for all three models.
- **Pros**: test stays clean through all design choices, and the whole pipeline is proven on val first.
- **Cons**: the RQ1 test table waits weeks, and feature 6 closes without a test number.

### Option 3: test only, no val pass
- **Pros**: one run, the least quota.
- **Cons**: the first bug would be found while spending the test fold, and E3 and E4 would get no development reading.

### Machine
Kaggle T4 was chosen over the laptop. It matches E1, E3 and E4, and is roughly 3.8 times faster on training steps (inference speed on the laptop was not measured). The cost is uploading `best.pt` and an ADR 0002 line. The laptop would avoid both, but would take about 8 h or more per fold.

### Sealing the test fold
A flag, a clean tree and one finished run were chosen. A flag alone proves nothing about the protocol. A separate test script could drift from the val path, and identical code is the point.

### Statistics
- **Holm per region.** This matches the proposal's wording, and each region is its own question. One family of 42 is stricter, but rarely changes a clear drop. A combined Dice and HD95 family spends power on a secondary metric.
- **Two sided.** A higher score at some pattern is a real finding, and one sided tests could never show it. The lost power is minor at this sample size.
- **Friedman within equal size groups, plus a matched pair marginal table.** Pairwise tests against complete input cannot answer which modality matters. All 91 pairwise tests would be heavier than RQ1 needs. Descriptive only would leave half of H1 untested. The first draft ran Friedman over all 14 patterns, which would pass almost by construction, because a pattern with one modality differs from a pattern with three. Grouping by the number missing removes that. The first draft's marginal table was also biased: patterns with a modality present hold 2.29 modalities on average, against 1.71 without it. Matched pairs remove that bias.
- **HD95 empties kept.** Dropping them would score different subjects per pattern and hide the worst failures. Wilcoxon uses ranks, so the penalty cannot dominate.
- **Statistics in `common/stats.py`.** Experiments cannot import from each other, so an E2 local copy would have to move anyway.

### Output format
PGFPlots was chosen because it is already installed with TeX Live and adds no lock change. `matplotlib` would need an ADR. Generated `.tex` goes to `writing/tables/`, because that folder is already ignored by git, and `writing/figures/` is not.

### Spec update 2026-11-04: `analyse.py` guards and the source of `alpha`
Milestone 3 found two contradictions in this spec. AC-12 said `analyse.py` reads only `metrics.csv` and `config.toml`, while the API table said it stops on a non `ok` eval. AC-14 said tables read only the analysis CSVs and `run.json`, while the star needs `alpha`.

**Status check.** The row check alone does not catch a died sweep. `load_metrics` builds its expected rows from the cases already in `metrics.csv`, so a sweep that stops between two cases looks complete.
- **Chosen: status plus case count.** `analyse.py` also reads `run.json`. Status catches the died sweep and the NaN failure. `n_cases` catches a `metrics.csv` cut short under status `ok`, for example a part copy from Kaggle.
- Status only: fixes the died sweep, but a cut short file passes `analyse.py` and is caught only if `check_eval.py` is run.
- Keep as built and fix the API table: no code change, but a part sweep can write a full verdict.
- Runner up in the spirit of reuse: call `check_rules()`. It refuses quick runs and needs the E1 run, so AC-15 would fail and `analyse.py` would stop being standalone.

`make_table.py` needs no status check of its own. It needs the analysis CSVs, and any folder analysed after build step 8 can hold them only at status `ok`. A folder analysed before that step and then resumed could keep stale CSVs, because a resume does not delete them. No real E2 folder exists, so this cannot happen in practice.

**Source of `alpha`.** The spec moved to match the code.
- **Chosen: `[eval] alpha` in the eval folder's `config.toml`.** One value, the one `analyse.py` already uses, and it sits inside `protocol_hash`. No code change.
- A `significant` column in `stats.csv`: AC-14 keeps its wording, but the columns, `make_table.py` and tests change for no gain.
- `alpha` in `run.json`: a new AC-10 key holding a copy of a value, so two places could differ.
- A constant in `make_table.py`: a second source, and it would silently ignore the config.

One small limit stays. The verdict uses unrounded p values, while the star reads `%.6g` from `stats.csv`. A p value within about 0.00000005 below alpha could be significant in the verdict and unstarred in the table. This is left as it is, because the verdict is the rule and the star is display.

### Spec update 2026-11-04: fixes from `/check review`

The review (`docs/reviews/2026-11-04-feat-e2-missing-eval.md`) found four points that the spec pinned, so the build could not change them alone. None of them touches the statistics, and no E2 number existed when they were decided.

**Resume keys (AC-5).**
- **Chosen: add `device` and `gpu`.** AC-9 trusts the `gpu` field, and a resume rebuilt the record from the first session, so a second machine could hide behind it.
- Log each session's `gpu` in `resumes` and fail `check_rules()`: the sweep would spend a whole session before anyone learnt the run could never pass.
- `seed` and `splits_sha256`, which the review also named, add nothing. `seed` is inside `config_hash`, and `splits.csv` is tracked, so `git_commit` pins it.

**Star (AC-14).**
- **Chosen: keep the two sided star, and say so in the caption.** It stays a direct reading of `p_holm`, and a clear improvement stays visible.
- Star only if worse: matches the H1a reading, but hides a real result, and a table would no longer match `stats.csv`. The HD95 table prints no difference, so its reader needs the caption either way.

**`make_table.py` guards (AC-14).** This reverses the note above that `make_table.py` needs no status check of its own. That note counted on the analysis CSVs existing only at status `ok`, but it left the case of a folder analysed and later resumed, and a table with a new run's caption on old CSVs is the worst way to fail. Three lines close it, with the same rule as `analyse.py`. Calling `check_rules()` instead was rejected for the same reason as before: it refuses quick runs and needs the E1 run.

**Sealing inference (AC-20).** A change to `predict()` after the val run passed the gate, because the AC-9 tie out compares stored files and scores nothing.
- **Chosen: hash the source text of a named list, plus the values of `MODALITIES`, `REGIONS` and `REGION_LABELS`.** A few lines, no file moves, and `build_model()` stays free for E4. The cross check found that `predict`, `CachedCase` and `label_to_regions` alone still left the load path and the constants open: editing `REGION_LABELS` would shift the ground truth with the hash unchanged.
- Hash all of `model.py` and `data.py`, with E4's builder in a new file: no leaks and less code, but any edit to `data.py`, E3's training transforms included, would force an E1 val rerun. Runner up.
- Move `predict()` to its own file: the hash stays pure file bytes, but E1's imports change, and `CachedCase` stays open unless it moves too.
- Git diff at the gate: needs full history on Kaggle, and refuses E4's legitimate change to `build_model()`.
- Rescoring a few val cases at the gate: the strongest check, since it also catches library drift, but the gate would need a GPU and the val cache, and AC-7 runs before any GPU work.
- Note it in Consequences only: leaves the gap open in the one place the study can least afford it.

**Cross check of this update (2026-11-04, `sonnet`).** It raised six gaps, all closed in the spec. One recommendation was not taken: checking `n` in `friedman.csv`, because `make_table.py` never reads that file. A label or length prefix per hashed part was also left out, because the file bytes before them have no separators either.

## Rationale

The test fold is the scarcest resource in the study, so the design protects it at every step: val is the rehearsal, the code gate is the lock, and `protocol_hash` is the evidence that rehearsal and real run used the same rules. This costs a delay in the RQ1 table, which is a fair price given that RQ2 is the main contribution and depends on a clean comparison.

Fixed folder names were chosen over timestamped ones because they make resume and the one run rule a single mechanism. A rerun must be an explicit, logged act.

Smaller calls, with the runner up:
- Tie out on the same GPU type, mean within 0.001, each case within 0.01. One flipped voxel in a small ET can move Dice by more than 0.001, so that limit applies to the mean only. Runner up: exact equality, which would fail on harmless noise.
- `zero_method="wilcox"`, the common default, because a zero difference carries no direction. Runner up: `pratt`.
- `method="asymptotic"`, so one rule covers every sample size. Runner up: `auto`, which changes its method with n and ties.
- One bootstrap index matrix, so rows are paired and the output is reproducible. Runner up: a new draw per row.
- `protocol_hash` blocks at the gate, before any GPU work, with no override flag: the way past it is a logged val rerun. Runner up: check only after the sweep, which could spend the sealed fold on a run that then fails. A runner up to whole file hashes is `git diff` between commits, which is just as sensitive to formatting edits and fails on a shallow clone.
- Resume keeps the strict `git_commit` rule, and `KAGGLE.md` pins the commit. Runner up: warn only, but `model.py`, `data.py` and `cache.py` sit outside `protocol_hash`, so a warning could mix code versions.
- Sensitivity and precision are stored but not tested. They support feature 11, and more tests would dilute RQ1.
- Threshold stays at 0.5. Tuning per pattern would give a different model per pattern, and E2 measures the frozen one.

## Cross check

A second independent read on 2026-11-04 (Sonnet, read only) covered the spec update above. It confirmed the design, `n_cases` on quick runs and after a resume, and the renumbered build steps. Its 7 gaps were applied on the engineer's choice: the check order and error style, absent `status` or `n_cases`, older files after a refusal, the test fixture change, the died sweep setup, the stale doc lines, and keeping `run.json` in any copy. A copy that swaps cases but keeps the count is not caught by `analyse.py` or `check_rules()`, which was judged negligible.

An independent read on 2026-11-04 (Sonnet, read only) found 18 items, all applied to `index.md` on the engineer's choice. The main ones: the Friedman and marginal faults above, the late protocol check, an overclaim about single opening, and the undefined formats that byte identical output needs. It also confirmed the row counts and the helper signatures against the code.

## Evidence

Measured 2026-11-04 from the repo:
- `run.json`: 26,577 s wall time, 0.60 s per iteration, val 256 s per point, peak reserved 1,136 MiB, Tesla T4.
- `splits.csv`: test 255 cases, 228 at timepoint 000, 7 empty ET, 1 empty TC, 1 `neg_values_flag`, 0 `flair_mask_flag`.
- `val_log.csv`: pattern 1 only, so no E2 number exists yet.
- Died sweep toy (2026-11-04, throwaway script, `analyse.py` as built at `e2c7b41`): 3 cases of a planned 114, each with all 45 rows, no `run.json`. Exit 0, with 90, 18 and 24 rows and a `verdict.txt`. Toy numbers, not a result.
- `pgfplots.sty` is installed. `matplotlib` and `statsmodels` are not in `uv.lock`.
