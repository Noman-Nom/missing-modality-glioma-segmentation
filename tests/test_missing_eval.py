"""Spec 0004 on a real synthetic E1 run: guards, the quick sweep, analyse, the tables,
determinism, the run record, the folder rule, resume, the test gate and check_rules (AC-1, AC-2,
AC-3, AC-5, AC-6, AC-7, AC-9, AC-10, AC-12, AC-13, AC-14, AC-15, AC-16). CPU only, no real data."""
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch
from common.data import MODALITIES, REGIONS
from common.evaluation import (
    EVAL_RUN_KEYS,
    METRIC_COLUMNS,
    PROTOCOL_FILES,
    check_rules,
    protocol_hash,
)
from common.missing import PATTERNS
from common.runlog import dump_config, load_config, load_run
from conftest import only_run, write_case, write_splits
from scipy.stats import chi2

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "experiments" / "0003-baseline"
E2 = REPO / "experiments" / "0004-missing-eval"


def load(folder, name):
    spec = importlib.util.spec_from_file_location(f"e2_{folder.name[:4]}_{name}", folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate, analyse, make_table, check_eval = (
    load(E2, n) for n in ("evaluate", "analyse", "make_table", "check_eval"))
ANALYSIS_FILES = ("stats.csv", "friedman.csv", "marginal.csv", "verdict.txt")
TABLE_NAMES = ("dice", "hd95", "marginal", "heatmap")


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="module")
def e1_run(tmp_path_factory):
    """A real finished E1 run on 2 train, 3 val and 2 test synthetic cases, tiny config, with
    metrics_val.csv, built once. Test cases reach the cache only after training, as in real life."""
    root = tmp_path_factory.mktemp("e1")
    data = root / "data"
    train = [f"BraTS-GLI-0000{i}-000" for i in range(1, 3)]
    val = [f"BraTS-GLI-0000{i}-000" for i in range(4, 7)]
    test = [f"BraTS-GLI-0000{i}-000" for i in range(7, 9)]
    for i, c in enumerate(train + val + test):
        write_case(data, c, seed=i)
    splits = root / "splits.csv"
    write_splits(splits, [(c, "train", 0, False) for c in train]
                 + [(c, "val", 0, False) for c in val] + [(c, "test", 0, False) for c in test])
    cfg = load_config(E1 / "config.toml")
    cfg["data"].update(splits_csv=str(splits), cache_dir=str(root / "cache"))
    cfg["train"].update(iterations=2, val_every=2, patch=[32, 32, 32], num_workers=0)
    cfg["model"].update(channels=[4, 8], strides=[2], num_res_units=1)
    cfg["infer"]["roi"] = [32, 32, 32]
    config = root / "config.toml"
    dump_config(cfg, config)
    assert load(E1, "preprocess").main(["--config", str(config), "--data-dir", str(data)]) == 0
    out = root / "results"
    assert load(E1, "train").main(["--config", str(config), "--out", str(out),
                                   "--device", "cpu"]) == 0
    run = only_run(out)
    assert load(E1, "evaluate").main(["--run", str(run), "--device", "cpu"]) == 0
    assert load(E1, "preprocess").main(["--config", str(config), "--data-dir", str(data),
                                        "--folds", "test"]) == 0
    torch.use_deterministic_algorithms(False)
    return run


def run_eval(run, out, *extra):
    return evaluate.main(["--run", str(run), "--fold", "val", "--out", str(out), "--device", "cpu",
                          *extra])


def edit_run(run, tmp_path, **fields):
    """A copy of the E1 run with some run.json fields changed."""
    copy = tmp_path / "e1-copy"
    shutil.copytree(run, copy)
    rec = load_run(copy) | fields
    (copy / "run.json").write_text(json.dumps(rec))
    return copy


@pytest.fixture(scope="module")
def quick_eval(e1_run, tmp_path_factory):
    """One quick sweep plus its analysis, shared by the read only tests below."""
    out = tmp_path_factory.mktemp("eval")
    assert run_eval(e1_run, out, "--quick") == 0
    (ev,) = out.iterdir()
    assert analyse.main(["--eval", str(ev)]) == 0
    torch.use_deterministic_algorithms(False)
    return ev


def rows_of(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# covers: AC-3
def test_quick_sweep_writes_the_exact_rows(quick_eval, e1_run):
    rows = rows_of(quick_eval / "metrics.csv")
    assert list(rows[0]) == METRIC_COLUMNS
    assert len(rows) == 2 * 15 * 3
    cases = sorted(load_run(e1_run)["val_cases"])[:2]
    assert [(r["case"], int(r["pattern_id"]), r["region"]) for r in rows] == [
        (c, p, g) for c in cases for p in range(1, 16) for g in REGIONS]
    assert {r["eval_id"] for r in rows} == {quick_eval.name}
    assert quick_eval.name == f"{e1_run.name}-val-quick"


def test_run_record_holds_the_spec_keys_and_the_source(quick_eval, e1_run):
    rec, source = load_run(quick_eval), load_run(e1_run)
    assert set(rec) == set(EVAL_RUN_KEYS)
    assert (rec["status"], rec["fold"], rec["n_cases"]) == ("ok", "val", 2)
    assert rec["source_run_id"] == source["run_id"]
    assert rec["source_config_hash"] == source["config_hash"]
    assert rec["predict_seconds"] > 0 and rec["metric_seconds"] > 0


# covers: AC-2. The eval config holds [eval] plus quick; E1 settings stay with the source run.
def test_eval_folder_config_is_the_eval_table_plus_quick(quick_eval):
    want = load_config(E2 / "config.toml")["eval"] | {"quick": True}
    assert load_config(quick_eval / "config.toml") == {"eval": want}


# covers: AC-2. The E1 run folder is read only.
def test_source_run_is_not_changed(e1_run, tmp_path):
    before = {p: p.stat().st_mtime_ns for p in e1_run.rglob("*")}
    assert run_eval(e1_run, tmp_path / "o", "--quick") == 0
    assert {p: p.stat().st_mtime_ns for p in e1_run.rglob("*")} == before


# covers: AC-2
@pytest.mark.parametrize("status", ["failed", "running", "interrupted"])
def test_source_run_not_ok_exits_1(e1_run, tmp_path, status):
    assert run_eval(edit_run(e1_run, tmp_path, status=status), tmp_path / "o") == 1
    assert not (tmp_path / "o").exists()


# covers: AC-2
def test_missing_best_pt_exits_1(e1_run, tmp_path):
    copy = edit_run(e1_run, tmp_path)
    (copy / "checkpoints" / "best.pt").unlink()
    assert run_eval(copy, tmp_path / "o") == 1


# covers: AC-2
def test_checkpoint_hash_mismatch_exits_1(e1_run, tmp_path):
    copy = edit_run(e1_run, tmp_path, config_hash="not the hash")
    assert run_eval(copy, tmp_path / "o") == 1


# covers: AC-2
def test_case_not_in_the_cache_exits_2(e1_run, tmp_path):
    copy = edit_run(e1_run, tmp_path, val_cases=["BraTS-GLI-00099-000"])
    assert run_eval(copy, tmp_path / "o") == 2
    assert not (tmp_path / "o").exists()


# covers: AC-2. The cache guard names the fold, so a train case cannot pass as val.
def test_train_case_listed_as_val_is_refused_by_the_cache(e1_run, tmp_path):
    rec = load_run(e1_run)
    copy = edit_run(e1_run, tmp_path, train_cases=[], val_cases=rec["train_cases"])
    assert run_eval(copy, tmp_path / "o") == 2


# covers: AC-1
def test_subject_shared_with_train_exits_2_before_any_work(e1_run, tmp_path):
    rec = load_run(e1_run)
    leak = rec["val_cases"][0][:-3] + "001"
    copy = edit_run(e1_run, tmp_path, train_cases=rec["train_cases"] + [leak])
    assert run_eval(copy, tmp_path / "o") == 2
    assert not (tmp_path / "o").exists()


# covers: AC-3
def test_nan_metric_sets_failed_and_exit_1(e1_run, tmp_path, monkeypatch):
    def nan_metrics(*a, **k):
        return {"dice": float("nan"), "hd95_mm": 0.0, "sensitivity": 0.0, "precision": 0.0,
                "gt_empty": False, "pred_empty": False}

    monkeypatch.setattr("common.evaluation.region_metrics", nan_metrics)
    assert run_eval(e1_run, tmp_path / "o", "--quick") == 1
    rec = load_run(tmp_path / "o" / f"{e1_run.name}-val-quick")
    assert rec["status"] == "failed" and "NaN metric" in rec["reason"]


# covers: AC-15
def test_two_quick_runs_are_byte_equal(e1_run, tmp_path, quick_eval):
    assert run_eval(e1_run, tmp_path / "second", "--quick") == 0
    second = tmp_path / "second" / quick_eval.name
    assert analyse.main(["--eval", str(second)]) == 0
    for name in ("metrics.csv", *ANALYSIS_FILES):
        assert (quick_eval / name).read_bytes() == (second / name).read_bytes(), name


def test_analyse_stats_rows(quick_eval):
    rows = rows_of(quick_eval / "stats.csv")
    assert len(rows) == 90
    assert [(r["metric"], r["region"], int(r["pattern_id"])) for r in rows] == [
        (m, g, p) for m in ("dice", "hd95_mm") for g in REGIONS for p in range(1, 16)]
    for r in rows:
        assert r["n"] == "2"
        assert r["pattern_name"] == PATTERNS[int(r["pattern_id"])].name
        tested = r["pattern_id"] != "1"
        for k in ("mean_diff", "ci_low", "ci_high", "n_nonzero", "wilcoxon_stat", "p_raw",
                  "p_holm"):
            assert bool(r[k]) == tested, k
        assert r["n_pred_empty"] != ""


# covers: AC-12
def test_analyse_friedman_and_marginal_rows(quick_eval):
    friedman = rows_of(quick_eval / "friedman.csv")
    assert [(r["metric"], r["region"], r["n_missing"]) for r in friedman] == [
        (m, g, str(k)) for m in ("dice", "hd95_mm") for g in REGIONS for k in (1, 2, 3)]
    marginal = rows_of(quick_eval / "marginal.csv")
    assert [(r["metric"], r["region"], r["modality"]) for r in marginal] == [
        (m, g, x) for m in ("dice", "hd95_mm") for g in REGIONS for x in MODALITIES]


# covers: AC-3 of the statistics path: d is pattern p minus pattern 1, matched on the case.
def test_analyse_differences_are_matched_on_case(quick_eval, tmp_path):
    rows = rows_of(quick_eval / "metrics.csv")
    shuffled = tmp_path / "ev"
    shuffled.mkdir()
    shutil.copy(quick_eval / "config.toml", shuffled)
    shutil.copy(quick_eval / "run.json", shuffled)
    with open(shuffled / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows[::-1])
    assert analyse.main(["--eval", str(shuffled)]) == 0
    assert (shuffled / "stats.csv").read_bytes() == (quick_eval / "stats.csv").read_bytes()


def falls_by_a_tenth(case, pattern):
    return 0.8 + case / 100 - (pattern > 1) * 0.1


def write_toy_metrics(path, drop=None, dice=falls_by_a_tenth, hd95=lambda case, pattern: 1.0,
                      cases=3, status="ok", n_cases=None):
    """A toy metrics.csv beside the [eval] config and a run.json. dice and hd95 are functions of
    (case, pattern). n_cases defaults to cases, so a toy folder is finished unless a test says so."""
    dump_config(load_config(E2 / "config.toml"), path.parent / "config.toml")
    (path.parent / "run.json").write_text(json.dumps(
        {"status": status, "n_cases": cases if n_cases is None else n_cases}))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        w.writeheader()
        for i in range(cases):
            for p in range(1, 16):
                for g in REGIONS:
                    if (i, p, g) == drop:
                        continue
                    w.writerow({"eval_id": "x", "case": f"BraTS-GLI-0000{i}-000",
                                "subject": f"0000{i}", "pattern_id": p, "region": g,
                                "pattern_name": PATTERNS[p].name, "gt_empty": False,
                                "dice": dice(i, p), "hd95_mm": hd95(i, p),
                                "pred_empty": p == 15 and i == 0})


def test_analyse_known_answer(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv")
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    rows = rows_of(tmp_path / "stats.csv")
    last = next(r for r in rows if r["region"] == "ET" and r["pattern_id"] == "15")
    first = next(r for r in rows if r["region"] == "ET" and r["pattern_id"] == "1")
    assert first["mean"] == "0.810000" and first["median"] == "0.810000"
    assert last["mean"] == "0.710000" and last["mean_diff"] == "-0.100000"
    assert last["n_nonzero"] == "3" and last["n_pred_empty"] == "1"
    assert float(last["p_holm"]) == pytest.approx(min(1.0, 14 * float(last["p_raw"])))
    # every difference is -0.1, so every bootstrap mean is too
    assert (float(last["ci_low"]), float(last["ci_high"])) == pytest.approx((-0.1, -0.1))


# covers: AC-12. Secondary metric: same tests as Dice, on its own rows.
def test_analyse_hd95_rows_use_their_own_family(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv", hd95=lambda i, p: 10.0 + p * (i + 1))
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    row = next(r for r in rows_of(tmp_path / "stats.csv")
               if (r["metric"], r["region"], r["pattern_id"]) == ("hd95_mm", "WT", "3"))
    assert row["mean"] == "16.000000" and row["median"] == "16.000000"  # 13, 16, 19
    assert row["mean_diff"] == "4.000000" and row["n_nonzero"] == "3"  # d is 2, 4, 6


# covers: AC-12. T1 adds exactly 0.1 at every case, the other three add nothing, so each pair
# difference is known. mean_present and mean_absent are means over 7 patterns, not 8.
def test_analyse_marginal_is_the_matched_pair_answer(tmp_path):
    def dice(case, pattern):
        return 0.5 + 0.1 * ("T1" in PATTERNS[pattern].present)

    write_toy_metrics(tmp_path / "metrics.csv", dice=dice)
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    rows = {(r["metric"], r["region"], r["modality"]): r for r in rows_of(tmp_path / "marginal.csv")}
    t1, t2 = rows["dice", "TC", "T1"], rows["dice", "TC", "T2"]
    assert float(t1["mean_diff"]) == pytest.approx(0.1)
    assert (float(t1["mean_present"]), float(t1["mean_absent"])) == pytest.approx((0.6, 0.5))
    assert (float(t1["ci_low"]), float(t1["ci_high"])) == pytest.approx((0.1, 0.1))
    assert float(t2["mean_diff"]) == pytest.approx(0.0)


# covers: AC-12. Each pair differs by exactly one modality, and S is never empty.
@pytest.mark.parametrize("modality", MODALITIES)
def test_marginal_pairs_differ_by_exactly_that_modality(modality):
    pairs = analyse.marginal_pairs(modality)
    assert len(pairs) == 7
    assert sum(1 for with_m, _ in pairs if with_m == 1) == 1
    for with_m, without_m in pairs:
        assert set(PATTERNS[with_m].present) - set(PATTERNS[without_m].present) == {modality}
        assert len(PATTERNS[with_m].present) == len(PATTERNS[without_m].present) + 1


# covers: AC-12. The groups compare patterns of one size, and every incomplete pattern is in one.
def test_friedman_groups_hold_patterns_of_equal_size():
    assert {k: len(v) for k, v in analyse.GROUPS.items()} == {1: 4, 2: 6, 3: 4}
    assert sorted(p for ids in analyse.GROUPS.values() for p in ids) == list(range(2, 16))
    for k, ids in analyse.GROUPS.items():
        assert {len(PATTERNS[p].missing) for p in ids} == {k}


# covers: AC-12. Hand worked: every case ranks the 4 patterns the same way, so with n = 4 and
# k = 4 the statistic is n(k-1) = 12 and W is 1. HD95 stays flat, so it hits the coded tied case.
def test_analyse_friedman_known_answer(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv", dice=lambda i, p: 0.1 * p + i, cases=4)
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    rows = {(r["metric"], r["region"], r["n_missing"]): r for r in rows_of(tmp_path / "friedman.csv")}
    dice = rows["dice", "WT", "1"]
    assert dice["n"] == "4" and float(dice["statistic"]) == pytest.approx(12)
    assert float(dice["kendall_w"]) == pytest.approx(1)
    assert float(dice["p_raw"]) == pytest.approx(chi2.sf(12, 3), rel=1e-4)
    flat = rows["hd95_mm", "WT", "1"]
    assert (flat["statistic"], flat["p_raw"], flat["p_holm"], flat["kendall_w"]) == (
        "0.000000", "1", "1", "0.000000")


# covers: AC-12. Two runs on one folder give the same bytes, so the seed really fixes the CI.
def test_analyse_twice_is_byte_identical(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv", dice=lambda i, p: 0.8 - 0.01 * p * ((i * 7) % 3))
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    first = {n: (tmp_path / n).read_bytes() for n in ANALYSIS_FILES}
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    assert first == {n: (tmp_path / n).read_bytes() for n in ANALYSIS_FILES}


# covers: AC-12, the rows of one family share one index matrix, so they are paired: two rows with
# the same differences get the same CI.
def test_analyse_every_row_uses_the_same_bootstrap_matrix(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv", dice=lambda i, p: 0.8 - (p > 1) * (i + 1) / 100)
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    tested = [r for r in rows_of(tmp_path / "stats.csv")
              if r["metric"] == "dice" and r["pattern_id"] != "1"]
    assert len({(r["ci_low"], r["ci_high"]) for r in tested}) == 1
    assert tested[0]["ci_low"] != tested[0]["ci_high"]


def toy_stats(mean_diff=-0.1, p_holm=0.001, **patterns):
    """Dice stats rows of all tested patterns, with overrides by name like TC_7=(0.1, 0.001)."""
    rows = []
    for g in REGIONS:
        for p in analyse.TESTED:
            diff, holm = patterns.get(f"{g}_{p}", (mean_diff, p_holm))
            rows.append({"metric": "dice", "region": g, "pattern_id": p, "mean_diff": diff,
                         "p_holm": holm})
    return rows


def toy_friedman(**p_holm):
    return [{"metric": "dice", "region": g, "n_missing": k, "p_holm": p_holm.get(g, 1.2e-05)}
            for g in REGIONS for k in (1, 2, 3)]


# covers: AC-13
def test_verdict_all_supported_is_the_exact_template():
    assert analyse.verdict(toy_stats(), toy_friedman(), 0.05) == [
        "H1a WT: supported", "H1a TC: supported", "H1a ET: supported",
        "H1b WT: supported (n_missing=1, p_holm=1.2e-05)",
        "H1b TC: supported (n_missing=1, p_holm=1.2e-05)",
        "H1b ET: supported (n_missing=1, p_holm=1.2e-05)"]


# covers: AC-13. One pattern fails on its sign, the other on Holm; IDs ascend whatever the order.
def test_verdict_names_the_failing_patterns_ascending():
    stats = toy_stats(TC_12=(-0.1, 0.06), TC_7=(0.02, 0.001))
    assert analyse.verdict(stats[::-1], toy_friedman(), 0.05)[1] == (
        "H1a TC: not supported (patterns 7, 12)")


# covers: AC-13. A significant change upward is a failure of H1a, because the test is two sided.
def test_verdict_h1a_reads_the_direction_from_the_sign():
    assert analyse.verdict(toy_stats(WT_2=(0.05, 0.0001)), toy_friedman(), 0.05)[0] == (
        "H1a WT: not supported (patterns 2)")


# covers: AC-13. Only the n_missing 1 group decides H1b, and the rule is strictly below alpha.
def test_verdict_h1b_uses_only_group_one_and_is_strict():
    rows = toy_friedman(ET=0.031)
    for r in rows:
        if r["n_missing"] != 1:
            r["p_holm"] = 0.9
    assert analyse.verdict(toy_stats(), rows, 0.05)[3] == (
        "H1b WT: supported (n_missing=1, p_holm=1.2e-05)")
    assert analyse.verdict(toy_stats(), rows, 0.01)[5] == (
        "H1b ET: not supported (n_missing=1, p_holm=0.031)")
    assert analyse.verdict(toy_stats(), toy_friedman(WT=0.05), 0.05)[3] == (
        "H1b WT: not supported (n_missing=1, p_holm=0.05)")


# covers: AC-13. HD95 never decides the verdict.
def test_verdict_ignores_hd95_rows():
    bad = [r | {"metric": "hd95_mm", "mean_diff": 1.0, "p_holm": 0.9} for r in toy_stats()]
    worse = [r | {"metric": "hd95_mm", "p_holm": 0.9} for r in toy_friedman()]
    good = analyse.verdict(toy_stats(), toy_friedman(), 0.05)
    assert analyse.verdict(toy_stats() + bad, toy_friedman() + worse, 0.05) == good


# covers: AC-13
def test_analyse_stores_the_printed_verdict(tmp_path, capsys):
    write_toy_metrics(tmp_path / "metrics.csv")
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    lines = (tmp_path / "verdict.txt").read_text().splitlines()
    assert len(lines) == 6 and lines[0].startswith("H1a WT: ")
    assert "\n".join(lines) in capsys.readouterr().out


def test_analyse_refuses_a_missing_row(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv", drop=(1, 7, "TC"))
    with pytest.raises(SystemExit, match="missing"):
        analyse.main(["--eval", str(tmp_path)])
    assert not (tmp_path / "stats.csv").exists()


def test_analyse_refuses_a_duplicate_row(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv")
    with open(tmp_path / "metrics.csv") as f:
        lines = f.readlines()
    (tmp_path / "metrics.csv").write_text("".join(lines + lines[-1:]))
    with pytest.raises(SystemExit, match="expected"):
        analyse.main(["--eval", str(tmp_path)])


OUTPUTS = ("stats.csv", "friedman.csv", "marginal.csv", "verdict.txt")


def assert_analysis_refused(folder, message):
    """analyse.py stops with this error and writes no output file."""
    with pytest.raises(SystemExit, match=message) as e:
        analyse.main(["--eval", str(folder)])
    assert str(e.value).startswith("ERROR: ")
    assert not [n for n in OUTPUTS if (folder / n).exists()]


# covers: AC-12
def test_analyse_refuses_a_folder_with_no_run_json(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv")
    (tmp_path / "run.json").unlink()
    assert_analysis_refused(tmp_path, "run.json missing")


# covers: AC-12. interrupted is in runlog.STATUSES; an eval never writes it, but it is still not ok.
@pytest.mark.parametrize("status", ["running", "failed", "interrupted"])
def test_analyse_refuses_every_status_that_is_not_ok(tmp_path, status):
    write_toy_metrics(tmp_path / "metrics.csv", status=status)
    assert_analysis_refused(tmp_path, f"status is '{status}'")


# covers: AC-12. Absent and null both count as a failed check, not as a KeyError.
@pytest.mark.parametrize(("key", "message"), [("status", "status is None"),
                                              ("n_cases", "expected n_cases None")])
@pytest.mark.parametrize("absent", [True, False], ids=["absent", "null"])
def test_analyse_refuses_a_status_or_case_count_with_no_value(tmp_path, key, message, absent):
    write_toy_metrics(tmp_path / "metrics.csv")
    edit_json(tmp_path, **{key: None})
    if absent:
        rec = load_run(tmp_path)
        del rec[key]
        (tmp_path / "run.json").write_text(json.dumps(rec))
    assert_analysis_refused(tmp_path, message)


# covers: AC-12. A died sweep: 3 whole cases of a planned 114 pass the row check, so the status
# is what refuses it.
def test_analyse_refuses_a_sweep_stopped_between_two_cases(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv", status="running", n_cases=114)
    assert analyse.load_metrics(tmp_path / "metrics.csv")["case"].nunique() == 3
    assert_analysis_refused(tmp_path, "status is 'running'")


# covers: AC-12. The same refusal on a folder the real sweep left, not a hand built one: the
# stop lands on a case boundary, so 1 of 2 cases is there and whole.
def test_analyse_refuses_a_real_stopped_sweep(e1_run, tmp_path, monkeypatch):
    folder = stop_after(e1_run, tmp_path, 15, monkeypatch)
    assert analyse.load_metrics(folder / "metrics.csv")["case"].nunique() == 1
    assert_analysis_refused(folder, "status is 'running'")


# covers: AC-12. A part copy from Kaggle: status ok, but the file was cut short. Too many is refused
# the same way.
@pytest.mark.parametrize("n_cases", [114, 2])
def test_analyse_refuses_a_case_count_that_differs_from_n_cases(tmp_path, n_cases):
    write_toy_metrics(tmp_path / "metrics.csv", n_cases=n_cases)
    assert_analysis_refused(tmp_path, f"3 cases, expected n_cases {n_cases}")


# covers: AC-12. The four checks run in the spec order: each folder below fails two of them, and
# the earlier one must speak.
def test_analyse_checks_run_in_the_spec_order(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv", status="failed")
    (tmp_path / "metrics.csv").unlink()
    assert_analysis_refused(tmp_path, "status is 'failed'")
    write_toy_metrics(tmp_path / "metrics.csv", status="failed", drop=(1, 7, "TC"))
    assert_analysis_refused(tmp_path, "status is 'failed'")
    write_toy_metrics(tmp_path / "metrics.csv", drop=(1, 7, "TC"), n_cases=114)
    assert_analysis_refused(tmp_path, "missing")


# covers: AC-12
def test_a_refusal_leaves_older_output_files_byte_equal(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv")
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    before = {n: (tmp_path / n).read_bytes() for n in OUTPUTS}
    edit_json(tmp_path, status="running")
    with pytest.raises(SystemExit, match="status is 'running'"):
        analyse.main(["--eval", str(tmp_path)])
    assert {n: (tmp_path / n).read_bytes() for n in OUTPUTS} == before


# covers: AC-12, AC-15. The quick_eval fixture runs analyse.py, and a quick run has to pass.
def test_analyse_accepts_a_quick_run_with_status_ok(quick_eval):
    rec = load_run(quick_eval)
    assert (rec["quick"], rec["status"], rec["n_cases"]) == (True, "ok", 2)
    assert all((quick_eval / n).is_file() for n in OUTPUTS)


@pytest.fixture(scope="module")
def tables(quick_eval, tmp_path_factory):
    out = tmp_path_factory.mktemp("tables")
    assert make_table.main(["--eval", str(quick_eval), "--out-dir", str(out)]) == 0
    return {n: (out / f"0004-missing-eval-val-{n}.tex").read_text() for n in TABLE_NAMES}


# covers: AC-14
@pytest.mark.parametrize("name", TABLE_NAMES)
def test_each_table_carries_its_provenance_and_val_note(tables, quick_eval, name):
    tex, rec = tables[name], load_run(quick_eval)
    assert tex.startswith(f"% generated by experiments/0004-missing-eval/make_table.py from run "
                          f"{quick_eval.name}\n")
    assert rec["config_hash"] in tex and rec["git_commit"] in tex and "seed 42" in tex
    assert "n = 2" in tex and "val fold" in tex
    assert ("Development fold, not a test result. The checkpoint was chosen on this fold at "
            "complete input, so pattern 1 is slightly optimistic.") in tex


# covers: AC-14
@pytest.mark.parametrize(("name", "rows"), [("dice", 15), ("hd95", 15), ("marginal", 8)])
def test_tables_have_the_spec_layout(tables, name, rows):
    tex = tables[name]
    assert "booktabs" not in tex and r"\toprule" in tex and r"\resizebox{\linewidth}{!}" in tex
    assert tex.count(r"\\") == 2 + rows


# covers: AC-14. The test caption is the spec's, and the val note is gone.
def test_test_fold_caption(quick_eval, tmp_path):
    ev = tmp_path / "ev"
    shutil.copytree(quick_eval, ev)
    edit_json(ev, fold="test")
    assert make_table.main(["--eval", str(ev), "--out-dir", str(tmp_path)]) == 0
    for n in TABLE_NAMES:
        tex = (tmp_path / f"0004-missing-eval-test-{n}.tex").read_text()
        assert "Test fold, opened once." in tex and "Development fold" not in tex


# covers: AC-14
def test_dice_cells_star_follows_holm_p_and_the_config_alpha():
    row = {"mean": "0.8", "mean_diff": "-0.1", "p_holm": "0.04", "ci_low": "-0.12",
           "ci_high": "-0.08"}
    assert make_table.dice_cells(row, 0.05) == ["0.800", "-0.100$^{*}$", "[-0.120, -0.080]"]
    assert make_table.dice_cells(row, 0.01) == ["0.800", "-0.100", "[-0.120, -0.080]"]
    assert make_table.dice_cells(row | {"mean_diff": "", "p_holm": ""}, 0.05) == ["0.800", "--", "--"]


# covers: AC-14. Through main: the star comes from alpha in the eval folder's config.toml. 3 toy
# cases give p_holm 1 everywhere, so one p value is set by hand, between the two alphas.
@pytest.mark.parametrize(("alpha", "starred"), [(0.05, True), (0.01, False)])
def test_table_star_follows_alpha_in_the_eval_folder_config(tmp_path, alpha, starred):
    ev = tmp_path / "ev"
    ev.mkdir()
    write_toy_metrics(ev / "metrics.csv")
    assert analyse.main(["--eval", str(ev)]) == 0

    def set_p(rows):
        next(r for r in rows if r["pattern_id"] == "2")["p_holm"] = "0.04"

    edit_csv(ev / "stats.csv", set_p)
    cfg = load_config(ev / "config.toml")
    dump_config(cfg | {"eval": cfg["eval"] | {"alpha": alpha}}, ev / "config.toml")
    edit_json(ev, fold="val", seed=42, eval_id="x", config_hash="c", git_commit="g")
    assert make_table.main(["--eval", str(ev), "--out-dir", str(tmp_path)]) == 0
    tex = (tmp_path / "0004-missing-eval-val-dice.tex").read_text()
    assert (make_table.STAR in tex) == starred


# covers: AC-14. HD95 leads with the median, 1 decimal, and pattern 1 has no star.
def test_hd95_cells():
    row = {"median": "12.34", "mean": "20.06", "n_pred_empty": "3", "p_holm": "0.01"}
    assert make_table.hd95_cells(row, 0.05) == ["12.3$^{*}$", "20.1", "3"]
    assert make_table.hd95_cells(row | {"p_holm": ""}, 0.05) == ["12.3", "20.1", "3"]


# covers: AC-14. The table comes from the analysis CSV, cell by cell.
def test_marginal_table_reads_the_analysis(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv",
                      dice=lambda i, p: 0.5 + 0.1 * ("T1" in PATTERNS[p].present))
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    tex = make_table.marginal_table(make_table.read_marginal(tmp_path),
                                    make_table.read_stats(tmp_path),
                                    {"fold": "val", "seed": 42, "eval_id": "x", "config_hash": "c",
                                     "git_commit": "g"})
    t1 = next(line for line in tex.splitlines() if line.startswith("Dice & T1 &"))
    assert t1.count("0.100 & [0.100, 0.100]") == 3


# covers: AC-14. 15 x 3 cells in scan order, pattern 1 at 0, greyscale, scale from the lowest to 0.
def test_heatmap_cells_and_scale(tmp_path):
    write_toy_metrics(tmp_path / "metrics.csv", dice=lambda i, p: 0.9 - 0.02 * (p - 1))
    assert analyse.main(["--eval", str(tmp_path)]) == 0
    tex = make_table.heatmap(make_table.read_stats(tmp_path),
                             {"fold": "val", "seed": 42, "eval_id": "x", "config_hash": "c",
                              "git_commit": "g"})
    lines = tex.splitlines()
    start = next(i for i, x in enumerate(lines) if "meta index=2" in x) + 1
    cells = [x.split() for x in lines[start:start + 45]]
    assert lines[start + 45] == "};"
    assert [c[:2] for c in cells] == [[str(x), str(y)] for y in range(15) for x in range(3)]
    assert cells[0][2] == "0.000000" and cells[-1][2] == "-0.280000"
    assert "point meta min=-0.280000, point meta max=0" in tex
    assert "matrix plot*" in tex and "gray(0cm)" in tex and "nodes near coords=" in tex


# covers: AC-14
def test_stub_inputs_the_four_files():
    tex = (REPO / "writing" / "tracer" / "e2-preview.tex").read_text()
    for n in TABLE_NAMES:
        assert rf"\input{{../tables/0004-missing-eval-val-{n}.tex}}" in tex
    for package in ("fontspec", "booktabs", "graphicx", "pgfplots"):
        assert rf"\usepackage{{{package}}}" in tex
    assert r"\pgfplotsset{compat=1.18}" in tex


# covers: AC-14
@pytest.mark.parametrize(("name", "message"), [("stats.csv", "89 rows in stats.csv"),
                                               ("marginal.csv", "23 rows in marginal.csv")])
def test_tables_refuse_a_wrong_row_count(quick_eval, tmp_path, name, message):
    shutil.copytree(quick_eval, tmp_path / "ev")
    path = tmp_path / "ev" / name
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(SystemExit, match=message):
        make_table.main(["--eval", str(tmp_path / "ev"), "--out-dir", str(tmp_path)])


# covers: AC-14
@pytest.mark.parametrize("name", ["stats.csv", "marginal.csv"])
def test_tables_refuse_a_missing_csv(quick_eval, tmp_path, name):
    shutil.copytree(quick_eval, tmp_path / "ev")
    (tmp_path / "ev" / name).unlink()
    with pytest.raises(SystemExit, match="missing"):
        make_table.main(["--eval", str(tmp_path / "ev"), "--out-dir", str(tmp_path)])

@pytest.fixture(scope="module")
def val_eval(e1_run, tmp_path_factory):
    """A finished non quick val eval (3 cases), as the test gate expects it."""
    out = tmp_path_factory.mktemp("full")
    assert run_eval(e1_run, out) == 0
    torch.use_deterministic_algorithms(False)
    return out / f"{e1_run.name}-val"


def edit_json(folder, **fields):
    (folder / "run.json").write_text(json.dumps(load_run(folder) | fields))


def edit_csv(path, change):
    """Rewrite a CSV after change(rows), which edits the rows in place or returns new ones."""
    rows = rows_of(path)
    rows = change(rows) or rows
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def copy_quick(quick_eval, tmp_path):
    copy = tmp_path / "o" / quick_eval.name
    shutil.copytree(quick_eval, copy)
    return copy


# covers: AC-10
def test_protocol_hash_is_the_files_in_the_spec_order(quick_eval):
    assert [Path(f).name for f in PROTOCOL_FILES] == [
        "config.toml", "analyse.py", "evaluation.py", "metrics.py", "missing.py", "stats.py"]
    h = hashlib.sha256()
    for name in PROTOCOL_FILES:
        h.update((REPO / name).read_bytes())
    assert h.hexdigest() == protocol_hash() == load_run(quick_eval)["protocol_hash"]


def test_a_finished_record_holds_timings_and_no_rerun_fields(quick_eval):
    rec = load_run(quick_eval)
    assert rec["predict_seconds"] > 0 and rec["metric_seconds"] > 0
    assert rec["wall_time_s"] >= rec["predict_seconds"] + rec["metric_seconds"]
    assert (rec["resumes"], rec["rerun_reason"], rec["superseded"], rec["reason"]) == (
        [], None, None, None)
    assert rec["peak_vram_reserved_mib"] is None  # CPU


def test_exception_while_loading_the_model_sets_failed(e1_run, tmp_path, monkeypatch):
    def boom(cfg):
        raise RuntimeError("no model\nsecond line")

    monkeypatch.setattr(evaluate, "build_model", boom)
    assert run_eval(e1_run, tmp_path, "--quick") == 1
    rec = load_run(tmp_path / f"{e1_run.name}-val-quick")
    assert (rec["status"], rec["reason"]) == ("failed", "RuntimeError: no model")


# covers: AC-6
def test_finished_folder_is_refused_and_left_alone(e1_run, quick_eval, tmp_path, capsys):
    copy = copy_quick(quick_eval, tmp_path)
    before = (copy / "metrics.csv").read_bytes()
    assert run_eval(e1_run, tmp_path / "o", "--quick") == 2
    assert "--rerun-reason" in capsys.readouterr().out
    assert (copy / "metrics.csv").read_bytes() == before
    assert [p.name for p in (tmp_path / "o").iterdir()] == [copy.name]


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_reason_is_no_reason(e1_run, quick_eval, tmp_path, blank):
    copy_quick(quick_eval, tmp_path)
    assert run_eval(e1_run, tmp_path / "o", "--quick", "--rerun-reason", blank) == 2


def test_a_reason_moves_the_finished_folder_and_is_logged(e1_run, quick_eval, tmp_path):
    copy = copy_quick(quick_eval, tmp_path)
    assert run_eval(e1_run, tmp_path / "o", "--quick", "--rerun-reason", "cache rebuilt") == 0
    (old,) = [p for p in (tmp_path / "o").iterdir() if p != copy]
    assert old.name.startswith(f"{copy.name}.superseded-")
    assert load_run(old)["status"] == "ok"
    assert (old / "metrics.csv").read_bytes() == (quick_eval / "metrics.csv").read_bytes()
    rec = load_run(copy)
    assert (rec["status"], rec["rerun_reason"], rec["superseded"]) == (
        "ok", "cache rebuilt", old.name)
    assert rec["resumes"] == []


def test_a_guard_failure_does_not_move_the_finished_folder(e1_run, quick_eval, tmp_path):
    copy_quick(quick_eval, tmp_path)
    run = edit_run(e1_run, tmp_path, status="failed")
    assert run_eval(run, tmp_path / "o", "--quick", "--rerun-reason", "x") == 1
    assert [p.name for p in (tmp_path / "o").iterdir()] == [quick_eval.name]


def test_a_reason_with_no_finished_folder_is_refused(e1_run, tmp_path, capsys):
    assert run_eval(e1_run, tmp_path / "o", "--quick", "--rerun-reason", "x") == 2
    assert "not one" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_a_reason_does_not_replace_an_unfinished_folder(e1_run, quick_eval, tmp_path):
    copy = copy_quick(quick_eval, tmp_path)
    edit_json(copy, status="failed")
    assert run_eval(e1_run, tmp_path / "o", "--quick", "--rerun-reason", "x") == 2
    assert load_run(copy)["status"] == "failed"


def test_an_empty_folder_is_a_start_that_died_before_its_record(e1_run, tmp_path):
    (tmp_path / f"{e1_run.name}-val-quick").mkdir()
    assert run_eval(e1_run, tmp_path, "--quick") == 0


def test_metrics_with_no_record_is_refused(e1_run, tmp_path):
    folder = tmp_path / f"{e1_run.name}-val-quick"
    folder.mkdir()
    (folder / "metrics.csv").write_text("anything\n")
    assert run_eval(e1_run, tmp_path, "--quick") == 2
    assert (folder / "metrics.csv").read_text() == "anything\n"


# covers: AC-5
def stop_after(run, out, n, monkeypatch):
    """A killed session: nothing runs after pair n, so run.json stays running."""
    monkeypatch.setenv("EVAL_STOP_AFTER", str(n))
    with pytest.raises(SystemExit):
        run_eval(run, out, "--quick")
    monkeypatch.delenv("EVAL_STOP_AFTER")
    return out / f"{run.name}-val-quick"


def test_stop_leaves_running_and_seven_whole_pairs(e1_run, tmp_path, monkeypatch):
    folder = stop_after(e1_run, tmp_path, 7, monkeypatch)
    rec = load_run(folder)
    assert rec["status"] == "running" and rec["predict_seconds"] > 0
    assert len(rows_of(folder / "metrics.csv")) == 21


@pytest.mark.parametrize("cut", [0, 5, 400], ids=["clean", "truncated line", "part of a pair"])
def test_resume_equals_an_uninterrupted_run(e1_run, quick_eval, tmp_path, monkeypatch, cut):
    folder = stop_after(e1_run, tmp_path, 7, monkeypatch)
    metrics = folder / "metrics.csv"
    with open(metrics, "r+b") as f:
        f.truncate(metrics.stat().st_size - cut)
    assert run_eval(e1_run, tmp_path, "--quick") == 0
    assert metrics.read_bytes() == (quick_eval / "metrics.csv").read_bytes()
    assert load_run(folder)["status"] == "ok"
    assert load_run(folder)["resumes"][0]["done_pairs"] == (7 if cut == 0 else 6)


def test_resume_records_the_entry_and_adds_up_the_timings(e1_run, tmp_path, monkeypatch):
    folder = stop_after(e1_run, tmp_path, 7, monkeypatch)
    edit_json(folder, peak_vram_reserved_mib=123.0)
    first = load_run(folder)
    assert run_eval(e1_run, tmp_path, "--quick") == 0
    rec = load_run(folder)
    (entry,) = rec["resumes"]
    assert set(entry) == {"utc", "done_pairs", "git_commit", "previous_status"}
    assert (entry["done_pairs"], entry["git_commit"], entry["previous_status"]) == (
        7, first["git_commit"], "running")
    for key in ("predict_seconds", "metric_seconds", "wall_time_s"):
        assert rec[key] > first[key]
    assert rec["peak_vram_reserved_mib"] == 123.0


def test_a_second_stop_keeps_the_first_entry_and_the_time(e1_run, quick_eval, tmp_path, monkeypatch):
    folder = stop_after(e1_run, tmp_path, 7, monkeypatch)
    first = load_run(folder)["predict_seconds"]
    stop_after(e1_run, tmp_path, 7, monkeypatch)
    assert load_run(folder)["predict_seconds"] > first
    assert run_eval(e1_run, tmp_path, "--quick") == 0
    assert [e["done_pairs"] for e in load_run(folder)["resumes"]] == [7, 14]
    assert (folder / "metrics.csv").read_bytes() == (quick_eval / "metrics.csv").read_bytes()


def test_failed_run_resumes_and_clears_the_reason(e1_run, quick_eval, tmp_path, monkeypatch):
    real = evaluate.sweep

    def fails(*a, **k):
        def boom(rows):
            raise RuntimeError("disk gone")

        real(*a[:4], boom, **k)

    monkeypatch.setattr(evaluate, "sweep", fails)
    assert run_eval(e1_run, tmp_path, "--quick") == 1
    monkeypatch.setattr(evaluate, "sweep", real)
    assert run_eval(e1_run, tmp_path, "--quick") == 0
    rec = load_run(tmp_path / quick_eval.name)
    assert (rec["status"], rec["reason"]) == ("ok", None)
    assert rec["resumes"][0]["previous_status"] == "failed"


def test_resume_with_every_pair_done_only_finishes(e1_run, quick_eval, tmp_path):
    copy = copy_quick(quick_eval, tmp_path)
    edit_json(copy, status="running")
    assert run_eval(e1_run, tmp_path / "o", "--quick") == 0
    rec = load_run(copy)
    assert rec["status"] == "ok" and rec["resumes"][0]["done_pairs"] == 30
    assert (copy / "metrics.csv").read_bytes() == (quick_eval / "metrics.csv").read_bytes()


@pytest.mark.parametrize("key", ["git_commit", "config_hash", "protocol_hash", "source_run_id",
                                 "source_config_hash", "source_best_iteration"])
def test_resume_mismatch_is_refused_before_any_cut(e1_run, tmp_path, monkeypatch, key, capsys):
    folder = stop_after(e1_run, tmp_path, 7, monkeypatch)
    edit_json(folder, **{key: "other"})
    metrics = folder / "metrics.csv"
    with open(metrics, "ab") as f:
        f.write(b"truncated")
    before = metrics.read_bytes()
    assert run_eval(e1_run, tmp_path, "--quick") == 2
    assert key in capsys.readouterr().out
    assert metrics.read_bytes() == before
    assert load_run(folder)["resumes"] == []


# covers: AC-9, AC-16
@pytest.fixture
def broken(val_eval, e1_run, tmp_path):
    """Copies of the val eval and the E1 run, so one rule can be broken at a time."""
    out = tmp_path / "o"
    shutil.copytree(val_eval, out / val_eval.name)
    run = tmp_path / "e1"
    shutil.copytree(e1_run, run)
    return out / val_eval.name, run


def test_rules_pass_on_a_real_val_eval(val_eval, e1_run):
    assert check_rules(val_eval, e1_run) == []


def test_rules_stop_on_a_missing_record(tmp_path):
    assert "run.json missing" in check_rules(tmp_path, tmp_path)[0]


@pytest.mark.parametrize(("fields", "message"), [
    ({"status": "running"}, "status is 'running'"),
    ({"quick": True}, "quick run"),
    ({"gpu": "other"}, "gpu is 'other'"),
    ({"n_cases": 4}, "rows, expected 180"),
])
def test_rules_refuse_a_broken_record(broken, fields, message):
    ev, run = broken
    edit_json(ev, **fields)
    assert any(message in e for e in check_rules(ev, run))


def test_rules_refuse_a_missing_row(broken):
    ev, run = broken
    edit_csv(ev / "metrics.csv", lambda rows: rows[:-1])
    errors = check_rules(ev, run)
    assert any("rows, expected 135" in e for e in errors)
    assert any("1 missing" in e for e in errors)


def test_rules_refuse_a_repeated_row(broken):
    ev, run = broken
    edit_csv(ev / "metrics.csv", lambda rows: rows[:-1] + rows[:1])
    assert any("1 repeated" in e for e in check_rules(ev, run))


def test_rules_refuse_a_nan_row(broken):
    ev, run = broken
    edit_csv(ev / "metrics.csv", lambda rows: rows[0].update(hd95_mm="nan"))
    assert any("no Dice or HD95" in e for e in check_rules(ev, run))


def test_rules_refuse_missing_metrics_files(broken):
    ev, run = broken
    (run / "metrics_val.csv").unlink()
    assert any("no tie out" in e for e in check_rules(ev, run))
    (ev / "metrics.csv").unlink()
    assert any("metrics.csv missing" in e for e in check_rules(ev, run))


def shift(rows, amounts):
    """Add amounts[region] to the WT, TC or ET Dice of every row."""
    for r in rows:
        r["dice"] = repr(float(r["dice"]) + amounts.get(r["region"], 0.0))


def test_tie_out_refuses_a_mean_off_by_0_002(broken):
    ev, run = broken
    edit_csv(run / "metrics_val.csv", lambda rows: shift(rows, {"TC": 0.002}))
    assert [e for e in check_rules(ev, run) if "Dice" in e] == [
        "TC mean Dice differs from E1 by 0.002000"]


def test_tie_out_keeps_a_mean_inside_the_limit(broken):
    ev, run = broken
    edit_csv(run / "metrics_val.csv", lambda rows: shift(rows, {"TC": 0.0009}))
    assert check_rules(ev, run) == []


def test_tie_out_refuses_one_case_off_with_the_mean_unchanged(broken):
    ev, run = broken

    def opposite(rows):
        wt = [r for r in rows if r["region"] == "WT"]
        wt[0]["dice"] = repr(float(wt[0]["dice"]) + 0.02)
        wt[1]["dice"] = repr(float(wt[1]["dice"]) - 0.02)

    edit_csv(run / "metrics_val.csv", opposite)
    (error,) = check_rules(ev, run)
    assert error.startswith("2 case Dice value(s) differ from E1")


def test_tie_out_refuses_another_case_set(broken):
    ev, run = broken
    edit_csv(run / "metrics_val.csv", lambda rows: rows[3:])
    assert any("case set differs" in e for e in check_rules(ev, run))


# covers: AC-7
def run_test(run, out, *extra):
    return evaluate.main(["--run", str(run), "--fold", "test", "--out", str(out),
                          "--device", "cpu", *extra])


@pytest.fixture
def gate(val_eval, e1_run, tmp_path, monkeypatch):
    """Everything open: a clean git_info and a good val eval. A test closes one rule. Any read of
    a test case raises, so a refused call proves that it touched none."""
    out = tmp_path / "o"
    shutil.copytree(val_eval, out / val_eval.name)
    run = tmp_path / "e1"
    shutil.copytree(e1_run, run)
    monkeypatch.setattr(evaluate, "git_info", lambda: ("abc", False))

    def no_read(*a, **k):
        raise AssertionError("the test fold was read")

    for name in ("eval_cases", "check_cache", "sweep"):
        monkeypatch.setattr(evaluate, name, no_read)
    monkeypatch.setattr("numpy.load", no_read)
    return out, out / val_eval.name, run, monkeypatch


def assert_refused(run, out, message, capsys, *extra):
    assert run_test(run, out, *extra) == 2
    assert message in capsys.readouterr().out
    assert not list(out.glob("*-test*"))


def test_gate_needs_open_test(gate, capsys):
    out, _, run, _ = gate
    assert_refused(run, out, "--open-test not given", capsys)


def test_gate_refuses_quick(gate, capsys):
    out, _, run, _ = gate
    assert_refused(run, out, "--quick is not allowed", capsys, "--open-test", "--quick")


@pytest.mark.parametrize("dirty", [True, None])
def test_gate_refuses_a_dirty_or_unknown_tree(gate, capsys, dirty):
    out, _, run, monkeypatch = gate
    monkeypatch.setattr(evaluate, "git_info", lambda: ("abc", dirty))
    assert_refused(run, out, f"git_dirty is {dirty}", capsys, "--open-test")


def test_gate_refuses_a_quick_e1_run(gate, capsys):
    out, _, run, _ = gate
    edit_json(run, quick=True)
    assert_refused(run, out, "the E1 run is a quick run", capsys, "--open-test")


def test_gate_refuses_no_val_eval(gate, capsys):
    out, val, run, _ = gate
    shutil.rmtree(val)
    assert_refused(run, out, "finish the val eval first", capsys, "--open-test")


def test_gate_refuses_an_unfinished_val_eval(gate, capsys):
    out, val, run, _ = gate
    edit_json(val, status="running")
    assert_refused(run, out, "val eval: status is 'running'", capsys, "--open-test")


def test_gate_reruns_the_rules_on_val(gate, capsys):
    out, _, run, _ = gate
    edit_csv(run / "metrics_val.csv", lambda rows: shift(rows, {"ET": 0.5}))
    assert_refused(run, out, "val eval: ET mean Dice differs", capsys, "--open-test")


def test_gate_refuses_a_changed_protocol(gate, capsys):
    out, val, run, _ = gate
    edit_json(val, protocol_hash="old")
    assert_refused(run, out, "rerun val with --rerun-reason", capsys, "--open-test")


def test_gate_prints_every_failed_rule(gate, capsys):
    out, val, run, monkeypatch = gate
    monkeypatch.setattr(evaluate, "git_info", lambda: ("abc", True))
    edit_json(val, protocol_hash="old")
    assert run_test(run, out, "--quick") == 2
    printed = capsys.readouterr().out
    for message in ("--open-test not given", "--quick is not allowed", "git_dirty is True",
                    "protocol_hash differs"):
        assert message in printed


def test_gate_refuses_open_test_on_val(gate, capsys):
    out, _, run, _ = gate
    assert evaluate.main(["--run", str(run), "--fold", "val", "--out", str(out), "--device", "cpu",
                          "--open-test"]) == 2
    assert "only for --fold test" in capsys.readouterr().out


def test_gate_open_scores_the_synthetic_test_fold_once(gate, e1_run, capsys):
    out, _, run, monkeypatch = gate
    monkeypatch.undo()
    monkeypatch.setattr(evaluate, "git_info", lambda: ("abc", False))
    assert run_test(run, out, "--open-test") == 0
    folder = out / f"{e1_run.name}-test"
    rows = rows_of(folder / "metrics.csv")
    assert len(rows) == 2 * 15 * 3
    assert {r["case"] for r in rows} == {"BraTS-GLI-00007-000", "BraTS-GLI-00008-000"}
    rec = load_run(folder)
    assert (rec["status"], rec["fold"], rec["git_dirty"]) == ("ok", "test", False)
    assert check_rules(folder, run) == []

    # AC-16: a list cut short consistently, n_cases included, is still not the test fold
    whole = (folder / "metrics.csv").read_bytes()
    edit_csv(folder / "metrics.csv", lambda rows: rows[:15 * 3])
    edit_json(folder, n_cases=1)
    assert check_rules(folder, run) == ["case set differs from the test fold: 1 missing, 0 extra"]
    (folder / "metrics.csv").write_bytes(whole)
    edit_json(folder, n_cases=2)
    assert check_rules(folder, run) == []

    # AC-1 as a rule: a subject shared with train is reported, not raised
    source = load_run(run)
    edit_json(run, train_cases=source["train_cases"] + ["BraTS-GLI-00007-000"])
    assert any("shares 1 subject" in e for e in check_rules(folder, run))
    edit_json(run, train_cases=source["train_cases"])

    edit_json(out / f"{e1_run.name}-val", protocol_hash="old")
    assert any("protocol_hash differs" in e for e in check_rules(folder, run))
    edit_json(out / f"{e1_run.name}-val", protocol_hash=protocol_hash())
    assert run_test(run, out, "--open-test") == 2  # one finished run (AC-6)
    assert "finished" in capsys.readouterr().out
    assert run_test(run, out, "--open-test", "--rerun-reason", "synthetic") == 0
    assert load_run(folder)["rerun_reason"] == "synthetic"


# covers: AC-16. source_run in run.json is the default, and a passing val prints the projection.
def test_check_eval_passes_and_projects_the_test_time(val_eval, capsys):
    assert check_eval.main([str(val_eval)]) == 0
    out = capsys.readouterr().out
    rec = load_run(val_eval)
    hours = (rec["predict_seconds"] + rec["metric_seconds"]) * 2 / 3600
    assert out.startswith("PASS: ") and f"Projected test time: {hours:.2f} hours" in out


# covers: AC-16. --run replaces a stale source_run, as after a move between machines.
def test_check_eval_run_flag_beats_a_moved_source(val_eval, e1_run, tmp_path):
    ev = tmp_path / "ev"
    shutil.copytree(val_eval, ev)
    edit_json(ev, source_run=str(tmp_path / "gone"))
    assert check_eval.main([str(ev)]) == 1
    assert check_eval.main([str(ev), "--run", str(e1_run)]) == 0


# covers: AC-16. Each reason is printed, and no projection comes from a failed folder.
def test_check_eval_prints_each_failure_and_exits_1(val_eval, e1_run, tmp_path, capsys):
    ev = tmp_path / "ev"
    shutil.copytree(val_eval, ev)
    edit_json(ev, status="failed", quick=True)
    assert check_eval.main([str(ev), "--run", str(e1_run)]) == 1
    out = capsys.readouterr().out
    assert "FAIL: status is 'failed'" in out and "FAIL: a quick run" in out
    assert "PASS" not in out and "Projected" not in out


# covers: AC-16
def test_check_eval_stops_on_a_folder_with_no_record(tmp_path, capsys):
    assert check_eval.main([str(tmp_path)]) == 1
    assert "run.json missing" in capsys.readouterr().out
