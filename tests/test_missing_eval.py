"""Spec 0004, thin thread on a real synthetic E1 run: guards, the quick sweep, analyse, the Dice
table and determinism (AC-1, AC-2, AC-3, AC-15). CPU only, no real data."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch
from common.data import REGIONS
from common.evaluation import EVAL_RUN_KEYS, METRIC_COLUMNS
from common.missing import PATTERNS
from common.runlog import dump_config, load_config, load_run
from conftest import only_run, write_case, write_splits

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "experiments" / "0003-baseline"
E2 = REPO / "experiments" / "0004-missing-eval"


def load(folder, name):
    spec = importlib.util.spec_from_file_location(f"e2_{folder.name[:4]}_{name}", folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate, analyse, make_table = (load(E2, n) for n in ("evaluate", "analyse", "make_table"))


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="module")
def e1_run(tmp_path_factory):
    """A real finished E1 run on 2 train and 3 val synthetic cases, tiny config, built once."""
    root = tmp_path_factory.mktemp("e1")
    data = root / "data"
    train = [f"BraTS-GLI-0000{i}-000" for i in range(1, 3)]
    val = [f"BraTS-GLI-0000{i}-000" for i in range(4, 7)]
    for i, c in enumerate(train + val):
        write_case(data, c, seed=i)
    splits = root / "splits.csv"
    write_splits(splits, [(c, "train", 0, False) for c in train]
                 + [(c, "val", 0, False) for c in val])
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
    torch.use_deterministic_algorithms(False)
    return only_run(out)


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


# The test gate of AC-7 is not built yet, so the test fold stays shut, whatever else is given.
def test_test_fold_is_refused_until_the_gate_exists(tmp_path, capsys):
    argv = ["--run", str(tmp_path / "nothing"), "--fold", "test", "--out", str(tmp_path / "o")]
    assert evaluate.main(argv) == 2
    assert "not built yet" in capsys.readouterr().out
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


def test_existing_eval_folder_is_refused(e1_run, tmp_path):
    (tmp_path / f"{e1_run.name}-val-quick").mkdir()
    assert run_eval(e1_run, tmp_path, "--quick") == 2


# covers: AC-15
def test_two_quick_runs_are_byte_equal(e1_run, tmp_path, quick_eval):
    assert run_eval(e1_run, tmp_path / "second", "--quick") == 0
    second = tmp_path / "second" / quick_eval.name
    assert analyse.main(["--eval", str(second)]) == 0
    for name in ("metrics.csv", "stats.csv"):
        assert (quick_eval / name).read_bytes() == (second / name).read_bytes(), name


def test_analyse_stats_rows(quick_eval):
    rows = rows_of(quick_eval / "stats.csv")
    assert len(rows) == 45
    assert [(r["region"], int(r["pattern_id"])) for r in rows] == [
        (g, p) for g in REGIONS for p in range(1, 16)]
    for r in rows:
        assert r["metric"] == "dice" and r["n"] == "2"
        assert r["pattern_name"] == PATTERNS[int(r["pattern_id"])].name
        tested = r["pattern_id"] != "1"
        for k in ("mean_diff", "n_nonzero", "wilcoxon_stat", "p_raw", "p_holm"):
            assert bool(r[k]) == tested, k


# covers: AC-3 of the statistics path: d is pattern p minus pattern 1, matched on the case.
def test_analyse_differences_are_matched_on_case(quick_eval, tmp_path):
    rows = rows_of(quick_eval / "metrics.csv")
    shuffled = tmp_path / "ev"
    shuffled.mkdir()
    with open(shuffled / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows[::-1])
    assert analyse.main(["--eval", str(shuffled)]) == 0
    assert (shuffled / "stats.csv").read_bytes() == (quick_eval / "stats.csv").read_bytes()


def write_toy_metrics(path, drop=None):
    """Dice falls by 0.1 at every incomplete pattern, one extra case to give 3 cases."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        w.writeheader()
        for i in range(3):
            for p in range(1, 16):
                for g in REGIONS:
                    if (i, p, g) == drop:
                        continue
                    w.writerow({"eval_id": "x", "case": f"BraTS-GLI-0000{i}-000",
                                "subject": f"0000{i}", "pattern_id": p, "region": g,
                                "pattern_name": PATTERNS[p].name, "gt_empty": False,
                                "dice": 0.8 + i / 100 - (p > 1) * 0.1, "hd95_mm": 1.0,
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


def test_dice_table_from_the_analysis(quick_eval, tmp_path):
    assert make_table.main(["--eval", str(quick_eval), "--out-dir", str(tmp_path)]) == 0
    tex = (tmp_path / "0004-missing-eval-val-dice.tex").read_text()
    rec = load_run(quick_eval)
    assert tex.startswith(f"% generated by experiments/0004-missing-eval/make_table.py from run "
                          f"{quick_eval.name}\n")
    assert rec["config_hash"] in tex and "seed 42" in tex and "n = 2" in tex
    assert "Development fold, not a test result." in tex
    assert "booktabs" not in tex and r"\toprule" in tex and r"\resizebox{\linewidth}{!}" in tex
    assert tex.count(r"\\") == 2 + 15


def test_dice_table_star_follows_holm_p_and_the_config_alpha():
    row = {"mean": "0.8", "mean_diff": "-0.1", "p_holm": "0.04", "ci_low": "", "ci_high": ""}
    assert make_table.dice_cells(row, 0.05) == ["0.800", "-0.100$^{*}$", "--"]
    assert make_table.dice_cells(row, 0.01) == ["0.800", "-0.100", "--"]
    assert make_table.dice_cells(row | {"mean_diff": "", "p_holm": ""}, 0.05) == ["0.800", "--", "--"]


def test_dice_table_refuses_a_wrong_row_count(quick_eval, tmp_path):
    shutil.copytree(quick_eval, tmp_path / "ev")
    path = tmp_path / "ev" / "stats.csv"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(SystemExit, match="44 Dice rows"):
        make_table.main(["--eval", str(tmp_path / "ev"), "--out-dir", str(tmp_path)])
