"""Spec 0001, AC-11 and AC-13 (part): failure handling and the experiment check, on synthetic
cases with a tiny CPU config. Also runs the thin thread train -> evaluate -> table -> check."""
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
from common.runlog import dump_config, load_config

EXP = Path(__file__).resolve().parents[1] / "experiments" / "0002-tracer-bullet"


def load(name):
    spec = importlib.util.spec_from_file_location(f"tracer_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train, evaluate, make_table, check = (load(n) for n in
                                      ("train", "evaluate", "make_table", "check_tracer"))


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def setup(tmp_path, synthetic_cases):
    data, splits, _, _ = synthetic_cases
    cfg = load_config(EXP / "config.toml")
    cfg["data"].update(n_train_subjects=2, n_val_subjects=1)
    cfg["train"].update(iterations=2, val_every=2, patch=[32, 32, 32], num_workers=0)
    cfg["model"].update(channels=[4, 8], strides=[2])
    cfg["infer"]["roi"] = [32, 32, 32]
    path = tmp_path / "config.toml"
    dump_config(cfg, path)
    out = tmp_path / "results"
    argv = ["--config", str(path), "--data-dir", str(data), "--splits", str(splits),
            "--out", str(out), "--device", "cpu"]
    return argv, out, data


def only_run(out):
    (run,) = out.iterdir()
    return run


def test_inject_inf_fails(setup, monkeypatch):
    argv, out, _ = setup
    monkeypatch.setenv("TRACER_INJECT_INF", "1")
    assert train.main(argv) == 1
    rec = json.loads((only_run(out) / "run.json").read_text())
    assert rec["status"] == "failed" and "non finite loss at iteration 1" in rec["reason"]
    assert rec["iterations"] == 0


def test_missing_file_exits_2_before_training(setup, synthetic_cases):
    argv, out, data = setup
    case = synthetic_cases[2][0]
    os.remove(data / case / f"{case}-t1c.nii.gz")
    assert train.main(argv) == 2
    assert not out.exists()  # no run folder, nothing trained


def test_no_data_dir_exits_2(setup, tmp_path):
    argv, _, _ = setup
    argv[argv.index("--data-dir") + 1] = str(tmp_path / "nope")
    assert train.main(argv) == 2


def test_thin_thread_and_check(setup, tmp_path, capsys):
    argv, out, data = setup
    assert train.main(argv) == 0
    run = only_run(out)
    rec = json.loads((run / "run.json").read_text())
    assert rec["status"] == "ok" and rec["iterations"] == 2 and rec["quick"] is False
    assert rec["val_wt_dice"][0][0] == 2 and rec["best_iteration"] == 2
    assert (run / "checkpoints" / "last.pt").is_file() and (run / "config.toml").is_file()
    assert load_config(run / "config.toml")["train"]["iterations"] == 2

    assert evaluate.main(["--run", str(run), "--data-dir", str(data), "--device", "cpu"]) == 0
    assert len((run / "metrics.csv").read_text().splitlines()) == 1 + 1 * 15 * 3

    tex = tmp_path / "table.tex"
    assert make_table.main(["--run", str(run), "--out", str(tex)]) == 0
    body = tex.read_text()
    assert "n = 1 val subjects" in body and "Not a result." in body
    assert body.count(r" \\") == 2 + 15  # two header rows, 15 pattern rows

    capsys.readouterr()
    assert check.main([str(run)]) == 1  # a 2 iteration run is not the full run
    assert "FAIL: AC-5" in capsys.readouterr().out

    rec["quick"] = True
    (run / "run.json").write_text(json.dumps(rec))
    assert check.main([str(run)]) == 1
    assert capsys.readouterr().out.strip().endswith("FAIL: quick run")


def test_check_rejects_quick(tmp_path, capsys):
    from common.runlog import RUN_KEYS
    run = tmp_path / "run"
    run.mkdir()
    rec = dict.fromkeys(RUN_KEYS)
    rec.update(run_id="r", status="ok", quick=True, val_cases=["BraTS-GLI-00005-000"])
    (run / "run.json").write_text(json.dumps(rec))
    (run / "config.toml").write_text("seed = 42\n")
    cols = check.COLUMNS
    lines = [",".join(cols)]
    for pid in range(1, 16):
        for region in ("WT", "TC", "ET"):
            lines.append(f"r,BraTS-GLI-00005-000,00005,{pid},x,{region},0.5,3.0,,,False,False")
    (run / "metrics.csv").write_text("\n".join(lines) + "\n")
    assert check.main([str(run)]) == 1
    assert capsys.readouterr().out.strip().endswith("FAIL: quick run")


def test_check_rejects_running(tmp_path, capsys):
    from common.runlog import RUN_KEYS
    run = tmp_path / "run"
    run.mkdir()
    rec = dict.fromkeys(RUN_KEYS)
    rec.update(status="running")
    (run / "run.json").write_text(json.dumps(rec))
    assert check.main([str(run)]) == 1
    assert "crashed run" in capsys.readouterr().out
