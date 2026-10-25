"""Spec 0001, AC-11 and AC-13 (part): failure handling and the experiment check, on synthetic
cases with a tiny CPU config. Also runs the thin thread train -> evaluate -> table -> check.
Spec 0002: last.pt only (AC-4), legacy run.json key (AC-4b), config hash check (AC-5), failed
status on any error (AC-8), run id suffix (AC-12)."""
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from common.runlog import RUN_KEYS, config_hash, dump_config, load_config

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
    assert rec["val_wt_dice"][0][0] == 2 and "best_iteration" not in rec
    assert [p.name for p in (run / "checkpoints").iterdir()] == ["last.pt"]
    assert (run / "config.toml").is_file() and rec["run_id"] == run.name
    assert load_config(run / "config.toml")["train"]["iterations"] == 2

    assert evaluate.main(["--run", str(run), "--data-dir", str(data), "--device", "cpu"]) == 0
    assert len((run / "metrics.csv").read_text().splitlines()) == 1 + 1 * 15 * 3

    tex = tmp_path / "table.tex"
    assert make_table.main(["--run", str(run), "--out", str(tex)]) == 0
    body = tex.read_text()
    assert "n = 1 val subjects" in body and "Not a result." in body
    assert body.count(r" \\") == 3 + 15  # three header rows, 15 pattern rows

    capsys.readouterr()
    assert check.main([str(run)]) == 1  # a 2 iteration run is not the full run
    assert "FAIL: AC-5" in capsys.readouterr().out

    rec["quick"] = True
    (run / "run.json").write_text(json.dumps(rec))
    assert check.main([str(run)]) == 1
    assert capsys.readouterr().out.strip().endswith("FAIL: quick run")


def quick_run(tmp_path, keys=RUN_KEYS):
    """A quick run folder that passes every structure check, with run.json holding `keys`."""
    run = tmp_path / "run"
    run.mkdir()
    rec = dict.fromkeys(keys)
    rec.update(run_id="r", status="ok", quick=True, val_cases=["BraTS-GLI-00005-000"])
    (run / "run.json").write_text(json.dumps(rec))
    (run / "config.toml").write_text("seed = 42\n")
    cols = check.COLUMNS
    lines = [",".join(cols)]
    for pid in range(1, 16):
        for region in ("WT", "TC", "ET"):
            lines.append(f"r,BraTS-GLI-00005-000,00005,{pid},x,{region},0.5,3.0,,,False,False")
    (run / "metrics.csv").write_text("\n".join(lines) + "\n")
    return run


def test_check_rejects_quick(tmp_path, capsys):
    run = quick_run(tmp_path)
    assert check.main([str(run)]) == 1
    assert capsys.readouterr().out.strip().endswith("FAIL: quick run")


# spec 0002, AC-4b: a run.json written before spec 0002 still passes the structure checks
def test_check_accepts_the_legacy_best_iteration_key(tmp_path, capsys):
    run = quick_run(tmp_path, keys=(*RUN_KEYS, "best_iteration"))
    assert check.main([str(run)]) == 1
    assert capsys.readouterr().out.strip().endswith("FAIL: quick run")


# spec 0002, AC-4b: any other extra or missing key still fails
@pytest.mark.parametrize("keys", [(*RUN_KEYS, "surprise"), (*RUN_KEYS, "best_iteration", "surprise"),
                                  tuple(k for k in RUN_KEYS if k != "seed")])
def test_check_rejects_other_key_differences(tmp_path, capsys, keys):
    run = quick_run(tmp_path, keys=keys)
    assert check.main([str(run)]) == 1
    assert "FAIL: run.json keys differ" in capsys.readouterr().out


# covers: AC-11 (Ctrl+C), AC-9 (exact keys on a real run, config copied)
def test_ctrl_c_writes_interrupted(setup, monkeypatch):
    argv, out, _ = setup

    def interrupted_after_three_steps(cfg, rec, run_dir, data_dir, device, gen, cfg_hash, losses):
        losses += [0.9, 0.8, 0.7]
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "train", interrupted_after_three_steps)
    assert train.main(argv) == 130
    run = only_run(out)
    rec = json.loads((run / "run.json").read_text())
    assert rec["status"] == "interrupted" and rec["reason"] == "Ctrl+C"
    assert rec["iterations"] == 3 and rec["loss_window"] == 1
    assert rec["loss_start_mean"] == 0.9 and rec["loss_end_mean"] == 0.7
    assert tuple(rec) == RUN_KEYS
    assert load_config(run / "config.toml")["train"]["iterations"] == 2


# spec 0002, AC-8: any other error writes failed with its type and first line, then raises again
def test_unexpected_error_writes_failed_and_raises(setup, monkeypatch, capsys):
    argv, out, _ = setup

    def broken(cfg, rec, run_dir, data_dir, device, gen, cfg_hash, losses):
        losses += [0.9, 0.8]
        raise RuntimeError("shape mismatch in layer 3\nsecond line")

    monkeypatch.setattr(train, "train", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        train.main(argv)
    rec = json.loads((only_run(out) / "run.json").read_text())
    assert rec["status"] == "failed" and rec["reason"] == "RuntimeError: shape mismatch in layer 3"
    assert rec["iterations"] == 2 and rec["wall_time_s"] is not None
    assert "status failed: RuntimeError: shape mismatch in layer 3" in capsys.readouterr().out


def test_failed_reason_without_a_message_is_the_type():
    assert train.failed_reason(ValueError()) == "ValueError"
    assert train.failed_reason(KeyError("x")) == "KeyError: 'x'"


# spec 0002, AC-12: same config in the same second gets a -1 suffix, and run_id is the folder name
def test_run_id_collision_gets_a_suffix(setup, monkeypatch):
    argv, out, _ = setup

    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC)

    cfg_hash = config_hash(load_config(argv[argv.index("--config") + 1]))
    stamp = f"20261025-120000-{cfg_hash[:8]}"
    (out / stamp).mkdir(parents=True)
    monkeypatch.setattr(train, "datetime", FrozenClock)
    monkeypatch.setattr(train, "train", lambda *a: None)
    assert train.main(argv) == 0
    run = out / f"{stamp}-1"
    rec = json.loads((run / "run.json").read_text())
    assert rec["run_id"] == run.name == f"{stamp}-1"
    assert not any((out / stamp).iterdir())  # the taken folder is left alone


def test_make_run_dir_counts_past_every_taken_suffix(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s-1").mkdir()
    assert train.make_run_dir(tmp_path, "s") == ("s-2", tmp_path / "s-2")


def eval_run(tmp_path, status="ok", checkpoint=True, ckpt_hash="h"):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    rec = dict.fromkeys(RUN_KEYS)
    rec.update(run_id="r", status=status, config_hash="h", data_dir=str(tmp_path / "data"),
               val_cases=["BraTS-GLI-00005-000"])
    (run / "run.json").write_text(json.dumps(rec))
    dump_config(load_config(EXP / "config.toml"), run / "config.toml")
    if checkpoint:
        torch.save({"model": {}, "iteration": 0, "config_hash": ckpt_hash},
                   run / "checkpoints" / "last.pt")
    return run


@pytest.mark.parametrize("status", ["failed", "interrupted", "running"])
def test_evaluate_refuses_a_run_that_is_not_ok(tmp_path, capsys, status):
    run = eval_run(tmp_path, status=status)
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 1
    assert "nothing evaluated" in capsys.readouterr().out
    assert not (run / "metrics.csv").exists()


def test_evaluate_refuses_a_missing_checkpoint(tmp_path, capsys):
    run = eval_run(tmp_path, checkpoint=False)
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 1
    assert "last.pt missing" in capsys.readouterr().out


def test_evaluate_reports_missing_inputs_with_exit_2(tmp_path, capsys):
    run = eval_run(tmp_path)
    (tmp_path / "data").mkdir()
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 2
    assert "input file(s) missing" in capsys.readouterr().out
    assert not (run / "metrics.csv").exists()


# spec 0002, AC-5: a checkpoint from another config is never scored
def test_evaluate_refuses_a_config_hash_mismatch(tmp_path, capsys):
    run = eval_run(tmp_path, ckpt_hash="other")
    (tmp_path / "data").mkdir()
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 1
    out = capsys.readouterr().out
    assert "config hash other differs from run.json h" in out and "nothing evaluated" in out
    assert not (run / "metrics.csv").exists()


def test_check_rejects_running(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    rec = dict.fromkeys(RUN_KEYS)
    rec.update(status="running")
    (run / "run.json").write_text(json.dumps(rec))
    assert check.main([str(run)]) == 1
    assert "crashed run" in capsys.readouterr().out
