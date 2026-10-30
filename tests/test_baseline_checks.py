"""Spec 0003, build plan 5 (checks): memory_test.py, check_baseline.py, and the determinism and
seed tests, on synthetic cases with a tiny CPU config. Covers AC-6 (wiring, not the GPU
number), AC-9, AC-15, AC-16."""
import csv
import json

import pytest
import torch
from common import training
from common.runlog import config_hash, dump_config
from conftest import only_run
from test_baseline_thread import evaluate, load, preprocess, train

memory_test, check = load("memory_test"), load("check_baseline")


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


def read_log(run):
    with open(run / "train_log.csv", newline="") as f:
        return list(csv.DictReader(f))


# covers: AC-6
def test_memory_test_on_cpu_writes_its_record_and_never_passes(setup, tmp_path):
    setup.cfg["memory_test"]["steps"] = 3
    dump_config(setup.cfg, setup.config)
    assert preprocess.main(setup.pre_argv) == 0
    out = tmp_path / "mem"
    argv = ["--config", str(setup.config), "--device", "cpu", "--out", str(out)]
    assert memory_test.main(argv) == 1
    r = json.loads((out / "memory_test.json").read_text())
    assert tuple(r) == memory_test.KEYS
    assert r["pass"] is False and r["peak_vram_reserved_mib"] is None
    assert r["val_cases"] == setup.val and r["config_hash"] == config_hash(setup.cfg)
    assert r["seconds_per_iteration"] > 0 and r["val_seconds"] > 0 and r["projected_hours"] > 0


# covers: AC-6
def test_largest_val_cases_by_voxels_ties_by_id(setup, monkeypatch):
    shapes = {"c": (10, 10, 10), "a": (5, 5, 5), "b": (10, 10, 10), "d": (20, 1, 1)}
    manifest = {c: {"shape_d": s[0], "shape_h": s[1], "shape_w": s[2]} for c, s in shapes.items()}
    monkeypatch.setattr(memory_test, "read_manifest", lambda d: manifest)
    assert memory_test.largest_val_cases(setup.cfg, ["d", "c", "a", "b"]) == ["b", "c"]


# covers: AC-6
def test_projected_hours():
    t = {"iterations": 40000, "val_every": 4000}
    # 0.5 s/iter -> 20000 s; 10 val points x 114 cases x 3 s = 3420 s
    assert memory_test.projected_hours(0.5, 3.0, 114, t) == pytest.approx(23420 / 3600)


@pytest.fixture
def evaluated(setup):
    """A 10 iteration run (val at 5 and 10), evaluated."""
    setup.cfg["train"].update(iterations=10, val_every=5, warmup=2)
    dump_config(setup.cfg, setup.config)
    assert preprocess.main(setup.pre_argv) == 0
    assert train.main(setup.train_argv) == 0
    run = only_run(setup.out)
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 0
    return run


def as_full_run(run, monkeypatch):
    """Scale the full run thresholds down to this tiny run and fake a GPU peak and a falling
    loss, so the PASS path can be checked on CPU."""
    for k, v in {"FULL_ITERATIONS": 10, "VAL_POINTS": 2, "BEST_SCORE_MIN": -1.0, "N_TRAIN": 3,
                 "N_VAL": 1, "LOSS_WINDOW": 5}.items():
        monkeypatch.setattr(check, k, v)
    log = read_log(run)
    for i, r in enumerate(log):
        r["loss"] = str(1.0 - 0.05 * i)
    with open(run / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(log[0]))
        w.writeheader()
        w.writerows(log)
    losses = [float(r["loss"]) for r in log]
    rec = json.loads((run / "run.json").read_text())
    rec.update(peak_vram_reserved_mib=900.0, loss_start_mean=sum(losses[:5]) / 5,
               loss_end_mean=sum(losses[-5:]) / 5)
    (run / "run.json").write_text(json.dumps(rec))
    return rec


# covers: AC-16
def test_check_fails_a_tiny_run_on_the_full_thresholds(evaluated, capsys):
    assert check.main([str(evaluated)]) == 1
    assert "FAIL: 10 iterations, expected 40000" in capsys.readouterr().out


# covers: AC-9, AC-16
def test_check_passes_a_run_that_meets_every_threshold(evaluated, monkeypatch, capsys):
    as_full_run(evaluated, monkeypatch)
    assert check.main([str(evaluated)]) == 0
    assert capsys.readouterr().out.strip().endswith("PASS")


# covers: AC-9, AC-16
@pytest.mark.parametrize("change, reason", [
    ({"status": "running"}, "crashed run"),
    ({"quick": True}, "FAIL: quick run"),
    ({"peak_vram_reserved_mib": 3584.0}, "peak reserved VRAM"),
    ({"best_score": -2.0}, "AC-9: best score"),
    ({"best_iteration": 7}, "not a val point"),
    ({"cache_hash": "0" * 64}, "cache hash"),
    ({"val_cases": ["BraTS-GLI-00009-000"]}, "AC-4: test fold"),
])
def test_check_reasons(evaluated, monkeypatch, capsys, change, reason):
    rec = as_full_run(evaluated, monkeypatch)
    if change.get("best_score") is not None:
        monkeypatch.setattr(check, "BEST_SCORE_MIN", 0.60)
    (evaluated / "run.json").write_text(json.dumps(dict(rec, **change)))
    assert check.main([str(evaluated)]) == 1
    assert reason in capsys.readouterr().out


# covers: AC-9, AC-16
def test_check_fails_when_the_loss_did_not_fall(evaluated, monkeypatch, capsys):
    rec = as_full_run(evaluated, monkeypatch)
    log = read_log(evaluated)
    for r in log:
        r["loss"] = "1.0"
    with open(evaluated / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(log[0]))
        w.writeheader()
        w.writerows(log)
    (evaluated / "run.json").write_text(json.dumps(dict(rec, loss_start_mean=1.0,
                                                        loss_end_mean=1.0)))
    assert check.main([str(evaluated)]) == 1
    assert "AC-9: loss did not fall" in capsys.readouterr().out


# covers: AC-16
def test_check_fails_without_metrics(setup, capsys):
    assert preprocess.main(setup.pre_argv) == 0
    assert train.main(setup.train_argv) == 0
    assert check.main([str(only_run(setup.out))]) == 1
    assert "metrics_val.csv missing" in capsys.readouterr().out


def quick_log(setup, run_dir, num_workers):
    cfg = train.apply_quick(setup.cfg)
    cfg["train"]["num_workers"] = num_workers
    run_dir.mkdir()
    code = training.train(cfg, run_dir, run_dir.name, setup.train[:2], setup.val[:1],
                          device=torch.device("cpu"))
    assert code == 0
    return [(float(r["loss"]), float(r["lr"])) for r in read_log(run_dir)]


# covers: AC-15
@pytest.mark.parametrize("num_workers", [0, 2])
def test_two_quick_runs_with_the_same_seed_match(setup, tmp_path, num_workers):
    assert preprocess.main(setup.pre_argv) == 0
    a = quick_log(setup, tmp_path / "a", num_workers)
    b = quick_log(setup, tmp_path / "b", num_workers)
    assert len(a) == 10
    assert a == pytest.approx(b, abs=1e-6)


# covers: AC-15
def test_seed_43_quick_run_records_its_seed_and_hash(setup):
    assert preprocess.main(setup.pre_argv) == 0
    assert train.main(setup.train_argv + ["--quick", "--seed", "43"]) == 0
    rec = json.loads((only_run(setup.out) / "run.json").read_text())
    base = train.effective_config(setup.config, quick=True)
    assert rec["seed"] == 43 and rec["quick"] is True
    assert rec["config_hash"] == config_hash(dict(base, seed=43)) != config_hash(base)
