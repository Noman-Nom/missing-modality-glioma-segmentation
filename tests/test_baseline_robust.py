"""Spec 0003, build plan 4 (robustness): failure statuses, the full last.pt state, --resume with
its guards and the best.pt rule, train.pid, preprocess --verify and the free disk check, on
synthetic cases with a tiny CPU config. Covers AC-2, AC-11, AC-12."""
import csv
import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from common import training
from common.cache import read_cache_json, read_manifest, write_case, write_manifest
from common.runlog import dump_config
from conftest import only_run
from test_baseline_thread import preprocess, train


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def ten(setup):
    """10 iterations, val every 5, cache built."""
    setup.cfg["train"].update(iterations=10, val_every=5, warmup=2)
    dump_config(setup.cfg, setup.config)
    assert preprocess.main(setup.pre_argv) == 0
    return setup


def rec_of(run):
    return json.loads((run / "run.json").read_text())


def iterations_in(path):
    with open(path, newline="") as f:
        return [int(r["iteration"]) for r in csv.DictReader(f)]


def resume(run):
    return train.main(["--resume", str(run), "--device", "cpu"])


def last_pt(run):
    return torch.load(run / "checkpoints" / "last.pt", weights_only=False)


# covers: AC-11, AC-12
def test_non_finite_loss_fails_then_resume_finishes(ten, monkeypatch):
    monkeypatch.setenv("BASELINE_INJECT_INF", "7")
    assert train.main(ten.train_argv) == 1
    run = only_run(ten.out)
    rec = rec_of(run)
    assert rec["status"] == "failed" and rec["failed_iteration"] == 7
    assert "non finite loss at iteration 7" in rec["reason"]
    assert rec["iterations"] == 6 and not (run / "train.pid").exists()
    state = last_pt(run)
    assert state["iteration"] == 5
    assert {"optimizer", "scheduler", "scaler", "best_score", "best_iteration",
            "scaler_skipped_steps", "hook_rng", "rng"} <= set(state)
    assert set(state["rng"]) == {"torch", "cuda", "numpy", "python"}

    monkeypatch.delenv("BASELINE_INJECT_INF")
    assert resume(run) == 0
    rec = rec_of(run)
    assert rec["status"] == "ok" and rec["iterations"] == 10
    assert rec["reason"] is None and rec["failed_iteration"] is None
    assert iterations_in(run / "train_log.csv") == list(range(1, 11))
    assert sorted(set(iterations_in(run / "val_log.csv"))) == [5, 10]
    assert [v["iteration"] for v in rec["val_history"]] == [5, 10]
    (r,) = rec["resumes"]
    assert r["from_iteration"] == 5 and r["previous_status"] == "failed"
    assert set(r) == {"from_iteration", "utc", "git_commit", "git_dirty", "previous_status"}
    assert last_pt(run)["iteration"] == 10 and not (run / "train.pid").exists()


# covers: AC-11
def test_resumed_lr_continues_the_schedule(ten, monkeypatch):
    assert train.main(ten.train_argv) == 0
    with open(only_run(ten.out) / "train_log.csv", newline="") as f:
        unbroken = [float(r["lr"]) for r in csv.DictReader(f)]
    out2 = ten.out.parent / "results2"
    monkeypatch.setenv("BASELINE_INJECT_INF", "8")
    argv = ten.train_argv[:ten.train_argv.index("--out")] + ["--out", str(out2), "--device", "cpu"]
    assert train.main(argv) == 1
    monkeypatch.delenv("BASELINE_INJECT_INF")
    run = only_run(out2)
    assert resume(run) == 0
    with open(run / "train_log.csv", newline="") as f:
        resumed = [float(r["lr"]) for r in csv.DictReader(f)]
    assert resumed == pytest.approx(unbroken, abs=1e-12)


# covers: AC-12
def test_ctrl_c_is_interrupted_and_resumable(ten):
    calls = []

    def hook(batch, rng):
        calls.append(1)
        if len(calls) == 7:
            raise KeyboardInterrupt
        return batch

    run = ten.out / "manual"
    run.mkdir(parents=True)
    code = training.train(ten.cfg, run, "manual", ten.train, ten.val,
                          device=torch.device("cpu"), batch_hook=hook)
    assert code == 130
    rec = rec_of(run)
    assert rec["status"] == "interrupted" and rec["reason"] == "Ctrl+C"
    assert resume(run) == 0 and rec_of(run)["status"] == "ok"


# covers: AC-12
def test_out_of_memory_fails_with_the_iteration(ten, monkeypatch):
    real = training.Trainer.step

    def step(self, it, *a, **k):
        if it == 6:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return real(self, it, *a, **k)

    monkeypatch.setattr(training.Trainer, "step", step)
    assert train.main(ten.train_argv) == 1
    rec = rec_of(only_run(ten.out))
    assert rec["status"] == "failed" and rec["failed_iteration"] == 6
    assert rec["reason"].startswith("CUDA out of memory")


# covers: AC-11
def test_best_pt_written_after_last_pt_is_deleted(ten, monkeypatch):
    monkeypatch.setenv("BASELINE_INJECT_INF", "7")
    assert train.main(ten.train_argv) == 1
    monkeypatch.delenv("BASELINE_INJECT_INF")
    run = only_run(ten.out)
    best = run / "checkpoints" / "best.pt"
    state = torch.load(best)
    torch.save(dict(state, iteration=10, score=0.99), best)  # as if a crash hit after best.pt
    rec = rec_of(run)
    (run / "run.json").write_text(json.dumps(dict(rec, best_iteration=10, best_score=0.99)))
    score_at_5 = last_pt(run)["best_score"]
    # the val point at 10 scores lower, so no new best.pt is written after the resume
    monkeypatch.setattr(training, "selection_score",
                        lambda rows: ({"WT": 0.0, "TC": 0.0, "ET": 0.0}, -1.0))
    assert resume(run) == 0
    rec = rec_of(run)
    assert not best.exists()
    assert rec["best_iteration"] == 5 and rec["best_score"] == score_at_5


# covers: AC-11
def test_resume_refusals(ten, monkeypatch):
    assert train.main(ten.train_argv) == 0
    run = only_run(ten.out)
    assert resume(run) == 2  # status ok is final

    rec = rec_of(run)
    (run / "run.json").write_text(json.dumps(dict(rec, status="running")))
    (run / "train.pid").write_text(str(os.getpid()))  # alive: this test process
    assert resume(run) == 2

    p = subprocess.Popen(["true"])
    p.wait()
    (run / "train.pid").write_text(str(p.pid))  # dead: a crashed run may continue
    assert train.resume_problems(run)[0] == []

    cfg = dict(ten.cfg, seed=7)
    dump_config(cfg, run / "config.toml")  # edited after the run started
    assert resume(run) == 2
    dump_config(ten.cfg, run / "config.toml")
    assert train.resume_problems(run)[0] == []

    (run / "run.json").write_text(json.dumps(dict(rec, status="failed", cache_hash="0" * 64)))
    assert resume(run) == 2
    assert train.main(["--resume", str(run), "--seed", "3"]) == 2  # only --device allowed


# covers: AC-11
def test_resume_refused_without_last_pt(ten, monkeypatch):
    monkeypatch.setenv("BASELINE_INJECT_INF", "3")
    assert train.main(ten.train_argv) == 1
    run = only_run(ten.out)
    problems, _, _ = train.resume_problems(run)
    assert any("no last.pt" in p for p in problems)
    assert resume(run) == 2


# covers: AC-2
def test_verify_passes_then_catches_a_changed_file(setup):
    assert preprocess.main(setup.pre_argv) == 0
    assert preprocess.main(setup.pre_argv + ["--verify"]) == 0
    case = setup.train[1]
    f = setup.cache / f"{case}.npz"
    data = bytearray(f.read_bytes())
    data[-10] ^= 0xFF
    f.write_bytes(bytes(data))
    assert preprocess.main(setup.pre_argv + ["--verify"]) == 1


# covers: AC-2
def test_verify_catches_an_image_that_does_not_match_a_recompute(setup, capsys):
    assert preprocess.main(setup.pre_argv) == 0
    case = setup.train[0]
    with np.load(setup.cache / f"{case}.npz") as z:
        parts = {k: z[k] for k in z.files}
    manifest = read_manifest(setup.cache)
    # a consistent sha256 but a wrong image: only the recompute can catch it
    manifest[case]["sha256"] = write_case(setup.cache, case, parts["image"] + 0.5,
                                          parts["label"], parts["spacing"], parts["crop_box"])
    write_manifest(setup.cache, manifest, read_cache_json(setup.cache))
    capsys.readouterr()
    assert preprocess.main(setup.pre_argv + ["--verify"]) == 1
    assert "differs from a float32 recompute" in capsys.readouterr().out


# covers: AC-2
def test_verify_needs_a_cache(setup):
    assert preprocess.main(setup.pre_argv + ["--verify"]) == 2


# covers: AC-2
def test_not_enough_free_disk_writes_nothing(setup, monkeypatch):
    monkeypatch.setattr(preprocess.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**6))
    assert preprocess.main(setup.pre_argv) == 2
    assert not setup.cache.exists() or not list(setup.cache.glob("*.npz"))
