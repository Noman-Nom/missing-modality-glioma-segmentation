"""Spec 0003, build plan 4 (robustness): failure statuses, the full last.pt state, --resume with
its guards and the best.pt rule, train.pid, preprocess --verify and the free disk check, on
synthetic cases with a tiny CPU config. Covers AC-2, AC-11, AC-12."""
import copy
import csv
import json
import os
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from common import training
from common.cache import read_cache_json, read_manifest, write_case, write_manifest
from common.runlog import RunRecord, dump_config
from common.seed import set_seed
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


def failed_at_7(ten, monkeypatch):
    """A run that fails at iteration 7, so last.pt is at 5. Returns the run folder."""
    monkeypatch.setenv("BASELINE_INJECT_INF", "7")
    assert train.main(ten.train_argv) == 1
    monkeypatch.delenv("BASELINE_INJECT_INF")
    return only_run(ten.out)


def trainer(setup):
    return training.Trainer(setup.cfg, torch.device("cpu"), setup.train,
                            set_seed(setup.cfg["seed"]))


# covers: AC-11
def test_full_state_round_trip_restores_every_stream(ten):
    a = trainer(ten)
    a.step(1)
    a.step(2)
    a.skipped = 3
    state = copy.deepcopy({"model": a.model.state_dict(), **a.full_state()})
    want = (a.hook_rng.random(), random.random(), np.random.random(), torch.rand(1).item())

    b = trainer(ten)
    b.load_full_state(state)

    assert (b.hook_rng.random(), random.random(), np.random.random(),
            torch.rand(1).item()) == want
    assert b.skipped == 3 and b.sched.last_epoch == 2
    assert b.opt.param_groups[0]["lr"] == a.opt.param_groups[0]["lr"]
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, b.model.state_dict()[k]), k
    saved, loaded = a.opt.state_dict()["state"], b.opt.state_dict()["state"]
    assert saved.keys() == loaded.keys()
    for i in saved:
        for k, v in saved[i].items():
            assert torch.equal(torch.as_tensor(v), torch.as_tensor(loaded[i][k])), (i, k)


# covers: AC-11
def test_resume_restores_last_pt_and_seeds_a_fresh_loader(ten, monkeypatch):
    run = failed_at_7(ten, monkeypatch)
    saved = last_pt(run)
    seen = {}
    real_init, real_load = training.Trainer.__init__, training.Trainer.load_full_state

    def init(self, cfg, device, cases, gen):
        seen["loader_seed"] = gen.initial_seed()
        real_init(self, cfg, device, cases, gen)

    def load(self, state):
        real_load(self, state)
        seen["last_epoch"] = self.sched.last_epoch
        seen["weights"] = {k: v.clone() for k, v in self.model.state_dict().items()}

    monkeypatch.setattr(training.Trainer, "__init__", init)
    monkeypatch.setattr(training.Trainer, "load_full_state", load)
    assert resume(run) == 0
    assert seen["loader_seed"] == ten.cfg["seed"] * 100003 + 5
    assert seen["last_epoch"] == 5
    for k, v in saved["model"].items():
        assert torch.equal(v, seen["weights"][k]), k


# covers: AC-11
def test_resume_on_another_commit_warns_and_records_it(ten, monkeypatch, capsys):
    run = failed_at_7(ten, monkeypatch)
    original = rec_of(run)["git_commit"]
    monkeypatch.setattr(training, "git_info", lambda: ("other-commit", True))
    capsys.readouterr()
    assert resume(run) == 0
    assert "WARNING: git commit other-commit" in capsys.readouterr().out
    rec = rec_of(run)
    (entry,) = rec["resumes"]
    assert set(entry) == {"from_iteration", "utc", "git_commit", "git_dirty", "previous_status"}
    assert (entry["from_iteration"], entry["git_commit"], entry["git_dirty"],
            entry["previous_status"]) == (5, "other-commit", True, "failed")
    assert rec["git_commit"] == original


# covers: AC-10, AC-11
def test_resume_keeps_the_wall_time_and_peaks_of_earlier_segments(ten, monkeypatch):
    run = failed_at_7(ten, monkeypatch)
    rec = rec_of(run)
    (run / "run.json").write_text(json.dumps(dict(
        rec, wall_time_s=1000.0, peak_vram_allocated_mib=3000.0, peak_vram_reserved_mib=3200.0)))
    assert resume(run) == 0
    after = rec_of(run)
    assert after["wall_time_s"] >= 1000.0
    assert (after["peak_vram_allocated_mib"], after["peak_vram_reserved_mib"]) == (3000.0, 3200.0)


# covers: AC-10
@pytest.mark.parametrize("base, now, want", [
    ((None, None), (100.0, 200.0), (100.0, 200.0)),  # the first segment
    ((500.0, 3000.0), (100.0, 200.0), (500.0, 3000.0)),  # a smaller resume never lowers it
    ((50.0, 60.0), (100.0, 200.0), (100.0, 200.0)),
])
def test_peaks_are_the_maximum_over_segments(tmp_path, monkeypatch, base, now, want):
    rec = RunRecord(tmp_path, keys=training.RUN_KEYS, defaults={})
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: now[0] * 2**20)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: now[1] * 2**20)
    training._record_progress(rec, SimpleNamespace(type="cuda"),
                              training.Progress(base_peaks=base))
    assert (rec.data["peak_vram_allocated_mib"], rec.data["peak_vram_reserved_mib"]) == want


# covers: AC-10
def test_run_json_is_current_on_disk_at_every_val_point(ten, monkeypatch):
    on_disk, real = [], training.validate

    def spy(*a, **k):
        on_disk.append(rec_of(only_run(ten.out)))
        return real(*a, **k)

    monkeypatch.setattr(training, "validate", spy)
    assert train.main(ten.train_argv) == 0
    second = on_disk[1]  # as the val at 10 starts: what the val at 5 wrote
    assert second["status"] == "running" and second["iterations"] == 5
    assert second["seconds_per_iteration"] > 0 and second["loss_window"] == 2
    assert [v["iteration"] for v in second["val_history"]] == [5]
    assert second["best_iteration"] == 5


# covers: AC-12
def test_out_of_memory_during_val_fails_and_can_resume(ten, monkeypatch):
    real, calls = training.validate, []

    def validate(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return real(*a, **k)

    monkeypatch.setattr(training, "validate", validate)
    assert train.main(ten.train_argv) == 1
    run = only_run(ten.out)
    rec = rec_of(run)
    assert rec["status"] == "failed" and rec["failed_iteration"] == 10
    assert rec["failure"] is None and last_pt(run)["iteration"] == 5
    assert resume(run) == 0
    assert iterations_in(run / "train_log.csv") == list(range(1, 11))
    assert [v["iteration"] for v in rec_of(run)["val_history"]] == [5, 10]


# covers: AC-12
def test_ctrl_c_right_after_last_pt_stays_resumable(ten, monkeypatch):
    """last.pt is written before the val row of train_log.csv: the same window a killed
    process hits. AC-12 says the last last.pt stays usable for --resume."""
    real, fired = RunRecord.write, []

    def write(self):
        if self.data["val_history"] and not fired:
            fired.append(1)
            raise KeyboardInterrupt  # the rec.write() that follows last.pt in the val point
        real(self)

    monkeypatch.setattr(RunRecord, "write", write)
    assert train.main(ten.train_argv) == 130
    run = only_run(ten.out)
    assert rec_of(run)["status"] == "interrupted" and last_pt(run)["iteration"] == 5
    assert resume(run) == 0
    assert iterations_in(run / "train_log.csv") == list(range(1, 11))


# covers: AC-11, AC-12
def test_every_last_pt_has_its_val_row_on_disk_already(ten, monkeypatch):
    """A killed process closes no file, so the row must be flushed before last.pt lands."""
    real, on_disk = training.save_checkpoint, {}

    def spy(path, **state):
        if path.name == "last.pt":
            on_disk[state["iteration"]] = iterations_in(only_run(ten.out) / "train_log.csv")
        real(path, **state)

    monkeypatch.setattr(training, "save_checkpoint", spy)
    assert train.main(ten.train_argv) == 0
    assert on_disk == {5: list(range(1, 6)), 10: list(range(1, 11))}
    with open(only_run(ten.out) / "train_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["iteration"] for r in rows if r["val_score"]] == ["5", "10"]
    assert all(rows[i]["val_wt"] and rows[i]["val_tc"] and rows[i]["val_et"] for i in (4, 9))


# covers: AC-12
def test_an_unexpected_error_marks_the_run_failed_and_still_raises(ten, monkeypatch):
    real = training.Trainer.step

    def step(self, it, *a, **k):
        if it == 6:
            raise ValueError("bad patch\nsecond line")
        return real(self, it, *a, **k)

    monkeypatch.setattr(training.Trainer, "step", step)
    with pytest.raises(ValueError, match="bad patch"):
        train.main(ten.train_argv)
    run = only_run(ten.out)
    rec = rec_of(run)
    assert rec["status"] == "failed" and rec["reason"] == "ValueError: bad patch"
    assert rec["failed_iteration"] == 6 and not (run / "train.pid").exists()
