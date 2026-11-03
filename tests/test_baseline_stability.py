"""Spec 0003, build plan 8 (stability fix): gradient clipping, the grad_norm column, the
grad_clip config check and the run.json failure diagnostics, on synthetic cases with a tiny CPU
config. Covers AC-7, AC-10, AC-12 (verify.md, Stability fix checks)."""
import csv
import json
import math

import pytest
import torch
from common import training
from common.runlog import dump_config
from common.seed import set_seed
from conftest import only_run
from test_baseline_thread import preprocess, train

HEADER = ["iteration", "loss", "dice_loss", "bce_loss", "lr", "grad_norm", "seconds",
          "val_wt", "val_tc", "val_et", "val_score"]


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


def strict_json(path):
    """run.json read with a reader that refuses NaN and Infinity."""
    def refuse(c):
        raise ValueError(f"not strict JSON: {c}")
    return json.loads(path.read_text(), parse_constant=refuse)


def trainer(setup):
    return training.Trainer(setup.cfg, torch.device("cpu"), setup.train,
                            set_seed(setup.cfg["seed"]))


# covers: AC-7, AC-10
def test_grad_norm_column_is_finite_on_every_row(ten):
    assert train.main(ten.train_argv) == 0
    with open(only_run(ten.out) / "train_log.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == HEADER == training.LOG_COLUMNS
    norms = [float(r[HEADER.index("grad_norm")]) for r in rows[1:]]
    assert len(norms) == 10 and all(math.isfinite(n) and n > 0 for n in norms)
    assert strict_json(only_run(ten.out) / "run.json")["failure"] is None


# covers: AC-7
def test_clip_bounds_the_gradient_and_logs_the_norm_before_it(ten):
    clip = 1e-3
    ten.cfg["train"]["grad_clip"] = clip
    tr = trainer(ten)
    s = tr.step(1)
    after = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad) for p in tr.model.parameters()
                     if p.grad is not None]))
    assert after <= clip + 1e-6
    assert s["grad_norm"] > clip  # the logged value is the norm before the clip


# covers: AC-7
@pytest.mark.parametrize("value", [0, 0.0, -1.0, None, True, "12"])
def test_bad_grad_clip_is_refused(ten, value):
    if value is None:
        del ten.cfg["train"]["grad_clip"]
    else:
        ten.cfg["train"]["grad_clip"] = value
    assert any("grad_clip" in p for p in training.check_config(ten.cfg))
    dump_config(ten.cfg, ten.config)
    assert train.main(ten.train_argv) == 2
    assert not ten.out.exists() or not any(ten.out.iterdir())


# covers: AC-12
def test_non_finite_loss_fills_failure_then_resume_clears_it(ten, monkeypatch):
    monkeypatch.setenv("BASELINE_INJECT_INF", "7")
    assert train.main(ten.train_argv) == 1
    run = only_run(ten.out)
    rec = strict_json(run / "run.json")
    assert tuple(rec) == training.RUN_KEYS and training.RUN_KEYS[-1] == "failure"
    f = rec["failure"]
    assert tuple(f) == training.FAILURE_KEYS
    (case,) = f["cases"]
    assert case in ten.train
    assert rec["reason"] == f"non finite loss at iteration 7 in {case}"
    assert f["iteration"] == 7 == rec["failed_iteration"]
    assert f["n_nonfinite_input"] > 0 and f["n_nonfinite_logits"] > 0
    assert f["max_abs_finite_logit"] is None
    assert f["params_finite"] is True and f["grad_scale"] is None  # CPU: scaler off
    assert f["first_nonfinite_module"]  # the first conv the input meets
    assert f["fp32_loss_finite"] is False and f["diagnostic_error"] is None
    assert {f["dice_loss"], f["bce_loss"]} & {"nan", "inf", "-inf"}

    monkeypatch.delenv("BASELINE_INJECT_INF")
    assert train.main(["--resume", str(run), "--device", "cpu"]) == 0
    rec = strict_json(run / "run.json")
    assert rec["status"] == "ok" and rec["failure"] is None


# covers: AC-12
def test_a_diagnostic_pass_that_raises_keeps_the_original_reason(ten, monkeypatch):
    def boom(self, x, amp):
        raise RuntimeError("diagnostic boom\nsecond line")

    monkeypatch.setattr(training.Trainer, "_forward", boom)
    monkeypatch.setenv("BASELINE_INJECT_INF", "3")
    assert train.main(ten.train_argv) == 1
    rec = strict_json(only_run(ten.out) / "run.json")
    assert rec["status"] == "failed"
    assert rec["reason"].startswith("non finite loss at iteration 3 in ")
    f = rec["failure"]
    assert f["diagnostic_error"] == "diagnostic boom"
    assert f["first_nonfinite_module"] is None and f["fp32_loss_finite"] is None
    assert f["n_nonfinite_input"] > 0  # the stats from the failing pass are still there


# covers: AC-12
def test_out_of_memory_leaves_failure_null(ten, monkeypatch):
    real = training.Trainer.step

    def step(self, it, *a, **k):
        if it == 6:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return real(self, it, *a, **k)

    monkeypatch.setattr(training.Trainer, "step", step)
    assert train.main(ten.train_argv) == 1
    rec = strict_json(only_run(ten.out) / "run.json")
    assert rec["status"] == "failed" and rec["failure"] is None


# covers: AC-12
def test_no_forward_hook_is_left_after_the_diagnostics(ten):
    tr = trainer(ten)
    with pytest.raises(training.NonFiniteLoss) as e:
        tr.step(1, inject_at=1)
    assert e.value.failure["first_nonfinite_module"] is not None
    assert all(not m._forward_hooks for m in tr.model.modules())
