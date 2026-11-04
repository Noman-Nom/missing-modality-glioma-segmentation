"""Spec 0003, build plan 9 (float32): amp off in the committed config, the 9 hour memory test
gate, the amp check in check_baseline.py and the gated Kaggle cell 4. Covers AC-6, AC-7, AC-16
(verify.md, float32 checks)."""
import json
import re

import pytest
import torch
from common.model import build_model
from common.runlog import config_hash, dump_config, load_config
from monai.networks.nets import UNet
from test_baseline_checks import as_full_run, evaluated  # noqa: F401  (evaluated is a fixture)
from test_baseline_thread import EXP, load

memory_test, check = load("memory_test"), load("check_baseline")


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


# covers: AC-7
def test_committed_config_trains_in_float32():
    assert load_config(EXP / "config.toml")["train"]["amp"] is False


# covers: AC-6
@pytest.mark.parametrize("hours, reason", [
    (9.0, None),
    (2.5, None),
    (9.1, "projected 9.1 h > 9"),
    (None, "projected hours not measured"),
])
def test_the_hours_gate(hours, reason):
    assert memory_test.MAX_HOURS == 9
    assert memory_test.failure_reason(None, 900.0, 3584, hours) == reason


# covers: AC-6
def test_the_failure_order_is_oom_gpu_vram_then_hours():
    f = memory_test.failure_reason
    assert f("CUDA out of memory: x", None, 3584, 99.0) == "CUDA out of memory: x"
    assert f(None, None, 3584, 99.0).startswith("no GPU")
    assert f(None, 3584.0, 3584, 99.0) == "peak reserved VRAM 3584.0 MiB >= 3584"


# covers: AC-16
def test_check_fails_a_run_trained_with_amp_on(evaluated, monkeypatch, capsys):  # noqa: F811
    rec = as_full_run(evaluated, monkeypatch)
    cfg = load_config(evaluated / "config.toml")
    cfg["train"]["amp"] = True
    dump_config(cfg, evaluated / "config.toml")
    (evaluated / "run.json").write_text(json.dumps(dict(rec, config_hash=config_hash(cfg))))
    assert check.main([str(evaluated)]) == 1
    assert "config.toml has amp on" in capsys.readouterr().out


# covers: AC-6
def test_kaggle_cell_4_trains_only_after_a_memory_test_pass():
    text = (EXP / "KAGGLE.md").read_text()
    cell = text[text.index("**Cell 4"):text.index("**Cell 5")]
    code = re.search(r"```python\n(.*?)```", cell, re.DOTALL).group(1)
    gate = code.index("raise SystemExit(1)")
    assert "memory_test.json" in code and 'mem["pass"] is not True' in code
    assert 'mem["config_hash"] != now' in code and "config_hash(load_config(" in code
    assert code.index("train.py") > gate
    compile(code.replace("!uv run", "pass  # "), "cell4", "exec")  # valid Python


# covers: AC-5, AC-6, AC-7, AC-8
def test_committed_config_holds_the_spec_values():
    cfg = load_config(EXP / "config.toml")
    t = cfg["train"]
    assert (t["iterations"], t["val_every"], t["warmup"]) == (40000, 4000, 500)
    assert (t["lr"], t["weight_decay"], t["grad_clip"]) == (1e-3, 1e-5, 12.0)
    assert (t["patch"], t["batch_size"]) == ([128, 128, 128], 1)
    assert (t["flip_p"], t["intensity_scale"], t["intensity_shift"]) == (0.5, 0.1, 0.1)
    assert cfg["memory_test"] == {"steps": 200, "limit_mib": 3584}


# covers: AC-6
def test_committed_config_builds_the_spec_unet():
    model = build_model(load_config(EXP / "config.toml"))
    assert isinstance(model, UNet)
    assert (model.dimensions, model.in_channels, model.out_channels) == (3, 4, 3)
    assert tuple(model.channels) == (32, 64, 128, 256, 320)
    assert tuple(model.strides) == (2, 2, 2, 2) and model.num_res_units == 2
    with torch.no_grad():
        assert model(torch.zeros(1, 4, 32, 32, 32)).shape == (1, 3, 32, 32, 32)
