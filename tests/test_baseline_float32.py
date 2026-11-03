"""Spec 0003, build plan 9 (float32): amp off in the committed config, the 9 hour memory test
gate, the amp check in check_baseline.py and the gated Kaggle cell 4. Covers AC-6, AC-7, AC-16
(verify.md, float32 checks)."""
import json
import re

import pytest
import torch
from common.runlog import config_hash, dump_config, load_config
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
