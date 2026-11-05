"""Spec 0005: E3 and E4 through the real E1 and E2 scripts on synthetic cases, on CPU. Covers
AC-2, AC-6, AC-7, AC-8, AC-9 and AC-10."""
import csv
import importlib.util
import shutil
from pathlib import Path

import pytest
import torch
from common import evaluation, training
from common.runlog import dump_config, load_config, load_run
from common.seed import set_seed
from common.strategies import dropout_hook, keep_batch, strategy_hooks
from conftest import only_run

REPO = Path(__file__).resolve().parents[1]
E1 = REPO / "experiments" / "0003-baseline"
E2 = REPO / "experiments" / "0004-missing-eval"


def load(folder, name):
    path = folder / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"s5_{folder.name[:4]}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


preprocess, train, evaluate = (load(E1, n) for n in ("preprocess", "train", "evaluate"))
sweep_script = load(E2, "evaluate")


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


def with_strategy(setup, name, **train_settings):
    """Give the tiny config a strategy and build its cache. Returns the train argv."""
    setup.cfg["strategy"] = {"name": name}
    setup.cfg["train"].update(train_settings)
    dump_config(setup.cfg, setup.config)
    assert preprocess.main(setup.pre_argv) == 0
    return setup.train_argv


def rows_of(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# covers: AC-7, AC-8, AC-9, AC-10
@pytest.mark.parametrize("name", ["dropout", "presence"])
def test_a_strategy_runs_through_the_e1_and_e2_scripts(setup, name, tmp_path):
    assert train.main(with_strategy(setup, name)) == 0
    run = only_run(setup.out)
    rec = load_run(run)
    assert rec["status"] == "ok" and rec["best_iteration"] == 2
    assert load_config(run / "config.toml")["strategy"] == {"name": name}
    assert {r["pattern_id"] for r in rows_of(run / "val_log.csv")} == {"1"}
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 0
    assert len(rows_of(run / "metrics_val.csv")) == len(setup.val) * 3
    out = tmp_path / "evals"
    assert sweep_script.main(["--run", str(run), "--fold", "val", "--out", str(out),
                              "--device", "cpu", "--quick"]) == 0
    (ev,) = out.iterdir()
    assert load_run(ev)["status"] == "ok"
    assert len(rows_of(ev / "metrics.csv")) == len(setup.val) * 15 * 3


# covers: AC-7. E3 has the E1 shape, so a lost hook would not crash; it would quietly train E1.
def test_e3_really_trains_on_dropped_input(setup):
    assert preprocess.main(setup.pre_argv) == 0
    argv = setup.train_argv

    def losses(name):
        if name:
            setup.cfg["strategy"] = {"name": name}
            dump_config(setup.cfg, setup.config)
        assert train.main(argv) == 0
        run = max(setup.out.iterdir())  # ids sort by time, the suffix only on a clash
        losses = [r["loss"] for r in rows_of(run / "train_log.csv")]
        shutil.rmtree(setup.out)
        return losses

    assert losses(None) != losses("dropout")


# covers: AC-10
def test_the_sweep_gives_the_e4_model_8_channels(setup, tmp_path, monkeypatch):
    assert train.main(with_strategy(setup, "presence")) == 0
    run = only_run(setup.out)
    seen, real = [], evaluation.predict

    def spy(model, image, cfg, device):
        seen.append(image.shape[0])
        return real(model, image, cfg, device)

    monkeypatch.setattr(evaluation, "predict", spy)
    assert sweep_script.main(["--run", str(run), "--fold", "val", "--out", str(tmp_path / "e"),
                              "--device", "cpu", "--quick"]) == 0
    assert seen == [8] * 15 * len(setup.val)


# covers: AC-7. A resume without the hooks would feed 4 channels to the 8 channel net and crash.
def test_an_e4_run_resumes_with_its_hooks(setup, monkeypatch):
    argv = with_strategy(setup, "presence", iterations=10, val_every=5, warmup=2)
    monkeypatch.setenv("BASELINE_INJECT_INF", "7")
    assert train.main(argv) == 1
    run = only_run(setup.out)
    monkeypatch.delenv("BASELINE_INJECT_INF")
    assert train.main(["--resume", str(run), "--device", "cpu"]) == 0
    rec = load_run(run)
    assert rec["status"] == "ok" and rec["iterations"] == 10 and len(rec["resumes"]) == 1


# covers: AC-6
def test_an_unknown_strategy_stops_a_new_run_and_a_resume(setup, monkeypatch):
    argv = with_strategy(setup, "dropout", iterations=10, val_every=5, warmup=2)
    bad = dict(setup.cfg, strategy={"name": "mean_fill"})
    dump_config(bad, setup.config)
    assert train.main(argv) == 2
    assert not setup.out.exists()

    dump_config(setup.cfg, setup.config)
    monkeypatch.setenv("BASELINE_INJECT_INF", "7")
    assert train.main(argv) == 1
    run = only_run(setup.out)
    dump_config(bad, run / "config.toml")
    assert train.main(["--resume", str(run), "--device", "cpu"]) == 2


# covers: AC-2. The hook has its own stream, so the loader gives the same patches either way.
def test_a_drawing_hook_does_not_change_the_patches(setup):
    with_strategy(setup, "dropout")

    def first_patches(hook):
        gen = set_seed(setup.cfg["seed"])
        tr = training.Trainer(setup.cfg, torch.device("cpu"), setup.train, gen)
        out = []
        for i in range(1, 4):
            tr.step(i, hook)
            out.append(next(tr.batches)["label"].clone())
        return out

    for a, b in zip(first_patches(keep_batch), first_patches(dropout_hook)):
        assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))
    assert strategy_hooks(setup.cfg)["batch_hook"] is dropout_hook


# covers: AC-3. The real loader's batch through Trainer.step, so the hook meets the true type.
@pytest.mark.parametrize("name", ["dropout", "presence"])
def test_the_real_batch_passes_through_the_hook(setup, name):
    with_strategy(setup, name)
    gen = set_seed(setup.cfg["seed"])
    tr = training.Trainer(setup.cfg, torch.device("cpu"), setup.train, gen)
    assert tr.step(1, strategy_hooks(setup.cfg)["batch_hook"])["loss"] > 0
