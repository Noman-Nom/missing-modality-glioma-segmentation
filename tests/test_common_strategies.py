"""Spec 0005: the strategy hooks, the presence channels, PresenceUNet and the seal. Covers AC-1 to
AC-6 and AC-11, with no training (tests/test_strategy_runs.py runs them end to end)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from common.data import MODALITIES
from common.evaluation import protocol_hash
from common.missing import PATTERNS, apply_pattern
from common.model import PresenceUNet, build_model, predict
from common.runlog import config_hash, load_config
from common.strategies import (
    append_presence,
    draw_patterns,
    dropout_hook,
    keep_batch,
    keep_image,
    presence_hook,
    presence_image_hook,
    strategy_hooks,
    strategy_name,
    strategy_problems,
)
from monai.data import MetaTensor
from monai.networks.nets import UNet

REPO = Path(__file__).resolve().parents[1]
E1_CONFIG = REPO / "experiments" / "0003-baseline" / "config.toml"
EXP = REPO / "experiments" / "0005-strategies"
TINY = {"channels": [4, 8], "strides": [2], "num_res_units": 1}
SEALED_HASH = "146635cf4521a239c4e8cba877528b4c688a6410cd3b817df671db4f1f57825b"


def batch(b=3, size=4):
    rng = torch.Generator().manual_seed(0)
    image = 1 + torch.rand(b, 4, size, size, size, generator=rng)  # no accidental zero
    return {"case": [f"c{i}" for i in range(b)], "image": MetaTensor(image),
            "label": torch.ones(b, 3, size, size, size), "wt": torch.ones(b, 1, size, size, size)}


def tiny_cfg(name=None):
    cfg = {"model": TINY}
    return cfg if name is None else cfg | {"strategy": {"name": name}}


# covers: AC-1
@pytest.mark.parametrize("file, name", [("e3.toml", "dropout"), ("e4.toml", "presence")])
def test_config_is_the_e1_config_plus_the_strategy(file, name):
    cfg = load_config(EXP / file)
    assert cfg.pop("strategy") == {"name": name}
    assert cfg == load_config(E1_CONFIG)


# covers: AC-7
def test_the_e1_config_keeps_its_hash():
    assert config_hash(load_config(E1_CONFIG)).startswith("f9b3afb6")


# covers: AC-2
def test_draws_cover_the_15_patterns_and_nothing_else():
    ids = draw_patterns(np.random.default_rng(43), 15000)
    assert set(ids.tolist()) == set(PATTERNS)
    assert np.bincount(ids)[1:].min() > 800  # about 1,000 each


# covers: AC-2
def test_e3_and_e4_draw_the_same_patterns_at_one_seed():
    dropped = dropout_hook(batch(5), np.random.default_rng(43))["image"]
    both = presence_hook(batch(5), np.random.default_rng(43))["image"]
    assert torch.equal(torch.as_tensor(dropped), torch.as_tensor(both[:, :4]))


# covers: AC-3
def test_dropout_zeroes_each_samples_own_pattern_and_nothing_else():
    before = batch()
    ids = draw_patterns(np.random.default_rng(1), 3)
    after = dropout_hook(before, np.random.default_rng(1))
    assert after["image"].shape == (3, 4, 4, 4, 4)
    for i, p in enumerate(ids.tolist()):
        expected = apply_pattern(before["image"][i], p)
        assert torch.equal(torch.as_tensor(after["image"][i]), torch.as_tensor(expected))
        for c, m in enumerate(MODALITIES):
            assert (after["image"][i, c] == 0).all() == (m in PATTERNS[p].missing)
    assert after["label"] is before["label"] and after["wt"] is before["wt"]
    assert (before["image"] != 0).all()  # the input batch is left alone


# covers: AC-4
def test_presence_channels_of_pattern_6():
    assert PATTERNS[6].present == ("T2", "FLAIR")
    out = append_presence(np.ones((4, 2, 3, 4), np.float16), 6)
    assert out.shape == (8, 2, 3, 4) and out.dtype == np.float32
    assert [out[c].max() for c in range(4, 8)] == [0, 0, 1, 1]
    assert [out[c].min() for c in range(4, 8)] == [0, 0, 1, 1]


# covers: AC-4
@pytest.mark.parametrize("pattern", PATTERNS)
def test_presence_is_the_same_for_numpy_torch_and_metatensor(pattern):
    image = np.random.default_rng(0).random((4, 2, 2, 2), dtype=np.float32)
    out = append_presence(image, pattern)
    for tensor in (torch.as_tensor(image), MetaTensor(image)):
        got = append_presence(tensor, pattern)
        assert isinstance(got, type(tensor)) and got.dtype == torch.float32
        np.testing.assert_array_equal(np.asarray(got), out)


# covers: AC-4
def test_presence_hook_uses_the_pattern_each_sample_drew():
    ids = draw_patterns(np.random.default_rng(2), 6).tolist()
    assert len(set(ids)) > 1
    out = presence_hook(batch(6), np.random.default_rng(2))["image"]
    assert out.shape == (6, 8, 4, 4, 4) and out.dtype == torch.float32
    for i, p in enumerate(ids):
        for c, m in enumerate(MODALITIES):
            assert (out[i, 4 + c] == float(m in PATTERNS[p].present)).all()
            assert (out[i, c] == 0).all() == (m in PATTERNS[p].missing)


# covers: AC-5
def test_e1_and_e3_models_keep_the_e1_shape():
    cfg = tiny_cfg()
    plain = UNet(spatial_dims=3, in_channels=4, out_channels=3, **TINY)
    for model in (build_model(cfg), build_model(tiny_cfg("dropout"))):
        assert type(model) is UNet
        assert {k: v.shape for k, v in model.state_dict().items()} == {
            k: v.shape for k, v in plain.state_dict().items()}


# covers: AC-5
def test_e4_model_takes_8_channels():
    model = build_model(tiny_cfg("presence"))
    assert isinstance(model, PresenceUNet)
    assert model(torch.ones(1, 8, 16, 16, 16)).shape == (1, 3, 16, 16, 16)
    with pytest.raises(RuntimeError):
        model(torch.ones(1, 4, 16, 16, 16))


# covers: AC-5
@pytest.mark.parametrize("pattern", [1, 6, 15])
def test_padding_gives_the_logits_of_constant_presence(pattern):
    model = build_model(tiny_cfg("presence")).eval()
    constant = append_presence(torch.rand(4, 16, 16, 16), pattern)[None]
    padded = constant.clone()
    padded[:, :, :4] = 0  # what predict() pads: zero in every channel, presence included
    with torch.no_grad():
        expected = model.net(torch.cat([padded[:, :4], constant[:, 4:]], dim=1))
        assert torch.equal(model(padded), expected)


# covers: AC-5
def test_predict_pads_a_small_case_and_the_net_still_sees_true_presence():
    cfg = tiny_cfg("presence") | {
        "infer": {"roi": [16, 16, 16], "sw_batch_size": 1, "overlap": 0.5, "threshold": 0.5},
        "train": {"amp": False}}
    model = build_model(cfg).eval()
    seen = []
    model.net.register_forward_pre_hook(lambda m, a: seen.append(a[0].clone()))
    pred = predict(model, presence_image_hook(np.ones((4, 8, 16, 16), np.float32), 6), cfg,
                   torch.device("cpu"))
    assert pred.shape == (3, 8, 16, 16)
    assert seen
    for x in seen:
        for c, want in zip(range(4, 8), (0, 0, 1, 1)):
            assert (x[:, c] == want).all()


# covers: AC-6
def test_strategy_lookup():
    assert strategy_name(tiny_cfg()) is None
    assert strategy_hooks(tiny_cfg()) == {"batch_hook": keep_batch, "image_hook": keep_image}
    assert strategy_hooks(tiny_cfg("dropout")) == {"batch_hook": dropout_hook,
                                                   "image_hook": keep_image}
    assert strategy_hooks(tiny_cfg("presence")) == {"batch_hook": presence_hook,
                                                    "image_hook": presence_image_hook}


# covers: AC-6
def test_strategy_problems():
    assert strategy_problems(tiny_cfg()) == []
    assert strategy_problems(tiny_cfg("presence")) == []
    assert strategy_problems(tiny_cfg("mean_fill"))
    assert strategy_problems({"strategy": {}})  # a section with no name is not E1


# covers: AC-11
def test_the_seal_holds():
    assert protocol_hash() == SEALED_HASH


# spec 0005, API surface: strategies imports data and missing only, so no cycle
@pytest.mark.parametrize("module", ["strategies", "model", "training"])
def test_each_module_imports_first(module):
    code = f"import sys; sys.path.insert(0, 'experiments'); import common.{module}"
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True, timeout=120)
