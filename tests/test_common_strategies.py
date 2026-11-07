"""Spec 0005: the strategy hooks, the presence channels, PresenceUNet and the seal. Covers AC-1 to
AC-6 and AC-11, with no training (tests/test_strategy_runs.py runs them end to end)."""
import subprocess
import sys
from functools import partial
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
    draw_exposure_patterns,
    draw_patterns,
    dropout_hook,
    expected_draw_counts,
    exposure_hook,
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
SEALED_HASH = "0bc76bfaba23b0d7b35775038b10a28b21650e5da374de4351374dfaa617f642"


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


# spec 0006, AC-3: values written down from the code before E3b was added. A change here means the
# seed 42 runs of E3 and E4 no longer match the seeds to come, so never repin these.
def test_old_pattern_draw_is_unchanged():
    assert draw_patterns(np.random.default_rng(42), 12).tolist() == [
        2, 12, 10, 7, 7, 13, 2, 11, 4, 2, 8, 15]


OLD_MASKS = [[1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 0], [1, 1, 1, 1]]
OLD_NEXT = 593941932  # what the generator gives next: the hooks use it up exactly as before


def channel_means(out):
    return out.mean(dim=(2, 3, 4)).int().tolist()


def test_old_dropout_hook_is_unchanged():
    rng = np.random.default_rng(43)
    out = dropout_hook({"image": torch.ones(6, 4, 2, 2, 2)}, rng)["image"]
    assert channel_means(out) == OLD_MASKS
    assert int(rng.integers(0, 2**31)) == OLD_NEXT


def test_old_presence_hook_is_unchanged():
    rng = np.random.default_rng(43)
    out = presence_hook({"image": torch.ones(6, 4, 2, 2, 2)}, rng)["image"]
    assert channel_means(out) == [m + m for m in OLD_MASKS]
    assert int(rng.integers(0, 2**31)) == OLD_NEXT


# spec 0006, AC-2
def test_e3b_config_is_e3_with_only_the_strategy_changed():
    e3b = load_config(REPO / "experiments" / "0007-confirmation" / "e3b.toml")
    assert e3b.pop("strategy") == {"name": "exposure", "complete_share": 0.5}
    e3 = load_config(EXP / "e3.toml")
    del e3["strategy"]
    assert e3b == e3


def test_exposure_draw_is_two_calls_in_the_fixed_order():
    """The rule, worked by hand from the generator's own two calls."""
    rng = np.random.default_rng(7)
    first, second = rng.random(1000), rng.integers(2, 16, size=1000)
    got = draw_exposure_patterns(np.random.default_rng(7), 1000, 0.5)
    assert got.tolist() == np.where(first < 0.5, 1, second).tolist()


def test_exposure_shares_complete_input_and_spreads_the_rest():
    ids = draw_exposure_patterns(np.random.default_rng(1), 140000, 0.5)
    counts = np.bincount(ids, minlength=16)
    assert PATTERNS[1].missing == ()
    assert counts[1] == pytest.approx(70000, abs=700)
    assert set(ids.tolist()) == set(PATTERNS)
    assert counts[2:].min() > 4700 and counts[2:].max() < 5300  # about 5,000 each


def test_exposure_batch_hook_changes_only_the_drawn_channels():
    b = batch(b=32)
    rng = np.random.default_rng(3)
    ids = draw_exposure_patterns(np.random.default_rng(3), 32, 0.5)
    out = exposure_hook(b, rng, 0.5)
    assert out["image"].shape == b["image"].shape  # 4 channels, no presence channels
    assert {1} < set(ids.tolist())
    for i, p in enumerate(ids):
        assert torch.equal(out["image"][i], apply_pattern(b["image"][i], int(p)))
    assert out["label"] is b["label"]


def test_exposure_hooks_bind_the_share_from_the_config():
    cfg = tiny_cfg("exposure")
    cfg["strategy"]["complete_share"] = 0.25
    hooks = strategy_hooks(cfg)
    assert isinstance(hooks["batch_hook"], partial)
    assert hooks["batch_hook"].func is exposure_hook
    assert hooks["batch_hook"].keywords == {"complete_share": 0.25}
    assert hooks["image_hook"] is keep_image
    assert next(build_model(cfg).parameters()).shape[1] == 4


@pytest.mark.parametrize("strategy, share", [("dropout", None), ("exposure", 0.5)])
def test_expected_counts_are_the_draws_the_hook_really_makes(strategy, share):
    section = {"name": strategy} | ({"complete_share": share} if share else {})
    cfg = {"seed": 42, "train": {"batch_size": 2}, "strategy": section}
    rng = np.random.default_rng(43)  # the Trainer seeds its generator with seed + 1
    seen = []
    hook = strategy_hooks(cfg)["batch_hook"]
    for _ in range(50):
        image = hook({"image": torch.ones(2, 4, 1, 1, 1)}, rng)["image"]
        seen += [sum(2 ** i for i, x in enumerate(s) if x) for s in channel_means(image)]
    counts = expected_draw_counts(cfg, 50)
    assert sum(counts.values()) == 100
    assert counts[1] == seen.count(15)  # all four channels kept means pattern 1


# spec 0006, AC-4
@pytest.mark.parametrize("share", [0, 1, -0.1, 1.5, float("nan"), "0.5", True])
def test_exposure_refuses_a_share_out_of_range(share):
    cfg = tiny_cfg("exposure")
    cfg["strategy"]["complete_share"] = share
    assert any("complete_share" in e for e in strategy_problems(cfg))


def test_exposure_needs_a_share():
    assert any("complete_share" in e for e in strategy_problems(tiny_cfg("exposure")))


@pytest.mark.parametrize("name", ["dropout", "presence"])
def test_other_strategies_refuse_a_share(name):
    cfg = tiny_cfg(name)
    cfg["strategy"]["complete_share"] = 0.5
    assert any("complete_share" in e for e in strategy_problems(cfg))


def test_a_good_share_passes():
    cfg = tiny_cfg("exposure")
    cfg["strategy"]["complete_share"] = 0.5
    assert strategy_problems(cfg) == []


# spec 0005, API surface: strategies imports data and missing only, so no cycle
@pytest.mark.parametrize("module", ["strategies", "model", "training"])
def test_each_module_imports_first(module):
    code = f"import sys; sys.path.insert(0, 'experiments'); import common.{module}"
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True, timeout=120)
