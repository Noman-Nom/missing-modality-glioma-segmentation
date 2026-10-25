"""Spec 0001: the shared 3D U-Net and sliding window prediction (CPU, tiny config)."""
import numpy as np
import pytest
import torch
from common.model import build_model, predict


def tiny_cfg(threshold=0.5):
    return {"model": {"channels": [4, 8], "strides": [2], "num_res_units": 1},
            "infer": {"roi": [16, 16, 16], "sw_batch_size": 1, "overlap": 0.25,
                      "threshold": threshold},
            "train": {"amp": True}}


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model(tiny_cfg()).eval()


def test_model_maps_4_modalities_to_3_regions(model):
    out = model(torch.zeros(1, 4, 16, 16, 16))
    assert out.shape == (1, 3, 16, 16, 16)


def test_predict_returns_bool_masks_with_the_input_shape(model):
    # a shape that is not a multiple of the roi, so sliding windows overlap and pad
    image = np.random.default_rng(0).normal(size=(4, 20, 24, 29)).astype(np.float32)
    pred = predict(model, image, tiny_cfg(), torch.device("cpu"))
    assert isinstance(pred, np.ndarray)
    assert pred.dtype == bool and pred.shape == (3, 20, 24, 29)


def test_predict_applies_the_config_threshold(model):
    image = np.random.default_rng(0).normal(size=(4, 16, 16, 16)).astype(np.float32)
    cpu = torch.device("cpu")
    # sigmoid is strictly inside (0, 1) for these logits
    assert predict(model, image, tiny_cfg(threshold=0.0), cpu).all()
    assert not predict(model, image, tiny_cfg(threshold=1.0), cpu).any()


def test_predict_is_repeatable(model):
    image = np.random.default_rng(1).normal(size=(4, 18, 18, 18)).astype(np.float32)
    cpu = torch.device("cpu")
    assert np.array_equal(predict(model, image, tiny_cfg(), cpu),
                          predict(model, image, tiny_cfg(), cpu))
