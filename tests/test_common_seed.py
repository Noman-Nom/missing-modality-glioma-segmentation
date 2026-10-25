"""Spec 0001, AC-10: same seed gives bitwise equal losses on CPU."""
import torch
from common.seed import determinism_state, set_seed
from monai.losses import DiceLoss
from monai.networks.nets import UNet


def three_steps(seed):
    g = set_seed(seed)
    model = UNet(spatial_dims=3, in_channels=4, out_channels=3, channels=(4, 8), strides=(2,),
                 num_res_units=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = DiceLoss(sigmoid=True)
    losses = []
    for _ in range(3):
        x = torch.randn(1, 4, 32, 32, 32, generator=g)
        y = (torch.rand(1, 3, 32, 32, 32, generator=g) > 0.5).float()
        loss = loss_fn(model(x), y) + torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_same_seed_same_loss():
    try:
        a, b = three_steps(42), three_steps(42)
        assert a == b
        assert three_steps(7) != a
        s = determinism_state()
        assert s["deterministic_algorithms"] and s["warn_only"] and s["cudnn_deterministic"]
        assert not s["cudnn_benchmark"] and s["cublas_workspace_config"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(False)
