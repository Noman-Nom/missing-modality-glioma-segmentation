"""Environment smoke test (ADR 0001, AC-3 and AC-4).

Fast tests (run by default):   uv run pytest tests/test_env_smoke.py -v
GPU test (human runs it):      uv run pytest tests/test_env_smoke.py -m slow -v -s

AC-3: a small MONAI 3D U-Net, forward and backward with autocast, peak GPU memory < 4 GB.
AC-4: Dice and HD95 on toy cases with known answers; NiBabel loads one BraTS 2023 case.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet

REPO = Path(__file__).resolve().parents[1]
BRATS_TRAIN = Path(
    os.environ.get(
        "BRATS_TRAIN_DIR",
        REPO / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    )
)
CASE = "BraTS-GLI-00000-000"
# BraTS 2023 GLI file suffixes. Order here is only for this test, not a model channel order.
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
LABELS_GLI = {0, 1, 2, 3}  # 0 background, 1 NCR, 2 ED, 3 ET (BraTS 2023 GLI)

VRAM_LIMIT_BYTES = 4 * 1024**3


def _one_hot_pair(mask: np.ndarray) -> torch.Tensor:
    """Binary (D,H,W) mask -> one-hot tensor (1, 2, D, H, W): background, foreground."""
    fg = torch.from_numpy(mask.astype(np.float32))
    return torch.stack([1 - fg, fg])[None]


# ---------------------------------------------------------------- AC-3 (GPU)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_unet_forward_backward_autocast_gpu():
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randn(1, 4, 96, 96, 96, device=device)
    y = torch.randint(0, 2, (1, 3, 96, 96, 96), device=device).float()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nGPU: {torch.cuda.get_device_name(0)} | params: {n_params:,} | loss: {loss.item():.4f}"
        f"\nPEAK allocated: {peak / 1024**2:.0f} MiB | reserved: {reserved / 1024**2:.0f} MiB"
        f" | limit: {VRAM_LIMIT_BYTES / 1024**2:.0f} MiB"
    )

    assert logits.shape == (1, 3, 96, 96, 96)
    assert torch.isfinite(loss).all()
    assert all(p.grad is not None for p in model.parameters())
    assert peak < VRAM_LIMIT_BYTES


# ---------------------------------------------------------------- AC-4 (toy metrics)


def test_dice_toy_case_known_answer():
    """Cube A = x 4..7, cube B = x 5..8 (same y, z 4..7). Overlap 3*4*4 = 48, sizes 64 + 64.
    Dice = 2*48 / 128 = 0.75."""
    a = np.zeros((16, 16, 16), dtype=bool)
    b = np.zeros_like(a)
    a[4:8, 4:8, 4:8] = True
    b[5:9, 4:8, 4:8] = True

    metric = DiceMetric(include_background=False, reduction="mean")
    metric(y_pred=_one_hot_pair(a), y=_one_hot_pair(b))
    assert metric.aggregate().item() == pytest.approx(0.75)

    metric.reset()
    metric(y_pred=_one_hot_pair(a), y=_one_hot_pair(a))
    assert metric.aggregate().item() == pytest.approx(1.0)


def test_hd95_toy_case_known_answer():
    """One voxel vs one voxel 3 voxels apart (spacing 1): every surface distance is 3, so HD95 = 3.
    Identical masks give 0."""
    a = np.zeros((16, 16, 16), dtype=bool)
    b = np.zeros_like(a)
    a[8, 8, 4] = True
    b[8, 8, 7] = True

    metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    metric(y_pred=_one_hot_pair(a), y=_one_hot_pair(b))
    assert metric.aggregate().item() == pytest.approx(3.0)

    metric.reset()
    metric(y_pred=_one_hot_pair(a), y=_one_hot_pair(a))
    assert metric.aggregate().item() == pytest.approx(0.0)


# ---------------------------------------------------------------- AC-4 (real BraTS case)


@pytest.mark.skipif(not BRATS_TRAIN.is_dir(), reason=f"BraTS training data not found at {BRATS_TRAIN}")
def test_nibabel_loads_brats_case_read_only():
    case_dir = BRATS_TRAIN / CASE
    assert case_dir.is_dir(), f"missing case folder {case_dir}"

    shapes = {}
    for mod in MODALITIES:
        img = nib.load(case_dir / f"{CASE}-{mod}.nii.gz")
        data = np.asarray(img.dataobj)
        assert data.ndim == 3
        assert np.isfinite(data).all()
        assert data.max() > 0
        shapes[mod] = data.shape
    assert len(set(shapes.values())) == 1, f"modality shapes differ: {shapes}"

    seg_img = nib.load(case_dir / f"{CASE}-seg.nii.gz")
    seg = np.asarray(seg_img.dataobj)
    assert seg.shape == next(iter(shapes.values()))
    assert set(np.unique(seg).astype(int)) <= LABELS_GLI
    assert (seg > 0).any()
    print(f"\n{CASE}: shape {seg.shape}, zooms {seg_img.header.get_zooms()}, labels {sorted(set(np.unique(seg).astype(int)))}")
