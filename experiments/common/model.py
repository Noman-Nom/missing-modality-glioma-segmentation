"""The compact 3D U-Net and sliding window prediction, shared by training and evaluation (spec 0001)."""
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from common.data import MODALITIES, REGIONS


def build_model(cfg):
    """MONAI 3D UNet, 4 channels in (MODALITIES order), 3 sigmoid region channels out (REGIONS)."""
    m = cfg["model"]
    return UNet(spatial_dims=3, in_channels=len(MODALITIES), out_channels=len(REGIONS),
                channels=m["channels"], strides=m["strides"], num_res_units=m["num_res_units"])


@torch.no_grad()
def predict(model, image, cfg, device):
    """Normalised image (4, X, Y, Z) NumPy -> binary region masks (3, X, Y, Z) NumPy bool.

    Buffers stay float32; only the model call runs under fp16 autocast (on CUDA with amp on).
    """
    inf = cfg["infer"]
    amp = cfg["train"]["amp"] and device.type == "cuda"

    def net(x):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            return model(x).float()

    x = torch.as_tensor(image, dtype=torch.float32)[None].to(device)
    logits = sliding_window_inference(
        x, roi_size=inf["roi"], sw_batch_size=inf["sw_batch_size"], predictor=net,
        overlap=inf["overlap"], mode="gaussian", padding_mode="constant", cval=0.0)
    return (torch.sigmoid(logits.float()) > inf["threshold"])[0].cpu().numpy()
