"""The compact 3D U-Net and sliding window prediction, shared by training and evaluation (spec 0001)."""
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from common.data import MODALITIES, REGIONS
from common.strategies import strategy_name


class PresenceUNet(torch.nn.Module):
    """The E4 network: 4 image channels, then 4 presence channels (spec 0005, AC-5).

    Each presence channel becomes its maximum over the patch. predict() pads with 0, and a 0 there
    would read as missing; a window always holds real voxels, so the maximum is the true value.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = len(MODALITIES)
        image, presence = x[:, :n], x[:, n:]
        presence = presence.amax(dim=(2, 3, 4), keepdim=True).expand_as(presence)
        return self.net(torch.cat([image, presence], dim=1))


def build_model(cfg):
    """MONAI 3D UNet, 4 channels in (MODALITIES order), 3 sigmoid region channels out (REGIONS).
    The presence strategy doubles the channels in and wraps the net in PresenceUNet."""
    m = cfg["model"]
    presence = strategy_name(cfg) == "presence"
    net = UNet(spatial_dims=3, in_channels=len(MODALITIES) * (2 if presence else 1),
               out_channels=len(REGIONS), channels=m["channels"], strides=m["strides"],
               num_res_units=m["num_res_units"])
    return PresenceUNet(net) if presence else net


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
