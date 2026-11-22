"""Frozen VGG-19 feature extractor and image ↔ tensor helpers for neural style transfer.

Gatys et al. read *content* from a deep layer (``conv4_2``: what is where) and *style* from several
layers (``conv1_1 … conv5_1``: textures at increasing scale). Max-pooling is replaced with
average-pooling, which the paper found gives smoother gradients and nicer images.
"""

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.data.cleaning import to_rgb

VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)

# Index of each named conv layer inside torchvision's ``vgg19().features``.
VGG19_LAYERS = {
    "conv1_1": 0, "conv1_2": 2,
    "conv2_1": 5, "conv2_2": 7,
    "conv3_1": 10, "conv3_2": 12, "conv3_3": 14, "conv3_4": 16,
    "conv4_1": 19, "conv4_2": 21, "conv4_3": 23, "conv4_4": 25,
    "conv5_1": 28, "conv5_2": 30, "conv5_3": 32, "conv5_4": 34,
}  # fmt: skip

STYLE_LAYER_SETS = {
    "all": ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"),  # Gatys et al. default
    "shallow": ("conv1_1", "conv2_1", "conv3_1"),  # colour and fine grain
    "deep": ("conv3_1", "conv4_1", "conv5_1"),  # larger motifs
}
CONTENT_LAYER = "conv4_2"


def load_image(img: Image.Image | str, size: int) -> Image.Image:
    """Open (if a path) and resize so the longest side is ``size`` pixels."""
    img = to_rgb(Image.open(img)) if isinstance(img, str) else to_rgb(img)
    scale = size / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return arr.unsqueeze(0).contiguous().to(device)  # contiguous: L-BFGS flattens the gradient


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = x.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8))


class VGGFeatures(nn.Module):
    """VGG-19 trunk truncated after the deepest requested layer; returns ``{layer_name: activation}``."""

    def __init__(self, layers: Sequence[str], pooling: str = "avg"):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        trunk = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        modules = []
        for m in trunk[: max(VGG19_LAYERS[n] for n in layers) + 1]:
            if isinstance(m, nn.MaxPool2d) and pooling == "avg":
                m = nn.AvgPool2d(kernel_size=2, stride=2)
            elif isinstance(m, nn.ReLU):
                m = nn.ReLU(inplace=False)
            modules.append(m)
        self.trunk = nn.Sequential(*modules)
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        self.index_to_name = {VGG19_LAYERS[n]: n for n in layers}
        self.register_buffer("mean", torch.tensor(VGG_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(VGG_STD).view(1, 3, 1, 1))

    def forward(self, x01: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x01 - self.mean) / self.std
        out = {}
        for i, m in enumerate(self.trunk):
            x = m(x)
            if i in self.index_to_name:
                out[self.index_to_name[i]] = x
        return out
