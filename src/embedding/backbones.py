"""Frozen, pre-trained networks used as image-embedding generators.

Every backbone takes a batch of ``[0, 1]`` tensors and returns a dict of pooled descriptors, so a
single forward pass yields every pooling variant compared during tuning:

* ``gap`` / ``gem`` — average / generalised-mean pooling of the last convolutional feature map;
* ``attn`` — CLIP's own attention-pooling head, which projects into CLIP's joint image–text space
  and is what makes **text → image** search possible.

ImageNet models come from ``torchvision``; CLIP weights (OpenAI) from ``open_clip``.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.embedding.pooling import gap, gem

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class Backbone(nn.Module):
    """A frozen feature extractor plus the normalisation constants it was trained with."""

    name: str
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    poolings: tuple[str, ...] = ("gap", "gem")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` in ``[0, 1]`` → model-specific standardisation."""
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    @torch.inference_mode()
    def forward(self, x01: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap = self.feature_map(self.normalize(x01))
        return {"gap": gap(fmap), "gem": gem(fmap)}


class TorchvisionBackbone(Backbone):
    def __init__(self, name: str, build: Callable[[], nn.Module], trunk: Callable[[nn.Module], nn.Module]):
        super().__init__()
        self.name = name
        self.trunk = trunk(build()).eval()

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class ClipBackbone(Backbone):
    """OpenAI CLIP image tower. For ResNet towers we also expose the raw conv map (gap / gem)."""

    mean, std = CLIP_MEAN, CLIP_STD

    def __init__(self, name: str, arch: str, pretrained: str = "openai"):
        super().__init__()
        import open_clip

        self.name = name
        self.model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.is_resnet = hasattr(self.model.visual, "attnpool")
        self.poolings = ("attn", "gap", "gem") if self.is_resnet else ("attn",)

    def _resnet_trunk(self, x: torch.Tensor) -> torch.Tensor:
        v = self.model.visual
        x = v.stem(x.to(v.conv1.weight.dtype))
        return v.layer4(v.layer3(v.layer2(v.layer1(x))))

    @torch.inference_mode()
    def forward(self, x01: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.normalize(x01)
        if not self.is_resnet:
            return {"attn": self.model.encode_image(x)}
        fmap = self._resnet_trunk(x)
        return {"attn": self.model.visual.attnpool(fmap), "gap": gap(fmap), "gem": gem(fmap)}

    @torch.inference_mode()
    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)


def _resnet50() -> TorchvisionBackbone:
    from torchvision.models import ResNet50_Weights, resnet50

    return TorchvisionBackbone(
        "resnet50",
        lambda: resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        lambda m: nn.Sequential(*list(m.children())[:-2]),  # drop avgpool + fc
    )


def _efficientnet_v2_s() -> TorchvisionBackbone:
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

    return TorchvisionBackbone(
        "efficientnet_v2_s",
        lambda: efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1),
        lambda m: m.features,
    )


def _convnext_tiny() -> TorchvisionBackbone:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    return TorchvisionBackbone(
        "convnext_tiny", lambda: convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1), lambda m: m.features
    )


BACKBONES: dict[str, Callable[[], Backbone]] = {
    "resnet50": _resnet50,
    "efficientnet_v2_s": _efficientnet_v2_s,
    "convnext_tiny": _convnext_tiny,
    "clip_rn50": lambda: ClipBackbone("clip_rn50", "RN50-quickgelu"),
    # Non-CNN reference point: CLIP's ViT-B/32 image tower.
    "clip_vit_b32": lambda: ClipBackbone("clip_vit_b32", "ViT-B-32-quickgelu"),
}

CNN_BACKBONES = ("resnet50", "efficientnet_v2_s", "convnext_tiny", "clip_rn50")


def load_backbone(name: str, device: torch.device) -> Backbone:
    if name not in BACKBONES:
        raise KeyError(f"Unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name]().to(device).eval()
