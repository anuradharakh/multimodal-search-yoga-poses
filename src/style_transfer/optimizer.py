"""Optimise a new image so it matches the photo's content and the artwork's style (L-BFGS)."""

import time
from dataclasses import dataclass, field

import torch
from PIL import Image

from src.style_transfer.losses import content_loss, gram_matrix, style_loss, total_variation
from src.style_transfer.vgg_features import (
    CONTENT_LAYER,
    STYLE_LAYER_SETS,
    VGG19_LAYERS,
    VGGFeatures,
    load_image,
    pil_to_tensor,
    tensor_to_pil,
)


@dataclass
class StyleParams:
    content_weight: float = 1.0
    style_weight: float = 1e6
    tv_weight: float = 1e-5
    steps: int = 300
    size: int = 384
    content_layers: tuple[str, ...] = (CONTENT_LAYER,)
    style_layers: tuple[str, ...] = STYLE_LAYER_SETS["all"]
    init: str = "content"  # "content" | "noise"
    seed: int = 0


@dataclass
class StyleResult:
    image: Image.Image
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


class StyleTransfer:
    """Builds the VGG trunk once and reuses it for many (content, style) runs."""

    ALL_LAYERS = tuple(
        sorted({*STYLE_LAYER_SETS["all"], *STYLE_LAYER_SETS["deep"], CONTENT_LAYER}, key=VGG19_LAYERS.get)
    )

    def __init__(self, device: torch.device):
        self.device = device
        self.vgg = VGGFeatures(self.ALL_LAYERS).to(device)

    def run(self, content: Image.Image, style: Image.Image, params: StyleParams | None = None) -> StyleResult:
        params = params or StyleParams()
        torch.manual_seed(params.seed)
        c = pil_to_tensor(load_image(content, params.size), self.device)
        s = pil_to_tensor(load_image(style, params.size), self.device)
        with torch.no_grad():
            content_targets = {n: f.detach() for n, f in self.vgg(c).items() if n in params.content_layers}
            style_targets = {n: gram_matrix(f).detach() for n, f in self.vgg(s).items() if n in params.style_layers}

        # Starting from the photo (not noise) keeps the pose geometry pinned while style is added.
        x = (c.clone() if params.init == "content" else torch.rand_like(c)).requires_grad_(True)
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=params.steps, history_size=50, line_search_fn="strong_wolfe")
        history: list[dict] = []

        def closure():
            opt.zero_grad()
            feats = self.vgg(x)
            lc = content_loss(feats, content_targets, params.content_layers)
            ls = style_loss(feats, style_targets, params.style_layers)
            ltv = total_variation(x)
            total = params.content_weight * lc + params.style_weight * ls + params.tv_weight * ltv
            total.backward()
            history.append({"content": lc.item(), "style": ls.item(), "tv": ltv.item(), "total": total.item()})
            return total

        t0 = time.perf_counter()
        opt.step(closure)
        with torch.no_grad():
            x.clamp_(0, 1)
        if self.device.type == "mps":
            torch.mps.synchronize()
        return StyleResult(tensor_to_pil(x), history, time.perf_counter() - t0)
