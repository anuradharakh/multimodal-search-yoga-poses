"""The three loss terms of neural style transfer.

    L(x) = α · L_content(x, c) + β · L_style(x, s) + γ · L_TV(x)

``β / α`` is the main trade-off knob: too low and the photo barely changes, too high and the pose
dissolves into texture.
"""

import torch
import torch.nn.functional as F


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W)`` → ``(B, C, C)`` channel correlations, normalised by ``C·H·W``.

    Correlations between feature channels capture colours and textures but discard *where* they
    occur — which is exactly what "style" means here.
    """
    b, c, h, w = feat.shape
    f = feat.reshape(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


def content_loss(feats: dict, targets: dict, layers) -> torch.Tensor:
    return sum(F.mse_loss(feats[n], targets[n]) for n in layers)


def style_loss(feats: dict, target_grams: dict, layers) -> torch.Tensor:
    return sum(F.mse_loss(gram_matrix(feats[n]), target_grams[n]) for n in layers) / len(layers)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference between neighbouring pixels — a smoothness prior against noise."""
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
