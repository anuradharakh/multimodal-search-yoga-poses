"""Collapse a ``(B, C, H, W)`` convolutional feature map into one ``(B, C)`` descriptor per image."""

import torch


def gap(x: torch.Tensor) -> torch.Tensor:
    """Global average pooling — every spatial position contributes equally."""
    return x.mean(dim=(-2, -1))


def gem(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalised-mean pooling (Radenović et al., 2018).

    ``p = 1`` is average pooling and ``p → ∞`` is max pooling; ``p = 3`` up-weights the strongest
    activations (the body) relative to weak, widespread ones (plain background).
    """
    return x.clamp(min=eps).pow(p).mean(dim=(-2, -1)).pow(1.0 / p)
