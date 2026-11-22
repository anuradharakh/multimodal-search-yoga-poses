"""Pick the compute device and seed the random number generators."""

import random

import numpy as np
import torch


def get_device(preference: str = "auto") -> torch.device:
    """CUDA → Apple MPS → CPU, unless a specific device is requested."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
