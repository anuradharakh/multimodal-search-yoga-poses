"""Turn a PIL image into the fixed-size square a CNN expects.

Two strategies are compared during tuning:

* ``crop`` — resize the short side to 224 and take the centre square (the ImageNet default);
* ``pad``  — *letterbox*: resize the long side to 224 and pad to a square with white, so hands and
  feet near the border — which define many yoga poses — are never cropped away.
"""

import numpy as np
import torch
from PIL import Image

from src.data.cleaning import to_rgb

PAD_COLOR = (255, 255, 255)


def letterbox(img: Image.Image, size: int = 224, fill=PAD_COLOR) -> Image.Image:
    img = to_rgb(img)
    scale = size / max(img.size)
    resized = img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.BICUBIC
    )
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def center_crop(img: Image.Image, size: int = 224) -> Image.Image:
    img = to_rgb(img)
    scale = size / min(img.size)
    img = img.resize(
        (max(size, round(img.width * scale)), max(size, round(img.height * scale))), Image.Resampling.BICUBIC
    )
    left, top = (img.width - size) // 2, (img.height - size) // 2
    return img.crop((left, top, left + size, top + size))


PREPROCESSORS = {"pad": letterbox, "crop": center_crop}


def to_uint8_tensor(img: Image.Image) -> torch.Tensor:
    """``(3, H, W)`` uint8 tensor; normalisation happens on the GPU inside each backbone."""
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)
