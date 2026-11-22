import numpy as np
import torch
from PIL import Image

from src.embedding.pooling import gap, gem
from src.embedding.preprocessing import center_crop, letterbox


def test_letterbox_keeps_whole_image_and_pads_white():
    out = np.asarray(letterbox(Image.new("RGB", (400, 100), "black"), 224))
    assert out.shape == (224, 224, 3)
    assert (out[0, 0] == 255).all() and (out[112, 112] == 0).all()


def test_center_crop_is_square():
    assert center_crop(Image.new("RGB", (300, 500)), 224).size == (224, 224)


def test_gem_between_average_and_max():
    x = torch.rand(2, 8, 7, 7)
    g, a, m = gem(x, p=3), gap(x), x.amax(dim=(-2, -1))
    assert torch.all(g >= a - 1e-6) and torch.all(g <= m + 1e-6)
    torch.testing.assert_close(gem(x, p=1), a, atol=1e-5, rtol=1e-5)
