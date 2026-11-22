"""Shared figure style: one colour-blind-validated palette and recessive axes for every chart."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GOOD, BAD = "#0ca30c", "#d03b3b"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "font.size": 9.5,
        "legend.frameon": False,
    }
)

BACKBONE_LABELS = {
    "resnet50": "ResNet-50 (ImageNet)",
    "efficientnet_v2_s": "EfficientNetV2-S (ImageNet)",
    "convnext_tiny": "ConvNeXt-T (ImageNet)",
    "clip_rn50": "CLIP ResNet-50",
    "clip_vit_b32": "CLIP ViT-B/32 (non-CNN ref.)",
    "ensemble": "Ensemble of 4 CNNs",
}


def save_figure(fig, path: Path, dpi: int = 140) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def thumb(path: str | Path, size: int = 256, square: bool = False) -> Image.Image:
    """Thumbnail; square pads onto a white square so grids line up."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    if not square:
        return img
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def show(ax, img, title: str | None = None, fontsize: float = 9) -> None:
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fontsize)
