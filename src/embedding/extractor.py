"""Batched, cached feature extraction — the slow part of the pipeline, run once per preprocessing.

For every image we embed the **original and its horizontal mirror** (most poses can be performed
facing either way) with every backbone, and save the raw pooled features to
``data/embeddings/<backbone>__<preprocess>__<pooling>__<view>.npy``. Rows follow ``manifest.csv``.
Later stages combine and post-process these matrices without touching the networks again.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.embedding.backbones import Backbone, load_backbone
from src.embedding.preprocessing import PREPROCESSORS, to_uint8_tensor
from src.utils.config_loader import get_path


class ImageDataset(Dataset):
    def __init__(self, paths: Sequence[str], preprocess: str, size: int):
        self.paths, self.fn, self.size = list(paths), PREPROCESSORS[preprocess], size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            return to_uint8_tensor(self.fn(im, self.size))


def feature_path(config: dict, backbone: str, preprocess: str, pooling: str, view: str) -> Path:
    return get_path(config, "embeddings_dir") / f"{backbone}__{preprocess}__{pooling}__{view}.npy"


@torch.inference_mode()
def embed_batch(backbone: Backbone, batch_u8: torch.Tensor, device: torch.device) -> dict[tuple[str, str], np.ndarray]:
    """``{(pooling, view): features}`` for one uint8 batch, views ``orig`` and ``flip``."""
    x = batch_u8.to(device).float().div_(255.0)
    out = {}
    for view, inp in (("orig", x), ("flip", torch.flip(x, dims=[-1]))):
        for pooling, feats in backbone(inp).items():
            out[(pooling, view)] = feats.float().cpu().numpy()
    return out


def is_cached(config: dict, backbone: str, preprocess: str) -> bool:
    pooling = "attn" if backbone == "clip_vit_b32" else "gap"
    return feature_path(config, backbone, preprocess, pooling, "orig").exists()


def extract_features(
    config: dict, paths: Sequence[str], backbones: Sequence[str], preprocess: str, device
) -> list[str]:
    """Extract every not-yet-cached backbone in one pass over the images; returns what was computed."""
    cfg = config["embedding"]
    todo = [b for b in backbones if cfg.get("overwrite") or not is_cached(config, b, preprocess)]
    if not todo:
        return []
    models = {b: load_backbone(b, device) for b in todo}
    loader = DataLoader(
        ImageDataset(paths, preprocess, cfg["input_size"]),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
    )
    chunks: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for batch in tqdm(loader, desc=f"Embedding [{preprocess}]"):
        for name, model in models.items():
            for (pooling, view), feats in embed_batch(model, batch, device).items():
                chunks.setdefault((name, pooling, view), []).append(feats)
    for (name, pooling, view), parts in chunks.items():
        np.save(feature_path(config, name, preprocess, pooling, view), np.concatenate(parts).astype(np.float32))
    return todo
