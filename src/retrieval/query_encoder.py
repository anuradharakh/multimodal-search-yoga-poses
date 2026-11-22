"""Embed a *new* query (an uploaded photo or a sentence) exactly like the database images."""

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

from src.embedding.backbones import load_backbone
from src.embedding.extractor import embed_batch
from src.embedding.postprocess import Projector, SearchConfig, combine_views
from src.embedding.preprocessing import PREPROCESSORS, to_uint8_tensor


class QueryEncoder:
    def __init__(self, cfg: SearchConfig, device: torch.device, projector: Projector | None = None, size: int = 224):
        self.cfg, self.device, self.size = cfg, device, size
        self.backbones = {name: load_backbone(name, device) for name, _ in cfg.members}
        self.projector = projector or Projector("l2")

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        fn = PREPROCESSORS[self.cfg.preprocess]
        batch = torch.stack([to_uint8_tensor(fn(im, self.size)) for im in images])
        parts = []
        for name, pooling in self.cfg.members:
            feats = embed_batch(self.backbones[name], batch, self.device)
            parts.append(combine_views(feats[(pooling, "orig")], feats[(pooling, "flip")], self.cfg.tta))
        return self.projector.transform(np.hstack(parts))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        model = self.backbones.get(self.cfg.backbone)
        if self.cfg.pooling != "attn" or not hasattr(model, "encode_text"):
            raise ValueError("Text queries need the CLIP configuration (pooling='attn').")
        return model.encode_text(list(texts), self.device).cpu().numpy()
