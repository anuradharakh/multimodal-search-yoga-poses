"""Text → image retrieval with CLIP-RN50: one query per pose, built from its Sanskrit / English name."""

import numpy as np
import pandas as pd

from src.data.manifest import load_manifest, split_masks
from src.data.pose_names import pose_names
from src.embedding.backbones import load_backbone
from src.embedding.postprocess import build_vectors
from src.evaluation.metrics import retrieval_metrics, topk_indices
from src.indexing.index_builder import CLIP_CONFIG

TEXT_TEMPLATES = {
    "sanskrit": "a photo of a person doing {sanskrit} yoga pose",
    "english": "a photo of a person doing the {english}, a yoga pose",
    "both": "a photo of a person doing {sanskrit} ({english}), a yoga pose",
}


def pose_text_embeddings(clip, template: str, device) -> tuple[list[str], np.ndarray]:
    names = pose_names()
    poses = list(names.index)
    templates = list(TEXT_TEMPLATES.values()) if template == "ensemble" else [TEXT_TEMPLATES[template]]
    per_template = [
        clip.encode_text(
            [t.format(sanskrit=names.loc[p, "sanskrit"], english=names.loc[p, "english"]) for p in poses], device
        )
        .cpu()
        .numpy()
        for t in templates
    ]
    emb = np.mean(per_template, axis=0)
    return poses, emb / np.linalg.norm(emb, axis=1, keepdims=True)


def text_search_eval(config: dict, device) -> pd.DataFrame:
    df = load_manifest(config)
    labels, g, _, test = split_masks(df)
    img, _ = build_vectors(config, CLIP_CONFIG, g)
    clip = load_backbone("clip_rn50", device)
    rows = []
    for template in [*TEXT_TEMPLATES, "ensemble"]:
        poses, txt = pose_text_embeddings(clip, template, device)
        m = retrieval_metrics(labels[g][topk_indices(txt, img[g], 10)], poses, labels[g])
        zero_shot = np.array(poses)[(img[test] @ txt.T).argmax(axis=1)]
        rows.append(
            {
                "prompt": template,
                **{f"text→image {k}": v for k, v in m.items()},
                "zero-shot acc (test)": float((zero_shot == labels[test]).mean()),
            }
        )
    return pd.DataFrame(rows)
