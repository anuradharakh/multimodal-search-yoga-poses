"""Text → image retrieval with CLIP-RN50: one query per pose, built from its Sanskrit / English name."""

import numpy as np
import pandas as pd

from src.data.manifest import load_manifest, split_masks
from src.data.pose_names import pose_names
from src.embedding.backbones import load_backbone
from src.embedding.postprocess import build_vectors
from src.evaluation.metrics import retrieval_metrics, topk_indices
from src.indexing.index_builder import CLIP_CONFIG
from src.indexing.text_index import pose_descriptions

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


# Query wordings for the name route — none is identical to the stored description.
NAME_QUERIES = {
    "English name only": "{english}",
    "Sanskrit name only": "{sanskrit}",
    "English, in a sentence": "a person doing the {english}",
}


def text_name_route_eval(config: dict, device) -> pd.DataFrame:
    """Text → stored pose-description embeddings → pose. With a correct top-1 pose every returned
    photo shows that pose, so text → image P@10 via this route equals the top-1 accuracy."""
    desc = pose_descriptions()
    clip = load_backbone("clip_rn50", device)
    stored = clip.encode_text(desc["description"].tolist(), device).cpu().numpy()
    poses = desc["pose"].to_numpy()
    rows = []
    for label, template in NAME_QUERIES.items():
        queries = [template.format(sanskrit=s, english=e) for s, e in zip(desc.sanskrit, desc.english, strict=True)]
        q = clip.encode_text(queries, device).cpu().numpy()
        top = np.argsort(-(q @ stored.T), axis=1)[:, :5]
        hit1 = float((poses[top[:, 0]] == poses).mean())
        hit5 = float((poses[top] == poses[:, None]).any(axis=1).mean())
        rows.append({"query wording": label, "pose top-1": hit1, "pose top-5": hit5, "text→image P@10": hit1})
    return pd.DataFrame(rows)
