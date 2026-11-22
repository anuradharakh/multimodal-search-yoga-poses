"""Validation grid search over every retrieval configuration.

Every combination of backbone × pooling × preprocessing × flip-TTA × post-processing is scored by
querying the **val** images against the **gallery** with exact cosine search. The best configuration
by ``tuning.selection_metric`` is saved; the **test** split is not looked at here.
"""

import itertools

import pandas as pd
from tqdm.auto import tqdm

from src.data.manifest import load_manifest, split_masks
from src.embedding.postprocess import SearchConfig, build_vectors
from src.evaluation.metrics import evaluate_vectors

# Pooling options each backbone exposes (CLIP ViT has only its attention-pool head).
BACKBONE_POOLINGS = {
    "resnet50": ("gap", "gem"),
    "efficientnet_v2_s": ("gap", "gem"),
    "convnext_tiny": ("gap", "gem"),
    "clip_rn50": ("attn", "gap", "gem"),
    "clip_vit_b32": ("attn",),
    "ensemble": ("mixed",),
}

ABLATION_STEPS: tuple[tuple[str, SearchConfig], ...] = (
    ("ResNet-50, GAP, centre-crop (ImageNet default)", SearchConfig("resnet50", "gap", "crop", False, "l2")),
    ("+ letterbox instead of crop", SearchConfig("resnet50", "gap", "pad", False, "l2")),
    ("+ horizontal-flip TTA", SearchConfig("resnet50", "gap", "pad", True, "l2")),
    ("+ GeM pooling", SearchConfig("resnet50", "gem", "pad", True, "l2")),
    ("+ PCA-whitening (256-d)", SearchConfig("resnet50", "gem", "pad", True, "pcaw256")),
    ("+ LDA projection instead (supervised)", SearchConfig("resnet50", "gem", "pad", True, "lda0.1")),
    ("backbone → CLIP-RN50 (GeM, LDA)", SearchConfig("clip_rn50", "gem", "pad", True, "lda0.1")),
    ("backbone → ensemble of 4 CNNs (LDA)", SearchConfig("ensemble", "mixed", "pad", True, "lda0.1")),
)


def all_configs(config: dict) -> list[SearchConfig]:
    t = config["tuning"]
    backbones = [*config["embedding"]["backbones"], "ensemble"]
    return [
        SearchConfig(bb, pool, prep, tta, post)
        for bb in backbones
        for pool, prep, tta, post in itertools.product(
            BACKBONE_POOLINGS[bb], config["embedding"]["preprocess"], t["tta"], t["post_processing"]
        )
    ]


def grid_search(config: dict, configs: list[SearchConfig] | None = None) -> pd.DataFrame:
    df = load_manifest(config)
    labels, g, v, _ = split_masks(df)
    ks = config["evaluation"]["ks"]
    rows = []
    for cfg in tqdm(configs or all_configs(config), desc="Grid search (val)"):
        vecs, _ = build_vectors(config, cfg, g, labels)
        rows.append({**cfg.to_dict(), "dim": vecs.shape[1], **evaluate_vectors(vecs, labels, g, v, ks)})
    return pd.DataFrame(rows).sort_values(config["tuning"]["selection_metric"], ascending=False).reset_index(drop=True)


def lookup(grid: pd.DataFrame, cfg: SearchConfig) -> pd.Series:
    sel = grid
    for k, v in cfg.to_dict().items():
        sel = sel[sel[k] == v]
    return sel.iloc[0]


def ablation_table(grid: pd.DataFrame) -> pd.DataFrame:
    """One change at a time from the textbook baseline to the selected configuration."""
    rows = []
    for label, cfg in ABLATION_STEPS:
        r = lookup(grid, cfg)
        rows.append({"step": label, "config": cfg.key, **{k: r[k] for k in ("P@1", "P@5", "P@10", "mAP@10")}})
    return pd.DataFrame(rows)


def best_per_backbone(grid: pd.DataFrame) -> pd.DataFrame:
    cols = ["backbone", "pooling", "preprocess", "tta", "post", "dim", "P@1", "P@5", "P@10", "mAP@10"]
    unsup = grid[grid.post.isin(["l2", "pcaw256"])].groupby("backbone").head(1)[cols].assign(setting="unsupervised")
    sup = grid[grid.post.str.startswith("lda")].groupby("backbone").head(1)[cols].assign(setting="+ LDA")
    return pd.concat([unsup, sup]).sort_values(["setting", "mAP@10"], ascending=[False, False]).reset_index(drop=True)
