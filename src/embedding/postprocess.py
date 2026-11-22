"""From cached raw features to final search vectors.

A retrieval configuration (:class:`SearchConfig`) fixes every tunable choice:

=============  ===================================================================================
``backbone``   a backbone name, or ``"ensemble"`` (concatenation of ``ENSEMBLE_MEMBERS``)
``pooling``    ``gap`` / ``gem`` / ``attn`` (``mixed`` for the ensemble)
``preprocess`` ``crop`` or ``pad`` (letterbox)
``tta``        average the original and mirrored embeddings (left/right invariance)
``post``       ``l2`` (none), ``pcaw256`` (PCA-whitening) or ``lda<shrinkage>`` — see :class:`Projector`
=============  ===================================================================================
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.embedding.extractor import feature_path

# Strongest descriptor of each CNN family, concatenated (each member L2-normalised first).
ENSEMBLE_MEMBERS: tuple[tuple[str, str], ...] = (
    ("clip_rn50", "gem"),
    ("resnet50", "gem"),
    ("convnext_tiny", "gap"),
    ("efficientnet_v2_s", "gem"),
)


def l2n(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


@dataclass(frozen=True)
class SearchConfig:
    backbone: str
    pooling: str
    preprocess: str = "pad"
    tta: bool = False
    post: str = "l2"

    @property
    def key(self) -> str:
        return f"{self.backbone}|{self.pooling}|{self.preprocess}|{'tta' if self.tta else 'single'}|{self.post}"

    @property
    def members(self) -> tuple[tuple[str, str], ...]:
        return ENSEMBLE_MEMBERS if self.backbone == "ensemble" else ((self.backbone, self.pooling),)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchConfig":
        return cls(**{k: d[k] for k in ("backbone", "pooling", "preprocess", "tta", "post")})


class Projector:
    """Post-processing fitted on **gallery** vectors only, then applied to gallery and queries alike.

    * ``pcaw<d>`` — PCA-whitening (Jégou & Chum, 2012): rotate onto principal axes and rescale each to
      unit variance, damping the few dominant generic directions (background, lighting). Unsupervised.
    * ``lda<s>`` — PCA to ≤ 512 dims, then shrinkage **Linear Discriminant Analysis** on the gallery's
      pose labels: the linear map that maximises between-pose vs. within-pose variance — a learned
      metric that emphasises *pose* over appearance. The CNNs stay frozen; only this map is learned.
    """

    def __init__(self, post: str):
        self.post, self.pca, self.lda = post, None, None

    def fit(self, x: np.ndarray, labels: np.ndarray | None = None) -> "Projector":
        from sklearn.decomposition import PCA

        if self.post.startswith("pcaw"):
            self.pca = PCA(min(int(self.post[4:]), x.shape[1], len(x) - 1), whiten=True, random_state=0).fit(x)
        elif self.post.startswith("lda"):
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            if labels is None:
                raise ValueError("LDA needs gallery labels")
            self.pca = PCA(min(512, x.shape[1], len(x) - 1), random_state=0).fit(x)
            self.lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=float(self.post[3:]))
            self.lda.fit(self.pca.transform(x), labels)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.post == "l2":
            return l2n(x).astype(np.float32)
        z = self.pca.transform(x)
        if self.lda is not None:
            z = self.lda.transform(z)
        return l2n(z).astype(np.float32)

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Projector":
        import joblib

        return joblib.load(path)


def combine_views(orig: np.ndarray, flip: np.ndarray | None, tta: bool) -> np.ndarray:
    """L2-normalise; with TTA, average the normalised original and mirrored vectors."""
    if not tta or flip is None:
        return l2n(orig)
    return l2n(l2n(orig) + l2n(flip))


def load_features(config: dict, cfg: SearchConfig) -> np.ndarray:
    """View-combined (and, for the ensemble, concatenated) features for every manifest row."""
    parts = []
    for backbone, pooling in cfg.members:
        orig = np.load(feature_path(config, backbone, cfg.preprocess, pooling, "orig"))
        flip = np.load(feature_path(config, backbone, cfg.preprocess, pooling, "flip")) if cfg.tta else None
        parts.append(combine_views(orig, flip, cfg.tta))
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def build_vectors(
    config: dict, cfg: SearchConfig, gallery_mask: np.ndarray, labels: np.ndarray | None = None
) -> tuple[np.ndarray, Projector]:
    """Final unit-length vectors for all rows; the projector is fitted on gallery rows only."""
    x = load_features(config, cfg)
    proj = Projector(cfg.post).fit(x[gallery_mask], None if labels is None else labels[gallery_mask])
    return proj.transform(x), proj


def save_selection(path: Path, cfg: SearchConfig, val_metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": cfg.to_dict(), "val_metrics": val_metrics}, indent=2))


def load_selection(path: Path) -> SearchConfig:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found — run `python run_pipeline.py --only tuning` first.")
    return SearchConfig.from_dict(json.loads(path.read_text())["config"])
