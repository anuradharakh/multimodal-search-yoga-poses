"""Retrieval metrics. A retrieved gallery image is *relevant* iff it shows the same pose as the query.

* **P@k** — share of the first *k* results with the query's pose;
* **mAP@k** — mean average precision over the first *k* results (rewards correct results ranked first),
  normalised by ``min(k, #relevant)`` so a perfect ranking scores 1;
* **Hit@5** — at least one correct pose among the first five (the "did the user find it" rate).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_KS = (1, 5, 10)


def topk_indices(queries: np.ndarray, gallery: np.ndarray, k: int, batch: int = 1024) -> np.ndarray:
    """Exact cosine top-k (inputs must be L2-normalised). Returns ``(n_queries, k)`` gallery indices."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), batch):
        sims = queries[s : s + batch] @ gallery.T
        part = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        order = np.take_along_axis(sims, part, axis=1).argsort(axis=1)[:, ::-1]
        out[s : s + batch] = np.take_along_axis(part, order, axis=1)
    return out


def relevance_matrix(retrieved_labels: np.ndarray, query_labels: np.ndarray) -> np.ndarray:
    """Boolean ``(n_queries, k)``: is the i-th result the same pose as the query?"""
    return retrieved_labels == np.asarray(query_labels)[:, None]


def precision_at_k(rel: np.ndarray, k: int) -> np.ndarray:
    return rel[:, :k].mean(axis=1)


def average_precision_at_k(rel: np.ndarray, k: int, n_relevant: np.ndarray) -> np.ndarray:
    """AP@k, normalised by ``min(k, #relevant in gallery)`` so a perfect ranking scores 1."""
    r = rel[:, :k].astype(float)
    hits = np.cumsum(r, axis=1)
    prec = hits / np.arange(1, k + 1)
    denom = np.minimum(k, np.maximum(n_relevant, 1))
    return (prec * r).sum(axis=1) / denom


def retrieval_metrics(
    retrieved_labels: np.ndarray, query_labels: Sequence, gallery_labels: Sequence, ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float]:
    """Mean P@k for each ``k`` plus mAP@max(k) and Hit@5 (≥1 correct pose among the first five)."""
    rel = relevance_matrix(np.asarray(retrieved_labels), np.asarray(query_labels))
    counts = pd.Series(gallery_labels).value_counts()
    n_rel = pd.Series(query_labels).map(counts).fillna(0).to_numpy()
    kmax = max(ks)
    out = {f"P@{k}": float(precision_at_k(rel, k).mean()) for k in ks}
    out[f"mAP@{kmax}"] = float(average_precision_at_k(rel, kmax, n_rel).mean())
    out["Hit@5"] = float(rel[:, :5].any(axis=1).mean())
    return out


def evaluate_vectors(
    vectors: np.ndarray, labels: np.ndarray, gallery_mask: np.ndarray, query_mask: np.ndarray, ks=DEFAULT_KS
) -> dict[str, float]:
    """Exact cosine search of ``query_mask`` rows against ``gallery_mask`` rows."""
    g_lab = labels[gallery_mask]
    idx = topk_indices(vectors[query_mask], vectors[gallery_mask], max(ks))
    return retrieval_metrics(g_lab[idx], labels[query_mask], g_lab, ks)
