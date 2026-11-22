"""Reciprocal Rank Fusion (Cormack et al., 2009) — merge ranked lists from several retrievers.

``score(d) = Σ_lists 1 / (rrf_k + rank_d)``. In text RAG it fuses BM25 with dense retrieval; here it
is evaluated as a way to fuse the tuned image vector with raw CLIP vectors (see the learning guide
for why it does *not* help on this task).
"""

from collections.abc import Sequence

import numpy as np


def rrf_fuse(ranked_lists: Sequence[Sequence[int]], rrf_k: int = 60, top_k: int | None = None) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores, key=scores.get, reverse=True)
    return fused[:top_k] if top_k else fused


def rrf_fuse_matrix(rankings: Sequence[np.ndarray], n_items: int, rrf_k: int = 60) -> np.ndarray:
    """Vectorised RRF for many queries: each ranking is ``(n_queries, depth)`` of item indices."""
    n_q = rankings[0].shape[0]
    scores = np.zeros((n_q, n_items))
    for r in rankings:
        depth = r.shape[1]
        np.add.at(scores, (np.arange(n_q)[:, None], r), 1.0 / (rrf_k + np.arange(1, depth + 1)))
    return np.argsort(-scores, axis=1)
