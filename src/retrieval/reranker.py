"""Second-stage reranking of the dense-retrieval candidates.

Dense retrieval scores every gallery image *independently* against the query. A reranker looks at
the short candidate list again with extra context — the same idea as the cross-encoder in a text
RAG system, adapted to images, where no "query–document pair" model exists.

**Neighbourhood-overlap reranking** (a light-weight form of k-reciprocal re-ranking, Zhong et al.,
CVPR 2017). If a candidate really shows the query's pose, its own nearest neighbours in the gallery
tend to be the *same images* as the query's nearest neighbours. So for each candidate ``c``::

    overlap(c) = |N_k(query) ∩ N_k(c)| / k
    score(c)   = (1 − λ) · cosine(query, c) + λ · overlap(c)

``N_k(query)`` is the top-``k`` of the dense candidate list (all ``fetch_k`` candidates when
``k ≥ fetch_k``); ``N_k(c)`` comes from a gallery neighbour table precomputed at indexing time, so
reranking costs a few set intersections per query (~0.02 ms).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.embedding.postprocess import l2n


@dataclass(frozen=True)
class RerankerParams:
    fetch_k: int = 50  # candidates taken from the vector database
    k: int = 10  # neighbourhood size
    weight: float = 0.3  # λ — share of the score given to neighbourhood overlap


def build_neighbor_table(vectors: np.ndarray, gallery_idx: np.ndarray, k_max: int, batch: int = 1024) -> np.ndarray:
    """For every row of ``vectors``: ids (rows) of its ``k_max`` nearest *gallery* items, itself excluded."""
    gallery = vectors[gallery_idx]
    out = np.empty((len(vectors), k_max), dtype=np.int32)
    for s in range(0, len(vectors), batch):
        sims = vectors[s : s + batch] @ gallery.T
        rows = np.arange(s, min(s + batch, len(vectors)))
        self_mask = gallery_idx[None, :] == rows[:, None]
        sims[self_mask] = -np.inf
        order = np.argsort(-sims, axis=1)[:, :k_max]
        out[s : s + batch] = gallery_idx[order]
    return out


def save_neighbor_table(table: np.ndarray, path: Path) -> None:
    np.save(path, table)


def load_neighbor_table(path: Path) -> np.ndarray:
    return np.load(path)


class NeighborhoodReranker:
    def __init__(self, neighbor_table: np.ndarray, params: RerankerParams):
        if params.k > neighbor_table.shape[1]:
            raise ValueError(f"k={params.k} exceeds the neighbour table width {neighbor_table.shape[1]}")
        self.table, self.params = neighbor_table, params

    def rerank(self, candidate_ids: Sequence[int], dense_scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Reorder one query's candidates; returns ``(ids, scores)`` sorted by the reranked score."""
        ids = np.asarray(candidate_ids)
        k, lam = self.params.k, self.params.weight
        query_nn = set(ids[:k].tolist())
        overlap = np.array([len(query_nn.intersection(self.table[c, :k].tolist())) / k for c in ids])
        scores = (1 - lam) * np.asarray(dense_scores, dtype=float) + lam * overlap
        order = np.argsort(-scores, kind="stable")
        return ids[order], scores[order]


def query_expansion(
    query: np.ndarray, candidate_vecs: np.ndarray, candidate_scores: np.ndarray, k: int, alpha: float = 3.0
) -> np.ndarray:
    """α-weighted query expansion (Radenović et al., 2018) — the alternative reranker compared in evaluation."""
    w = np.clip(candidate_scores[:k], 0, None) ** alpha
    return l2n(query + (w[:, None] * candidate_vecs[:k]).sum(axis=0))
