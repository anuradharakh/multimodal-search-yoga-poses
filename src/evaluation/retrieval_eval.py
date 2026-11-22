"""Second-stage evaluation: tune the reranker on val, compare alternatives, score the full pipeline on test.

* **Reranker tuning (val).** ``fetch_k × k × λ`` grid on exact-search candidates (identical to what
  Qdrant returns — local Qdrant is exact too, and the final numbers below go through Qdrant).
* **Alternatives (val).** Hybrid RRF of the tuned vector with raw CLIP vectors, and α-query expansion.
* **Final (test, through Qdrant).** Dense retrieval alone vs. dense + reranker — the reported result.
"""

import itertools
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.manifest import load_manifest, split_masks
from src.embedding.postprocess import SearchConfig, build_vectors, load_selection
from src.evaluation.metrics import retrieval_metrics
from src.indexing.index_builder import open_store, selection_path
from src.indexing.vector_store import labels_of
from src.retrieval.reranker import NeighborhoodReranker, RerankerParams, query_expansion
from src.retrieval.rrf import rrf_fuse_matrix


def _vectors(config: dict):
    df = load_manifest(config)
    labels, g, v, t = split_masks(df)
    vecs, _ = build_vectors(config, load_selection(selection_path(config)), g, labels)
    return df, labels, g, v, t, vecs


def dense_candidates(vecs: np.ndarray, gallery_idx: np.ndarray, query_idx: np.ndarray, fetch: int):
    """Exact cosine candidates: global row ids and scores, best first."""
    sims = vecs[query_idx] @ vecs[gallery_idx].T
    order = np.argsort(-sims, axis=1)[:, :fetch]
    return gallery_idx[order], np.take_along_axis(sims, order, axis=1)


def rerank_all(reranker: NeighborhoodReranker, ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.stack([reranker.rerank(i, s)[0] for i, s in zip(ids, scores, strict=True)])


def _metrics(ranked_ids, labels, query_idx, gallery_idx, ks):
    return retrieval_metrics(labels[ranked_ids[:, : max(ks)]], labels[query_idx], labels[gallery_idx], ks)


def tune_reranker(config: dict, table: np.ndarray) -> pd.DataFrame:
    df, labels, g, v, _, vecs = _vectors(config)
    gi, qi, ks = np.flatnonzero(g), np.flatnonzero(v), config["evaluation"]["ks"]
    space = config["reranker"]["search_space"]
    ids, scores = dense_candidates(vecs, gi, qi, max(space["fetch_k"]))
    rows = [{"fetch_k": 0, "k": 0, "weight": 0.0, **_metrics(ids, labels, qi, gi, ks)}]  # no reranker
    for fetch, k, w in tqdm(
        list(itertools.product(space["fetch_k"], space["k"], space["weight"])), desc="Reranker grid (val)"
    ):
        rr = NeighborhoodReranker(table, RerankerParams(fetch, k, w))
        rows.append(
            {
                "fetch_k": fetch,
                "k": k,
                "weight": w,
                **_metrics(rerank_all(rr, ids[:, :fetch], scores[:, :fetch]), labels, qi, gi, ks),
            }
        )
    return pd.DataFrame(rows).sort_values(config["tuning"]["selection_metric"], ascending=False).reset_index(drop=True)


def select_reranker(grid: pd.DataFrame, selection: dict) -> pd.Series:
    """Best ``metric`` among settings whose guardrail metric stays within the allowed drop."""
    baseline = grid[grid.fetch_k == 0].iloc[0]
    floor = baseline[selection["guardrail_metric"]] - selection["max_guardrail_drop"]
    ok = grid[(grid.fetch_k > 0) & (grid[selection["guardrail_metric"]] >= floor)]
    if ok.empty:
        return baseline
    return ok.sort_values(selection["metric"], ascending=False).iloc[0]


def compare_alternatives(config: dict, table: np.ndarray, best: RerankerParams) -> pd.DataFrame:
    """Dense baseline vs. hybrid RRF vs. query expansion vs. the neighbourhood reranker (val)."""
    df, labels, g, v, _, vecs = _vectors(config)
    gi, qi, ks = np.flatnonzero(g), np.flatnonzero(v), config["evaluation"]["ks"]
    ids, scores = dense_candidates(vecs, gi, qi, 100)
    rows = [{"method": "dense retrieval only (baseline)", **_metrics(ids, labels, qi, gi, ks)}]

    clip, _ = build_vectors(config, SearchConfig("clip_rn50", "attn", "pad", True, "l2"), g)
    clip_ids, _ = dense_candidates(clip, gi, qi, 100)
    pos = {gid: j for j, gid in enumerate(gi)}
    to_local = np.vectorize(pos.get)
    fused = gi[rrf_fuse_matrix([to_local(ids), to_local(clip_ids)], len(gi), config["reranker"]["rrf_k"])]
    rows.append({"method": "hybrid: tuned + raw CLIP, fused with RRF", **_metrics(fused, labels, qi, gi, ks)})

    for k in (3, 10):
        q2 = np.stack([query_expansion(vecs[q], vecs[ids[i, :k]], scores[i, :k], k) for i, q in enumerate(qi)])
        qe_ids, _ = dense_candidates(np.vstack([vecs, q2]), gi, np.arange(len(vecs), len(vecs) + len(qi)), 100)
        rows.append({"method": f"α-query expansion (k={k})", **_metrics(qe_ids, labels, qi, gi, ks)})

    rr = NeighborhoodReranker(table, best)
    reranked = rerank_all(rr, ids[:, : best.fetch_k], scores[:, : best.fetch_k])
    label = f"neighbourhood reranker (fetch_k={best.fetch_k}, k={best.k}, λ={best.weight})"
    rows.append({"method": label, **_metrics(reranked, labels, qi, gi, ks)})
    return pd.DataFrame(rows)


def evaluate_through_qdrant(config: dict, table: np.ndarray, params: RerankerParams, split: str) -> dict:
    """Query the real index (gallery filter) with ``split`` images; dense-only and reranked metrics."""
    df, labels, g, _, _, vecs = _vectors(config)
    qi, gi, ks = np.flatnonzero((df["split"] == split).to_numpy()), np.flatnonzero(g), config["evaluation"]["ks"]
    store = open_store(config)
    t0 = time.perf_counter()
    hits = store.search_batch(vecs[qi], using="image", k=params.fetch_k, split="gallery")
    db_ms = (time.perf_counter() - t0) / len(qi) * 1000
    store.close()
    ids = np.array([[h.id for h in row] for row in hits])
    scores = np.array([[h.score for h in row] for row in hits])
    dense = retrieval_metrics(labels_of(hits)[:, : max(ks)], labels[qi], labels[gi], ks)
    rr = NeighborhoodReranker(table, params)
    t1 = time.perf_counter()
    reranked_ids = rerank_all(rr, ids, scores)
    rerank_ms = (time.perf_counter() - t1) / len(qi) * 1000
    reranked = _metrics(reranked_ids, labels, qi, gi, ks)
    return {
        "dense": dense,
        "dense+rerank": reranked,
        "ms_per_query": {"vector_db": db_ms, "rerank": rerank_ms},
        "_ids": reranked_ids,
        "_qi": qi,
    }


def per_pose_report(labels: np.ndarray, qi: np.ndarray, ranked_ids: np.ndarray, k: int = 5) -> pd.DataFrame:
    rows = []
    retrieved = labels[ranked_ids[:, :k]]
    for pose in np.unique(labels[qi]):
        sel = labels[qi] == pose
        r = retrieved[sel]
        confused = pd.Series(r[r != pose]).value_counts()
        rows.append(
            {
                "pose": pose,
                "n_queries": int(sel.sum()),
                f"P@{k}": float((r == pose).mean()),
                "most_confused_with": confused.index[0] if len(confused) else "",
                "confusion_share": float(confused.iloc[0] / r.size) if len(confused) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(f"P@{k}").reset_index(drop=True)
