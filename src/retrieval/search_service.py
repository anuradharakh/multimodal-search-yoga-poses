"""Query-time search service: encode → dense retrieval from Qdrant → rerank. Used by the app and studies.

The searchable database is the **gallery** split; val/test images are held-out query photos.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.embedding.postprocess import load_selection
from src.indexing.index_builder import CLIP_CONFIG, load_projector, open_store, selection_path
from src.retrieval.query_encoder import QueryEncoder
from src.retrieval.reranker import NeighborhoodReranker, RerankerParams, load_neighbor_table
from src.utils.config_loader import get_path
from src.utils.device import get_device


def neighbor_table_path(config: dict) -> Path:
    return get_path(config, "index_dir") / "gallery_neighbors.npy"


def reranker_selection_path(config: dict) -> Path:
    return get_path(config, "output_dir") / "metrics" / "selected_reranker.json"


def load_reranker_params(config: dict) -> RerankerParams:
    path = reranker_selection_path(config)
    if path.exists():
        return RerankerParams(**json.loads(path.read_text())["params"])
    return RerankerParams(**{k: config["reranker"]["default"][k] for k in ("fetch_k", "k", "weight")})


@dataclass
class SearchHit:
    id: int
    path: str
    pose: str
    pose_display: str
    score: float  # final score (reranked if the reranker ran)
    dense_score: float  # cosine similarity from the vector database
    dense_rank: int  # 1-based rank before reranking


@dataclass
class SearchResult:
    hits: list[SearchHit] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class PoseSearch:
    def __init__(self, config: dict, device=None):
        self.config = config
        self.device = device or get_device(config["project"]["device"])
        self.cfg = load_selection(selection_path(config))
        self.image_encoder = QueryEncoder(self.cfg, self.device, load_projector(config))
        self.clip_encoder = QueryEncoder(CLIP_CONFIG, self.device)
        self.store = open_store(config)
        self.params = load_reranker_params(config)
        self.reranker = NeighborhoodReranker(load_neighbor_table(neighbor_table_path(config)), self.params)

    def _search(self, vector, using: str, k: int, rerank: bool, timings: dict) -> SearchResult:
        t0 = time.perf_counter()
        fetch = max(k, self.params.fetch_k) if rerank else k
        hits = self.store.search(vector, using=using, k=fetch, split="gallery")
        timings["vector_db"] = (time.perf_counter() - t0) * 1000
        dense = {h.id: (h, h.score, r) for r, h in enumerate(hits, start=1)}
        if rerank:
            t1 = time.perf_counter()
            ids, scores = self.reranker.rerank([h.id for h in hits], [h.score for h in hits])
            timings["rerank"] = (time.perf_counter() - t1) * 1000
        else:
            ids, scores = [h.id for h in hits], [h.score for h in hits]
        out = []
        for i, s in zip(ids[:k], scores[:k], strict=False):
            h, d, r = dense[int(i)]
            p = h.payload
            out.append(SearchHit(int(i), p["path"], p["pose"], p["pose_display"], float(s), float(d), r))
        return SearchResult(out, timings)

    def search_image(self, img: Image.Image, k: int = 12, rerank: bool = True, vector: str = "image") -> SearchResult:
        t0 = time.perf_counter()
        encoder = self.image_encoder if vector == "image" else self.clip_encoder
        vec = encoder.encode_images([img])[0]
        timings = {"embed": (time.perf_counter() - t0) * 1000}
        return self._search(vec, vector, k, rerank and vector == "image", timings)

    def search_text(self, text: str, k: int = 12) -> SearchResult:
        t0 = time.perf_counter()
        vec = self.clip_encoder.encode_texts([text])[0]
        return self._search(vec, "clip", k, False, {"embed": (time.perf_counter() - t0) * 1000})

    def predict_pose(self, img: Image.Image, k: int = 10) -> tuple[str, float]:
        """Similarity-weighted kNN vote over the top-k results."""
        votes: dict[str, float] = {}
        for h in self.search_image(img, k=k).hits:
            votes[h.pose] = votes.get(h.pose, 0.0) + max(h.dense_score, 0.0)
        best = max(votes, key=votes.get)
        return best, votes[best] / max(sum(votes.values()), 1e-9)

    def count(self) -> int:
        return self.store.count()

    def close(self) -> None:
        self.store.close()
