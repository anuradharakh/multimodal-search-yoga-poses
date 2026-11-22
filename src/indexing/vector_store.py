"""Thin wrapper around an embedded (serverless, on-disk) Qdrant vector database.

One collection holds one point per image with two **named vectors**:

* ``image`` — the tuned image-retrieval embedding chosen by the grid search;
* ``clip``  — the CLIP-RN50 joint image–text embedding, which also powers **text → image** search.

Payload: ``path`` (relative to the cleaned-image folder), ``pose``, ``pose_display`` and ``split``,
so results can be filtered, e.g. to the gallery split during evaluation or to a single pose.
Qdrant's local mode needs no server and stores data under ``data/indexes/qdrant``; switching to a
Qdrant server or Qdrant Cloud is a one-line change (``QdrantClient(url=...)``).
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


@dataclass
class Hit:
    id: int
    score: float
    payload: dict


class VectorStore:
    def __init__(self, path: Path | str | None, collection: str):
        # path=None → in-memory instance (used by the unit tests)
        self.client = QdrantClient(path=str(path)) if path else QdrantClient(location=":memory:")
        self.collection = collection

    def recreate(self, dims: Mapping[str, int]) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={name: qm.VectorParams(size=d, distance=qm.Distance.COSINE) for name, d in dims.items()},
        )
        with warnings.catch_warnings():  # local mode ignores payload indexes; they matter on a Qdrant server
            warnings.simplefilter("ignore", UserWarning)
            for field in ("split", "pose"):
                self.client.create_payload_index(self.collection, field, qm.PayloadSchemaType.KEYWORD)

    def upsert(self, vectors: Mapping[str, np.ndarray], payloads: pd.DataFrame, batch: int = 512) -> None:
        records = payloads.to_dict("records")
        for s in range(0, len(records), batch):
            points = [
                qm.PointStruct(
                    id=s + i,
                    vector={name: vec[s + i].tolist() for name, vec in vectors.items()},
                    payload=records[s + i],
                )
                for i in range(min(batch, len(records) - s))
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    @staticmethod
    def _filter(split: str | None = None, pose: str | None = None, exclude_path: str | None = None) -> qm.Filter | None:
        must = []
        if split:
            must.append(qm.FieldCondition(key="split", match=qm.MatchValue(value=split)))
        if pose:
            must.append(qm.FieldCondition(key="pose", match=qm.MatchValue(value=pose)))
        must_not = [qm.FieldCondition(key="path", match=qm.MatchValue(value=exclude_path))] if exclude_path else []
        return qm.Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None

    def search(
        self,
        vector: np.ndarray,
        using: str,
        k: int = 10,
        split: str | None = None,
        pose: str | None = None,
        exclude_path: str | None = None,
    ) -> list[Hit]:
        res = self.client.query_points(
            self.collection,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            using=using,
            limit=k,
            query_filter=self._filter(split, pose, exclude_path),
            with_payload=True,
        )
        return [Hit(p.id, p.score, p.payload) for p in res.points]

    def search_batch(self, vectors: np.ndarray, using: str, k: int = 10, split: str | None = None) -> list[list[Hit]]:
        flt = self._filter(split)
        reqs = [
            qm.QueryRequest(query=v.astype(np.float32).tolist(), using=using, limit=k, filter=flt, with_payload=True)
            for v in vectors
        ]
        out = []
        for s in range(0, len(reqs), 256):
            for r in self.client.query_batch_points(self.collection, requests=reqs[s : s + 256]):
                out.append([Hit(p.id, p.score, p.payload) for p in r.points])
        return out

    def close(self) -> None:
        self.client.close()


def labels_of(hits: Sequence[Sequence[Hit]]) -> np.ndarray:
    return np.array([[h.payload["pose"] for h in row] for row in hits])
