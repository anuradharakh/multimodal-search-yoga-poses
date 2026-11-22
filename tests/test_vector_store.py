import numpy as np
import pandas as pd

from src.embedding.postprocess import l2n
from src.indexing.vector_store import VectorStore, labels_of


def test_qdrant_roundtrip_filters_and_exclusion():
    vecs = l2n(np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32))
    payload = pd.DataFrame(
        {
            "path": [f"p{i}.jpg" for i in range(6)],
            "pose": ["a", "a", "b", "b", "c", "c"],
            "pose_display": list("AABBCC"),
            "split": ["gallery", "test"] * 3,
        }
    )
    store = VectorStore(path=None, collection="t")
    store.recreate({"image": 8})
    store.upsert({"image": vecs}, payload)
    assert store.count() == 6
    top = store.search(vecs[0], using="image", k=1)[0]
    assert top.payload["path"] == "p0.jpg" and abs(top.score - 1) < 1e-5
    assert all(h.payload["path"] != "p0.jpg" for h in store.search(vecs[0], "image", k=3, exclude_path="p0.jpg"))
    batch = store.search_batch(vecs[[1, 3]], using="image", k=3, split="gallery")
    assert all(h.payload["split"] == "gallery" for row in batch for h in row)
    assert labels_of(batch).shape == (2, 3)
