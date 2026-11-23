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


def test_second_local_client_on_same_folder_raises_clear_error(tmp_path):
    import pytest

    from src.indexing.vector_store import IndexLockedError

    first = VectorStore(tmp_path / "qdrant", "t")
    try:
        with pytest.raises(IndexLockedError, match="already open in another process"):
            VectorStore(tmp_path / "qdrant", "t")
    finally:
        first.close()


def test_unreferenced_client_in_same_process_does_not_block_reopening(tmp_path):
    stale = VectorStore(tmp_path / "qdrant", "t")
    stale.recreate({"image": 4})
    stale.cycle = stale  # a reference cycle: only the garbage collector can free it
    del stale  # dropped without close(), as when Streamlit discards a cached service
    fresh = VectorStore(tmp_path / "qdrant", "t")
    assert fresh.client.collection_exists("t")
    fresh.close()


def test_with_collection_shares_the_client():
    store = VectorStore(path=None, collection="a")
    other = store.with_collection("b")
    assert other.client is store.client and other.collection == "b"
