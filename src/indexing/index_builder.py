"""Build the Qdrant collection from the selected configuration.

Every cleaned image becomes one point (id = manifest row) with two named vectors:

* ``image`` — the tuned embedding (selected in the tuning stage), used for image → image search;
* ``clip``  — CLIP-RN50's joint image–text embedding, used for text → image search.

The projection fitted on the gallery is saved next to the index so query images are embedded
exactly like the database images.
"""

import json
import time
from pathlib import Path

from src.data.manifest import load_manifest, split_masks
from src.embedding.postprocess import Projector, SearchConfig, build_vectors, load_selection
from src.indexing.vector_store import VectorStore
from src.utils.config_loader import get_path

CLIP_CONFIG = SearchConfig("clip_rn50", "attn", "pad", False, "l2")
PAYLOAD_COLUMNS = ["path", "pose", "pose_display", "split"]


def selection_path(config: dict) -> Path:
    return get_path(config, "output_dir") / "metrics" / "selected_config.json"


def projector_path(config: dict) -> Path:
    return get_path(config, "index_dir") / "projector.joblib"


def open_store(config: dict) -> VectorStore:
    return VectorStore(get_path(config, "index_dir") / "qdrant", config["indexing"]["collection"])


def build_index(config: dict) -> dict:
    cfg = load_selection(selection_path(config))
    df = load_manifest(config)
    labels, gallery, _, _ = split_masks(df)
    image_vecs, projector = build_vectors(config, cfg, gallery, labels)
    projector.save(projector_path(config))
    clip_vecs, _ = build_vectors(config, CLIP_CONFIG, gallery)

    t0 = time.perf_counter()
    store = open_store(config)
    store.recreate({"image": image_vecs.shape[1], "clip": clip_vecs.shape[1]})
    store.upsert({"image": image_vecs, "clip": clip_vecs}, df[PAYLOAD_COLUMNS])
    meta = {
        "config": cfg.key,
        "points": store.count(),
        "dims": {"image": int(image_vecs.shape[1]), "clip": int(clip_vecs.shape[1])},
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    store.close()
    (get_path(config, "index_dir") / "index_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_projector(config: dict) -> Projector:
    return Projector.load(projector_path(config))
