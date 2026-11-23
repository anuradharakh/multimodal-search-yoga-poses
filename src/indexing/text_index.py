"""The *text* branch of the search architecture: Text → text-embedding generator → vector DB.

Each of the 107 poses gets one short description ("a photo of a person doing Vrksasana (Tree
Pose), a yoga pose"), embedded with CLIP-RN50's text encoder and stored in its own Qdrant
collection. A text query is embedded with the same encoder and matched against these
descriptions (text ↔ text, which CLIP does far better than text ↔ image for pose names); the
best-matching pose then selects gallery photos through a payload filter.
"""

import pandas as pd

from src.data.pose_names import display_name, pose_names
from src.indexing.vector_store import VectorStore

DESCRIPTION_TEMPLATE = "a photo of a person doing {sanskrit} ({english}), a yoga pose"


def pose_descriptions() -> pd.DataFrame:
    """One row per pose: ``pose``, ``pose_display``, ``sanskrit``, ``english``, ``description``."""
    names = pose_names().reset_index().rename(columns={"folder": "pose"})
    names["pose_display"] = names["pose"].map(display_name)
    names["description"] = [
        DESCRIPTION_TEMPLATE.format(sanskrit=s, english=e) for s, e in zip(names.sanskrit, names.english, strict=True)
    ]
    return names[["pose", "pose_display", "sanskrit", "english", "description"]]


def build_text_index(store: VectorStore, collection: str, clip, device) -> int:
    """Embed every pose description with the CLIP text encoder and (re)load ``collection``."""
    df = pose_descriptions()
    vectors = clip.encode_text(df["description"].tolist(), device).cpu().numpy()
    text_store = store.with_collection(collection)
    text_store.recreate({"text": vectors.shape[1]})
    text_store.upsert({"text": vectors}, df)
    return text_store.count()
