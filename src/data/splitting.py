"""Stratified gallery / validation / test split.

* **gallery** — the images stored in the vector database (what a user searches over);
* **val** — query images used to *tune* the pipeline;
* **test** — query images used once, at the end, to report results.

The provided train/val/test folders are merged and re-split *after* de-duplication, so no identical
image can sit in both the gallery and a query split.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, fractions: dict, seed: int) -> pd.Series:
    """Label each row ``gallery`` / ``val`` / ``test`` with per-pose proportions preserved."""
    idx = df.index.to_numpy()
    gallery, rest = train_test_split(idx, train_size=fractions["gallery"], stratify=df["pose"], random_state=seed)
    val_share = fractions["val"] / (fractions["val"] + fractions["test"])
    val, test = train_test_split(rest, train_size=val_share, stratify=df.loc[rest, "pose"], random_state=seed)
    split = pd.Series("gallery", index=df.index)
    split.loc[val] = "val"
    split.loc[test] = "test"
    return split
