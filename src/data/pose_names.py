"""Sanskrit ↔ English names for the 107 poses (folder names in the dataset are lower-case Sanskrit)."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).with_name("pose_names.csv")


@lru_cache(maxsize=1)
def pose_names() -> pd.DataFrame:
    """Table indexed by dataset folder name with ``sanskrit`` and ``english`` columns."""
    return pd.read_csv(_CSV).set_index("folder")


def display_name(pose: str) -> str:
    """``'bakasana'`` → ``'Bakasana (Crow Pose)'``."""
    names = pose_names()
    if pose not in names.index:
        return pose.title()
    row = names.loc[pose]
    return f"{row.sanskrit} ({row.english})"


def english_name(pose: str) -> str:
    names = pose_names()
    return names.loc[pose, "english"] if pose in names.index else pose.title()
