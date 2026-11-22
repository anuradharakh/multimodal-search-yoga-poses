"""Read the cleaned-dataset manifest — the single source of truth for image paths, labels and splits."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config_loader import get_path


def image_dir(config: dict) -> Path:
    return get_path(config, "processed_dir") / "images"


def manifest_path(config: dict) -> Path:
    return get_path(config, "processed_dir") / "manifest.csv"


def load_manifest(config: dict) -> pd.DataFrame:
    """Manifest with an ``abs_path`` column. Row order is the row order of every feature matrix."""
    path = manifest_path(config)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found — run `python run_pipeline.py --only data_prep` first.")
    df = pd.read_csv(path)
    root = image_dir(config)
    df["abs_path"] = df["path"].map(lambda p: str(root / p))
    return df


def split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``labels, gallery_mask, val_mask, test_mask`` as NumPy arrays."""
    return (
        df["pose"].to_numpy(),
        (df["split"] == "gallery").to_numpy(),
        (df["split"] == "val").to_numpy(),
        (df["split"] == "test").to_numpy(),
    )
