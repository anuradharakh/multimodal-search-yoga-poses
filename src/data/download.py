"""Locate the raw Kaggle *Yoga Poses 107* dataset, downloading it with kagglehub if needed."""

from pathlib import Path

from src.utils.logger import log_info


def get_raw_dataset_dir(config: dict) -> Path:
    """Folder containing ``train/ val/ test/<pose>/<image>``.

    Uses ``source.raw_dir`` when set; otherwise downloads (≈1 GB, public, no credentials) into the
    kagglehub cache, which is reused on later runs.
    """
    if config["source"].get("raw_dir"):
        return Path(config["source"]["raw_dir"])
    import kagglehub

    log_info(f"Downloading {config['source']['kaggle_dataset']} with kagglehub (cached after first run)")
    root = Path(kagglehub.dataset_download(config["source"]["kaggle_dataset"]))
    return root / "107 yoga poses"
