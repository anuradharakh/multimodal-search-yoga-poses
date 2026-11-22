"""Load ``config.yaml`` and resolve its relative paths against the repository root."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Read the YAML config. Relative ``config_path`` values are looked up from the repo root."""
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_path(config: dict, key: str) -> Path:
    """Absolute path for ``config["paths"][key]`` (created on demand)."""
    path = Path(config["paths"][key])
    path = path if path.is_absolute() else REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path
