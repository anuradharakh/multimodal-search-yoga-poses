"""Clean the raw dataset: drop ambiguous duplicates, unreadable and tiny files; normalise to RGB JPEG.

The raw data mixes PNGs with transparency, JPEGs and GIFs, and contains byte-identical images filed
under *two different poses*. Such images have an ambiguous label, so every copy is dropped; plain
within-pose duplicates keep one copy. Each kept image is converted to RGB (transparency flattened
onto white) and downscaled so its longest side is at most ``max_side`` pixels.
"""

import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jpe", ".gif", ".bmp", ".webp"}


@dataclass
class CleaningReport:
    files_seen: int = 0
    unreadable: int = 0
    too_small: int = 0
    duplicate_within_pose: int = 0
    ambiguous_label: int = 0
    kept: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL mode to RGB, compositing transparent pixels onto white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        return background
    return img.convert("RGB")


def list_raw_files(raw_dir: Path) -> list[tuple[str, str, Path]]:
    """``(original_split, pose, path)`` for every image, in a deterministic order."""
    files = []
    for split_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for pose_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            files += [
                (split_dir.name, pose_dir.name, f) for f in sorted(pose_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTS
            ]
    return files


def find_ambiguous_hashes(files: list[tuple[str, str, Path]]) -> tuple[dict[Path, str], set[str]]:
    """MD5 of every file, plus the set of hashes that occur under more than one pose."""
    digests = {f: hashlib.md5(f.read_bytes()).hexdigest() for _, _, f in tqdm(files, desc="Hashing")}
    poses_per_hash: dict[str, set[str]] = defaultdict(set)
    for _, pose, f in files:
        poses_per_hash[digests[f]].add(pose)
    return digests, {h for h, poses in poses_per_hash.items() if len(poses) > 1}


def clean_images(raw_dir: Path, image_dir: Path, min_side: int, max_side: int) -> tuple[pd.DataFrame, CleaningReport]:
    """Write cleaned JPEGs to ``image_dir/<pose>/<md5[:16]>.jpg``; return one row per kept image."""
    files = list_raw_files(raw_dir)
    report = CleaningReport(files_seen=len(files))
    digests, ambiguous = find_ambiguous_hashes(files)
    seen: set[str] = set()
    rows = []
    for orig_split, pose, f in tqdm(files, desc="Cleaning"):
        digest = digests[f]
        if digest in ambiguous:
            report.ambiguous_label += 1
            continue
        if digest in seen:
            report.duplicate_within_pose += 1
            continue
        try:
            with Image.open(f) as im:
                im.load()
                img = to_rgb(im)
        except Exception:
            report.unreadable += 1
            continue
        w, h = img.size
        if min(w, h) < min_side:
            report.too_small += 1
            continue
        seen.add(digest)
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        out = image_dir / pose / f"{digest[:16]}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, "JPEG", quality=95)
        rows.append(
            {
                "path": f"{pose}/{out.name}",
                "pose": pose,
                "orig_split": orig_split,
                "orig_width": w,
                "orig_height": h,
                "md5": digest,
            }
        )
    report.kept = len(rows)
    return pd.DataFrame(rows), report
