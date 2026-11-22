import numpy as np
import pandas as pd
from PIL import Image

from src.data.cleaning import find_ambiguous_hashes, to_rgb
from src.data.pose_names import display_name, pose_names
from src.data.splitting import stratified_split


def test_pose_names_cover_107_poses():
    assert len(pose_names()) == 107
    assert display_name("bakasana") == "Bakasana (Crow Pose)"


def test_to_rgb_flattens_transparency_onto_white():
    out = np.asarray(to_rgb(Image.new("RGBA", (4, 4), (255, 0, 0, 0))))
    assert out.shape == (4, 4, 3) and (out == 255).all()


def test_identical_bytes_under_two_poses_are_ambiguous(tmp_path):
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    Image.new("RGB", (8, 8), "red").save(a)
    b.write_bytes(a.read_bytes())
    Image.new("RGB", (8, 8), "blue").save(c)
    digests, ambiguous = find_ambiguous_hashes([("train", "x", a), ("test", "y", b), ("train", "x", c)])
    assert digests[a] in ambiguous and digests[c] not in ambiguous


def test_stratified_split_proportions():
    df = pd.DataFrame({"pose": np.repeat(["a", "b", "c"], 40)})
    split = stratified_split(df, {"gallery": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    counts = pd.crosstab(df["pose"], split)
    assert (counts["gallery"] == 28).all() and (counts["val"] + counts["test"] == 12).all()
