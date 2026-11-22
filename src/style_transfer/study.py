"""The style-transfer study: which artworks and settings keep the yoga pose readable?

Experiments (content photos from the held-out **test** split):

1. **Style grid** — every content pose × every artwork at the default weights.
2. **Weight sweep** — style/content ratio β/α over ``weight_ratios`` for ``sweep_pairs``.
3. **Style-layer ablation** — Gram matrices from shallow, all, or deep VGG layers.
4. **Initialisation** — start from the content photo vs. from white noise (2× iterations).

Every output is scored with :class:`~src.style_transfer.pose_metrics.StyleMetrics` and the pose
search engine. Runs are cached as JSON next to the images, so an interrupted study resumes.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.data.manifest import image_dir, load_manifest
from src.style_transfer.optimizer import StyleParams, StyleTransfer
from src.style_transfer.pose_metrics import PoseDetector, StyleMetrics
from src.style_transfer.vgg_features import STYLE_LAYER_SETS, load_image
from src.utils.config_loader import get_path
from src.utils.device import get_device


def base_params(config: dict) -> StyleParams:
    s = config["style_transfer"]
    return StyleParams(
        content_weight=float(s["content_weight"]),
        style_weight=float(s["style_weight"]),
        tv_weight=float(s["tv_weight"]),
        steps=int(s["steps"]),
        size=int(s["size"]),
    )


def select_content_images(config: dict, detector: PoseDetector) -> pd.DataFrame:
    """Per pose, the test photo whose skeleton the keypoint detector finds most confidently.

    A confident skeleton on the *original* makes the pose-preservation metric meaningful — otherwise
    OKS would measure detector failure rather than stylisation.
    """
    s = config["style_transfer"]
    df = load_manifest(config)
    rows = []
    for pose in s["content_poses"]:
        cand = df[(df.pose == pose) & (df.split == "test") & (df[["orig_width", "orig_height"]].min(axis=1) >= 256)]
        best = None
        for _, r in cand.head(s["candidates_per_pose"]).iterrows():
            det = detector(load_image(r.abs_path, s["size"]))
            if det is None:
                continue
            quality = det.score * float((det.keypoint_scores > 0).mean())
            if best is None or quality > best[0]:
                best = (quality, r)
        if best is not None:
            rows.append({"pose": pose, "path": best[1].path, "quality": best[0]})
    return pd.DataFrame(rows)


def content_table_path(config: dict) -> Path:
    return get_path(config, "output_dir") / "metrics" / "style_content_images.csv"


def load_content_table(config: dict) -> pd.DataFrame:
    """Chosen content photos with an absolute ``abs_path`` column."""
    contents = pd.read_csv(content_table_path(config))
    contents["abs_path"] = contents.path.map(lambda p: str(image_dir(config) / p))
    return contents


class Study:
    def __init__(self, config: dict, device=None):
        self.config = config
        self.device = device or get_device(config["project"]["device"])
        self.base = base_params(config)
        self.out_dir = get_path(config, "style_dir")
        self.st = StyleTransfer(self.device)
        self.detector = PoseDetector(self.device)
        self.metrics = StyleMetrics(self.device, self.detector)
        styles_dir = get_path(config, "styles_dir")
        self.styles = {
            k: Image.open(styles_dir / f"{k}.jpg").convert("RGB") for k in config["style_transfer"]["styles"]
        }
        self._search = None

    @property
    def search(self):
        if self._search is None:
            from src.retrieval.search_service import PoseSearch

            self._search = PoseSearch(self.config, self.device)
        return self._search

    def run_one(self, experiment: str, pose: str, content_path: str, style: str, params: StyleParams, tag: str) -> dict:
        stem = f"{pose.replace(' ', '_')}__{style}__{tag}"
        folder = self.out_dir / experiment
        folder.mkdir(parents=True, exist_ok=True)
        img_path, meta_path = folder / f"{stem}.jpg", folder / f"{stem}.json"
        if img_path.exists() and meta_path.exists():
            return json.loads(meta_path.read_text())

        content = load_image(content_path, params.size)
        style_img = load_image(self.styles[style], params.size)
        res = self.st.run(content, style_img, params)
        res.image.save(img_path, quality=95)
        pred, conf = self.search.predict_pose(res.image)
        row = {
            "experiment": experiment,
            "pose": pose,
            "style": style,
            "tag": tag,
            "image": f"{experiment}/{img_path.name}",
            "seconds": round(res.seconds, 1),
            "evals": len(res.history),
            **self.metrics.score(res.image, content, style_img),
            "pose_pred": pred,
            "pose_pred_conf": conf,
            "pose_retrieved": pred == pose,
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(params).items()},
            "loss_history": res.history[:: max(1, len(res.history) // 60)],
        }
        meta_path.write_text(json.dumps(row))
        return row

    def baseline_rows(self, contents: pd.DataFrame) -> list[dict]:
        """Score the *unstylised* photos too, as the reference point (OKS = 1, SSIM = 1)."""
        rows = []
        for _, c in contents.iterrows():
            img = load_image(c.abs_path, self.base.size)
            det = self.metrics.reference_pose(img)
            pred, conf = self.search.predict_pose(img)
            rows.append(
                {
                    "experiment": "original",
                    "pose": c.pose,
                    "style": "none",
                    "tag": "original",
                    "person_score": det.score if det else 0.0,
                    "pose_oks": 1.0,
                    "ssim": 1.0,
                    "content_dist": 0.0,
                    "pose_pred": pred,
                    "pose_pred_conf": conf,
                    "pose_retrieved": pred == c.pose,
                }
            )
        return rows

    def jobs(self, poses: list[str]) -> list[tuple[str, str, str, StyleParams, str]]:
        s = self.config["style_transfer"]
        jobs = [("grid", pose, style, self.base, "default") for pose in poses for style in s["styles"]]
        for pose, style in s["sweep_pairs"]:
            jobs += [
                ("weights", pose, style, replace(self.base, style_weight=float(r)), f"sw{float(r):.0e}")
                for r in s["weight_ratios"]
            ]
        for pose, style in s["ablation_pairs"]:
            jobs += [
                ("layers", pose, style, replace(self.base, style_layers=layers), name)
                for name, layers in STYLE_LAYER_SETS.items()
            ]
            jobs.append(("init", pose, style, replace(self.base, init="noise", steps=self.base.steps * 2), "noise"))
        return jobs

    def run(self) -> pd.DataFrame:
        contents = select_content_images(self.config, self.detector)
        contents.to_csv(content_table_path(self.config), index=False)
        contents = load_content_table(self.config)
        path_of = dict(zip(contents.pose, contents.abs_path, strict=True))
        jobs = self.jobs(list(contents.pose))
        rows = [self.run_one(e, p, path_of[p], s, prm, t) for e, p, s, prm, t in tqdm(jobs, desc="Style transfer")]
        rows += self.baseline_rows(contents)
        return pd.DataFrame(rows).drop(columns=["loss_history", "params"], errors="ignore")


def summarize_by_style(config: dict, runs: pd.DataFrame) -> pd.DataFrame:
    grid = runs[runs.experiment == "grid"]
    s = (
        grid.groupby("style")
        .agg(
            pose_oks=("pose_oks", "mean"),
            ssim=("ssim", "mean"),
            content_dist=("content_dist", "mean"),
            style_dist_log10=("style_dist_log10", "mean"),
            person_score=("person_score", "mean"),
            pose_retrieved=("pose_retrieved", "mean"),
            seconds=("seconds", "mean"),
        )
        .sort_values("pose_oks", ascending=False)
    )
    s.insert(0, "artwork", s.index.map(config["style_transfer"]["styles"]))
    return s
