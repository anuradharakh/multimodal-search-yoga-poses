"""Pose-preservation metrics — quantitative checks for *"was the style transferred while the yoga pose was preserved?"*

Visual inspection alone is subjective, so every stylised image is scored on:

=====================  ==========================================================================
``content_dist``       relative VGG ``conv4_2`` feature distance to the content photo — the quantity
                       the content loss minimises (lower = closer to the photo)
``style_dist``         Gram-matrix distance to the artwork over ``conv1_1…conv5_1`` (log10; lower =
                       more of the artwork's texture/colour statistics)
``ssim``               structural similarity of the grayscale images (pixel-level structure kept)
``pose_oks``           **Object Keypoint Similarity** between the 17 COCO body keypoints that a
                       Keypoint R-CNN detects on the photo and on the stylised image (1 = identical
                       skeleton) — a direct measure of whether the *pose* survived
``person_score``       detector confidence that a person is still visible in the stylised image
``pose_retrieved``     whether the pose-search index (part 1) still returns the correct pose as the
                       kNN vote for the stylised image
=====================  ==========================================================================
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

from src.style_transfer.losses import gram_matrix
from src.style_transfer.vgg_features import CONTENT_LAYER, STYLE_LAYER_SETS, VGGFeatures, pil_to_tensor

# COCO per-keypoint falloff constants (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
COCO_SIGMAS = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)


@dataclass
class PoseDetection:
    keypoints: np.ndarray  # (17, 2) pixel coordinates
    keypoint_scores: np.ndarray  # (17,)
    box: np.ndarray  # (4,) x1, y1, x2, y2
    score: float


class PoseDetector:
    """torchvision Keypoint R-CNN (ResNet-50-FPN, COCO) — returns the most confident person."""

    def __init__(self, device: torch.device):
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

        # Detection models are run on CPU: some MPS kernels used by RoIAlign/NMS are slow or missing.
        self.device = torch.device("cpu") if device.type == "mps" else device
        self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, img: Image.Image) -> PoseDetection | None:
        x = pil_to_tensor(img, self.device)[0]
        out = self.model([x])[0]
        if len(out["scores"]) == 0:
            return None
        i = int(out["scores"].argmax())
        return PoseDetection(
            keypoints=out["keypoints"][i, :, :2].cpu().numpy(),
            keypoint_scores=out["keypoints_scores"][i].cpu().numpy(),
            box=out["boxes"][i].cpu().numpy(),
            score=float(out["scores"][i]),
        )


def object_keypoint_similarity(ref: PoseDetection, other: PoseDetection | None, min_kp_score: float = 0.0) -> float:
    """COCO OKS of ``other`` against ``ref`` (treated as ground truth); 0 if no person was detected."""
    if other is None:
        return 0.0
    visible = ref.keypoint_scores > min_kp_score
    if not visible.any():
        visible[:] = True
    x1, y1, x2, y2 = ref.box
    area = max((x2 - x1) * (y2 - y1), 1.0)
    d2 = ((ref.keypoints - other.keypoints) ** 2).sum(axis=1)
    e = d2 / (2 * area * (2 * COCO_SIGMAS) ** 2)
    return float(np.exp(-e)[visible].mean())


def gray_ssim(a: Image.Image, b: Image.Image) -> float:
    ga = np.asarray(a.convert("L"), dtype=np.float32) / 255.0
    gb = np.asarray(b.convert("L").resize(a.size), dtype=np.float32) / 255.0
    return float(structural_similarity(ga, gb, data_range=1.0))


class StyleMetrics:
    def __init__(self, device: torch.device, pose_detector: "PoseDetector | None" = None):
        self.device = device
        self.vgg = VGGFeatures((*STYLE_LAYER_SETS["all"], CONTENT_LAYER)).to(device)
        self.detector = pose_detector or PoseDetector(device)
        self._ref_cache: dict[int, PoseDetection | None] = {}

    @torch.no_grad()
    def feature_distances(self, out: Image.Image, content: Image.Image, style: Image.Image) -> tuple[float, float]:
        fo = self.vgg(pil_to_tensor(out, self.device))
        fc = self.vgg(pil_to_tensor(content.resize(out.size), self.device))
        fs = self.vgg(pil_to_tensor(style, self.device))
        content_dist = float((fo[CONTENT_LAYER] - fc[CONTENT_LAYER]).norm() / fc[CONTENT_LAYER].norm())
        style_dist = float(
            np.mean([F.mse_loss(gram_matrix(fo[n]), gram_matrix(fs[n])).item() for n in STYLE_LAYER_SETS["all"]])
        )
        return content_dist, float(np.log10(style_dist + 1e-12))

    def reference_pose(self, content: Image.Image) -> PoseDetection | None:
        key = hash(content.tobytes())
        if key not in self._ref_cache:
            self._ref_cache[key] = self.detector(content)
        return self._ref_cache[key]

    def score(self, out: Image.Image, content: Image.Image, style: Image.Image) -> dict[str, float]:
        content = content.resize(out.size)
        content_dist, style_dist = self.feature_distances(out, content, style)
        ref = self.reference_pose(content)
        det = self.detector(out)
        return {
            "content_dist": content_dist,
            "style_dist_log10": style_dist,
            "ssim": gray_ssim(content, out),
            "pose_oks": object_keypoint_similarity(ref, det) if ref else float("nan"),
            "person_score": det.score if det else 0.0,
        }
