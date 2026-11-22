import numpy as np
import torch

from src.style_transfer.losses import gram_matrix, total_variation
from src.style_transfer.pose_metrics import PoseDetection, object_keypoint_similarity


def test_gram_matrix_shape_symmetry_and_normalisation():
    f = torch.randn(1, 4, 5, 6)
    g = gram_matrix(f)
    assert g.shape == (1, 4, 4)
    torch.testing.assert_close(g, g.transpose(1, 2))
    torch.testing.assert_close(g[0, 0, 0], (f[0, 0] ** 2).sum() / (4 * 5 * 6))


def test_total_variation_zero_for_constant_image():
    assert total_variation(torch.ones(1, 3, 8, 8)).item() == 0.0


def _det(kps):
    return PoseDetection(np.asarray(kps, float), np.ones(17), np.array([0, 0, 100, 200.0]), 0.99)


def test_oks_is_one_for_identical_pose_and_falls_with_distance():
    kps = np.random.default_rng(0).uniform(0, 100, size=(17, 2))
    ref = _det(kps)
    near, far = object_keypoint_similarity(ref, _det(kps + 2)), object_keypoint_similarity(ref, _det(kps + 20))
    assert object_keypoint_similarity(ref, _det(kps)) == 1.0 and 1.0 > near > far > 0.0
    assert object_keypoint_similarity(ref, None) == 0.0
