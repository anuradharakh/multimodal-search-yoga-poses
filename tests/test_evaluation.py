import numpy as np
import pytest

from src.embedding.postprocess import Projector, SearchConfig, combine_views, l2n
from src.evaluation.metrics import average_precision_at_k, retrieval_metrics, topk_indices


def test_topk_returns_most_similar_first():
    q = np.array([[0.1, 0.9, 0.3, 0.0]], dtype=np.float32)
    assert topk_indices(q, np.eye(4, dtype=np.float32), 3).tolist() == [[1, 2, 0]]


def test_average_precision_hand_computed():
    ap = average_precision_at_k(np.array([[1, 0, 1, 0]], dtype=bool), k=4, n_relevant=np.array([2]))
    assert ap[0] == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_retrieval_metrics_perfect_and_wrong():
    gallery = ["a", "a", *["b"] * 9]
    retrieved = np.array([["a"] * 2 + ["b"] * 8, ["a"] * 10])
    m = retrieval_metrics(retrieved, ["a", "b"], gallery, ks=(1, 5, 10))
    assert m["P@1"] == pytest.approx(0.5) and m["mAP@10"] == pytest.approx(0.5)


def test_flip_tta_is_symmetric_and_unit_norm():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 16)), rng.normal(size=(5, 16))
    np.testing.assert_allclose(combine_views(a, b, True), combine_views(b, a, True), rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(combine_views(a, b, True), axis=1), 1.0, rtol=1e-6)


def test_pca_whitening_roundtrip(tmp_path):
    x = l2n(np.random.default_rng(0).normal(size=(500, 32)))
    p = Projector("pcaw16").fit(x)
    np.testing.assert_allclose(np.cov(p.pca.transform(x), rowvar=False), np.eye(16), atol=1e-3)
    p.save(tmp_path / "p.joblib")
    np.testing.assert_allclose(Projector.load(tmp_path / "p.joblib").transform(x), p.transform(x), rtol=1e-5)


def test_lda_projection_separates_classes():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 50)
    x = rng.normal(size=(4, 20))[labels] * 0.5 + rng.normal(size=(200, 20))
    z = Projector("lda0.1").fit(x, labels).transform(x)
    same = labels[:, None] == labels[None, :]
    assert z.shape == (200, 3) and (z @ z.T)[same].mean() > (z @ z.T)[~same].mean() + 0.5


def test_search_config_roundtrip():
    cfg = SearchConfig("ensemble", "mixed", "pad", True, "lda0.1")
    assert SearchConfig.from_dict(cfg.to_dict()) == cfg and len(cfg.members) == 4
