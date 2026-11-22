import numpy as np

from src.retrieval.reranker import NeighborhoodReranker, RerankerParams, build_neighbor_table
from src.retrieval.rrf import rrf_fuse, rrf_fuse_matrix


def test_neighbor_table_excludes_self_and_uses_gallery_only():
    vecs = np.eye(5, dtype=np.float32) + 0.1
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    gallery = np.array([0, 1, 2])
    table = build_neighbor_table(vecs, gallery, k_max=2)
    assert table.shape == (5, 2)
    assert all(i not in table[i] for i in range(5)) and set(table.ravel()) <= set(gallery)


def test_reranker_promotes_candidate_sharing_the_query_neighbourhood():
    # Candidates 1 and 2 have nearly equal dense scores; only 2 shares neighbours with the query's top list.
    table = np.array([[9, 9], [7, 8], [0, 3], [0, 2], [0, 1]] + [[0, 0]] * 5)
    rr = NeighborhoodReranker(table, RerankerParams(fetch_k=4, k=2, weight=0.5))
    ids, scores = rr.rerank([3, 1, 2, 0], [0.90, 0.80, 0.79, 0.50])
    assert ids.tolist().index(2) < ids.tolist().index(1)
    assert np.all(np.diff(scores) <= 0)


def test_reranker_with_zero_weight_keeps_dense_order():
    table = np.zeros((4, 3), dtype=int)
    ids, _ = NeighborhoodReranker(table, RerankerParams(4, 3, 0.0)).rerank([2, 0, 3, 1], [0.9, 0.8, 0.7, 0.6])
    assert ids.tolist() == [2, 0, 3, 1]


def test_rrf_rewards_items_ranked_well_by_both_lists():
    fused_list = rrf_fuse([[1, 2, 3], [2, 1, 3]])
    assert set(fused_list[:2]) == {1, 2} and fused_list[-1] == 3
    fused = rrf_fuse_matrix([np.array([[0, 1, 2]]), np.array([[1, 0, 2]])], n_items=3)
    assert set(fused[0, :2].tolist()) == {0, 1} and fused[0, 2] == 2
