"""Unit tests for Reciprocal Rank Fusion (rag/rrf_fusion.py)."""
from __future__ import annotations

from rag.rrf_fusion import reciprocal_rank_fusion


def test_single_list_preserves_order():
    lst = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
    fused = reciprocal_rank_fusion([lst])
    assert [d for d, _ in fused] == ["a", "b", "c"]


def test_item_in_both_lists_ranks_higher():
    # "b" is mid in both lists; "a" tops only one. Consensus should lift "b".
    l1 = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
    l2 = [("z", 9.0), ("b", 8.0), ("y", 7.0)]
    fused = dict(reciprocal_rank_fusion([l1, l2]))
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["z"]


def test_scores_are_scale_independent():
    # Dense cosine (0..1) vs BM25 (0..30) — RRF ignores magnitudes, uses ranks.
    dense = [("a", 0.91), ("b", 0.90)]
    bm25 = [("b", 28.0), ("a", 3.0)]
    fused = reciprocal_rank_fusion([dense, bm25])
    # a: 1/(60+1)+1/(60+2); b: 1/(60+2)+1/(60+1) -> equal, tie broken by id.
    assert {d for d, _ in fused} == {"a", "b"}


def test_k_constant_changes_weighting():
    lst = [("a", 1.0), ("b", 1.0)]
    small_k = dict(reciprocal_rank_fusion([lst], k=1))
    big_k = dict(reciprocal_rank_fusion([lst], k=1000))
    # Smaller k amplifies the gap between rank 1 and rank 2.
    assert (small_k["a"] - small_k["b"]) > (big_k["a"] - big_k["b"])


def test_top_k_truncation():
    lst = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    fused = reciprocal_rank_fusion([lst], top_k=2)
    assert len(fused) == 2
    assert [d for d, _ in fused] == ["a", "b"]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_deterministic_tie_break_by_id():
    l1 = [("x", 1.0)]
    l2 = [("a", 1.0)]
    fused = reciprocal_rank_fusion([l1, l2])
    # Equal RRF scores -> alphabetical.
    assert [d for d, _ in fused] == ["a", "x"]
