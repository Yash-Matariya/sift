"""Reciprocal Rank Fusion is a pure function — fully testable, no services."""
from app.retrieval import rrf_fuse


def test_item_in_both_lists_ranks_highest():
    # id 2 appears in both lists, so it should fuse to the top.
    vector = [1, 2, 3]
    keyword = [4, 2, 5]
    fused = rrf_fuse([vector, keyword], k=5)
    assert fused[0] == 2


def test_respects_k_limit():
    fused = rrf_fuse([[1, 2, 3, 4, 5], [6, 7, 8]], k=3)
    assert len(fused) == 3


def test_single_list_preserves_order():
    assert rrf_fuse([[10, 20, 30]], k=3) == [10, 20, 30]


def test_higher_rank_beats_lower():
    # id 1 is rank 0 in one list; id 9 is rank 2 in one list -> 1 should win.
    fused = rrf_fuse([[1, 8, 9]], k=3)
    assert fused.index(1) < fused.index(9)


def test_empty_lists():
    assert rrf_fuse([[], []], k=5) == []
