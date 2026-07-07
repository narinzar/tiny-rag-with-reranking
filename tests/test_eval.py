"""Tests for precision@k / recall@k and substring relevance labeling."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import Chunk  # noqa: E402
from src.eval import (  # noqa: E402
    aggregate,
    precision_at_k,
    recall_at_k,
    relevant_chunk_ids,
)


def test_precision_at_k_basic():
    retrieved = [3, 1, 7, 2, 9]
    relevant = {1, 2, 5}
    # top-3 = [3, 1, 7], one relevant (id 1) -> 1/3
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    # top-5 has ids 1 and 2 relevant -> 2/5
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(2 / 5)


def test_recall_at_k_basic():
    retrieved = [3, 1, 7, 2, 9]
    relevant = {1, 2, 5}
    # top-3 contains only id 1 of 3 relevant -> 1/3
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    # top-5 contains ids 1 and 2 of 3 relevant -> 2/3
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)


def test_perfect_and_zero():
    retrieved = [10, 11, 12]
    relevant = {10, 11, 12}
    assert precision_at_k(retrieved, relevant, 3) == 1.0
    assert recall_at_k(retrieved, relevant, 3) == 1.0

    assert precision_at_k([1, 2, 3], {9}, 3) == 0.0
    assert recall_at_k([1, 2, 3], {9}, 3) == 0.0


def test_recall_with_no_relevant_is_zero():
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_k_larger_than_retrieved():
    retrieved = [1, 2]
    relevant = {1}
    # Only two retrieved; precision divides by the actual count (2), not k.
    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(1 / 2)
    assert recall_at_k(retrieved, relevant, 5) == 1.0


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, 0)
    with pytest.raises(ValueError):
        recall_at_k([1], {1}, -1)


def test_aggregate_averages_across_queries():
    per_retrieved = [[1, 2, 3], [4, 5, 6]]
    per_relevant = [{1}, {5, 6}]
    # q0 precision@3 = 1/3, recall = 1/1 = 1
    # q1 precision@3 = 2/3, recall = 2/2 = 1
    summ = aggregate(per_retrieved, per_relevant, 3)
    assert summ.n_queries == 2
    assert summ.precision == pytest.approx((1 / 3 + 2 / 3) / 2)
    assert summ.recall == pytest.approx(1.0)


def test_aggregate_length_mismatch_raises():
    with pytest.raises(ValueError):
        aggregate([[1]], [{1}, {2}], 3)


def test_relevant_chunk_ids_substring_matching():
    chunks = [
        Chunk(text="Alice fell down the rabbit-hole slowly.", start=0, end=39),
        Chunk(text="Nothing relevant here at all.", start=39, end=68),
        Chunk(text="He had served in Afghanistan for years.", start=68, end=107),
    ]
    rel = relevant_chunk_ids(chunks, ["down the rabbit-hole", "Afghanistan"])
    assert rel == {0, 2}


def test_relevant_chunk_ids_is_whitespace_insensitive():
    # A newline inside the chunk should still match a space-joined needle.
    chunks = [Chunk(text="the White\nRabbit was late", start=0, end=25)]
    rel = relevant_chunk_ids(chunks, ["White Rabbit"])
    assert rel == {0}
