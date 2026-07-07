"""Tests for chunking strategies: sizes, overlap, offsets, and no lost text."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import (  # noqa: E402
    adaptive,
    chunk_text,
    count_tokens,
    fixed_token,
    sentence_aware,
    sliding_window,
)

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "A short one. "
    "Sentences of varying length appear throughout this small paragraph of text. "
    "Tiny. "
    "Here is another moderately sized sentence that carries a few more words along. "
    "End here."
)


def _words(s: str):
    return re.findall(r"\S+", s)


def test_offsets_map_back_to_source():
    for strat in ["fixed_token", "sentence_aware", "sliding_window", "adaptive"]:
        chunks = chunk_text(SAMPLE, strategy=strat)
        assert chunks, f"{strat} produced no chunks"
        for ch in chunks:
            # The stored text must equal the source slice at its offsets.
            assert ch.text == SAMPLE[ch.start : ch.end]
            assert 0 <= ch.start < ch.end <= len(SAMPLE)


def test_fixed_token_respects_size():
    size = 5
    chunks = fixed_token(SAMPLE, chunk_tokens=size)
    # Every chunk but possibly the last holds exactly `size` tokens.
    for ch in chunks[:-1]:
        assert count_tokens(ch.text) == size
    assert 1 <= count_tokens(chunks[-1].text) <= size
    # Concatenated token count equals the source token count.
    total = sum(count_tokens(c.text) for c in chunks)
    assert total == count_tokens(SAMPLE)


def test_sliding_window_overlap_is_correct():
    size, overlap = 6, 2
    chunks = sliding_window(SAMPLE, chunk_tokens=size, overlap_tokens=overlap)
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        a_words = _words(a.text)
        b_words = _words(b.text)
        # The last `overlap` words of a equal the first `overlap` words of b.
        assert a_words[-overlap:] == b_words[:overlap]
    # Non-final chunks are full size.
    for ch in chunks[:-1]:
        assert count_tokens(ch.text) == size


def test_sliding_window_rejects_bad_overlap():
    with pytest.raises(ValueError):
        sliding_window(SAMPLE, chunk_tokens=4, overlap_tokens=4)
    with pytest.raises(ValueError):
        sliding_window(SAMPLE, chunk_tokens=4, overlap_tokens=5)


def test_sentence_aware_stays_within_budget_when_possible():
    budget = 20
    chunks = sentence_aware(SAMPLE, chunk_tokens=budget)
    # Each chunk is under budget unless it is a single oversized sentence.
    for ch in chunks:
        n = count_tokens(ch.text)
        # allow single-sentence overflow only
        assert n <= budget or "." not in ch.text[:-1]


def test_adaptive_respects_target_size():
    target = 12
    chunks = adaptive(SAMPLE, target_tokens=target, min_tokens=4)
    for ch in chunks:
        # After splitting long sentences, no chunk exceeds the target.
        assert count_tokens(ch.text) <= target


def test_adaptive_loses_no_sentence_text():
    # Every word that belongs to a sentence must appear in exactly one chunk,
    # in order. We reconstruct the sequence of words from chunks and compare to
    # the sequence of words taken from the sentence spans of the source.
    from src.chunking import _sentence_spans

    target = 10
    chunks = adaptive(SAMPLE, target_tokens=target, min_tokens=3)

    source_words = []
    for s, e in _sentence_spans(SAMPLE):
        source_words.extend(_words(SAMPLE[s:e]))

    chunk_words = []
    for ch in chunks:
        chunk_words.extend(_words(ch.text))

    assert chunk_words == source_words


def test_adaptive_splits_a_very_long_sentence():
    long_sentence = "word " * 300 + "end."
    target = 50
    chunks = adaptive(long_sentence, target_tokens=target)
    assert len(chunks) >= 6  # 301 words / 50 -> 7 windows
    for ch in chunks:
        assert count_tokens(ch.text) <= target


def test_empty_text_yields_no_chunks():
    for strat in ["fixed_token", "sentence_aware", "sliding_window", "adaptive"]:
        assert chunk_text("   ", strategy=strat) == []


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        chunk_text(SAMPLE, strategy="nope")
