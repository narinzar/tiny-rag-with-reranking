"""Chunking strategies that split raw text into chunks with source offsets.

Every strategy returns a list of ``Chunk`` objects. A ``Chunk`` carries the text
plus the ``[start, end)`` character offsets into the original document, so a
retrieved chunk can always be mapped back to exactly where it came from.

Strategies
----------
- fixed_token:      contiguous windows of a fixed token count.
- sentence_aware:   pack whole sentences until a token budget is reached.
- sliding_window:   fixed-token windows with a configurable token overlap.
- adaptive:         my own strategy. It walks sentences and MERGES short ones up
                    to a target token size, and SPLITS any single sentence that
                    is longer than the target on its own. This keeps chunks near
                    a target size while respecting sentence boundaries when it
                    can, and it never drops or duplicates source text.

"Token" here means a whitespace-delimited word. That keeps the module free of a
heavy tokenizer dependency while still giving a stable, monotonic size measure
that correlates with model tokens closely enough for chunk-size decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class Chunk:
    """A single chunk of text with its offsets into the source document."""

    text: str
    start: int
    end: int
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid offsets: start={self.start} end={self.end}")

    @property
    def n_tokens(self) -> int:
        return count_tokens(self.text)


# --------------------------------------------------------------------------- #
# Token and sentence helpers
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"\S+")
# Sentence terminator followed by whitespace. We keep the terminator with the
# sentence and split on the whitespace that follows it.
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    """Count whitespace-delimited tokens."""
    return len(_WORD_RE.findall(text))


def _word_spans(text: str) -> List[tuple]:
    """Return (start, end) character spans for every whitespace-delimited word."""
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _sentence_spans(text: str) -> List[tuple]:
    """Split into sentences, returning (start, end) character spans.

    Offsets are into the original ``text``. Trailing/leading whitespace between
    sentences is not part of any span, but every non-space character belongs to
    exactly one span, so no text content is lost.
    """
    spans: List[tuple] = []
    pos = 0
    n = len(text)
    for m in _SENT_END_RE.finditer(text):
        end = m.start()
        # Trim to the actual sentence content (no leading/trailing space).
        seg = text[pos:end]
        stripped_start = pos + (len(seg) - len(seg.lstrip()))
        stripped_end = pos + len(seg.rstrip())
        if stripped_end > stripped_start:
            spans.append((stripped_start, stripped_end))
        pos = m.end()
    # Tail after the last terminator.
    if pos < n:
        seg = text[pos:n]
        stripped_start = pos + (len(seg) - len(seg.lstrip()))
        stripped_end = pos + len(seg.rstrip())
        if stripped_end > stripped_start:
            spans.append((stripped_start, stripped_end))
    return spans


def _make_chunk(text: str, start: int, end: int, strategy: str, index: int) -> Chunk:
    return Chunk(
        text=text[start:end],
        start=start,
        end=end,
        meta={"strategy": strategy, "index": index},
    )


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

def fixed_token(text: str, chunk_tokens: int = 128) -> List[Chunk]:
    """Contiguous windows of ``chunk_tokens`` words each (no overlap)."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    spans = _word_spans(text)
    chunks: List[Chunk] = []
    idx = 0
    for i in range(0, len(spans), chunk_tokens):
        window = spans[i : i + chunk_tokens]
        if not window:
            continue
        start = window[0][0]
        end = window[-1][1]
        chunks.append(_make_chunk(text, start, end, "fixed_token", idx))
        idx += 1
    return chunks


def sentence_aware(text: str, chunk_tokens: int = 128) -> List[Chunk]:
    """Pack whole sentences until adding the next would exceed ``chunk_tokens``.

    A single sentence longer than the budget becomes its own (oversized) chunk;
    ``adaptive`` is the strategy that also splits such sentences.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    sents = _sentence_spans(text)
    chunks: List[Chunk] = []
    idx = 0
    cur_start = None
    cur_end = None
    cur_tokens = 0
    for s_start, s_end in sents:
        s_tokens = count_tokens(text[s_start:s_end])
        if cur_start is None:
            cur_start, cur_end, cur_tokens = s_start, s_end, s_tokens
            continue
        if cur_tokens + s_tokens > chunk_tokens:
            chunks.append(_make_chunk(text, cur_start, cur_end, "sentence_aware", idx))
            idx += 1
            cur_start, cur_end, cur_tokens = s_start, s_end, s_tokens
        else:
            cur_end = s_end
            cur_tokens += s_tokens
    if cur_start is not None:
        chunks.append(_make_chunk(text, cur_start, cur_end, "sentence_aware", idx))
    return chunks


def sliding_window(
    text: str, chunk_tokens: int = 128, overlap_tokens: int = 32
) -> List[Chunk]:
    """Fixed-token windows that overlap by ``overlap_tokens`` words.

    Consecutive chunks share ``overlap_tokens`` words so that a fact spanning a
    boundary is not cut in half. ``overlap_tokens`` must be < ``chunk_tokens``.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be in [0, chunk_tokens)")
    spans = _word_spans(text)
    if not spans:
        return []
    step = chunk_tokens - overlap_tokens
    chunks: List[Chunk] = []
    idx = 0
    i = 0
    n = len(spans)
    while i < n:
        window = spans[i : i + chunk_tokens]
        start = window[0][0]
        end = window[-1][1]
        chunks.append(_make_chunk(text, start, end, "sliding_window", idx))
        idx += 1
        if i + chunk_tokens >= n:
            break
        i += step
    return chunks


def adaptive(
    text: str, target_tokens: int = 128, min_tokens: int = 48
) -> List[Chunk]:
    """My own strategy: merge short sentences, split long ones.

    Walk the document sentence by sentence:

    - If a single sentence is longer than ``target_tokens``, emit it as one or
      more word-window chunks of at most ``target_tokens`` words (SPLIT).
    - Otherwise accumulate sentences into a buffer. Flush the buffer when adding
      the next sentence would push it past ``target_tokens``, but only if the
      buffer already holds at least ``min_tokens`` words. This MERGES a run of
      short sentences into one chunk near the target size instead of producing a
      cloud of tiny, context-poor chunks.

    The result is chunks that cluster tightly around ``target_tokens`` while
    honoring sentence boundaries wherever a sentence fits. Every source character
    that belongs to a sentence lands in exactly one chunk (verified in tests).
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if not (0 <= min_tokens <= target_tokens):
        raise ValueError("min_tokens must be in [0, target_tokens]")

    sents = _sentence_spans(text)
    chunks: List[Chunk] = []
    idx = 0

    buf_start = None
    buf_end = None
    buf_tokens = 0

    def flush() -> None:
        nonlocal idx, buf_start, buf_end, buf_tokens
        if buf_start is not None:
            chunks.append(_make_chunk(text, buf_start, buf_end, "adaptive", idx))
            idx += 1
            buf_start, buf_end, buf_tokens = None, None, 0

    for s_start, s_end in sents:
        s_tokens = count_tokens(text[s_start:s_end])

        if s_tokens > target_tokens:
            # Long sentence: flush any buffer, then split this sentence into
            # word windows of at most target_tokens.
            flush()
            word_spans = _word_spans(text[s_start:s_end])
            for i in range(0, len(word_spans), target_tokens):
                win = word_spans[i : i + target_tokens]
                w_start = s_start + win[0][0]
                w_end = s_start + win[-1][1]
                chunks.append(_make_chunk(text, w_start, w_end, "adaptive", idx))
                idx += 1
            continue

        if buf_start is None:
            buf_start, buf_end, buf_tokens = s_start, s_end, s_tokens
            continue

        if buf_tokens + s_tokens > target_tokens and buf_tokens >= min_tokens:
            flush()
            buf_start, buf_end, buf_tokens = s_start, s_end, s_tokens
        else:
            buf_end = s_end
            buf_tokens += s_tokens

    flush()
    return chunks


# --------------------------------------------------------------------------- #
# Registry / dispatch
# --------------------------------------------------------------------------- #

STRATEGIES: Dict[str, Callable[..., List[Chunk]]] = {
    "fixed_token": fixed_token,
    "sentence_aware": sentence_aware,
    "sliding_window": sliding_window,
    "adaptive": adaptive,
}


def chunk_text(text: str, strategy: str = "adaptive", **kwargs) -> List[Chunk]:
    """Dispatch to a named strategy. Extra kwargs pass straight through."""
    if strategy not in STRATEGIES:
        raise KeyError(
            f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy](text, **kwargs)
