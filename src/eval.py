"""Retrieval metrics and a chunk-size sweep.

The eval set maps each query to a set of relevant chunk ids. Because chunk ids
depend on how the corpus was chunked, we resolve relevance by SUBSTRING: a query
is labeled with one or more answer substrings that must appear verbatim in a
relevant passage, and a chunk counts as relevant if it contains any of those
substrings. This keeps the labeled data stable across chunking strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from .chunking import Chunk, chunk_text


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #

def precision_at_k(retrieved: Sequence[int], relevant: set, k: int) -> float:
    """Fraction of the top-``k`` retrieved ids that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    hits = sum(1 for cid in top if cid in relevant)
    return hits / len(top)


def recall_at_k(retrieved: Sequence[int], relevant: set, k: int) -> float:
    """Fraction of relevant ids that appear in the top-``k`` retrieved ids."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    top = set(list(retrieved)[:k])
    hits = len(top & set(relevant))
    return hits / len(relevant)


@dataclass
class MetricSummary:
    """Averaged precision/recall over a set of queries."""

    k: int
    precision: float
    recall: float
    n_queries: int


def aggregate(
    per_query_retrieved: Sequence[Sequence[int]],
    per_query_relevant: Sequence[set],
    k: int,
) -> MetricSummary:
    """Average precision@k and recall@k across queries."""
    if len(per_query_retrieved) != len(per_query_relevant):
        raise ValueError("retrieved/relevant lists must align")
    n = len(per_query_retrieved)
    if n == 0:
        return MetricSummary(k=k, precision=0.0, recall=0.0, n_queries=0)
    p = sum(
        precision_at_k(r, rel, k)
        for r, rel in zip(per_query_retrieved, per_query_relevant)
    ) / n
    rec = sum(
        recall_at_k(r, rel, k)
        for r, rel in zip(per_query_retrieved, per_query_relevant)
    ) / n
    return MetricSummary(k=k, precision=p, recall=rec, n_queries=n)


# --------------------------------------------------------------------------- #
# Relevance labeling by substring
# --------------------------------------------------------------------------- #

def relevant_chunk_ids(chunks: Sequence[Chunk], answer_substrings: Sequence[str]) -> set:
    """Return ids of chunks whose text contains any answer substring.

    Matching is case-insensitive and whitespace-insensitive so that chunk
    boundaries collapsing a newline into a space still match.
    """
    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    needles = [norm(a) for a in answer_substrings if a.strip()]
    out = set()
    for i, ch in enumerate(chunks):
        hay = norm(ch.text)
        if any(n in hay for n in needles):
            out.add(i)
    return out


# --------------------------------------------------------------------------- #
# Chunk-size sweep
# --------------------------------------------------------------------------- #

@dataclass
class SweepRow:
    strategy: str
    chunk_tokens: int
    n_chunks: int
    n_queries_with_relevant: int
    precision: float
    recall: float


def sweep_chunk_sizes(
    documents: Dict[str, str],
    queries: Sequence[dict],
    sizes: Sequence[int],
    k: int,
    retrieve_fn: Callable[[List[Chunk], str, int], List[int]],
    strategy: str = "adaptive",
    strategy_kwargs_for_size: Callable[[int], dict] | None = None,
) -> List[SweepRow]:
    """Sweep chunk sizes and report retrieval quality for each.

    Parameters
    ----------
    documents:
        Mapping of doc id -> raw text. Chunked and concatenated into one pool.
    queries:
        Each item has keys ``question`` and ``answers`` (list of substrings).
    sizes:
        Candidate chunk-token targets to try.
    k:
        Cutoff for precision@k / recall@k.
    retrieve_fn:
        Callable ``(chunks, question, k) -> list[chunk_id]`` that does the actual
        embedding + FAISS retrieval. Injected so this module stays free of heavy
        dependencies and is easy to unit test.
    strategy:
        Chunking strategy name.
    strategy_kwargs_for_size:
        Maps a size to the strategy kwargs. Defaults to passing the size as the
        strategy's primary size argument (``chunk_tokens`` or ``target_tokens``).
    """
    if strategy_kwargs_for_size is None:
        def strategy_kwargs_for_size(size: int) -> dict:  # type: ignore[misc]
            key = "target_tokens" if strategy == "adaptive" else "chunk_tokens"
            return {key: size}

    rows: List[SweepRow] = []
    for size in sizes:
        kwargs = strategy_kwargs_for_size(size)
        chunks: List[Chunk] = []
        for _doc_id, text in documents.items():
            chunks.extend(chunk_text(text, strategy=strategy, **kwargs))

        per_retrieved: List[List[int]] = []
        per_relevant: List[set] = []
        n_with_rel = 0
        for q in queries:
            rel = relevant_chunk_ids(chunks, q["answers"])
            if rel:
                n_with_rel += 1
            retrieved = retrieve_fn(chunks, q["question"], k)
            per_retrieved.append(retrieved)
            per_relevant.append(rel)

        summary = aggregate(per_retrieved, per_relevant, k)
        rows.append(
            SweepRow(
                strategy=strategy,
                chunk_tokens=size,
                n_chunks=len(chunks),
                n_queries_with_relevant=n_with_rel,
                precision=summary.precision,
                recall=summary.recall,
            )
        )
    return rows
