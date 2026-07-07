"""Cross-encoder reranker that re-scores retrieved chunks against the query.

A bi-encoder scores query and chunk independently, so it can miss fine-grained
relevance. A cross-encoder reads the (query, chunk) pair jointly and produces a
sharper relevance score, at higher cost. We therefore only rerank the top-k that
the FAISS retriever already surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _pick_device(device: str | None) -> str:
    if device is not None:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class RerankResult:
    """A reranked item: which retrieved chunk, and its cross-encoder score."""

    chunk_id: int
    score: float


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = _pick_device(device)
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self,
        query: str,
        chunk_ids: Sequence[int],
        chunk_texts: Sequence[str],
        top_k: int | None = None,
    ) -> List[RerankResult]:
        """Re-score ``chunk_texts`` against ``query`` and return sorted results.

        ``chunk_ids`` and ``chunk_texts`` are aligned. Returns results sorted by
        descending cross-encoder score, truncated to ``top_k`` if given.
        """
        if len(chunk_ids) != len(chunk_texts):
            raise ValueError("chunk_ids and chunk_texts must align")
        if len(chunk_ids) == 0:
            return []
        pairs = [[query, t] for t in chunk_texts]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        ranked = sorted(
            (
                RerankResult(chunk_id=int(cid), score=float(sc))
                for cid, sc in zip(chunk_ids, scores)
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
