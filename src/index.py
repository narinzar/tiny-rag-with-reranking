"""FAISS index over normalized embeddings (inner product == cosine similarity)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class Hit:
    """A single retrieval result."""

    chunk_id: int
    score: float


class FaissIndex:
    """Thin wrapper around a FAISS inner-product index.

    Vectors are assumed to be L2-normalized (see ``Embedder``), so inner product
    equals cosine similarity. Chunk ids are the row positions of the vectors that
    were added, in order.
    """

    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._n = 0

    @property
    def size(self) -> int:
        return self._n

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected (n, {self.dim}) vectors, got {vectors.shape}"
            )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        self._n += vectors.shape[0]

    def search(self, query_vecs: np.ndarray, k: int) -> List[List[Hit]]:
        """Return top-``k`` hits for each query row, sorted by descending score."""
        if query_vecs.ndim == 1:
            query_vecs = query_vecs[None, :]
        query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)
        k = min(k, self._n) if self._n > 0 else 0
        if k == 0:
            return [[] for _ in range(query_vecs.shape[0])]
        scores, ids = self.index.search(query_vecs, k)
        results: List[List[Hit]] = []
        for row_scores, row_ids in zip(scores, ids):
            hits = [
                Hit(chunk_id=int(cid), score=float(sc))
                for cid, sc in zip(row_ids, row_scores)
                if cid != -1
            ]
            results.append(hits)
        return results

    def save(self, path: str) -> None:
        import faiss

        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        import faiss

        raw = faiss.read_index(path)
        obj = cls.__new__(cls)
        obj.index = raw
        obj.dim = raw.d
        obj._n = raw.ntotal
        return obj
