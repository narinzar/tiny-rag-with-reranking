"""Bi-encoder embedding of text with a sentence-transformers model.

The embedder produces L2-normalized vectors so that an inner-product FAISS index
computes cosine similarity directly.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


class Embedder:
    """Wrap a sentence-transformers bi-encoder for batched, normalized encoding."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        # Imported lazily so the module can be imported without the heavy deps
        # present (e.g. when only running the pure-python chunking tests).
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = _pick_device(device)
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        """Encode ``texts`` into an ``(n, dim)`` float32 array of unit vectors."""
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single string into a ``(dim,)`` unit vector."""
        return self.encode([text], show_progress=False)[0]
