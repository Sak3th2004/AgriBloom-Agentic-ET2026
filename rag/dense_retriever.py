"""
Dense semantic retriever using sentence-transformers embeddings.

Embeds the corpus once and searches by cosine similarity. Runs fully local (on
GPU if available) with an in-memory numpy index — no server required. If a
Qdrant URL is configured it can be swapped in later; for now local numpy keeps
it free and offline. Degrades gracefully: if sentence-transformers can't load,
``available`` is False and the hybrid retriever just uses BM25.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from rag.ingestion import Document

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class DenseRetriever:
    def __init__(self, documents: list[Document], model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self.documents = documents
        self.model_name = model_name
        self._model = None
        self._matrix: Optional[np.ndarray] = None
        self._ids = [d.id for d in documents]
        self._error: Optional[str] = None
        self._build()

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception as e:
            self._error = f"embedder load failed: {e}"
            logger.warning("DenseRetriever: %s — dense retrieval disabled", self._error)
            return False

    def _build(self) -> None:
        if not self.documents or not self._ensure_model():
            return
        try:
            embs = self._model.encode(
                [d.text for d in self.documents],
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            self._matrix = _l2_normalize(embs)
            logger.info("DenseRetriever: indexed %d docs (dim=%d)", *self._matrix.shape)
        except Exception as e:
            self._error = f"embedding failed: {e}"
            logger.warning("DenseRetriever: %s", self._error)

    @property
    def available(self) -> bool:
        return self._matrix is not None

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        if not self._ensure_model():
            return None
        vec = self._model.encode([query], convert_to_numpy=True).astype("float32")
        return _l2_normalize(vec)[0]

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return ``[(doc_id, cosine), ...]`` high→low, or [] if unavailable."""
        if not self.available:
            return []
        q = self.embed_query(query)
        if q is None:
            return []
        sims = self._matrix @ q
        order = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i])) for i in order]
