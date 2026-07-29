"""
Cross-encoder re-ranking.

RRF gives a good candidate set; a cross-encoder then scores each (query, doc)
pair jointly for much sharper top-k precision. Uses
``cross-encoder/ms-marco-MiniLM-L-6-v2``. Degrades gracefully: if the model
can't load, :func:`rerank` returns the candidates unchanged.
"""
from __future__ import annotations

import logging
from typing import Optional

from rag.ingestion import Document

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._error: Optional[str] = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            return True
        except Exception as e:
            self._error = f"reranker load failed: {e}"
            logger.warning("CrossEncoderReranker: %s — rerank skipped", self._error)
            return False

    @property
    def available(self) -> bool:
        return self._ensure_model()

    def rerank(
        self, query: str, candidates: list[Document], top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return ``[(Document, score), ...]`` best→worst.

        If the model is unavailable, returns the input order with score 0.0 so
        callers get a uniform shape and the pipeline keeps working.
        """
        if not candidates:
            return []
        if not self._ensure_model():
            return [(d, 0.0) for d in candidates[:top_k]]
        try:
            pairs = [(query, d.text) for d in candidates]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
            return [(d, float(s)) for d, s in ranked[:top_k]]
        except Exception as e:
            logger.warning("CrossEncoderReranker: predict failed: %s", e)
            return [(d, 0.0) for d in candidates[:top_k]]
