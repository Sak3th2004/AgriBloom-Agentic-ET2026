"""
Hybrid RAG orchestrator.

Pipeline:
    query ─┬─(optional HyDE expansion)
           ├─▶ BM25 keyword search      ─┐
           └─▶ dense semantic search    ─┴─▶ RRF fusion ─▶ cross-encoder rerank ─▶ top-k

Every stage degrades gracefully:
  * no dense model  -> BM25-only (still fused, just one list)
  * no reranker     -> RRF order preserved
  * no LLM for HyDE -> raw query used

Public entry point :func:`get_hybrid_retriever` returns a cached singleton, and
:meth:`HybridRetriever.query` returns results in the SAME shape as the V1
``rag_query`` (``[{text, metadata, relevance_score}]``) so it's a drop-in.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from rag.bm25_retriever import BM25Retriever
from rag.dense_retriever import DenseRetriever
from rag.hyde import hyde_query_text
from rag.ingestion import Document, load_corpus
from rag.reranker import CrossEncoderReranker
from rag.rrf_fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        documents: Optional[list[Document]] = None,
        enable_dense: bool = True,
        enable_rerank: bool = True,
    ) -> None:
        self.documents = documents if documents is not None else load_corpus()
        self._by_id = {d.id: d for d in self.documents}
        self.bm25 = BM25Retriever(self.documents)
        self.dense = DenseRetriever(self.documents) if enable_dense else None
        self.reranker = CrossEncoderReranker() if enable_rerank else None

    def query(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        use_hyde: bool = False,
        crop: str = "",
    ) -> list[dict[str, Any]]:
        """Retrieve the top_k most relevant documents for ``query``."""
        search_text = hyde_query_text(query) if use_hyde else query

        ranked_lists: list[list[tuple[str, float]]] = []
        bm25_hits = self.bm25.search(search_text, top_k=candidate_k)
        if bm25_hits:
            ranked_lists.append(bm25_hits)
        if self.dense is not None and self.dense.available:
            dense_hits = self.dense.search(search_text, top_k=candidate_k)
            if dense_hits:
                ranked_lists.append(dense_hits)

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(ranked_lists, top_k=candidate_k)
        candidates = [self._by_id[doc_id] for doc_id, _ in fused if doc_id in self._by_id]

        # Optional crop filter (keep general advisories, which apply to all).
        if crop:
            crop_l = crop.lower()
            filtered = [
                d for d in candidates
                if d.metadata.get("crop", "").lower() in (crop_l, "general")
            ]
            candidates = filtered or candidates  # never return empty on over-filter

        # Rerank only if the cross-encoder actually loaded; otherwise fall back
        # to RRF order + RRF score (meaningful) rather than the reranker's 0.0.
        if self.reranker is not None and self.reranker.available:
            ranked = self.reranker.rerank(query, candidates, top_k=top_k)
            return [self._as_result(d, score) for d, score in ranked]

        fused_scores = dict(fused)
        return [
            self._as_result(d, fused_scores.get(d.id, 0.0))
            for d in candidates[:top_k]
        ]

    @staticmethod
    def _as_result(doc: Document, score: float) -> dict[str, Any]:
        return {"text": doc.text, "metadata": doc.metadata, "relevance_score": float(score)}


_INSTANCE: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HybridRetriever()
    return _INSTANCE


def hybrid_query(query: str, crop: str = "", n_results: int = 5, use_hyde: bool = False) -> list[dict]:
    """Drop-in replacement for V1 ``rag_query`` using hybrid retrieval."""
    return get_hybrid_retriever().query(query, top_k=n_results, crop=crop, use_hyde=use_hyde)
