"""AgriBloom V2 Hybrid RAG package.

Hybrid retrieval = BM25 (keyword) + dense embeddings, fused with Reciprocal
Rank Fusion, then cross-encoder re-ranked, with optional HyDE query expansion.
See :mod:`rag.hybrid` for the orchestrator entry point.
"""
from rag.rrf_fusion import reciprocal_rank_fusion  # noqa: F401
