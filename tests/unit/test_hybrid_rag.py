"""Unit tests for BM25 + the hybrid orchestrator (dense/rerank disabled).

These run with no ML models: dense retrieval and reranking are turned off so
the BM25 + RRF + ingestion path is exercised deterministically.
"""
from __future__ import annotations

from rag.bm25_retriever import BM25Retriever, tokenize
from rag.hybrid import HybridRetriever
from rag.hyde import hyde_query_text
from rag.ingestion import Document


def _corpus():
    return [
        Document("d1", "Grape downy mildew is a fungal disease treated with copper oxychloride spray.",
                 {"crop": "grape", "disease": "downy mildew"}),
        Document("d2", "Rice blast is managed with tricyclazole and resistant varieties in Kharif season.",
                 {"crop": "rice", "disease": "blast"}),
        Document("d3", "Wheat rust control uses propiconazole; grow resistant wheat cultivars.",
                 {"crop": "wheat", "disease": "rust"}),
        Document("d4", "General IPM advice: use Trichoderma and neem oil before chemicals.",
                 {"crop": "general", "disease": "ipm"}),
    ]


def test_tokenize_basic():
    assert tokenize("Grape Downy-Mildew 2024!") == ["grape", "downy", "mildew", "2024"]


def test_bm25_ranks_keyword_match_first():
    bm25 = BM25Retriever(_corpus())
    hits = bm25.search("grape downy mildew treatment", top_k=5)
    assert hits, "expected BM25 hits"
    assert hits[0][0] == "d1"


def test_bm25_no_match_returns_empty():
    bm25 = BM25Retriever(_corpus())
    assert bm25.search("xylophone submarine") == []


def test_bm25_scores_descending():
    bm25 = BM25Retriever(_corpus())
    hits = bm25.search("rice blast Kharif")
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_query_bm25_only_shape_and_ranking():
    hr = HybridRetriever(documents=_corpus(), enable_dense=False, enable_rerank=False)
    results = hr.query("how to treat grape downy mildew", top_k=3)
    assert results
    # Drop-in shape matches V1 rag_query.
    assert set(results[0].keys()) == {"text", "metadata", "relevance_score"}
    assert results[0]["metadata"]["crop"] == "grape"


def test_hybrid_crop_filter_keeps_general():
    hr = HybridRetriever(documents=_corpus(), enable_dense=False, enable_rerank=False)
    results = hr.query("neem oil and trichoderma", top_k=5, crop="rice")
    crops = {r["metadata"]["crop"] for r in results}
    # Only rice + general should survive the crop filter.
    assert crops <= {"rice", "general"}


def test_hybrid_over_filter_falls_back_not_empty():
    hr = HybridRetriever(documents=_corpus(), enable_dense=False, enable_rerank=False)
    # A crop with no docs must not yield an empty result (fallback to candidates).
    results = hr.query("downy mildew", top_k=3, crop="banana")
    assert results  # graceful: never empty due to over-filtering


def test_hyde_falls_back_to_raw_query_without_llm():
    # generate() raises -> hyde must return the raw query unchanged.
    def broken_gen(_prompt):
        raise RuntimeError("no llm")

    assert hyde_query_text("grape disease", generate=broken_gen) == "grape disease"


def test_hyde_appends_hypotheticals():
    out = hyde_query_text("grape disease", n=2, generate=lambda p: "Use copper fungicide.")
    assert out.startswith("grape disease")
    assert "copper fungicide" in out
