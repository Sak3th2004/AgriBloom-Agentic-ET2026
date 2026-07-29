"""
Reciprocal Rank Fusion (Algorithm 2).

Fuses several ranked lists (e.g. BM25 + dense) into one, using only the RANK of
each item in each list — robust to the fact that BM25 scores and cosine
similarities are on totally different scales.

    RRF(d) = Σ_r  1 / (k + rank_r(d))

where ``rank_r`` is d's 1-based position in ranked list r (missing = skip),
and ``k`` (default 60) dampens the influence of top ranks. Pure + testable.
"""
from __future__ import annotations

from typing import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists of ``(doc_id, score)`` into one RRF-ranked list.

    Args:
        ranked_lists: each inner list is already ordered best→worst. Only the
            order matters; the per-list scores are ignored by design.
        k: RRF damping constant (60 is the standard from Cormack et al. 2009).
        top_k: optional truncation of the fused result.

    Returns:
        ``[(doc_id, rrf_score), ...]`` ordered by RRF score high→low. Ties are
        broken by doc_id for deterministic output.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused[:top_k] if top_k is not None else fused
