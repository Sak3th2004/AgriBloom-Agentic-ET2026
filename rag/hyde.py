"""
HyDE — Hypothetical Document Embeddings (Algorithm 3).

Instead of embedding the (often short, keyword-y) farmer query directly, we ask
an LLM to write a hypothetical ICAR-style answer and retrieve with THAT. It
closes the query↔document semantic gap (Gao et al. 2022). We average several
hypotheticals for stability. Uses the free ``genai_handler`` router; degrades to
the raw query if generation is unavailable.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_HYDE_PROMPT = (
    "Write a short, specific, technical answer to this agricultural question, "
    "as if quoting an ICAR expert advisory (2-3 sentences, name treatments/"
    "practices where relevant):\n\nQuestion: {query}\n\nAnswer:"
)


def _default_generate(prompt: str) -> str:
    from utils import genai_handler

    return genai_handler._generate(prompt)  # free multi-backend router


def generate_hypotheticals(
    query: str,
    n: int = 2,
    generate: Optional[Callable[[str], str]] = None,
) -> list[str]:
    """Generate ``n`` hypothetical answers for ``query`` (best-effort)."""
    gen = generate or _default_generate
    out: list[str] = []
    for _ in range(max(1, n)):
        try:
            text = (gen(_HYDE_PROMPT.format(query=query)) or "").strip()
            if text:
                out.append(text)
        except Exception as e:
            logger.warning("HyDE generation failed: %s", e)
            break
    return out


def hyde_query_text(
    query: str,
    n: int = 2,
    generate: Optional[Callable[[str], str]] = None,
) -> str:
    """Return an expanded query = original + hypothetical answers.

    Concatenating the original query with the hypotheticals is a robust form of
    HyDE for a shared embedder: it enriches the query with domain vocabulary
    without discarding the literal terms. Falls back to the raw query.
    """
    hyps = generate_hypotheticals(query, n=n, generate=generate)
    if not hyps:
        return query
    return query + "\n" + "\n".join(hyps)
