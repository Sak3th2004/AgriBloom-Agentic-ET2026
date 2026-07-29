"""
BM25 Okapi keyword retriever — self-contained, no external dependency.

Implemented directly (rather than pulling ``rank_bm25``) so the keyword side of
hybrid retrieval is dependency-free and fully unit-testable, and so the ranking
maths is transparent for the write-up.

BM25 score of document D for query Q:

    score(D, Q) = Σ_{t∈Q} IDF(t) · (f(t,D)·(k1+1)) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))

with IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 ).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from rag.ingestion import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer (letters + digits)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 Okapi index over a list of :class:`Document`."""

    def __init__(self, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = [tokenize(d.text) for d in self.documents]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._tf: list[Counter] = [Counter(toks) for toks in self._doc_tokens]
        self._df = self._document_frequencies()
        self._n = len(self.documents)

    def _document_frequencies(self) -> dict[str, int]:
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        return dict(df)

    def _idf(self, term: str) -> float:
        n_t = self._df.get(term, 0)
        # BM25+ style IDF with +1 to keep it non-negative.
        return math.log((self._n - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        tf = self._tf[doc_idx]
        dl = self._doc_len[doc_idx]
        score = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0))
            score += self._idf(term) * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return ``[(doc_id, score), ...]`` ranked high→low, non-zero scores only."""
        q_tokens = tokenize(query)
        scored = [
            (self.documents[i].id, self.score(q_tokens, i))
            for i in range(self._n)
        ]
        scored = [(doc_id, s) for doc_id, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]
