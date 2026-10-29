"""Deterministic BM25 retrieval over the knowledge base."""

from __future__ import annotations

import math
from collections import Counter

from .config import RetrievalConfig
from .schema import Doc, Query
from .text import tokens


class BM25:
    def __init__(self, docs: list[Doc], cfg: RetrievalConfig) -> None:
        self.docs = docs
        self.k1, self.b = cfg.k1, cfg.b
        self.tf = [Counter(tokens(f"{d.title} {d.text}")) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        q_terms = tokens(query)
        out = []
        for tf, length in zip(self.tf, self.lengths):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            for t in q_terms:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


def retrieve(docs: list[Doc], queries: list[Query], cfg: RetrievalConfig) -> list[dict]:
    """Top k per query, sorted by score desc then doc_id asc, scores rounded."""
    index = BM25(docs, cfg)
    results = []
    for q in queries:
        scored = sorted(
            ((round(s, 4), d) for s, d in zip(index.score(q.user_question), docs)),
            key=lambda pair: (-pair[0], pair[1].doc_id),
        )
        results.append(
            {
                "query_id": q.query_id,
                "retrieved": [
                    {"doc_id": d.doc_id, "score": s, "title": d.title, "text": d.text}
                    for s, d in scored[: cfg.k]
                ],
            }
        )
    return results


def is_hit(retrieved: list[dict], expected_doc_ids: list[str]) -> bool:
    """An expected doc counts only if it was retrieved with a positive score."""
    return any(r["doc_id"] in expected_doc_ids and r.get("score", 1.0) > 0 for r in retrieved)
