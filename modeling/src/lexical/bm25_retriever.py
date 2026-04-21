from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


class BM25Retriever:
    def __init__(self, config: BM25Config):
        self.config = config
        self.doc_ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def fit(self, doc_ids: list[str], tokenized_docs: list[list[str]]) -> None:
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.config.k1, b=self.config.b)

    def retrieve(self, tokenized_query: list[str], top_k: int) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25Retriever must be fit before retrieve.")
        scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=float)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

