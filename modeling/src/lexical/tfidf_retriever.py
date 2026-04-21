from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TFIDFConfig:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.95
    use_idf: bool = True
    sublinear_tf: bool = True
    max_features: int | None = 50_000


class TFIDFRetriever:
    def __init__(self, config: TFIDFConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            use_idf=config.use_idf,
            sublinear_tf=config.sublinear_tf,
            max_features=config.max_features,
        )
        self.doc_matrix = None
        self.doc_ids: list[str] = []

    def fit(self, doc_ids: list[str], texts: list[str]) -> None:
        self.doc_ids = doc_ids
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        if self.doc_matrix is None:
            raise RuntimeError("TFIDFRetriever must be fit before retrieve.")
        query_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.doc_matrix).ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

