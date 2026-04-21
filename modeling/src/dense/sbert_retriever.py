from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SBERTConfig:
    model_name: str
    batch_size: int = 16
    normalize_embeddings: bool = True


class SBERTRetriever:
    def __init__(self, config: SBERTConfig, cache_dir: str):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(config.model_name)
        self.doc_ids: list[str] = []
        self.doc_embeddings: np.ndarray | None = None

    def _cache_key(self, corpus_fingerprint: str) -> str:
        payload = {
            "model_name": self.config.model_name,
            "normalize_embeddings": self.config.normalize_embeddings,
            "corpus_fingerprint": corpus_fingerprint,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def fit(self, doc_ids: list[str], texts: list[str], corpus_fingerprint: str) -> None:
        self.doc_ids = doc_ids
        key = self._cache_key(corpus_fingerprint)
        emb_path = self.cache_dir / f"sbert_{key}.npy"
        ids_path = self.cache_dir / f"sbert_{key}_doc_ids.json"

        if emb_path.exists() and ids_path.exists():
            self.doc_embeddings = np.load(emb_path)
            self.doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            return

        self.doc_embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(emb_path, self.doc_embeddings)
        ids_path.write_text(json.dumps(self.doc_ids), encoding="utf-8")

    def retrieve(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        if self.doc_embeddings is None:
            raise RuntimeError("SBERTRetriever must be fit before retrieve.")
        query_emb = self.model.encode(
            [query_text],
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )[0]
        scores = self.doc_embeddings @ query_emb
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

