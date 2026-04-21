from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class DenseConfig:
    model_name: str
    pooling: str = "mean"
    max_length: int = 384
    batch_size: int = 16
    normalize_embeddings: bool = True


class SciBERTRetriever:
    def __init__(self, config: DenseConfig, cache_dir: str):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.eval()
        self.doc_ids: list[str] = []
        self.doc_embeddings: np.ndarray | None = None

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "cls":
            return hidden_states[:, 0, :]
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_embeddings = hidden_states * mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / token_counts

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**encoded)
                pooled = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
                if self.config.normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu().numpy())
        return np.vstack(embeddings)

    def _cache_key(self, corpus_fingerprint: str) -> str:
        payload = {
            "model_name": self.config.model_name,
            "pooling": self.config.pooling,
            "max_length": self.config.max_length,
            "normalize_embeddings": self.config.normalize_embeddings,
            "corpus_fingerprint": corpus_fingerprint,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def fit(self, doc_ids: list[str], texts: list[str], corpus_fingerprint: str) -> None:
        self.doc_ids = doc_ids
        key = self._cache_key(corpus_fingerprint)
        emb_path = self.cache_dir / f"scibert_{key}.npy"
        ids_path = self.cache_dir / f"scibert_{key}_doc_ids.json"

        if emb_path.exists() and ids_path.exists():
            self.doc_embeddings = np.load(emb_path)
            self.doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            return

        self.doc_embeddings = self._encode_batch(texts)
        np.save(emb_path, self.doc_embeddings)
        ids_path.write_text(json.dumps(self.doc_ids), encoding="utf-8")

    def retrieve(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        if self.doc_embeddings is None:
            raise RuntimeError("SciBERTRetriever must be fit before retrieve.")
        query_emb = self._encode_batch([query_text])[0]
        scores = self.doc_embeddings @ query_emb
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

