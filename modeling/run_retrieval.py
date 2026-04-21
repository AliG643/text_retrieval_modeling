from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.dense.sbert_retriever import SBERTConfig, SBERTRetriever
from src.dense.scibert_retriever import DenseConfig, SciBERTRetriever
from src.lexical.bm25_retriever import BM25Config, BM25Retriever
from src.lexical.tfidf_retriever import TFIDFConfig, TFIDFRetriever
from src.loader.corpus_loader import load_and_prepare_corpus
from src.preprocess.lexical_preprocess import batch_normalize_texts, batch_tokenize_texts, tokenize_for_bm25
from src.preprocess.semantic_preprocess import batch_normalize_semantic, normalize_text_for_semantic
from src.utils.io_schema import ensure_parent_dir, validate_query_df, validate_result_df


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_queries(queries_csv_path: str) -> pd.DataFrame:
    query_df = pd.read_csv(queries_csv_path)
    validate_query_df(query_df)
    query_df["query_id"] = query_df["query_id"].astype(str).str.strip()
    query_df["query_text"] = query_df["query_text"].astype(str).str.strip()
    return query_df


def corpus_fingerprint(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    for row in df[["doc_id", "document_text"]].itertuples(index=False):
        hasher.update(str(row.doc_id).encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update(str(row.document_text).encode("utf-8"))
        hasher.update(b"\x1e")
    return hasher.hexdigest()


def format_results(model_name: str, query_id: str, ranked: list[tuple[str, float]]) -> list[dict]:
    return [
        {
            "query_id": query_id,
            "pmid": str(doc_id),
            "rank": rank,
            "score": float(score),
            "model_name": model_name,
        }
        for rank, (doc_id, score) in enumerate(ranked, start=1)
    ]


def export_results(df: pd.DataFrame, output_dir: str, model_name: str, top_k_values: list[int]) -> list[str]:
    saved_paths: list[str] = []
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for k in top_k_values:
        subset = df[df["rank"] <= k].copy()
        output_path = model_dir / f"{model_name}_top{k}.csv"
        subset.to_csv(output_path, index=False)
        saved_paths.append(str(output_path))
    return saved_paths


def append_run_log(log_path: str, payload: dict) -> None:
    ensure_parent_dir(log_path)
    with Path(log_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def run_single_model(model_name: str, config: dict) -> None:
    paths = config["paths"]
    retrieval_cfg = config["retrieval"]

    corpus_df, quality_summary = load_and_prepare_corpus(
        corpus_csv_path=paths["corpus_csv"],
        document_mode=retrieval_cfg.get("document_mode", "title_abstract"),
    )
    query_df = load_queries(paths["queries_csv"])
    corpus_hash = corpus_fingerprint(corpus_df)

    doc_ids = corpus_df["doc_id"].astype(str).tolist()
    doc_texts = corpus_df["document_text"].astype(str).tolist()

    max_k = max(retrieval_cfg["top_k_values"])
    all_rows: list[dict] = []

    if model_name == "tfidf":
        tfidf_cfg = config["tfidf"]
        retriever = TFIDFRetriever(
            TFIDFConfig(
                ngram_range=tuple(tfidf_cfg["ngram_range"]),
                min_df=tfidf_cfg["min_df"],
                max_df=tfidf_cfg["max_df"],
                use_idf=tfidf_cfg["use_idf"],
                sublinear_tf=tfidf_cfg["sublinear_tf"],
                max_features=tfidf_cfg["max_features"],
            )
        )
        retriever.fit(doc_ids, batch_normalize_texts(doc_texts))
        for query_id, query_text in query_df[["query_id", "query_text"]].itertuples(index=False):
            ranked = retriever.retrieve(query_text=batch_normalize_texts([query_text])[0], top_k=max_k)
            all_rows.extend(format_results("tfidf", query_id, ranked))

    elif model_name == "bm25":
        bm25_cfg = config["bm25"]
        retriever = BM25Retriever(BM25Config(k1=bm25_cfg["k1"], b=bm25_cfg["b"]))
        retriever.fit(doc_ids, batch_tokenize_texts(doc_texts, remove_stopwords=True))
        for query_id, query_text in query_df[["query_id", "query_text"]].itertuples(index=False):
            ranked = retriever.retrieve(tokenized_query=tokenize_for_bm25(query_text), top_k=max_k)
            all_rows.extend(format_results("bm25", query_id, ranked))

    elif model_name == "scibert":
        dense_cfg = config["dense"]
        retriever = SciBERTRetriever(
            DenseConfig(
                model_name=dense_cfg["scibert_model_name"],
                pooling=dense_cfg["pooling"],
                max_length=dense_cfg["max_length"],
                batch_size=dense_cfg["batch_size"],
                normalize_embeddings=dense_cfg["normalize_embeddings"],
            ),
            cache_dir=paths["embedding_cache_dir"],
        )
        normalized_docs = batch_normalize_semantic(doc_texts)
        retriever.fit(doc_ids=doc_ids, texts=normalized_docs, corpus_fingerprint=corpus_hash)
        for query_id, query_text in query_df[["query_id", "query_text"]].itertuples(index=False):
            ranked = retriever.retrieve(query_text=normalize_text_for_semantic(query_text), top_k=max_k)
            all_rows.extend(format_results("scibert", query_id, ranked))

    elif model_name == "sbert":
        dense_cfg = config["dense"]
        retriever = SBERTRetriever(
            SBERTConfig(
                model_name=dense_cfg["sbert_model_name"],
                batch_size=dense_cfg["batch_size"],
                normalize_embeddings=dense_cfg["normalize_embeddings"],
            ),
            cache_dir=paths["embedding_cache_dir"],
        )
        normalized_docs = batch_normalize_semantic(doc_texts)
        retriever.fit(doc_ids=doc_ids, texts=normalized_docs, corpus_fingerprint=corpus_hash)
        for query_id, query_text in query_df[["query_id", "query_text"]].itertuples(index=False):
            ranked = retriever.retrieve(query_text=normalize_text_for_semantic(query_text), top_k=max_k)
            all_rows.extend(format_results("sbert", query_id, ranked))

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    result_df = pd.DataFrame(all_rows, columns=["query_id", "pmid", "rank", "score", "model_name"])
    validate_result_df(result_df)
    output_paths = export_results(
        df=result_df,
        output_dir=paths["output_dir"],
        model_name=model_name,
        top_k_values=retrieval_cfg["top_k_values"],
    )

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "query_count": int(query_df["query_id"].nunique()),
        "corpus_size": int(len(corpus_df)),
        "top_k_values": retrieval_cfg["top_k_values"],
        "output_paths": output_paths,
        "corpus_fingerprint": corpus_hash,
        "document_mode": retrieval_cfg.get("document_mode", "title_abstract"),
        "quality_summary": quality_summary.as_dict(),
    }
    append_run_log(paths["run_log_path"], run_meta)
    print(json.dumps(run_meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one retrieval model.")
    parser.add_argument("--config", default="modeling/src/config/defaults.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["tfidf", "bm25", "scibert", "sbert"],
        help="Model to execute.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_single_model(model_name=args.model, config=config)


if __name__ == "__main__":
    main()

