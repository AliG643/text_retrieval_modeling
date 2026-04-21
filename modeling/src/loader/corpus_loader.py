from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io_schema import DataQualitySummary, validate_corpus_df


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_document_text(title: str, abstract: str, mode: str = "title_abstract") -> str:
    if mode == "title_only":
        return title
    if mode == "abstract_only":
        return abstract
    return f"{title} [SEP] {abstract}".strip()


def load_and_prepare_corpus(corpus_csv_path: str, document_mode: str = "title_abstract") -> tuple[pd.DataFrame, DataQualitySummary]:
    corpus_path = Path(corpus_csv_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus CSV not found: {corpus_path}")

    raw_df = pd.read_csv(corpus_path)
    validate_corpus_df(raw_df)

    df = raw_df.copy()
    raw_rows = len(df)
    df["pmid"] = df["pmid"].astype(str).str.strip()
    df["title"] = df["title"].apply(_safe_text)
    df["abstract"] = df["abstract"].apply(_safe_text)

    duplicate_mask = df.duplicated(subset=["pmid"], keep="first")
    missing_title_mask = df["title"].eq("")
    missing_abstract_mask = df["abstract"].eq("")
    invalid_mask = duplicate_mask | missing_title_mask | missing_abstract_mask

    cleaned_df = df.loc[~invalid_mask].copy()
    cleaned_df = cleaned_df.rename(columns={"pmid": "doc_id"})
    cleaned_df["document_text"] = cleaned_df.apply(
        lambda row: build_document_text(row["title"], row["abstract"], mode=document_mode),
        axis=1,
    )

    summary = DataQualitySummary(
        raw_rows=raw_rows,
        duplicate_pmid_rows=int(duplicate_mask.sum()),
        missing_title_rows=int(missing_title_mask.sum()),
        missing_abstract_rows=int(missing_abstract_mask.sum()),
        dropped_rows=int(invalid_mask.sum()),
        kept_rows=len(cleaned_df),
    )
    return cleaned_df[["doc_id", "title", "abstract", "document_text"]], summary

