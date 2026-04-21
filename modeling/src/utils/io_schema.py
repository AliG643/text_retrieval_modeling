from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_CORPUS_COLUMNS = ["pmid", "title", "abstract"]
REQUIRED_QUERY_COLUMNS = ["query_id", "query_text"]
REQUIRED_RESULT_COLUMNS = ["query_id", "pmid", "rank", "score", "model_name"]


@dataclass
class DataQualitySummary:
    raw_rows: int
    duplicate_pmid_rows: int
    missing_title_rows: int
    missing_abstract_rows: int
    dropped_rows: int
    kept_rows: int

    def as_dict(self) -> dict:
        return {
            "raw_rows": self.raw_rows,
            "duplicate_pmid_rows": self.duplicate_pmid_rows,
            "missing_title_rows": self.missing_title_rows,
            "missing_abstract_rows": self.missing_abstract_rows,
            "dropped_rows": self.dropped_rows,
            "kept_rows": self.kept_rows,
        }


def _validate_columns(df: pd.DataFrame, required: Iterable[str], kind: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{kind} is missing required columns: {missing}")


def validate_corpus_df(df: pd.DataFrame) -> None:
    _validate_columns(df, REQUIRED_CORPUS_COLUMNS, "Corpus CSV")


def validate_query_df(df: pd.DataFrame) -> None:
    _validate_columns(df, REQUIRED_QUERY_COLUMNS, "Query CSV")


def validate_result_df(df: pd.DataFrame) -> None:
    _validate_columns(df, REQUIRED_RESULT_COLUMNS, "Result dataframe")


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

