from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s\-']")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text_for_lexical(text: str) -> str:
    lowered = text.lower()
    stripped = _NON_WORD_PATTERN.sub(" ", lowered)
    normalized = _WHITESPACE_PATTERN.sub(" ", stripped).strip()
    return normalized


def tokenize_for_bm25(text: str, remove_stopwords: bool = True) -> list[str]:
    normalized = normalize_text_for_lexical(text)
    tokens = normalized.split()
    if not remove_stopwords:
        return tokens
    return [token for token in tokens if token not in ENGLISH_STOP_WORDS]


def batch_normalize_texts(texts: Iterable[str]) -> list[str]:
    return [normalize_text_for_lexical(text) for text in texts]


def batch_tokenize_texts(texts: Iterable[str], remove_stopwords: bool = True) -> list[list[str]]:
    return [tokenize_for_bm25(text, remove_stopwords=remove_stopwords) for text in texts]

