from __future__ import annotations

import re
from typing import Iterable


_BAD_CHARS_PATTERN = re.compile(r"[\u0000-\u001f]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text_for_semantic(text: str) -> str:
    cleaned = _BAD_CHARS_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def batch_normalize_semantic(texts: Iterable[str]) -> list[str]:
    return [normalize_text_for_semantic(text) for text in texts]

