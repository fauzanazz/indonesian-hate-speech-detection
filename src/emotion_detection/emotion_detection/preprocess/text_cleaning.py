"""Minimal text normalisation helpers for Indonesian tweets."""

from __future__ import annotations

import re
from typing import Iterable, List


URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")


def normalize_text(text: str) -> str:
    """Lowercase and remove lightweight social artefacts."""

    if not isinstance(text, str):
        text = str(text)

    cleaned = text.strip().lower()
    cleaned = URL_PATTERN.sub("", cleaned)
    cleaned = MENTION_PATTERN.sub("", cleaned)
    cleaned = HASHTAG_PATTERN.sub(r"\1", cleaned)
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def clean_corpus(texts: Iterable[str]) -> List[str]:
    """Apply :func:`normalize_text` to an iterable of strings."""

    return [normalize_text(text) for text in texts]

