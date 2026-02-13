#!/usr/bin/env python3
"""
Speech text preprocessing for vote prediction.

- Clean XML artifacts, formatting noise
- Basic tokenization (regex-based, no spaCy required)
- Sentence splitting for long speeches
- Optional: spaCy nl_core_news_lg for Dutch NER (install separately)
"""

import re
from typing import Optional


def clean_speech_text(text: str) -> str:
    """
    Remove common artifacts from parliamentary speech text.
    - Speaker attribution lines (Mevrouw X (VVD):)
    - Extra whitespace, newlines
    - XML remnants if any
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove speaker attribution (first line ending with :)
    lines = text.strip().split("\n")
    if lines:
        first = lines[0]
        if re.match(r"^(?:De heer|Mevrouw|De voorzitter|Minister|Staatssecretaris)\b", first, re.I):
            lines = lines[1:]
        elif first.strip().endswith(":") and len(first) < 120:
            lines = lines[1:]

    # Join and normalize whitespace
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_simple(text: str) -> list[str]:
    """
    Simple word tokenization (no external deps).
    Splits on whitespace and punctuation, keeps words.
    """
    if not text:
        return []
    # Keep alphanumeric + Dutch chars (ë, ï, etc.)
    tokens = re.findall(r"\b[\w']+\b", text, re.UNICODE)
    return [t for t in tokens if len(t) > 0]


def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitting. Splits on . ! ? followed by space or end.
    """
    if not text:
        return []
    # Split on sentence end
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def preprocess_pipeline(text: str, clean: bool = True) -> str:
    """
    Full preprocessing pipeline for a single speech.
    Returns cleaned text ready for feature extraction.
    """
    if clean:
        text = clean_speech_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Stance keywords for topic/stance detection (Dutch)
STANCE_VOOR = ["steun", "voor", "stem voor", "stemmen voor", "aanvaard", "aannemen"]
STANCE_TEGEN = ["tegen", "verwerp", "onaanvaardbaar", "stem tegen", "afwijzen", "ontraden"]


def count_stance_keywords(text: str) -> tuple[int, int]:
    """
    Count explicit stance markers in text.
    Returns (n_voor, n_tegen).
    """
    text_lower = text.lower()
    n_voor = sum(1 for w in STANCE_VOOR if w in text_lower)
    n_tegen = sum(1 for w in STANCE_TEGEN if w in text_lower)
    return n_voor, n_tegen
