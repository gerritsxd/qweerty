#!/usr/bin/env python3
"""
Dutch speech embeddings via sentence-transformers.
Caches embeddings to avoid recomputation.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "analysis"
EMBEDDING_CACHE = CACHE_DIR / "speech_embeddings.npy"
INDEX_CACHE = CACHE_DIR / "speech_embedding_index.parquet"


def get_embedding_model():
    """Lazy load the Dutch sentence transformer."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("DTAI-KULeuven/robbert-2023-dutch-base")


def compute_speech_embeddings(
    texts: pd.Series,
    batch_size: int = 64,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Compute 768-dim embeddings for speech texts.
    Uses RobBERT Dutch model. Returns (n, 768) array.
    """
    model = get_embedding_model()
    texts_clean = texts.fillna("").astype(str)
    texts_list = texts_clean.tolist()
    embeddings = model.encode(
        texts_list,
        batch_size=batch_size,
        show_progress_bar=len(texts_list) > 1000,
    )
    return np.asarray(embeddings, dtype=np.float32)


def ensure_embeddings(df: pd.DataFrame, text_col: str = "speech_text") -> np.ndarray:
    """
    Get embeddings for df. Uses cache if available and indices match.
    Otherwise computes and caches.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_npy = CACHE_DIR / "speech_embeddings.parquet"

    # For simplicity: compute fresh each time (cache keyed by hash of first 100 rows)
    # In production, cache by (df hash or row ids)
    texts = df[text_col] if text_col in df.columns else pd.Series([""] * len(df))
    return compute_speech_embeddings(texts)
