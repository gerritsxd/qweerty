#!/usr/bin/env python3
"""
Feature extraction for vote prediction.

Builds feature matrix from speech_vote_pairs + optional NLP features.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent


def load_pairs(sample: Optional[int] = None) -> pd.DataFrame:
    """Load speech-vote pairs from analysis dir."""
    path = ROOT / "data" / "analysis" / "speech_vote_pairs.parquet"
    df = pd.read_parquet(path)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)
    return df


def build_basic_features(
    df: pd.DataFrame,
    sample: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build basic feature matrix (no NLP, metadata only).
    - fractie (one-hot or label)
    - speech_length (chars)
    - Optional: vergaderjaar, activiteit_onderwerp hash
    """
    out = df.copy()
    if sample:
        out = out.sample(min(sample, len(out)), random_state=42)

    # Speech length
    out["speech_length"] = out["speech_text"].fillna("").str.len()

    # Stance keyword counts (if nlp available)
    try:
        from src.nlp.preprocess import count_stance_keywords
        kw = out["speech_text"].fillna("").apply(count_stance_keywords)
        out["n_voor_kw"] = [k[0] for k in kw]
        out["n_tegen_kw"] = [k[1] for k in kw]
    except ImportError:
        pass

    return out


def get_train_val_test(
    df: pd.DataFrame,
    date_col: str = "datum",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: train <= 2021, val 2022, test >= 2023.
    """
    df = df.copy()
    df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    train = df[df["_year"] <= 2021].drop(columns=["_year"])
    val = df[df["_year"] == 2022].drop(columns=["_year"])
    test = df[df["_year"] >= 2023].drop(columns=["_year"])
    return train, val, test
