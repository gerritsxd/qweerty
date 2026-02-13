#!/usr/bin/env python3
"""
Vote prediction models.

Baseline 1: Party only
Baseline 2: Party + metadata
Model A: + TF-IDF of speech
Model B: + embeddings
...
"""

from typing import Any, Optional

import pandas as pd
import numpy as np


def train_baseline_party(
    train: pd.DataFrame,
    target_col: str = "vote",
    fractie_col: str = "fractie",
) -> dict[str, Any]:
    """
    Baseline: predict majority vote per party.
    Returns dict with party -> majority vote mapping.
    """
    majority = train.groupby(fractie_col)[target_col].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    return majority.to_dict()


def predict_baseline_party(
    model: dict[str, Any],
    df: pd.DataFrame,
    fractie_col: str = "fractie",
    default: str = "Voor",
) -> np.ndarray:
    """Predict using party baseline."""
    return df[fractie_col].map(lambda x: model.get(x, default)).values


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy and per-class metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1_macro}
