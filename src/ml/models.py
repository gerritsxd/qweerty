#!/usr/bin/env python3
"""
Vote prediction models.

Baseline 1: Party only
Baseline 2: Party + metadata
Model A: Party + TF-IDF of speech
Model B: + embeddings
...
"""

from typing import Any, Optional

import pandas as pd
import numpy as np


def _get_text_col(df: pd.DataFrame) -> pd.Series:
    """Get speech text, preferring cleaned version."""
    if "speech_text" in df.columns:
        return df["speech_text"].fillna("")
    return pd.Series([""] * len(df))


def train_model_a(
    train: pd.DataFrame,
    target_col: str = "vote",
    fractie_col: str = "fractie",
    text_col: Optional[str] = None,
    max_features: int = 5000,
    max_df: float = 0.95,
    min_df: int = 2,
) -> Any:
    """
    Model A: Party (one-hot) + TF-IDF of speech -> Logistic Regression.
    Returns fitted sklearn Pipeline.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    text = _get_text_col(train) if text_col is None else train[text_col].fillna("")
    X_party = train[[fractie_col]].fillna("Onbekend")
    y = train[target_col]

    # Drop rows with rare/invalid target
    valid = y.isin(["Voor", "Tegen", "Niet deelgenomen"])
    if valid.sum() < 100:
        valid = y.notna()
    X_party = X_party[valid]
    text = text[valid]
    y = y[valid]

    ct = ColumnTransformer(
        [
            ("party", OneHotEncoder(handle_unknown="ignore"), [fractie_col]),
            ("tfidf", TfidfVectorizer(max_features=max_features, max_df=max_df, min_df=min_df), text_col or "speech_text"),
        ],
        remainder="drop",
    )
    # We need to pass both; ColumnTransformer with text needs the right column
    X_combined = pd.DataFrame({"party": X_party[fractie_col], "text": text})
    # Simpler: use a custom approach - concat party one-hot with tfidf
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    party_enc = OneHotEncoder(handle_unknown="ignore")
    X_party_enc = party_enc.fit_transform(X_party)

    tfidf = TfidfVectorizer(max_features=max_features, max_df=max_df, min_df=min_df)
    X_tfidf = tfidf.fit_transform(text.astype(str))

    from scipy.sparse import hstack
    X = hstack([X_party_enc, X_tfidf])

    clf = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    clf.fit(X, y_enc)

    return {
        "party_enc": party_enc,
        "tfidf": tfidf,
        "clf": clf,
        "label_enc": le,
    }


def predict_model_a(model: dict, df: pd.DataFrame, fractie_col: str = "fractie") -> np.ndarray:
    """Predict using Model A."""
    from scipy.sparse import hstack

    text = _get_text_col(df).astype(str)
    X_party = model["party_enc"].transform(df[[fractie_col]].fillna("Onbekend"))
    X_tfidf = model["tfidf"].transform(text)
    X = hstack([X_party, X_tfidf])
    pred_enc = model["clf"].predict(X)
    return model["label_enc"].inverse_transform(pred_enc)


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
