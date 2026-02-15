#!/usr/bin/env python3
"""
Vote prediction models.

Baseline 1: Party only
Baseline 2: Party + metadata
Model A: Party + TF-IDF of speech
Model B: + embeddings
...
"""

from pathlib import Path
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
    ngram_range: tuple[int, int] = (1, 1),
    use_extra_features: bool = False,
    use_besluit_tfidf: bool = False,
    besluit_max_features: int = 500,
    use_speech_position: bool = False,
    use_speaker_loyalty: bool = False,
    use_kabinetsappreciatie: bool = False,
    use_zaak_soort: bool = False,
    use_is_coalition: bool = False,
) -> Any:
    """
    Model A: Party (one-hot) + TF-IDF of speech -> Logistic Regression.
    Optionally adds: speech_length, stance keywords, besluit_tekst TF-IDF,
    speech_position, speaker_loyalty.
    Returns fitted model dict.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from scipy.sparse import hstack
    from scipy.sparse import csr_matrix
    import numpy as np

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
    train_sub = train.loc[valid]

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    party_enc = OneHotEncoder(handle_unknown="ignore")
    X_party_enc = party_enc.fit_transform(X_party)

    tfidf = TfidfVectorizer(
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
    )
    X_tfidf = tfidf.fit_transform(text.astype(str))

    X_list = [X_party_enc, X_tfidf]

    if use_besluit_tfidf and "besluit_tekst" in train_sub.columns:
        besluit_text = train_sub["besluit_tekst"].fillna("").astype(str)
        tfidf_besluit = TfidfVectorizer(
            max_features=besluit_max_features,
            max_df=0.95,
            min_df=1,
            ngram_range=(1, 2),
        )
        X_tfidf_besluit = tfidf_besluit.fit_transform(besluit_text)
        X_list.append(X_tfidf_besluit)
    else:
        tfidf_besluit = None

    extra_parts = []
    if use_extra_features:
        speech_len = train_sub["speech_text"].fillna("").str.len().values.reshape(-1, 1)
        try:
            from src.nlp.preprocess import count_stance_keywords
            kw = train_sub["speech_text"].fillna("").apply(count_stance_keywords)
            n_voor = np.array([k[0] for k in kw]).reshape(-1, 1)
            n_tegen = np.array([k[1] for k in kw]).reshape(-1, 1)
            extra_parts.append(np.hstack([speech_len, n_voor, n_tegen]))
        except Exception:
            extra_parts.append(speech_len)
    if use_speech_position and "speech_position" in train_sub.columns:
        pos = train_sub["speech_position"].fillna(0.5).values.reshape(-1, 1)
        extra_parts.append(pos)
    if use_speaker_loyalty and "speaker_loyalty" in train_sub.columns:
        loyal = train_sub["speaker_loyalty"].fillna(0.5).values.reshape(-1, 1)
        extra_parts.append(loyal)
    if use_kabinetsappreciatie and "kabinetsappreciatie" in train_sub.columns:
        ka = train_sub["kabinetsappreciatie"].fillna("Onbekend").astype(str)
        ka_enc = OneHotEncoder(handle_unknown="ignore")
        extra_parts.append(ka_enc.fit_transform(ka.values.reshape(-1, 1)))
        model_ka_enc = ka_enc
    else:
        model_ka_enc = None
    if use_zaak_soort and "zaak_soort" in train_sub.columns:
        zs = train_sub["zaak_soort"].fillna("Onbekend").astype(str)
        zs_enc = OneHotEncoder(handle_unknown="ignore")
        extra_parts.append(zs_enc.fit_transform(zs.values.reshape(-1, 1)))
        model_zs_enc = zs_enc
    else:
        model_zs_enc = None
    if use_is_coalition and "is_coalition" in train_sub.columns:
        ic = train_sub["is_coalition"].fillna(0).values.reshape(-1, 1)
        extra_parts.append(csr_matrix(ic))
    if extra_parts:
        parts_to_stack = []
        for p in extra_parts:
            if hasattr(p, "toarray"):
                parts_to_stack.append(p)
            else:
                parts_to_stack.append(csr_matrix(np.asarray(p)))
        extra = hstack(parts_to_stack)
        X_list.append(extra)

    X = hstack(X_list)

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X, y_enc)

    return {
        "party_enc": party_enc,
        "tfidf": tfidf,
        "tfidf_besluit": tfidf_besluit,
        "clf": clf,
        "label_enc": le,
        "use_extra_features": use_extra_features,
        "use_besluit_tfidf": use_besluit_tfidf,
        "use_speech_position": use_speech_position,
        "use_speaker_loyalty": use_speaker_loyalty,
        "use_kabinetsappreciatie": use_kabinetsappreciatie,
        "use_zaak_soort": use_zaak_soort,
        "use_is_coalition": use_is_coalition,
        "ka_enc": model_ka_enc,
        "zs_enc": model_zs_enc,
    }


def predict_model_a(model: dict, df: pd.DataFrame, fractie_col: str = "fractie") -> np.ndarray:
    """Predict using Model A."""
    from scipy.sparse import hstack, csr_matrix
    import numpy as np

    text = _get_text_col(df).astype(str)
    X_party = model["party_enc"].transform(df[[fractie_col]].fillna("Onbekend"))
    X_tfidf = model["tfidf"].transform(text)
    X_list = [X_party, X_tfidf]

    if model.get("use_besluit_tfidf") and model.get("tfidf_besluit") is not None:
        besluit_text = df["besluit_tekst"].fillna("") if "besluit_tekst" in df.columns else pd.Series([""] * len(df))
        X_tfidf_besluit = model["tfidf_besluit"].transform(besluit_text.astype(str))
        X_list.append(X_tfidf_besluit)

    extra_parts = []
    if model.get("use_extra_features"):
        speech_len = df["speech_text"].fillna("").str.len().values.reshape(-1, 1)
        try:
            from src.nlp.preprocess import count_stance_keywords
            kw = df["speech_text"].fillna("").apply(count_stance_keywords)
            n_voor = np.array([k[0] for k in kw]).reshape(-1, 1)
            n_tegen = np.array([k[1] for k in kw]).reshape(-1, 1)
            extra_parts.append(np.hstack([speech_len, n_voor, n_tegen]))
        except Exception:
            extra_parts.append(speech_len)
    if model.get("use_speech_position") and "speech_position" in df.columns:
        pos = df["speech_position"].fillna(0.5).values.reshape(-1, 1)
        extra_parts.append(pos)
    if model.get("use_speaker_loyalty") and "speaker_loyalty" in df.columns:
        loyal = df["speaker_loyalty"].fillna(0.5).values.reshape(-1, 1)
        extra_parts.append(loyal)
    if model.get("use_kabinetsappreciatie") and model.get("ka_enc") is not None and "kabinetsappreciatie" in df.columns:
        ka = df["kabinetsappreciatie"].fillna("Onbekend").astype(str)
        extra_parts.append(model["ka_enc"].transform(ka.values.reshape(-1, 1)))
    if model.get("use_zaak_soort") and model.get("zs_enc") is not None and "zaak_soort" in df.columns:
        zs = df["zaak_soort"].fillna("Onbekend").astype(str)
        extra_parts.append(model["zs_enc"].transform(zs.values.reshape(-1, 1)))
    if model.get("use_is_coalition") and "is_coalition" in df.columns:
        ic = df["is_coalition"].fillna(0).values.reshape(-1, 1)
        extra_parts.append(csr_matrix(ic))
    if extra_parts:
        parts_to_stack = []
        for p in extra_parts:
            if hasattr(p, "toarray"):
                parts_to_stack.append(p)
            else:
                parts_to_stack.append(csr_matrix(np.asarray(p)))
        extra = hstack(parts_to_stack)
        X_list.append(extra)

    X = hstack(X_list)
    pred_enc = model["clf"].predict(X)
    return model["label_enc"].inverse_transform(pred_enc)


def predict_proba_model_a(
    model: dict, df: pd.DataFrame, fractie_col: str = "fractie"
) -> dict[str, float]:
    """Predict and return class probabilities (for first row)."""
    X = build_X_for_model_a(df, model, fractie_col)
    if hasattr(X, "toarray"):
        X = X.toarray()
    proba = model["clf"].predict_proba(X)[0]
    classes = model["label_enc"].classes_
    return dict(zip(classes, proba))


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


def build_X_for_model_a(
    df: pd.DataFrame,
    model: dict,
    fractie_col: str = "fractie",
) -> Any:
    """Build feature matrix for Model A pipeline (shared by predict and tree models)."""
    from scipy.sparse import hstack, csr_matrix

    text = _get_text_col(df).astype(str)
    X_party = model["party_enc"].transform(df[[fractie_col]].fillna("Onbekend"))
    X_tfidf = model["tfidf"].transform(text)
    X_list = [X_party, X_tfidf]

    if model.get("use_besluit_tfidf") and model.get("tfidf_besluit") is not None:
        besluit_text = df["besluit_tekst"].fillna("") if "besluit_tekst" in df.columns else pd.Series([""] * len(df))
        X_tfidf_besluit = model["tfidf_besluit"].transform(besluit_text.astype(str))
        X_list.append(X_tfidf_besluit)

    extra_parts = []
    if model.get("use_extra_features"):
        speech_len = df["speech_text"].fillna("").str.len().values.reshape(-1, 1)
        try:
            from src.nlp.preprocess import count_stance_keywords
            kw = df["speech_text"].fillna("").apply(count_stance_keywords)
            n_voor = np.array([k[0] for k in kw]).reshape(-1, 1)
            n_tegen = np.array([k[1] for k in kw]).reshape(-1, 1)
            extra_parts.append(np.hstack([speech_len, n_voor, n_tegen]))
        except Exception:
            extra_parts.append(speech_len)
    if model.get("use_speech_position") and "speech_position" in df.columns:
        pos = df["speech_position"].fillna(0.5).values.reshape(-1, 1)
        extra_parts.append(pos)
    if model.get("use_speaker_loyalty") and "speaker_loyalty" in df.columns:
        loyal = df["speaker_loyalty"].fillna(0.5).values.reshape(-1, 1)
        extra_parts.append(loyal)
    if model.get("use_kabinetsappreciatie") and model.get("ka_enc") is not None and "kabinetsappreciatie" in df.columns:
        ka = df["kabinetsappreciatie"].fillna("Onbekend").astype(str)
        extra_parts.append(model["ka_enc"].transform(ka.values.reshape(-1, 1)))
    if model.get("use_zaak_soort") and model.get("zs_enc") is not None and "zaak_soort" in df.columns:
        zs = df["zaak_soort"].fillna("Onbekend").astype(str)
        extra_parts.append(model["zs_enc"].transform(zs.values.reshape(-1, 1)))
    if model.get("use_is_coalition") and "is_coalition" in df.columns:
        ic = df["is_coalition"].fillna(0).values.reshape(-1, 1)
        extra_parts.append(csr_matrix(ic))
    if extra_parts:
        parts_to_stack = []
        for p in extra_parts:
            if hasattr(p, "toarray"):
                parts_to_stack.append(p)
            else:
                parts_to_stack.append(csr_matrix(np.asarray(p)))
        extra = hstack(parts_to_stack)
        X_list.append(extra)

    return hstack(X_list)


def train_model_gb(
    train: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Gradient Boosting on same features as Model A.
    kwargs passed to train_model_a for feature setup; classifier is GradientBoostingClassifier.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    model_a = train_model_a(train, **kwargs)
    X = build_X_for_model_a(train, model_a, kwargs.get("fractie_col", "fractie"))

    valid = train["vote"].isin(["Voor", "Tegen", "Niet deelgenomen"])
    if valid.sum() < 100:
        valid = train["vote"].notna()
    y = train.loc[valid, kwargs.get("target_col", "vote")]

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_sub = X[valid.values]
    if hasattr(X_sub, "toarray"):
        X_sub = X_sub.toarray()

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_sub, y_enc)

    model_a["clf"] = clf
    model_a["label_enc"] = le
    return model_a


def train_model_rf(
    train: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Random Forest on same features as Model A.
    kwargs passed to train_model_a for feature setup; classifier is RandomForestClassifier.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    model_a = train_model_a(train, **kwargs)
    X = build_X_for_model_a(train, model_a, kwargs.get("fractie_col", "fractie"))

    valid = train["vote"].isin(["Voor", "Tegen", "Niet deelgenomen"])
    if valid.sum() < 100:
        valid = train["vote"].notna()
    y = train.loc[valid, kwargs.get("target_col", "vote")]

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_sub = X[valid.values]
    if hasattr(X_sub, "toarray"):
        X_sub = X_sub.toarray()

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    clf.fit(X_sub, y_enc)

    model_a["clf"] = clf
    model_a["label_enc"] = le
    return model_a


def train_model_xgb(
    train: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    XGBoost on same features as Model A.
    Handles feature interactions better than Logistic Regression.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        XGBClassifier = lambda **kw: GradientBoostingClassifier(
            n_estimators=kw.get("n_estimators", 100),
            max_depth=kw.get("max_depth", 6),
            random_state=kw.get("random_state", 42),
        )

    from sklearn.preprocessing import LabelEncoder

    model_a = train_model_a(train, **kwargs)
    X = build_X_for_model_a(train, model_a, kwargs.get("fractie_col", "fractie"))

    valid = train["vote"].isin(["Voor", "Tegen", "Niet deelgenomen"])
    if valid.sum() < 100:
        valid = train["vote"].notna()
    y = train.loc[valid, kwargs.get("target_col", "vote")]

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_sub = X[valid.values]
    if hasattr(X_sub, "toarray"):
        X_sub = X_sub.toarray()

    clf = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_sub, y_enc)

    model_a["clf"] = clf
    model_a["label_enc"] = le
    return model_a


def train_ensemble_two_stage(
    train: pd.DataFrame,
    confidence_threshold: float = 0.75,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Two-stage ensemble: Party baseline + speech override when confident.
    Stage 1: Party prediction. Stage 2: Override with speech model only when proba > threshold.
    """
    model_baseline = train_baseline_party(train)
    model_speech = train_model_a(train, **kwargs)

    return {
        "model_baseline": model_baseline,
        "model_speech": model_speech,
        "confidence_threshold": confidence_threshold,
    }


def predict_ensemble_two_stage(
    model: dict,
    df: pd.DataFrame,
    fractie_col: str = "fractie",
) -> np.ndarray:
    """Predict using two-stage ensemble."""
    pred_baseline = predict_baseline_party(model["model_baseline"], df)
    pred_speech = predict_model_a(model["model_speech"], df)
    X = build_X_for_model_a(df, model["model_speech"], fractie_col)
    if hasattr(X, "toarray"):
        X = X.toarray()
    proba_arr = model["model_speech"]["clf"].predict_proba(X)
    max_proba = proba_arr.max(axis=1)

    result = np.where(max_proba >= model["confidence_threshold"], pred_speech, pred_baseline)
    return result


def _build_structural_X(
    df: pd.DataFrame,
    model: dict,
    fractie_col: str = "fractie",
) -> np.ndarray:
    """Build feature matrix for structural model."""
    from sklearn.preprocessing import OneHotEncoder
    from scipy.sparse import hstack, csr_matrix

    parts = []
    # fractie one-hot
    fractie = df[[fractie_col]].fillna("Onbekend")
    parts.append(model["fractie_enc"].transform(fractie))
    # topic_cluster one-hot
    if "topic_cluster" in df.columns and model.get("topic_enc") is not None:
        tc = df[["topic_cluster"]].astype(int).astype(str)
        parts.append(model["topic_enc"].transform(tc))
    # categorical
    for col, enc_key in [("zaak_soort", "zaak_enc"), ("kabinetsappreciatie", "ka_enc")]:
        if enc_key in model and model[enc_key] is not None and col in df.columns:
            c = df[col].fillna("Onbekend").astype(str)
            parts.append(model[enc_key].transform(c.values.reshape(-1, 1)))
    # numeric
    num_cols = ["is_coalition", "speaker_loyalty", "speech_position",
                "party_domain_voor_rate", "party_domain_vote_count",
                "party_recent_voor_rate", "speaker_topic_loyalty"]
    num_arr = []
    for col in num_cols:
        if col in df.columns:
            num_arr.append(df[col].fillna(0.5 if "rate" in col or "loyalty" in col or "position" in col else 0).values)
    if num_arr:
        parts.append(csr_matrix(np.column_stack(num_arr)))
    return hstack(parts)


def train_structural_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "vote",
    fractie_col: str = "fractie",
) -> dict[str, Any]:
    """
    XGBoost on structured features: fractie, topic_cluster, zaak_soort,
    kabinetsappreciatie, is_coalition, historical features, speaker_loyalty, speech_position.
    """
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from scipy.sparse import hstack, csr_matrix

    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        XGBClassifier = lambda **kw: GradientBoostingClassifier(
            n_estimators=kw.get("n_estimators", 500),
            max_depth=kw.get("max_depth", 6),
            random_state=kw.get("random_state", 42),
        )

    train_sub = train[train[target_col].isin(["Voor", "Tegen"])].copy()
    if len(train_sub) < 100:
        raise ValueError("Not enough training samples for structural model")

    fractie_enc = OneHotEncoder(handle_unknown="ignore")
    fractie_enc.fit(train_sub[[fractie_col]].fillna("Onbekend"))

    topic_enc = None
    if "topic_cluster" in train_sub.columns:
        topic_enc = OneHotEncoder(handle_unknown="ignore")
        topic_enc.fit(train_sub[["topic_cluster"]].astype(int).astype(str))

    zaak_enc = None
    if "zaak_soort" in train_sub.columns:
        zaak_enc = OneHotEncoder(handle_unknown="ignore")
        zaak_enc.fit(train_sub[["zaak_soort"]].fillna("Onbekend").astype(str))
    ka_enc = None
    if "kabinetsappreciatie" in train_sub.columns:
        ka_enc = OneHotEncoder(handle_unknown="ignore")
        ka_enc.fit(train_sub[["kabinetsappreciatie"]].fillna("Onbekend").astype(str))

    model = {
        "fractie_enc": fractie_enc,
        "topic_enc": topic_enc,
        "zaak_enc": zaak_enc,
        "ka_enc": ka_enc,
    }

    X_train = _build_structural_X(train_sub, model, fractie_col)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    y_train = train_sub[target_col].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train.astype(str))

    clf = XGBClassifier(n_estimators=500, max_depth=6, random_state=42, scale_pos_weight=1.0)
    clf.fit(X_train, y_enc)

    model["clf"] = clf
    model["label_enc"] = le
    return model


def predict_structural_model(
    model: dict,
    df: pd.DataFrame,
    fractie_col: str = "fractie",
) -> np.ndarray:
    """Predict using structural model."""
    X = _build_structural_X(df, model, fractie_col)
    if hasattr(X, "toarray"):
        X = X.toarray()
    pred_enc = model["clf"].predict(X)
    return model["label_enc"].inverse_transform(pred_enc)


def predict_proba_structural_model(
    model: dict,
    df: pd.DataFrame,
    fractie_col: str = "fractie",
) -> np.ndarray:
    """Return Voor probability (index 0) for each row."""
    X = _build_structural_X(df, model, fractie_col)
    if hasattr(X, "toarray"):
        X = X.toarray()
    proba = model["clf"].predict_proba(X)
    # LabelEncoder order: classes_[0] is first, etc. Voor=0, Tegen=1 typically
    classes = model["label_enc"].classes_
    voor_idx = np.where(classes == "Voor")[0]
    if len(voor_idx) == 0:
        return 1.0 - proba[:, 1]  # assume Tegen is index 1
    return proba[:, voor_idx[0]]


def train_ensemble_stacked(
    val: pd.DataFrame,
    proba_struct: np.ndarray,
    proba_robbert: np.ndarray,
    y_true: np.ndarray,
    structural_model: dict,
    train: pd.DataFrame,
) -> dict[str, Any]:
    """
    Meta-learner stacking: combine structural + RobBERT probabilities.
    Features: proba_struct, proba_robbert, agreement, confidence_delta, key raw features.
    """
    from sklearn.linear_model import LogisticRegression

    agree = (proba_struct > 0.5) == (proba_robbert > 0.5)
    conf_delta = np.abs(proba_struct - proba_robbert)
    X_meta = np.column_stack([
        proba_struct,
        proba_robbert,
        agree.astype(float),
        conf_delta,
    ])
    if "is_coalition" in val.columns:
        X_meta = np.column_stack([X_meta, val["is_coalition"].fillna(0).values])
    if "party_domain_voor_rate" in val.columns:
        X_meta = np.column_stack([X_meta, val["party_domain_voor_rate"].fillna(0.5).values])
    if "topic_cluster" in val.columns:
        X_meta = np.column_stack([X_meta, val["topic_cluster"].fillna(0).values])

    y_enc = (y_true == "Voor").astype(int)
    meta = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    meta.fit(X_meta, y_enc)
    return {"meta": meta, "n_features": X_meta.shape[1]}


def predict_ensemble_stacked(
    model: dict,
    df: pd.DataFrame,
    proba_struct: np.ndarray,
    proba_robbert: np.ndarray,
    structural_model: dict,
) -> np.ndarray:
    """Predict using stacked ensemble."""
    agree = (proba_struct > 0.5) == (proba_robbert > 0.5)
    conf_delta = np.abs(proba_struct - proba_robbert)
    X_meta = np.column_stack([
        proba_struct,
        proba_robbert,
        agree.astype(float),
        conf_delta,
    ])
    if "is_coalition" in df.columns:
        X_meta = np.column_stack([X_meta, df["is_coalition"].fillna(0).values])
    if "party_domain_voor_rate" in df.columns:
        X_meta = np.column_stack([X_meta, df["party_domain_voor_rate"].fillna(0.5).values])
    if "topic_cluster" in df.columns:
        X_meta = np.column_stack([X_meta, df["topic_cluster"].fillna(0).values])

    pred_enc = model["meta"].predict(X_meta)
    return np.where(pred_enc == 1, "Voor", "Tegen")


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute accuracy and per-class metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1_macro}


# --- RobBERT fine-tuned transformer ---

ROBBERT_MODEL_ID = "DTAI-KULeuven/robbert-2023-dutch-base"
MAX_SEQ_LENGTH = 256  # 256 is enough — party/besluit/topic at the start are never truncated


def _build_robbert_input_text(
    row: pd.Series,
    fractie_col: str = "fractie",
    speech_col: str = "speech_text",
    topic_col: str = "agendapunt_onderwerp",
    besluit_col: str = "besluit_tekst",
    sep: str = " </s> ",
    truncation_chars: dict | None = None,
) -> str:
    """Build input: [party] </s> [besluit_tekst] </s> [topic] </s> [speech].

    Uses </s> (RobBERT's actual SEP token) instead of literal '[SEP]'.
    Puts the most informative context (party + what's being voted on) first
    so it's never truncated.
    truncation_chars: optional dict with keys besluit, topic, speech (char limits).
    """
    tc = truncation_chars or {}
    n_besluit = tc.get("besluit", 300)
    n_topic = tc.get("topic", 200)
    n_speech = tc.get("speech", 1500)
    party = str(row.get(fractie_col) or "Onbekend")
    besluit = str(row.get(besluit_col) or "")[:n_besluit]
    topic = str(row.get(topic_col) or "")[:n_topic]
    speech = str(row.get(speech_col) or "")[:n_speech]
    return f"{party}{sep}{besluit}{sep}{topic}{sep}{speech}".strip()


def train_model_robbert(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str = "vote",
    fractie_col: str = "fractie",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-5,
    fp16: bool = True,
    max_length: int = MAX_SEQ_LENGTH,
    save_path: str | Path | None = None,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    dropout: float = 0.3,
    focal_gamma: float = 2.0,
    unfreeze_schedule: dict[int, int] | None = None,
    gradient_checkpointing: bool = False,
    score_subset: int | None = 15000,
    accum_steps: int | None = None,
    early_stopping_patience: int | None = None,
    checkpoint_every_n_epochs: int | None = None,
    truncation_chars: dict | None = None,
) -> dict[str, Any]:
    """
    Iterative self-improving RobBERT training for vote prediction.

    The model discovers its own relations from text alone — no hand-crafted
    numeric features. Training uses:

    1. **Focal loss** — rewards confident correct predictions, focuses learning
       budget on hard/uncertain cases. The model gets more "reward" (lower loss)
       when it's right, and the hardest cases get amplified gradients.

    2. **Progressive unfreezing** — starts with only the classifier head trainable,
       then gradually unfreezes encoder layers. This lets the model first learn
       what patterns to look for, then adapt its language understanding.

    3. **Per-epoch sample reweighting** — after each epoch the model scores every
       training sample. Next epoch, samples the model is uncertain about get
       higher weight. Already-mastered patterns fade into the background.

    4. **Per-epoch discovery logging** — prints what the model has learned:
       per-party accuracy, confidence distribution, improvement from last epoch.

    5. **Gradient accumulation** — when encoder layers are unfrozen, we accumulate
       gradients over multiple mini-batches to keep GPU memory usage low while
       maintaining effective batch size.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm
    from collections import defaultdict

    # --- Default unfreeze schedule: epoch -> how many layers to UNfreeze from top ---
    if unfreeze_schedule is None:
        unfreeze_schedule = {0: 0, 3: 2, 6: 4}
    if accum_steps is None:
        accum_steps = 6  # v2 default for 512 tokens
    if score_subset is None:
        score_subset = len(train)  # score all (64GB RAM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # --- Prepare data ---
    valid = train[target_col].isin(["Voor", "Tegen"])
    train_sub = train.loc[valid].copy().reset_index(drop=True)
    val_sub = val[val[target_col].isin(["Voor", "Tegen"])].copy().reset_index(drop=True)
    if len(val_sub) < 10:
        val_sub = train_sub.tail(min(500, len(train_sub)))

    label_map = {"Voor": 0, "Tegen": 1}
    inv_label = {0: "Voor", 1: "Tegen"}
    y_train = np.array([label_map[v] for v in train_sub[target_col]])
    y_val = np.array([label_map[v] for v in val_sub[target_col]])

    # Class balance for focal loss alpha
    n_voor = (y_train == 0).sum()
    n_tegen = (y_train == 1).sum()
    alpha = torch.tensor(
        [len(y_train) / (2 * n_voor), len(y_train) / (2 * n_tegen)],
        dtype=torch.float32,
    ).to(device)
    print(f"Samples: {len(y_train):,} train, {len(y_val):,} val")
    print(f"Balance: Voor={n_voor} ({100*n_voor/len(y_train):.0f}%), "
          f"Tegen={n_tegen} ({100*n_tegen/len(y_train):.0f}%)")
    print(f"Focal loss: gamma={focal_gamma}, alpha={alpha.cpu().tolist()}")

    # Track per-sample parties for discovery logging
    parties_train = train_sub[fractie_col].fillna("Onbekend").values
    parties_val = val_sub[fractie_col].fillna("Onbekend").values

    # --- Tokenize ---
    tokenizer = AutoTokenizer.from_pretrained(ROBBERT_MODEL_ID)
    print("Tokenizing texts...")
    def _text(row):
        return _build_robbert_input_text(row, fractie_col, truncation_chars=truncation_chars)
    texts_train = [_text(row) for _, row in train_sub.iterrows()]
    texts_val = [_text(row) for _, row in val_sub.iterrows()]

    enc_train = tokenizer(texts_train, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
    enc_val = tokenizer(texts_val, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")

    class VoteDataset(Dataset):
        def __init__(self, input_ids, attention_mask, labels, sample_weights=None):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels
            self.weights = sample_weights  # per-sample weight for loss

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {
                "input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels": torch.tensor(self.labels[i], dtype=torch.long),
                "idx": torch.tensor(i, dtype=torch.long),
            }
            if self.weights is not None:
                item["weight"] = torch.tensor(self.weights[i], dtype=torch.float32)
            return item

    # Start with uniform weights
    sample_weights = np.ones(len(y_train), dtype=np.float32)

    # Inference uses 4x larger batch (no gradients = less memory)
    infer_batch = batch_size * 4
    ds_val = VoteDataset(enc_val["input_ids"], enc_val["attention_mask"], y_val)
    loader_val = DataLoader(ds_val, batch_size=infer_batch,
                            pin_memory=(device.type == "cuda"))

    # --- Build model (text-only, no hand-crafted features) ---
    base = AutoModel.from_pretrained(ROBBERT_MODEL_ID)
    hidden_size = base.config.hidden_size
    n_encoder_layers = len(base.encoder.layer)

    classifier = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(256, 64),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(64, 2),
    )
    model = _RobbertVoteModel(base, classifier)
    # Gradient checkpointing is toggled dynamically: OFF when encoder frozen, ON when unfrozen
    # (avoids unnecessary double forward pass through frozen layers)
    gc_enabled = False
    model = model.to(device)

    # --- Focal loss ---
    def focal_loss(logits, targets, sample_w=None):
        """
        Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        When the model is confident and CORRECT: (1 - p_t) is small -> low loss (reward!)
        When the model is confident and WRONG: (1 - p_t) is large -> high loss (punishment!)
        When uncertain: moderate loss -> signal to keep learning

        gamma controls how much to focus on hard cases (higher = more focus).
        """
        ce = torch.nn.functional.cross_entropy(logits, targets, weight=alpha, reduction="none")
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** focal_gamma) * ce
        if sample_w is not None:
            fl = fl * sample_w
        return fl.mean()

    # --- Helper: set which layers are trainable ---
    def set_trainable_layers(n_unfrozen_from_top: int):
        """Freeze/unfreeze encoder layers. Always train classifier."""
        # Freeze everything in encoder first
        for param in base.parameters():
            param.requires_grad = False
        # Unfreeze top N layers
        if n_unfrozen_from_top > 0:
            for i in range(max(0, n_encoder_layers - n_unfrozen_from_top), n_encoder_layers):
                for param in base.encoder.layer[i].parameters():
                    param.requires_grad = True
        # Classifier always trainable
        for param in classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

    # --- Helper: compute per-sample confidence on training set ---
    @torch.no_grad()
    def compute_train_confidence():
        """Score training data for sample reweighting (subset or all)."""
        model.eval()
        n = len(y_train)
        all_correct = np.zeros(n, dtype=bool)
        all_conf = np.full(n, 0.5, dtype=np.float32)
        all_pred = np.zeros(n, dtype=np.int64)

        k = min(score_subset, n) if score_subset is not None else n
        idx = np.arange(n) if k >= n else np.random.choice(n, k, replace=False)
        sub_ds = VoteDataset(
            enc_train["input_ids"][idx],
            enc_train["attention_mask"][idx],
            y_train[idx],
        )
        sub_loader = DataLoader(sub_ds, batch_size=infer_batch, shuffle=False)

        results_c, results_f, results_p = [], [], []
        for batch in tqdm(sub_loader, desc="  Scoring samples", leave=False):
            inp = {k: v.to(device) for k, v in batch.items()
                   if k in ("input_ids", "attention_mask")}
            labels_b = batch["labels"].to(device)
            if fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(**inp)
            else:
                logits = model(**inp)
            probs = torch.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)
            results_c.append((preds == labels_b).cpu().numpy())
            results_f.append(probs.max(dim=1).values.cpu().numpy())
            results_p.append(preds.cpu().numpy())

        all_correct[idx] = np.concatenate(results_c)
        all_conf[idx] = np.concatenate(results_f)
        all_pred[idx] = np.concatenate(results_p)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        return all_correct, all_conf, all_pred

    # --- Helper: log discovered patterns ---
    def log_discoveries(epoch, val_correct, val_conf, val_pred, train_correct, train_conf):
        """Print what the model has learned this epoch."""
        # Per-party val accuracy
        party_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for i, party in enumerate(parties_val):
            party_stats[party]["total"] += 1
            if val_correct[i]:
                party_stats[party]["correct"] += 1

        print(f"\n  --- Epoch {epoch+1} Discovery Report ---")
        print(f"  Confidence: mean={val_conf.mean():.3f}, "
              f"high(>0.8)={100*(val_conf>0.8).mean():.0f}%, "
              f"low(<0.6)={100*(val_conf<0.6).mean():.0f}%")

        # Sort parties by accuracy
        party_rows = []
        for party, stats in sorted(party_stats.items(), key=lambda x: -x[1]["total"]):
            if stats["total"] >= 5:
                acc = stats["correct"] / stats["total"]
                party_rows.append((party, acc, stats["total"]))
        if party_rows:
            print("  Per-party val accuracy (>=5 samples):")
            for party, acc, n in party_rows[:10]:
                bar = "#" * int(acc * 20)
                print(f"    {party:20s} {acc*100:5.1f}% ({n:4d}) |{bar}")

        # Overall confidence-accuracy correlation
        high_conf_mask = val_conf > 0.8
        low_conf_mask = val_conf < 0.6
        if high_conf_mask.sum() > 0:
            high_acc = val_correct[high_conf_mask].mean()
            print(f"  High-confidence (>0.8) accuracy: {high_acc*100:.1f}% "
                  f"({high_conf_mask.sum()} samples)")
        if low_conf_mask.sum() > 0:
            low_acc = val_correct[low_conf_mask].mean()
            print(f"  Low-confidence  (<0.6) accuracy: {low_acc*100:.1f}% "
                  f"({low_conf_mask.sum()} samples)")

        # Training sample difficulty distribution
        mastered = (train_correct & (train_conf > 0.8)).sum()
        learning = (train_correct & (train_conf <= 0.8)).sum()
        struggling = (~train_correct).sum()
        print(f"  Training samples: mastered={mastered}, learning={learning}, "
              f"struggling={struggling}")
        print()

    # --- Training loop ---
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_state_dict = None  # keep best model weights in memory
    prev_unfrozen = -1
    step_count = 0
    epochs_without_improvement = 0

    # Pre-compute total steps for LR schedule
    total_steps = len(range(0, len(y_train), batch_size)) * epochs

    for epoch in range(epochs):
        # --- Progressive unfreezing ---
        # Find the right unfreeeze level for this epoch
        n_unfrozen = 0
        for e_thresh, n_unf in sorted(unfreeze_schedule.items()):
            if epoch >= e_thresh:
                n_unfrozen = n_unf
        if n_unfrozen != prev_unfrozen:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Toggle gradient checkpointing: only when encoder layers are trainable
            if gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
                if n_unfrozen > 0 and not gc_enabled:
                    base.gradient_checkpointing_enable()
                    gc_enabled = True
                    print("  [GradCheckpoint ON — encoder layers unfrozen]")
                elif n_unfrozen == 0 and gc_enabled:
                    base.gradient_checkpointing_disable()
                    gc_enabled = False
                    print("  [GradCheckpoint OFF — encoder frozen]")
            # Fast path: detach encoder output when fully frozen (no backward through 124M params)
            model.freeze_encoder_grad = (n_unfrozen == 0)
            if n_unfrozen == 0:
                print("  [Encoder detached — backward only through classifier head]")
            trainable, total = set_trainable_layers(n_unfrozen)
            print(f"\n>> Epoch {epoch+1}: Unfreezing top {n_unfrozen}/{n_encoder_layers} "
                  f"encoder layers ({trainable:,}/{total:,} params trainable)")
            # Re-create optimizer with new param set
            no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
            param_groups = [
                {
                    "params": [p for n, p in model.named_parameters()
                               if p.requires_grad and not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters()
                               if p.requires_grad and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            opt = torch.optim.AdamW(param_groups, lr=lr)
            # Warmup after unfreezing
            n_warmup = int(0.1 * total_steps / epochs)

            def make_lr_lambda(warm, total_ep_steps):
                def fn(step):
                    if step < warm:
                        return step / max(warm, 1)
                    return max(0.05, 1.0 - (step - warm) / max(total_ep_steps - warm, 1))
                return fn

            remaining_steps = total_steps - step_count
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, make_lr_lambda(n_warmup, remaining_steps)
            )
            scaler = torch.amp.GradScaler("cuda") if (fp16 and device.type == "cuda") else None
            prev_unfrozen = n_unfrozen

        # --- Build weighted dataloader for this epoch ---
        ds_train = VoteDataset(enc_train["input_ids"], enc_train["attention_mask"],
                               y_train, sample_weights)
        # Use WeightedRandomSampler so hard samples appear more often
        sampler_weights = torch.from_numpy(sample_weights).double()
        sampler = WeightedRandomSampler(sampler_weights, num_samples=len(y_train),
                                        replacement=True)
        loader_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=(device.type == "cuda"))

        # --- Train one epoch ---
        # Use gradient accumulation when encoder layers are unfrozen (saves GPU memory)
        epoch_accum = accum_steps if n_unfrozen > 0 else 1
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        opt.zero_grad()
        for batch_idx, batch in enumerate(tqdm(loader_train, desc=f"Epoch {epoch+1}/{epochs}")):
            inp = {k: v.to(device, non_blocking=True)
                   for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            labels = batch["labels"].to(device, non_blocking=True)
            sw = batch.get("weight")
            if sw is not None:
                sw = sw.to(device, non_blocking=True)

            if fp16 and device.type == "cuda" and scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(**inp)
                    loss = focal_loss(logits, labels, sw) / epoch_accum
                scaler.scale(loss).backward()
                if (batch_idx + 1) % epoch_accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    sched.step()
                    step_count += 1
            else:
                logits = model(**inp)
                loss = focal_loss(logits, labels, sw) / epoch_accum
                loss.backward()
                if (batch_idx + 1) % epoch_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                    sched.step()
                    step_count += 1

            epoch_loss += loss.item() * epoch_accum * labels.size(0)
            pred = logits.argmax(dim=1)
            epoch_correct += (pred == labels).sum().item()
            epoch_total += labels.size(0)

        train_acc = epoch_correct / epoch_total
        train_loss = epoch_loss / epoch_total

        # --- Validate ---
        model.eval()
        val_correct_arr = np.zeros(len(y_val), dtype=bool)
        val_conf_arr = np.zeros(len(y_val), dtype=np.float32)
        val_pred_arr = np.zeros(len(y_val), dtype=np.int64)
        val_loss_sum = 0.0
        offset = 0
        with torch.no_grad():
            for batch in loader_val:
                inp = {k: v.to(device) for k, v in batch.items()
                       if k in ("input_ids", "attention_mask")}
                labels = batch["labels"].to(device)
                if fp16 and device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(**inp)
                else:
                    logits = model(**inp)
                logits = logits.float()
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                val_loss_sum += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                conf = probs.max(dim=1).values
                bs = labels.size(0)
                val_correct_arr[offset:offset+bs] = (preds == labels).cpu().numpy()
                val_conf_arr[offset:offset+bs] = conf.cpu().numpy()
                val_pred_arr[offset:offset+bs] = preds.cpu().numpy()
                offset += bs

        val_acc = val_correct_arr.mean()
        val_loss = val_loss_sum / len(y_val)
        n_pred_voor = (val_pred_arr == 0).sum()
        n_pred_tegen = (val_pred_arr == 1).sum()

        # F1
        from sklearn.metrics import f1_score as _f1
        val_f1 = _f1(y_val, val_pred_arr, average="macro", zero_division=0)

        print(
            f"  Epoch {epoch+1}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.1f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}% val_f1={val_f1:.3f} | "
            f"pred: Voor={n_pred_voor} Tegen={n_pred_tegen}"
        )

        # --- Compute training confidence for sample reweighting ---
        train_correct_arr, train_conf_arr, train_pred_arr = compute_train_confidence()

        # --- Discovery logging ---
        log_discoveries(epoch, val_correct_arr, val_conf_arr, val_pred_arr,
                        train_correct_arr, train_conf_arr)

        # --- Update sample weights for next epoch ---
        # Mastered (correct + high confidence): low weight (model already knows this)
        # Uncertain (low confidence): high weight (model is still learning)
        # Wrong: highest weight (model needs to focus here)
        new_weights = np.ones(len(y_train), dtype=np.float32)
        mastered = train_correct_arr & (train_conf_arr > 0.85)
        uncertain = train_conf_arr < 0.65
        wrong = ~train_correct_arr

        new_weights[mastered] = 0.3      # already learned, fade out
        new_weights[uncertain] = 1.5     # still figuring out, focus more
        new_weights[wrong] = 2.0         # getting wrong, focus most

        # Smooth transition (don't change weights too abruptly)
        sample_weights = 0.5 * sample_weights + 0.5 * new_weights
        # Normalize
        sample_weights = sample_weights / sample_weights.mean()

        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            # Store best weights in memory (deep copy)
            import copy
            best_state_dict = copy.deepcopy(model.state_dict())
            if save_path:
                save_model_robbert(
                    {"model": model, "tokenizer": tokenizer,
                     "label_map": label_map, "inv_label": inv_label, "device": device},
                    str(save_path),
                )
                print(f"  >> New best! val_acc={val_acc*100:.1f}% f1={val_f1:.3f} — model saved.")
        else:
            epochs_without_improvement += 1

        # Periodic checkpoint (crash safety)
        if checkpoint_every_n_epochs and save_path and (epoch + 1) % checkpoint_every_n_epochs == 0:
            ckpt_path = str(Path(save_path).parent / f"{Path(save_path).name}_checkpoint_epoch{epoch+1}")
            save_model_robbert(
                {"model": model, "tokenizer": tokenizer,
                 "label_map": label_map, "inv_label": inv_label, "device": device},
                ckpt_path,
            )
            print(f"  >> Checkpoint saved: {ckpt_path}")

        # Early stopping
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(f"  >> Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
            break

    # Restore best model weights before returning
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"  >> Restored best model weights (val_acc={best_val_acc*100:.1f}%)")
    del best_state_dict  # free memory

    print(f"\n{'='*60}")
    print(f"Training complete. Best val accuracy: {best_val_acc*100:.1f}% (f1={best_val_f1:.3f})")
    print(f"{'='*60}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_map": label_map,
        "inv_label": inv_label,
        "device": device,
    }


def save_model_robbert(model_dict: dict, path: str | Path | None = None) -> Path:
    """Save RobBERT model, tokenizer, and config to disk."""
    import json
    import torch

    root = Path(__file__).resolve().parent.parent.parent
    out_dir = Path(path or root / "models" / "robbert_vote_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model_dict["model"].state_dict(), out_dir / "model.pt")
    model_dict["tokenizer"].save_pretrained(out_dir)
    config = {
        "label_map": model_dict["label_map"],
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return out_dir


def load_model_robbert(path: str | Path | None = None) -> dict[str, Any]:
    """Load RobBERT model from checkpoint."""
    import json
    import torch
    from transformers import AutoModel, AutoTokenizer

    root = Path(__file__).resolve().parent.parent.parent
    model_dir = Path(path or root / "models" / "robbert_vote_classifier")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"RobBERT checkpoint not found at {model_dir}")

    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    label_map = config["label_map"]
    inv_label = {int(v): k for k, v in label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    base = AutoModel.from_pretrained(ROBBERT_MODEL_ID)
    hidden_size = base.config.hidden_size
    classifier = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 64),
        torch.nn.GELU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(64, 2),
    )
    model = _RobbertVoteModel(base, classifier)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_map": label_map,
        "inv_label": inv_label,
        "device": device,
    }


def _RobbertVoteModel(encoder, classifier):
    """RobBERT + classification head. Text-only, no hand-crafted features."""
    import torch

    class _Model(torch.nn.Module):
        def __init__(self, enc, clf):
            super().__init__()
            self.encoder = enc
            self.classifier = clf
            self.freeze_encoder_grad = False  # when True, detach encoder output

        def forward(self, input_ids, attention_mask, **kwargs):
            if self.freeze_encoder_grad:
                # Fast path: no backward through encoder at all
                with torch.no_grad():
                    out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :].detach()
            else:
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
            return self.classifier(cls)

    return _Model(encoder, classifier)


def predict_model_robbert(
    model_dict: dict,
    df: pd.DataFrame,
    fractie_col: str = "fractie",
    batch_size: int = 32,
    max_length: int | None = None,
) -> np.ndarray:
    """Predict Voor/Tegen using RobBERT model."""
    import torch

    device = model_dict["device"]
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    inv_label = model_dict["inv_label"]
    max_len = max_length or MAX_SEQ_LENGTH

    model.eval()
    texts = [_build_robbert_input_text(row, fractie_col) for _, row in df.iterrows()]

    # Tokenize on CPU, only move each batch to GPU to avoid OOM
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")

    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            inp = {
                "input_ids": enc["input_ids"][i:end].to(device),
                "attention_mask": enc["attention_mask"][i:end].to(device),
            }
            logits = model(**inp)
            pred_enc = logits.argmax(dim=1).cpu().numpy()
            preds.extend([inv_label[int(p)] for p in pred_enc])
    return np.array(preds)


def get_robbert_attention(
    model_dict: dict,
    row: pd.Series,
    fractie_col: str = "fractie",
) -> tuple[np.ndarray, list[str], str]:
    """
    Get attention weights for a single row. Returns (attention_from_cls, tokens, prediction).
    attention_from_cls: (seq_len,) - how much CLS attends to each token (mean over heads, last layer).
    """
    import torch

    device = model_dict["device"]
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    inv_label = model_dict["inv_label"]

    text = _build_robbert_input_text(row, fractie_col)
    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=MAX_SEQ_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        out = model.encoder(**enc, output_attentions=True)
        attentions = out.attentions
        attn_last = attentions[-1][0].cpu().numpy()
        cls_attn = attn_last[:, 0, :].mean(axis=0)
        logits = model(**enc)
        pred_enc = logits.argmax(dim=1).item()

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().numpy())
    pred_label = inv_label[pred_enc]
    return cls_attn, tokens, pred_label


def predict_proba_model_robbert(
    model_dict: dict,
    df: pd.DataFrame,
    fractie_col: str = "fractie",
    batch_size: int = 32,
) -> dict[str, float]:
    """Predict and return class probabilities for first row."""
    import torch

    device = model_dict["device"]
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    inv_label = model_dict["inv_label"]

    model.eval()
    row = df.iloc[0]
    text = _build_robbert_input_text(row, fractie_col)

    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=MAX_SEQ_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {inv_label[i]: float(probs[i]) for i in range(len(inv_label))}


def predict_proba_model_robbert_batch(
    model_dict: dict,
    df: pd.DataFrame,
    fractie_col: str = "fractie",
    batch_size: int = 32,
    max_length: int | None = None,
) -> np.ndarray:
    """Return Voor probability for each row (n_samples,) array."""
    import torch

    device = model_dict["device"]
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    inv_label = model_dict["inv_label"]
    max_len = max_length or MAX_SEQ_LENGTH

    model.eval()
    texts = [_build_robbert_input_text(row, fractie_col) for _, row in df.iterrows()]
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")

    probs_voor = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            inp = {
                "input_ids": enc["input_ids"][i:end].to(device),
                "attention_mask": enc["attention_mask"][i:end].to(device),
            }
            logits = model(**inp)
            probs = torch.softmax(logits.float(), dim=1)
            voor_idx = 0 if inv_label.get(0) == "Voor" else 1
            probs_voor.append(probs[:, voor_idx].cpu().numpy())
    return np.concatenate(probs_voor)
