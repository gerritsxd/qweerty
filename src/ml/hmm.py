#!/usr/bin/env python3
"""
Hidden Markov Model regime detection for vote prediction.

Fits a 2-state or 3-state HMM per party on vote sequences from training data.
States: cooperative (high Voor), oppositional (high Tegen), optionally independent (mixed).
Uses Viterbi decode to infer current regime for each row.
"""

from typing import Optional

import numpy as np
import pandas as pd


def _build_party_vote_sequence(
    df: pd.DataFrame,
    fractie_col: str = "fractie",
    vote_col: str = "vote",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
) -> dict[str, list[int]]:
    """
    Per-party ordered sequence of votes (1=Voor, 0=Tegen).
    One vote per (besluit, fractie) - mode across speakers.
    """
    sub = df[df[vote_col].isin(["Voor", "Tegen"])].copy()
    if sub.empty:
        return {}
    sub["_voor"] = (sub[vote_col] == "Voor").astype(int)
    agg = sub.groupby([besluit_col, fractie_col]).agg(
        datum=(datum_col, "first"),
        _voor=("_voor", lambda x: int(x.mode().iloc[0]) if len(x.mode()) > 0 else 1),
    ).reset_index()
    agg[datum_col] = pd.to_datetime(agg[datum_col], errors="coerce")
    agg = agg.sort_values([fractie_col, datum_col])

    sequences = {}
    for party, grp in agg.groupby(fractie_col):
        seq = grp["_voor"].tolist()
        if len(seq) >= 2:
            sequences[party] = seq
    return sequences


def fit_party_hmm(
    train: pd.DataFrame,
    n_states: int = 2,
    fractie_col: str = "fractie",
    vote_col: str = "vote",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
) -> dict:
    """
    Fit HMM per party on training vote sequences.
    Returns dict with party -> fitted model (or None if too few samples).
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        return {}

    sequences = _build_party_vote_sequence(train, fractie_col, vote_col, datum_col, besluit_col)
    models = {}
    for party, seq in sequences.items():
        if len(seq) < 10:
            models[party] = None
            continue
        X = np.array(seq, dtype=float).reshape(-1, 1)
        try:
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", random_state=42)
            model.fit(X)
            models[party] = model
        except Exception:
            models[party] = None
    return models


def decode_regime(
    models: dict,
    party: str,
    vote_sequence: list[int],
) -> tuple[int, np.ndarray]:
    """
    Viterbi-decode the most likely hidden state at the end of the sequence.
    Returns (state_index, state_probs). GaussianHMM has no predict_proba, so we
    use soft one-hot: decoded state gets 0.8, others 0.2/(n-1).
    """
    if party not in models or models[party] is None or len(vote_sequence) < 2:
        n = 2
        return 0, np.ones(n) / n
    model = models[party]
    X = np.array(vote_sequence, dtype=float).reshape(-1, 1)
    try:
        _, states = model.decode(X)
        last_state = int(states[-1])
        n = model.n_components
        probs = np.full(n, (1.0 - 0.8) / max(1, n - 1))
        probs[last_state] = 0.8
        probs = probs / probs.sum()
        return last_state, probs
    except Exception:
        return 0, np.ones(model.n_components) / model.n_components


def add_hmm_regime_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    models: Optional[dict] = None,
    n_states: int = 2,
    fractie_col: str = "fractie",
    vote_col: str = "vote",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add party_regime (0..n_states-1) and party_regime_prob_0, party_regime_prob_1, ...
    to each dataframe. Fits HMM on train if models not provided.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    if models is None:
        models = fit_party_hmm(train, n_states, fractie_col, vote_col, datum_col, besluit_col)

    if not models:
        train["party_regime"] = 0
        val["party_regime"] = 0
        test["party_regime"] = 0
        for s in range(n_states):
            train[f"party_regime_prob_{s}"] = 1.0 / n_states
            val[f"party_regime_prob_{s}"] = 1.0 / n_states
            test[f"party_regime_prob_{s}"] = 1.0 / n_states
        return train, val, test

    def _add_regime_to_df(
        df: pd.DataFrame,
        party_decisions: pd.DataFrame,
    ) -> None:
        """Add regime features using cumulative vote sequences per party."""
        regime_col = []
        prob_cols = [[] for _ in range(n_states)]

        for _, row in df.iterrows():
            frac = row[fractie_col]
            dt = pd.to_datetime(row[datum_col], errors="coerce")
            bid = row.get(besluit_col)

            sub = party_decisions[
                (party_decisions[fractie_col] == frac)
                & (party_decisions[datum_col] < dt)
                & (party_decisions[besluit_col] != bid)
            ].sort_values(datum_col)
            seq = sub["_voor"].tolist()

            state, probs = decode_regime(models, frac, seq)
            regime_col.append(state)
            for s in range(n_states):
                prob_cols[s].append(probs[s] if s < len(probs) else 1.0 / n_states)

        df["party_regime"] = regime_col
        for s in range(n_states):
            df[f"party_regime_prob_{s}"] = prob_cols[s]

    # Build party decisions for each split
    def _build_dec(df_in: pd.DataFrame) -> pd.DataFrame:
        sub = df_in[df_in[vote_col].isin(["Voor", "Tegen"])].copy()
        if sub.empty:
            return pd.DataFrame()
        sub["_voor"] = (sub[vote_col] == "Voor").astype(int)
        agg = sub.groupby([besluit_col, fractie_col]).agg(
            datum=(datum_col, "first"),
            _voor=("_voor", lambda x: int(x.mode().iloc[0]) if len(x.mode()) > 0 else 1),
        ).reset_index()
        agg[datum_col] = pd.to_datetime(agg[datum_col], errors="coerce")
        return agg

    dec_train = _build_dec(train)
    dec_val = _build_dec(pd.concat([train, val], ignore_index=True))
    dec_test = _build_dec(pd.concat([train, val, test], ignore_index=True))

    _add_regime_to_df(train, dec_train)
    _add_regime_to_df(val, dec_val)
    _add_regime_to_df(test, dec_test)

    return train, val, test
