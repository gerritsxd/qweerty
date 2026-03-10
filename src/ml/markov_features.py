#!/usr/bin/env python3
"""
Markov chain-based sequential features for vote prediction.

Layer 1: Transition features (party/MP last vote, streak, transition prob, rolling rate)
Layer 2: Topic-conditioned transition matrices and cross-topic momentum
Layer 3: Delegated to src/ml/hmm.py (regime detection)
Layer 4: Markov P(Voor) predictor for ensemble

All lookups use only data before the current row's datum to prevent leakage.
Transition matrices and HMMs are fit on training data only.
"""

from typing import Optional

import numpy as np
import pandas as pd


def _to_voor_binary(vote: str) -> int:
    """1 = Voor, 0 = Tegen. Niet deelgenomen -> NaN handled by caller."""
    if vote == "Voor":
        return 1
    if vote == "Tegen":
        return 0
    return -1  # Niet deelgenomen or unknown


def _build_party_decisions(
    df: pd.DataFrame,
    fractie_col: str = "fractie",
    vote_col: str = "vote",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
) -> pd.DataFrame:
    """
    One row per (besluit_id, fractie): party's vote (mode across speakers) and datum.
    Used to build sequences for transition features.
    """
    sub = df[df[vote_col].isin(["Voor", "Tegen"])].copy()
    if sub.empty:
        return pd.DataFrame(columns=[besluit_col, fractie_col, datum_col, vote_col, "_voor"])
    sub["_voor"] = (sub[vote_col] == "Voor").astype(int)
    # Mode per (besluit, fractie)
    agg = sub.groupby([besluit_col, fractie_col]).agg(
        datum=(datum_col, "first"),
        _voor=("_voor", lambda x: int(x.mode().iloc[0]) if len(x.mode()) > 0 else 1),
    ).reset_index()
    agg[vote_col] = np.where(agg["_voor"] == 1, "Voor", "Tegen")
    agg[datum_col] = pd.to_datetime(agg[datum_col], errors="coerce")
    return agg


def _build_mp_decisions(
    df: pd.DataFrame,
    persoon_col: str = "persoon_id",
    vote_col: str = "vote",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
) -> pd.DataFrame:
    """One row per (besluit_id, persoon_id): MP's vote and datum."""
    sub = df[df[vote_col].isin(["Voor", "Tegen"])].copy()
    if sub.empty:
        return pd.DataFrame(columns=[besluit_col, persoon_col, datum_col, vote_col, "_voor"])
    sub["_voor"] = (sub[vote_col] == "Voor").astype(int)
    agg = sub.groupby([besluit_col, persoon_col]).agg(
        datum=(datum_col, "first"),
        _voor=("_voor", "first"),
    ).reset_index()
    agg[vote_col] = np.where(agg["_voor"] == 1, "Voor", "Tegen")
    agg[datum_col] = pd.to_datetime(agg[datum_col], errors="coerce")
    return agg


def _fit_party_transition_matrix(
    party_decisions: pd.DataFrame,
    fractie_col: str = "fractie",
) -> dict[str, np.ndarray]:
    """
    Per-party 2x2 transition matrix: P(next | prev).
    Rows: prev (0=Tegen, 1=Voor), Cols: next (0=Tegen, 1=Voor).
    Returns dict party -> (2,2) array. Fallback 0.5 for unseen.
    """
    matrices = {}
    for party, grp in party_decisions.groupby(fractie_col):
        grp = grp.sort_values("datum").reset_index(drop=True)
        if len(grp) < 2:
            matrices[party] = np.array([[0.5, 0.5], [0.5, 0.5]])
            continue
        prev = grp["_voor"].values[:-1]
        next_ = grp["_voor"].values[1:]
        counts = np.zeros((2, 2))
        for p, n in zip(prev, next_):
            if 0 <= p <= 1 and 0 <= n <= 1:
                counts[int(p), int(n)] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrices[party] = counts / row_sums
    return matrices


def _fit_party_topic_transition_matrix(
    party_decisions: pd.DataFrame,
    df_full: pd.DataFrame,
    fractie_col: str = "fractie",
    topic_col: str = "topic_cluster",
    besluit_col: str = "besluit_id",
) -> dict[tuple[str, int], np.ndarray]:
    """
    Per (party, topic_cluster) 2x2 transition matrix.
    Keys: (fractie, topic_cluster). Fallback: party-level matrix, then 0.5.
    """
    if topic_col not in df_full.columns:
        return {}
    # Merge topic into party_decisions
    besluit_to_topic = df_full.groupby(besluit_col)[topic_col].first().to_dict()
    party_dec = party_decisions.copy()
    party_dec["_topic"] = party_dec[besluit_col].map(
        lambda x: besluit_to_topic.get(x, 0)
    )
    matrices = {}
    for (party, topic), grp in party_dec.groupby([fractie_col, "_topic"]):
        grp = grp.sort_values("datum").reset_index(drop=True)
        if len(grp) < 2:
            continue
        prev = grp["_voor"].values[:-1]
        next_ = grp["_voor"].values[1:]
        counts = np.zeros((2, 2))
        for p, n in zip(prev, next_):
            if 0 <= p <= 1 and 0 <= n <= 1:
                counts[int(p), int(n)] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrices[(party, int(topic))] = counts / row_sums
    return matrices


def _get_prev_state(
    row_datum,
    row_fractie: str,
    row_besluit,
    party_decisions: pd.DataFrame,
    fractie_col: str = "fractie",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
) -> Optional[int]:
    """Previous vote (0 or 1) for this party before row_datum, excluding current besluit."""
    sub = party_decisions[
        (party_decisions[fractie_col] == row_fractie)
        & (party_decisions[datum_col] < row_datum)
        & (party_decisions[besluit_col] != row_besluit)
    ]
    if sub.empty:
        return None
    last = sub.sort_values(datum_col).iloc[-1]
    return int(last["_voor"])


def _get_prev_state_topic(
    row_datum,
    row_fractie: str,
    row_topic: int,
    row_besluit,
    party_decisions: pd.DataFrame,
    df_full: pd.DataFrame,
    fractie_col: str = "fractie",
    topic_col: str = "topic_cluster",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
) -> Optional[int]:
    """Previous vote for this party on this topic before row_datum."""
    if topic_col not in df_full.columns:
        return None
    besluit_to_topic = df_full.groupby(besluit_col)[topic_col].first().to_dict()
    sub = party_decisions[
        (party_decisions[fractie_col] == row_fractie)
        & (party_decisions[datum_col] < row_datum)
        & (party_decisions[besluit_col] != row_besluit)
    ].copy()
    sub["_topic"] = sub[besluit_col].map(lambda x: besluit_to_topic.get(x, -1))
    sub = sub[sub["_topic"] == row_topic]
    if sub.empty:
        return None
    last = sub.sort_values(datum_col).iloc[-1]
    return int(last["_voor"])


def _get_streak(
    row_datum,
    row_fractie: str,
    row_besluit,
    party_decisions: pd.DataFrame,
    fractie_col: str = "fractie",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
    target_vote: int = 1,
    max_streak: int = 10,
) -> int:
    """Consecutive Voor (target_vote=1) or Tegen (0) votes before this one."""
    sub = party_decisions[
        (party_decisions[fractie_col] == row_fractie)
        & (party_decisions[datum_col] < row_datum)
        & (party_decisions[besluit_col] != row_besluit)
    ].sort_values(datum_col, ascending=False)
    if sub.empty:
        return 0
    streak = 0
    for _, r in sub.iterrows():
        if int(r["_voor"]) == target_vote:
            streak += 1
            if streak >= max_streak:
                break
        else:
            break
    return streak


def _get_rolling_voor_rate(
    row_datum,
    row_fractie: str,
    row_besluit,
    party_decisions: pd.DataFrame,
    window: int = 5,
    fractie_col: str = "fractie",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
) -> float:
    """Voor rate over last `window` votes for this party before row_datum."""
    sub = party_decisions[
        (party_decisions[fractie_col] == row_fractie)
        & (party_decisions[datum_col] < row_datum)
        & (party_decisions[besluit_col] != row_besluit)
    ].sort_values(datum_col, ascending=False).head(window)
    if sub.empty:
        return 0.5
    return float(sub["_voor"].mean())


def build_markov_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    fractie_col: str = "fractie",
    vote_col: str = "vote",
    datum_col: str = "datum",
    besluit_col: str = "besluit_id",
    topic_col: str = "topic_cluster",
    persoon_col: str = "persoon_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add Layer 1 and Layer 2 Markov features to train, val, test.

    Layer 1: party_last_vote, party_last_topic_vote, party_voor_streak,
             party_transition_prob, party_rolling_voor_rate_5,
             mp_last_vote, mp_transition_prob (when persoon_id available)

    Layer 2: party_topic_transition_prob, cross_topic_momentum (simplified)

    Transition matrices and topic matrices are fit on TRAIN only.
    For val/test, we use train's matrices and look up previous state from
    train+val (for val) or train+val+test (for test) - but only rows with
    datum < current row's datum.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    # Build party decisions from TRAIN only for fitting transition matrices
    train_party_dec = _build_party_decisions(
        train, fractie_col, vote_col, datum_col, besluit_col
    )
    train_mp_dec = _build_mp_decisions(
        train, persoon_col, vote_col, datum_col, besluit_col
    )

    # Fit transition matrices on train
    party_trans = _fit_party_transition_matrix(train_party_dec, fractie_col)
    party_topic_trans = _fit_party_topic_transition_matrix(
        train_party_dec, train, fractie_col, topic_col, besluit_col
    )
    mp_trans = _fit_party_transition_matrix(
        train_mp_dec.rename(columns={persoon_col: fractie_col}),
        fractie_col,
    ) if persoon_col in train.columns and not train_mp_dec.empty else {}

    # Party decisions for context: train only, train+val, train+val+test
    party_dec_val = _build_party_decisions(
        pd.concat([train, val], ignore_index=True), fractie_col, vote_col, datum_col, besluit_col
    )
    party_dec_test = _build_party_decisions(
        pd.concat([train, val, test], ignore_index=True), fractie_col, vote_col, datum_col, besluit_col
    )
    mp_dec_val = _build_mp_decisions(
        pd.concat([train, val], ignore_index=True), persoon_col, vote_col, datum_col, besluit_col
    ) if persoon_col in train.columns else pd.DataFrame()
    mp_dec_test = _build_mp_decisions(
        pd.concat([train, val, test], ignore_index=True), persoon_col, vote_col, datum_col, besluit_col
    ) if persoon_col in train.columns else pd.DataFrame()

    # Topic similarity for cross_topic_momentum: use simple inverse distance
    # (same topic=1, adjacent=0.5, else 0.25). We don't have TF-IDF centroids
    # here, so we use topic_cluster identity: same=1, else 0.3
    def _cross_topic_momentum(row, party_dec, df_hist):
        """Weighted recent Voor rate by topic similarity."""
        if topic_col not in df_hist.columns:
            return 0.5
        row_topic = row.get(topic_col, 0)
        if pd.isna(row_topic):
            return 0.5
        row_topic = int(row_topic)
        row_dt = row.get("_dt") if "_dt" in row.index else pd.to_datetime(row[datum_col], errors="coerce")
        sub = party_dec[
            (party_dec[fractie_col] == row[fractie_col])
            & (party_dec[datum_col] < row_dt)
            & (party_dec[besluit_col] != row.get(besluit_col))
        ].sort_values(datum_col, ascending=False).head(10)
        if sub.empty:
            return 0.5
        besluit_to_topic = df_hist.groupby(besluit_col)[topic_col].first().to_dict()
        weights = []
        voor_vals = []
        for _, r in sub.iterrows():
            t = int(besluit_to_topic.get(r[besluit_col], -1))
            sim = 1.0 if t == row_topic else 0.3
            weights.append(sim)
            voor_vals.append(r["_voor"])
        if sum(weights) == 0:
            return 0.5
        return float(np.average(voor_vals, weights=weights))

    def _add_features_to_df(
        df: pd.DataFrame,
        party_decisions: pd.DataFrame,
        mp_decisions: pd.DataFrame,
        df_hist: pd.DataFrame,
    ) -> None:
        """Add Markov features in-place to df."""
        df["_dt"] = pd.to_datetime(df[datum_col], errors="coerce")

        party_last = []
        party_last_topic = []
        party_streak = []
        party_trans_prob = []
        party_rolling = []
        party_topic_trans_prob = []
        cross_mom = []

        for _, row in df.iterrows():
            dt = row["_dt"]
            frac = row[fractie_col]
            bid = row.get(besluit_col)
            topic = row.get(topic_col, 0)
            if pd.isna(topic):
                topic = 0
            topic = int(topic)

            # party_decisions already contains all votes; helpers filter by datum < row
            prev = _get_prev_state(dt, frac, bid, party_decisions, fractie_col, datum_col, besluit_col)
            prev_topic = _get_prev_state_topic(
                dt, frac, topic, bid, party_decisions, df_hist,
                fractie_col, topic_col, datum_col, besluit_col,
            ) if topic_col in df.columns else None

            party_last.append(prev if prev is not None else -1)
            party_last_topic.append(prev_topic if prev_topic is not None else -1)
            party_streak.append(
                _get_streak(dt, frac, bid, party_decisions, fractie_col, datum_col, besluit_col, 1, 10)
            )
            party_rolling.append(
                _get_rolling_voor_rate(dt, frac, bid, party_decisions, 5, fractie_col, datum_col, besluit_col)
            )

            # Transition prob: P(Voor | prev)
            mat = party_trans.get(frac, np.array([[0.5, 0.5], [0.5, 0.5]]))
            p_idx = prev if prev is not None else 0
            p_idx = max(0, min(1, p_idx))
            prob = float(mat[p_idx, 1])
            party_trans_prob.append(prob)

            # Topic-conditioned transition
            tmat = party_topic_trans.get((frac, topic))
            if tmat is not None and prev is not None and 0 <= prev <= 1:
                tprob = float(tmat[int(prev), 1])
            else:
                tprob = prob
            party_topic_trans_prob.append(tprob)

            cross_mom.append(_cross_topic_momentum(row, party_decisions, df_hist))

        df["party_last_vote"] = party_last
        df["party_last_topic_vote"] = party_last_topic
        df["party_voor_streak"] = party_streak
        df["party_transition_prob"] = party_trans_prob
        df["party_rolling_voor_rate_5"] = party_rolling
        df["party_topic_transition_prob"] = party_topic_trans_prob
        df["cross_topic_momentum"] = cross_mom

        # MP-level (optional)
        if persoon_col in df.columns and mp_trans:
            mp_last = []
            mp_trans_prob = []
            for _, row in df.iterrows():
                pid = row[persoon_col]
                dt = row["_dt"]
                bid = row.get(besluit_col)
                mp_dec_sub = mp_decisions[
                    (mp_decisions[persoon_col] == pid)
                    & (mp_decisions[datum_col] < dt)
                    & (mp_decisions[besluit_col] != bid)
                ].sort_values(datum_col)
                if mp_dec_sub.empty:
                    mp_last.append(-1)
                    mp_trans_prob.append(0.5)
                else:
                    last_row = mp_dec_sub.iloc[-1]
                    prev_mp = int(last_row["_voor"])
                    mp_last.append(prev_mp)
                    mat = mp_trans.get(pid, np.array([[0.5, 0.5], [0.5, 0.5]]))
                    p_idx = max(0, min(1, prev_mp))
                    mp_trans_prob.append(float(mat[p_idx, 1]))
            df["mp_last_vote"] = mp_last
            df["mp_transition_prob"] = mp_trans_prob
        else:
            df["mp_last_vote"] = -1
            df["mp_transition_prob"] = 0.5

        df.drop(columns=["_dt"], errors="ignore", inplace=True)

    # For each row we need "all votes before this row". Train: use train. Val: use train. Test: use train+val.
    _add_features_to_df(train, train_party_dec, train_mp_dec, train)
    _add_features_to_df(val, party_dec_val, mp_dec_val, pd.concat([train, val], ignore_index=True))
    _add_features_to_df(test, party_dec_test, mp_dec_test, pd.concat([train, val, test], ignore_index=True))

    # Fill NaN for party_last_vote / party_last_topic_vote (-1) -> use 0.5 for transition prob
    for df in (train, val, test):
        df["party_last_vote"] = df["party_last_vote"].replace(-1, np.nan)
        df["party_last_topic_vote"] = df["party_last_topic_vote"].replace(-1, np.nan)

    # Layer 3: HMM regime detection
    try:
        from src.ml.hmm import add_hmm_regime_features
        train, val, test = add_hmm_regime_features(
            train, val, test,
            fractie_col=fractie_col,
            vote_col=vote_col,
            datum_col=datum_col,
            besluit_col=besluit_col,
        )
    except Exception:
        pass  # hmmlearn may not be installed; regime features optional

    return train, val, test


def predict_markov_proba(
    df: pd.DataFrame,
    train: pd.DataFrame,
    fractie_col: str = "fractie",
    topic_col: str = "topic_cluster",
) -> np.ndarray:
    """
    Layer 4: Combine transition prob, topic prob, rolling rate, cross-topic momentum,
    and regime state into a single P(Voor) estimate for the ensemble.
    """
    if "party_transition_prob" not in df.columns or "party_topic_transition_prob" not in df.columns:
        return np.full(len(df), 0.5, dtype=np.float32)

    p1 = df["party_transition_prob"].fillna(0.5).values
    p2 = df["party_topic_transition_prob"].fillna(0.5).values
    p3 = df.get("party_rolling_voor_rate_5", pd.Series(0.5, index=df.index)).fillna(0.5).values
    p4 = df.get("cross_topic_momentum", pd.Series(0.5, index=df.index)).fillna(0.5).values

    # Regime: state 0 typically = cooperative (high Voor), state 1 = oppositional
    if "party_regime_prob_0" in df.columns:
        p5 = df["party_regime_prob_0"].fillna(0.5).values
        proba = 0.30 * p1 + 0.30 * p2 + 0.12 * p3 + 0.12 * p4 + 0.16 * p5
    else:
        proba = 0.35 * p1 + 0.35 * p2 + 0.15 * p3 + 0.15 * p4
    return np.clip(proba.astype(np.float32), 0.01, 0.99)
