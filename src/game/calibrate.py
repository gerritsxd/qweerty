#!/usr/bin/env python3
"""
Calibrate game-theoretic model parameters from training data.

Fits payoff weights w1, w2, w3, w4 and temperature using maximum likelihood
(or grid search) on the training split.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.game.simulation import run_simulation, payoff_to_probability
from src.game.payoffs import compute_payoffs_batch
from src.game.players import PartyProfiler, get_party_ideal_points
from src.game.bills import BillPositioner, get_bill_positions

ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = ROOT / "data" / "analysis"


def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy loss."""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def calibrate_weights(
    train: pd.DataFrame,
    val: Optional[pd.DataFrame] = None,
    n_iter: int = 50,
    fractie_col: str = "fractie",
    vote_col: str = "vote",
) -> dict:
    """
    Calibrate w1, w2, w3, w4, temperature by minimizing BCE on train.
    Uses coordinate descent / grid search over weight combinations.
    """
    train_vt = train[train[vote_col].isin(["Voor", "Tegen"])].copy()
    if train_vt.empty:
        return {"w1": 0.5, "w2": 0.3, "w3": 0.1, "w4": 0.1, "temperature": 1.0}
    y_true = (train_vt[vote_col] == "Voor").astype(int).values

    prof = PartyProfiler()
    pos = BillPositioner(party_profiler=prof, train_df=train)
    party_ideals = get_party_ideal_points(train_vt, fractie_col=fractie_col, profiler=prof)
    bill_positions = get_bill_positions(train_vt, positioner=pos, train_df=train)

    best_loss = float("inf")
    best_weights = {"w1": 0.5, "w2": 0.3, "w3": 0.1, "w4": 0.1, "temperature": 1.0}

    # Grid search over weight combinations
    w1_candidates = [0.4, 0.5, 0.6]
    w2_candidates = [0.2, 0.3, 0.4]
    w3_candidates = [0.05, 0.1, 0.15]
    w4_candidates = [0.05, 0.1, 0.15]
    temp_candidates = [0.5, 1.0, 1.5, 2.0]

    for w1 in w1_candidates:
        for w2 in w2_candidates:
            for w3 in w3_candidates:
                for w4 in w4_candidates:
                    total = w1 + w2 + w3 + w4
                    if total < 0.1:
                        continue
                    # Normalize so weights sum to 1
                    w1n, w2n, w3n, w4n = w1 / total, w2 / total, w3 / total, w4 / total
                    payoff_voor, payoff_tegen = compute_payoffs_batch(
                        train_vt, party_ideals, bill_positions,
                        w1=w1n, w2=w2n, w3=w3n, w4=w4n,
                        fractie_col=fractie_col,
                    )
                    for temp in temp_candidates:
                        proba = payoff_to_probability(payoff_voor, payoff_tegen, temp)
                        loss = _binary_cross_entropy(y_true, proba)
                        if loss < best_loss:
                            best_loss = loss
                            best_weights = {"w1": w1n, "w2": w2n, "w3": w3n, "w4": w4n, "temperature": temp}

    return best_weights


def load_and_calibrate(
    train_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
) -> dict:
    """
    Load train/val from parquet and calibrate.
    """
    train_path = train_path or ANALYSIS_DIR / "train.parquet"
    val_path = val_path or ANALYSIS_DIR / "val.parquet"
    if not train_path.exists():
        return {"w1": 0.5, "w2": 0.3, "w3": 0.1, "w4": 0.1, "temperature": 1.0}
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path) if val_path.exists() else None
    return calibrate_weights(train, val)
