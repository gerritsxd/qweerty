#!/usr/bin/env python3
"""
Equilibrium solver for the game-theoretic vote prediction model.

- Simple best-response: vote Voor if U(Voor) > U(Tegen)
- Converts payoff difference to P(Voor) via sigmoid
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.game.players import PartyProfiler, get_party_ideal_points
from src.game.bills import BillPositioner, get_bill_positions
from src.game.payoffs import compute_payoffs_batch

ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = ROOT / "data" / "analysis"


def payoff_to_probability(payoff_voor: np.ndarray, payoff_tegen: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert payoff difference to P(Voor) using sigmoid.
    P(Voor) = 1 / (1 + exp(-(U_voor - U_tegen) / temperature))
    """
    diff = payoff_voor - payoff_tegen
    return 1.0 / (1.0 + np.exp(-diff / max(temperature, 0.01)))


def run_simulation(
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    party_profiler: Optional[PartyProfiler] = None,
    bill_positioner: Optional[BillPositioner] = None,
    w1: float = 0.45,
    w2: float = 0.25,
    w3: float = 0.1,
    w4: float = 0.1,
    w5: float = 0.1,
    temperature: float = 1.0,
    fractie_col: str = "fractie",
) -> np.ndarray:
    """
    Run game-theoretic simulation on df.
    Returns P(Voor) for each row, shape (n_rows,).
    """
    prof = party_profiler or PartyProfiler()
    pos = bill_positioner or BillPositioner(party_profiler=prof, train_df=train_df)

    party_ideals = get_party_ideal_points(df, fractie_col=fractie_col, profiler=prof)
    bill_positions = get_bill_positions(df, positioner=pos, train_df=train_df)

    payoff_voor, payoff_tegen = compute_payoffs_batch(
        df, party_ideals, bill_positions,
        w1=w1, w2=w2, w3=w3, w4=w4, w5=w5,
        fractie_col=fractie_col,
    )
    return payoff_to_probability(payoff_voor, payoff_tegen, temperature)


def predict_vote_probabilities(
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    calibrated_weights: Optional[dict] = None,
) -> np.ndarray:
    """
    Main entry point: predict P(Voor) for each row using game-theoretic model.
    calibrated_weights: optional dict with w1, w2, w3, w4, w5, temperature
    """
    weights = calibrated_weights or {}
    return run_simulation(
        df,
        train_df=train_df,
        w1=weights.get("w1", 0.45),
        w2=weights.get("w2", 0.25),
        w3=weights.get("w3", 0.1),
        w4=weights.get("w4", 0.1),
        w5=weights.get("w5", 0.1),
        temperature=weights.get("temperature", 1.0),
    )
