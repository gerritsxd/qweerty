#!/usr/bin/env python3
"""
Payoff functions for the game-theoretic vote prediction model.

Five components:
1. Policy utility: alignment between party ideal point and bill position
2. Coalition utility: cost/benefit of voting with/against government
3. Electoral utility: polling-based incentives
4. Reciprocity utility: log-rolling from historical co-voting
5. Discipline utility: penalty for deviating from party line (DPBD-informed)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.game.external_data import CHES_DIMENSIONS, get_polling_for_date, load_polling_data

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
ANALYSIS_DIR = ROOT / "data" / "analysis"


def policy_utility(
    party_ideal: np.ndarray,
    bill_position: np.ndarray,
    vote_voor: bool,
) -> float:
    """
    Policy utility: negative distance in policy space.
    If vote_voor: utility = -distance (closer = better)
    If vote_tegen: utility = -distance to "opposite" of bill (further = better)
    """
    dist = np.linalg.norm(party_ideal - bill_position)
    max_dist = np.sqrt(len(party_ideal)) * 10  # max possible
    norm_dist = dist / max(max_dist, 1e-6)
    if vote_voor:
        return 1.0 - norm_dist  # [0, 1], higher when close
    else:
        return norm_dist  # higher when far (opposition)


def coalition_utility(
    fractie: str,
    vote_voor: bool,
    government_vote_voor: bool,
    is_coalition: bool,
    strength: float = 0.5,
) -> float:
    """
    Coalition parties: bonus for voting with government, penalty for breaking ranks.
    Opposition: bonus for differentiating from government.
    """
    if is_coalition:
        if vote_voor == government_vote_voor:
            return strength  # bonus for loyalty
        else:
            return -strength  # penalty for rebellion
    else:
        if vote_voor != government_vote_voor:
            return strength * 0.5  # opposition differentiates
        else:
            return -strength * 0.3  # opposition voting with gov is less rewarded


def electoral_utility(
    fractie: str,
    vote_voor: bool,
    government_vote_voor: bool,
    polling_share: float,
    total_poll_share: float,
) -> float:
    """
    Parties trailing in polls may vote more strategically to differentiate.
    Leading parties want stability.
    """
    if total_poll_share <= 0:
        return 0.0
    share_pct = polling_share / total_poll_share
    if share_pct < 0.10:
        # Small party: slight incentive to differentiate
        if vote_voor != government_vote_voor:
            return 0.1
        return -0.05
    if share_pct > 0.25:
        # Large party: prefer stability (vote with gov if coalition)
        return 0.0
    return 0.0


def reciprocity_utility(
    fractie: str,
    vote_voor: bool,
    other_party_voor_rate: float,
    strength: float = 0.2,
) -> float:
    """
    If we have historical co-voting with another party that supports this bill,
    slight bonus for voting same way.
    other_party_voor_rate: P(party X votes Voor) on similar bills - proxy for "allies"
    """
    if vote_voor and other_party_voor_rate > 0.7:
        return strength
    if not vote_voor and other_party_voor_rate < 0.3:
        return strength
    return 0.0


def discipline_utility(
    vote_voor: bool,
    party_line_voor: bool,
    dpbd_consistency: float,
    strength: float = 0.5,
) -> float:
    """
    Penalty for deviating from party line, scaled by historical discipline.
    High dpbd_consistency (e.g. 0.98) -> strong penalty for rebellion.
    Low dpbd_consistency (new/small parties) -> weaker penalty.
    party_line_voor: True if party typically votes Voor on this type of bill.
    """
    if dpbd_consistency <= 0 or dpbd_consistency >= 1:
        return 0.0
    if vote_voor == party_line_voor:
        return dpbd_consistency * strength
    return -dpbd_consistency * strength


def compute_payoffs(
    party_ideal: np.ndarray,
    bill_position: np.ndarray,
    fractie: str,
    vote_voor: bool,
    is_coalition: bool,
    government_vote_voor: bool,
    party_domain_voor_rate: float = 0.5,
    polling_share: float = 0.0,
    total_poll_share: float = 1.0,
    w1: float = 0.45,
    w2: float = 0.25,
    w3: float = 0.1,
    w4: float = 0.1,
    w5: float = 0.1,
    coalition_strength: float = 0.5,
    dpbd_consistency: float = 0.0,
    party_line_voor: Optional[bool] = None,
) -> float:
    """
    Total payoff for a party voting Voor (if vote_voor) or Tegen (otherwise).

    U = w1*policy + w2*coalition + w3*electoral + w4*reciprocity + w5*discipline
    """
    u1 = policy_utility(party_ideal, bill_position, vote_voor)
    u2 = coalition_utility(fractie, vote_voor, government_vote_voor, is_coalition, coalition_strength)
    u3 = electoral_utility(fractie, vote_voor, government_vote_voor, polling_share, total_poll_share)
    u4 = reciprocity_utility(fractie, vote_voor, party_domain_voor_rate, 0.2)
    u5 = 0.0
    if dpbd_consistency > 0 and party_line_voor is not None:
        u5 = discipline_utility(vote_voor, party_line_voor, dpbd_consistency, 0.5)
    return w1 * u1 + w2 * u2 + w3 * u3 + w4 * u4 + w5 * u5


def compute_payoffs_batch(
    df: pd.DataFrame,
    party_ideals: np.ndarray,
    bill_positions: np.ndarray,
    government_vote_voor: Optional[np.ndarray] = None,
    w1: float = 0.45,
    w2: float = 0.25,
    w3: float = 0.1,
    w4: float = 0.1,
    w5: float = 0.1,
    fractie_col: str = "fractie",
    is_coalition_col: str = "is_coalition",
    party_domain_col: str = "party_domain_voor_rate",
    datum_col: str = "datum",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute payoffs for Voor and Tegen for each row.
    Returns (payoff_voor, payoff_tegen) each shape (n_rows,).
    Uses dpbd_category_rate for reciprocity when available; dpbd_consistency for discipline.
    """
    n = len(df)
    payoff_voor = np.zeros(n, dtype=np.float32)
    payoff_tegen = np.zeros(n, dtype=np.float32)

    if government_vote_voor is None:
        gov_col = "kabinetsappreciatie" if "kabinetsappreciatie" in df.columns else None
        if gov_col:
            government_vote_voor = ~df[gov_col].fillna("").str.contains("Ontraden", case=False, na=False).values
        else:
            government_vote_voor = np.ones(n, dtype=bool)

    polling = load_polling_data()
    has_dpbd = "dpbd_consistency" in df.columns

    for i in range(n):
        row = df.iloc[i]
        fractie = row.get(fractie_col, "Onbekend")
        is_coal = bool(row.get(is_coalition_col, 0))
        p_domain = float(row.get(party_domain_col, 0.5))
        # Prefer dpbd_category_rate for reciprocity when available
        if has_dpbd:
            cat_rate = row.get("dpbd_category_rate")
            if cat_rate is not None and not pd.isna(cat_rate):
                p_domain = float(cat_rate)
            elif (vr := row.get("dpbd_voor_rate")) is not None and not pd.isna(vr):
                p_domain = float(vr)
        gov_voor = government_vote_voor[i] if hasattr(government_vote_voor, "__getitem__") else True

        poll_share = 0.0
        total_poll = 1.0
        if polling is not None and not polling.empty:
            dt = row.get(datum_col)
            p = get_polling_for_date(polling, dt)
            if p:
                poll_share = p.get(fractie, 0.0) or 0.0
                total_poll = sum(p.values()) or 1.0

        ideal = party_ideals[i] if i < len(party_ideals) else np.full(len(CHES_DIMENSIONS), 5.0)
        bill = bill_positions[i] if i < len(bill_positions) else np.full(len(CHES_DIMENSIONS), 5.0)

        # Discipline: party line from dpbd_category_rate > dpbd_voor_rate > party_domain
        dpbd_cons = 0.0
        party_line_voor = None
        if has_dpbd:
            cons = row.get("dpbd_consistency")
            if cons is not None and not pd.isna(cons):
                dpbd_cons = float(cons)
            for rate_col in ("dpbd_category_rate", "dpbd_voor_rate", party_domain_col):
                r = row.get(rate_col)
                if r is not None and not pd.isna(r):
                    party_line_voor = float(r) > 0.5
                    break
            else:
                party_line_voor = p_domain > 0.5

        payoff_voor[i] = compute_payoffs(
            ideal, bill, fractie, True, is_coal, gov_voor,
            p_domain, poll_share, total_poll, w1, w2, w3, w4, w5,
            dpbd_consistency=dpbd_cons, party_line_voor=party_line_voor,
        )
        payoff_tegen[i] = compute_payoffs(
            ideal, bill, fractie, False, is_coal, gov_voor,
            p_domain, poll_share, total_poll, w1, w2, w3, w4, w5,
            dpbd_consistency=dpbd_cons, party_line_voor=party_line_voor,
        )
    return payoff_voor, payoff_tegen
