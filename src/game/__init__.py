"""
Game-theoretic simulation for vote prediction.

Models the Tweede Kamer as a multi-player strategic game where parties
are rational agents with policy utility functions, coalition incentives,
and electoral pressures.

On first import, writes CSV files from embedded defaults if they don't
exist yet — so ``git pull`` on VPS is all you need.
"""

from src.game.external_data import (
    load_ches_positions,
    load_manifesto_positions,
    load_polling_data,
    load_world_events,
    load_coalition_agreements,
    ensure_external_data,
    CHES_DIMENSIONS,
)

ensure_external_data()
from src.game.players import PartyProfiler, get_party_ideal_points, build_mp_deviation_profiles
from src.game.bills import BillPositioner, get_bill_positions
from src.game.payoffs import compute_payoffs, compute_payoffs_batch
from src.game.simulation import run_simulation, predict_vote_probabilities
from src.game.calibrate import calibrate_weights, load_and_calibrate

__all__ = [
    "load_ches_positions",
    "load_manifesto_positions",
    "load_polling_data",
    "load_world_events",
    "load_coalition_agreements",
    "CHES_DIMENSIONS",
    "PartyProfiler",
    "get_party_ideal_points",
    "build_mp_deviation_profiles",
    "BillPositioner",
    "get_bill_positions",
    "compute_payoffs",
    "compute_payoffs_batch",
    "run_simulation",
    "predict_vote_probabilities",
    "calibrate_weights",
    "load_and_calibrate",
]
