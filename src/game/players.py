#!/usr/bin/env python3
"""
Party ideal points and MP deviation profiles.

Builds multi-dimensional policy vectors for each party from:
- CHES expert survey data
- CMP manifesto data
- Historical voting (revealed preferences)
- Time-varying shifts from world events and polling
"""

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.game.external_data import (
    load_ches_positions,
    load_manifesto_positions,
    load_polling_data,
    load_world_events,
    CHES_DIMENSIONS,
    get_polling_for_date,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
ANALYSIS_DIR = ROOT / "data" / "analysis"

# Map fractie names (as in Stemming) to CHES party_abbrev
FRACTIE_TO_CHES: dict[str, str] = {
    "VVD": "VVD",
    "CDA": "CDA",
    "PVV": "PVV",
    "D66": "D66",
    "SP": "SP",
    "GroenLinks": "GroenLinks",
    "ChristenUnie": "ChristenUnie",
    "PvdA": "PvdA",
    "SGP": "SGP",
    "Partij voor de Dieren": "PvdD",
    "PvdD": "PvdD",
    "FVD": "FVD",
    "DENK": "DENK",
    "50Plus": "50Plus",
    "Volt": "Volt",
    "JA21": "JA21",
    "BBB": "BBB",
    "NSC": "NSC",
    "GroenLinks-PvdA": "GroenLinks-PvdA",
    "BIJ1": "BIJ1",
    "Fractie Den Haan": "Fractie Den Haan",
}


class PartyProfiler:
    """
    Builds and caches party ideal points over time.
    """

    def __init__(
        self,
        ches_path: Optional[Path] = None,
        manifesto_path: Optional[Path] = None,
        polling_path: Optional[Path] = None,
        events_path: Optional[Path] = None,
        voting_df: Optional[pd.DataFrame] = None,
    ):
        self.ches = load_ches_positions(ches_path)
        self.manifesto = load_manifesto_positions(manifesto_path)
        self.polling = load_polling_data(polling_path)
        self.events = load_world_events(events_path)
        self.voting_df = voting_df
        self._ideal_cache: dict[tuple[str, str], np.ndarray] = {}
        self._build_ches_lookup()

    def _build_ches_lookup(self) -> None:
        """Build party_abbrev -> year -> dict of dimension values."""
        self._ches_by_party: dict[str, dict[int, dict[str, float]]] = {}
        if self.ches.empty:
            return
        for _, row in self.ches.iterrows():
            abbrev = str(row.get("party_abbrev", "")).strip()
            yr = int(row.get("year", 2019))
            d = {}
            for dim in CHES_DIMENSIONS:
                if dim in row and pd.notna(row[dim]):
                    d[dim] = float(row[dim])
            if d:
                self._ches_by_party.setdefault(abbrev, {})[yr] = d

    def _interpolate_ches(self, party_abbrev: str, dt: date) -> dict[str, float]:
        """Get CHES position for party at date, interpolating between years."""
        key = FRACTIE_TO_CHES.get(party_abbrev, party_abbrev)
        if key not in self._ches_by_party:
            return {}
        by_year = self._ches_by_party[key]
        if not by_year:
            return {}
        years = sorted(by_year.keys())
        yr = dt.year if hasattr(dt, "year") else pd.to_datetime(dt).year
        if yr <= years[0]:
            return by_year[years[0]].copy()
        if yr >= years[-1]:
            return by_year[years[-1]].copy()
        # Find bracketing years
        y_before = max(y for y in years if y <= yr)
        y_after = min(y for y in years if y >= yr)
        if y_before == y_after:
            return by_year[y_before].copy()
        t = (yr - y_before) / (y_after - y_before)
        before = by_year[y_before]
        after = by_year[y_after]
        dims = set(before.keys()) & set(after.keys())
        return {d: (1 - t) * before[d] + t * after[d] for d in dims}

    def _apply_event_shifts(self, base: dict[str, float], dt: date) -> dict[str, float]:
        """Shift base positions by world events up to dt."""
        if self.events is None or (hasattr(self.events, "empty") and self.events.empty):
            return base.copy()
        out = base.copy()
        dt_pd = pd.to_datetime(dt)
        for _, ev in self.events.iterrows():
            ev_date = ev.get("date")
            if pd.isna(ev_date):
                continue
            if pd.to_datetime(ev_date) > dt_pd:
                continue
            dims_str = str(ev.get("affected_dimensions", ""))
            direction = str(ev.get("direction", ""))
            mag = float(ev.get("magnitude", 0.5))
            dims = [d.strip() for d in dims_str.split(";") if d.strip()]
            for d in dims:
                if d in out:
                    if "hawkish" in direction or "restrictive" in direction or "right" in direction:
                        out[d] = min(10, out[d] + mag * 0.5)
                    elif "left" in direction or "open" in direction or "pro" in direction:
                        out[d] = max(0, out[d] - mag * 0.5)
                    elif "energy" in direction or "environment" in direction:
                        out[d] = out[d] + mag * 0.3
        return out

    def _apply_polling_shift(self, base: dict[str, float], party: str, dt: date) -> dict[str, float]:
        """
        Shift towards median if party is trailing in polls (electoral pressure).
        """
        if self.polling is None or self.polling.empty:
            return base.copy()
        polls = get_polling_for_date(self.polling, dt)
        if not polls:
            return base.copy()
        party_share = polls.get(party, 0)
        if party_share == 0:
            return base.copy()
        total = sum(polls.values())
        if total == 0:
            return base.copy()
        share_pct = party_share / total
        # If share < 15%, slight shift towards center (5.0) on lrgen
        if share_pct < 0.15 and "lrgen" in base:
            shift = 0.1 * (5.0 - base["lrgen"])
            base = base.copy()
            base["lrgen"] = np.clip(base["lrgen"] + shift, 0, 10)
        return base

    def get_party_ideal_point(
        self,
        fractie: str,
        dt: date,
        apply_events: bool = True,
        apply_polling: bool = True,
    ) -> np.ndarray:
        """
        Get party ideal point as vector of dimension values.
        Returns array of shape (n_dims,) with values in [0, 10].
        """
        cache_key = (fractie, str(dt), apply_events, apply_polling)
        if cache_key in self._ideal_cache:
            return self._ideal_cache[cache_key]
        ches_key = FRACTIE_TO_CHES.get(fractie, fractie)
        base = self._interpolate_ches(ches_key, dt)
        if not base:
            # Fallback: use 5.0 for all dimensions (neutral)
            base = {d: 5.0 for d in CHES_DIMENSIONS}
        if apply_events:
            base = self._apply_event_shifts(base, dt)
        if apply_polling:
            base = self._apply_polling_shift(base, fractie, dt)
        vec = np.array([base.get(d, 5.0) for d in CHES_DIMENSIONS], dtype=np.float32)
        self._ideal_cache[cache_key] = vec
        return vec


def build_mp_deviation_profiles(
    train: pd.DataFrame,
    persoon_col: str = "persoon_id",
    fractie_col: str = "fractie",
    topic_col: str = "topic_cluster",
    vote_col: str = "vote",
) -> tuple[dict[str, float], dict[tuple[str, int], float]]:
    """
    Build MP-level deviation profiles from training data.

    Returns:
        mp_overall_loyalty: persoon_id -> mean loyalty (0-1, 1=always votes with party)
        mp_topic_loyalty: (persoon_id, topic_cluster) -> loyalty in that topic
    """
    train_sub = train[train[vote_col].isin(["Voor", "Tegen"])].copy()
    if train_sub.empty or topic_col not in train_sub.columns:
        return {}, {}

    party_topic_majority = (
        train_sub.groupby([fractie_col, topic_col])[vote_col]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Voor")
        .to_dict()
    )
    train_sub["_maj"] = train_sub.apply(
        lambda r: party_topic_majority.get((r[fractie_col], r[topic_col]), "Voor"),
        axis=1,
    )
    train_sub["_match"] = (train_sub[vote_col] == train_sub["_maj"]).astype(float)

    mp_overall = train_sub.groupby(persoon_col)["_match"].mean().to_dict()
    mp_topic = (
        train_sub.groupby([persoon_col, topic_col])["_match"]
        .mean()
        .to_dict()
    )
    return mp_overall, mp_topic


def get_mp_deviation(
    persoon_id: str,
    topic_cluster: int,
    mp_overall_loyalty: dict[str, float],
    mp_topic_loyalty: dict[tuple[str, int], float],
    n_topic_clusters: int = 20,
) -> float:
    """
    Get MP deviation probability for a given (persoon, topic).
    Deviation = 1 - loyalty; higher = more likely to rebel.
    """
    key = (str(persoon_id), int(topic_cluster))
    loyalty = mp_topic_loyalty.get(key)
    if loyalty is not None:
        return 1.0 - float(loyalty)
    overall = mp_overall_loyalty.get(str(persoon_id), 0.5)
    return 1.0 - float(overall)


def get_party_ideal_points(
    df: pd.DataFrame,
    fractie_col: str = "fractie",
    date_col: str = "datum",
    profiler: Optional[PartyProfiler] = None,
) -> np.ndarray:
    """
    Get party ideal points for each row in df.
    Returns array of shape (n_rows, n_dims).
    """
    prof = profiler or PartyProfiler()
    n_dims = len(CHES_DIMENSIONS)
    result = np.full((len(df), n_dims), 5.0, dtype=np.float32)
    for idx, (_, row) in enumerate(df.iterrows()):
        fractie = row.get(fractie_col) or "Onbekend"
        dt = row.get(date_col)
        if pd.isna(dt):
            continue
        vec = prof.get_party_ideal_point(fractie, dt)
        result[idx] = vec
    return result
