#!/usr/bin/env python3
"""
Bill positioning in policy space.

Maps each bill (besluit) to a multi-dimensional policy vector using:
- topic_cluster (from TF-IDF + KMeans on besluit + agendapunt text)
- kabinetsappreciatie (government position)
- zaak_soort (motion, amendment, etc.)
- Optional: sponsor party from ZaakActor (when available)
"""

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.game.external_data import CHES_DIMENSIONS
from src.game.players import PartyProfiler, FRACTIE_TO_CHES

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"

# Map topic_cluster (0-19) to primary policy dimension index (0-7)
# Heuristic: clusters 0-2=lrgen, 3-4=immigration, 5-6=environment, 7-8=eu, etc.
TOPIC_TO_DIM: dict[int, int] = {i: i % len(CHES_DIMENSIONS) for i in range(20)}

# Kabinetsappreciatie: government support level -> bill position shift
# "Overgenomen" = gov supports -> bill is center/coalition-aligned
# "Ontraden" = gov opposes -> bill is more oppositional
KA_TO_BIAS: dict[str, float] = {
    "Overgenomen": 0.2,   # Slight right/center bias
    "Ontraden": -0.2,     # Slight left/opposition bias
    "Onbekend": 0.0,
}


class BillPositioner:
    """
    Positions bills in the same policy space as party ideal points.
    """

    def __init__(
        self,
        party_profiler: Optional[PartyProfiler] = None,
        train_df: Optional[pd.DataFrame] = None,
    ):
        self.profiler = party_profiler or PartyProfiler()
        self._topic_centroids: dict[int, np.ndarray] = {}
        self._build_topic_centroids(train_df)

    def _build_topic_centroids(self, train_df: Optional[pd.DataFrame]) -> None:
        """
        Build typical bill position per topic_cluster from training data.
        Use mean party ideal point of parties that vote Voor in that topic.
        """
        if train_df is None or train_df.empty or "topic_cluster" not in train_df.columns:
            # Default: neutral (5.0) for all dimensions
            for tc in range(20):
                self._topic_centroids[tc] = np.full(len(CHES_DIMENSIONS), 5.0, dtype=np.float32)
            return
        train_vt = train_df[train_df["vote"].isin(["Voor", "Tegen"])].copy()
        for tc in range(20):
            sub = train_vt[train_vt["topic_cluster"] == tc]
            if sub.empty:
                self._topic_centroids[tc] = np.full(len(CHES_DIMENSIONS), 5.0, dtype=np.float32)
                continue
            # Parties that vote Voor more often in this topic -> bill is "closer" to them
            voor_rate = sub.groupby("fractie")["vote"].apply(lambda x: (x == "Voor").mean())
            parties = voor_rate[voor_rate > 0.5].index.tolist()
            if not parties:
                # Use overall mean
                dates = sub["datum"].dropna()
                if dates.empty:
                    self._topic_centroids[tc] = np.full(len(CHES_DIMENSIONS), 5.0, dtype=np.float32)
                    continue
                dt = pd.to_datetime(dates.iloc[0]).date()
                vecs = [self.profiler.get_party_ideal_point(p, dt) for p in sub["fractie"].unique()[:5]]
            else:
                dates = sub["datum"].dropna()
                dt = pd.to_datetime(dates.iloc[0]).date() if not dates.empty else date(2019, 1, 1)
                vecs = [self.profiler.get_party_ideal_point(p, dt) for p in parties[:5]]
            self._topic_centroids[tc] = np.mean(vecs, axis=0).astype(np.float32)

    def get_bill_position(
        self,
        topic_cluster: int,
        kabinetsappreciatie: str = "Onbekend",
        zaak_soort: str = "Onbekend",
        sponsor_party: Optional[str] = None,
        datum=None,
    ) -> np.ndarray:
        """
        Get bill position as vector in policy space.
        """
        tc = int(topic_cluster) if topic_cluster is not None else 0
        base = self._topic_centroids.get(tc, np.full(len(CHES_DIMENSIONS), 5.0, dtype=np.float32)).copy()
        # Apply kabinetsappreciatie bias on lrgen
        ka = str(kabinetsappreciatie or "Onbekend")[:50]
        bias = KA_TO_BIAS.get(ka, 0.0)
        base[0] = np.clip(base[0] + bias * 2, 0, 10)
        # If sponsor party known, blend with sponsor's ideal point
        if sponsor_party and self.profiler:
            import datetime
            dt = datum if datum else datetime.date(2019, 1, 1)
            if hasattr(dt, "date"):
                dt = dt.date()
            sponsor_vec = self.profiler.get_party_ideal_point(sponsor_party, dt)
            base = 0.7 * base + 0.3 * sponsor_vec
        return base.astype(np.float32)


def get_bill_positions(
    df: pd.DataFrame,
    topic_col: str = "topic_cluster",
    ka_col: str = "kabinetsappreciatie",
    zaak_col: str = "zaak_soort",
    date_col: str = "datum",
    positioner: Optional[BillPositioner] = None,
    train_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Get bill positions for each row in df.
    Returns array of shape (n_rows, n_dims).
    """
    pos = positioner or BillPositioner(train_df=train_df)
    n_dims = len(CHES_DIMENSIONS)
    result = np.full((len(df), n_dims), 5.0, dtype=np.float32)
    for i, row in df.iterrows():
        tc = row.get(topic_col, 0)
        ka = row.get(ka_col, "Onbekend")
        zs = row.get(zaak_col, "Onbekend")
        dt = row.get(date_col)
        vec = pos.get_bill_position(tc, ka, zs, None, dt)
        result[i] = vec
    return result
