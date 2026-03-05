#!/usr/bin/env python3
"""
Loaders for external game-theory data.

All reference data is embedded as Python constants so the module works
immediately after ``git pull`` with zero file dependencies. If CSV files
exist under ``data/external/``, they override the built-in defaults
(useful for updating positions without changing code).

- CHES: Chapel Hill Expert Survey party positions
- Peilingwijzer: Dutch polling data
- World events: Timeline of events that shift party positions
- Coalition agreements: Regeerakkoord key policy summaries
"""

from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_DIR = ROOT / "data" / "external"

CHES_DIMENSIONS = [
    "lrgen",            # Left-right general
    "galtan",           # Green-Alternative-Libertarian vs Traditional-Authoritarian-Nationalist
    "eu_position",      # EU integration (0=anti, 10=pro)
    "immigration",      # Immigration (0=open, 10=restrictive)
    "environment",      # Environment (0=economy first, 10=environment first)
    "redistribution",   # Redistribution (0=low, 10=high)
    "law_order",        # Law and order (0=liberal, 10=strict)
    "decentralization", # Decentralization (0=central, 10=decentral)
]

# ── Embedded reference data ──────────────────────────────────────────────────
# Sources: CHES 2019, Peilingwijzer, Dutch political history.
# Kept as plain dicts so there are no import-time file reads.

_CHES_ROWS = [
    # party_name, party_abbrev, year, lrgen, galtan, eu_position, immigration, environment, redistribution, law_order, decentralization
    ("VVD",               "VVD",              2019, 7.8, 5.2, 8.2, 6.5, 4.5, 3.2, 7.5, 6.8),
    ("CDA",               "CDA",              2019, 5.5, 6.2, 7.5, 5.0, 5.2, 4.8, 6.5, 5.5),
    ("PVV",               "PVV",              2019, 5.0, 8.5, 2.0, 9.5, 2.5, 5.5, 8.5, 6.0),
    ("D66",               "D66",              2019, 5.2, 3.5, 8.8, 4.5, 7.5, 5.0, 4.0, 6.5),
    ("SP",                "SP",               2019, 2.5, 3.8, 4.5, 5.0, 6.0, 9.0, 4.5, 5.0),
    ("GroenLinks",        "GroenLinks",       2019, 2.8, 2.5, 8.5, 5.5, 9.5, 8.5, 3.5, 6.0),
    ("ChristenUnie",      "ChristenUnie",     2019, 5.0, 6.0, 7.2, 4.5, 6.5, 5.5, 5.5, 5.2),
    ("PvdA",              "PvdA",             2019, 3.5, 3.2, 8.0, 5.0, 7.0, 8.0, 4.0, 5.5),
    ("SGP",               "SGP",              2019, 6.5, 9.0, 4.5, 6.0, 4.0, 4.5, 7.5, 7.0),
    ("Partij voor de Dieren", "PvdD",         2019, 2.2, 2.0, 7.5, 5.0, 9.8, 7.5, 3.0, 5.5),
    ("FVD",               "FVD",              2019, 6.0, 8.0, 1.5, 8.5, 3.0, 4.0, 8.0, 7.5),
    ("DENK",              "DENK",             2019, 4.0, 5.0, 6.5, 3.5, 5.5, 6.5, 4.5, 5.0),
    ("50Plus",            "50Plus",           2019, 5.5, 5.5, 6.0, 5.0, 4.5, 5.5, 5.5, 5.0),
    ("Volt",              "Volt",             2019, 5.0, 3.0, 9.5, 5.0, 7.5, 5.0, 4.0, 4.0),
    ("JA21",              "JA21",             2019, 7.0, 7.0, 4.0, 7.5, 3.5, 3.5, 7.0, 6.5),
    ("BBB",               "BBB",              2023, 5.5, 6.5, 5.0, 5.5, 4.5, 4.5, 5.5, 8.0),
    ("NSC",               "NSC",              2023, 5.8, 6.0, 6.5, 5.5, 5.0, 5.0, 6.0, 6.0),
    ("GroenLinks-PvdA",   "GroenLinks-PvdA",  2023, 3.2, 3.0, 8.2, 5.0, 7.5, 8.2, 4.0, 5.5),
]
_CHES_COLS = ["party_name", "party_abbrev", "year"] + CHES_DIMENSIONS

_POLLING_ROWS = [
    # date, VVD, PVV, CDA, D66, SP, GroenLinks, PvdA, ChristenUnie, SGP, PvdD, FVD, BBB, NSC, Volt, JA21, GroenLinks-PvdA
    ("2018-01-01", 21, 13, 13, 12, 9,  9,  6, 4, 3, 3, 2,  None, None, None, None, None),
    ("2019-01-01", 16, 16, 14, 12, 9,  10, 9, 5, 3, 4, 3,  None, None, None, None, None),
    ("2020-01-01", 22, 18, 12, 11, 8,  8,  7, 5, 2, 4, 2,  None, None, None, None, None),
    ("2021-01-01", 22, 17, 15, 15, 8,  8,  6, 5, 2, 4, 2,  None, None, None, None, None),
    ("2022-01-01", 24, 11, 14, 10, 7,  6,  5, 5, 2, 4, 5,  10,   None, None, None, None),
    ("2023-01-01", 25, 18, 5,  8,  5,  5,  4, 3, 2, 4, 3,  18,   4,    None, None, None),
    ("2024-01-01", 25, 24, 4,  7,  4,  4,  4, 3, 2, 4, 2,  7,    18,   None, None, 8),
    ("2025-01-01", 23, 26, 4,  7,  5,  4,  4, 3, 2, 4, 2,  6,    16,   None, None, 9),
]
_POLLING_COLS = [
    "date", "VVD", "PVV", "CDA", "D66", "SP", "GroenLinks", "PvdA",
    "ChristenUnie", "SGP", "PvdD", "FVD", "BBB", "NSC", "Volt", "JA21",
    "GroenLinks-PvdA",
]

_EVENT_ROWS = [
    # event_id, date, description, affected_dimensions, direction, magnitude
    ("mh17",             "2014-07-17", "MH17 plane crash Ukraine",              "eu_position;law_order",                    "hawkish",           0.3),
    ("refugee_crisis",   "2015-09-01", "European refugee crisis peak",          "immigration;eu_position",                  "restrictive",       0.5),
    ("brexit_ref",       "2016-06-23", "Brexit referendum",                     "eu_position",                              "eu_skeptic",        0.2),
    ("trump",            "2017-01-20", "Trump inauguration",                    "law_order;immigration",                    "populist_shift",    0.2),
    ("covid_start",      "2020-03-15", "COVID-19 lockdown Netherlands",         "redistribution;decentralization;law_order","state_expansion",   0.4),
    ("covid_peak",       "2021-01-01", "COVID peak winter",                     "redistribution;law_order",                 "state_expansion",   0.3),
    ("ukraine_invasion", "2022-02-24", "Russia invades Ukraine",                "eu_position;law_order;environment",        "hawkish;energy",    0.6),
    ("energy_crisis",    "2022-08-01", "Energy crisis Europe",                  "environment;redistribution;eu_position",   "energy_security",   0.4),
    ("housing_crisis",   "2022-01-01", "Housing shortage peak",                 "redistribution;decentralization",          "state_intervention",0.3),
    ("farmers_protests", "2022-06-01", "Farmers protests nitrogen",             "environment;decentralization",             "agriculture",       0.4),
    ("election_2023",    "2023-11-22", "Provincial elections BBB win",          "environment;decentralization",             "agriculture_shift", 0.5),
    ("schoof_cabinet",   "2024-07-02", "Schoof cabinet installed",              "immigration;law_order",                    "right_shift",       0.4),
]
_EVENT_COLS = ["event_id", "date", "description", "affected_dimensions", "direction", "magnitude"]

_COALITION_AGREEMENTS = {
    "rutte_ii": (
        "Bruggen slaan (2012)\n"
        "Coalition: VVD + PvdA\n"
        "Key themes: austerity, healthcare reform, EU cooperation, labor market\n"
        "Policy domains: economic (center-right), social (compromise), EU (pro)"
    ),
    "rutte_iii": (
        "Vertrouwen in de toekomst (2017)\n"
        "Coalition: VVD + CDA + D66 + ChristenUnie\n"
        "Key themes: climate agreement, tax reform, healthcare, migration\n"
        "Policy domains: economic (center), environment (ambitious), immigration (controlled)"
    ),
    "rutte_iv": (
        "Omzien naar elkaar, vooruitkijken naar de toekomst (2021)\n"
        "Coalition: VVD + CDA + D66 + ChristenUnie\n"
        "Key themes: climate, nitrogen, housing, healthcare\n"
        "Policy domains: environment (nitrogen), housing (construction), EU (pro)"
    ),
    "schoof": (
        "Hope, courage and pride (2024)\n"
        "Coalition: PVV + VVD + NSC + BBB\n"
        "Key themes: immigration, asylum, law and order, agriculture, decentralization\n"
        "Policy domains: immigration (restrictive), law_order (strict), environment (farmer-friendly), decentralization"
    ),
}

# ── Builders: embedded data → DataFrames ─────────────────────────────────────

def _build_ches_df() -> pd.DataFrame:
    return pd.DataFrame(_CHES_ROWS, columns=_CHES_COLS)

def _build_polling_df() -> pd.DataFrame:
    df = pd.DataFrame(_POLLING_ROWS, columns=_POLLING_COLS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def _build_events_df() -> pd.DataFrame:
    df = pd.DataFrame(_EVENT_ROWS, columns=_EVENT_COLS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ── Public loaders: CSV override → embedded fallback ─────────────────────────

def load_ches_positions(path: Optional[Path] = None) -> pd.DataFrame:
    """Load CHES party positions. Uses CSV if present, otherwise built-in data."""
    p = path or EXTERNAL_DIR / "ches_nl.csv"
    if p.exists():
        return pd.read_csv(p)
    return _build_ches_df()


def load_manifesto_positions(path: Optional[Path] = None) -> pd.DataFrame:
    """Load CMP manifesto positions. CSV-only (no built-in fallback)."""
    p = path or EXTERNAL_DIR / "manifesto_nl.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_polling_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load polling data. Uses CSV if present, otherwise built-in data."""
    p = path or EXTERNAL_DIR / "peilingwijzer.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return _build_polling_df()


def load_world_events(path: Optional[Path] = None) -> pd.DataFrame:
    """Load world event timeline. Uses CSV if present, otherwise built-in data."""
    p = path or EXTERNAL_DIR / "world_events.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return _build_events_df()


def load_coalition_agreements(dir_path: Optional[Path] = None) -> dict[str, str]:
    """Load coalition agreement texts. Uses files if present, otherwise built-in."""
    d = dir_path or EXTERNAL_DIR / "coalition_agreements"
    if d.exists() and any(d.glob("*.txt")):
        return {f.stem: f.read_text(encoding="utf-8") for f in d.glob("*.txt")}
    return dict(_COALITION_AGREEMENTS)


def get_polling_for_date(df: pd.DataFrame, date) -> dict[str, float]:
    """Get interpolated polling share per party for a given date."""
    if df is None or df.empty:
        return {}
    df = df.sort_values("date").reset_index(drop=True)
    date = pd.to_datetime(date, errors="coerce")
    if pd.isna(date):
        return {}
    party_cols = [c for c in df.columns if c != "date"]
    if not party_cols:
        return {}
    before = df[df["date"] <= date]
    after = df[df["date"] >= date]
    if before.empty and after.empty:
        return {}
    if before.empty:
        row = after.iloc[0]
    elif after.empty:
        row = before.iloc[-1]
    else:
        b = before.iloc[-1]
        a = after.iloc[0]
        span = (a["date"] - b["date"]).total_seconds()
        if span == 0:
            row = b
        else:
            t = (date - b["date"]).total_seconds() / span
            row = {
                c: (1 - t) * b[c] + t * a[c]
                if pd.notna(b[c]) and pd.notna(a[c])
                else (b[c] if pd.notna(b[c]) else a[c])
                for c in party_cols
            }
            row = pd.Series(row)
    return {c: float(row[c]) for c in party_cols if pd.notna(row.get(c))}


# ── Bootstrap: write CSVs from embedded data if they don't exist ─────────────

def ensure_external_data() -> None:
    """
    Write CSV files from embedded defaults if they don't exist yet.
    Call after ``git pull`` or during pipeline setup.
    Safe to call multiple times — never overwrites existing files.
    """
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    ches_path = EXTERNAL_DIR / "ches_nl.csv"
    if not ches_path.exists():
        _build_ches_df().to_csv(ches_path, index=False)

    polling_path = EXTERNAL_DIR / "peilingwijzer.csv"
    if not polling_path.exists():
        _build_polling_df().to_csv(polling_path, index=False)

    events_path = EXTERNAL_DIR / "world_events.csv"
    if not events_path.exists():
        _build_events_df().to_csv(events_path, index=False)

    ca_dir = EXTERNAL_DIR / "coalition_agreements"
    ca_dir.mkdir(parents=True, exist_ok=True)
    for name, text in _COALITION_AGREEMENTS.items():
        p = ca_dir / f"{name}.txt"
        if not p.exists():
            p.write_text(text, encoding="utf-8")
