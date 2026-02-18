"""
Tweede Kamer — Person Matcher
==============================
Bridges the gap between speech data (Vlos/XML persoon_ids) and canonical
OData API Persoon.Id by matching on (normalized_surname, party).

The XML Verslag system uses different IDs than the OData API (Parlis/Sesam),
so there is zero overlap on UUID-based IDs.  This module provides a
deterministic, name-based mapping that covers ~95% of speech rows.

Usage:
    from src.person_matcher import PersonMatcher
    matcher = PersonMatcher.from_parquet("data/processed")
    canonical_id = matcher.match("Nispen van", "SP")
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ─── Dutch tussenvoegsel list ───────────────────────────────────────────
TUSSENVOEGSELS = frozenset({
    "van", "de", "den", "der", "het", "ter", "ten",
    "op", "in", "tot", "te", "bij", "uit", "vd",
})

# ─── Party name normalisation map ───────────────────────────────────────
# Maps every known variant (from both speech data and Fractie table) to
# the canonical Fractie.Afkorting used in FractieZetelPersoon.
_PARTY_ALIASES: dict[str, str] = {
    # Speech data names → Fractie.Afkorting
    "groenlinks":           "GL",
    "groenlinks-pvda":      "GroenLinks-PvdA",
    "christenunie":         "ChristenUnie",
    "nsc":                  "Nieuw Sociaal Contract",
    "d66":                  "D66",
    "pvda":                 "PvdA",
    "pvdd":                 "PvdD",
    "vvd":                  "VVD",
    "cda":                  "CDA",
    "sp":                   "SP",
    "pvv":                  "PVV",
    "sgp":                  "SGP",
    "denk":                 "DENK",
    "fvd":                  "FVD",
    "bbb":                  "BBB",
    "50plus":               "50PLUS",
    "bij1":                 "BIJ1",
    "volt":                 "Volt",
    "ja21":                 "JA21",
    "groep van haga":       "Groep Van Haga",
    "van haga":             "Van Haga",
    "fractie den haan":     "Fractie Den Haan",
    "bontes":               "BONTES",
    "houwers":              "Houwers",
    "klein":                "Klein",
    "krol":                 "Krol",
    "monasch":              "Monasch",
    "van vliet":            "Van Vliet",
    "van kooten-arissen":   "vKA",
    "groep markuszower":    "Groep Markuszower",
    "50plus/baay-timmerman": "50PLUS/Baay-Timmerman",
    "50plus/klein":         "50PLUS/Klein",
    "omtzigt":              "Omtzigt",
    # Identity mappings (Fractie.Afkorting → itself)
    "gl":                   "GL",
    "cu":                   "ChristenUnie",
    "nieuw sociaal contract": "Nieuw Sociaal Contract",
}


def _norm_party(raw: str | None) -> str:
    """Normalise a party name to its canonical Fractie.Afkorting."""
    if not raw or not isinstance(raw, str):
        return ""
    key = raw.strip().lower()
    return _PARTY_ALIASES.get(key, raw.strip())


def _extract_surname(full_name: str | None) -> str:
    """
    Strip Dutch tussenvoegsels from speech-style names.

    Speech data stores names like 'Nispen van', 'Groot de', 'Lee van der'.
    The Persoon table stores 'Nispen', 'Groot', 'Lee' with a separate
    Tussenvoegsel column.  This function strips the trailing tussenvoegsels.
    """
    if not full_name or not isinstance(full_name, str):
        return ""
    parts = full_name.strip().split()
    core = [p for p in parts if p.lower() not in TUSSENVOEGSELS]
    return " ".join(core) if core else full_name.strip()


def _norm_name(s: str | None) -> str:
    """Lower-case, strip accents and non-alpha for surname comparison."""
    if not s or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = s.replace("ë", "e").replace("ï", "i").replace("ö", "o")
    s = s.replace("ü", "u").replace("é", "e").replace("è", "e")
    s = s.replace("ä", "a").replace("á", "a").replace("à", "a")
    return re.sub(r"[^a-z]", "", s)


class PersonMatcher:
    """
    Deterministic mapper: (speech_achternaam, speech_fractie) → Persoon.Id.
    """

    def __init__(self, lookup: dict[tuple[str, str], str],
                 name_only: dict[str, str]):
        self._lookup = lookup        # (norm_name, Afkorting) → Persoon.Id
        self._name_only = name_only  # norm_name → Persoon.Id  (unambiguous)

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_parquet(cls, data_dir: str | Path) -> "PersonMatcher":
        """Build matcher from processed parquet tables."""
        data_dir = Path(data_dir)

        persoon = pd.read_parquet(data_dir / "Persoon.parquet")
        fzp = pd.read_parquet(data_dir / "FractieZetelPersoon.parquet")
        fzs = pd.read_parquet(data_dir / "FractieZetel.parquet")
        fractie = pd.read_parquet(data_dir / "Fractie.parquet")

        # Persoon → FractieZetelPersoon → FractieZetel → Fractie
        fzp_full = (
            fzp.merge(
                fzs[["Id", "Fractie_Id"]],
                left_on="FractieZetel_Id", right_on="Id", suffixes=("", "_z"),
            ).merge(
                fractie[["Id", "Afkorting"]],
                left_on="Fractie_Id", right_on="Id", suffixes=("", "_f"),
            )
        )

        persoon_party = persoon[["Id", "Achternaam", "Tussenvoegsel"]].merge(
            fzp_full[["Persoon_Id", "Afkorting", "Van", "TotEnMet"]],
            left_on="Id", right_on="Persoon_Id", how="inner",
        )

        # (norm_name, Afkorting) → Persoon.Id
        # When duplicates exist (same name switched parties), prefer most recent
        persoon_party["_name"] = persoon_party["Achternaam"].apply(_norm_name)
        persoon_party = persoon_party.sort_values("Van", ascending=False, na_position="last")

        lookup: dict[tuple[str, str], str] = {}
        for _, row in persoon_party.iterrows():
            key = (row["_name"], row["Afkorting"])
            if key not in lookup:
                lookup[key] = row["Id"]

        # Also build name-only lookup for unambiguous surnames
        from collections import Counter
        name_counts = Counter(persoon_party["_name"])
        name_only: dict[str, str] = {}
        for _, row in persoon_party.iterrows():
            n = row["_name"]
            if name_counts[n] == 1 and n not in name_only:
                name_only[n] = row["Id"]

        return cls(lookup, name_only)

    # ── Matching ─────────────────────────────────────────────────────────

    def match(self, achternaam: str | None, fractie: str | None) -> str | None:
        """
        Match a speech speaker to their canonical Persoon.Id.

        Args:
            achternaam: Name as it appears in speech data (e.g. "Nispen van")
            fractie: Party as it appears in speech data (e.g. "SP")

        Returns:
            Persoon.Id (UUID string) or None if no match found.
        """
        name = _norm_name(_extract_surname(achternaam))
        if not name:
            return None

        party = _norm_party(fractie)

        # Strategy 1: exact (name, party) match
        pid = self._lookup.get((name, party))
        if pid:
            return pid

        # Strategy 2: try all party aliases in case of slight differences
        # e.g. speech has "FvD" (lowercase v) and Fractie.Afkorting is "FVD"
        for alias_key, canonical in _PARTY_ALIASES.items():
            if canonical == party:
                continue
            pid = self._lookup.get((name, canonical))
            if pid:
                return pid

        # Strategy 3: compound name — try hyphenated prefix
        # e.g. speech "Dik" → Persoon "Dik-Faber"
        if party:
            for (n, p), pid in self._lookup.items():
                if p == party and n.startswith(name) and len(n) > len(name):
                    return pid

        # Strategy 4: name-only (unambiguous)
        pid = self._name_only.get(name)
        if pid:
            return pid

        return None

    def match_df(self, df: pd.DataFrame,
                 name_col: str = "achternaam",
                 party_col: str = "fractie",
                 result_col: str = "canonical_persoon_id") -> pd.DataFrame:
        """Add a canonical_persoon_id column to a DataFrame."""
        df = df.copy()
        df[result_col] = df.apply(
            lambda r: self.match(r.get(name_col), r.get(party_col)),
            axis=1,
        )
        return df

    # ── Diagnostics ──────────────────────────────────────────────────────

    def coverage_report(self, df: pd.DataFrame,
                        name_col: str = "achternaam",
                        party_col: str = "fractie") -> None:
        """Print match statistics for a DataFrame."""
        matched = df.apply(
            lambda r: self.match(r.get(name_col), r.get(party_col)) is not None,
            axis=1,
        )
        total = len(df)
        n_matched = matched.sum()
        print(f"  Matched:   {n_matched:,}/{total:,} ({n_matched/max(total,1)*100:.1f}%)")

        unmatched = df[~matched][[name_col, party_col]].drop_duplicates()
        if len(unmatched) > 0:
            print(f"  Unmatched unique (name, party): {len(unmatched)}")
            counts = df[~matched].groupby([name_col, party_col]).size().sort_values(ascending=False)
            for (name, party), cnt in counts.head(15).items():
                print(f"    {name:25s} | {party:25s} ({cnt} rows)")
