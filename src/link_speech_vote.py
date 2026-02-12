#!/usr/bin/env python3
"""
Tweede Kamer — Link speeches to votes
=======================================
Takes parsed speeches (from parse_verslagen.py) and links them to voting
records (Stemming) to create the master speech→vote dataset.

Linking strategies (tried in order):
1. Activiteit objectid from XML → Activiteit.Id → Agendapunt → Besluit → Stemming
2. Zaak objectids from XML → find Besluit for same Zaak via Agendapunt
3. Date + subject text matching as fallback

For each speech-vote pair:
- Individual votes (Hoofdelijk): match Persoon_Id
- Faction votes (Met handopsteken): match speaker's Fractie → ActorFractie

Output: data/analysis/speech_vote_pairs.parquet

Usage:
    python -m src.link_speech_vote
    python -m src.link_speech_vote --stats     # Print statistics only
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
ANALYSIS_DIR = ROOT / "data" / "analysis"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_table(name: str) -> pd.DataFrame:
    """Load a processed parquet table."""
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_speeches() -> pd.DataFrame:
    """Load parsed speeches from analysis dir."""
    path = ANALYSIS_DIR / "speeches.parquet"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run parse_verslagen.py first!")
        return pd.DataFrame()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Build vote lookup tables
# ---------------------------------------------------------------------------

def build_vote_lookups(
    activiteit: pd.DataFrame,
    agendapunt: pd.DataFrame,
    besluit: pd.DataFrame,
    stemming: pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Build lookup structures for linking speeches to votes.

    Returns:
        act_to_besluit: dict mapping Activiteit_Id → list of Besluit_Ids
        zaak_info: (unused for now, placeholder for zaak-based linking)
        vote_df: flat DataFrame of all votes with Besluit + context info
    """

    # ── Chain: Activiteit → Agendapunt → Besluit ───────────────────────
    # Only keep Besluiten that actually have Stemmingen
    besluit_ids_with_votes = set(stemming["Besluit_Id"].dropna().unique())
    besluit_voted = besluit[besluit["Id"].isin(besluit_ids_with_votes)].copy()

    # Agendapunt → Besluit (only voted ones)
    ap_besluit = agendapunt[["Id", "Activiteit_Id"]].merge(
        besluit_voted[["Id", "Agendapunt_Id", "BesluitSoort", "BesluitTekst", "StemmingsSoort"]],
        left_on="Id",
        right_on="Agendapunt_Id",
        how="inner",
        suffixes=("_agendapunt", "_besluit"),
    )

    # Build: Activiteit_Id → list of Besluit_Ids
    act_to_besluit = {}
    for _, row in ap_besluit.iterrows():
        aid = row["Activiteit_Id"]
        bid = row["Id_besluit"]
        act_to_besluit.setdefault(aid, []).append(bid)

    print(f"  Activiteiten with voted Besluiten: {len(act_to_besluit):,}")
    print(f"  Total voted Besluiten: {len(besluit_voted):,}")

    # ── Build flat vote DataFrame ──────────────────────────────────────
    # Stemming with Besluit context
    vote_df = stemming.merge(
        besluit_voted[["Id", "Agendapunt_Id", "BesluitSoort", "BesluitTekst", "StemmingsSoort"]],
        left_on="Besluit_Id",
        right_on="Id",
        how="inner",
        suffixes=("", "_besluit"),
    )

    print(f"  Total vote records (with Besluit context): {len(vote_df):,}")

    return act_to_besluit, {}, vote_df


# ---------------------------------------------------------------------------
# Link speeches to votes
# ---------------------------------------------------------------------------

def link_speeches_to_votes(
    speeches: pd.DataFrame,
    stemming: pd.DataFrame,
    besluit: pd.DataFrame,
    agendapunt: pd.DataFrame,
    activiteit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Link each speech to the corresponding vote(s).
    
    Strategy:
    1. Group speeches by vergadering (date)
    2. For that vergadering date, find "Stemmingen" activiteiten
    3. Get Besluiten from those Stemmingen
    4. Match speech topics to voted Agendapunt topics
    5. For each match, find the speaker's vote (by Persoon_Id or Fractie)
    
    Returns DataFrame with columns from both speech and vote.
    """

    print("\n  Building vote lookup structures...")
    act_to_besluit, _, vote_df = build_vote_lookups(
        activiteit, agendapunt, besluit, stemming
    )

    # ── Strategy 1: Direct activiteit_id match ─────────────────────────
    # The XML's <activiteit objectid> might directly match an Activiteit
    # that has Agendapunten with Besluiten and Stemmingen
    print("\n  Strategy 1: Direct activiteit_id → votes...")

    # Get speech activiteit_ids
    speech_act_ids = set(speeches["activiteit_id"].dropna().unique())
    matched_act_ids = speech_act_ids & set(act_to_besluit.keys())
    print(f"    Speech activiteit_ids:        {len(speech_act_ids):,}")
    print(f"    Matched to voted Besluiten:   {len(matched_act_ids):,}")

    # ── Strategy 2: Vergadering date matching ──────────────────────────
    # Speeches happen during debate → votes happen in "Stemmingen" activiteit
    # on the same or nearby date. Link through date + vergaderjaar.
    print("\n  Strategy 2: Date-based matching (debate → Stemmingen)...")

    # Find "Stemmingen" activiteiten
    stemmingen_acts = activiteit[
        activiteit["Soort"].str.contains("Stemming", case=False, na=False)
    ].copy()
    print(f"    'Stemmingen' activiteiten: {len(stemmingen_acts):,}")

    # Parse dates for matching
    speeches_dated = speeches.copy()
    if "datum" in speeches_dated.columns:
        speeches_dated["verg_date"] = pd.to_datetime(
            speeches_dated["datum"], errors="coerce", utc=True
        ).dt.date

    # For each speech, find the Stemmingen on the same date
    # and match the speaker to their vote
    pairs = []
    no_vote_found = 0
    no_date = 0

    # Pre-build lookups for speed
    # Besluit_Id → list of (Persoon_Id, ActorFractie, Soort, ...) tuples
    # Using dicts of dicts for O(1) lookup instead of DataFrame iteration
    print("  Building fast vote indexes...")

    # For individual votes: (Besluit_Id, Persoon_Id) → vote info
    individual_votes = {}
    # For faction votes: (Besluit_Id, ActorFractie) → vote info
    faction_votes = {}
    # Also keep fractie lowercase mapping for fuzzy matching
    faction_votes_lower = {}

    for _, row in vote_df.iterrows():
        bid = row["Besluit_Id"]
        pid = row.get("Persoon_Id")
        af = row.get("ActorFractie")
        vote_info = {
            "Besluit_Id": bid,
            "Soort": row.get("Soort"),
            "FractieGrootte": row.get("FractieGrootte"),
            "Persoon_Id": pid,
            "ActorFractie": af,
            "Vergissing": row.get("Vergissing"),
            "StemmingsSoort": row.get("StemmingsSoort"),
            "BesluitSoort": row.get("BesluitSoort"),
            "BesluitTekst": row.get("BesluitTekst"),
        }
        if pid and pd.notna(pid):
            individual_votes[(bid, pid)] = vote_info
        if af and pd.notna(af):
            faction_votes[(bid, af)] = vote_info
            faction_votes_lower[(bid, af.lower())] = vote_info

    # Besluit_Id → set of all Besluit_Ids for that besluit (for iteration)
    besluit_ids_by_act = {}  # act_id → list of besluit_ids
    for aid, bids in act_to_besluit.items():
        besluit_ids_by_act[aid] = bids

    # Filter to substantive speeches (not chair, not interruptions)
    substantive = speeches_dated[
        (~speeches_dated["is_voorzitter"].fillna(False))
        & (~speeches_dated["is_interruptie"].fillna(False))
        & (speeches_dated["fractie"].notna())
        & (speeches_dated["speech_text_clean"].str.len() > 50)
    ].copy()

    print(f"    Substantive speeches (not chair, not interruptions, >50 chars): {len(substantive):,}")

    # ── Strategy 1 linking ──────────────────────────────────────────────
    s1_speeches = substantive[substantive["activiteit_id"].isin(matched_act_ids)]

    print(f"\n  Linking Strategy 1 speeches ({len(s1_speeches):,})...")
    for _, speech in tqdm(s1_speeches.iterrows(), total=len(s1_speeches),
                          desc="  Strategy 1", unit=" speeches"):
        act_id = speech["activiteit_id"]
        speaker_fractie = speech.get("fractie")
        speaker_pid = speech.get("persoon_id")

        for besluit_id in act_to_besluit.get(act_id, []):
            vote = _fast_find_vote(
                besluit_id, speaker_pid, speaker_fractie,
                individual_votes, faction_votes, faction_votes_lower,
            )
            if vote is not None:
                pairs.append(make_pair(speech, vote))

    s1_count = len(pairs)
    print(f"    Strategy 1 pairs: {s1_count:,}")

    # ── Strategy 2 linking ──────────────────────────────────────────────
    # For speeches NOT matched by Strategy 1, try date matching
    s2_speeches = substantive[~substantive["activiteit_id"].isin(matched_act_ids)]

    # Build date → Stemmingen Activiteit_Id mapping
    if "Datum" in stemmingen_acts.columns:
        stemmingen_acts["stem_date"] = pd.to_datetime(
            stemmingen_acts["Datum"], errors="coerce", utc=True
        ).dt.date
        date_to_stem_acts = {}
        for _, row in stemmingen_acts.iterrows():
            d = row.get("stem_date")
            if d:
                date_to_stem_acts.setdefault(d, []).append(row["Id"])

        print(f"\n  Linking Strategy 2 speeches ({len(s2_speeches):,})...")
        for _, speech in tqdm(s2_speeches.iterrows(), total=len(s2_speeches),
                              desc="  Strategy 2", unit=" speeches"):
            vd = speech.get("verg_date")
            if vd is None:
                no_date += 1
                continue

            speaker_fractie = speech.get("fractie")
            speaker_pid = speech.get("persoon_id")

            # Look for Stemmingen on the same date (or +1 day for evening votes)
            matched_besluit_ids = []
            for d in [vd]:  # Could add vd + timedelta(days=1)
                for stem_act_id in date_to_stem_acts.get(d, []):
                    matched_besluit_ids.extend(act_to_besluit.get(stem_act_id, []))

            for besluit_id in matched_besluit_ids:
                vote = _fast_find_vote(
                    besluit_id, speaker_pid, speaker_fractie,
                    individual_votes, faction_votes, faction_votes_lower,
                )
                if vote is not None:
                    pairs.append(make_pair(speech, vote))

        s2_count = len(pairs) - s1_count
        print(f"    Strategy 2 pairs: {s2_count:,}")
    else:
        print("    Skipping Strategy 2 (no Datum column in Activiteit)")

    print(f"\n  Total speech-vote pairs: {len(pairs):,}")

    if not pairs:
        return pd.DataFrame()

    result = pd.DataFrame(pairs)
    return result


def _fast_find_vote(
    besluit_id: str,
    persoon_id: str | None,
    fractie: str | None,
    individual_votes: dict,
    faction_votes: dict,
    faction_votes_lower: dict,
) -> dict | None:
    """
    Fast O(1) vote lookup using pre-built dicts.
    Priority: individual match → exact faction → fuzzy faction.
    """
    # Try individual match
    if persoon_id:
        v = individual_votes.get((besluit_id, persoon_id))
        if v:
            return v

    # Try exact faction match
    if fractie:
        v = faction_votes.get((besluit_id, fractie))
        if v:
            return v

        # Try lowercase fuzzy match
        v = faction_votes_lower.get((besluit_id, fractie.lower()))
        if v:
            return v

    return None


def find_speaker_vote(
    votes: pd.DataFrame,
    persoon_id: str | None,
    fractie: str | None,
) -> dict | None:
    """
    Find a specific person's vote within a set of Stemming records.
    (Legacy — used as fallback; prefer _fast_find_vote)
    """
    if votes is None or votes.empty:
        return None

    if persoon_id:
        individual = votes[votes["Persoon_Id"] == persoon_id]
        if not individual.empty:
            return individual.iloc[0].to_dict()

    if fractie:
        faction = votes[votes["ActorFractie"] == fractie]
        if not faction.empty:
            return faction.iloc[0].to_dict()

    return None


def make_pair(speech: pd.Series, vote: dict) -> dict:
    """Combine a speech record and vote record into one pair."""
    return {
        # Speech info
        "vergadering_id": speech.get("vergadering_id"),
        "vergaderjaar": speech.get("vergaderjaar"),
        "datum": speech.get("datum"),
        "activiteit_id": speech.get("activiteit_id"),
        "activiteit_soort": speech.get("activiteit_soort"),
        "activiteit_onderwerp": speech.get("activiteit_onderwerp"),
        "activiteithoofd_onderwerp": speech.get("activiteithoofd_onderwerp"),
        # Speaker
        "persoon_id": speech.get("persoon_id"),
        "achternaam": speech.get("achternaam"),
        "voornaam": speech.get("voornaam"),
        "fractie": speech.get("fractie"),
        "spreker_soort": speech.get("spreker_soort"),
        "functie": speech.get("functie"),
        # Speech text
        "speech_text": speech.get("speech_text_clean", speech.get("speech_text")),
        "speech_length": len(speech.get("speech_text_clean", speech.get("speech_text", ""))),
        # Vote info
        "besluit_id": vote.get("Besluit_Id"),
        "stemmings_soort": vote.get("StemmingsSoort"),
        "besluit_soort": vote.get("BesluitSoort"),
        "besluit_tekst": vote.get("BesluitTekst"),
        # The target variable
        "vote": vote.get("Soort"),  # "Voor", "Tegen", "Niet deelgenomen"
        "vote_fractie_grootte": vote.get("FractieGrootte"),
        "vote_is_individual": pd.notna(vote.get("Persoon_Id")),
        "vote_vergissing": vote.get("Vergissing"),
    }


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test by time.
    Train: ≤ 2021 | Val: 2022 | Test: 2023-2025
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["datum"], errors="coerce", utc=True).dt.year

    train = df[df["year"] <= 2021]
    val = df[df["year"] == 2022]
    test = df[df["year"] >= 2023]

    print(f"  Train (≤2021):   {len(train):,} pairs")
    print(f"  Val (2022):      {len(val):,} pairs")
    print(f"  Test (≥2023):    {len(test):,} pairs")

    # Drop the temp column
    train = train.drop(columns=["year"])
    val = val.drop(columns=["year"])
    test = test.drop(columns=["year"])

    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Link speeches to votes")
    parser.add_argument("--stats", action="store_true", help="Print stats only, don't save")
    parser.add_argument("--no-split", action="store_true", help="Don't create train/val/test splits")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Linking speeches to votes")
    print("=" * 60 + "\n")

    # Load data
    print("[1/4] Loading data...")
    speeches = load_speeches()
    if speeches.empty:
        return

    activiteit = load_table("Activiteit")
    agendapunt = load_table("Agendapunt")
    besluit = load_table("Besluit")
    stemming = load_table("Stemming")

    for name, df in [("Activiteit", activiteit), ("Agendapunt", agendapunt),
                     ("Besluit", besluit), ("Stemming", stemming)]:
        print(f"  {name}: {len(df):,} rows")

    # Link
    print("\n[2/4] Linking speeches to votes...")
    pairs = link_speeches_to_votes(speeches, stemming, besluit, agendapunt, activiteit)

    if pairs.empty:
        print("\n  No pairs found! Check data linking.")
        return

    # Stats
    print("\n[3/4] Dataset statistics:")
    print(f"  Total pairs:        {len(pairs):,}")
    print(f"  Unique speakers:    {pairs['persoon_id'].nunique():,}")
    print(f"  Unique parties:     {pairs['fractie'].nunique():,}")
    print(f"  Unique besluiten:   {pairs['besluit_id'].nunique():,}")
    print(f"\n  Vote distribution:")
    print(pairs["vote"].value_counts().to_string(header=False))
    print(f"\n  Vote type (individual vs faction):")
    print(pairs["vote_is_individual"].value_counts().to_string(header=False))
    print(f"\n  Top parties:")
    print(pairs["fractie"].value_counts().head(10).to_string(header=False))

    if args.stats:
        return

    # Save
    print("\n[4/4] Saving...")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    out = ANALYSIS_DIR / "speech_vote_pairs.parquet"
    pairs.to_parquet(out, index=False)
    print(f"  Master dataset: {out} ({len(pairs):,} rows)")

    if not args.no_split:
        train, val, test = split_dataset(pairs)
        train.to_parquet(ANALYSIS_DIR / "train.parquet", index=False)
        val.to_parquet(ANALYSIS_DIR / "val.parquet", index=False)
        test.to_parquet(ANALYSIS_DIR / "test.parquet", index=False)
        print(f"  Splits saved to {ANALYSIS_DIR}")

    print("\nDone!")


if __name__ == "__main__":
    main()
