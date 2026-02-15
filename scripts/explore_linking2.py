"""More linking analysis."""
import pandas as pd
from pathlib import Path

# Tables
all_tables = [f.stem for f in Path("data/processed").glob("*.parquet")]
zaak_related = [t for t in all_tables if "zaak" in t.lower() or "agendapunt" in t.lower()]
print("Zaak/Agendapunt related tables:", zaak_related)

# Agendapunt Onderwerp samples (voted ones)
ap = pd.read_parquet("data/processed/Agendapunt.parquet")
besluit = pd.read_parquet("data/processed/Besluit.parquet")
stemming = pd.read_parquet("data/processed/Stemming.parquet")
voted_besluit_ids = set(stemming["Besluit_Id"].dropna().unique())
voted_b = besluit[besluit["Id"].isin(voted_besluit_ids)]
voted_aps = ap[ap["Id"].isin(voted_b["Agendapunt_Id"].dropna().unique())]
print("\nVoted agendapunt onderwerp samples:")
for _, row in voted_aps.head(5).iterrows():
    print(f"  {str(row['Onderwerp'])[:120]}")

# Check speech zaak_ids format vs Zaak.Id format
speeches = pd.read_parquet("data/analysis/speeches.parquet")
zaak = pd.read_parquet("data/processed/Zaak.parquet")
print("\nSpeech zaak_ids format:", speeches["zaak_ids"].dropna().iloc[0][:80])
print("Zaak.Id format:", zaak["Id"].iloc[0])
print("Zaak.Nummer sample:", zaak["Nummer"].head(3).tolist())

# Volume estimate: date-only pairing
act = pd.read_parquet("data/processed/Activiteit.parquet")
sub = speeches[
    (~speeches["is_voorzitter"])
    & (~speeches["is_interruptie"])
    & (speeches["fractie"].notna())
    & (speeches["speech_text_clean"].str.len() > 50)
]
sub_dates = pd.to_datetime(sub["datum"], errors="coerce", utc=True).dt.date
stem_acts = act[act["Soort"] == "Stemmingen"]
stem_dates_set = set(pd.to_datetime(stem_acts["Datum"], errors="coerce", utc=True).dt.date.dropna())
speeches_on_vote_days = sub[sub_dates.isin(stem_dates_set)]
print(f"\nSubstantive speeches on days with votes: {len(speeches_on_vote_days):,}")
print(f"Unique speakers on vote days: {speeches_on_vote_days['persoon_id'].nunique()}")

# What is the activiteit_onderwerp overlap like?
pairs = pd.read_parquet("data/analysis/speech_vote_pairs.parquet")
print(f"\nCurrent pairs: {len(pairs):,}")
print(f"activiteit_onderwerp samples from pairs:")
for _, row in pairs.head(5).iterrows():
    print(f"  speech topic: {str(row.get('activiteit_onderwerp',''))[:80]}")
    print(f"  besluit_tekst: {str(row.get('besluit_tekst',''))[:80]}")
    print()

# How many unique (speaker, date) pairs exist on vote days?
speeches_on_vote_days = speeches_on_vote_days.copy()
speeches_on_vote_days["vote_date"] = sub_dates[speeches_on_vote_days.index]
unique_speaker_dates = speeches_on_vote_days.groupby(["persoon_id", "vote_date"]).size()
print(f"Unique (speaker, date) pairs on vote days: {len(unique_speaker_dates):,}")

# How many votes per day?
stemming_besluit = stemming.merge(voted_b[["Id", "Agendapunt_Id"]], left_on="Besluit_Id", right_on="Id", how="inner")
besluit_ap = stemming_besluit.merge(voted_aps[["Id", "Activiteit_Id"]], left_on="Agendapunt_Id", right_on="Id", how="inner")
act_dates = act[["Id", "Datum"]].rename(columns={"Id": "act_id"})
besluit_ap = besluit_ap.merge(act_dates, left_on="Activiteit_Id", right_on="act_id", how="left")
besluit_ap["vote_date"] = pd.to_datetime(besluit_ap["Datum"], errors="coerce", utc=True).dt.date
votes_per_day = besluit_ap.groupby("vote_date")["Besluit_Id"].nunique()
print(f"\nAvg voted besluiten per voting day: {votes_per_day.mean():.1f}")
print(f"Max voted besluiten per voting day: {votes_per_day.max()}")
