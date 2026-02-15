"""Explore linking chain potential - read only analysis."""
import pandas as pd

ap = pd.read_parquet("data/processed/Agendapunt.parquet")
besluit = pd.read_parquet("data/processed/Besluit.parquet")
zaak = pd.read_parquet("data/processed/Zaak.parquet")
stemming = pd.read_parquet("data/processed/Stemming.parquet")
speeches = pd.read_parquet("data/analysis/speeches.parquet")

# How many besluiten have a Stemming? (voted decisions)
voted_besluit_ids = set(stemming["Besluit_Id"].dropna().unique())
print(f"Voted besluiten: {len(voted_besluit_ids):,}")

# Agendapunten linked to voted besluiten
voted_besluit = besluit[besluit["Id"].isin(voted_besluit_ids)]
print(f"Voted besluiten rows: {len(voted_besluit):,}")
ap_col = voted_besluit["Agendapunt_Id"].notna().sum()
print(f"  with Agendapunt_Id: {ap_col:,}")
voted_ap_ids = set(voted_besluit["Agendapunt_Id"].dropna().unique())
print(f"Agendapunten with voted besluiten: {len(voted_ap_ids):,}")

# Agendapunten -> Activiteit
voted_aps = ap[ap["Id"].isin(voted_ap_ids)]
voted_act_ids = set(voted_aps["Activiteit_Id"].dropna().unique())
print(f"Activiteiten with voted besluiten (via Agendapunt): {len(voted_act_ids):,}")

# How many speeches have activiteit_id matching?
speech_act_ids = set(speeches["activiteit_id"].dropna().unique())
overlap = speech_act_ids & voted_act_ids
print(f"Speech activiteiten: {len(speech_act_ids):,}")
print(f"Overlap (speech act_ids in voted act_ids): {len(overlap):,}")

# If we could link via Agendapunt, how many substantive speeches?
sub = speeches[
    (~speeches["is_voorzitter"])
    & (~speeches["is_interruptie"])
    & (speeches["fractie"].notna())
    & (speeches["speech_text_clean"].str.len() > 50)
]
sub_in_voted = sub[sub["activiteit_id"].isin(voted_act_ids)]
print(f"Substantive speeches in voted activiteiten: {len(sub_in_voted):,}")

# Zaak-based analysis
print(f"\nAgendapunt columns: {list(ap.columns)}")
print(f"Besluit columns: {list(besluit.columns)}")

# Check if ZaakActor or similar links Zaak to Agendapunt
za = pd.read_parquet("data/processed/ZaakActor.parquet")
print(f"\nZaakActor: {len(za):,}, cols: {list(za.columns)}")

# Check speech zaak_ids
speech_zaak = speeches["zaak_ids"].dropna()
print(f"\nSpeeches with zaak_ids: {len(speech_zaak):,}")
# Explode zaak_ids
all_speech_zaak_ids = set()
for zids in speech_zaak:
    for zid in str(zids).split(";"):
        zid = zid.strip()
        if zid:
            all_speech_zaak_ids.add(zid)
print(f"Unique zaak_ids from speeches: {len(all_speech_zaak_ids):,}")

zaak_ids_set = set(zaak["Id"].dropna().astype(str))
valid_speech_zaak = all_speech_zaak_ids & zaak_ids_set
print(f"Valid zaak_ids (found in Zaak table): {len(valid_speech_zaak):,}")

# Try: speech zaak_id -> Zaak -> find Besluit for same Zaak
# Besluit links to Agendapunt, Agendapunt links to Activiteit
# But we need Zaak -> Besluit. Check if there's a direct link.
# Look for Zaak_Id in Besluit or Agendapunt
for col in besluit.columns:
    if "zaak" in col.lower():
        print(f"  Besluit has Zaak column: {col}")
for col in ap.columns:
    if "zaak" in col.lower():
        print(f"  Agendapunt has Zaak column: {col}")

# Check Kabinetsappreciatie distribution for voted Zaak
# First need to find which Zaken lead to votes
# Path: Zaak -> ? -> Besluit -> Stemming
# Without a direct Zaak_Id on Besluit, we need an intermediate table
print("\n--- Checking for Zaak-Besluit link ---")
print("Besluit columns:", list(besluit.columns))

# Alternative: date-only linking without topic filter - how many pairs?
print("\n--- Date-only linking estimate ---")
from datetime import date
act = pd.read_parquet("data/processed/Activiteit.parquet")
stem_acts = act[act["Soort"] == "Stemmingen"]
print(f"Stemmingen activiteiten: {len(stem_acts):,}")
stem_dates = pd.to_datetime(stem_acts["Datum"], errors="coerce", utc=True).dt.date.dropna().unique()
speech_dates = pd.to_datetime(sub["datum"], errors="coerce", utc=True).dt.date.dropna().unique()
common_dates = set(stem_dates) & set(speech_dates)
print(f"Dates with both speeches and votes: {len(common_dates):,}")
