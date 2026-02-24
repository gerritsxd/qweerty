"""Quick exploration of current seating data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

DATA = Path("data/processed")

fzp = pd.read_parquet(DATA / "FractieZetelPersoon.parquet")
fz = pd.read_parquet(DATA / "FractieZetel.parquet")
fractie = pd.read_parquet(DATA / "Fractie.parquet")
persoon = pd.read_parquet(DATA / "Persoon.parquet")

# Current MPs (no end date)
current = fzp[fzp["TotEnMet"].isna()].copy()
print(f"Active MPs: {len(current)}")

# Join to get party info
current = current.merge(fz[["Id", "Fractie_Id"]], left_on="FractieZetel_Id", right_on="Id", suffixes=("", "_zetel"))
current = current.merge(fractie[["Id", "Afkorting", "NaamNL", "AantalZetels"]], left_on="Fractie_Id", right_on="Id", suffixes=("", "_fractie"))
current = current.merge(persoon[["Id", "Achternaam", "Roepnaam", "Voornamen"]], left_on="Persoon_Id", right_on="Id", suffixes=("", "_persoon"))

# Party breakdown
party_counts = current.groupby("Afkorting").size().sort_values(ascending=False)
print("\nParty breakdown (current):")
for party, count in party_counts.items():
    print(f"  {party:25s} {count:3d} seats")
print(f"  {'TOTAL':25s} {party_counts.sum():3d}")

# Sample MPs
print("\nSample MPs:")
for _, r in current.sort_values("Afkorting").head(10).iterrows():
    name = f"{r['Roepnaam'] or r['Voornamen']} {r['Achternaam']}"
    print(f"  {r['Afkorting']:15s} {name}")
