"""Estimate pairs if we match speech topic to Agendapunt.Onderwerp instead of BesluitTekst."""
import pandas as pd
import re

ap = pd.read_parquet("data/processed/Agendapunt.parquet")
besluit = pd.read_parquet("data/processed/Besluit.parquet")
stemming = pd.read_parquet("data/processed/Stemming.parquet")
act = pd.read_parquet("data/processed/Activiteit.parquet")
speeches = pd.read_parquet("data/analysis/speeches.parquet")

# Dutch stopwords
STOP = {"de", "het", "een", "van", "en", "in", "op", "te", "voor", "met",
        "dat", "die", "dit", "is", "zijn", "worden", "om", "aan", "bij",
        "als", "naar", "over", "uit", "tot", "door"}

def tokenize(text):
    words = set(re.findall(r"\b[a-z]{3,}\b", str(text).lower()))
    return words - STOP

# Build: for each voted besluit, get the Agendapunt.Onderwerp
voted_bids = set(stemming["Besluit_Id"].dropna().unique())
voted_b = besluit[besluit["Id"].isin(voted_bids)].copy()
voted_b = voted_b.merge(
    ap[["Id", "Onderwerp", "Activiteit_Id"]],
    left_on="Agendapunt_Id", right_on="Id", suffixes=("", "_ap")
)
voted_b["ap_onderwerp"] = voted_b["Onderwerp"]

# Get Stemmingen activiteiten -> dates
stem_acts = act[act["Soort"] == "Stemmingen"].copy()
stem_acts["stem_date"] = pd.to_datetime(stem_acts["Datum"], errors="coerce", utc=True).dt.date

# Map date -> list of (besluit_id, ap_onderwerp_tokens)
date_to_votes = {}
for _, row in voted_b.iterrows():
    bid = row["Id"]
    # Find the Stemmingen activity for this besluit
    act_id = row["Activiteit_Id"]
    stem_row = stem_acts[stem_acts["Id"] == act_id]
    if stem_row.empty:
        continue
    d = stem_row.iloc[0]["stem_date"]
    if pd.isna(d):
        continue
    tokens = tokenize(row["ap_onderwerp"])
    date_to_votes.setdefault(d, []).append((bid, tokens, str(row["ap_onderwerp"])[:80]))

print(f"Dates with votes: {len(date_to_votes)}")
print(f"Total (date, besluit) entries: {sum(len(v) for v in date_to_votes.values()):,}")

# Now: for substantive speeches on those dates, count matches
sub = speeches[
    (~speeches["is_voorzitter"])
    & (~speeches["is_interruptie"])
    & (speeches["fractie"].notna())
    & (speeches["speech_text_clean"].str.len() > 50)
].copy()
sub["verg_date"] = pd.to_datetime(sub["datum"], errors="coerce", utc=True).dt.date

match_count = 0
match_samples = []
no_match = 0
for i, (_, speech) in enumerate(sub.iterrows()):
    if i >= 50000:  # sample first 50K for speed
        break
    d = speech["verg_date"]
    if pd.isna(d) or d not in date_to_votes:
        continue
    speech_topic = " ".join(filter(None, [
        str(speech.get("activiteit_onderwerp") or ""),
        str(speech.get("activiteithoofd_onderwerp") or ""),
    ]))
    speech_tokens = tokenize(speech_topic)
    if not speech_tokens:
        continue
    matched_this = False
    for bid, vote_tokens, vote_topic in date_to_votes[d]:
        shared = speech_tokens & vote_tokens
        if len(shared) >= 1:
            match_count += 1
            matched_this = True
            if len(match_samples) < 5:
                match_samples.append((speech_topic[:60], vote_topic[:60], shared))
    if not matched_this:
        no_match += 1

print(f"\nFrom first 50K speeches:")
print(f"  Matched pairs: {match_count:,}")
print(f"  Speeches with no match: {no_match:,}")
print(f"\nSample matches:")
for st, vt, shared in match_samples:
    print(f"  SPEECH: {st}")
    print(f"  VOTE:   {vt}")
    print(f"  SHARED: {shared}")
    print()

# Estimate total
total_sub = len(sub)
ratio = match_count / min(50000, total_sub)
estimated_total = int(ratio * total_sub)
print(f"Estimated total pairs (Agendapunt matching): ~{estimated_total:,}")
