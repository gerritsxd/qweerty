# Linking Speech Speakers to Canonical Person IDs

## The Problem

Our speech data comes from **Verslag XML** (transcripts), served by the **Vlos** system.
The vote data (Stemming, ActiviteitActor) comes from **Parlis/Sesam**.

These two systems use **completely different UUID spaces** — there is **0 overlap** between:
- `activiteit_id` / `persoon_id` from Verslag XML (Vlos)
- `Activiteit.Id` / `Persoon.Id` in the OData API (Parlis/Sesam)

## Solution: `PersonMatcher`

We built `src/person_matcher.py` — a deterministic name+party matcher that maps
speech speakers to their canonical `Persoon.Id` with **99.3% coverage** and
**99.5%+ verification** against the Stemming table.

### How it works

1. **Build lookup**: Join `Persoon` → `FractieZetelPersoon` → `FractieZetel` → `Fractie`
   to get every (surname, party) → `Persoon.Id` mapping.

2. **Normalise names**: Strip Dutch tussenvoegsels from speech-style names
   (e.g. "Nispen van" → "Nispen") and normalise case/accents.

3. **Normalise parties**: Map speech party names to Fractie.Afkorting
   (e.g. "GroenLinks" → "GL", "NSC" → "Nieuw Sociaal Contract").

4. **Match with fallbacks**:
   - Exact (name, party) match
   - Try all known party aliases
   - Compound name prefix match (e.g. "Dik" → "Dik-Faber")
   - Unambiguous name-only match (when surname is globally unique)

### Coverage

| Metric | Value |
|--------|-------|
| Speech rows matched | 152,460 / 153,557 (99.3%) |
| Faction-level vote verification | 341/341 (100%) |
| Individual-level vote verification | 198/199 (99.5%) |
| Unmatched unique (name, party) pairs | 16 |

### Usage

```python
from src.person_matcher import PersonMatcher

matcher = PersonMatcher.from_parquet("data/processed")
canonical_id = matcher.match("Nispen van", "SP")
# Returns: 'fdc445bd-f8f...' (Persoon.Id)
```

## Why Faction-Level Linking Works

99.2% of Tweede Kamer votes are faction-level ("Met handopsteken"), meaning every
party member votes the same way. The current speech→vote linker (`src/link_speech_vote.py`)
matches through `(date, topic, party)` — not individual Persoon_Id.

The `PersonMatcher` adds individual-level resolution on top, enabling:
- Hemicycle visualisations with per-MP prediction overlays
- Individual vote prediction analysis
- Speaker-level accuracy metrics

## Remaining Gaps (16 unmatched name/party pairs)

Mostly edge cases: "El Abassi" (Arabic prefix), "Wout van 't" (unusual tussenvoegsel),
"Gündogan" (Turkish characters), and a few historical members. These account for <1% of data.
