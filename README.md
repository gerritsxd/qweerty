# Tweede Kamer Open Data Pipeline

Data pipeline for fetching and preprocessing **all** parliamentary data from the [Dutch House of Representatives (Tweede Kamer)](https://opendata.tweedekamer.nl/) OData API.

## What's in the data?

35+ entity types covering the full Dutch parliamentary process:

| Category | Entities |
|---|---|
| **People** | Persoon, PersoonContactinformatie, PersoonGeschenk, PersoonLoopbaan, PersoonNevenfunctie, PersoonNevenfunctieInkomsten, PersoonOnderwijs, PersoonReis |
| **Parties (Fracties)** | Fractie, FractieAanvullendGegeven, FractieZetel, FractieZetelPersoon, FractieZetelVacature |
| **Committees** | Commissie, CommissieContactinformatie, CommissieZetel, CommissieZetelVast/VervangerPersoon/Vacature |
| **Activities** | Activiteit, ActiviteitActor, Agendapunt, Reservering |
| **Documents** | Document, DocumentActor, DocumentVersie, DocumentPublicatie, DocumentPublicatieMetadata, Kamerstukdossier |
| **Decisions & Votes** | Besluit, Stemming |
| **Cases** | Zaak, ZaakActor |
| **Meetings** | Vergadering, Verslag, Toezegging |
| **Rooms** | Zaal |

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> && cd <repo-name>

# 2. Create virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run the full pipeline (fetch + preprocess)
python pipeline.py
```

That's it. The pipeline will download all data from the API and produce clean Parquet files in `data/processed/`.

## Usage

```bash
# Full pipeline (fetch + preprocess all entities)
python pipeline.py

# Fetch raw data only (JSON)
python pipeline.py --fetch-only

# Preprocess only (if raw data already downloaded)
python pipeline.py --preprocess-only

# Process specific entities only
python pipeline.py --entities Persoon Fractie Stemming

# List all available entities
python pipeline.py --list-entities

# Generate summary of downloaded data
python pipeline.py --summary

# Verbose logging
python pipeline.py -v
```

## Project Structure

```
.
├── pipeline.py          # Main CLI entry point
├── config.yaml          # Pipeline configuration (entities, API settings, etc.)
├── requirements.txt     # Python dependencies
├── src/
│   ├── fetch.py         # OData API fetcher with pagination & retries
│   └── preprocess.py    # Data cleaning & normalization pipeline
├── data/
│   ├── raw/             # Raw JSON from API (git-ignored)
│   └── processed/       # Clean Parquet/CSV files (git-ignored)
└── README.md
```

## Configuration

Edit `config.yaml` to customize:

- **API settings**: rate limiting, retries, timeouts
- **Entities**: enable/disable specific entity types
- **Preprocessing**: date parsing, null column removal, output format (parquet/csv)

## How it works

1. **Fetch**: For each enabled entity, the pipeline queries the OData v4 API at `https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/{Entity}` with `$filter=Verwijderd eq false` to exclude deleted records. It handles pagination (250 records/page) automatically via `@odata.nextLink`.

2. **Preprocess**: Raw JSON is loaded into pandas DataFrames, then:
   - Redundant `Verwijderd` column is dropped
   - UUID IDs are standardized to lowercase
   - String columns are normalized (whitespace stripped)
   - Datetime columns are auto-detected and parsed
   - All-null columns are removed
   - Output is saved as Parquet (or CSV)

3. **Summary**: A `_summary.csv` is generated with record counts, column counts, and file sizes for every entity.

## For the team

The `data/` directory is **git-ignored** -- each team member runs the pipeline themselves to fetch fresh data. This keeps the repo lightweight and ensures everyone works with up-to-date data.

```bash
# After cloning, just run:
python pipeline.py
```

## API Reference

- Portal: https://opendata.tweedekamer.nl/
- OData API docs: https://opendata.tweedekamer.nl/documentatie/odata-api
- Information model: https://opendata.tweedekamer.nl/documentatie/informatiemodel
- Base URL: `https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0`
