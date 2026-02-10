# Tweede Kamer Open Data — Dataset Documentation

Comprehensive guide to the parliamentary data from the Dutch House of Representatives (Tweede Kamer). Use this document to explore the dataset and understand its possibilities for analysis and research.

---

## 1. Overview

### What is this dataset?

The dataset contains **all** publicly available parliamentary data from the Dutch Tweede Kamer (and Eerste Kamer) via the official [Open Data Portaal](https://opendata.tweedekamer.nl/) OData API. It covers the full legislative process: people, parties, committees, documents, votes, activities, meetings, and more.

### Data sources & timeline

| Source system | Entities | Earliest data |
|--------------|----------|---------------|
| **Parlis** | Activiteit, Document, Zaak, Besluit, Stemming, Agendapunt, etc. | 1 September 2008 |
| **Sesam** | Persoon, Fractie, Commissie, seat assignments | 1 September 2012 |
| **Vlos** | Vergadering, Verslag, Toezegging | 25 June 2013 |

Older data may be incomplete; records are retained when relevant to activities after these dates.

### File structure

| Location | Format | Contents |
|----------|--------|----------|
| `data/raw/` | JSON | Raw API responses (one file per entity) |
| `data/processed/` | Parquet | Cleaned, normalized tables ready for analysis |

### Key concepts (Dutch terms)

| Term | Meaning |
|------|---------|
| **Fractie** | Parliamentary party/caucus |
| **Kamer** | Chamber (Tweede Kamer = lower house, Eerste Kamer = upper house) |
| **Vergaderjaar** | Parliamentary year (starts 3rd Tuesday of September) |
| **Zaak** | Case — a legislative item (motion, bill, etc.) |
| **Besluit** | Decision taken on an agenda item |
| **Stemming** | Individual vote (Voor/Tegen/Niet deelgenomen) |
| **Kamerstuk** | Parliamentary document |
| **Commissie** | Parliamentary committee |

---

## 2. Entity Reference

### 2.1 People (Actors)

#### **Persoon**
*Members of parliament and related persons (MPs, ministers, etc.)*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID (primary key) |
| `Nummer` | Internal ID |
| `Voornamen`, `Achternaam`, `Roepnaam` | Name fields |
| `Geslacht` | man / vrouw |
| `Functie` | Eerste Kamerlid / Tweede Kamerlid / Oud Kamerlid |
| `Fractielabel` | Current party (only for Eerste Kamerleden) |
| `Geboortedatum`, `Geboorteplaats`, `Geboorteland` | Birth info |
| `Woonplaats`, `Land` | Residence |

**Links to:** FractieZetelPersoon, ActiviteitActor, DocumentActor, Stemming, ZaakActor, PersoonContactinformatie, PersoonGeschenk, PersoonLoopbaan, PersoonNevenfunctie, PersoonOnderwijs, PersoonReis

---

#### **PersoonContactinformatie**
*Contact details (email, phone, etc.) for persons*

Links to `Persoon` via `Persoon_Id`.

---

#### **PersoonGeschenk**
*Gifts received by parliament members (transparency register)*

**Links to:** Persoon

---

#### **PersoonLoopbaan**
*Career history of persons*

**Links to:** Persoon

---

#### **PersoonNevenfunctie**
*Side functions / secondary jobs (bijbanen)*

**Links to:** Persoon

---

#### **PersoonNevenfunctieInkomsten**
*Income from side functions*

**Links to:** PersoonNevenfunctie

---

#### **PersoonOnderwijs**
*Education history*

**Links to:** Persoon

---

#### **PersoonReis**
*Travel records of parliament members (official trips)*

**Links to:** Persoon

---

### 2.2 Political parties (Fracties)

#### **Fractie**
*Parliamentary parties/caucuses*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID |
| `Naam` | Party name |
| `Afkorting` | Abbreviation (e.g. VVD, CDA, D66) |

**Links to:** FractieZetel, FractieZetelPersoon, FractieZetelVacature, Stemming, ActiviteitActor, DocumentActor, ZaakActor

---

#### **FractieZetel**
*Seats allocated to a party*

**Links to:** Fractie, FractieZetelPersoon, FractieZetelVacature

---

#### **FractieZetelPersoon**
*Person occupying a party seat (current/former MPs)*

| Key columns | Description |
|-------------|-------------|
| `FractieZetel_Id` | Links to seat |
| `Persoon_Id` | Links to person |
| `Van`, `TotEnMet` |Period (start/end dates) |

**Links to:** FractieZetel, Persoon

---

#### **FractieZetelVacature**
*Vacant seats (unfilled)*

**Links to:** FractieZetel, Fractie

---

### 2.3 Committees

#### **Commissie**
*Parliamentary committees (e.g. VWS, BZK, EZK)*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID |
| `Naam` | Committee name |
| `Afkorting` | Abbreviation |

**Links to:** CommissieContactinformatie, CommissieZetel, ActiviteitActor, DocumentActor, ZaakActor

---

#### **CommissieContactinformatie**
*Committee contact info*

**Links to:** Commissie

---

#### **CommissieZetel**
*Committee seats*

**Links to:** Commissie, CommissieZetelVastPersoon, CommissieZetelVastVacature, CommissieZetelVervangerPersoon, CommissieZetelVervangerVacature

---

#### **CommissieZetelVastPersoon** / **CommissieZetelVervangerPersoon**
*Permanent and substitute members*

**Links to:** CommissieZetel, Persoon

---

#### **CommissieZetelVastVacature** / **CommissieZetelVervangerVacature**
*Vacant committee seats*

**Links to:** CommissieZetel, Fractie

---

### 2.4 Activities & agenda

#### **Activiteit**
*Parliamentary activities (meetings, debates, hearings, etc.)*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID |
| `Soort` | Activity type (e.g. Plenair debat, Hoorzitting, Vragenuur) |
| `Nummer` | Internal number |
| `Onderwerp` | Subject |
| `Datum` | Date |
| `Aanvangstijd`, `Eindtijd` | Start/end time |
| `Status` | Gepland / Uitgevoerd / Geannuleerd / Verplaatst / Vervallen |
| `Vergaderjaar` | Parliamentary year |
| `Kamer` | Eerste Kamer / Tweede Kamer / Beide |
| `Locatie` | Location |
| `Besloten` | Boolean (closed session) |
| `Voortouwcommissie_Id` | Lead committee |

**Links to:** ActiviteitActor, Agendapunt, Document, Reservering, Zaak

---

#### **ActiviteitActor**
*Who participated in which activity (MPs, committees, parties)*

| Key columns | Description |
|-------------|-------------|
| `Activiteit_Id` | Links to activity |
| `Persoon_Id` | Links to person |
| `ActorFractie` | Party name |
| `ActorCommissie` | Committee name |

**Links to:** Activiteit, Persoon, Fractie, Commissie

---

#### **Agendapunt**
*Agenda items within activities*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID |
| `Activiteit_Id` | Parent activity |
| `Nummer` | Agenda number |
| `Onderwerp` | Subject |

**Links to:** Activiteit, Besluit, Document, Zaak

---

#### **Reservering**
*Room reservations for activities*

**Links to:** Activiteit, Zaal

---

### 2.5 Documents

#### **Document**
*Parliamentary documents (bills, motions, reports, etc.)*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID |
| `Soort` | Document type (100+ types: Motie, Amendement, Voorstel van wet, etc.) |
| `DocumentNummer` | Official document number |
| `Titel` | Title |
| `Onderwerp` | Subject |
| `Datum` | Date |
| `Vergaderjaar` | Parliamentary year |
| `Kamer` | 1 (Eerste) / 2 (Tweede) / 3 (Both) |
| `Volgnummer` | Sequence within dossier |

**Links to:** DocumentActor, DocumentVersie, Kamerstukdossier, Activiteit, Agendapunt, Zaak

---

#### **DocumentActor**
*Actors associated with documents (authors, initiators)*

**Links to:** Document, Persoon, Fractie, Commissie

---

#### **DocumentVersie**
*Document versions (revisions)*

**Links to:** Document, DocumentPublicatie, DocumentPublicatieMetadata

---

#### **DocumentPublicatie** / **DocumentPublicatieMetadata**
*Publication metadata*

**Links to:** DocumentVersie

---

#### **Kamerstukdossier**
*Parliamentary dossiers (collections of related documents)*

**Links to:** Document, Zaak

---

### 2.6 Decisions & voting

#### **Besluit**
*Decisions taken on agenda items*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID |
| `Agendapunt_Id` | Links to agenda item |
| `StemmingsSoort` | Hoofdelijk / Met handopsteken / Zonder stemming |
| `BesluitSoort` | Type of decision |
| `BesluitTekst` | Decision text |
| `Status` | Besluit / Voorstel / Concept |

**Links to:** Agendapunt, Zaak, Stemming

---

#### **Stemming**
*Individual votes on decisions*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID |
| `Besluit_Id` | Links to decision |
| `Persoon_Id` | Links to person |
| `Fractie_Id` | Links to party |
| `Soort` | **Voor** / **Tegen** / **Niet deelgenomen** |
| `ActorNaam` | Name of voter |
| `ActorFractie` | Party of voter |
| `FractieGrootte` | Party size |
| `Vergissing` | Whether vote was a mistake |

**Links to:** Besluit, Persoon, Fractie

---

### 2.7 Cases (Zaken)

#### **Zaak**
*Parliamentary cases (motions, bills, interpellations, etc.)*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID |
| `Soort` | Case type (Motie, Initiatiefwetgeving, Schriftelijke vragen, etc.) |
| `Titel` | Title |
| `Status` | Vrijgegeven |
| `Onderwerp` | Subject |
| `GestartOp` | Start date |
| `Vergaderjaar` | Parliamentary year |
| `Kabinetsappreciatie` | Government position (Ontraden, Overgenomen, etc.) |
| `Afgedaan` | Whether case is closed |

**Links to:** ZaakActor, Activiteit, Agendapunt, Besluit, Document, Kamerstukdossier

---

#### **ZaakActor**
*Actors involved in cases (initiators, sponsors)*

**Links to:** Zaak, Persoon, Fractie, Commissie

---

### 2.8 Meetings & reports

#### **Vergadering**
*Completed meetings with available reports*

| Key columns | Description |
|-------------|-------------|
| `Id` | UUID |
| `Soort` | Commissie / Plenair |
| `Titel` | Title |
| `Datum` | Date |
| `Aanvangstijd`, `Sluiting` | Start/end |
| `Zaal` | Room |
| `Kamer` | Which chamber |

**Links to:** Verslag

---

#### **Verslag**
*Meeting reports / minutes (stenograms)*

**Links to:** Vergadering

---

#### **Toezegging**
*Promises / commitments made by ministers during debates*

**Links to:** Persoon

---

### 2.9 Facilities

#### **Zaal**
*Rooms/halls in parliament*

**Links to:** Reservering

---

## 3. Entity Relationships (Summary)

```
                    ┌─────────────┐
                    │   Persoon   │
                    └──────┬──────┘
           ┌──────────────┼──────────────┐
           │              │              │
    FractieZetelPersoon   Stemming   ActiviteitActor
           │              │              │
           ▼              ▼              ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Fractie   │ │   Besluit   │ │  Activiteit │
    └─────────────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           │               │               ├──────────┬──────────┐
           │               │               ▼          ▼          ▼
           │               │        ┌────────────┐ Agendapunt  Reservering
           │               │        │  Document  │     │         │
           │               │        └──────┬─────┘     │         │
           │               │               │           │         ▼
           │               │               │           │    ┌─────────┐
           │               │               │           │    │  Zaal   │
           │               │               │           │    └─────────┘
           │               │               │           │
           │               │               ├───────────┼──────────────┐
           │               │               ▼           ▼              │
           │               │        ┌────────────┐  Zaak               │
           │               └───────►│   Zaak     │◄───────────────────┘
           │                        └────────────┘
           │
           └──────────────► CommissieZetel, DocumentActor, ZaakActor
```

---

## 4. Exploration Possibilities

### Political analysis

1. **Voting behavior** — Join `Stemming` with `Persoon` and `Fractie` to analyze party cohesion, rebel votes, cross-party voting.
2. **Party discipline** — Compare `Voor`/`Tegen`/`Niet deelgenomen` rates per party over time.
3. **Cabinet support** — Link `Zaak` (Kabinetsappreciatie) with `Stemming` to see how parties vote on government positions.

### Legislative process

4. **Document flow** — Track documents from `Kamerstukdossier` → `Document` → `Zaak` through the legislative pipeline.
5. **Committee workload** — Count `Activiteit` per `Commissie` over time.
6. **Initiatiefwetgeving** — Filter `Zaak` by `Soort = Initiatiefwetgeving` and join with `ZaakActor` for initiators.

### People & transparency

7. **MP profiles** — Combine `Persoon` with `PersoonLoopbaan`, `PersoonOnderwijs`, `PersoonNevenfunctie`, `PersoonNevenfunctieInkomsten`.
8. **Gifts & travel** — `PersoonGeschenk`, `PersoonReis` for transparency/ethics analysis.
9. **Tenure** — Use `FractieZetelPersoon.Van` / `TotEnMet` for tenure and turnover analysis.

### Meetings & participation

10. **Attendance** — `ActiviteitActor` shows who attended which activities.
11. **Debate participation** — Filter `Activiteit` by `Soort` (e.g. Plenair debat) and join with `ActiviteitActor`.
12. **Meeting minutes** — `Vergadering` links to `Verslag` for full transcript access.

### Network & text analysis

13. **Co-sponsorship** — Use `DocumentActor` or `ZaakActor` for co-sponsorship networks.
14. **Document similarity** — Use `Document.Titel`, `Document.Onderwerp` for topic modeling.
15. **Promise tracking** — `Toezegging` links ministers to promises made during debates.

### Time series

16. **Volume over time** — Count documents, votes, activities per `Vergaderjaar`.
17. **Trends** — Compare `Activiteit.Soort` distribution across years.
18. **Seasonality** — `Vergadering.Datum` for meeting patterns.

---

## 5. Quick Start Queries

### Active MPs (Tweede Kamer) with party
```python
# Join Persoon with FractieZetelPersoon, FractieZetel, Fractie
# Filter: FractieZetelPersoon.TotEnMet == null
# Filter: Persoon.Functie == "Tweede Kamerlid"
```

### All votes on a specific decision
```python
# Stemming[Stemming.Besluit_Id == "..."]
# Join with Persoon, Fractie for names
```

### Documents per party (initiators)
```python
# DocumentActor → Document → Fractie
# Filter by Document.Soort (e.g. "Motie")
```

### Activities in a vergaderjaar
```python
# Activiteit[Activiteit.Vergaderjaar == "2023-2024"]
# Join with ActiviteitActor for participants
```

---

## 6. Official resources

| Resource | URL |
|----------|-----|
| Open Data Portal | https://opendata.tweedekamer.nl/ |
| OData API docs | https://opendata.tweedekamer.nl/documentatie/odata-api |
| Information model | https://opendata.tweedekamer.nl/documentatie/informatiemodel |
| API base URL | https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0 |
| Entity-specific docs | https://opendata.tweedekamer.nl/documentatie/{EntityName} |

---

## 7. Data quality notes

- **FractieAanvullendGegeven** — Returns 404 from API; not available in this dataset.
- **Verwijderd** — All records are pre-filtered to `Verwijderd eq false`; this column is dropped during preprocessing.
- **IDs** — All `Id` columns are GUIDs; `_Id` suffix indicates foreign keys to other entities.
- **Nulls** — All-null columns are dropped in preprocessing.
- **Dates** — Datetime columns are parsed to UTC in processed files.
