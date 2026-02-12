# Besluit — Parliamentary Decisions

**Tweede Kamer der Staten-Generaal · Open Data**

## Overview

**Besluit** records represent decisions taken on agenda items in the Dutch Parliament. Each record links to an **Agendapunt** (agenda item) and describes what was decided — e.g. "Ingediend" (submitted), "Behandeld" (processed), "Aangenomen" (adopted), "Verworpen" (rejected).

- **Total records:** 645,649  
- **Source file:** `data/raw/Besluit.json`  
- **Source:** [opendata.tweedekamer.nl](https://opendata.tweedekamer.nl/)

---

## Columns

| Column | Description |
|--------|-------------|
| `Id` | Unique identifier (UUID) |
| `Agendapunt_Id` | Link to the agenda item this decision applies to |
| `StemmingsSoort` | Type of vote (e.g. "Met handopsteken", "Hoofdelijke stemming"), or null |
| `BesluitSoort` | Kind of decision (e.g. Ingediend, Behandelen, Stemmen - aangenomen) |
| `BesluitTekst` | Short text of the decision (e.g. "Aangenomen.", "Verworpen") |
| `Opmerking` | Optional remark |
| `Status` | Status (typically "Besluit") |
| `AgendapuntZaakBesluitVolgorde` | Order of this decision within the agenda item |
| `GewijzigdOp` | Last modified date |

---

## Common decision types (BesluitSoort)

| Type | Meaning |
|------|---------|
| Ingediend | Submitted |
| Behandelen [en afdoen] | To be treated [and disposed of] |
| Stemmen - aangenomen | Vote — adopted |
| Stemmen - verworpen | Vote — rejected |
| Stemmen - ingetrokken | Vote — withdrawn |
| Stemmen - aangehouden | Vote — adjourned |
| LVIS - rondgezonden en gepubliceerd | Distributed and published |
| Afvoeren van de stand der werkzaamheden | Removed from order of business |
| Agenderen - algemeen overleg | Scheduling — general consultation |
| [Vrij tekstveld / geen Parlisproces] | Free text / no Parlis process |

---

## Sample records (first 50)

| # | BesluitSoort | BesluitTekst | StemmingsSoort | Date |
|---|--------------|--------------|----------------|------|
| 1 | Ingediend | Ingediend. | — | 2019-10-21 |
| 2 | Behandelen [en afdoen] | Behandeld | — | 2010-02-22 |
| 3 | Afvoeren van de stand der werkzaamheden (plenair) | Afgevoerd van de stand der werkzaamheden. | — | 2013-03-01 |
| 4 | LVIS - rondgezonden en gepubliceerd | Rondgezonden en gepubliceerd. | — | 2022-03-09 |
| 5 | Behandelen [en afdoen] | Behandeld. | — | 2018-05-17 |
| 6 | **Stemmen - aangenomen** | **Aangenomen.** | Met handopsteken | 2015-12-03 |
| 7 | Ingediend | Ingediend | — | 2022-07-06 |
| 8 | LVIS - rondgezonden en gepubliceerd | Rondgezonden en gepubliceerd | — | 2012-11-22 |
| 9 | [Vrij tekstveld / geen Parlisproces] | Reeds behandeld in het schriftelijk overleg Eurogroep/Ecofinraad. | — | 2018-10-10 |
| 10 | Behandelen [en afdoen] | Behandeld. | — | 2025-03-31 |
| 11 | Stemmen - ingetrokken | Ingetrokken. | — | 2019-10-21 |
| 12 | **Stemmen - verworpen** | **Verworpen** | Met handopsteken | 2019-10-21 |
| 13 | Ingediend | Ingediend. | — | 2019-10-21 |
| 14 | [Vrij tekstveld / geen Parlisproces] | Er zal een algemeen overleg Klimaat worden gepland in januari 2018. | — | 2017-11-22 |
| 15 | LVIS - rondgezonden en gepubliceerd | Rondgezonden en gepubliceerd | — | 2012-12-28 |
| 16 | Behandelen [en afdoen] | Behandeld. | — | 2023-06-15 |
| 17 | Ingediend | Ingediend. | — | 2019-10-21 |
| 18 | Behandelen [en afdoen] | Behandeld | — | 2009-07-10 |
| 19 | Agenderen - algemeen overleg | Agenderen voor het algemeen overleg Bouwopgave op 19 juni 2019. | — | 2019-10-21 |
| 20 | Termijn - vervallen in verband met verstrijken termijn | Vervallen in verband met verstrijken termijn. | — | 2022-02-25 |
| 21 | Leveren inbreng | Inbreng geleverd | — | 2012-06-15 |
| 22 | **Stemmen - aangenomen** | **Aangenomen.** | Met handopsteken | 2019-10-21 |
| 23 | LVIS - rondgezonden en gepubliceerd | Rondgezonden en gepubliceerd. | — | 2013-09-04 |
| 24 | Behandelen [en afdoen] | Behandeld. | — | 2024-06-20 |
| 25 | Behandelen [en afdoen] | Behandeld. | — | 2011-05-18 |
| 26 | Behandelen [en afdoen] | Behandeld. | — | 2025-03-19 |
| 27 | Behandelen [en afdoen] | Behandeld. | — | 2020-09-04 |
| 28 | Agenderen - algemeen overleg | Agenderen voor het te zijner tijd te houden algemeen overleg Sportbeleid. | — | 2020-02-19 |
| 29 | Aanhouden - tot de volgende procedurevergadering | Aangehouden tot de volgende procedurevergadering. | — | 2016-04-13 |
| 30 | Termijn - vervallen in verband met verstrijken termijn | Vervallen in verband met verstrijken termijn. | — | 2019-10-21 |
| 31 | Verzoek - bewindspersoon verzoeken om ... | Bewindspersonen verzoeken hoe zij staan tegenover openbare technische briefings door ambtenaren | — | 2016-03-03 |
| 32 | Inbreng - verslag wetsvoorstel | Inbrengdatum voor het verslag vaststellen op 10 oktober 2013 om 10.00 uur. | — | 2013-10-02 |
| 33 | Termijn - vervallen in verband met verstrijken termijn | Vervallen in verband met verstrijken termijn. | — | 2019-10-21 |
| 34 | Agenderen - algemeen overleg | Agenderen voor algemeen overleg EU-Uitbreiding op 10 december 2014. | — | 2014-11-06 |
| 35 | Agenderen - algemeen overleg | Agenderen voor het te zijner tijd te voeren algemeen overleg Dierenwelzijn. | — | 2017-05-17 |
| 36 | LVIS - rondgezonden en gepubliceerd | Rondgezonden en gepubliceerd. | — | 2019-10-21 |
| 37 | LVIS - rondgezonden en gepubliceerd | Rondgezonden en gepubliceerd. | — | 2025-11-20 |
| 38 | Betrekken - desgewenst betrekken bij andere zaak (relateren) | Desgewenst betrekken bij de begrotingsbehandeling SZW en agenderen voor te zijner tijd te voeren algemeen overleg over integratiebeleid | — | 2012-12-05 |
| 39 | LVIS - rondgezonden en gepubliceerd | Rondgezonden en gepubliceerd. | — | 2021-01-13 |
| 40 | Ingediend | Ingediend | — | 2021-06-15 |
| 41 | Agenderen - algemeen overleg | Dossier ouderbetrokkenheid maken en onderhavige brief aan toevoegen. AO plannen na ontvangst van beleidsnotitie. | — | 2012-12-13 |
| 42 | **Stemmen - verworpen** | **Verworpen** | Met handopsteken | 2019-10-21 |
| 43 | [Vrij tekstveld / geen Parlisproces] | De commissie stemt niet in met het verzoek. | — | 2024-01-15 |
| 44 | Behandelen [en afdoen] | Behandeld | — | 2009-12-09 |
| 45 | Behandelen [en afdoen] | Behandeld. | — | 2025-03-05 |
| 46 | Ingediend | Ingediend | — | 2025-06-19 |
| 47–50 | *(more of the same variety)* | | | |

---

## Linking to other data

- **Agendapunt_Id** → links to `Agendapunt` (agenda items)
- **Besluit** → links to `Stemming` (individual votes) via `Besluit_Id` for decisions that were put to a vote
