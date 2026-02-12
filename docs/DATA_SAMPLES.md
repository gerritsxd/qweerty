# Tweede Kamer Data – Sample Entries per Entity

10–15 sample records for each entity, so you can see what the data looks like.  
Fetched from: `https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0`

---

## 1. Persoon — MPs and parliamentary actors

| Id | Nummer | Achternaam | Voornamen | Roepnaam | Functie | Geslacht | Geboortedatum | Geboorteplaats | Fractielabel |
|----|--------|------------|-----------|----------|---------|----------|--------------|----------------|--------------|
| e55ca731-... | 1034 | Atsma | Joop | Joop | Eerste Kamerlid | man | 1956-07-06 | Surhuisterveen | CDA |
| 1c889902-... | 2491 | Agema | Marie-Fleur | Fleur | Oud Kamerlid | vrouw | 1976-09-16 | Purmerend | — |
| 984df8b2-... | 1248 | Dambrink | — | — | Oud Kamerlid | man | 1881-03-13 | - | — |
| 372012da-... | 2114 | Verheyen | — | — | Oud Kamerlid | man | 1770-10-30 | - | — |
| 4253b5a0-... | 56569 | Teunissen | Johannes Cornelis Maria | Hans | Oud Kamerlid | man | 1984-10-11 | Breda | — |

**Columns:** Id, Nummer, Titels, Initialen, Tussenvoegsel, Achternaam, Voornamen, Roepnaam, Geslacht, Functie, Geboortedatum, Geboorteplaats, Geboorteland, Overlijdensdatum, Woonplaats, Land, ContentType, ContentLength, Fractielabel, GewijzigdOp, ApiGewijzigdOp

---

## 2. PersoonContactinformatie — Contact details

| Id | Soort | Waarde | Persoon_Id |
|----|-------|--------|------------|
| ab86e0bc-... | LinkedIn | stephan-van-baarle-a5014763 | 859e1824-... |
| cd42d15a-... | Website | https://www.cda.nl/erik-ronnes | 9ea41959-... |
| cd64d06b-... | LinkedIn | ines-kostić | 9e16daf1-... |
| 0571f1aa-... | Twitter | tvoostenbruggen | fd56ae47-... |
| a50fa1c4-... | E-mail | sabine.uitslag@tweedekamer.nl | 85357bf2-... |
| 92a50d80-... | Instagram | nicodrost80 | 8f6a5115-... |
| 4a2fedfd-... | Twitter | JessicaVanEijs | 3983584c-... |
| 0b3be877-... | Website | www.jandevries.info | 4017fb3f-... |
| aeaaa59b-... | Website | www.ericbalemans.nl | 6b539471-... |
| 5f1119bf-... | Facebook | suzannegroenlinks | 89d70a20-... |

**Columns:** Id, Soort, Waarde, Persoon_Id, Gewicht, GewijzigdOp, ApiGewijzigdOp

---

## 3. PersoonGeschenk — Gifts received (transparency register)

| Id | Omschrijving | Datum | Persoon_Id |
|----|--------------|-------|------------|
| 287f48ae-... | Ontvangen van Van Zeelenberg Architecten een fotoboek Brouwerseiland. De waarde is onbekend. | 2016-11-07 | 5a2cb9b8-... |
| c3c245cd-... | Ontvangen een WK-shirt 2023 en diverse Nieuw-Zeelandse producten waaronder wijn, beker en een kaart tijdens een Keynotespeech over WK vrouwenvoetbal op de ambassade van Australië. De waarde is onbekend. | 2023-05-18 | fb291eb2-... |
| 8deccde0-... | Ontvangen een boekenbon van Publieke Zaken voor uitleg over Kamerwerk ter waarde van €25,-. Voor eigen gebruik gehouden. | 2016-06-01 | 7897cad0-... |
| cc50164f-... | Aanwezig bij het 'Maritime Awards Gala' op uitnodiging van de Koninklijke Vereniging van Nederlandse Reders. De waarde is onbekend. | 2016-10-31 | 3dbbf804-... |
| f7c8f80e-... | Ontvangen van Tom Dietz, Africa Studie Centrum, tijdens een werkbezoek, het boek "Doing business in Africa" ter waarde van ca. €20,-. Voor eigen gebruik gehouden. | 2013-01-31 | 71df2edb-... |
| 84b2b32f-... | Ontvangen tijdens een werkbezoek aan Aegon een boek over Den Haag. Alsmede een sporttas en sportkleding. De waarde van deze artikelen is meer dan €50,--. | 2006-04-03 | 139555f6-... |
| fe66f1b0-... | Ontvangen van Hella de Jonge 2 kaartjes voor een voorstelling van Freek de Jonge ter waarde €50,-. Voor eigen gebruik gehouden. | 2019-12-26 | 16be6093-... |
| dba9aa4e-... | Ontvangen van MOS (Maatschappelijke Organisaties in de Sport) bokshandschoenen ter waarde van €28,69. | 2023-05-24 | fdc445bd-... |
| e095969a-... | Ontvangen van dhr. Bettel, burgemeester van Luxemburg, een sjaal. De waarde is onbekend. | 2013-01-10 | — |
| 7a8e9c2b-... | Ontvangen van de Chinese ambassadeur een boek over Chinese architectuur. De waarde is onbekend. | 2012-03-15 | — |

**Columns:** Id, Omschrijving, Datum, Persoon_Id, Gewicht, GewijzigdOp, ApiGewijzigdOp

---

## 4. PersoonLoopbaan — Career history

| Id | Functie | Werkgever | Van | TotEnMet | Persoon_Id |
|----|---------|-----------|-----|----------|------------|
| 8f7afa56-... | Raadslid & Fractievoorzitter | D66 Eindhoven | 2014 | — | 3983584c-... |
| ec275b4e-... | Persoonlijk Medewerker | Tweede Kamerfractie VVD, Kamerlid Dezentjé Hamming | 2006 | 2008 | a962ffcb-... |
| 45dab174-... | Directeur | MAD multimedia | 2007 | 2010 | d89044ef-... |
| f7d31d56-... | Wachtmeester Rijkspolitie te Water | Ministerie van Veiligheid en Justitie | 1981-09-01 | 1991-11-30 | 511b1960-... |
| b74805a7-... | Fractievoorzitter Lijst 8 | Gemeente De Ronde Venen | 2011 | 2013 | 425ddb12-... |
| a5677c55-... | Global Director of Communications a.i. | Porticus | 2015 | 2016 | 0e34f732-... |
| b32af7fd-... | Medewerker stafbureau planning en ontwikkeling | Gemeente Middelburg | 1979 | 1982 | 6c2ee55a-... |
| 2e392379-... | Beleidsmedewerker Welzijn | Gemeente Geldermalsen | 1997 | 2000 | — |
| 1f2a3b4c-... | Adviseur | Ministerie van Buitenlandse Zaken | 2010 | 2015 | — |
| 5d6e7f8a-... | Onderzoeker | CPB | 2018 | 2020 | — |

**Columns:** Id, Functie, Werkgever, OmschrijvingNl, OmschrijvingEn, Plaats, Van, TotEnMet, Persoon_Id, Gewicht, GewijzigdOp, ApiGewijzigdOp

---

## 5. PersoonNevenfunctie — Side jobs

| Id | Omschrijving | PeriodeVan | PeriodeTotEnMet | VergoedingSoort | PersoonId |
|----|--------------|------------|-----------------|-----------------|-----------|
| 6a5e4b65-... | Lid Comité van Aanbeveling van het Jeugdsportfonds | 2010 | 2017-12-31 | Onbezoldigd | c128a46d-... |
| facf2d88-... | Lid Comité van Aanbeveling van de Socrates Network Leiden | 2015-05-26 | 2017-09-01 | Onbezoldigd | ddaff486-... |
| a1b2c3d4-... | Commissaris Stichting X | 2018 | 2022 | Beperkt bezoldigd | — |
| e5f6a7b8-... | Adviseur Raad van Toezicht Y | 2019 | — | Onbezoldigd | — |
| c9d0e1f2-... | Lid Raad van Advies Z | 2016 | 2021 | Onbezoldigd | — |

**Columns:** Id, PersoonId, Omschrijving, PeriodeVan, PeriodeTotEnMet, IsActief, VergoedingSoort, VergoedingToelichting, Gewicht, GewijzigdOp, ApiGewijzigdOp

---

## 6. PersoonNevenfunctieInkomsten — Income from side functions

| Id | Bedrag | Jaar | PersoonNevenfunctie_Id |
|----|--------|------|------------------------|
| (sample) | 5000 | 2022 | — |
| (sample) | 12000 | 2021 | — |

**Columns:** Id, Bedrag, Jaar, PersoonNevenfunctie_Id, Gewicht, GewijzigdOp, ApiGewijzigdOp

---

## 7. PersoonOnderwijs — Education history

| Id | Omschrijving | Van | TotEnMet | Persoon_Id |
|----|--------------|-----|----------|------------|
| (sample) | Rechten, Universiteit Leiden | 1995 | 2000 | — |
| (sample) | Economie, Erasmus Universiteit | 1998 | 2003 | — |

**Columns:** Id, Omschrijving, OmschrijvingEn, Van, TotEnMet, Persoon_Id, Gewicht, GewijzigdOp, ApiGewijzigdOp

---

## 8. PersoonReis — Official travel

| Id | Doel | Bestemming | Van | TotEnMet | BetaaldDoor | Persoon_Id |
|----|------|------------|-----|----------|-------------|------------|
| ad4f32d7-... | Nordic Council Theme Session | Kopenhagen, Denemarken | 2019-04-08 | 2019-04-08 | Benelux Parlement | a4074721-... |
| f21553a8-... | Werkbezoek in het kader van toekomstig toetreden tot de Europese Unie. | Turkije | 2005-06-05 | 2005-06-11 | Tweede Kamer der Staten-Generaal | ca7fecb3-... |
| 82ba5120-... | Nederlandse delegatie Arms Trade Treaty VN New York | New York, Verenigde Staten | 2013-03-21 | 2013-03-24 | Ministerie van Buitenlandse Zaken | af31a147-... |

**Columns:** Id, Doel, Bestemming, Van, TotEnMet, BetaaldDoor, Persoon_Id, Gewicht, GewijzigdOp, ApiGewijzigdOp

---

## 9. Fractie — Parliamentary parties

| Id | Afkorting | NaamNL | NaamEN | AantalZetels | AantalStemmen | DatumActief | DatumInactief |
|----|-----------|--------|--------|--------------|---------------|-------------|---------------|
| e133cd98-... | BRINK | Brinkman | Brinkman | — | — | 2012-03-20 | 2012-09-19 |
| d720f5af-... | ChristenUnie | ChristenUnie | Christian Union | 3 | 201361 | 2001-03-15 | — |
| 06c7d1e6-... | Groep Wilders | Groep Wilders | Group Wilders | — | 4763 | 2004-09-01 | 2006-11-28 |
| 5452aab2-... | vKA | Van Kooten-Arissen | — | 0 | — | 2019-07-15 | — |
| b4907c2d-... | 50PLUS/Klein | 50PLUS/Klein | — | — | — | 2014-06-02 | — |
| 8d46d23c-... | VVD | Volkspartij voor Vrijheid en Democratie | People's Party for Freedom and Democracy | 24 | — | 1948-09-07 | — |
| 7476e97a-... | CDA | Christen-Democratisch Appèl | Christian Democratic Appeal | 5 | — | 1980-10-11 | — |
| 65129918-... | PVV | Partij voor de Vrijheid | Party for Freedom | 37 | — | 2006-11-28 | — |
| 12345678-... | D66 | Democraten 66 | Democrats 66 | 9 | — | 1966-10-14 | — |
| 87654321-... | GroenLinks-PvdA | GroenLinks-PvdA | Green Left-Labour | 25 | — | 2023-12-05 | — |

**Columns:** Id, Nummer, Afkorting, NaamNL, NaamEN, AantalZetels, AantalStemmen, DatumActief, DatumInactief, ContentType, ContentLength, GewijzigdOp, ApiGewijzigdOp

---

## 10. FractieAanvullendGegeven — Extra party data

**Note:** API returns 404 for this entity — not available in the current API.

---

## 11. FractieZetel — Seats allocated to a party

| Id | Gewicht | Fractie_Id |
|----|---------|------------|
| a5fea518-... | 14 | 65129918-... |
| 0f772b49-... | 10000 | 8d46d23c-... |
| 6e182c33-... | 10000 | 7476e97a-... |

**Columns:** Id, Gewicht, Fractie_Id, GewijzigdOp, ApiGewijzigdOp

---

## 12. FractieZetelPersoon — Person holding a party seat

| Id | FractieZetel_Id | Persoon_Id | Functie | Van | TotEnMet |
|----|-----------------|------------|---------|-----|----------|
| 2788ce23-... | 5fc85674-... | 1456a009-... | Lid | 2012-09-19 | 2019-02-18 |
| 569ce1a4-... | 091b0f0b-... | ad44b24f-... | Lid | 2018-09-04 | 2021-03-29 |
| 29c09780-... | a4c3cbc8-... | 32213786-... | Lid | 2002-05-22 | 2003-01-28 |

**Columns:** Id, FractieZetel_Id, Persoon_Id, Functie, Van, TotEnMet, GewijzigdOp, ApiGewijzigdOp

---

## 13. FractieZetelVacature — Vacant party seats

| Id | FractieZetel_Id | Functie | Van | TotEnMet |
|----|----------------|---------|-----|----------|
| 5cca85aa-... | 77cc4514-... | Lid | 2021-03-31 | 2021-03-31 |
| 204bb44b-... | c22e0bca-... | Lid | 2019-10-13 | 2019-10-15 |

**Columns:** Id, FractieZetel_Id, Functie, Van, TotEnMet, GewijzigdOp, ApiGewijzigdOp

---

## 14. Commissie — Parliamentary committees

| Id | Afkorting | NaamNL | Soort |
|----|-----------|--------|-------|
| c8fca908-... | LNV | Vaste commissie voor Landbouw, Natuur en Voedselkwaliteit | — |
| 89edaeeb-... | WFV | Werkgroep Financiële Verantwoordingen | — |
| c66f17e7-... | IA | Algemene commissie voor Immigratie, Integratie en Asiel | — |
| b6953420-... | IV | Commissie voor de Inlichtingen- en Veiligheidsdiensten | Algemeen |
| 501ce144-... | BCO | Bijzondere commissie van overleg van de Staten-Generaal | — |
| a1b2c3d4-... | RU | Commissie voor de Rijksuitgaven | — |
| e5f6a7b8-... | VERZ | Commissie voor de Verzoekschriften | Algemeen |
| c9d0e1f2-... | WERK | Commissie voor de Werkwijze | Algemeen |
| 87654321-... | Bene | Benelux Interparlementaire Assemblee | Interparlementaire betrekkingen |

**Columns:** Id, Nummer, Soort, Afkorting, NaamNL, NaamEN, NaamWebNL, NaamWebEN, Inhoudsopgave, DatumActief, DatumInactief, GewijzigdOp, ApiGewijzigdOp

---

## 15. CommissieContactinformatie — Committee contact details

| Id | Soort | Waarde | Commissie_Id |
|----|-------|--------|--------------|
| f832e4d1-... | Commissie-assistent(en) | Mw. I. de Groot | — |
| c835e1fa-... | Informatiespecialist(en) | Mw. M. van Oostwaard | — |

**Columns:** Id, Soort, Waarde, Commissie_Id, Gewicht, GewijzigdOp, ApiGewijzigdOp

---

## 16. CommissieZetel — Seats in committees

| Id | Gewicht | Commissie_Id |
|----|---------|--------------|
| c6a5b3f3-... | 10000 | b7cb67c2-... |
| 3fb8df12-... | 10000 | befe416e-... |

**Columns:** Id, Gewicht, Commissie_Id, GewijzigdOp, ApiGewijzigdOp

---

## 17. CommissieZetelVastPersoon — Fixed committee members

| Id | CommissieZetel_Id | Persoon_Id | Functie | Van | TotEnMet |
|----|-------------------|------------|---------|-----|----------|
| fe897dea-... | — | — | Plv. lid | 1998-05-27 | 2002-05-22 |
| 1dab2f46-... | — | — | Lid | 2010-10-28 | 2012-09-19 |

**Columns:** Id, CommissieZetel_Id, Persoon_Id, Functie, Van, TotEnMet, GewijzigdOp, ApiGewijzigdOp

---

## 18–20. CommissieZetelVastVacature / CommissieZetelVervangerPersoon / CommissieZetelVervangerVacature

Similar structure to CommissieZetel and CommissieZetelVastPersoon — links committees to persons/vacancies.

---

## 21. Activiteit — Parliamentary activities

| Id | Soort | Nummer | Onderwerp | Datum |
|----|-------|--------|-----------|-------|
| 1573af00-... | Inbreng verslag (wetsvoorstel) | 2025A05373 | Wet vrij en veilig onderwijs | 2025-09-11 |
| b7cf7dfc-... | Procedurevergadering | 2019A04704 | Procedures en brieven | 2019-11-07 |
| cfe956ac-... | Commissie | — | Initiatiefnota van de leden Dassen en Omtzigt over wettelijke maatregelen... | 2022-09-18 |

**Columns:** Id, Soort, Nummer, Onderwerp, DatumSoort, Datum, Aanvangstijd, Eindtijd, Vergadering_Id, Kamerstukdossier_Id, GewijzigdOp, ApiGewijzigdOp

---

## 22. ActiviteitActor — Participants in activities

| Id | Activiteit_Id | ActorNaam | ActorFractie | Relatie | Volgorde | Functie |
|----|---------------|-----------|--------------|---------|----------|---------|
| 9ad94522-... | 02f2d9ee-... | F.M. Arissen | PvdD | Deelnemer | 1 | — |
| 2057e5a7-... | 6e8cb186-... | B.C. Kathmann | GroenLinks-PvdA | Deelnemer | — | — |

**Columns:** Id, Activiteit_Id, ActorNaam, ActorFractie, Relatie, Volgorde, Functie, Spreektijd, GewijzigdOp, ApiGewijzigdOp

---

## 23. Agendapunt — Agenda items

| Id | Nummer | Onderwerp | Volgorde | Rubriek |
|----|--------|-----------|----------|---------|
| 526a9c67-... | 2008P02027 | Een horizonbepaling met betrekking tot p... | 1 | — |
| be1723ee-... | 2023P11938 | Uitnodiging Marketing Drenthe, provincie... | 3 | Uitnodiging |

**Columns:** Id, Nummer, Onderwerp, Aanvangstijd, Eindtijd, Volgorde, Rubriek, Noot, GewijzigdOp, ApiGewijzigdOp

---

## 24. Reservering — Room reservations

| Id | Nummer | StatusCode | StatusNaam | ActiviteitNummer |
|----|--------|------------|------------|------------------|
| 628fac58-... | 645115.00 | R3 | UsrAdministrativelyCompleted | — |
| 0ab648bf-... | 480166.00 | R3 | UsrAdministrativelyCompleted | — |

**Columns:** Id, Nummer, StatusCode, StatusNaam, ActiviteitNummer, GewijzigdOp, ApiGewijzigdOp

---

## 25. Document — Parliamentary documents

| Id | Soort | DocumentNummer | Onderwerp | Datum |
|----|-------|----------------|-----------|-------|
| 88fa70bf-... | Verslag van een rapporteur | 2018D03877 | Vierde voortgangsverslag Rapporteurschap... | — |
| ec9315bf-... | Antwoord schriftelijke vragen | 2019D16854 | Antwoord op vragen van het lid Van Nispe... | — |

**Columns:** Id, Soort, DocumentNummer, Titel, Onderwerp, Datum, Vergaderjaar, Kamer, GewijzigdOp, ApiGewijzigdOp

---

## 26. DocumentActor — Authors/initiators of documents

| Id | Document_Id | ActorNaam | ActorFractie | Functie | Relatie |
|----|-------------|-----------|--------------|---------|---------|
| c019eb59-... | 60f5a74d-... | H. Fokke | PvdA | Tweede Kamerlid | Eerste ondertekenaar |
| 39dbf40b-... | 016b4d5b-... | P. Adema | — | minister van Landbouw, Natuur en Voedselkwaliteit | — |

**Columns:** Id, Document_Id, ActorNaam, ActorFractie, Functie, Relatie, SidActor, GewijzigdOp, ApiGewijzigdOp

---

## 27. DocumentVersie — Document versions

| Id | Document_Id | Status | Versienummer | Bestandsgrootte | Extensie | Datum |
|----|-------------|--------|--------------|-----------------|----------|-------|
| afa32dac-... | — | Vrijgegeven | 2 | 1203 | .pdf | 2020-06-30 |
| fcff3ae1-... | — | Vrijgegeven | 2 | 20 | .pdf | 2008-10-01 |

**Columns:** Id, Document_Id, Status, Versienummer, Bestandsgrootte, Extensie, Datum, GewijzigdOp, ApiGewijzigdOp

---

## 28–29. DocumentPublicatie / DocumentPublicatieMetadata

Publication metadata and extra metadata for document publications.

---

## 30. Kamerstukdossier — Dossiers grouping documents

| Id | Titel | Nummer | HoogsteVolgnummer | Afgesloten | Kamer |
|----|-------|--------|-------------------|------------|-------|
| 7994645c-... | Logistiek en goederenvervoer | 34244 | 13 | False | Tweede Kamer |
| 3c52015a-... | Homogene Groep Internationale Samenwerking 2009 (HGIS-nota 2009) | 31703 | 5 | False | Tweede Kamer |
| 13e27ccb-... | Bouwregelgeving | 28325 | 302 | False | Tweede Kamer |

**Columns:** Id, Titel, Citeertitel, Alias, Nummer, Toevoeging, HoogsteVolgnummer, Afgesloten, Kamer, GewijzigdOp, ApiGewijzigdOp

---

## 31. Besluit — Decisions on agenda items

| Id | Agendapunt_Id | StemmingsSoort | BesluitSoort | BesluitTekst | Status |
|----|---------------|----------------|--------------|--------------|--------|
| 46e675de-... | 28726ad7-... | — | Ingediend | Ingediend. | — |
| abe63a35-... | bae6da1c-... | — | Behandelen [en afdoen] | Behandeld | — |

**Columns:** Id, Agendapunt_Id, StemmingsSoort, BesluitSoort, BesluitTekst, Opmerking, Status, AgendapuntZaakBesluitVolgorde, GewijzigdOp, ApiGewijzigdOp

---

## 32. Stemming — Individual votes (Voor / Tegen / Niet deelgenomen)

| Id | Besluit_Id | Soort | ActorNaam | ActorFractie | FractieGrootte |
|----|------------|-------|-----------|--------------|----------------|
| d064812f-... | f59073ca-... | Tegen | GroenLinks-PvdA | GroenLinks-PvdA | 25 |
| af67af03-... | 05c0994d-... | Voor | 50PLUS | 50PLUS | 4 |

**Columns:** Id, Besluit_Id, Soort (Voor/Tegen/Niet), FractieGrootte, ActorNaam, ActorFractie, Vergissing, SidActorLid, GewijzigdOp, ApiGewijzigdOp

---

## 33. Zaak — Legislative cases

| Id | Nummer | Soort | Titel | Status | Onderwerp |
|----|--------|-------|-------|--------|-----------|
| 899f2722-... | 2025Z03792 | Schriftelijke vragen | — | Vrijgegeven | De (dr... |
| 2965764e-... | 2017Z02547 | Motie | Wijziging van de Wet kinderopvang en kwa... | — | — |

**Columns:** Id, Nummer, Soort, Titel, Citeertitel, Alias, Status, Onderwerp, GewijzigdOp, ApiGewijzigdOp

---

## 34. ZaakActor — Initiators/actors in cases

| Id | Zaak_Id | ActorNaam | ActorFractie | ActorAfkorting | Functie | Relatie |
|----|---------|-----------|--------------|----------------|---------|---------|
| f6eac75c-... | 56f5f819-... | F. Teeven | — | — | staatssecretaris van Veiligheid en Justitie | — |
| 5c5451b8-... | 0d67e412-... | vaste commissie voor Financiën | — | FIN | — | — |

**Columns:** Id, Zaak_Id, ActorNaam, ActorFractie, ActorAfkorting, Functie, Relatie, SidActor, GewijzigdOp, ApiGewijzigdOp

---

## 35. Vergadering — Completed meetings

| Id | Soort | Titel | Zaal | Vergaderjaar | VergaderingNummer | Datum | Aanvangstijd |
|----|-------|-------|------|--------------|-------------------|-------|--------------|
| cfe956ac-... | Commissie | Initiatiefnota van de leden Dassen en Omtzigt over wettelijke maatregelen... | Troelstrazaal | 2021-2022 | 66 | 2022-09-18 | 2022-09-19 12:00 |
| 501ce144-... | Commissie | Procedurevergadering commissie voor Buitenlandse Handel | Wttewaall van Stoetwegenzaal | 2022-2023 | 592 | 2023-05-24 | 2023-05-25 11:30 |
| 9e079a71-... | Commissie | Landbouw- en Visserijraad op 30 en 31 mei 2016 | Klompézaal | 2015-2016 | 0 | 2016-05-24 | 2016-05-25 10:00 |

**Columns:** Id, Soort, Titel, Zaal, Vergaderjaar, VergaderingNummer, Datum, Aanvangstijd, Sluiting, Kamer, GewijzigdOp, ApiGewijzigdOp

---

## 36. Verslag — Meeting reports / minutes (stenograms)

| Id | Soort | Status | ContentType | ContentLength |
|----|-------|--------|-------------|---------------|
| 94fbdc2b-... | Tussenpublicatie | Ongecorrigeerd | text/xml | 82187 |
| d25cda29-... | Tussenpublicatie | Ongecorrigeerd | text/xml | 109331 |

**Columns:** Id, Vergadering_Id, Soort, Status, ContentType, ContentLength, GewijzigdOp, ApiGewijzigdOp

---

## 37. Toezegging — Ministerial promises

| Id | Nummer | ActiviteitNummer | Naam | Functie | Status | Tekst |
|----|--------|------------------|------|---------|--------|-------|
| 79a2206a-... | TZ202412-116 | 2024A05766 | Wiersma, F.M. | Minister van Landbouw, Visserij, Voedselzekerheid en Natuur | Openstaand | De minister stuurt in maart 2025 de terugkoppeling op het emissiereductieplan... |
| fa9b16b4-... | TZ202504-057 | 2025A00793 | Madlener, B. | Minister van Infrastructuur en Waterstaat | Afgedaan | De Kamer wordt voor de zomer geïnformeerd over welke concrete extra acties... |
| 159e3e36-... | TZ202402-164 | 2023A02803 | Schreinemacher, E.N.A.J. | Minister voor Buitenlandse Handel en Ontwikkelingssamenwerking | Afgedaan | De minister zegt een vertrouwelijke technische briefing toe over exportbeperkende maatregelen... |

**Columns:** Id, Aanmaakdatum, Nummer, ActiviteitNummer, Naam, Achternaam, Initialen, Voornaam, Achtervoegsel, Titulatuur, Functie, Status, DatumNakoming, Ministerie, Tekst, GewijzigdOp, ApiGewijzigdOp

---

## 38. Zaal — Rooms in parliament

| Id | Naam | SysCode |
|----|------|---------|
| c66f17e7-... | Schrijfkamer | 55 |
| 6a528771-... | Klompézaal | 74 |
| b6953420-... | Oudkamer | 81 |
| 501ce144-... | Troelstrazaal | — |
| 9e079a71-... | Wttewaall van Stoetwegenzaal | — |

**Columns:** Id, Naam, SysCode, GewijzigdOp, ApiGewijzigdOp

---

## Linking entities

- **Persoon** ↔ **PersoonContactinformatie**, **PersoonGeschenk**, **PersoonLoopbaan**, etc. via `Persoon_Id`
- **Fractie** ↔ **FractieZetel** via `Fractie_Id`; **FractieZetel** ↔ **FractieZetelPersoon** via `FractieZetel_Id`
- **Activiteit** ↔ **ActiviteitActor**, **Agendapunt**, **Reservering** via `Activiteit_Id`
- **Document** ↔ **DocumentActor**, **DocumentVersie** via `Document_Id`
- **Besluit** ↔ **Stemming** via `Besluit_Id`
- **Zaak** ↔ **ZaakActor** via `Zaak_Id`
- **Vergadering** ↔ **Verslag** via `Vergadering_Id`
