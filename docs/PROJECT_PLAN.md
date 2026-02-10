# Far-Right Rise Analysis — Full Project Plan

## Project overview

A semester-long (16-week) university project for 5 people analyzing the rise of far-right parties (PVV, FvD, JA21) in the Dutch Tweede Kamer. Combines parliamentary open data with NLP, machine learning, and social science theory.

**Deliverables**: Research paper, presentation, interactive dashboard.

---

## Team roles

| Role | Person | Focus area |
|------|--------|------------|
| **Person 1** | Data Engineer | Pipeline, data cleaning, linking external datasets, infrastructure |
| **Person 2** | NLP Researcher | Text analysis, topic modeling, sentiment, populism detection |
| **Person 3** | ML Engineer | Vote prediction, motion outcomes, classification models |
| **Person 4** | Social Science Researcher | Theory, literature review, research design, interpretation |
| **Person 5** | Visualization & Integration Lead | Dashboard, paper writing, presentation, project management |

---

## Phase 1: Foundation (Weeks 1-3)

### ALL: Project kickoff and orientation

- Read [DATASET.md](DATASET.md) and the [README.md](../README.md)
- Run the pipeline (`python pipeline.py`) so everyone has local data
- Explore the official API docs: https://opendata.tweedekamer.nl/documentatie/informatiemodel
- Agree on research questions (see Phase 2)
- Set up shared Git workflow (feature branches, PR reviews)

### Person 1 — Data infrastructure

**Week 1-2: Data audit and enrichment**

- Run pipeline, verify all 38 entities downloaded
- Profile every parquet file: row counts, column types, null rates, date ranges
- Create a data dictionary notebook (`notebooks/01_data_audit.ipynb`) documenting every column in every entity
- Identify the far-right parties in `Fractie`:
  - PVV (Partij voor de Vrijheid) — Wilders, entered 2006
  - FvD (Forum voor Democratie) — Baudet, entered 2017
  - JA21 — split from FvD, entered 2021
  - LPF (Lijst Pim Fortuyn) — 2002-2006 (historical, may be partially in data)

**Week 2-3: External data linking**

- Download and integrate external datasets:
  - **Election results** (Kiesraad): seat counts per party per election
  - **Polls** (Peilingwijzer / I&O Research): monthly party support
  - **Chapel Hill Expert Survey (CHES)**: party positions on left-right, immigration, EU
  - **Manifesto Project (MARPOR)**: coded party manifestos
  - **CBS StatLine**: demographics, immigration numbers, unemployment
- Create a unified time index (by vergaderjaar or calendar year) for merging
- Build helper functions in `src/utils.py`:
  - `get_party_mps(fractie, date)` — who sits for a party at a given time
  - `get_far_right_ids()` — returns set of Fractie_Ids for PVV, FvD, JA21
  - `get_voting_record(persoon_id)` — all votes for an MP
  - `classify_topic(zaak)` — keyword-based topic tagger (v1)

**Week 3: Document text fetcher**

- Build a script (`src/fetch_documents.py`) to download actual document PDFs/text via:
  `https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Document/{id}/resource`
- Focus on key document types: Motie, Amendement, Schriftelijke vragen, Brief regering
- Extract text from PDFs (use `pymupdf` or `pdfplumber`)
- Store as `data/texts/{DocumentId}.txt`
- This is rate-limited — prioritize far-right authored documents and immigration-related cases

### Person 2 — NLP groundwork

**Week 1-2: Literature review on Dutch NLP for parliamentary text**

- Survey existing Dutch NLP models:
  - **BERTje** (Dutch BERT): https://github.com/wietsedv/bertje
  - **RobBERT** (Dutch RoBERTa): https://github.com/iPieter/RobBERT
  - **multilingual models**: XLM-RoBERTa
- Read key papers:
  - Rooduijn & Pauwels (2011) — measuring populism in party manifestos
  - Hawkins et al. (2019) — Team Populism coding scheme
  - Grundl (2020) — populism dictionary approach
- Document findings in `docs/NLP_LITERATURE.md`

**Week 2-3: Baseline text processing**

- Build text preprocessing pipeline (`src/nlp/preprocess.py`):
  - Dutch tokenization (spaCy `nl_core_news_lg`)
  - Stopword removal, lemmatization
  - Named entity recognition for people, parties, organizations
- Create keyword dictionaries for key topics:
  - Immigration: immigratie, asiel, vluchtelingen, integratie, islam, grenzen...
  - Economy: werkgelegenheid, belasting, economie, begroting...
  - EU/sovereignty: Europa, Brussel, soevereiniteit, nexit...
  - Security: veiligheid, criminaliteit, terrorisme, politie...
  - Healthcare: zorg, gezondheid, ziekenhuis...
- Test on sample of Document titles and Zaak onderwerpen

### Person 3 — ML groundwork

**Week 1-2: Data exploration for ML**

- Create `notebooks/02_voting_exploration.ipynb`:
  - Load Stemming, join with Besluit, Zaak, Fractie, Persoon
  - Compute basic stats: votes per party, Voor/Tegen rates
  - Visualize party cohesion over time
  - Identify far-right voting patterns vs. mainstream
- Create `notebooks/03_motion_exploration.ipynb`:
  - Load Zaak where Soort = "Motie"
  - Join with ZaakActor (sponsors), Besluit (outcome), Stemming (votes)
  - Compute motion acceptance rates by party

**Week 2-3: Feature engineering plan**

- Document planned features for vote prediction:
  - Party membership (one-hot)
  - Topic of the Zaak (from keyword classifier or topic model)
  - Cabinet appreciation (Kabinetsappreciatie)
  - Sponsoring party
  - Historical voting similarity between parties
  - Time features (vergaderjaar, month)
  - MP tenure, committee memberships
- Set up ML experiment framework:
  - `src/ml/features.py` — feature extraction
  - `src/ml/models.py` — model training
  - `src/ml/evaluate.py` — evaluation metrics
  - Use `sklearn` pipelines, `mlflow` or simple CSV logging for experiment tracking

### Person 4 — Social science research

**Week 1-3: Full literature review**

Research and write up the following theories (target: `docs/LITERATURE_REVIEW.md`):

**A. Demand-side theories (why voters turn to far-right)**

- **Modernization losers thesis** (Betz 1994): Globalization creates economic losers who turn to far-right
  - Measure: correlate unemployment/immigration data (CBS) with far-right vote share
- **Cultural backlash thesis** (Norris & Inglehart 2019): Reaction against progressive value change
  - Measure: topic salience of identity/immigration in parliamentary debate
- **Relative deprivation theory** (Gurr 1970): Perceived gap between expectations and reality
  - Measure: sentiment in government responses vs. opposition rhetoric

**B. Supply-side theories (what far-right parties do)**

- **Issue ownership theory** (Petrocik 1996): Parties "own" certain issues; far-right owns immigration
  - Measure: share of immigration-related Zaak/Document initiated by far-right vs. others
- **Populism theory** (Mudde 2004): Thin-centered ideology: "pure people" vs. "corrupt elite"
  - Measure: populism scores in parliamentary texts (NLP)
- **Mainstreaming thesis** (Akkerman et al. 2016): Mainstream parties adopt far-right positions
  - Measure: topic and rhetoric convergence over time between VVD/CDA and PVV/FvD

**C. Institutional theories (how parliament shapes outcomes)**

- **Agenda-setting theory** (Baumgartner & Jones 1993): Who controls what gets discussed
  - Measure: share of agenda items initiated by far-right, topic distribution over time
- **Coalition theory**: Cordon sanitaire vs. cooperation
  - Measure: voting alignment networks, co-sponsorship patterns
- **Responsiveness theory** (Hobolt & Klemmensen 2008): Do mainstream parties respond to far-right?
  - Measure: topic adoption lag — does mainstream pick up far-right topics after delay?

**Deliverable**: 15-20 page literature review with hypotheses mapped to measurable indicators.

### Person 5 — Project management and visualization setup

**Week 1-2: Project infrastructure**

- Set up project board (GitHub Projects or Trello)
- Define milestone schedule (this plan, adapted)
- Create `docs/RESEARCH_DESIGN.md` — formalize research questions:
  1. How has the parliamentary agenda on immigration/identity changed over time?
  2. Do far-right parties exhibit distinctive rhetorical patterns (populism, negativity)?
  3. Can we predict voting behavior, and what does the model reveal about far-right alignment?
  4. Have mainstream parties shifted their rhetoric/voting toward far-right positions?
  5. What factors predict the success/failure of far-right motions?

**Week 2-3: Dashboard skeleton**

- Choose framework: **Streamlit** (fastest for Python team) or **Plotly Dash**
- Create `app/dashboard.py` with placeholder pages:
  - Overview (party seat counts over time)
  - Voting analysis
  - Topic trends
  - Network visualization
  - Model results
- Set up basic data loading from processed parquet files

---

## Phase 2: Core Analysis (Weeks 4-9)

### Person 1 — Data pipelines for analysis

**Week 4-5: Voting dataset**

- Create `src/analysis/voting.py`:
  - Build a flat voting table: one row per (Stemming, Persoon, Fractie, Besluit, Zaak)
  - Add topic labels (from Person 2's keyword classifier)
  - Add temporal features
  - Export as `data/analysis/voting_flat.parquet`
- Create party-level aggregation: per-party Voor/Tegen percentages per Besluit
- Create party-pair voting similarity matrix per vergaderjaar

**Week 5-6: Document corpus**

- Process downloaded document texts into clean corpus
- Create `data/analysis/document_corpus.parquet`:
  - Columns: DocumentId, FractieId, PartyName, Soort, Titel, Onderwerp, FullText, Date, Vergaderjaar
  - Filter to key types: Motie, Amendement, Schriftelijke vragen
- Create far-right vs. mainstream subsets

**Week 6-7: Network datasets**

- Build co-sponsorship network from ZaakActor:
  - Nodes = parties, edges = number of co-sponsored cases
  - Per vergaderjaar
- Build voting alignment network from Stemming:
  - Nodes = parties, edges = voting agreement percentage
  - Per vergaderjaar and per topic

**Week 7-9: External data integration**

- Merge election results, polls, CHES positions into time-indexed dataset
- Create `data/analysis/context_timeline.parquet`:
  - Year, far-right seat share, poll numbers, immigration numbers, unemployment
  - CHES party positions on key dimensions

### Person 2 — NLP analysis

**Week 4-5: Topic modeling**

- Apply BERTopic to Document titles + onderwerpen + Zaak titles:
  - Use Dutch sentence-transformers (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
  - Extract 20-50 topics
  - Manually label top topics (immigration, economy, EU, etc.)
  - Save topic assignments per document/zaak
- Create `notebooks/04_topic_modeling.ipynb` with:
  - Topic distribution over time
  - Topic distribution per party
  - Far-right topic focus vs. mainstream

**Week 5-7: Sentiment and populism analysis**

- Sentiment analysis on document texts:
  - Test zero-shot with RobBERT
  - If poor, fine-tune on 200-300 manually labeled sentences
  - Score: negative/neutral/positive per document
- Populism detection:
  - Implement dictionary-based approach (Rooduijn & Pauwels word list)
  - Count populism markers per document: anti-elite, people-centrism, exclusionism
  - Validate against CHES expert scores
  - If time: fine-tune classifier on Team Populism labeled data

**Week 7-9: Rhetoric convergence analysis**

- For each vergaderjaar, compute:
  - Average populism score per party
  - Average sentiment per party
  - Topic distribution per party
- Test **mainstreaming hypothesis**:
  - Does VVD/CDA populism score increase after PVV/FvD gains?
  - Does mainstream topic distribution shift toward immigration after far-right success?
  - Use Granger causality or simple lagged correlation

### Person 3 — Machine learning models

**Week 4-6: Vote prediction model**

- Build feature matrix from `voting_flat.parquet`:
  - Features: party (one-hot), topic, cabinet appreciation, vergaderjaar, sponsor party, MP tenure
  - Target: Voor / Tegen / Niet deelgenomen
- Train/test split: temporal (train <= 2020, test 2021-2025)
- Models to try:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost / LightGBM
  - (Optional) LSTM on voting sequences
- **Target metrics**:
  - Overall accuracy, macro F1
  - Per-party accuracy (especially PVV, FvD)
  - Feature importance analysis
- Key analysis: **When does the model fail?** — these are the interesting cases (party rebellion, cross-party alignment)
- Create `notebooks/05_vote_prediction.ipynb`

**Week 6-8: Motion outcome prediction**

- Build feature matrix:
  - Features: sponsoring party, topic, number of co-sponsors, cabinet appreciation, vergaderjaar
  - Target: motion accepted / rejected (from Besluit + Stemming aggregation)
- Same model pipeline as vote prediction
- Key analysis: What predicts far-right motion success? Are more passing over time?
- Create `notebooks/06_motion_prediction.ipynb`

**Week 8-9: Party text classification**

- Binary classifier: far-right authored vs. mainstream authored documents
- Features: TF-IDF or Dutch BERT embeddings of document text
- Train on documents with known DocumentActor party
- **Key output**: What words/topics are most distinctive for far-right? How has this changed over time?
- Temporal analysis: train on 2010-2015, test on 2020-2025 — does the boundary blur? (mainstreaming signal)
- Create `notebooks/07_party_classification.ipynb`

### Person 4 — Hypothesis testing and interpretation

**Week 4-5: Formalize hypotheses**

- Write formal, testable hypotheses (in `docs/HYPOTHESES.md`):

  **H1 (Issue salience)**: The share of immigration-related parliamentary items has increased over time, and far-right parties initiate a disproportionate share.

  **H2 (Populism)**: Far-right party documents score significantly higher on populism metrics than mainstream party documents.

  **H3 (Mainstreaming)**: Mainstream-right parties (VVD, CDA) show increasing populism scores and immigration topic focus over time, especially after elections where far-right gains seats.

  **H4 (Voting alignment)**: Voting alignment between mainstream-right and far-right has increased over time on immigration-related votes but not on other topics.

  **H5 (Motion success)**: Far-right motion acceptance rates have increased over time, and immigration-related motions show the largest increase.

  **H6 (Predictability)**: Votes on immigration-related topics are less predictable by party affiliation alone than votes on other topics, indicating cross-party dynamics.

**Week 5-7: Statistical testing**

- For each hypothesis, define:
  - Null and alternative hypothesis
  - Statistical test (t-test, chi-square, regression, time series)
  - Effect size measure
  - Confounders to control for
- Work with Person 2 (NLP results) and Person 3 (ML results) to run tests
- Create `notebooks/08_hypothesis_tests.ipynb`

**Week 7-9: Qualitative validation**

- Select key cases (specific debates, votes, motions) that illustrate quantitative findings
- Write case studies for the paper:
  - Example: A specific immigration debate where mainstream parties voted with PVV
  - Example: A motion by FvD that was adopted with VVD support
- Cross-reference with news coverage and academic literature

### Person 5 — Dashboard and visualization

**Week 4-6: Core dashboard pages**

- **Page 1: Parliamentary landscape**
  - Party seat counts over time (stacked area chart)
  - Far-right seat share trend line
  - Election markers on timeline
- **Page 2: Topic trends**
  - Stacked area chart of topic shares over time
  - Immigration topic highlighted
  - Per-party topic distribution (grouped bar)
- **Page 3: Voting analysis**
  - Party voting alignment heatmap (interactive, per vergaderjaar)
  - Far-right voting agreement with each party over time
  - Drill-down: click a party pair to see specific votes

**Week 6-8: Advanced dashboard pages**

- **Page 4: Network view**
  - Co-sponsorship network graph (nodes = parties, edges = co-sponsored motions)
  - Animated over time (slider by vergaderjaar)
  - Highlight far-right connections
- **Page 5: NLP results**
  - Populism score trends per party
  - Sentiment trends per party
  - Word clouds per party per period
  - Topic model visualization (intertopic distance map)
- **Page 6: Model results**
  - Vote prediction accuracy dashboard
  - Feature importance charts
  - Confusion matrices
  - "Interesting failures" explorer — browse votes where model was wrong

**Week 8-9: Polish and interactivity**

- Add filters: date range, party selection, topic filter
- Add tooltips and explanations for non-technical audience
- Performance optimization (cache parquet loads)

---

## Phase 3: Integration and Writing (Weeks 10-13)

### ALL: Paper writing

Structure (target: 30-40 pages):

```
1. Introduction (Person 4 + Person 5)
   - Research question, relevance, Dutch political context

2. Literature Review (Person 4)
   - Theories, hypotheses, prior work

3. Data and Methods (Person 1 + Person 2 + Person 3)
   - Data sources, pipeline, NLP methods, ML models

4. Results
   4.1 Descriptive: Topic trends, agenda shifts (Person 2)
   4.2 NLP: Populism and sentiment analysis (Person 2)
   4.3 ML: Vote prediction and feature analysis (Person 3)
   4.4 Networks: Voting alignment and co-sponsorship (Person 1)
   4.5 Hypothesis tests (Person 4)

5. Discussion (Person 4 + Person 5)
   - Interpretation through social science lens
   - Limitations

6. Conclusion (Person 4)

Appendix: Data dictionary, model hyperparameters,
         full results tables (Person 1 + Person 3)
```

### Task division for writing weeks

| Week | Person 1 | Person 2 | Person 3 | Person 4 | Person 5 |
|------|----------|----------|----------|----------|----------|
| 10 | Data/methods section | NLP results section | ML results section | Lit review polish | Introduction draft |
| 11 | Network results | NLP results polish | ML results polish | Hypothesis results | Discussion draft |
| 12 | Appendix, data dict | Review Person 3 | Review Person 2 | Discussion, conclusion | Integration, editing |
| 13 | Final data checks | Figures polish | Figures polish | Final review | Final formatting |

---

## Phase 4: Presentation and Delivery (Weeks 14-16)

### Week 14: Presentation prep

- Person 5 creates slide deck structure
- Each person drafts slides for their section (3-5 slides each)
- Target: 20-25 slides total, 20-30 min presentation

### Week 15: Rehearsal and polish

- Full team rehearsal
- Dashboard demo preparation
- Peer review of paper (swap sections, everyone reads everything)

### Week 16: Delivery

- Final presentation
- Submit paper
- Push all code to GitHub (clean notebooks, requirements, documentation)

---

## Repository structure (target)

```
.
├── app/
│   └── dashboard.py              # Streamlit dashboard
├── data/
│   ├── raw/                      # Raw JSON (gitignored)
│   ├── processed/                # Clean parquet (gitignored)
│   ├── texts/                    # Document full texts (gitignored)
│   ├── analysis/                 # Analysis-ready datasets (gitignored)
│   └── external/                 # Election results, polls, CHES (gitignored)
├── docs/
│   ├── DATASET.md                # Dataset documentation
│   ├── PROJECT_PLAN.md           # This file
│   ├── LITERATURE_REVIEW.md      # Social science literature
│   ├── NLP_LITERATURE.md         # NLP methods literature
│   ├── RESEARCH_DESIGN.md        # Research questions and design
│   ├── HYPOTHESES.md             # Formal hypotheses
│   └── paper/                    # LaTeX or Word paper
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_voting_exploration.ipynb
│   ├── 03_motion_exploration.ipynb
│   ├── 04_topic_modeling.ipynb
│   ├── 05_vote_prediction.ipynb
│   ├── 06_motion_prediction.ipynb
│   ├── 07_party_classification.ipynb
│   └── 08_hypothesis_tests.ipynb
├── src/
│   ├── fetch.py                  # API fetcher (exists)
│   ├── preprocess.py             # Data preprocessor (exists)
│   ├── fetch_documents.py        # Document text downloader (new)
│   ├── utils.py                  # Helper functions (new)
│   ├── nlp/
│   │   ├── preprocess.py         # Text preprocessing
│   │   ├── topics.py             # Topic modeling
│   │   ├── sentiment.py          # Sentiment analysis
│   │   └── populism.py           # Populism detection
│   ├── ml/
│   │   ├── features.py           # Feature extraction
│   │   ├── models.py             # Model training
│   │   └── evaluate.py           # Evaluation
│   └── analysis/
│       ├── voting.py             # Voting dataset builder
│       └── networks.py           # Network analysis
├── config.yaml
├── pipeline.py
├── requirements.txt
├── Makefile
└── README.md
```

---

## Key milestones

| Week | Milestone | Gate |
|------|-----------|------|
| 3 | Foundation complete | Everyone has data, lit review drafted, pipelines working |
| 6 | Core analysis halfway | Topic model trained, vote prediction baseline, hypotheses formalized |
| 9 | All analysis complete | All notebooks done, all models evaluated, dashboard functional |
| 13 | Paper complete | Full draft reviewed by all members |
| 16 | Final delivery | Paper submitted, presentation given, dashboard demo |

---

## Risk mitigation

| Risk | Mitigation |
|------|------------|
| Document text download is too slow / rate-limited | Prioritize: only download motions and schriftelijke vragen from far-right parties and immigration-related cases. Fall back to title+onderwerp text only. |
| NLP models perform poorly on parliamentary Dutch | Use keyword-based approaches as baseline; these are interpretable and don't need training data. Layer ML on top only if it improves. |
| Mixed skill levels | Person 4 (social science) and Person 5 (viz) don't need deep ML skills. Person 1 supports data needs for everyone. Pair programming sessions weekly. |
| Scope creep | Stick to the 6 hypotheses. Everything else is "nice to have". |
| External data hard to get | CHES and election data are freely downloadable. If CBS data is complicated, skip it and use election results as the main external source. |
