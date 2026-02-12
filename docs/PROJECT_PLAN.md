# Speech-to-Vote Prediction — Full Project Plan

## Project overview

A semester-long (16-week) university project for 5 people. **Core question: Can we predict how a parliamentarian will vote on a bill based on what they say in the debate?**

MP gives a speech → we extract features from the speech text → we predict their vote (Voor / Tegen / Niet deelgenomen).

Uses the Dutch Tweede Kamer open data: debate transcripts (Verslagen), votes (Stemming), and the full parliamentary information model.

**Deliverables**: Research paper, presentation, interactive dashboard, trained prediction model.

---

## The core idea

```
┌─────────────────────┐      ┌──────────────┐      ┌────────────────┐
│  Debate transcript   │      │   Features   │      │   Prediction   │
│  (Verslag XML)       │─────►│  from speech │─────►│  Voor / Tegen  │
│                      │      │  text + meta │      │  / Niet        │
│  "Voorzitter, dit    │      │              │      │                │
│   wetsvoorstel is    │      │  sentiment,  │      │  XGBoost /     │
│   onacceptabel..."   │      │  topic, tone │      │  BERT / etc.   │
└─────────────────────┘      └──────────────┘      └────────────────┘
```

**Data chain:**
- `Vergadering` → `Verslag` (transcript XML with individual speech segments)
- `Vergadering` → `Activiteit` → `Agendapunt` → `Besluit` → `Stemming` (votes)
- Link: speech by Person X during debate on Agendapunt Y → their vote on Besluit for Agendapunt Y

---

## Team roles

| Role | Person | Focus area |
|------|--------|------------|
| **Person 1** | Data Engineer | Pipeline, speech extraction, linking speeches to votes, infrastructure |
| **Person 2** | NLP Researcher | Speech text processing, embeddings, feature extraction from text |
| **Person 3** | ML Engineer | Prediction models, experiment tracking, evaluation |
| **Person 4** | Social Science Researcher | Theory, literature review, interpretation, qualitative analysis |
| **Person 5** | Visualization & Integration Lead | Dashboard, paper writing, presentation, project management |

---

## Phase 1: Foundation (Weeks 1-3)

### ALL: Project kickoff and orientation

- Read [DATASET.md](DATASET.md) and the [README.md](../README.md)
- Run the pipeline (`python pipeline.py`) so everyone has local data
- Explore the official API docs: https://opendata.tweedekamer.nl/documentatie/informatiemodel
- **Key focus**: understand how Verslag (transcripts) connect to Stemming (votes)
- Set up shared Git workflow (feature branches, PR reviews)

### Person 1 — Data infrastructure

**Week 1-2: Data audit and speech data acquisition**

- Run pipeline, verify all entities downloaded
- **Critical task**: Fetch `Verslag` entity (meeting transcripts)
  - Verslag contains XML text of debates (stenograms)
  - Each Verslag links to a Vergadering via `Vergadering_Id`
  - Download the actual XML content via:
    `https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0/Verslag/{id}/resource`
  - Store as `data/texts/{VerslagId}.xml`
- Profile every parquet file: row counts, column types, null rates, date ranges
- Create `notebooks/01_data_audit.ipynb` documenting every entity

**Week 2-3: Speech-to-vote linking pipeline**

- Build the critical data link — this is the hardest data engineering task:
  1. Parse Verslag XML to extract individual speech segments
     - Each speech has: speaker name, party, text content, timestamp
     - Verslag XML format uses `<spreker>` and `<tekst>` elements
  2. Link speeches to agenda items:
     - Verslag → Vergadering → Activiteit → Agendapunt
     - Match speech segments to the Agendapunt being discussed (by order/time)
  3. Link agenda items to votes:
     - Agendapunt → Besluit → Stemming
  4. Final join: Speech(person, text) → Vote(person, Voor/Tegen)
- Build helper functions in `src/utils.py`:
  - `parse_verslag_xml(xml_path)` → list of speech segments
  - `get_party_mps(fractie, date)` — who sits for a party at a given time
  - `get_voting_record(persoon_id)` — all votes for an MP
  - `link_speech_to_vote(verslag_id)` — returns (speech_text, vote) pairs
- Create `data/analysis/speech_vote_pairs.parquet`:
  - Columns: Persoon_Id, PersonName, Fractie, SpeechText, Agendapunt_Id, Besluit_Id, Vote (Voor/Tegen/Niet), Date, Vergaderjaar
  - This is the **master dataset** for the whole project

**Week 3: Data quality check**

- How many speech-vote pairs can we link? (target: thousands)
- What's the coverage? (% of votes where we have a preceding speech)
- What's the class balance? (Voor vs. Tegen vs. Niet deelgenomen)
- Write up findings in `notebooks/01b_data_quality.ipynb`

### Person 2 — NLP groundwork

**Week 1-2: Literature review on speech-to-vote prediction**

- Survey existing work:
  - **Predicting Congressional Votes** (Thomas et al., 2006) — classic speech-vote paper
  - **Capturing the Style of Parliamentary Debates** (Rheault et al., 2016)
  - **Predicting Votes from Debate** (Stab et al., 2018) — argument mining
  - **How to Frame a Politician** (Johnson & Goldwasser, 2018) — framing and voting
  - **BERT for legislative text** (recent papers using transformers on parliamentary data)
- Survey Dutch NLP models:
  - **BERTje** (Dutch BERT): https://github.com/wietsedv/bertje
  - **RobBERT** (Dutch RoBERTa): https://github.com/iPieter/RobBERT
  - **Multilingual models**: XLM-RoBERTa
- Document findings in `docs/NLP_LITERATURE.md`

**Week 2-3: Baseline text processing**

- Build speech text preprocessing pipeline (`src/nlp/preprocess.py`):
  - Clean XML artifacts, formatting noise
  - Dutch tokenization (spaCy `nl_core_news_lg`)
  - Sentence splitting (speeches can be long)
  - Named entity recognition for people, parties, organizations
- Create keyword dictionaries for topic detection:
  - Immigration, Economy, EU/sovereignty, Security, Healthcare, Education, etc.
- Compute basic speech statistics:
  - Average speech length per party
  - Vocabulary richness
  - Most common topics per party

### Person 3 — ML groundwork

**Week 1-2: Voting data exploration**

- Create `notebooks/02_voting_exploration.ipynb`:
  - Load Stemming, join with Besluit, Agendapunt, Fractie, Persoon
  - Compute basic stats: votes per party, Voor/Tegen rates
  - How predictable are votes from party alone? (majority-class baseline)
  - Visualize party cohesion over time
- Create `notebooks/03_baseline_predictions.ipynb`:
  - **Baseline 1**: Always predict majority vote of the party → what accuracy?
  - **Baseline 2**: Predict based on party + topic → what accuracy?
  - These baselines tell us: how much does speech text *add* beyond party membership?

**Week 2-3: Feature engineering plan**

- Document planned feature categories:
  - **Speech features** (from NLP): sentiment, topic, key phrases, speech length, argumentative structure
  - **Speaker features**: party, tenure, committee memberships, historical voting pattern
  - **Bill features**: topic of the Zaak, cabinet appreciation (Kabinetsappreciatie), sponsoring party
  - **Context features**: vergaderjaar, coalition/opposition status
- Set up ML experiment framework:
  - `src/ml/features.py` — feature extraction
  - `src/ml/models.py` — model training
  - `src/ml/evaluate.py` — evaluation metrics
  - Use `sklearn` pipelines, simple CSV logging for experiment tracking

### Person 4 — Social science research

**Week 1-3: Literature review**

Research and write up (target: `docs/LITERATURE_REVIEW.md`):

**A. Parliamentary speech and voting behavior**

- **Signaling theory** (Mayhew 1974): MPs use speeches to signal positions to voters, not to persuade colleagues
  - Implication: speech may predict vote because both reflect the same underlying position
- **Party discipline theory** (Bowler et al. 1999): Party whips control votes; speeches may deviate more than votes
  - Implication: speeches may show personal opinion, votes show party line — mismatches are interesting
- **Deliberation theory** (Habermas 1996): Debate actually changes minds
  - Implication: if speeches predict votes perfectly, deliberation may not matter; if poorly, debate has real effects

**B. Computational political science**

- **Text-as-data** (Grimmer & Stewart 2013): Using text to measure political positions
- **Ideal point estimation** (Clinton et al. 2004): Placing politicians on a spectrum from votes
- **Wordscoring / Wordfish** (Laver et al. 2003; Slapin & Proksch 2008): Placing politicians on a spectrum from text
- **Speech-vote alignment** (Lauderdale & Herzog 2016): Do words predict positions?

**C. Dutch parliamentary politics**

- Coalition vs. opposition dynamics in the Netherlands
- Role of the Tweede Kamer in legislative process
- Party discipline in Dutch politics (relatively strong)

**Deliverable**: 15-20 page literature review with hypotheses.

### Person 5 — Project management and visualization setup

**Week 1-2: Project infrastructure**

- Set up project board (GitHub Projects or Trello)
- Define milestone schedule (this plan)
- Create `docs/RESEARCH_DESIGN.md` — formalize research questions:
  1. Can parliamentary speech predict individual voting behavior?
  2. How much predictive power does speech text add beyond party membership alone?
  3. What speech features (sentiment, topic, key phrases) are most predictive?
  4. When do speeches and votes diverge? (rebellion, cross-party dynamics)
  5. Do prediction patterns differ across parties, topics, or time periods?

**Week 2-3: Dashboard skeleton**

- Choose framework: **Streamlit** (fastest for Python team) or **Plotly Dash**
- Create `app/dashboard.py` with placeholder pages:
  - Overview (dataset stats, speech-vote pair counts)
  - Speech explorer (browse speeches with vote outcomes)
  - Model results
  - Prediction demo (paste a speech → get vote prediction)
- Set up basic data loading from processed parquet files

---

## Phase 2: Core Analysis (Weeks 4-9)

### Person 1 — Data pipelines

**Week 4-5: Master dataset refinement**

- Refine `speech_vote_pairs.parquet`:
  - Clean edge cases: MPs who speak but don't vote, fractie-level votes vs. individual
  - Handle "Niet deelgenomen" — decide: include as 3rd class or exclude?
  - Add speaker metadata: party, tenure at time of speech, committee roles
  - Add bill metadata: topic, cabinet appreciation, sponsor party
- Create train/validation/test splits:
  - **Temporal split**: train ≤ 2021, validation 2022, test 2023-2025
  - Alternative: **speaker split** (train on some MPs, test on unseen MPs)
- Export as `data/analysis/train.parquet`, `data/analysis/val.parquet`, `data/analysis/test.parquet`

**Week 5-7: Aggregated analysis datasets**

- Party-level voting alignment matrix per vergaderjaar
- Per-MP speech statistics: average sentiment, topic distribution, speech frequency
- Per-topic prediction difficulty: which topics are hardest to predict?
- Export for dashboard use

**Week 7-9: Additional data**

- If speech XML download is incomplete, build fallback:
  - Use Activiteit.Onderwerp + Zaak.Titel as proxy for speech topic
  - Use ActiviteitActor to know who participated (even without transcript text)
- Optionally fetch external data:
  - Election results for context
  - Coalition composition per vergaderjaar

### Person 2 — NLP feature engineering

**Week 4-5: Speech embeddings**

- Generate embeddings for all speech texts:
  - **Option A**: Sentence-level embeddings with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - **Option B**: Fine-tuned RobBERT embeddings
  - **Option C**: TF-IDF (interpretable baseline)
- Handle long speeches: truncate, chunk + average, or use sliding window
- Create `notebooks/04_speech_embeddings.ipynb`:
  - Visualize embedding space with t-SNE/UMAP
  - Color by party → do parties cluster?
  - Color by vote → do Voor/Tegen separate?

**Week 5-7: Feature extraction from speech text**

- Build feature extraction pipeline (`src/nlp/features.py`):
  - **Sentiment**: positive/negative/neutral (RobBERT zero-shot or Dutch sentiment model)
  - **Topic**: keyword-based or BERTopic assignment
  - **Stance markers**: "voor", "tegen", "steun", "verwerp", "onaanvaardbaar" — explicit vote signals in text
  - **Argumentative features**: number of claims, evidence, rebuttals (simplified argument mining)
  - **Rhetorical features**: speech length, sentence complexity, question marks (asking vs. stating)
  - **Named entities**: which parties/people are mentioned
- Create `data/analysis/speech_features.parquet`:
  - One row per speech, all extracted features as columns

**Week 7-9: Advanced NLP**

- **Fine-tuned classifier** (if base model underperforms):
  - Fine-tune RobBERT on speech text → vote prediction directly (end-to-end)
  - Compare with feature-based approach
- **Attention analysis**: Which words does the model attend to when predicting Voor vs. Tegen?
  - Use SHAP or attention weights to explain predictions
- **Cross-party language analysis**:
  - Do MPs from different parties use different language when voting the same way?
  - Do opposition speeches sound different from coalition speeches on the same topic?

### Person 3 — Prediction models

**Week 4-6: Core vote prediction models**

Build prediction pipeline using `speech_vote_pairs.parquet` + `speech_features.parquet`:

**Model progression** (each builds on the previous):

| Model | Features | Purpose |
|-------|----------|---------|
| **Baseline 1** | Party only (one-hot) | Lower bound — how much does party explain? |
| **Baseline 2** | Party + topic + cabinet appreciation | Structured metadata only (no speech text) |
| **Model A** | Baseline 2 + TF-IDF of speech | Does text add value over metadata? |
| **Model B** | Baseline 2 + speech embeddings | Dense text representation |
| **Model C** | Baseline 2 + extracted speech features (sentiment, stance, etc.) | Interpretable speech features |
| **Model D** | Fine-tuned RobBERT (end-to-end) | Full neural approach |

- Algorithms for Models A-C: Logistic Regression, Random Forest, XGBoost/LightGBM
- Temporal train/test split
- **Target metrics**: Accuracy, macro F1, per-party F1, AUC-ROC
- Create `notebooks/05_vote_prediction.ipynb`

**Week 6-8: Analysis of predictions**

- **Feature importance**: What speech features matter most? (SHAP values)
- **Error analysis**: When does the model fail?
  - Party rebellions (MP votes against party majority)
  - Conscience votes (free votes, no party whip)
  - Close votes (narrow margins)
- **Ablation study**: Remove speech features → how much does accuracy drop?
  - This answers: "Does speech add predictive value beyond party + metadata?"
- Create `notebooks/06_error_analysis.ipynb`

**Week 8-9: Extended models**

- **Per-topic models**: Train separate models for immigration, economy, healthcare
  - Are some topics harder to predict?
- **Per-party models**: Train models for specific parties
  - Which parties are most predictable from speech? Least?
- **Temporal analysis**: Train on 2013-2018, test on 2019-2025
  - Does prediction accuracy degrade over time? (concept drift)
- **Rebellion detector**: Binary classifier — will this MP rebel (vote against party)?
  - Features: speech text, historical rebellion rate, topic
  - This is the most practically interesting sub-model
- Create `notebooks/07_extended_models.ipynb`

### Person 4 — Hypotheses and interpretation

**Week 4-5: Formalize hypotheses**

Write formal, testable hypotheses (in `docs/HYPOTHESES.md`):

**H1 (Speech predicts vote)**: Individual speech text significantly predicts voting behavior beyond party membership alone.

**H2 (Stance words)**: Explicit stance markers in speech (e.g. "steun", "verwerp", "onaanvaardbaar") are the strongest textual predictors of voting.

**H3 (Sentiment signal)**: Speeches with more negative sentiment toward a bill predict a "Tegen" vote; positive sentiment predicts "Voor".

**H4 (Party discipline)**: Prediction accuracy is higher for parties with strong party discipline (e.g. PVV, SGP) and lower for parties with more internal diversity (e.g. D66, GroenLinks-PvdA).

**H5 (Topic variation)**: Prediction difficulty varies by topic — emotionally charged topics (immigration, identity) show more speech-vote divergence than technical topics (budget, infrastructure).

**H6 (Coalition effect)**: Coalition party MPs are more predictable (they follow party line + coalition agreement) than opposition MPs.

**H7 (Rebellion signal)**: When an MP will rebel against their party, linguistic markers in their speech differ from party-line speeches.

**Week 5-7: Statistical testing**

- For each hypothesis, define:
  - Null and alternative hypothesis
  - Statistical test (t-test, chi-square, regression, permutation test)
  - Effect size measure
- Work with Person 2 and Person 3 to validate
- Create `notebooks/08_hypothesis_tests.ipynb`

**Week 7-9: Qualitative analysis**

- Select 20-30 interesting cases:
  - Speeches where prediction was correct with high confidence
  - Speeches where prediction was wrong (surprises)
  - Rebellion cases: MP speaks one way, votes another (or vice versa)
- Manually analyze these speeches: what was the context?
- Cross-reference with news coverage of key debates
- Write up case studies for the paper

### Person 5 — Dashboard and visualization

**Week 4-6: Core dashboard pages**

- **Page 1: Dataset overview**
  - Total speeches, votes, speech-vote pairs
  - Party distribution, time distribution
  - Interactive filters: date range, party, topic
- **Page 2: Speech explorer**
  - Browse individual speeches with their vote outcome
  - Highlight stance words and sentiment markers in text
  - Filter by party, topic, correct/incorrect prediction
- **Page 3: Model comparison**
  - Bar chart: accuracy of each model (Baseline 1 → Model D)
  - Shows the "lift" from adding speech features
  - Per-party accuracy breakdown

**Week 6-8: Advanced dashboard pages**

- **Page 4: Feature importance**
  - SHAP summary plot: which features drive predictions
  - Interactive: select a speech → see feature contributions
  - Word cloud of most predictive terms for Voor vs. Tegen
- **Page 5: Error analysis**
  - Confusion matrix per party
  - Rebellion cases highlighted
  - Timeline of prediction accuracy (does it change over vergaderjaren?)
- **Page 6: Live prediction demo**
  - Text input box: paste a speech excerpt
  - Select party + topic context
  - Model outputs: predicted vote + confidence + explanation
  - "This speech sounds 78% like a Tegen vote because of: negative sentiment, words X, Y, Z"

**Week 8-9: Polish and interactivity**

- Add filters, tooltips, explanations
- Performance optimization
- Export static figures for the paper

---

## Phase 3: Integration and Writing (Weeks 10-13)

### ALL: Paper writing

Structure (target: 30-40 pages):

```
1. Introduction (Person 4 + Person 5)
   - Can we predict votes from speeches?
   - Why this matters (transparency, deliberation theory)
   - Dutch parliamentary context

2. Literature Review (Person 4)
   - Speech-to-vote prediction prior work
   - Party discipline, signaling, deliberation
   - Computational political science methods

3. Data and Methods (Person 1 + Person 2 + Person 3)
   - Tweede Kamer open data pipeline
   - Speech extraction and linking methodology
   - NLP feature engineering
   - Model architectures and evaluation design

4. Results
   4.1 Dataset description: speech-vote pairs stats (Person 1)
   4.2 NLP features: what speeches look like (Person 2)
   4.3 Prediction results: model comparison + ablation (Person 3)
   4.4 Feature importance: what in speech predicts votes (Person 2 + 3)
   4.5 Error analysis: when predictions fail (Person 3 + 4)
   4.6 Hypothesis tests (Person 4)

5. Discussion (Person 4 + Person 5)
   - What does it mean that speech predicts votes?
   - Signaling vs. deliberation: evidence from our results
   - Party discipline differences
   - Limitations and threats to validity

6. Conclusion (Person 4)

Appendix: Data dictionary, model hyperparameters,
         full results tables, case studies (ALL)
```

### Task division for writing weeks

| Week | Person 1 | Person 2 | Person 3 | Person 4 | Person 5 |
|------|----------|----------|----------|----------|----------|
| 10 | Data/methods section | NLP features section | ML results section | Lit review polish | Introduction draft |
| 11 | Dataset description | Feature importance | Error analysis section | Hypothesis results | Discussion draft |
| 12 | Appendix, data dict | Review Person 3 | Review Person 2 | Discussion, conclusion | Integration, editing |
| 13 | Final data checks | Figures polish | Figures polish | Final review | Final formatting |

---

## Phase 4: Presentation and Delivery (Weeks 14-16)

### Week 14: Presentation prep

- Person 5 creates slide deck structure
- Each person drafts slides for their section (3-5 slides each)
- Target: 20-25 slides total, 20-30 min presentation
- **Key demo**: live prediction on the dashboard — paste a speech, get a vote prediction

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
│   ├── texts/                    # Verslag XML transcripts (gitignored)
│   ├── analysis/                 # Analysis-ready datasets (gitignored)
│   │   ├── speech_vote_pairs.parquet   # MASTER DATASET
│   │   ├── speech_features.parquet     # NLP features per speech
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── external/                 # Election results, etc. (gitignored)
├── docs/
│   ├── DATASET.md                # Dataset documentation
│   ├── PROJECT_PLAN.md           # This file
│   ├── LITERATURE_REVIEW.md      # Academic literature
│   ├── NLP_LITERATURE.md         # NLP methods literature
│   ├── RESEARCH_DESIGN.md        # Research questions and design
│   ├── HYPOTHESES.md             # Formal hypotheses
│   └── paper/                    # LaTeX or Word paper
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 01b_data_quality.ipynb
│   ├── 02_voting_exploration.ipynb
│   ├── 03_baseline_predictions.ipynb
│   ├── 04_speech_embeddings.ipynb
│   ├── 05_vote_prediction.ipynb
│   ├── 06_error_analysis.ipynb
│   ├── 07_extended_models.ipynb
│   └── 08_hypothesis_tests.ipynb
├── src/
│   ├── fetch.py                  # API fetcher (exists)
│   ├── preprocess.py             # Data preprocessor (exists)
│   ├── fetch_verslagen.py        # Verslag XML downloader (new)
│   ├── parse_verslagen.py        # XML speech parser (new)
│   ├── link_speech_vote.py       # Speech-to-vote linker (new)
│   ├── utils.py                  # Helper functions (new)
│   ├── nlp/
│   │   ├── preprocess.py         # Text preprocessing
│   │   ├── features.py           # Speech feature extraction
│   │   ├── embeddings.py         # Speech embeddings
│   │   └── stance.py             # Stance/sentiment detection
│   ├── ml/
│   │   ├── features.py           # Feature matrix builder
│   │   ├── models.py             # Model training
│   │   └── evaluate.py           # Evaluation + SHAP
│   └── analysis/
│       └── voting.py             # Voting dataset builder
├── models/                       # Saved trained models (gitignored)
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
| 3 | Foundation complete | Speech-vote pairs dataset exists, baselines computed, lit review drafted |
| 6 | Core models trained | Vote prediction models A-D trained and evaluated, features extracted |
| 9 | All analysis complete | Error analysis done, hypotheses tested, dashboard functional |
| 13 | Paper complete | Full draft reviewed by all members |
| 16 | Final delivery | Paper submitted, presentation given, dashboard demo |

---

## Risk mitigation

| Risk | Mitigation |
|------|------------|
| **Verslag XML download is slow / rate-limited** | Start downloading Week 1. Prioritize plenaire vergaderingen (most votes happen there). Fall back to Activiteit metadata if transcripts unavailable. |
| **Speech-vote linking is ambiguous** | Start with plenaire stemmingen where the link is clearest (one debate → one vote session). Skip committee-level votes initially. |
| **Speech text doesn't add much over party label** | This is actually a finding! "Party membership explains 95% of votes" is a valid result about party discipline. Focus the paper on the 5% where speech matters (rebellions, close votes). |
| **Dutch NLP models perform poorly** | Use keyword-based features (stance words, topic keywords) as interpretable baseline. Layer ML on top only if it improves. |
| **Not enough speech-vote pairs** | Expand to fractie-level votes (party votes, not individual). Use ActiviteitActor (participation records) as weak proxy for speech. |
| **Mixed skill levels** | Person 4 and 5 don't need deep ML skills. Person 1 handles all data plumbing. Pair programming sessions weekly. |
| **Scope creep** | Core scope = vote prediction from speech. Everything else (rebellion detection, temporal analysis, per-topic models) is bonus. Stick to the 7 hypotheses. |

---

## What makes this project interesting

1. **It's testable**: We have ground truth (actual votes) to evaluate against.
2. **Speech adds signal**: If it doesn't, that itself is a finding about party discipline.
3. **Error analysis is the gold**: The cases where speech says one thing but the vote says another reveal party rebellions, strategic behavior, and the limits of deliberation.
4. **Live demo is compelling**: Paste a speech excerpt → get a vote prediction. Great for the presentation.
5. **Combines NLP + ML + political science**: Genuine interdisciplinary work.
6. **No one has done this for the Dutch parliament** with this dataset and modern NLP/ML methods.
