# Tweede Kamer Speech-to-Vote — Complete Codebase Reference

**Explicit explanation of every module, model, script, and pipeline in the repository.**

---

## 1. Repository Structure

```
poop/
├── app/
│   └── dashboard.py              # Streamlit web app
├── config.yaml                    # Pipeline configuration
├── pipeline.py                    # Main OData fetch + preprocess entry point
├── visualize.py                   # Matplotlib dashboard (static PNG)
├── requirements.txt               # Python dependencies
├── Makefile                       # Convenience targets
├── data/
│   ├── raw/                       # Raw JSON from API (gitignored)
│   ├── processed/                 # Parquet tables (gitignored)
│   ├── texts/verslagen/           # Verslag XML transcripts (gitignored)
│   └── analysis/                  # speech_vote_pairs, train/val/test (gitignored)
├── docs/
│   ├── CODEBASE_REFERENCE.md      # This file
│   ├── DATASET.md                 # Entity documentation
│   ├── PROJECT_PLAN.md            # Full project plan
│   └── ...
├── notebooks/                     # Jupyter notebooks for exploration
├── outputs/                       # overnight_report.txt
├── models/                        # Saved RobBERT checkpoints (gitignored)
├── scripts/                       # Standalone scripts
└── src/
    ├── fetch.py                   # OData API fetcher
    ├── preprocess.py              # JSON → Parquet preprocessor
    ├── fetch_verslagen.py         # Verslag transcript fetcher
    ├── parse_verslagen.py         # XML → speech records parser
    ├── link_speech_vote.py        # Speech-to-vote linker
    ├── build_speech_dataset.py    # Orchestrates full speech pipeline
    ├── nlp/
    │   └── preprocess.py          # Text preprocessing, stance keywords
    └── ml/
        ├── features.py            # Feature extraction (topics, history, etc.)
        ├── embeddings.py          # Sentence embeddings (optional)
        └── models.py              # All prediction models
```

---

## 2. Data Pipeline (OData → Parquet)

### 2.1 `pipeline.py` — Main entry point

**Usage:**
```bash
python pipeline.py                    # Full: fetch + preprocess
python pipeline.py --fetch-only       # Only fetch raw JSON
python pipeline.py --preprocess-only  # Only preprocess (needs raw)
python pipeline.py --entities Persoon Fractie Stemming  # Specific entities
python pipeline.py --summary          # Print summary table
python pipeline.py --list-entities    # List all 38 entity types
```

**What it does:**
1. Loads `config.yaml`
2. Calls `TweedeKamerFetcher.fetch_all()` — fetches all enabled entities from the OData API
3. Calls `TweedeKamerPreprocessor.preprocess_all()` — cleans, normalizes, exports to Parquet
4. Generates `_summary.csv` with row counts per entity

---

### 2.2 `src/fetch.py` — TweedeKamerFetcher

**Purpose:** Fetch all records from the Tweede Kamer OData v4 API with pagination.

**Key methods:**
- `fetch_entity(entity_name)` — Paginated GET, follows `@odata.nextLink`, returns list of dicts
- `save_raw(entity_name, records)` — Saves to `data/raw/{Entity}.json`
- `fetch_all(entities)` — Loops over config entities, fetches and saves each

**API URL pattern:** `{base_url}/{Entity}?$filter=Verwijderd eq false&$format=application/json;odata.metadata=none`

**Retry logic:** Exponential backoff on 429, 5xx, connection errors (max 5 retries).

---

### 2.3 `src/preprocess.py` — TweedeKamerPreprocessor

**Purpose:** Load raw JSON, clean, normalize, export to Parquet/CSV.

**Steps per entity:**
1. Load JSON from `data/raw/{Entity}.json`
2. Drop `Verwijderd` column (always false)
3. Standardize `Id` column (lowercase)
4. Normalize strings (strip, collapse whitespace)
5. Parse datetime columns (auto-detect by regex: `Datum`, `GewijzigdOp`, `Van`, etc.)
6. Drop all-null columns
7. Save to `data/processed/{Entity}.parquet`

---

## 3. Speech-to-Vote Pipeline

### 3.1 `src/build_speech_dataset.py` — Orchestrator

**Usage:**
```bash
python -m src.build_speech_dataset              # Full pipeline
python -m src.build_speech_dataset --skip-fetch # Skip API (XMLs exist)
python -m src.build_speech_dataset --skip-parse # Skip parsing (speeches.parquet exists)
python -m src.build_speech_dataset --stats      # Stats only
```

**Steps:**
1. **Fetch** — `fetch_verslagen.py`: get Verslag metadata, download XMLs
2. **Parse** — `parse_verslagen.py`: extract speeches from XML
3. **Link** — `link_speech_vote.py`: link speeches to votes, output `speech_vote_pairs.parquet`

---

### 3.2 `src/fetch_verslagen.py` — Verslag transcript fetcher

**Purpose:** Fetch debate transcript metadata and download XML content.

**Steps:**
1. GET `{base_url}/Verslag?$filter=Verwijderd eq false` — paginated metadata
2. Save to `data/raw/Verslag.json`
3. For each Vergadering, pick **best** Verslag (priority: Gecorrigeerd > Ongecorrigeerd > Tussenpublicatie > Casco)
4. Download XML via `GET {base_url}/Verslag/{Id}/resource` → `data/texts/verslagen/{Id}.xml`

**Options:**
- `--limit N` — max verslagen to fetch
- `--skip-xml` — metadata only
- `--only-full-text` — skip Casco (skeleton, no speech text)
- `--redownload` — overwrite existing XMLs

---

### 3.3 `src/parse_verslagen.py` — XML parser

**Purpose:** Parse Verslag XML into structured speech records.

**XML structure (from Tweede Kamer):**
- Root: `<vlosCoreDocument>`
- `<vergadering>` — session metadata (datum, vergaderjaar, titel)
- `<activiteit>` — agenda section (Vragenuur, Opening, etc.)
- `<activiteithoofd>` — sub-agenda item with `<zaken>` (Zaak references)
- `<activiteitdeel soort="Spreekbeurt">` — one speech turn
- `<spreker>` — `objectid` = Persoon_Id, `fractie`, `achternaam`, etc.
- `<woordvoerder>` → `<tekst>` → `<alinea>` → `<alineaitem>` — actual speech text

**Output columns:** `verslag_id`, `vergadering_id`, `vergaderjaar`, `datum`, `activiteit_id`, `activiteit_soort`, `activiteit_onderwerp`, `activiteithoofd_onderwerp`, `zaak_ids`, `zaak_soorten`, `persoon_id`, `spreker_soort`, `fractie`, `achternaam`, `voornaam`, `spreker_soort`, `is_voorzitter`, `is_interruptie`, `speech_text`, `speech_text_clean`, `speech_start`, `speech_end`

**`speech_text_clean`:** Removes speaker attribution line (e.g. "Mevrouw Lodders (VVD):").

---

### 3.4 `src/link_speech_vote.py` — Speech-to-vote linker

**Purpose:** Link each speech to the corresponding vote(s).

**Linking strategies:**
1. **Strategy 1:** Direct `activiteit_id` match — XML `<activiteit objectid>` → `Activiteit.Id` → `Agendapunt` → `Besluit` → `Stemming` (rarely matches)
2. **Strategy 2:** Date matching — same-day speeches linked to Stemmingen on that date. For each speech: find speaker's vote by `Persoon_Id` (individual) or `ActorFractie` (faction-level)

**Vote lookup:**
- Individual votes: `Persoon_Id` in Stemming (Hoofdelijk)
- Faction votes: `ActorFractie` in Stemming (Met handopsteken)

**Output:** `speech_vote_pairs.parquet` with columns: `speech_text`, `fractie`, `persoon_id`, `vote` (Voor/Tegen/Niet deelgenomen), `besluit_id`, `besluit_tekst`, `agendapunt_onderwerp`, `datum`, etc.

**Splits:** `train.parquet` (≤2021), `val.parquet` (2022), `test.parquet` (≥2023).

---

## 4. NLP Module

### 4.1 `src/nlp/preprocess.py`

**Functions:**
- `clean_speech_text(text)` — Remove speaker attribution lines, normalize whitespace
- `tokenize_simple(text)` — Regex word tokenization (no spaCy)
- `split_sentences(text)` — Split on `. ! ?`
- `preprocess_pipeline(text)` — Full clean + normalize
- `count_stance_keywords(text)` — Returns `(n_voor, n_tegen)` using Dutch keywords:
  - **Voor:** steun, voor, stem voor, aanvaard, aannemen
  - **Tegen:** tegen, verwerp, onaanvaardbaar, stem tegen, afwijzen, ontraden

---

## 5. ML Module — Models

### 5.1 `src/ml/features.py` — Feature extraction

**Functions:**

| Function | Purpose |
|----------|---------|
| `load_pairs()` | Load `speech_vote_pairs.parquet` |
| `get_train_val_test(df)` | Temporal split: train ≤2021, val 2022, test ≥2023 |
| `build_basic_features(df)` | Add `speech_length`, `n_voor_kw`, `n_tegen_kw` |
| `enrich_with_zaak_features(df)` | Match `agendapunt_onderwerp` to Zaak → `kabinetsappreciatie`, `zaak_soort` |
| `add_enhanced_features(train, val, test)` | Add `speaker_loyalty`, `speech_position`, `is_coalition` |
| `cluster_topics(train, val, test)` | TF-IDF + KMeans on `besluit_tekst` + `agendapunt_onderwerp` → `topic_cluster` (0..19) |
| `build_historical_features(train, val, test)` | Add `party_domain_voor_rate`, `party_domain_vote_count`, `party_recent_voor_rate`, `speaker_topic_loyalty` |

**Coalition parties:** Hardcoded in `_COALITION_PARTIES` by year (e.g. 2024: PVV, VVD, NSC, BBB).

---

### 5.2 `src/ml/models.py` — All prediction models

#### Baseline 1: Party majority only

```python
train_baseline_party(train)  # Returns dict: {party -> majority vote}
predict_baseline_party(model, df)  # For each row: predict majority of that party
```

**Logic:** Predict the most common vote for that party in training data.

---

#### Model A: Party + TF-IDF of speech (Logistic Regression)

```python
train_model_a(train, max_features=5000, use_besluit_tfidf=True, ...)
predict_model_a(model, df)
```

**Features:**
- Party one-hot
- TF-IDF of `speech_text` (5000 features, unigrams)
- Optional: TF-IDF of `besluit_tekst`, `speech_length`, stance keywords, `speech_position`, `speaker_loyalty`, `kabinetsappreciatie`, `zaak_soort`, `is_coalition`

**Classifier:** LogisticRegression with `class_weight="balanced"`.

---

#### Model variants on same features (XGBoost, Random Forest, Gradient Boosting)

```python
train_model_xgb(train, **kwargs)   # XGBoost on Model A features
train_model_rf(train, **kwargs)   # Random Forest
train_model_gb(train, **kwargs)   # GradientBoostingClassifier
```

---

#### Structural model (XGBoost, no speech text)

```python
train_structural_model(train, val, test)
predict_structural_model(model, df)
predict_proba_structural_model(model, df)  # Returns Voor probability
```

**Features:**

- `fractie` one-hot
- `topic_cluster` one-hot
- `zaak_soort` one-hot
- `kabinetsappreciatie` one-hot
- `is_coalition` (0/1)
- `speaker_loyalty` (0–1)
- `speech_position` (0–1)
- `party_domain_voor_rate`, `party_domain_vote_count`
- `party_recent_voor_rate`
- `speaker_topic_loyalty`

**Classifier:** XGBClassifier, 500 estimators, max_depth 6.

---

#### RobBERT v2 (fine-tuned transformer)

```python
train_model_robbert(train, val, epochs=15, batch_size=32, max_length=512, ...)
predict_model_robbert(model_dict, df)
predict_proba_model_robbert_batch(model_dict, df)  # Voor probability per row
```

**Input format:** `[party] </s> [besluit_tekst] </s> [topic] </s> [speech]`
- Truncation: besluit 500 chars, topic 300 chars, speech 3000 chars (configurable)

**Architecture:**
- Base: `DTAI-KULeuven/robbert-2023-dutch-base` (125M params)
- Classifier: `768 → 256 (GELU, Dropout) → 64 (GELU, Dropout) → 2`

**Training:**
- **Focal loss**: `FL = -alpha_t * (1 - p_t)^gamma * log(p_t)` — focuses on hard examples
- **Progressive unfreezing**: Epoch 0–2: classifier only; 3–4: unfreeze 2 layers; 5–7: 4 layers; 8+: 6 layers
- **Encoder detaching**: When encoder frozen, skip backward pass (6× speedup)
- **Sample reweighting**: Per epoch, reweight samples by confidence (mastered → 0.3, uncertain → 1.5, wrong → 2.0)
- **Gradient checkpointing**: When encoder unfrozen, to save memory
- **Early stopping**: Patience 4 epochs
- **Checkpoint**: Save every 5 epochs

**Returns:** Best model (by val accuracy), not final epoch.

---

#### Ensemble (stacked meta-learner)

```python
train_ensemble_stacked(val, proba_struct, proba_robbert, y_true, structural_model, train)
predict_ensemble_stacked(model, df, proba_struct, proba_robbert, structural_model)
```

**Meta-features:**
- `proba_struct` (structural model Voor probability)
- `proba_robbert` (RobBERT Voor probability)
- `agree` (1 if both predict same class)
- `conf_delta` (|proba_struct - proba_robbert|)
- Optional: `is_coalition`, `party_domain_voor_rate`, `topic_cluster`

**Meta-learner:** LogisticRegression, binary (Voor vs Tegen).

---

### 5.3 `src/ml/embeddings.py`

**Purpose:** Compute 768-dim embeddings for speech text (optional, not used in main pipeline).

```python
compute_speech_embeddings(texts)  # Uses RobBERT via sentence-transformers
ensure_embeddings(df, text_col="speech_text")  # With cache
```

---

## 6. Scripts

### 6.1 `scripts/overnight_run.py` — Full training pipeline

**Purpose:** Crash-resilient orchestrator for full model training.

**Steps:**
1. Load `speech_vote_pairs.parquet`, build features, split train/val/test
2. **Topic clustering** (cache: `overnight_topics.pkl`)
3. **Historical features** (cache: `overnight_historical.pkl`)
4. **Structural model** (cache: `overnight_structural.pkl`)
5. **RobBERT v2** — train, save to `models/robbert_v2/`
6. **Ensemble** — stack structural + RobBERT
7. **Report** — write `outputs/overnight_report.txt`

**RobBERT config:** 15 epochs, batch 32, max_length 512, fp16, gradient_checkpointing, unfreeze_schedule `{0:0, 2:2, 4:4, 7:6, 10:8}`, early_stopping 4, checkpoint every 5.

---

### 6.2 `scripts/complete_pipeline.py`

**Purpose:** Resume from checkpoint: load cached data + structural model + saved RobBERT, run predictions → ensemble → report. Use when overnight run froze mid-training.

---

### 6.3 `scripts/train_robbert.py`

**Purpose:** Standalone RobBERT training script.

```bash
python scripts/train_robbert.py --epochs 10 --batch_size 16 --fp16
```

---

### 6.4 `scripts/eval_robbert.py`, `scripts/eval_checkpoint.py`, `scripts/quick_eval.py`

**Purpose:** Evaluate trained RobBERT model on val/test set.

---

### 6.5 `scripts/explore_linking.py`, `explore_linking2.py`, `explore_linking3.py`

**Purpose:** Exploratory scripts for speech-vote linking logic.

---

### 6.6 `scripts/check_zaak_besluit.py`

**Purpose:** Check Zaak–Besluit linkage.

---

### 6.7 `scripts/export_to_excel.py`, `scripts/build_sqlite.py`

**Purpose:** Export Parquet to CSV for Excel; build SQLite DB from Parquet.

---

## 7. App — Streamlit Dashboard

### `app/dashboard.py`

**Pages:**
1. **Overview** — KPI cards (pairs, speakers, parties, date range), vote distribution pie, temporal coverage bar, pairs per party
2. **Speech Explorer** — Filter by party, vote, year; search; expandable speech cards with vote color
3. **Model Results** — Baseline vs Model A comparison, confusion matrix
4. **Prediction Demo** — Paste speech text; live prediction (if model loaded)
5. **Methodology** — Pipeline and model descriptions

**Usage:** `streamlit run app/dashboard.py`

---

## 8. Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_audit.ipynb` | Data dictionary, column types, null rates |
| `01b_data_quality.ipynb` | Speech-vote pair coverage, quality checks |
| `02_voting_exploration.ipynb` | Stemming join, party cohesion, Voor/Tegen rates |
| `03_baseline_predictions.ipynb` | Party baseline, structural baseline |
| `05_vote_prediction.ipynb` | Model training, evaluation |
| `06_model_interpretability.ipynb` | SHAP, feature importance, attention |

---

## 9. Config — `config.yaml`

**API:**
- `base_url`: `https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0`
- `page_size`: 250
- `request_delay`: 0.25s
- `max_retries`: 5
- `retry_backoff`: 2.0

**Paths:**
- `raw_data`: `data/raw`
- `processed_data`: `data/processed`

**Entities:** 38 types (Persoon, Fractie, Stemming, Besluit, Activiteit, etc.) — each with `enabled`, `description`.

**Preprocessing:**
- `parse_dates`: true
- `drop_all_null_columns`: true
- `normalize_strings`: true
- `output_format`: parquet

---

## 10. Dependencies — `requirements.txt`

```
requests, pandas, pyarrow, pyyaml, tqdm
matplotlib, seaborn
scikit-learn, streamlit, shap
sentence-transformers, xgboost
transformers, torch, accelerate
```

---

## 11. Makefile Targets

| Target | Command |
|--------|---------|
| `make setup` | Create venv, install deps |
| `make data` | `pipeline.py` (fetch + preprocess) |
| `make fetch` | `pipeline.py --fetch-only` |
| `make process` | `pipeline.py --preprocess-only` |
| `make summary` | `pipeline.py --summary` |
| `make viz` | `visualize.py` |
| `make speeches` | `python -m src.build_speech_dataset` |
| `make speeches-quick` | `build_speech_dataset --skip-fetch` |
| `make speeches-stats` | `build_speech_dataset --stats` |
| `make clean` | Delete raw + processed data |
| `make nuke` | Delete data + venv |

---

## 12. visualize.py — Static Dashboard

**Purpose:** Generate `dashboard.png` — multi-panel matplotlib figure.

**Panels:**
- A: Dataset sizes (horizontal bar per entity)
- B: Current party seats (top 15)
- C: Gender distribution (pie)
- D: Meetings per year (stacked by Soort)
- E: Promises by status (donut)
- F: Gifts per year (line)
- G: Top travel destinations
- H: Promises by ministry
- I: Birth decade distribution
- J: Dossiers by Kamer
- K: Previous careers (top 12)

**Requires:** Processed Parquet files in `data/processed/`.
