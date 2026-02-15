# Predicting Parliamentary Votes from Debate Speeches

**Can we predict how a Dutch parliamentarian will vote based on what they say in debate?**

This project uses data from the [Tweede Kamer](https://opendata.tweedekamer.nl/) (Dutch House of Representatives) to predict individual voting behavior (Voor/Tegen) from debate transcripts, party metadata, and legislative context. We combine a fine-tuned Dutch transformer model (RobBERT) with structured features in a stacked ensemble that achieves **71.9% test accuracy** on unseen post-2023 votes.

---

## Results at a Glance

| Model | Val Acc | Test Acc | Test F1 (macro) |
|---|---|---|---|
| Baseline (party majority) | 62.9% | 60.7% | 0.469 |
| Structural (XGBoost) | 60.4% | 53.3% | 0.517 |
| **RobBERT v2** (fine-tuned) | **71.6%** | **70.0%** | **0.696** |
| **Ensemble** (stacked) | **74.0%** | **71.9%** | **0.711** |

> The ensemble improves **+11.2 percentage points** over the party-majority baseline on test data.

### Per-Party Test Accuracy (Ensemble)

| Party | Accuracy | Samples | | Party | Accuracy | Samples |
|---|---|---|---|---|---|---|
| NSC | 87.5% | 3,813 | | DENK | 66.5% | 2,387 |
| VVD | 87.3% | 4,580 | | Volt | 66.3% | 1,568 |
| CDA | 82.9% | 3,183 | | PvdD | 64.7% | 2,463 |
| ChristenUnie | 77.5% | 3,180 | | GroenLinks-PvdA | 62.3% | 4,275 |
| SGP | 76.2% | 3,198 | | PvdA | 59.6% | 443 |
| GroenLinks | 74.0% | 492 | | Groep Van Haga | 56.1% | 360 |
| BBB | 73.6% | 3,385 | | BIJ1 | 57.3% | 117 |
| D66 | 72.9% | 4,048 | | PVV | 51.4% | 2,992 |
| SP | 71.5% | 3,891 | | FVD | 51.1% | 1,920 |
| JA21 | 67.6% | 1,460 | | | | |

**Key insight**: Coalition/governing parties (VVD, NSC, CDA, CU) are highly predictable (77-87%), while populist opposition parties (PVV, FVD) are near coin-flip (~51%), suggesting less consistent voting patterns relative to their rhetoric.

---

## Dataset

- **Source**: Tweede Kamer Open Data API (OData v4) + Verslag XML transcripts
- **Speech-vote pairs**: ~153,000 linked records
- **Temporal split**: Train <= 2021 (92,919) | Val = 2022 (12,414) | Test >= 2023 (48,021)
- **Classes**: Voor (For) / Tegen (Against) — binary classification
- **Data chain**: `Vergadering` -> `Verslag` (transcript XML) -> speech segments; `Activiteit` -> `Agendapunt` -> `Besluit` -> `Stemming` (votes). Speeches and votes are linked through person + agenda item matching.

---

## Model Architecture

### Pipeline Overview

```
Raw Data (OData API + XML)
    |
    v
Feature Engineering
    |-- Topic clustering (TF-IDF + KMeans on besluit_tekst)
    |-- Historical features (party-domain voting rates, speaker loyalty)
    |-- Structural features (party, coalition status, zaak_soort, etc.)
    |
    v
+-----------------------------------+    +-----------------------------------+
|  Structural Model (XGBoost)       |    |  RobBERT v2 (Transformer)        |
|  Features: party, topic, coalition|    |  Input: [party] </s> [besluit]   |
|  zaak_soort, historical rates,    |    |         </s> [topic] </s>        |
|  speaker loyalty                  |    |         [speech_text]            |
+-----------------------------------+    +-----------------------------------+
    |                                        |
    v                                        v
    P(structural)                       P(robbert)
          \                               /
           \                             /
            +---------------------------+
            |  Ensemble Meta-Learner    |
            |  (Logistic Regression)    |
            |  Features: P(struct),     |
            |  P(robbert), agreement,   |
            |  confidence delta,        |
            |  coalition, party rates   |
            +---------------------------+
                       |
                       v
                 Final Prediction
                 (Voor / Tegen)
```

### RobBERT v2 — Fine-Tuned Dutch Transformer

- **Base model**: [DTAI-KULeuven/robbert-2023-dutch-base](https://huggingface.co/DTAI-KULeuven/robbert-2023-dutch-base) (125M params)
- **Input format**: `[fractie] </s> [besluit_tekst] </s> [topic] </s> [speech_text]` (max 512 tokens)
- **Classifier head**: 3-layer MLP (768 -> 256 -> 64 -> 2) with GELU + Dropout(0.3)
- **Training techniques**:
  - **Focal loss** (gamma=2.0) — focuses on hard/uncertain examples
  - **Progressive unfreezing** — starts with frozen encoder, gradually unfreezes top layers
  - **Encoder detaching** — `torch.no_grad()` on frozen encoder for 6x speedup in early epochs
  - **Dynamic sample reweighting** — per-epoch confidence scoring, upweighting misclassified samples
  - **Gradient accumulation** (2 steps) + **gradient checkpointing** (dynamic)
  - **Early stopping** (patience=4)

### Structural Model (XGBoost)

Features: fractie one-hot, topic cluster, zaak_soort, kabinetsappreciatie, is_coalition, party_domain_voor_rate, party_recent_voor_rate, speaker_topic_loyalty.

### Ensemble (Stacked Meta-Learner)

Combines structural and RobBERT probability outputs with agreement indicators, confidence deltas, and raw features. Trained as logistic regression on the validation set.

---

## Project Structure

```
.
├── app/
│   └── dashboard.py              # Streamlit dashboard (overview, explorer, model results, demo)
├── data/
│   ├── raw/                      # Raw JSON from OData API (gitignored)
│   ├── processed/                # Clean Parquet files (gitignored)
│   ├── analysis/                 # speech_vote_pairs.parquet, train/val/test splits (gitignored)
│   └── texts/                    # Verslag XML transcripts (gitignored)
├── docs/
│   ├── DATASET.md                # Entity documentation for 35+ data types
│   ├── PROJECT_PLAN.md           # Full 16-week project plan with team roles
│   ├── Besluit_README.md         # Decision data documentation
│   └── DATA_SAMPLES.md           # Example data records
├── models/                       # Trained model checkpoints (gitignored, ~475MB each)
│   ├── robbert_v2/               # Best RobBERT checkpoint
│   └── robbert_v2_checkpoint_epoch5/
├── notebooks/
│   └── 06_model_interpretability.ipynb  # SHAP analysis and feature importance
├── outputs/
│   └── overnight_report.txt      # Latest training run results
├── scripts/
│   ├── overnight_run.py          # Full training pipeline orchestrator
│   ├── train_robbert.py          # Standalone RobBERT training
│   ├── eval_robbert.py           # Model evaluation
│   ├── complete_pipeline.py      # Resume pipeline from checkpoint
│   └── quick_eval.py             # Fast model evaluation
├── src/
│   ├── fetch.py                  # OData API fetcher (pagination, retries)
│   ├── preprocess.py             # Raw data cleaning and normalization
│   ├── fetch_verslagen.py        # Verslag XML downloader
│   ├── parse_verslagen.py        # XML speech segment parser
│   ├── link_speech_vote.py       # Speech-to-vote linker (core data join)
│   ├── build_speech_dataset.py   # Full data pipeline orchestrator
│   ├── ml/
│   │   ├── models.py             # All model implementations (1400+ lines)
│   │   ├── features.py           # Feature engineering pipeline
│   │   └── embeddings.py         # Sentence-transformer embeddings
│   └── nlp/
│       └── preprocess.py         # Dutch text preprocessing
├── config.yaml                   # Pipeline configuration
├── pipeline.py                   # Main CLI entry point
├── requirements.txt              # Python dependencies
└── Makefile                      # One-command team workflow
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended for RobBERT training)
- ~20GB disk space for data + models

### Installation

```bash
# Clone
git clone https://github.com/gerritsxd/qweerty.git
cd qweerty

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Fetching Data

```bash
# Full pipeline: fetch all 35+ entities from OData API + preprocess
python pipeline.py

# Build speech-vote dataset (requires Verslag XMLs)
python src/build_speech_dataset.py
```

### Training Models

```bash
# Full overnight pipeline (topic clustering -> features -> structural -> RobBERT -> ensemble)
python scripts/overnight_run.py

# Or train RobBERT standalone
python scripts/train_robbert.py
```

### Running the Dashboard

```bash
streamlit run app/dashboard.py
```

---

## Training Pipeline Details

The `scripts/overnight_run.py` orchestrator runs the full pipeline:

1. **Load data** — loads `speech_vote_pairs.parquet`, builds train/val/test splits
2. **Topic clustering** — TF-IDF + KMeans on besluit_tekst and agendapunt_onderwerp
3. **Historical features** — party-domain voting rates, speaker-topic loyalty scores
4. **Structural model** — trains XGBoost on structured features
5. **RobBERT v2 training** — fine-tunes the transformer with progressive unfreezing
6. **Ensemble stacking** — trains meta-learner on structural + RobBERT outputs
7. **Evaluation report** — writes per-model accuracy, classification report, per-party breakdown

Each step is crash-resilient (cached intermediate results as `.pkl` files for resume).

### RobBERT Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 15 (early stop patience=4) |
| Batch size | 32 |
| Max sequence length | 512 tokens |
| Gradient accumulation | 2 steps |
| Unfreezing schedule | Epoch 0: 0 layers, 2: 2, 4: 4, 7: 6, 10: 8 |
| Loss function | Focal loss (gamma=2.0) |
| Scoring subset | 25,000 samples (for speed) |
| Truncation | besluit: 500 chars, topic: 300, speech: 3,000 |

---

## Key Findings

1. **Speech text matters**: RobBERT (70.0% test) significantly outperforms the party-majority baseline (60.7%), showing that what MPs say carries predictive signal beyond party membership alone.

2. **Ensemble is best**: Combining structural features with the transformer yields the best results (71.9%), suggesting the two approaches capture complementary information.

3. **Government parties are predictable**: VVD (87.3%), NSC (87.5%), CDA (82.9%) follow consistent patterns. Their speeches align closely with their votes.

4. **Populist parties are unpredictable**: PVV (51.4%) and FVD (51.1%) are nearly random, suggesting their voting behavior is less correlated with speech content or follows different dynamics.

5. **The model is well-calibrated on confident predictions**: When the model is >80% confident, it achieves 95.8% accuracy. Low-confidence predictions (<60%) only reach 58.4%.

---

## Dependencies

Core: `pandas`, `pyarrow`, `scikit-learn`, `xgboost`
NLP: `transformers`, `torch`, `sentence-transformers`, `accelerate`
Viz: `matplotlib`, `seaborn`, `streamlit`, `shap`

See `requirements.txt` for full list with versions.

---

## Team & Workflow

See `docs/PROJECT_PLAN.md` for the full 16-week plan, team roles, hypotheses, and milestone schedule.

**Data is never committed to git.** Everyone runs the pipeline (`python pipeline.py`) to reproduce identical data locally. Only code, config, and documentation are tracked.

---

## API Reference

- Portal: https://opendata.tweedekamer.nl/
- OData API: https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0
- Information model: https://opendata.tweedekamer.nl/documentatie/informatiemodel
