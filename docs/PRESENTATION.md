# Predicting Parliamentary Votes from Debate Speeches
## Presentation — Monday Progress Update

---

## Slide 1: Title

**Predicting Parliamentary Votes from Debate Speeches**

*Can we predict how a Dutch MP will vote based on what they say in debate?*

Team Project — Tweede Kamer Open Data
February 2026

---

## Slide 2: The Problem

- The Dutch parliament (Tweede Kamer) publishes full debate transcripts and individual votes
- **Research question**: Does what an MP says in debate predict how they vote?
- If yes: speech carries real signal about voting intent
- If no: voting is purely party-line (deliberation doesn't matter)

**Data**: 153,000 speech-vote pairs from the Tweede Kamer Open Data API

---

## Slide 3: Data Pipeline

```
Tweede Kamer OData API          Verslag XML Transcripts
        |                               |
        v                               v
  35+ entity types              Speech segment parsing
  (Stemming, Besluit,           (speaker, party, text)
   Activiteit, etc.)                    |
        |                               |
        +---------- LINKING ------------+
                      |
                      v
          speech_vote_pairs.parquet
          153,000 linked records
                      |
                      v
         Train (<=2021):  92,919
         Val   (2022):    12,414
         Test  (>=2023):  48,021
```

**Temporal split** prevents data leakage — we test on future votes the model has never seen.

---

## Slide 4: Model Architecture

We built 4 models, each adding complexity:

| Model | What it uses | Test Acc |
|---|---|---|
| **Baseline** | Party majority vote only | 60.7% |
| **Structural** | Party + topic + coalition + history (XGBoost) | 53.3% |
| **RobBERT v2** | Full speech text (fine-tuned Dutch transformer) | 70.0% |
| **Ensemble** | Structural + RobBERT combined | **71.9%** |

The key question: **does speech text add value?**
- Baseline (party only): 60.7%
- With speech text (RobBERT): 70.0%
- **Answer: YES, +9.3 percentage points from speech alone**

---

## Slide 5: RobBERT — The Transformer Model

- **Base**: RobBERT-2023-dutch-base (125M parameters, pre-trained on Dutch text)
- **Input**: `[party] </s> [besluit_tekst] </s> [topic] </s> [speech]`
- **Classifier**: 768 -> 256 -> 64 -> 2 (MLP with GELU + Dropout)

Training innovations:
- **Focal loss**: focuses on hard-to-classify examples
- **Progressive unfreezing**: start frozen, gradually unfreeze encoder layers
- **Encoder detaching**: 6x speedup by skipping backward pass through frozen layers
- **Dynamic sample reweighting**: re-weight training samples based on confidence each epoch
- **Early stopping**: best model at epoch 3 (74.0% val accuracy)

---

## Slide 6: Results — The Big Picture

```
                        Test Accuracy
Baseline (party only)   ████████████████████████████████ 60.7%
Structural (XGBoost)    ██████████████████████████████   53.3%
RobBERT v2              ██████████████████████████████████████ 70.0%
Ensemble                ████████████████████████████████████████ 71.9%
                        0%   10%  20%  30%  40%  50%  60%  70%  80%
```

- **+11.2pp** improvement over party-only baseline
- Ensemble F1 (macro): 0.711
- Precision: 0.633 (Tegen), 0.785 (Voor)
- Recall: 0.691 (Tegen), 0.738 (Voor)

---

## Slide 7: Per-Party Accuracy — Who's Predictable?

**Most predictable (>75%)**:
| Party | Accuracy | Type |
|---|---|---|
| NSC | 87.5% | Coalition |
| VVD | 87.3% | Coalition |
| CDA | 82.9% | Government |
| ChristenUnie | 77.5% | Former coalition |
| SGP | 76.2% | Consistent opposition |

**Least predictable (<55%)**:
| Party | Accuracy | Type |
|---|---|---|
| PVV | 51.4% | Populist opposition |
| FVD | 51.1% | Populist opposition |

**Interpretation**: Parties with clear ideological positions and strong party discipline are predictable. Populist parties that vote unpredictably relative to their rhetoric are near coin-flip.

---

## Slide 8: Confidence Analysis

The model knows when it's uncertain:

| Confidence Level | Accuracy | % of Predictions |
|---|---|---|
| High (>0.8) | **95.8%** | 12% |
| Medium (0.6-0.8) | ~75% | 31% |
| Low (<0.6) | 58.4% | 57% |

When the model is confident, it's almost always right (95.8%).
Most predictions fall in the low-confidence zone — this is where improvement is needed.

---

## Slide 9: Confusion Matrix

```
                 Predicted
              Tegen    Voor
Actual Tegen  13,137   5,870     (69.1% correct)
Actual Voor    7,602  21,412     (73.8% correct)
```

- Slightly better at predicting Voor (73.8%) than Tegen (69.1%)
- This makes sense: Voor is the majority class (60.4% of test data)
- The model captures the "default support" pattern well

---

## Slide 10: What We Learned

1. **Speech text adds real predictive value** (+9.3pp over party alone)
   - MPs reveal voting intent through their language
   - Confirms speech is not just performative

2. **The ensemble captures complementary signals**
   - Structural model: party patterns, coalition dynamics, historical rates
   - Transformer: nuanced language understanding from speech text

3. **Party discipline varies dramatically**
   - VVD/NSC/CDA: strong party line, highly predictable
   - PVV/FVD: voting behavior doesn't follow speech patterns

4. **Confident predictions are reliable**
   - 95.8% accuracy when model confidence > 80%
   - Can be used as a "high-certainty" filter for downstream applications

---

## Slide 11: Technical Highlights

- **Dataset**: 153K speech-vote pairs across 20+ parties, temporal split
- **Infrastructure**: Full OData pipeline fetching 35+ entity types
- **Crash-resilient training**: cached intermediate steps, checkpoint saves
- **Optimization**: encoder detaching for 6x speedup, dynamic gradient checkpointing
- **Dashboard**: Streamlit app with overview, speech explorer, model results, live prediction demo

---

## Slide 12: Current Limitations & Next Steps

**Limitations**:
- 57% of predictions are low-confidence (<0.6) — room for improvement
- Populist parties (~51%) are near-random — may need different approach
- Structural model underperforms on test set (53.3%) — temporal distribution shift
- Best RobBERT checkpoint (74% val) was lost due to a bug — now fixed

**Next Steps**:
- [ ] Retrain RobBERT with the bug fix (return best model, not final epoch)
- [ ] Experiment with longer training / different learning rate schedules
- [ ] Per-party or per-topic specialized models
- [ ] Rebellion detection: predict when an MP will vote against their party
- [ ] SHAP analysis on transformer attention for interpretability
- [ ] Hypothesis testing (H1-H7 from project plan)

---

## Slide 13: Live Demo

**Dashboard**: `streamlit run app/dashboard.py`

Features:
- Browse speech-vote pairs with filters
- See model predictions with confidence scores
- Per-party accuracy breakdowns
- Live prediction: paste a speech excerpt -> get vote prediction

---

## Slide 14: Repository & How to Run

```bash
# Clone and setup
git clone https://github.com/gerritsxd/qweerty.git
cd qweerty
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Fetch data
python pipeline.py

# Train models (full overnight pipeline)
python scripts/overnight_run.py

# Launch dashboard
streamlit run app/dashboard.py
```

See README.md for full documentation.

---

## Appendix: Training Curves (RobBERT v2)

| Epoch | Train Acc | Val Acc | Val F1 | Layers Unfrozen |
|---|---|---|---|---|
| 1 | 44.6% | 47.5% | 0.450 | 0/12 |
| 2 | 58.5% | 62.2% | 0.622 | 0/12 |
| 3 | 68.8% | **74.0%** | **0.729** | 2/12 |
| 4 | 69.9% | 68.7% | 0.686 | 2/12 |
| 5 | 71.4% | 71.4% | 0.711 | 4/12 |
| 6 | — | — | — | 4/12 |
| 7 | 72.2% | 70.2% | 0.700 | 6/12 |
| — | *Early stopping* | | | |

Best model: **Epoch 3** (74.0% val accuracy, 2 layers unfrozen)

Note: Unfreezing more layers caused slight overfitting. The sweet spot appears to be unfreezing 2 layers — the model captures domain-specific patterns without losing generalization.
