# Game Theory as Complementary Signal: Implementation Plan

## Role in the Ensemble

The game-theoretic model provides a **principled party baseline** that complements:

| Model | Signal | Strength |
|-------|--------|----------|
| **Structural** | Party one-hot, topic, kabinetsappreciatie | Strong party identity |
| **RobBERT** | Speech text → vote intent | Captures speech overrides |
| **Game theory** | Policy + coalition + discipline + reciprocity | Captures *when* parties deviate |
| **Markov** | Historical co-voting patterns | Temporal consistency |

Game theory adds value when:
- Coalition discipline matters (coalition parties voting with government)
- Party discipline varies (high-discipline parties rarely rebel)
- Policy alignment is informative (CHES ideal points vs bill position)

---

## Phase 1: DPBD Discipline Integration ✓

**Goal:** Use 70 years of DPBD voting to inform party discipline.

**Changes:**
1. Add **discipline_utility** (w5): penalty for deviating from party line, scaled by `dpbd_consistency`
   - Party line = Voor if `dpbd_category_rate` (or `dpbd_voor_rate`) > 0.5 else Tegen
   - High-consistency parties (e.g. CDA 98%) get strong penalty for rebellion
   - Low-consistency parties (e.g. new/small) get weaker penalty

2. Use **dpbd_category_rate** for reciprocity when available (better domain match than `party_domain_voor_rate`)

**Files:** `src/game/payoffs.py`, `src/game/simulation.py`, `src/game/calibrate.py`

---

## Phase 2: Better Calibration ✓

**Goal:** Find payoff weights that maximize log-likelihood on training data.

**Changes:**
1. Add w5 to grid search
2. Expand grid (finer resolution on w1, w2)
3. Optional: scipy.optimize for continuous weights (future)

**Files:** `src/game/calibrate.py`

---

## Phase 3: Pipeline Integration ✓

**Goal:** Ensure game runs with DPBD features.

**Prerequisite:** `add_enhanced_features` must run before game (adds `dpbd_*` columns).  
The complete_pipeline loads from cache that includes DPBD when built with `add_enhanced_features`.

**No code change** if cache has DPBD. Fallback: game uses `party_domain_voor_rate` when `dpbd_*` missing.

---

## Phase 4 (Future): Speech-Informed Bill Positions

**Goal:** Use RobBERT hidden states to position bills in policy space from debate text.

**Idea:** Encode bill/agendapunt text with RobBERT → project to CHES dimensions via probe → use as bill position instead of topic centroid.

**Effort:** Medium. Requires trained probe from mechanistic interp.

---

## Phase 5 (Future): Strategic Interdependence

**Goal:** Model belief about coalition outcome; parties best-respond given beliefs.

**Idea:** Iterative best-response: each party computes P(bill passes) given others' expected votes; if pivotal, adjust; repeat until convergence.

**Effort:** High. Requires per-besluit aggregation of party votes.

---

## Success Metrics

- **Standalone:** Game-theoretic accuracy on val/test (target: beat party baseline 62.6%)
- **Ensemble:** Does adding proba_game improve stacked ensemble over structural+RobBERT only?
- **Ablation:** Compare ensemble with vs without game on same split

---

## Implementation Checklist

- [x] Plan document
- [x] discipline_utility in payoffs.py
- [x] dpbd_category_rate fallback for reciprocity
- [x] w5 in compute_payoffs_batch, simulation, calibrate
- [x] Run validation (game standalone ~66% with defaults; calibration with quick=True)
