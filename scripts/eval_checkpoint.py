#!/usr/bin/env python3
"""Evaluate a specific RobBERT checkpoint against val and test sets."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pickle
from src.ml.models import (
    load_model_robbert,
    predict_model_robbert,
    predict_proba_model_robbert_batch,
    evaluate,
)
from sklearn.metrics import classification_report

OUTPUTS = ROOT / "outputs"

# Load data
print("Loading data...", flush=True)
with open(OUTPUTS / "overnight_historical.pkl", "rb") as f:
    train, val, test = pickle.load(f)
train = train[train["vote"].isin(["Voor", "Tegen"])]
val = val[val["vote"].isin(["Voor", "Tegen"])]
test = test[test["vote"].isin(["Voor", "Tegen"])]
print(f"Data: train={len(train):,} val={len(val):,} test={len(test):,}")

# Evaluate each checkpoint
checkpoints = {
    "robbert_v2 (epoch7-final)": ROOT / "models" / "robbert_v2",
    "robbert_v2_checkpoint_epoch5": ROOT / "models" / "robbert_v2_checkpoint_epoch5",
    "robbert_vote_classifier (prev-run)": ROOT / "models" / "robbert_vote_classifier",
}

for name, ckpt_path in checkpoints.items():
    if not (ckpt_path / "model.pt").exists():
        print(f"\n{name}: checkpoint not found, skipping")
        continue
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"  Path: {ckpt_path}")
    print("  Loading model...", flush=True)
    model = load_model_robbert(str(ckpt_path))
    
    print("  Predicting val...", flush=True)
    pred_val = predict_model_robbert(model, val, max_length=512)
    r_val = evaluate(val["vote"].values, pred_val)
    
    print("  Predicting test...", flush=True)
    pred_test = predict_model_robbert(model, test, max_length=512)
    r_test = evaluate(test["vote"].values, pred_test)
    
    print(f"\n  Val:  acc={r_val['accuracy']*100:.1f}% f1={r_val.get('f1_macro', 0):.3f}")
    print(f"  Test: acc={r_test['accuracy']*100:.1f}% f1={r_test.get('f1_macro', 0):.3f}")
    
    # Per-party test accuracy
    if "fractie" in test.columns:
        test_copy = test.copy()
        test_copy["pred"] = pred_test
        test_copy["correct"] = test_copy["vote"] == test_copy["pred"]
        party_acc = test_copy.groupby("fractie").agg(
            total=("correct", "count"),
            correct=("correct", "sum"),
        )
        party_acc["accuracy"] = party_acc["correct"] / party_acc["total"]
        party_acc = party_acc[party_acc["total"] >= 100].sort_values("total", ascending=False)
        print(f"\n  Per-party test accuracy (>=100 samples):")
        for party, row in party_acc.head(15).iterrows():
            bar = "#" * int(row["accuracy"] * 20)
            print(f"    {party:25s} {row['accuracy']*100:5.1f}% ({int(row['total']):5d}) |{bar}")
    
    # Free GPU memory
    import torch
    del model
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("Evaluation complete.")
