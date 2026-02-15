#!/usr/bin/env python3
"""Evaluate saved RobBERT model on test set."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.features import load_pairs, get_train_val_test, build_basic_features, add_enhanced_features
from src.ml.models import (
    load_model_robbert, predict_model_robbert, evaluate,
    train_baseline_party, predict_baseline_party,
)

print("Loading data...")
df = load_pairs()
df = df[df["datum"].notna()]
df = build_basic_features(df)
train, val, test = get_train_val_test(df)
train = train[train["vote"].isin(["Voor", "Tegen"])]
val = val[val["vote"].isin(["Voor", "Tegen"])]
test = test[test["vote"].isin(["Voor", "Tegen"])]

print(f"Val: {len(val):,} | Test: {len(test):,}")

# Baselines
majority = train["vote"].value_counts().idxmax()
bl = train_baseline_party(train)

print("\nLoading saved RobBERT model...")
model_dict = load_model_robbert()

print("Predicting on val...")
pred_val = predict_model_robbert(model_dict, val)
r_val = evaluate(val["vote"].values, pred_val)

print("Predicting on test...")
pred_test = predict_model_robbert(model_dict, test)
r_test = evaluate(test["vote"].values, pred_test)

# Baselines
maj_val = (val["vote"] == majority).mean()
maj_test = (test["vote"] == majority).mean()
bl_val = evaluate(val["vote"].values, predict_baseline_party(bl, val))
bl_test = evaluate(test["vote"].values, predict_baseline_party(bl, test))

print("\n" + "=" * 55)
print("            FINAL RESULTS")
print("=" * 55)
print(f"{'Model':<25} {'Val Acc':>10} {'Test Acc':>10} {'Test F1':>10}")
print("-" * 55)
print(f"{'Majority (always Voor)':<25} {maj_val*100:>9.1f}% {maj_test*100:>9.1f}%")
print(f"{'Party baseline':<25} {bl_val['accuracy']*100:>9.1f}% {bl_test['accuracy']*100:>9.1f}%")
print(f"{'RobBERT (iterative)':<25} {r_val['accuracy']*100:>9.1f}% {r_test['accuracy']*100:>9.1f}% {r_test['f1_macro']:>9.3f}")
print("-" * 55)
print(f"{'Improvement vs baseline':<25} {(r_val['accuracy']-bl_val['accuracy'])*100:>+9.1f}pp {(r_test['accuracy']-bl_test['accuracy'])*100:>+9.1f}pp")
print("=" * 55)

# Per-class breakdown
from sklearn.metrics import classification_report
print("\nTest set classification report:")
print(classification_report(test["vote"].values, pred_test, digits=3))
