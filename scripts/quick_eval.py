#!/usr/bin/env python3
"""Quick accuracy evaluation for all models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.features import load_pairs, get_train_val_test, build_basic_features, add_enhanced_features
from src.ml.models import (
    train_baseline_party, predict_baseline_party,
    train_model_a, predict_model_a,
    train_model_gb, train_model_rf, train_model_xgb,
    train_ensemble_two_stage, predict_ensemble_two_stage,
    evaluate,
)

MODEL_KW = dict(
    max_features=2000, ngram_range=(1, 1), min_df=1,
    use_besluit_tfidf=True, use_speech_position=True, use_speaker_loyalty=True,
    use_kabinetsappreciatie=True, use_zaak_soort=True, use_is_coalition=True,
)

def main():
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    print(f"Loading {sample} pairs...")
    df = load_pairs(sample=sample)
    df = df[df["datum"].notna()]
    df = build_basic_features(df)
    train, val, test = get_train_val_test(df)
    train = train[train["vote"].isin(["Voor", "Tegen"])]
    val = val[val["vote"].isin(["Voor", "Tegen"])]
    test = test[test["vote"].isin(["Voor", "Tegen"])]
    train, val, test = add_enhanced_features(train, val, test)
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}\n")

    results = []
    model_b = train_baseline_party(train)
    r_b_val = evaluate(val["vote"].values, predict_baseline_party(model_b, val))
    r_b_test = evaluate(test["vote"].values, predict_baseline_party(model_b, test))
    results.append(("Baseline", r_b_val["accuracy"], r_b_test["accuracy"]))

    model_a = train_model_a(train, **MODEL_KW)
    r_a_val = evaluate(val["vote"].values, predict_model_a(model_a, val))
    r_a_test = evaluate(test["vote"].values, predict_model_a(model_a, test))
    results.append(("Model A", r_a_val["accuracy"], r_a_test["accuracy"]))

    model_gb = train_model_gb(train, **MODEL_KW)
    r_gb_val = evaluate(val["vote"].values, predict_model_a(model_gb, val))
    r_gb_test = evaluate(test["vote"].values, predict_model_a(model_gb, test))
    results.append(("GradientBoost", r_gb_val["accuracy"], r_gb_test["accuracy"]))

    model_rf = train_model_rf(train, **MODEL_KW)
    r_rf_val = evaluate(val["vote"].values, predict_model_a(model_rf, val))
    r_rf_test = evaluate(test["vote"].values, predict_model_a(model_rf, test))
    results.append(("RandomForest", r_rf_val["accuracy"], r_rf_test["accuracy"]))

    try:
        model_xgb = train_model_xgb(train, **MODEL_KW)
        r_xgb_val = evaluate(val["vote"].values, predict_model_a(model_xgb, val))
        r_xgb_test = evaluate(test["vote"].values, predict_model_a(model_xgb, test))
        results.append(("XGBoost", r_xgb_val["accuracy"], r_xgb_test["accuracy"]))
    except Exception as e:
        print(f"XGBoost skipped: {e}")

    try:
        model_ens = train_ensemble_two_stage(train, confidence_threshold=0.7, **MODEL_KW)
        r_ens_val = evaluate(val["vote"].values, predict_ensemble_two_stage(model_ens, val))
        r_ens_test = evaluate(test["vote"].values, predict_ensemble_two_stage(model_ens, test))
        results.append(("Ensemble", r_ens_val["accuracy"], r_ens_test["accuracy"]))
    except Exception as e:
        print(f"Ensemble skipped: {e}")

    print("Model          | Val Acc  | Test Acc")
    print("---------------|----------|----------")
    for name, val_acc, test_acc in results:
        print(f"{name:14} | {val_acc*100:6.1f}%  | {test_acc*100:6.1f}%")

if __name__ == "__main__":
    main()
