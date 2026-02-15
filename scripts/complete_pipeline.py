#!/usr/bin/env python3
"""
Complete the overnight pipeline from the point where it froze.
Loads cached data + structural model + saved RobBERT v2, then runs:
  predictions -> ensemble -> final report.
"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"
REPORT_PATH = OUTPUTS / "overnight_report.txt"
RESULTS = {}

start_time = datetime.now()


def log(msg: str):
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def main():
    import pickle
    import torch
    from src.ml.features import (
        load_pairs,
        get_train_val_test,
        build_basic_features,
        add_enhanced_features,
    )
    from src.ml.models import (
        train_baseline_party,
        predict_baseline_party,
        predict_structural_model,
        predict_proba_structural_model,
        load_model_robbert,
        predict_model_robbert,
        predict_proba_model_robbert_batch,
        train_ensemble_stacked,
        predict_ensemble_stacked,
        evaluate,
    )

    # --- Load & prepare data (reuse cached topic + historical features) ---
    log("\n[Completion] Loading data from cache...")
    pkl_topics = OUTPUTS / "overnight_topics.pkl"
    pkl_hist = OUTPUTS / "overnight_historical.pkl"
    pkl_struct = OUTPUTS / "overnight_structural.pkl"

    if pkl_hist.exists():
        with open(pkl_hist, "rb") as f:
            train, val, test = pickle.load(f)
        log("  Loaded train/val/test from overnight_historical.pkl (includes topics + historical)")
    elif pkl_topics.exists():
        with open(pkl_topics, "rb") as f:
            train, val, test = pickle.load(f)
        log("  Loaded train/val/test from overnight_topics.pkl")
    else:
        log("  No cached data found, loading from scratch...")
        df = load_pairs()
        df = df[df["datum"].notna()]
        df = build_basic_features(df)
        train, val, test = get_train_val_test(df)
        train = train[train["vote"].isin(["Voor", "Tegen"])]
        val = val[val["vote"].isin(["Voor", "Tegen"])]
        test = test[test["vote"].isin(["Voor", "Tegen"])]
        train, val, test = add_enhanced_features(train, val, test)

    # Filter to Voor/Tegen (should already be filtered but just in case)
    train = train[train["vote"].isin(["Voor", "Tegen"])]
    val = val[val["vote"].isin(["Voor", "Tegen"])]
    test = test[test["vote"].isin(["Voor", "Tegen"])]
    log(f"  Data: train={len(train):,} val={len(val):,} test={len(test):,}")

    # --- Baseline ---
    log("\n[Completion] Computing baselines...")
    bl = train_baseline_party(train)
    pred_bl_val = predict_baseline_party(bl, val)
    pred_bl_test = predict_baseline_party(bl, test)
    r_bl_val = evaluate(val["vote"].values, pred_bl_val)
    r_bl_test = evaluate(test["vote"].values, pred_bl_test)
    RESULTS["baseline"] = {"val": r_bl_val, "test": r_bl_test, "pred_val": pred_bl_val, "pred_test": pred_bl_test}
    log(f"  Baseline: val={r_bl_val['accuracy']*100:.1f}% test={r_bl_test['accuracy']*100:.1f}%")

    # --- Structural model ---
    structural_model = None
    if pkl_struct.exists():
        with open(pkl_struct, "rb") as f:
            structural_model = pickle.load(f)
        log("\n[Completion] Loaded structural model from cache")
        pred_struct_val = predict_structural_model(structural_model, val)
        pred_struct_test = predict_structural_model(structural_model, test)
        proba_struct_val = predict_proba_structural_model(structural_model, val)
        proba_struct_test = predict_proba_structural_model(structural_model, test)
        r_struct_val = evaluate(val["vote"].values, pred_struct_val)
        r_struct_test = evaluate(test["vote"].values, pred_struct_test)
        RESULTS["structural"] = {
            "val": r_struct_val, "test": r_struct_test,
            "pred_val": pred_struct_val, "pred_test": pred_struct_test,
            "proba_val": proba_struct_val, "proba_test": proba_struct_test,
        }
        log(f"  Structural: val={r_struct_val['accuracy']*100:.1f}% test={r_struct_test['accuracy']*100:.1f}%")
    else:
        log("  WARNING: No structural model cache found, skipping.")

    # --- RobBERT v2 ---
    robbert_path = ROOT / "models" / "robbert_v2"
    robbert_model = None
    log("\n[Completion] Loading saved RobBERT v2 model...")
    try:
        robbert_model = load_model_robbert(str(robbert_path))
        log("  RobBERT v2 model loaded successfully")
    except Exception as e:
        log(f"  ERROR loading RobBERT: {e}")
        traceback.print_exc()

    if robbert_model is not None:
        log("[Completion] Running RobBERT predictions on val set...")
        pred_rob_val = predict_model_robbert(robbert_model, val, max_length=512)
        log("[Completion] Running RobBERT predictions on test set...")
        pred_rob_test = predict_model_robbert(robbert_model, test, max_length=512)
        log("[Completion] Computing RobBERT probabilities (val)...")
        proba_rob_val = predict_proba_model_robbert_batch(robbert_model, val, max_length=512)
        log("[Completion] Computing RobBERT probabilities (test)...")
        proba_rob_test = predict_proba_model_robbert_batch(robbert_model, test, max_length=512)
        r_rob_val = evaluate(val["vote"].values, pred_rob_val)
        r_rob_test = evaluate(test["vote"].values, pred_rob_test)
        RESULTS["robbert"] = {
            "val": r_rob_val, "test": r_rob_test,
            "pred_val": pred_rob_val, "pred_test": pred_rob_test,
            "proba_val": proba_rob_val, "proba_test": proba_rob_test,
        }
        log(f"  RobBERT v2: val={r_rob_val['accuracy']*100:.1f}% test={r_rob_test['accuracy']*100:.1f}%")

    # --- Ensemble stacking ---
    if structural_model is not None and robbert_model is not None:
        log("\n[Completion] Training ensemble stacker...")
        try:
            ensemble_model = train_ensemble_stacked(
                val,
                RESULTS["structural"]["proba_val"],
                RESULTS["robbert"]["proba_val"],
                val["vote"].values,
                structural_model,
                train,
            )
            if ensemble_model is not None:
                pred_ens_val = predict_ensemble_stacked(
                    ensemble_model, val,
                    RESULTS["structural"]["proba_val"],
                    RESULTS["robbert"]["proba_val"],
                    structural_model,
                )
                pred_ens_test = predict_ensemble_stacked(
                    ensemble_model, test,
                    RESULTS["structural"]["proba_test"],
                    RESULTS["robbert"]["proba_test"],
                    structural_model,
                )
                r_ens_val = evaluate(val["vote"].values, pred_ens_val)
                r_ens_test = evaluate(test["vote"].values, pred_ens_test)
                RESULTS["ensemble"] = {"val": r_ens_val, "test": r_ens_test,
                                       "pred_val": pred_ens_val, "pred_test": pred_ens_test}
                log(f"  Ensemble: val={r_ens_val['accuracy']*100:.1f}% test={r_ens_test['accuracy']*100:.1f}%")
        except Exception as e:
            log(f"  ERROR in ensemble: {e}")
            traceback.print_exc()

    # --- Final report ---
    log("\n" + "=" * 60)
    write_final_report(RESULTS, val, test)
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\nCompletion runtime: {elapsed/60:.1f} minutes")
    log("Pipeline completion finished.")


def write_final_report(results: dict, val, test):
    from sklearn.metrics import classification_report, confusion_matrix

    lines = []
    lines.append("FINAL RESULTS")
    lines.append("=" * 60)
    lines.append(f"{'Model':<25} {'Val Acc':>10} {'Test Acc':>10} {'Test F1':>10}")
    lines.append("-" * 55)

    for name, data in results.items():
        if isinstance(data, dict) and "val" in data and "test" in data:
            rv = data["val"]
            rt = data["test"]
            lines.append(f"{name:<25} {rv['accuracy']*100:>9.1f}% {rt['accuracy']*100:>9.1f}% {rt.get('f1_macro', 0):>9.3f}")

    lines.append("-" * 55)
    candidates = [k for k, v in results.items() if isinstance(v, dict) and "test" in v]
    best_name = max(candidates, key=lambda k: results[k]["test"]["accuracy"], default="(none)") if candidates else "(none)"
    lines.append(f"Best model: {best_name}")

    if best_name in results and "pred_test" in results.get(best_name, {}):
        pred = results[best_name]["pred_test"]
        lines.append("\nTest classification report:")
        lines.append(classification_report(test["vote"].values, pred, digits=3))
        cm = confusion_matrix(test["vote"].values, pred, labels=["Tegen", "Voor"])
        lines.append("Confusion matrix (Tegen, Voor):")
        lines.append(str(cm))

        # Per-party accuracy breakdown
        if "fractie" in test.columns:
            test_copy = test.copy()
            test_copy["pred"] = pred
            test_copy["correct"] = test_copy["vote"] == test_copy["pred"]
            party_acc = test_copy.groupby("fractie").agg(
                total=("correct", "count"),
                correct=("correct", "sum"),
            )
            party_acc["accuracy"] = party_acc["correct"] / party_acc["total"]
            party_acc = party_acc[party_acc["total"] >= 5].sort_values("total", ascending=False)
            lines.append("\nPer-party test accuracy (>=5 samples):")
            for party, row in party_acc.head(20).iterrows():
                bar = "#" * int(row["accuracy"] * 20)
                lines.append(f"  {party:25s} {row['accuracy']*100:5.1f}% ({int(row['total']):5d}) |{bar}")

    lines.append("=" * 60)

    report = "\n".join(lines)
    log(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
