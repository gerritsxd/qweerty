#!/usr/bin/env python3
"""
Overnight training pipeline: crash-resilient orchestrator.

Runs: topic clustering -> historical features -> structural model -> RobBERT v2 -> ensemble -> report.
Each step wrapped in try/except; on error logs and continues. Saves intermediate results for resume.
"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
REPORT_PATH = OUTPUTS / "overnight_report.txt"
RESULTS = {}


def log(msg: str, also_print: bool = True):
    """Append to report file and optionally print."""
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    if also_print:
        print(msg, flush=True)


def run_step(name: str, fn, *args, **kwargs):
    """Run a step, catch errors, log, continue."""
    log(f"\n{'='*60}\n[{datetime.now().isoformat()}] STEP: {name}\n{'='*60}")
    try:
        result = fn(*args, **kwargs)
        log(f"  OK: {name} completed successfully.")
        return result
    except Exception as e:
        log(f"  ERROR in {name}: {e}")
        log(traceback.format_exc(), also_print=False)
        return None


def main():
    start_time = datetime.now()
    # Clear or init report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Overnight Run started at {start_time.isoformat()}\n")
        f.write("=" * 60 + "\n")

    log("Overnight pipeline starting...")

    # --- Load data (shared) ---
    from src.ml.features import (
        load_pairs,
        get_train_val_test,
        build_basic_features,
        add_enhanced_features,
        cluster_topics,
        build_historical_features,
    )
    from src.ml.models import (
        train_baseline_party,
        predict_baseline_party,
        train_structural_model,
        predict_structural_model,
        predict_proba_structural_model,
        train_model_robbert,
        save_model_robbert,
        load_model_robbert,
        predict_model_robbert,
        predict_proba_model_robbert_batch,
        train_ensemble_stacked,
        predict_ensemble_stacked,
        evaluate,
    )

    df = run_step("Load data", lambda: load_pairs())
    if df is None:
        log("FATAL: Could not load data. Aborting.")
        return
    df = df[df["datum"].notna()]
    df = build_basic_features(df)
    train, val, test = get_train_val_test(df)
    train = train[train["vote"].isin(["Voor", "Tegen"])]
    val = val[val["vote"].isin(["Voor", "Tegen"])]
    test = test[test["vote"].isin(["Voor", "Tegen"])]
    train, val, test = add_enhanced_features(train, val, test)
    log(f"Data: train={len(train):,} val={len(val):,} test={len(test):,}")

    # --- Step 1: Topic clustering ---
    pkl1 = OUTPUTS / "overnight_topics.pkl"
    if pkl1.exists():
        import pickle
        with open(pkl1, "rb") as f:
            train, val, test = pickle.load(f)
        log("  (Resumed from overnight_topics.pkl)")
    else:
        result = run_step("Topic clustering", cluster_topics, train, val, test)
        if result is not None:
            train, val, test = result
            import pickle
            with open(pkl1, "wb") as f:
                pickle.dump((train, val, test), f)

    # --- Step 2: Historical features ---
    pkl2 = OUTPUTS / "overnight_historical.pkl"
    if pkl2.exists():
        import pickle
        with open(pkl2, "rb") as f:
            train, val, test = pickle.load(f)
        log("  (Resumed from overnight_historical.pkl)")
    else:
        result = run_step("Historical features", build_historical_features, train, val, test)
        if result is not None:
            train, val, test = result
            import pickle
            with open(pkl2, "wb") as f:
                pickle.dump((train, val, test), f)

    # --- Step 3: Structural model ---
    pkl3 = OUTPUTS / "overnight_structural.pkl"
    structural_model = None
    if pkl3.exists():
        import pickle
        with open(pkl3, "rb") as f:
            structural_model = pickle.load(f)
        log("  (Resumed structural model from overnight_structural.pkl)")
    else:
        structural_model = run_step("Structural model", train_structural_model, train, val, test)
        if structural_model is not None:
            import pickle
            with open(pkl3, "wb") as f:
                pickle.dump(structural_model, f)

    # Baselines
    bl = train_baseline_party(train)
    pred_bl_val = predict_baseline_party(bl, val)
    pred_bl_test = predict_baseline_party(bl, test)
    r_bl_val = evaluate(val["vote"].values, pred_bl_val)
    r_bl_test = evaluate(test["vote"].values, pred_bl_test)
    RESULTS["baseline"] = {"val": r_bl_val, "test": r_bl_test, "pred_val": pred_bl_val, "pred_test": pred_bl_test}

    # Structural predictions
    if structural_model is not None:
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

    # --- Step 4: RobBERT v2 ---
    robbert_path = ROOT / "models" / "robbert_v2"
    robbert_path.mkdir(parents=True, exist_ok=True)
    robbert_model = None
    robbert_result = run_step(
        "RobBERT v2 training",
        lambda: train_model_robbert(
            train, val,
            epochs=15,
            batch_size=32,
            max_length=512,
            save_path=str(robbert_path),
            gradient_checkpointing=True,
            score_subset=25000,
            unfreeze_schedule={0: 0, 2: 2, 4: 4, 7: 6, 10: 8},
            accum_steps=2,
            early_stopping_patience=4,
            checkpoint_every_n_epochs=5,
            truncation_chars={"besluit": 500, "topic": 300, "speech": 3000},
        ),
    )
    if robbert_result is not None:
        # train_model_robbert already saves best model during training
        # and now returns the best model (not final epoch), so just use it directly
        robbert_model = robbert_result
        pred_rob_val = predict_model_robbert(robbert_model, val, max_length=512)
        pred_rob_test = predict_model_robbert(robbert_model, test, max_length=512)
        proba_rob_val = predict_proba_model_robbert_batch(robbert_model, val, max_length=512)
        proba_rob_test = predict_proba_model_robbert_batch(robbert_model, test, max_length=512)
        r_rob_val = evaluate(val["vote"].values, pred_rob_val)
        r_rob_test = evaluate(test["vote"].values, pred_rob_test)
        RESULTS["robbert"] = {
            "val": r_rob_val, "test": r_rob_test,
            "pred_val": pred_rob_val, "pred_test": pred_rob_test,
            "proba_val": proba_rob_val, "proba_test": proba_rob_test,
        }
        log(f"  RobBERT v2: val={r_rob_val['accuracy']*100:.1f}% test={r_rob_test['accuracy']*100:.1f}%")

    # --- Step 5: Ensemble stacking ---
    if structural_model is not None and robbert_model is not None:
        ensemble_model = run_step(
            "Ensemble stacking",
            train_ensemble_stacked,
            val,
            RESULTS["structural"]["proba_val"],
            RESULTS["robbert"]["proba_val"],
            val["vote"].values,
            structural_model,
            train,
        )
        if ensemble_model is not None:
            proba_struct_val = RESULTS["structural"]["proba_val"]
            proba_struct_test = RESULTS["structural"]["proba_test"]
            proba_rob_val = RESULTS["robbert"]["proba_val"]
            proba_rob_test = RESULTS["robbert"]["proba_test"]
            pred_ens_val = predict_ensemble_stacked(
                ensemble_model, val, proba_struct_val, proba_rob_val, structural_model
            )
            pred_ens_test = predict_ensemble_stacked(
                ensemble_model, test, proba_struct_test, proba_rob_test, structural_model
            )
            r_ens_val = evaluate(val["vote"].values, pred_ens_val)
            r_ens_test = evaluate(test["vote"].values, pred_ens_test)
            RESULTS["ensemble"] = {"val": r_ens_val, "test": r_ens_test}
            log(f"  Ensemble: val={r_ens_val['accuracy']*100:.1f}% test={r_ens_test['accuracy']*100:.1f}%")

    # --- Step 6: Final report ---
    run_step("Evaluation report", write_final_report, RESULTS, val, test, start_time)

    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\nTotal runtime: {elapsed/3600:.2f} hours")
    log("Overnight pipeline finished.")


def write_final_report(results: dict, val, test, start_time):
    """Write outputs/overnight_report.txt with full comparison."""
    from sklearn.metrics import classification_report, confusion_matrix

    lines = []
    lines.append("\n" + "=" * 60)
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
            for party, row in party_acc.head(15).iterrows():
                lines.append(f"  {party:25s} {row['accuracy']*100:5.1f}% ({int(row['total']):4d})")

    elapsed = (datetime.now() - start_time).total_seconds()
    lines.append(f"\nTotal runtime: {elapsed/3600:.2f} hours")
    lines.append("=" * 60)

    report = "\n".join(lines)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
