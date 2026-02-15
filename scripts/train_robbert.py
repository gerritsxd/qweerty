#!/usr/bin/env python3
"""
Iterative self-improving RobBERT training for vote prediction.

The model discovers its own relations from text (party + besluit + topic + speech).
No hand-crafted numeric features — just language understanding.

Training uses:
  - Focal loss: rewards confident correct predictions, focuses on hard cases
  - Progressive unfreezing: head-only -> top layers -> deeper layers
  - Per-epoch sample reweighting: mastered patterns fade, hard cases amplified
  - Discovery logging: see what the model learns each epoch

Usage:
  python scripts/train_robbert.py --epochs 10 --batch_size 16 --fp16
  python scripts/train_robbert.py --epochs 15 --lr 3e-5 --focal_gamma 2.5
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train RobBERT with iterative self-improvement")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (default: 3e-5)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use fp16 (default: True)")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16", help="Disable fp16")
    parser.add_argument("--sample", type=int, default=None, help="Max samples (default: all data)")
    parser.add_argument("--output_dir", type=str, default=None, help="Model save path")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)")
    args = parser.parse_args()

    from src.ml.features import load_pairs, get_train_val_test, build_basic_features, add_enhanced_features
    from src.ml.models import (
        train_model_robbert,
        predict_model_robbert,
        save_model_robbert,
        evaluate,
        train_baseline_party,
        predict_baseline_party,
    )

    print("=" * 60)
    print("RobBERT Iterative Self-Improving Training")
    print("=" * 60)

    print("\nLoading data...")
    df = load_pairs(sample=args.sample)
    df = df[df["datum"].notna()]
    df = build_basic_features(df)
    train, val, test = get_train_val_test(df)
    train = train[train["vote"].isin(["Voor", "Tegen"])]
    val = val[val["vote"].isin(["Voor", "Tegen"])]
    test = test[test["vote"].isin(["Voor", "Tegen"])]
    # Still call add_enhanced_features so besluit_tekst enrichment happens
    train, val, test = add_enhanced_features(train, val, test)

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # Print baseline for comparison
    print("\n--- Baselines ---")
    majority_label = train["vote"].value_counts().idxmax()
    majority_acc_val = (val["vote"] == majority_label).mean()
    majority_acc_test = (test["vote"] == majority_label).mean()
    print(f"Majority class ({majority_label}): val={majority_acc_val*100:.1f}% test={majority_acc_test*100:.1f}%")

    bl = train_baseline_party(train)
    p_val_bl = predict_baseline_party(bl, val)
    p_test_bl = predict_baseline_party(bl, test)
    r_val_bl = evaluate(val["vote"].values, p_val_bl)
    r_test_bl = evaluate(test["vote"].values, p_test_bl)
    print(f"Party baseline:          val={r_val_bl['accuracy']*100:.1f}% test={r_test_bl['accuracy']*100:.1f}%")
    print()

    out_dir = args.output_dir or str(ROOT / "models" / "robbert_vote_classifier")
    print("Starting iterative training...\n")
    model_dict = train_model_robbert(
        train,
        val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fp16=args.fp16,
        save_path=out_dir,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
    )
    save_model_robbert(model_dict, out_dir)
    print(f"Model saved to {out_dir}")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    pred_val = predict_model_robbert(model_dict, val)
    pred_test = predict_model_robbert(model_dict, test)
    r_val = evaluate(val["vote"].values, pred_val)
    r_test = evaluate(test["vote"].values, pred_test)

    print(f"Val  accuracy: {r_val['accuracy']*100:.1f}%  f1={r_val['f1_macro']:.3f}")
    print(f"Test accuracy: {r_test['accuracy']*100:.1f}%  f1={r_test['f1_macro']:.3f}")
    print()
    print(f"vs. majority baseline:  val +{(r_val['accuracy']-majority_acc_val)*100:.1f}pp  "
          f"test +{(r_test['accuracy']-majority_acc_test)*100:.1f}pp")
    print(f"vs. party baseline:     val +{(r_val['accuracy']-r_val_bl['accuracy'])*100:.1f}pp  "
          f"test +{(r_test['accuracy']-r_test_bl['accuracy'])*100:.1f}pp")

    # Save training log
    log_dir = ROOT / "outputs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "robbert_training_log.csv"
    import csv
    row = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "focal_gamma": args.focal_gamma,
        "dropout": args.dropout,
        "val_accuracy": r_val["accuracy"],
        "val_f1": r_val["f1_macro"],
        "test_accuracy": r_test["accuracy"],
        "test_f1": r_test["f1_macro"],
        "baseline_val": r_val_bl["accuracy"],
        "baseline_test": r_test_bl["accuracy"],
    }
    write_header = not log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Log appended to {log_file}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f"\nFATAL ERROR: {e}", flush=True)
        sys.exit(1)
