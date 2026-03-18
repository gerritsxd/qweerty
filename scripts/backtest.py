#!/usr/bin/env python3
"""
Walk-forward backtest for vote prediction.

Trains on expanding windows (all data before year Y) and tests on year Y.
Reports accuracy per test year to assess temporal robustness and degradation.
Uses structural model + baseline (fast). Optionally includes Markov + game theory.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
REPORT_PATH = OUTPUTS / "backtest_report.txt"


def log(msg: str, also_print: bool = True):
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    if also_print:
        print(msg, flush=True)


def get_split_by_year(
    df: pd.DataFrame,
    test_year: int,
    date_col: str = "datum",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train = all before test_year, test = test_year only."""
    df = df.copy()
    df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    train = df[df["_year"] < test_year].drop(columns=["_year"]).reset_index(drop=True)
    test = df[df["_year"] == test_year].drop(columns=["_year"]).reset_index(drop=True)
    return train, test


def main():
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Walk-Forward Backtest\n")
        f.write("=" * 60 + "\n\n")

    from src.ml.features import (
        load_pairs,
        build_basic_features,
        add_enhanced_features,
        cluster_topics,
        build_historical_features,
    )
    from src.ml.markov_features import build_markov_features, predict_markov_proba
    from src.ml.models import (
        train_baseline_party,
        predict_baseline_party,
        train_structural_model,
        predict_structural_model,
        predict_proba_structural_model,
        train_ensemble_stacked,
        predict_ensemble_stacked,
        evaluate,
    )

    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    log(f"Loading data{f' (sample={sample})' if sample else ''}...")
    df = load_pairs(sample=sample)
    df = df[df["datum"].notna()]
    df = build_basic_features(df)
    df = df[df["vote"].isin(["Voor", "Tegen"])]

    df["_year"] = pd.to_datetime(df["datum"], errors="coerce").dt.year
    available_years = sorted(df["_year"].dropna().astype(int).unique())
    df = df.drop(columns=["_year"])

    test_years = [y for y in available_years if y >= 2018 and y <= 2025]
    if not test_years:
        log("No suitable test years found.")
        return

    log(f"Test years: {test_years}\n")

    all_results = []

    for test_year in test_years:
        log(f"--- Test year {test_year} ---")
        train_full, test = get_split_by_year(df, test_year)
        train_full = train_full[train_full["vote"].isin(["Voor", "Tegen"])].reset_index(drop=True)
        test = test[test["vote"].isin(["Voor", "Tegen"])].reset_index(drop=True)

        if len(train_full) < 5000:
            log(f"  Skipping: train too small ({len(train_full):,})")
            continue
        if len(test) < 100:
            log(f"  Skipping: test too small ({len(test):,})")
            continue

        n_val = min(2000, max(500, len(train_full) // 10))
        val = train_full.tail(n_val).reset_index(drop=True)
        train = train_full.iloc[:-n_val].reset_index(drop=True)

        log(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        train, val, test = add_enhanced_features(train, val, test)
        train, val, test = cluster_topics(train, val, test)
        train, val, test = build_historical_features(train, val, test)

        has_markov = False
        try:
            train, val, test = build_markov_features(train, val, test)
            has_markov = True
        except Exception as e:
            log(f"  Markov features skipped: {e}")

        bl = train_baseline_party(train)
        pred_bl = predict_baseline_party(bl, test)
        r_bl = evaluate(test["vote"].values, pred_bl)

        try:
            struct = train_structural_model(train, val, test)
            pred_struct = predict_structural_model(struct, test)
            proba_struct = predict_proba_structural_model(struct, test)
            r_struct = evaluate(test["vote"].values, pred_struct)
        except Exception as e:
            log(f"  Structural failed: {e}")
            r_struct = {"accuracy": 0}
            proba_struct = None
            struct = None

        proba_game_val = None
        proba_game_test = None
        proba_markov_val = None
        proba_markov_test = None
        if has_markov:
            try:
                from src.game.simulation import predict_vote_probabilities
                from src.game.calibrate import calibrate_weights
                calibrated = calibrate_weights(train, val, quick=True)
                proba_game_val = predict_vote_probabilities(val, train_df=train, calibrated_weights=calibrated)
                proba_game_test = predict_vote_probabilities(test, train_df=train, calibrated_weights=calibrated)
                proba_markov_val = predict_markov_proba(val, train)
                proba_markov_test = predict_markov_proba(test, train)
            except Exception as e:
                log(f"  Game/Markov skipped: {e}")

        r_ens = r_struct
        if struct is not None and proba_struct is not None:
            proba_struct_val = predict_proba_structural_model(struct, val)
            try:
                ens = train_ensemble_stacked(
                    val,
                    proba_struct_val,
                    proba_struct_val,
                    val["vote"].values,
                    struct,
                    train,
                    proba_game=proba_game_val,
                    proba_markov=proba_markov_val,
                )
                pred_ens = predict_ensemble_stacked(
                    ens, test,
                    proba_struct,
                    proba_struct,
                    struct,
                    proba_game=proba_game_test,
                    proba_markov=proba_markov_test,
                )
                r_ens = evaluate(test["vote"].values, pred_ens)
            except Exception as e:
                log(f"  Ensemble fallback to structural: {e}")

        row = {
            "year": test_year,
            "n_train": len(train),
            "n_test": len(test),
            "baseline": r_bl["accuracy"] * 100,
            "structural": r_struct["accuracy"] * 100,
            "ensemble": r_ens["accuracy"] * 100,
        }
        all_results.append(row)
        log(f"  Baseline: {row['baseline']:.1f}% | Structural: {row['structural']:.1f}% | Ensemble: {row['ensemble']:.1f}%")

    log("\n" + "=" * 60)
    log("BACKTEST SUMMARY")
    log("=" * 60)
    log(f"{'Year':<6} {'Train':>8} {'Test':>8} {'Baseline':>10} {'Structural':>10} {'Ensemble':>10}")
    log("-" * 60)
    for r in all_results:
        log(f"{r['year']:<6} {r['n_train']:>8,} {r['n_test']:>8,} {r['baseline']:>9.1f}% {r['structural']:>9.1f}% {r['ensemble']:>9.1f}%")

    if all_results:
        avg_ens = sum(r["ensemble"] for r in all_results) / len(all_results)
        avg_bl = sum(r["baseline"] for r in all_results) / len(all_results)
        avg_struct = sum(r["structural"] for r in all_results) / len(all_results)
        log("-" * 60)
        log(f"Average baseline:   {avg_bl:.1f}%")
        log(f"Average structural: {avg_struct:.1f}%")
        log(f"Average ensemble:   {avg_ens:.1f}%")
        log(f"Ensemble vs baseline: +{avg_ens - avg_bl:.1f}pp")

    log("\nBacktest complete.")


if __name__ == "__main__":
    main()
