#!/usr/bin/env python3
"""
Tweede Kamer — Full speech-to-vote dataset builder
====================================================
Orchestrates the entire pipeline:
  1. Fetch Verslag metadata + download XMLs  (fetch_verslagen.py)
  2. Parse XMLs into speech records           (parse_verslagen.py)
  3. Link speeches to votes                   (link_speech_vote.py)

Run this on the computer that has the full dataset.

Usage:
    python -m src.build_speech_dataset                # Full pipeline
    python -m src.build_speech_dataset --skip-fetch    # Skip API calls (XMLs already downloaded)
    python -m src.build_speech_dataset --skip-parse    # Skip parsing (speeches.parquet exists)
    python -m src.build_speech_dataset --stats          # Just show stats on existing data
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Build the full speech-to-vote dataset")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip Verslag fetching (XMLs exist)")
    parser.add_argument("--skip-parse", action="store_true", help="Skip XML parsing (speeches.parquet exists)")
    parser.add_argument("--stats", action="store_true", help="Show stats on existing dataset")
    parser.add_argument("--fetch-limit", type=int, default=None, help="Max Verslagen to fetch")
    parser.add_argument("--parse-limit", type=int, default=None, help="Max XMLs to parse")
    parser.add_argument("--only-full-text", action="store_true",
                        help="Skip Casco verslagen during fetch")
    args = parser.parse_args()

    start = time.time()

    print("\n" + "=" * 60)
    print("  Speech-to-Vote Dataset Builder")
    print("  Tweede Kamer Open Data")
    print("=" * 60)

    # Step 1: Fetch Verslagen
    if not args.skip_fetch and not args.skip_parse and not args.stats:
        print("\n" + "-" * 60)
        print("  STEP 1: Fetch Verslag metadata + download XMLs")
        print("-" * 60)

        from src.fetch_verslagen import (
            load_config, fetch_verslag_metadata, save_verslag_metadata,
            pick_best_verslagen, download_verslag_xml,
        )

        config = load_config()
        records = fetch_verslag_metadata(config, limit=args.fetch_limit)
        save_verslag_metadata(records)
        best = pick_best_verslagen(records)

        if args.only_full_text:
            best = [r for r in best if r.get("Status") != "Casco"]
            print(f"  Filtered to non-Casco: {len(best):,}")

        download_verslag_xml(best, config)
    else:
        if not args.stats:
            print("\n  [Skipping fetch - using existing XML files]")

    # Step 2: Parse XMLs
    if not args.skip_parse and not args.stats:
        print("\n" + "-" * 60)
        print("  STEP 2: Parse Verslag XMLs into speech records")
        print("-" * 60)

        from src.parse_verslagen import parse_all_verslagen, save_speeches

        df = parse_all_verslagen(limit=args.parse_limit)
        if df.empty:
            print("\n  ERROR: No speeches extracted. Check XML files.")
            return
        save_speeches(df)
    else:
        if not args.stats:
            print("\n  [Skipping parse - using existing speeches.parquet]")

    # Step 3: Link speeches to votes
    print("\n" + "-" * 60)
    if args.stats:
        print("  STATS: Existing dataset")
    else:
        print("  STEP 3: Link speeches to votes")
    print("-" * 60)

    from src.link_speech_vote import main as link_main

    # Temporarily replace sys.argv so the linker's argparse doesn't see our args
    old_argv = sys.argv
    sys.argv = ["link_speech_vote.py", "--stats"] if args.stats else ["link_speech_vote.py"]
    try:
        link_main()
    finally:
        sys.argv = old_argv

    # Summary
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  Pipeline complete! ({elapsed:.0f}s)")
    print("=" * 60)

    analysis_dir = ROOT / "data" / "analysis"
    for f in sorted(analysis_dir.glob("*.parquet")):
        import pandas as pd
        df = pd.read_parquet(f)
        print(f"  {f.name}: {len(df):,} rows, {len(df.columns)} cols")

    print(f"\n  Output: {analysis_dir}/")
    print("  Master dataset: speech_vote_pairs.parquet")
    print("  Splits: train.parquet, val.parquet, test.parquet")


if __name__ == "__main__":
    main()
