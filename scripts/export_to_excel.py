#!/usr/bin/env python3
"""
Export parquet files to CSV for Excel
=====================================
Excel can open CSV files directly. Large tables are limited to avoid
Excel row limits (~1M rows) and slow loading.

Usage:
    python scripts/export_to_excel.py              # All entities, 10k rows each
    python scripts/export_to_excel.py --limit 5000 # Custom row limit
    python scripts/export_to_excel.py Persoon      # Single entity
"""

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "exports" / "excel"
DEFAULT_LIMIT = 10_000  # Excel handles ~1M rows but 10k keeps files snappy


def main():
    ap = argparse.ArgumentParser(description="Export parquet to CSV for Excel")
    ap.add_argument("entities", nargs="*", help="Entity names (e.g. Persoon, Fractie). If empty, export all.")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Max rows per table (default {DEFAULT_LIMIT})")
    ap.add_argument("--no-limit", action="store_true", help="No row limit (full export)")
    ap.add_argument("-o", "--out", type=Path, default=OUT_DIR, help="Output directory")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    parquet_files = [f for f in parquet_files if not f.stem.startswith("_")]

    if args.entities:
        parquet_files = [f for f in parquet_files if f.stem in args.entities]
        if not parquet_files:
            print(f"No matching entities: {args.entities}")
            sys.exit(1)

    limit = None if args.no_limit else args.limit

    for pf in parquet_files:
        name = pf.stem
        df = pd.read_parquet(pf)
        n = len(df)
        if limit and n > limit:
            df = df.head(limit)
            suffix = f" (first {limit:,} of {n:,})"
        else:
            suffix = f" ({n:,} rows)"
        out_path = args.out / f"{name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  {name}.csv{suffix}")
    print(f"\nSaved to: {args.out.resolve()}")
    print("Open .csv files in Excel, LibreOffice Calc, or Google Sheets.")


if __name__ == "__main__":
    main()
