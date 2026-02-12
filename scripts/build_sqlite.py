#!/usr/bin/env python3
"""
Build SQLite database from parquet files
========================================
Creates a single SQLite database you can open with DB Browser for SQLite,
DBeaver, or any SQL client.

Usage:
    python scripts/build_sqlite.py              # Creates tweedekamer.db
    python scripts/build_sqlite.py -o my.db     # Custom output path
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "tweedekamer.db"


def main():
    ap = argparse.ArgumentParser(description="Build SQLite DB from parquet")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="Output database path")
    args = ap.parse_args()

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    parquet_files = [f for f in parquet_files if not f.stem.startswith("_")]

    if not parquet_files:
        print("No parquet files found in data/processed/")
        sys.exit(1)

    conn = sqlite3.connect(args.out)

    for pf in parquet_files:
        name = pf.stem
        df = pd.read_parquet(pf)
        # SQLite table names: use as-is (Pandas handles this)
        df.to_sql(name, conn, if_exists="replace", index=False)
        print(f"  {name}: {len(df):,} rows")

    conn.close()
    print(f"\nDatabase saved: {args.out.resolve()}")
    print("Open with: DB Browser for SQLite, DBeaver, or: sqlite3 tweedekamer.db")


if __name__ == "__main__":
    main()
