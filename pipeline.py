#!/usr/bin/env python3
"""
Tweede Kamer Open Data Pipeline
================================
Fetches, preprocesses, and exports all parliamentary data from the
Dutch House of Representatives (Tweede Kamer) OData API.

Usage:
    # Full pipeline: fetch + preprocess all entities
    python pipeline.py

    # Fetch only (download raw JSON)
    python pipeline.py --fetch-only

    # Preprocess only (requires raw data already downloaded)
    python pipeline.py --preprocess-only

    # Fetch & preprocess specific entities
    python pipeline.py --entities Persoon Fractie Stemming

    # Generate summary of processed data
    python pipeline.py --summary

    # Use a custom config file
    python pipeline.py --config my_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.fetch import TweedeKamerFetcher
from src.preprocess import TweedeKamerPreprocessor


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load pipeline configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Config file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_entities(entities: dict, selected: list[str] | None) -> dict:
    """Filter entities dict to only include selected names (if specified)."""
    if not selected:
        return entities
    filtered = {}
    for name in selected:
        if name in entities:
            filtered[name] = entities[name]
        else:
            print(f"Warning: Entity '{name}' not found in config, skipping.")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tweede Kamer Open Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--fetch-only", action="store_true",
        help="Only fetch raw data, skip preprocessing",
    )
    parser.add_argument(
        "--preprocess-only", action="store_true",
        help="Only preprocess (requires raw data already fetched)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Generate a summary table of all processed data",
    )
    parser.add_argument(
        "--entities", nargs="+", default=None,
        help="Only process specific entities (space-separated names)",
    )
    parser.add_argument(
        "--list-entities", action="store_true",
        help="List all available entity types and exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    entities = filter_entities(config["entities"], args.entities)

    # List mode
    if args.list_entities:
        print(f"\nAvailable entity types ({len(config['entities'])}):\n")
        for name, cfg in config["entities"].items():
            status = "enabled" if cfg.get("enabled", True) else "disabled"
            desc = cfg.get("description", "")
            print(f"  [{status:>8}] {name:<40} {desc}")
        print()
        return

    # Summary mode
    if args.summary:
        preprocessor = TweedeKamerPreprocessor(config)
        summary = preprocessor.generate_summary(entities)
        print(f"\n{summary.to_string(index=False)}\n")
        return

    print(r"""
  _____ _          _      _  __
 |_   _|_ __ ___  ___  __| | ___  | |/ /__ _ _ __ ___   ___ _ __
   | | \ V  V / -_) -_) _` |/ -_) | ' </ _` | '  \/ -_) '_|
   |_|  \_/\_/\___\___\__,_|\___| |_|\_\__,_|_|_|_\___|_|

  Open Data Pipeline
  API: https://gegevensmagazijn.tweedekamer.nl/OData/v4/2.0
    """)

    fetcher = TweedeKamerFetcher(config)
    preprocessor = TweedeKamerPreprocessor(config)

    # Fetch
    if not args.preprocess_only:
        fetcher.fetch_all(entities)

    # Preprocess
    if not args.fetch_only:
        preprocessor.preprocess_all(entities)

        # Always generate summary after preprocessing
        summary = preprocessor.generate_summary(entities)
        print(summary.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
