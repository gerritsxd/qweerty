#!/usr/bin/env python3
"""
Tweede Kamer — Verslag (debate transcript) fetcher
====================================================
1. Fetches all Verslag metadata records from the OData API
2. Downloads the actual XML content for each Verslag

The XML contains the full stenographic transcript with individual speeches,
speaker identification, party membership, and agenda item structure.

Usage:
    python -m src.fetch_verslagen                    # Fetch all
    python -m src.fetch_verslagen --limit 100        # Fetch first 100
    python -m src.fetch_verslagen --skip-xml         # Metadata only
    python -m src.fetch_verslagen --only-full-text   # Skip Casco (skeleton) verslagen
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
RAW_DIR = ROOT / "data" / "raw"
XML_DIR = ROOT / "data" / "texts" / "verslagen"

VERSLAG_STATUS_PRIORITY = {
    # Prefer the most complete version
    "Gecorrigeerd": 1,
    "Ongecorrigeerd": 2,
    "Voorpublicatie": 3,
    "Tussenpublicatie": 4,
    "Casco": 5,
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Fetch Verslag metadata (same paginated OData approach as fetch.py)
# ---------------------------------------------------------------------------

def fetch_verslag_metadata(config: dict, limit: int | None = None) -> list[dict]:
    """Fetch all Verslag records from the OData API."""
    base_url = config["api"]["base_url"]
    delay = config["api"]["request_delay"]
    timeout = config["api"]["timeout"]
    max_retries = config["api"]["max_retries"]
    backoff = config["api"]["retry_backoff"]

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    # Get count first
    count_url = (
        f"{base_url}/Verslag"
        f"?$filter=Verwijderd eq false"
        f"&$top=0&$count=true"
        f"&$format=application/json;odata.metadata=minimal"
    )
    try:
        resp = session.get(count_url, timeout=timeout)
        resp.raise_for_status()
        total = resp.json().get("@odata.count", None)
    except Exception:
        total = None

    if limit and total:
        total = min(total, limit)

    url = (
        f"{base_url}/Verslag"
        f"?$filter=Verwijderd eq false"
        f"&$format=application/json;odata.metadata=none"
    )

    all_records = []
    pbar = tqdm(total=total, desc="Fetching Verslag metadata", unit=" records")

    while url:
        for attempt in range(max_retries):
            try:
                resp = session.get(url, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = backoff ** (attempt + 1)
                logger.warning("Retry %d/%d: %s (wait %.1fs)", attempt + 1, max_retries, e, wait)
                time.sleep(wait)

        records = data.get("value", [])
        all_records.extend(records)
        pbar.update(len(records))

        if limit and len(all_records) >= limit:
            all_records = all_records[:limit]
            break

        url = data.get("@odata.nextLink")
        if url:
            time.sleep(delay)

    pbar.close()
    print(f"  Fetched {len(all_records):,} Verslag records")
    return all_records


def save_verslag_metadata(records: list[dict]) -> Path:
    """Save Verslag metadata as JSON."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "Verslag.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  Saved metadata to {out}")
    return out


# ---------------------------------------------------------------------------
# Pick best Verslag per Vergadering
# ---------------------------------------------------------------------------

def pick_best_verslagen(records: list[dict]) -> list[dict]:
    """
    For each Vergadering, keep only the best (most complete) Verslag.
    A single vergadering can have multiple verslagen (Casco, Tussenpublicatie,
    Ongecorrigeerd, Gecorrigeerd). We want the most complete one.
    """
    by_vergadering: dict[str, list[dict]] = {}
    for r in records:
        vid = r.get("Vergadering_Id")
        if vid:
            by_vergadering.setdefault(vid, []).append(r)

    best = []
    for vid, verslagen in by_vergadering.items():
        # Sort by status priority (lower = better)
        verslagen.sort(
            key=lambda v: VERSLAG_STATUS_PRIORITY.get(v.get("Status", ""), 99)
        )
        best.append(verslagen[0])

    # Also include verslagen without a Vergadering_Id (shouldn't happen, but be safe)
    orphans = [r for r in records if not r.get("Vergadering_Id")]
    best.extend(orphans)

    print(f"  {len(records):,} total verslagen -> {len(best):,} best per vergadering")
    return best


# ---------------------------------------------------------------------------
# Download XML content
# ---------------------------------------------------------------------------

def download_verslag_xml(
    records: list[dict],
    config: dict,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """
    Download the XML resource for each Verslag.
    Returns dict of Verslag_Id → xml file path.
    """
    base_url = config["api"]["base_url"]
    delay = config["api"]["request_delay"]
    timeout = config["api"]["timeout"]
    max_retries = config["api"]["max_retries"]
    backoff = config["api"]["retry_backoff"]

    XML_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    # XML content — accept XML
    session.headers.update({"Accept": "text/xml, application/xml, */*"})

    results = {}
    skipped = 0
    failed = 0

    pbar = tqdm(records, desc="Downloading Verslag XMLs", unit=" files")

    for rec in pbar:
        vid = rec["Id"]
        out_path = XML_DIR / f"{vid}.xml"

        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            results[vid] = out_path
            skipped += 1
            continue

        url = f"{base_url}/Verslag/{vid}/resource"

        success = False
        for attempt in range(max_retries):
            try:
                resp = session.get(url, timeout=timeout)
                if resp.status_code == 404:
                    # No resource available for this verslag
                    logger.debug("No resource for Verslag %s (404)", vid)
                    break
                resp.raise_for_status()

                with open(out_path, "wb") as f:
                    f.write(resp.content)

                results[vid] = out_path
                success = True
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning("Failed to download %s after %d retries: %s", vid, max_retries, e)
                    failed += 1
                else:
                    wait = backoff ** (attempt + 1)
                    time.sleep(wait)

        time.sleep(delay)

    pbar.close()
    print(f"  Downloaded: {len(results):,} | Skipped (existing): {skipped:,} | Failed: {failed:,}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Verslag transcripts")
    parser.add_argument("--limit", type=int, default=None, help="Max verslagen to fetch")
    parser.add_argument("--skip-xml", action="store_true", help="Only fetch metadata, skip XML download")
    parser.add_argument("--only-full-text", action="store_true",
                        help="Skip Casco (skeleton) verslagen — only download those with actual text")
    parser.add_argument("--redownload", action="store_true", help="Re-download existing XML files")
    args = parser.parse_args()

    config = load_config()

    print("\n" + "=" * 60)
    print("  Fetching Verslag (debate transcript) data")
    print("=" * 60 + "\n")

    # Step 1: Fetch metadata
    print("[1/3] Fetching Verslag metadata from API...")
    records = fetch_verslag_metadata(config, limit=args.limit)
    save_verslag_metadata(records)

    # Step 2: Pick best per vergadering
    print("\n[2/3] Selecting best version per vergadering...")
    best = pick_best_verslagen(records)

    if args.only_full_text:
        before = len(best)
        best = [r for r in best if r.get("Status") != "Casco"]
        print(f"  Filtered out Casco verslagen: {before} -> {len(best)}")

    # Step 3: Download XMLs
    if not args.skip_xml:
        print(f"\n[3/3] Downloading XML transcripts ({len(best):,} files)...")
        download_verslag_xml(best, config, skip_existing=not args.redownload)
    else:
        print("\n[3/3] Skipping XML download (--skip-xml)")

    print("\nDone!")
    print(f"  Metadata: {RAW_DIR / 'Verslag.json'}")
    print(f"  XMLs:     {XML_DIR}/")


if __name__ == "__main__":
    main()
