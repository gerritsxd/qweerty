"""
Tweede Kamer Open Data Fetcher
Handles paginated OData API calls for all entity types.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TweedeKamerFetcher:
    """Fetches all data from the Tweede Kamer OData v4 API with pagination."""

    def __init__(self, config: dict):
        self.base_url = config["api"]["base_url"]
        self.page_size = config["api"]["page_size"]
        self.request_delay = config["api"]["request_delay"]
        self.max_retries = config["api"]["max_retries"]
        self.retry_backoff = config["api"]["retry_backoff"]
        self.timeout = config["api"]["timeout"]
        self.raw_path = Path(config["paths"]["raw_data"])
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
        })

    def _request_with_retry(self, url: str) -> dict:
        """Make a GET request with exponential backoff retry."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    # Rate limited -- wait longer
                    wait = self.retry_backoff ** (attempt + 2)
                    logger.warning("Rate limited (429). Waiting %.1fs...", wait)
                    time.sleep(wait)
                    last_exception = e
                elif resp.status_code >= 500:
                    wait = self.retry_backoff ** (attempt + 1)
                    logger.warning(
                        "Server error %d on attempt %d/%d. Retrying in %.1fs...",
                        resp.status_code, attempt + 1, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    last_exception = e
                else:
                    raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                wait = self.retry_backoff ** (attempt + 1)
                logger.warning(
                    "Connection error on attempt %d/%d: %s. Retrying in %.1fs...",
                    attempt + 1, self.max_retries, e, wait,
                )
                time.sleep(wait)
                last_exception = e
        raise RuntimeError(
            f"Failed after {self.max_retries} retries: {last_exception}"
        ) from last_exception

    def get_entity_count(self, entity_name: str) -> int | None:
        """Get the total count of non-deleted entities (if the API supports $count)."""
        url = (
            f"{self.base_url}/{entity_name}"
            f"?$filter=Verwijderd eq false"
            f"&$top=0&$count=true"
            f"&$format=application/json;odata.metadata=minimal"
        )
        try:
            data = self._request_with_retry(url)
            return data.get("@odata.count")
        except Exception:
            return None

    def fetch_entity(self, entity_name: str, description: str = "") -> list[dict[str, Any]]:
        """
        Fetch ALL non-deleted records for a single entity type.
        Handles pagination via @odata.nextLink.
        Returns list of all record dicts.
        """
        all_records: list[dict[str, Any]] = []

        # Try to get total count for progress bar
        total = self.get_entity_count(entity_name)
        desc = f"{entity_name}"
        if description:
            desc = f"{entity_name} ({description})"

        pbar = tqdm(total=total, desc=desc, unit=" records")

        url: str | None = (
            f"{self.base_url}/{entity_name}"
            f"?$filter=Verwijderd eq false"
            f"&$format=application/json;odata.metadata=none"
        )

        page = 0
        while url:
            data = self._request_with_retry(url)
            records = data.get("value", [])
            all_records.extend(records)
            pbar.update(len(records))

            # OData pagination: follow @odata.nextLink
            url = data.get("@odata.nextLink")
            page += 1

            if url:
                time.sleep(self.request_delay)

        pbar.close()
        logger.info(
            "Fetched %d records for %s (%d pages)",
            len(all_records), entity_name, page,
        )
        return all_records

    def save_raw(self, entity_name: str, records: list[dict[str, Any]]) -> Path:
        """Save raw JSON records to disk."""
        out_file = self.raw_path / f"{entity_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d records to %s", len(records), out_file)
        return out_file

    def fetch_and_save(self, entity_name: str, description: str = "") -> Path:
        """Fetch all records for an entity and save as raw JSON."""
        records = self.fetch_entity(entity_name, description)
        return self.save_raw(entity_name, records)

    def fetch_all(self, entities: dict) -> dict[str, Path]:
        """
        Fetch all enabled entities from config.
        Returns dict of entity_name -> output file path.
        """
        results: dict[str, Path] = {}
        enabled = {
            name: cfg for name, cfg in entities.items()
            if cfg.get("enabled", True)
        }
        logger.info("Fetching %d entity types...", len(enabled))
        print(f"\n{'='*60}")
        print(f"  Fetching {len(enabled)} entity types from Tweede Kamer API")
        print(f"{'='*60}\n")

        for entity_name, entity_cfg in enabled.items():
            desc = entity_cfg.get("description", "")
            try:
                path = self.fetch_and_save(entity_name, desc)
                results[entity_name] = path
            except Exception as e:
                logger.error("Failed to fetch %s: %s", entity_name, e)
                print(f"  [ERROR] Failed to fetch {entity_name}: {e}")

        print(f"\n{'='*60}")
        print(f"  Done! Fetched {len(results)}/{len(enabled)} entities")
        print(f"{'='*60}\n")
        return results
