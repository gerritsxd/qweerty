"""
Tweede Kamer Data Preprocessor
Cleans, normalizes, and exports fetched data to analysis-ready formats.
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Columns that typically contain datetime strings from the API
DATETIME_PATTERNS = re.compile(
    r"(Datum|GewijzigdOp|ApiGewijzigdOp|Geboortedatum|Overlijdensdatum|"
    r"DatumActief|DatumInactief|DatumSoort|Van|TotEnMet|Aanvang|Einde|"
    r"DatumRegistratie|DatumVerzoek|DatumVergadering|Aangemaakt)",
    re.IGNORECASE,
)


class TweedeKamerPreprocessor:
    """Preprocesses raw JSON data into clean, analysis-ready DataFrames."""

    def __init__(self, config: dict):
        self.raw_path = Path(config["paths"]["raw_data"])
        self.processed_path = Path(config["paths"]["processed_data"])
        self.processed_path.mkdir(parents=True, exist_ok=True)

        prep_cfg = config.get("preprocessing", {})
        self.parse_dates = prep_cfg.get("parse_dates", True)
        self.drop_all_null_cols = prep_cfg.get("drop_all_null_columns", True)
        self.normalize_strings = prep_cfg.get("normalize_strings", True)
        self.output_format = prep_cfg.get("output_format", "parquet")

    def load_raw(self, entity_name: str) -> pd.DataFrame:
        """Load raw JSON file into a DataFrame."""
        raw_file = self.raw_path / f"{entity_name}.json"
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_file}")

        with open(raw_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            logger.warning("No records found for %s", entity_name)
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info("Loaded %d records x %d columns for %s", len(df), len(df.columns), entity_name)
        return df

    def _clean_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip and normalize whitespace in string columns."""
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .replace("None", pd.NA)
                .replace("nan", pd.NA)
                .replace("", pd.NA)
            )
        return df

    def _parse_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and parse datetime columns."""
        for col in df.columns:
            if DATETIME_PATTERNS.search(col):
                try:
                    df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
                    logger.debug("Parsed datetime column: %s", col)
                except Exception:
                    logger.debug("Could not parse %s as datetime, skipping", col)
        return df

    def _drop_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns where every value is null."""
        before = len(df.columns)
        df = df.dropna(axis=1, how="all")
        dropped = before - len(df.columns)
        if dropped > 0:
            logger.info("Dropped %d all-null columns", dropped)
        return df

    def _drop_verwijderd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the Verwijderd column since we already filtered for false."""
        if "Verwijderd" in df.columns:
            df = df.drop(columns=["Verwijderd"])
        return df

    def _standardize_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure Id column is lowercase string (UUID format)."""
        if "Id" in df.columns:
            df["Id"] = df["Id"].astype(str).str.strip().str.lower()
        return df

    def preprocess(self, entity_name: str) -> pd.DataFrame:
        """Run the full preprocessing pipeline on a single entity."""
        df = self.load_raw(entity_name)
        if df.empty:
            return df

        # Always drop Verwijderd (redundant -- always false)
        df = self._drop_verwijderd(df)

        # Standardize IDs
        df = self._standardize_id_column(df)

        # Optional steps based on config
        if self.normalize_strings:
            df = self._clean_strings(df)

        if self.parse_dates:
            df = self._parse_datetime_columns(df)

        if self.drop_all_null_cols:
            df = self._drop_empty_columns(df)

        logger.info(
            "Preprocessed %s: %d records x %d columns",
            entity_name, len(df), len(df.columns),
        )
        return df

    def save(self, df: pd.DataFrame, entity_name: str) -> Path:
        """Save processed DataFrame to configured output format."""
        if self.output_format == "parquet":
            out_file = self.processed_path / f"{entity_name}.parquet"
            df.to_parquet(out_file, index=False, engine="pyarrow")
        elif self.output_format == "csv":
            out_file = self.processed_path / f"{entity_name}.csv"
            df.to_csv(out_file, index=False, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        logger.info("Saved processed data to %s", out_file)
        return out_file

    def preprocess_and_save(self, entity_name: str) -> Path | None:
        """Preprocess a single entity and save output."""
        try:
            df = self.preprocess(entity_name)
            if df.empty:
                logger.warning("Skipping %s (empty DataFrame)", entity_name)
                return None
            return self.save(df, entity_name)
        except FileNotFoundError:
            logger.warning("No raw data for %s, skipping", entity_name)
            return None

    def preprocess_all(self, entities: dict) -> dict[str, Path | None]:
        """Preprocess all enabled entities."""
        results: dict[str, Path | None] = {}
        enabled = {
            name: cfg for name, cfg in entities.items()
            if cfg.get("enabled", True)
        }

        print(f"\n{'='*60}")
        print(f"  Preprocessing {len(enabled)} entity types")
        print(f"{'='*60}\n")

        for entity_name in tqdm(enabled, desc="Preprocessing", unit=" entities"):
            results[entity_name] = self.preprocess_and_save(entity_name)

        success = sum(1 for v in results.values() if v is not None)
        print(f"\n{'='*60}")
        print(f"  Done! Preprocessed {success}/{len(enabled)} entities")
        print(f"  Output: {self.processed_path}/")
        print(f"{'='*60}\n")
        return results

    def generate_summary(self, entities: dict) -> pd.DataFrame:
        """Generate a summary table of all processed datasets."""
        rows = []
        enabled = {
            name: cfg for name, cfg in entities.items()
            if cfg.get("enabled", True)
        }

        for entity_name, entity_cfg in enabled.items():
            raw_file = self.raw_path / f"{entity_name}.json"

            if self.output_format == "parquet":
                proc_file = self.processed_path / f"{entity_name}.parquet"
            else:
                proc_file = self.processed_path / f"{entity_name}.csv"

            row = {
                "entity": entity_name,
                "description": entity_cfg.get("description", ""),
                "raw_exists": raw_file.exists(),
                "processed_exists": proc_file.exists(),
                "raw_size_mb": round(raw_file.stat().st_size / 1024 / 1024, 2) if raw_file.exists() else None,
                "processed_size_mb": round(proc_file.stat().st_size / 1024 / 1024, 2) if proc_file.exists() else None,
            }

            # Get record count from processed file
            if proc_file.exists():
                try:
                    if self.output_format == "parquet":
                        df = pd.read_parquet(proc_file)
                    else:
                        df = pd.read_csv(proc_file)
                    row["records"] = len(df)
                    row["columns"] = len(df.columns)
                except Exception:
                    row["records"] = None
                    row["columns"] = None
            else:
                row["records"] = None
                row["columns"] = None

            rows.append(row)

        summary = pd.DataFrame(rows)
        summary_file = self.processed_path / "_summary.csv"
        summary.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")
        return summary
