#!/usr/bin/env python3
"""
Tweede Kamer — Parse Verslag XML into structured speech data
=============================================================
Reads downloaded Verslag XML files and extracts:
- Individual speech segments with full text
- Speaker identification (Persoon_Id, name, party, role)
- Activiteit context (objectid, subject, type)
- Zaak references (linked cases/motions)
- Timestamps

Output: data/analysis/speeches.parquet — one row per speech segment.

Usage:
    python -m src.parse_verslagen                 # Parse all XMLs
    python -m src.parse_verslagen --limit 50      # Parse first 50
    python -m src.parse_verslagen --sample         # Parse 5 and print results
"""

import argparse
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
XML_DIR = ROOT / "data" / "texts" / "verslagen"
OUT_DIR = ROOT / "data" / "analysis"

# The Verslag XML namespace
NS = {"v": "http://www.tweedekamer.nl/ggm/vergaderverslag/v1.0"}


# ---------------------------------------------------------------------------
# XML text extraction helpers
# ---------------------------------------------------------------------------

def extract_text(elem) -> str:
    """
    Recursively extract all text from an element and its children.
    Handles <alinea>, <alineaitem>, <nadruk>, etc.
    Returns clean text with paragraphs separated by newlines.
    """
    if elem is None:
        return ""

    paragraphs = []
    # Look for <alinea> elements (paragraphs)
    alineas = elem.findall(".//v:alinea", NS)
    if not alineas:
        # Fallback: no namespace or different structure
        alineas = elem.findall(".//{http://www.tweedekamer.nl/ggm/vergaderverslag/v1.0}alinea")
    if not alineas:
        # Last fallback: just get all text
        return " ".join((elem.itertext() or "")).strip()

    for alinea in alineas:
        # Get all text content within this paragraph
        parts = []
        for item in alinea.iter():
            if item.text:
                parts.append(item.text)
            if item.tail:
                parts.append(item.tail)
        text = " ".join(parts).strip()
        # Clean up multiple spaces
        text = re.sub(r"\s+", " ", text)
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def extract_text_nons(elem) -> str:
    """Extract text without namespace (for some XML variants)."""
    if elem is None:
        return ""
    parts = []
    for t in elem.itertext():
        parts.append(t)
    text = " ".join(parts).strip()
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Speaker extraction
# ---------------------------------------------------------------------------

def parse_spreker(spreker_elem) -> dict:
    """Parse a <spreker> element into a dict."""
    if spreker_elem is None:
        return {}

    def get(tag):
        el = spreker_elem.find(f"v:{tag}", NS)
        if el is None:
            # Try without namespace
            el = spreker_elem.find(tag)
        return el.text.strip() if el is not None and el.text else None

    return {
        "persoon_id": spreker_elem.get("objectid"),
        "spreker_soort": spreker_elem.get("soort"),  # "Tweede Kamerlid", "Minister", etc.
        "fractie": get("fractie"),
        "achternaam": get("achternaam"),
        "voornaam": get("voornaam"),
        "verslagnaam": get("verslagnaam"),
        "functie": get("functie"),
    }


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_verslag_xml(xml_path: Path) -> list[dict]:
    """
    Parse a single Verslag XML file into a list of speech records.
    
    Each record is one speech segment (one person speaking once),
    including interruptions as separate records.
    
    Returns list of dicts with keys:
        verslag_id, vergadering_id, vergadering_soort, vergadering_titel,
        vergaderjaar, datum,
        activiteit_id, activiteit_soort, activiteit_onderwerp,
        activiteithoofd_id, activiteithoofd_onderwerp,
        zaak_ids, zaak_soorten,
        persoon_id, spreker_soort, fractie, achternaam, voornaam, verslagnaam, functie,
        is_voorzitter, is_interruptie,
        speech_text, speech_start, speech_end
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        logger.warning("XML parse error in %s: %s", xml_path.name, e)
        return []

    root = tree.getroot()
    verslag_id = xml_path.stem  # filename is the Verslag Id

    speeches = []

    # Find <vergadering> element
    vergadering = root.find("v:vergadering", NS)
    if vergadering is None:
        # Try without namespace
        vergadering = root.find("vergadering")
    if vergadering is None:
        logger.debug("No <vergadering> in %s", xml_path.name)
        return []

    def vget(tag):
        el = vergadering.find(f"v:{tag}", NS)
        if el is None:
            el = vergadering.find(tag)
        return el.text.strip() if el is not None and el.text else None

    verg_info = {
        "verslag_id": verslag_id,
        "vergadering_id": vergadering.get("objectid"),
        "vergadering_soort": vergadering.get("soort"),
        "vergadering_titel": vget("titel"),
        "vergaderjaar": vget("vergaderjaar"),
        "datum": vget("datum"),
    }

    # Iterate over <activiteit> elements (agenda sections)
    for activiteit in vergadering.findall("v:activiteit", NS):
        act_id = activiteit.get("objectid")
        act_soort = activiteit.get("soort")

        def aget(tag):
            el = activiteit.find(f"v:{tag}", NS)
            return el.text.strip() if el is not None and el.text else None

        act_info = {
            "activiteit_id": act_id,
            "activiteit_soort": act_soort,
            "activiteit_onderwerp": aget("onderwerp"),
        }

        # Iterate over <activiteithoofd> (sub-agenda items)
        for hoofd in activiteit.findall("v:activiteithoofd", NS):
            hoofd_id = hoofd.get("objectid")

            def hget(tag):
                el = hoofd.find(f"v:{tag}", NS)
                return el.text.strip() if el is not None and el.text else None

            hoofd_info = {
                "activiteithoofd_id": hoofd_id,
                "activiteithoofd_onderwerp": hget("onderwerp"),
            }

            # Extract zaak references
            zaak_ids = []
            zaak_soorten = []
            zaken_elem = hoofd.find("v:zaken", NS)
            if zaken_elem is not None:
                for zaak in zaken_elem.findall("v:zaak", NS):
                    zid = zaak.get("objectid")
                    zsoort = zaak.get("soort")
                    if zid:
                        zaak_ids.append(zid)
                    if zsoort:
                        zaak_soorten.append(zsoort)

            zaak_info = {
                "zaak_ids": ";".join(zaak_ids) if zaak_ids else None,
                "zaak_soorten": ";".join(zaak_soorten) if zaak_soorten else None,
            }

            # Iterate over <activiteitdeel> (individual speech turns)
            for deel in hoofd.findall(".//v:activiteitdeel", NS):
                if deel.get("soort") != "Spreekbeurt":
                    continue

                deel_spreker = deel.find("v:spreker", NS)
                deel_info = parse_spreker(deel_spreker) if deel_spreker else {}

                start = None
                end = None
                el_start = deel.find("v:markeertijdbegin", NS)
                el_end = deel.find("v:markeertijdeind", NS)
                if el_start is not None and el_start.text:
                    start = el_start.text.strip()
                if el_end is not None and el_end.text:
                    end = el_end.text.strip()

                # Find <woordvoerder> elements (the actual speech content)
                for item in deel.findall(".//v:activiteititem", NS):
                    for wv in item.findall("v:woordvoerder", NS):
                        # Speaker info (may override deel-level speaker)
                        wv_spreker = wv.find("v:spreker", NS)
                        speaker = parse_spreker(wv_spreker) if wv_spreker else deel_info.copy()

                        is_vz_el = wv.find("v:isvoorzitter", NS)
                        is_voorzitter = (
                            is_vz_el is not None
                            and is_vz_el.text
                            and is_vz_el.text.strip().lower() == "true"
                        )

                        # Main speech text
                        tekst_elem = wv.find("v:tekst", NS)
                        speech_text = extract_text(tekst_elem)

                        if speech_text and len(speech_text.strip()) > 10:
                            record = {
                                **verg_info,
                                **act_info,
                                **hoofd_info,
                                **zaak_info,
                                **speaker,
                                "is_voorzitter": is_voorzitter,
                                "is_interruptie": False,
                                "speech_text": speech_text.strip(),
                                "speech_start": start,
                                "speech_end": end,
                            }
                            speeches.append(record)

                        # Interruptions (other speakers interjecting)
                        for interr in wv.findall("v:interrumpant", NS):
                            int_spreker = interr.find("v:spreker", NS)
                            int_speaker = parse_spreker(int_spreker) if int_spreker else {}

                            int_vz_el = interr.find("v:isvoorzitter", NS)
                            int_is_vz = (
                                int_vz_el is not None
                                and int_vz_el.text
                                and int_vz_el.text.strip().lower() == "true"
                            )

                            int_tekst = interr.find("v:tekst", NS)
                            int_text = extract_text(int_tekst)

                            if int_text and len(int_text.strip()) > 10:
                                int_record = {
                                    **verg_info,
                                    **act_info,
                                    **hoofd_info,
                                    **zaak_info,
                                    **int_speaker,
                                    "is_voorzitter": int_is_vz,
                                    "is_interruptie": True,
                                    "speech_text": int_text.strip(),
                                    "speech_start": start,
                                    "speech_end": end,
                                }
                                speeches.append(int_record)

    return speeches


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def parse_all_verslagen(limit: int | None = None) -> pd.DataFrame:
    """Parse all downloaded Verslag XMLs into a single DataFrame."""
    xml_files = sorted(XML_DIR.glob("*.xml"))

    if not xml_files:
        print(f"  No XML files found in {XML_DIR}")
        return pd.DataFrame()

    if limit:
        xml_files = xml_files[:limit]

    print(f"  Parsing {len(xml_files):,} XML files...")

    all_speeches = []
    failed = 0

    from tqdm import tqdm
    for xml_path in tqdm(xml_files, desc="Parsing Verslag XMLs", unit=" files"):
        try:
            speeches = parse_verslag_xml(xml_path)
            all_speeches.extend(speeches)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", xml_path.name, e)
            failed += 1

    if not all_speeches:
        print("  No speeches extracted!")
        return pd.DataFrame()

    df = pd.DataFrame(all_speeches)

    # Clean up the speaker attribution line (first line often is "Mevrouw X (VVD):")
    df["speech_text_clean"] = df["speech_text"].apply(clean_speech_text)

    print(f"\n  Results:")
    print(f"    Total speeches:     {len(df):,}")
    print(f"    Unique speakers:    {df['persoon_id'].nunique():,}")
    print(f"    Unique vergaderingen: {df['vergadering_id'].nunique():,}")
    print(f"    Unique activiteiten:  {df['activiteit_id'].nunique():,}")
    print(f"    With party label:   {df['fractie'].notna().sum():,}")
    print(f"    Interrupties:       {df['is_interruptie'].sum():,}")
    print(f"    Chair speeches:     {df['is_voorzitter'].sum():,}")
    print(f"    Failed files:       {failed:,}")

    return df


def clean_speech_text(text: str) -> str:
    """
    Remove the speaker attribution line from the beginning of a speech.
    E.g., 'Mevrouw Lodders (VVD):\nVoorzitter. ...' → 'Voorzitter. ...'
    Also: 'De voorzitter:\nAan de orde is...' → 'Aan de orde is...'
    """
    if not text:
        return text

    lines = text.split("\n")
    if not lines:
        return text

    first = lines[0]
    # Common attribution patterns
    patterns = [
        r"^(?:De heer|Mevrouw|De voorzitter|Minister|Staatssecretaris)\b.*?:\s*$",
        r"^[A-Z][a-z]+ .+?\([A-Za-z\-/]+\)\s*:\s*$",
    ]
    for pat in patterns:
        if re.match(pat, first, re.IGNORECASE):
            return "\n".join(lines[1:]).strip()

    # If the first line ends with ":", it's probably attribution
    if first.strip().endswith(":") and len(first) < 120:
        return "\n".join(lines[1:]).strip()

    return text


def save_speeches(df: pd.DataFrame) -> Path:
    """Save parsed speeches to parquet."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "speeches.parquet"
    df.to_parquet(out, index=False)
    print(f"  Saved {len(df):,} speeches to {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse Verslag XMLs into structured speeches")
    parser.add_argument("--limit", type=int, default=None, help="Max XML files to parse")
    parser.add_argument("--sample", action="store_true", help="Parse 5 files and print results")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Parsing Verslag XML files into speech data")
    print("=" * 60 + "\n")

    limit = 5 if args.sample else args.limit
    df = parse_all_verslagen(limit=limit)

    if df.empty:
        print("\nNo data to save.")
        return

    if args.sample:
        print("\n--- Sample speeches ---")
        for _, row in df.head(10).iterrows():
            print(f"\n[{row.get('fractie', '?')}] {row.get('verslagnaam', '?')} "
                  f"({'VZ' if row.get('is_voorzitter') else 'MP'}) "
                  f"— {row.get('activiteit_onderwerp', '?')[:60]}")
            text = row.get("speech_text_clean", row.get("speech_text", ""))
            print(f"  {text[:200]}...")
    else:
        save_speeches(df)

    print("\nDone!")


if __name__ == "__main__":
    main()
