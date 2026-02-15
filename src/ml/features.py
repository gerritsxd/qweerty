#!/usr/bin/env python3
"""
Feature extraction for vote prediction.

Builds feature matrix from speech_vote_pairs + optional NLP features.
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
_STOP = {"de", "het", "een", "van", "en", "in", "op", "te", "voor", "met",
         "dat", "die", "dit", "is", "zijn", "worden", "om", "aan", "bij",
         "als", "naar", "over", "uit", "tot", "door"}


def _tokenize(text: str) -> set:
    if not text or not isinstance(text, str):
        return set()
    words = re.findall(r"\b[a-z]{3,}\b", str(text).lower())
    return {w for w in words if w not in _STOP}


def enrich_with_zaak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add kabinetsappreciatie and zaak_soort by matching agendapunt_onderwerp to Zaak.
    Uses token overlap; infers zaak_soort from text if no match.
    """
    ap_col = "agendapunt_onderwerp" if "agendapunt_onderwerp" in df.columns else "besluit_tekst"
    out = df.copy()
    out["kabinetsappreciatie"] = "Onbekend"
    out["zaak_soort"] = "Onbekend"

    zaak_path = DATA_DIR / "Zaak.parquet"
    if not zaak_path.exists():
        return out

    zaak = pd.read_parquet(zaak_path)
    zaak = zaak[zaak["Soort"].isin(["Motie", "Amendement"])].copy()
    zaak["_text"] = (zaak["Titel"].fillna("") + " " + zaak["Onderwerp"].fillna("")).str.strip()
    zaak["_tokens"] = zaak["_text"].apply(_tokenize)

    word_to_zaak: dict[str, list[tuple[str, str, str]]] = {}
    for _, row in zaak.iterrows():
        t_z = row["_tokens"] if isinstance(row["_tokens"], set) else set()
        ka = str(row.get("Kabinetsappreciatie") or "Onbekend")[:80]
        soort = str(row.get("Soort") or "Onbekend")
        for w in t_z:
            word_to_zaak.setdefault(w, []).append((ka, soort))

    unique_ap = out[ap_col].fillna("").astype(str).unique()
    ap_to_zaak = {}
    for ap in unique_ap:
        if not ap or len(ap) < 5:
            continue
        t_ap = _tokenize(ap)
        if not t_ap:
            continue
        scores: dict[tuple[str, str], int] = {}
        for w in t_ap:
            for (ka, soort) in word_to_zaak.get(w, []):
                key = (ka, soort)
                scores[key] = scores.get(key, 0) + 1
        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            if best[1] >= 2:
                ap_to_zaak[ap] = best[0]  # (ka, soort)
            else:
                soort = "Motie" if "motie" in ap.lower() else "Amendement" if "amendement" in ap.lower() else "Onbekend"
                ap_to_zaak[ap] = ("Onbekend", soort)
        else:
            soort = "Motie" if "motie" in ap.lower() else "Amendement" if "amendement" in ap.lower() else "Onbekend"
            ap_to_zaak[ap] = ("Onbekend", soort)

    def lookup(ap):
        return ap_to_zaak.get(str(ap or ""), ("Onbekend", "Onbekend"))

    mapped = out[ap_col].fillna("").astype(str).apply(lookup)
    out["kabinetsappreciatie"] = [m[0] for m in mapped]
    out["zaak_soort"] = [m[1] for m in mapped]
    return out


def load_pairs(sample: Optional[int] = None) -> pd.DataFrame:
    """Load speech-vote pairs from analysis dir."""
    path = ROOT / "data" / "analysis" / "speech_vote_pairs.parquet"
    df = pd.read_parquet(path)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)
    return df


def build_basic_features(
    df: pd.DataFrame,
    sample: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build basic feature matrix (no NLP, metadata only).
    - fractie (one-hot or label)
    - speech_length (chars)
    - Optional: vergaderjaar, activiteit_onderwerp hash
    """
    out = df.copy()
    if sample:
        out = out.sample(min(sample, len(out)), random_state=42)

    # Speech length
    out["speech_length"] = out["speech_text"].fillna("").str.len()

    # Stance keyword counts (if nlp available)
    try:
        from src.nlp.preprocess import count_stance_keywords
        kw = out["speech_text"].fillna("").apply(count_stance_keywords)
        out["n_voor_kw"] = [k[0] for k in kw]
        out["n_tegen_kw"] = [k[1] for k in kw]
    except ImportError:
        pass

    return out


def get_train_val_test(
    df: pd.DataFrame,
    date_col: str = "datum",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: train <= 2021, val 2022, test >= 2023.
    """
    df = df.copy()
    df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    train = df[df["_year"] <= 2021].drop(columns=["_year"])
    val = df[df["_year"] == 2022].drop(columns=["_year"])
    test = df[df["_year"] >= 2023].drop(columns=["_year"])
    return train, val, test


def compute_speaker_loyalty(
    train: pd.DataFrame,
    df: pd.DataFrame,
    persoon_col: str = "persoon_id",
    fractie_col: str = "fractie",
    vote_col: str = "vote",
    besluit_col: str = "besluit_id",
) -> pd.Series:
    """
    Compute speaker loyalty ratio: how often does this speaker vote with party majority?
    Uses train to compute party majority per (besluit, fractie), then loyalty per persoon.
    For val/test: use loyalty from train, or 0.5 for unseen speakers.
    """
    train_sub = train[train[vote_col].isin(["Voor", "Tegen"])].copy()
    if train_sub.empty:
        return pd.Series(0.5, index=df.index)

    # Party majority per (besluit_id, fractie)
    party_majority = (
        train_sub.groupby([besluit_col, fractie_col])[vote_col]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .to_dict()
    )

    def matches_party(row):
        key = (row[besluit_col], row[fractie_col])
        maj = party_majority.get(key)
        if maj is None:
            return None
        return 1.0 if row[vote_col] == maj else 0.0

    train_sub["_match"] = train_sub.apply(matches_party, axis=1)
    train_sub = train_sub[train_sub["_match"].notna()]

    loyalty = train_sub.groupby(persoon_col)["_match"].mean().to_dict()

    return df[persoon_col].map(lambda x: loyalty.get(x, 0.5)).values


# Coalition parties by period (approximate)
_COALITION_PARTIES = {
    2017: {"VVD", "CDA", "D66", "ChristenUnie"},
    2018: {"VVD", "CDA", "D66", "ChristenUnie"},
    2019: {"VVD", "CDA", "D66", "ChristenUnie"},
    2020: {"VVD", "CDA", "D66", "ChristenUnie"},
    2021: {"VVD", "CDA", "D66", "ChristenUnie"},
    2022: {"VVD", "D66", "CDA", "ChristenUnie"},
    2023: {"VVD", "D66", "CDA", "ChristenUnie"},
    2024: {"PVV", "VVD", "NSC", "BBB"},
    2025: {"PVV", "VVD", "NSC", "BBB"},
}


def add_enhanced_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add Step 2 features: speaker_loyalty, speech_position, kabinetsappreciatie,
    zaak_soort, is_coalition.
    Call after get_train_val_test. Modifies in place and returns.
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    loyalty_train = compute_speaker_loyalty(train, train)
    loyalty_val = compute_speaker_loyalty(train, val)
    loyalty_test = compute_speaker_loyalty(train, test)
    train["speaker_loyalty"] = loyalty_train
    val["speaker_loyalty"] = loyalty_val
    test["speaker_loyalty"] = loyalty_test

    if "speech_position" not in train.columns:
        train["speech_position"] = 0.5
    if "speech_position" not in val.columns:
        val["speech_position"] = 0.5
    if "speech_position" not in test.columns:
        test["speech_position"] = 0.5
    train["speech_position"] = train["speech_position"].fillna(0.5)
    val["speech_position"] = val["speech_position"].fillna(0.5)
    test["speech_position"] = test["speech_position"].fillna(0.5)

    for df in (train, val, test):
        if "kabinetsappreciatie" not in df.columns or "zaak_soort" not in df.columns:
            enriched = enrich_with_zaak_features(df)
            df["kabinetsappreciatie"] = enriched["kabinetsappreciatie"]
            df["zaak_soort"] = enriched["zaak_soort"]
        df["_year"] = pd.to_datetime(df["datum"], errors="coerce").dt.year
        df["is_coalition"] = df.apply(
            lambda r: 1 if r.get("fractie") in _COALITION_PARTIES.get(int(r["_year"]) if pd.notna(r["_year"]) else 2020, set()) else 0,
            axis=1,
        )
        df.drop(columns=["_year"], errors="ignore", inplace=True)

    return train, val, test


def cluster_topics(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_clusters: int = 20,
    max_features: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cluster votes into policy domains using TF-IDF + KMeans on besluit_tekst + agendapunt_onderwerp.
    Adds topic_cluster column (0..n_clusters-1) to each dataframe.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    def _text(row):
        b = str(row.get("besluit_tekst") or "")[:500]
        a = str(row.get("agendapunt_onderwerp") or "")[:300]
        return f"{b} {a}".strip() or "onbekend"

    train = train.copy()
    val = val.copy()
    test = test.copy()

    texts_train = train.apply(_text, axis=1)
    texts_val = val.apply(_text, axis=1)
    texts_test = test.apply(_text, axis=1)

    tfidf = TfidfVectorizer(max_features=max_features, max_df=0.95, min_df=2)
    X_train = tfidf.fit_transform(texts_train.astype(str))
    X_val = tfidf.transform(texts_val.astype(str))
    X_test = tfidf.transform(texts_test.astype(str))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train["topic_cluster"] = kmeans.fit_predict(X_train)
    val["topic_cluster"] = kmeans.predict(X_val)
    test["topic_cluster"] = kmeans.predict(X_test)

    return train, val, test


def build_historical_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    fractie_col: str = "fractie",
    vote_col: str = "vote",
    topic_col: str = "topic_cluster",
    persoon_col: str = "persoon_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add historical voting features from training data only (no leakage).
    - party_domain_voor_rate: % Voor by party in this topic cluster
    - party_domain_vote_count: votes by party in this topic
    - party_recent_voor_rate: party Voor rate (rolling, from train)
    - speaker_topic_loyalty: does this speaker break from party on this topic?
    """
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train_sub = train[train[vote_col].isin(["Voor", "Tegen"])].copy()
    if "topic_cluster" not in train_sub.columns:
        return train, val, test

    # Party-domain stats (from train)
    party_domain = (
        train_sub.groupby([fractie_col, topic_col])[vote_col]
        .agg(lambda x: (x == "Voor").mean())
        .reset_index()
        .rename(columns={vote_col: "party_domain_voor_rate"})
    )
    party_domain_count = (
        train_sub.groupby([fractie_col, topic_col]).size().reset_index(name="party_domain_vote_count")
    )
    party_domain = party_domain.merge(party_domain_count, on=[fractie_col, topic_col])

    def _lookup_party_domain(df):
        merged = df[[fractie_col, topic_col]].merge(
            party_domain, on=[fractie_col, topic_col], how="left"
        )
        return merged["party_domain_voor_rate"].fillna(0.5), merged["party_domain_vote_count"].fillna(0)

    for df in (train, val, test):
        vr, vc = _lookup_party_domain(df)
        df["party_domain_voor_rate"] = vr.values
        df["party_domain_vote_count"] = vc.values

    # Party recent rate (simple: overall party rate from train)
    party_overall = (
        train_sub.groupby(fractie_col)[vote_col]
        .agg(lambda x: (x == "Voor").mean())
        .to_dict()
    )
    for df in (train, val, test):
        df["party_recent_voor_rate"] = df[fractie_col].map(lambda x: party_overall.get(x, 0.5))

    # Speaker-topic loyalty: does this speaker vote with party majority on this topic?
    party_topic_majority = (
        train_sub.groupby([fractie_col, topic_col])[vote_col]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Voor")
        .to_dict()
    )
    train_sub["_maj"] = train_sub.apply(
        lambda r: party_topic_majority.get((r[fractie_col], r[topic_col]), "Voor"), axis=1
    )
    train_sub["_match"] = (train_sub[vote_col] == train_sub["_maj"]).astype(float)
    speaker_topic = (
        train_sub.groupby([persoon_col, topic_col])["_match"]
        .mean()
        .reset_index()
        .rename(columns={"_match": "speaker_topic_loyalty"})
    )

    def _lookup_speaker_topic(df):
        merged = df[[persoon_col, topic_col]].merge(
            speaker_topic, on=[persoon_col, topic_col], how="left"
        )
        return merged["speaker_topic_loyalty"].fillna(0.5)

    for df in (train, val, test):
        df["speaker_topic_loyalty"] = _lookup_speaker_topic(df).values

    return train, val, test
