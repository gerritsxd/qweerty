#!/usr/bin/env python3
"""Pre-compute hemicycle data for the top besluiten in the test set."""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "processed"

from src.person_matcher import PersonMatcher

TOP_N = 20

def main():
    matcher = PersonMatcher.from_parquet(DATA)
    print(f"PersonMatcher: {len(matcher._lookup)} keys")

    from src.ml.features import load_pairs, get_train_val_test, build_basic_features
    df = load_pairs()
    df = df[df["datum"].notna()]
    df = build_basic_features(df)
    _, _, test = get_train_val_test(df)
    test = test[test["vote"].isin(["Voor", "Tegen"])]
    print(f"Test set: {len(test):,}")

    stemming = pd.read_parquet(DATA / "Stemming.parquet")
    persoon = pd.read_parquet(DATA / "Persoon.parquet")
    fzp = pd.read_parquet(DATA / "FractieZetelPersoon.parquet")
    fzs = pd.read_parquet(DATA / "FractieZetel.parquet")
    fractie_tbl = pd.read_parquet(DATA / "Fractie.parquet")

    # Find top besluiten
    bc = test.groupby("besluit_id").agg(
        n_speakers=("achternaam", "nunique"),
        n_parties=("fractie", "nunique"),
        topic=("agendapunt_onderwerp", "first"),
        besluit=("besluit_tekst", "first"),
        datum=("datum", "first"),
    ).reset_index()
    bc["score"] = bc["n_speakers"] * bc["n_parties"]
    valid = set(stemming["Besluit_Id"].unique())
    bc = bc[bc["besluit_id"].isin(valid)]
    bc = bc.nlargest(TOP_N, "score")
    print(f"Top {len(bc)} besluiten selected")

    # Load model
    from src.ml.models import load_model_robbert, predict_proba_model_robbert_batch
    model_path = ROOT / "models" / "robbert_v2"
    if not (model_path / "model.pt").exists():
        model_path = ROOT / "models" / "robbert_vote_classifier"
    model = load_model_robbert(str(model_path))

    # Historical party bias
    party_bias = {}
    for party in test["fractie"].unique():
        pt = test[test["fractie"] == party]
        party_bias[party] = float((pt["vote"] == "Voor").mean())

    # Build chamber members (active)
    fzp_full = (
        fzp.merge(fzs[["Id", "Fractie_Id"]], left_on="FractieZetel_Id", right_on="Id", suffixes=("", "_z"))
           .merge(fractie_tbl[["Id", "Afkorting"]], left_on="Fractie_Id", right_on="Id", suffixes=("", "_f"))
    )
    active = fzp_full[fzp_full["TotEnMet"].isna() | (fzp_full["TotEnMet"] > pd.Timestamp.now(tz="UTC"))]
    active_members = active.merge(
        persoon[["Id", "Achternaam", "Tussenvoegsel", "Roepnaam"]],
        left_on="Persoon_Id", right_on="Id", how="left", suffixes=("", "_p"),
    )
    active_members["display_name"] = active_members.apply(
        lambda r: f"{r.get('Roepnaam','') or ''} {r.get('Tussenvoegsel','') or ''} {r['Achternaam']}".strip(),
        axis=1,
    )

    output = {"besluiten": [], "parties": {}}

    for _, row in bc.iterrows():
        bid = row["besluit_id"]
        topic = str(row["topic"] or "")[:120]
        besluit_txt = str(row["besluit"] or "")
        datum = str(row["datum"])[:10] if pd.notna(row["datum"]) else ""
        n_speakers = int(row["n_speakers"])
        n_parties = int(row["n_parties"])

        print(f"\n  [{bid[:8]}] {topic[:60]}...")

        # Get votes for this besluit
        votes = stemming[stemming["Besluit_Id"] == bid]
        if votes.empty:
            continue

        has_individual = votes["Persoon_Id"].notna().sum() > 10
        faction_votes = votes[votes["ActorFractie"].notna()].set_index("ActorFractie")["Soort"].to_dict()

        # Build chamber for this besluit
        if has_individual:
            ind = votes[votes["Persoon_Id"].notna()]
            members = ind.merge(
                persoon[["Id", "Achternaam", "Tussenvoegsel", "Roepnaam"]],
                left_on="Persoon_Id", right_on="Id", how="left", suffixes=("", "_p"),
            )
            members["display_name"] = members.apply(
                lambda r: f"{r.get('Roepnaam','') or ''} {r.get('Tussenvoegsel','') or ''} {r['Achternaam']}".strip(),
                axis=1,
            )
            chamber = pd.DataFrame({
                "pid": members["Persoon_Id"],
                "name": members["display_name"],
                "party": members["ActorFractie"],
                "vote": members["Soort"],
            }).drop_duplicates(subset=["pid"])
        else:
            relevant = active_members[active_members["Afkorting"].isin(faction_votes.keys())]
            chamber = pd.DataFrame({
                "pid": relevant["Persoon_Id"],
                "name": relevant["display_name"],
                "party": relevant["Afkorting"],
                "vote": relevant["Afkorting"].map(faction_votes),
            }).drop_duplicates(subset=["pid"])

        # Get predictions
        sub = test[test["besluit_id"] == bid]
        direct_probs = {}
        party_probs_local = {}
        if len(sub) > 0:
            probs = predict_proba_model_robbert_batch(model, sub, max_length=512)
            sub = sub.copy()
            sub["prob_voor"] = probs
            sub["cpid"] = sub.apply(lambda r: matcher.match(r["achternaam"], r["fractie"]), axis=1)
            for _, s in sub.iterrows():
                cpid = s["cpid"]
                if cpid and pd.notna(cpid):
                    direct_probs[cpid] = float(s["prob_voor"])
                party_probs_local.setdefault(s["fractie"], []).append(float(s["prob_voor"]))

        party_avg = {p: float(np.mean(v)) for p, v in party_probs_local.items()}

        seats = []
        for _, m in chamber.iterrows():
            pid = m["pid"]
            party = m["party"]
            if pid in direct_probs:
                prob = direct_probs[pid]
                source = "model"
            elif party in party_avg:
                prob = party_avg[party]
                source = "party_avg"
            elif party in party_bias:
                prob = party_bias[party]
                source = "historical"
            else:
                prob = 0.5
                source = "default"

            seats.append({
                "name": m["name"],
                "party": party,
                "prob": round(prob, 3),
                "source": source,
                "vote": m["vote"],
            })

        n_voor = sum(1 for s in seats if s["vote"] == "Voor")
        n_tegen = sum(1 for s in seats if s["vote"] == "Tegen")
        n_direct = sum(1 for s in seats if s["source"] == "model")
        correct = sum(1 for s in seats if (s["prob"] > 0.5 and s["vote"] == "Voor") or (s["prob"] <= 0.5 and s["vote"] == "Tegen"))

        output["besluiten"].append({
            "id": bid,
            "topic": topic,
            "besluit": besluit_txt,
            "datum": datum,
            "n_speakers": n_speakers,
            "n_parties": n_parties,
            "n_voor": n_voor,
            "n_tegen": n_tegen,
            "n_direct": n_direct,
            "accuracy": round(correct / max(len(seats), 1) * 100, 1),
            "seats": seats,
        })
        print(f"    {len(seats)} seats, {n_direct} direct, acc={correct}/{len(seats)}")

    out_path = ROOT / "outputs" / "hemicycle_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    # Embed JSON into HTML so it works when opened via file:// (no CORS)
    html_path = ROOT / "outputs" / "hemicycle.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        data_str = json.dumps(output, ensure_ascii=False)
        if "__HEMICYCLE_DATA__" in html:
            html = html.replace("__HEMICYCLE_DATA__", data_str)
            html_path.write_text(html, encoding="utf-8")
            print(f"Updated {html_path} with embedded data ({len(html) // 1024} KB)")

if __name__ == "__main__":
    main()
