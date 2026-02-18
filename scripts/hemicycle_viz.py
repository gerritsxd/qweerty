#!/usr/bin/env python3
"""
Tweede Kamer Hemicycle Visualization
=====================================
Shows ALL 150 seats colored by predicted voting probability (Voor/Tegen).

For each seat:
  - Speakers who debated: direct RobBERT model prediction
  - Same-party colleagues: party-average from speakers
  - Parties with no speakers: historical party bias from test set

Usage:
    python scripts/hemicycle_viz.py                  # Best vote from test set
    python scripts/hemicycle_viz.py --no-model       # Actual votes only
    python scripts/hemicycle_viz.py --save out.png
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "processed"

from src.person_matcher import PersonMatcher

PARTY_COLORS = {
    "PVV":              "#002F6C",
    "Nieuw Sociaal Contract": "#003D6A",
    "NSC":              "#003D6A",
    "VVD":              "#FF6600",
    "BBB":              "#94C11F",
    "GroenLinks-PvdA":  "#5FA525",
    "GL":               "#5FA525",
    "GroenLinks":       "#5FA525",
    "PvdA":             "#E12B1A",
    "CDA":              "#007B5F",
    "SP":               "#ED1C24",
    "D66":              "#00AA55",
    "FVD":              "#8B1A1A",
    "PvdD":             "#006C2E",
    "ChristenUnie":     "#00A3E0",
    "SGP":              "#E85B0A",
    "DENK":             "#00A7E1",
    "Volt":             "#502379",
    "JA21":             "#142856",
    "BIJ1":             "#FFD700",
    "50PLUS":           "#93117E",
    "Groep Van Haga":   "#6B3FA0",
}

PARTY_ORDER = [
    "SP", "PvdD", "BIJ1", "DENK", "GroenLinks-PvdA", "GL",
    "GroenLinks", "PvdA", "Volt", "D66", "50PLUS",
    "Nieuw Sociaal Contract", "NSC",
    "CDA", "ChristenUnie", "VVD", "SGP", "BBB",
    "JA21", "PVV", "FVD", "Groep Van Haga",
]


# ─── Geometry ────────────────────────────────────────────────────────────

def build_hemicycle_coords(n_seats, n_rows=9, angle_spread=np.pi * 0.82,
                           r_inner=3.2, r_outer=7.5):
    """Generate (x, y) for seats in a hemicycle arc layout."""
    row_radii = np.linspace(r_inner, r_outer, n_rows)
    arc_lengths = row_radii * angle_spread
    raw_per_row = arc_lengths / arc_lengths.sum() * n_seats
    seats_per_row = np.round(raw_per_row).astype(int)

    diff = n_seats - seats_per_row.sum()
    for i in range(abs(diff)):
        idx = (n_rows - 1 - i) % n_rows if diff > 0 else i % n_rows
        seats_per_row[idx] += 1 if diff > 0 else -1

    coords = []
    for r, n in zip(row_radii, seats_per_row):
        start = (np.pi - angle_spread) / 2
        end = np.pi - start
        for a in np.linspace(start, end, n):
            coords.append((r * np.cos(a), r * np.sin(a)))
    return np.array(coords)


# ─── Data loading ────────────────────────────────────────────────────────

def load_chamber_for_besluit(besluit_id: str):
    """Build the full chamber for a Besluit (all voters + their actual votes)."""
    stemming = pd.read_parquet(DATA / "Stemming.parquet")
    persoon = pd.read_parquet(DATA / "Persoon.parquet")
    fzp = pd.read_parquet(DATA / "FractieZetelPersoon.parquet")
    fzs = pd.read_parquet(DATA / "FractieZetel.parquet")
    fractie_tbl = pd.read_parquet(DATA / "Fractie.parquet")

    votes = stemming[stemming["Besluit_Id"] == besluit_id].copy()
    if votes.empty:
        print(f"  WARNING: No Stemming records for Besluit {besluit_id}")
        return pd.DataFrame()

    has_individual = votes["Persoon_Id"].notna().sum() > 10

    if has_individual:
        ind = votes[votes["Persoon_Id"].notna()].copy()
        members = ind.merge(
            persoon[["Id", "Achternaam", "Tussenvoegsel", "Roepnaam"]],
            left_on="Persoon_Id", right_on="Id", how="left", suffixes=("", "_p"),
        )
        members["display_name"] = members.apply(
            lambda r: f"{r.get('Roepnaam', '') or ''} {r.get('Tussenvoegsel', '') or ''} {r.get('Achternaam', '')}".strip(),
            axis=1,
        )
        return pd.DataFrame({
            "persoon_id": members["Persoon_Id"],
            "name": members["display_name"],
            "achternaam": members["Achternaam"],
            "party": members["ActorFractie"],
            "vote": members["Soort"],
        }).reset_index(drop=True)

    # Faction vote — expand to individual members
    faction_votes = votes[votes["ActorFractie"].notna()].set_index("ActorFractie")["Soort"].to_dict()

    fzp_full = (
        fzp.merge(fzs[["Id", "Fractie_Id"]], left_on="FractieZetel_Id", right_on="Id", suffixes=("", "_z"))
           .merge(fractie_tbl[["Id", "Afkorting"]], left_on="Fractie_Id", right_on="Id", suffixes=("", "_f"))
    )
    active = fzp_full[fzp_full["TotEnMet"].isna() | (fzp_full["TotEnMet"] > pd.Timestamp.now(tz="UTC"))]
    active = active[active["Afkorting"].isin(faction_votes.keys())]

    members = active.merge(
        persoon[["Id", "Achternaam", "Tussenvoegsel", "Roepnaam"]],
        left_on="Persoon_Id", right_on="Id", how="left", suffixes=("", "_p"),
    )
    members["display_name"] = members.apply(
        lambda r: f"{r.get('Roepnaam', '') or ''} {r.get('Tussenvoegsel', '') or ''} {r.get('Achternaam', '')}".strip(),
        axis=1,
    )
    members["vote"] = members["Afkorting"].map(faction_votes)

    return pd.DataFrame({
        "persoon_id": members["Persoon_Id"],
        "name": members["display_name"],
        "achternaam": members["Achternaam"],
        "party": members["Afkorting"],
        "vote": members["vote"],
    }).drop_duplicates(subset=["persoon_id"]).reset_index(drop=True)


def get_best_besluit(test_df):
    """Find the besluit in the test set with the most speakers + parties."""
    bc = test_df.groupby("besluit_id").agg(
        n_speakers=("achternaam", "nunique"),
        n_parties=("fractie", "nunique"),
        topic=("agendapunt_onderwerp", "first"),
        besluit=("besluit_tekst", "first"),
    ).reset_index()
    bc["score"] = bc["n_speakers"] * bc["n_parties"]

    stemming = pd.read_parquet(DATA / "Stemming.parquet")
    valid = set(stemming["Besluit_Id"].unique())
    bc = bc[bc["besluit_id"].isin(valid)]
    if bc.empty:
        raise ValueError("No test-set besluiten found in Stemming table")
    return bc.nlargest(1, "score").iloc[0]


# ─── Prediction logic ───────────────────────────────────────────────────

def predict_all_seats(model_dict, test_df, besluit_id, chamber_df, matcher):
    """
    Compute a probability for EVERY seat in the chamber.

    Strategy:
      1. Direct prediction — speakers who debated this topic
      2. Party propagation — same-party avg from speakers
      3. Historical party bias — from all test set predictions
    """
    from src.ml.models import predict_proba_model_robbert_batch

    # --- Step 1: Direct predictions for this besluit's speakers ---
    sub = test_df[test_df["besluit_id"] == besluit_id].copy()
    direct_probs = {}
    party_probs = {}
    if len(sub) > 0:
        probs = predict_proba_model_robbert_batch(model_dict, sub, max_length=512)
        sub["prob_voor"] = probs
        sub["canonical_pid"] = sub.apply(
            lambda r: matcher.match(r["achternaam"], r["fractie"]), axis=1
        )

        for _, row in sub.iterrows():
            cpid = row["canonical_pid"]
            if cpid and pd.notna(cpid):
                direct_probs[cpid] = row["prob_voor"]

            party = row["fractie"]
            party_probs.setdefault(party, []).append(row["prob_voor"])

    print(f"  Direct speaker predictions: {len(direct_probs)}")
    print(f"  Parties with speakers: {len(party_probs)}")

    # --- Step 2: Compute party-average from speakers ---
    party_avg = {}
    for party, probs_list in party_probs.items():
        party_avg[party] = np.mean(probs_list)

    # --- Step 3: Historical party bias from full test set ---
    print("  Computing historical party biases...", flush=True)
    party_bias = {}
    for party in chamber_df["party"].unique():
        party_test = test_df[test_df["fractie"] == party]
        if len(party_test) > 0:
            n_voor = (party_test["vote"] == "Voor").sum()
            party_bias[party] = n_voor / len(party_test)
        else:
            party_bias[party] = 0.5

    # --- Assign probabilities to every seat ---
    chamber = chamber_df.copy()
    chamber["prob_voor"] = np.nan
    chamber["prediction_source"] = "none"
    chamber["predicted"] = None

    for idx, row in chamber.iterrows():
        pid = row["persoon_id"]
        party = row["party"]

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

        chamber.at[idx, "prob_voor"] = prob
        chamber.at[idx, "prediction_source"] = source
        chamber.at[idx, "predicted"] = "Voor" if prob > 0.5 else "Tegen"

    for src in ["model", "party_avg", "historical", "default"]:
        n = (chamber["prediction_source"] == src).sum()
        if n > 0:
            print(f"    {src:12s}: {n:3d} seats")

    return chamber


# ─── Rendering ───────────────────────────────────────────────────────────

def plot_hemicycle(voters_df, besluit_info=None, save_path=None):
    """Render the full hemicycle with per-seat probabilities."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.set_aspect("equal")
    ax.axis("off")

    vote_cmap = LinearSegmentedColormap.from_list("vote",
        ["#d62828", "#e85d04", "#f9c74f", "#80b918", "#2a9d8f"])

    party_rank = {p: i for i, p in enumerate(PARTY_ORDER)}
    df = voters_df.copy()
    df["_rank"] = df["party"].map(party_rank).fillna(99)
    df = df.sort_values(["_rank", "name"]).reset_index(drop=True)

    n_seats = len(df)
    coords = build_hemicycle_coords(n_seats)
    if len(coords) < n_seats:
        coords = np.vstack([coords, np.zeros((n_seats - len(coords), 2))])
    df["x"] = coords[:n_seats, 0]
    df["y"] = coords[:n_seats, 1]

    # Draw all seats
    for _, row in df.iterrows():
        x, y = row["x"], row["y"]
        prob = row.get("prob_voor", 0.5)
        source = row.get("prediction_source", "none")
        actual = row.get("vote", "")

        color = vote_cmap(prob)

        if source == "model":
            edge_color = "white"
            edge_width = 2.0
            size = 260
            alpha = 1.0
            zorder = 5
        elif source == "party_avg":
            edge_color = "#ffffff60"
            edge_width = 1.0
            size = 180
            alpha = 0.85
            zorder = 4
        elif source in ("historical", "default"):
            edge_color = "#ffffff30"
            edge_width = 0.6
            size = 140
            alpha = 0.55
            zorder = 3
        else:
            if actual in ("Voor", "Tegen"):
                color = "#2a9d8f" if actual == "Voor" else "#d62828"
                alpha = 0.3
            else:
                color = PARTY_COLORS.get(row["party"], "#555555")
                alpha = 0.2
            edge_color = "#ffffff15"
            edge_width = 0.5
            size = 120
            zorder = 2

        ax.scatter(x, y, s=size, c=[color], alpha=alpha,
                   edgecolors=edge_color, linewidths=edge_width, zorder=zorder)

        # Actual-vote indicator triangle for model-predicted seats
        if source == "model" and actual in ("Voor", "Tegen"):
            marker = "^" if actual == "Voor" else "v"
            act_color = "#2a9d8f" if actual == "Voor" else "#d62828"
            ax.scatter(x, y - 0.42, s=40, c=[act_color], marker=marker,
                       alpha=0.95, edgecolors="none", zorder=6)

        # Probability label on model-predicted seats
        if source == "model":
            pct = f"{prob*100:.0f}%"
            txt_color = "white" if 0.25 < prob < 0.75 else "#cccccc"
            ax.text(x, y + 0.42, pct, ha="center", va="bottom",
                    fontsize=5, fontweight="bold", color=txt_color,
                    alpha=0.9, zorder=7)

    # Party labels along the outer edge
    party_groups = df.groupby("party").agg(
        x_mean=("x", "mean"), y_max=("y", "max"), y_min=("y", "min"),
        count=("name", "count"),
        avg_prob=("prob_voor", "mean"),
    ).reset_index()
    for _, p in party_groups.iterrows():
        if p["count"] >= 2:
            label = p["party"]
            # Shorten long names
            label = label.replace("Nieuw Sociaal Contract", "NSC")
            label = label.replace("GroenLinks-PvdA", "GL-PvdA")
            label = label.replace("ChristenUnie", "CU")

            pct = f"{p['avg_prob']*100:.0f}% V"
            ax.text(p["x_mean"], p["y_min"] - 0.55, label,
                    ha="center", va="top", fontsize=6.5, color="white",
                    fontweight="bold", alpha=0.7, zorder=8)
            ax.text(p["x_mean"], p["y_min"] - 0.95, pct,
                    ha="center", va="top", fontsize=5.5, color="#f9c74f",
                    fontweight="normal", alpha=0.6, zorder=8)

    # Title
    title = "TWEEDE KAMER \u2014 Predicted Vote Probabilities"
    if besluit_info:
        topic = str(besluit_info.get("topic", ""))[:100]
        besluit_txt = str(besluit_info.get("besluit", ""))[:120]
        title += f"\n{topic}"
        ax.text(0, -0.8, besluit_txt, ha="center", va="top",
                fontsize=9, color="#888888", style="italic", zorder=8)

    ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=18)

    # Podium
    podium = mpatches.FancyBboxPatch((-1.5, -0.2), 3.0, 0.8,
                                      boxstyle="round,pad=0.18",
                                      facecolor="#12122a", edgecolor="#ffffff15",
                                      linewidth=1.5, zorder=2)
    ax.add_patch(podium)
    ax.text(0, 0.18, "V O O R Z I T T E R", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#ffffff40", zorder=3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=vote_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.10, 0.06, 0.35, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["100% Tegen", "75%", "Toss-up", "75%", "100% Voor"])
    cbar.ax.tick_params(labelsize=8, colors="white", length=0)
    cbar.outline.set_edgecolor("#ffffff20")

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="#0a0a1a", markerfacecolor=vote_cmap(0.85),
                   markersize=12, markeredgecolor="white", markeredgewidth=2,
                   label="Speaker \u2014 direct model prediction"),
        plt.Line2D([0], [0], marker="o", color="#0a0a1a", markerfacecolor=vote_cmap(0.7),
                   markersize=10, markeredgecolor="#ffffff60", markeredgewidth=1,
                   label="Same party \u2014 party-average prediction"),
        plt.Line2D([0], [0], marker="o", color="#0a0a1a", markerfacecolor=vote_cmap(0.5),
                   markersize=8, markeredgecolor="#ffffff30", markeredgewidth=0.6,
                   label="No speakers \u2014 historical party bias"),
        plt.Line2D([0], [0], marker="^", color="#2a9d8f", linestyle="None",
                   markersize=8, label="Actual vote: Voor"),
        plt.Line2D([0], [0], marker="v", color="#d62828", linestyle="None",
                   markersize=8, label="Actual vote: Tegen"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=8, facecolor="#0a0a1a", edgecolor="#ffffff20",
              labelcolor="white", framealpha=0.95, borderpad=1.2)

    # Stats box
    n_total = len(df)
    n_model = (df["prediction_source"] == "model").sum()
    n_party = (df["prediction_source"] == "party_avg").sum()
    n_hist = (df["prediction_source"].isin(["historical", "default"])).sum()
    actual_voor = (df["vote"] == "Voor").sum()
    actual_tegen = (df["vote"] == "Tegen").sum()
    pred_voor = (df["prob_voor"] > 0.5).sum()
    pred_tegen = (df["prob_voor"] <= 0.5).sum()

    correct_all = (df["predicted"] == df["vote"]).sum()
    acc_all = correct_all / n_total * 100

    if n_model > 0:
        model_mask = df["prediction_source"] == "model"
        correct_model = (df.loc[model_mask, "predicted"] == df.loc[model_mask, "vote"]).sum()
        acc_model = correct_model / n_model * 100
        model_line = f"Speaker accuracy:  {acc_model:.0f}% ({correct_model}/{n_model})"
    else:
        model_line = "Speaker accuracy:  N/A"

    stats_lines = [
        f"Predicted:  {pred_voor:3d} Voor / {pred_tegen:3d} Tegen",
        f"Actual:     {actual_voor:3d} Voor / {actual_tegen:3d} Tegen",
        f"",
        f"Overall accuracy:  {acc_all:.0f}% ({correct_all}/{n_total})",
        model_line,
        f"",
        f"Sources:  {n_model} direct  {n_party} party-avg  {n_hist} historical",
    ]
    ax.text(0.99, 0.02, "\n".join(stats_lines), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="#cccccc",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#0a0a1a",
                      edgecolor="#ffffff20", alpha=0.95),
            zorder=10)

    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.02, right=0.98)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tweede Kamer hemicycle vote visualization")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()

    matcher = PersonMatcher.from_parquet(DATA)
    print(f"PersonMatcher loaded ({len(matcher._lookup)} lookup keys)")

    print("Loading test data...", flush=True)
    from src.ml.features import load_pairs, get_train_val_test, build_basic_features
    df = load_pairs()
    df = df[df["datum"].notna()]
    df = build_basic_features(df)
    _, _, test = get_train_val_test(df)
    test = test[test["vote"].isin(["Voor", "Tegen"])]
    print(f"  Test set: {len(test):,} records")

    print("Finding best vote to visualize...", flush=True)
    sample = get_best_besluit(test)
    besluit_id = sample["besluit_id"]
    print(f"  Topic: {sample['topic']}")
    print(f"  {sample['n_speakers']} speakers, {sample['n_parties']} parties")
    print(f"  Besluit: {str(sample['besluit'])[:100]}")

    print("Loading full chamber...", flush=True)
    chamber = load_chamber_for_besluit(besluit_id)
    print(f"  {len(chamber)} MPs in chamber "
          f"({(chamber['vote']=='Voor').sum()} Voor, "
          f"{(chamber['vote']=='Tegen').sum()} Tegen)")

    besluit_info = {
        "topic": sample["topic"],
        "besluit": sample["besluit"],
        "n_speakers": sample["n_speakers"],
        "n_parties": sample["n_parties"],
    }

    if not args.no_model:
        print("Loading RobBERT model...", flush=True)
        from src.ml.models import load_model_robbert
        model_path = ROOT / "models" / "robbert_v2"
        if not (model_path / "model.pt").exists():
            model_path = ROOT / "models" / "robbert_vote_classifier"
        model = load_model_robbert(str(model_path))

        print("Predicting for ALL seats...", flush=True)
        chamber = predict_all_seats(model, test, besluit_id, chamber, matcher)

        import torch
        del model
        torch.cuda.empty_cache()
    else:
        chamber["prob_voor"] = chamber["vote"].map({"Voor": 1.0, "Tegen": 0.0}).fillna(0.5)
        chamber["prediction_source"] = "actual"
        chamber["predicted"] = chamber["vote"]

    save_path = args.save or str(ROOT / "outputs" / "hemicycle_vote.png")
    print("Rendering hemicycle...", flush=True)
    plot_hemicycle(chamber, besluit_info, save_path=save_path)
    print("Done!")


if __name__ == "__main__":
    main()
