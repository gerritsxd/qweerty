#!/usr/bin/env python3
"""
Tweede Kamer Data — Overview Dashboard
=======================================
Generates a multi-panel visual overview of all fetched parliamentary data.

Usage:
    python visualize.py                  # Generate and save dashboard
    python visualize.py --open           # Generate and open in viewer
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Style ──────────────────────────────────────────────────────────────────

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_COLOR = "#e6edf3"
ACCENT = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#7ee787"
ACCENT4 = "#d2a8ff"
ACCENT5 = "#ff7b72"
GRID_COLOR = "#21262d"
MUTED = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlepad": 14,
    "text.color": TEXT_COLOR,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


DATA_DIR = Path("data/processed")
OUT_FILE = Path("dashboard.png")


# ─── Helper: load parquet safely ────────────────────────────────────────────

def load(name: str) -> pd.DataFrame:
    f = DATA_DIR / f"{name}.parquet"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_parquet(f)


def format_k(x, _):
    """Format tick labels with k suffix."""
    if x >= 1000:
        return f"{x/1000:.0f}k"
    return f"{int(x)}"


def add_panel_label(ax, label, x=-0.02, y=1.08):
    """Add a letter label (A, B, C...) to a subplot."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=ACCENT,
            va="bottom", ha="left")


# ─── Load all data ──────────────────────────────────────────────────────────

def main():
    print("Loading data...")

    fracties = load("Fractie")
    personen = load("Persoon")
    vergaderingen = load("Vergadering")
    toezeggingen = load("Toezegging")
    reizen = load("PersoonReis")
    geschenken = load("PersoonGeschenk")
    dossiers = load("Kamerstukdossier")
    commissies = load("Commissie")
    nevenfuncties = load("PersoonNevenfunctie")
    zetel_persoon = load("FractieZetelPersoon")
    loopbaan = load("PersoonLoopbaan")

    # ─── Figure layout: 4 rows x 3 cols ─────────────────────────────────
    fig = plt.figure(figsize=(22, 28))
    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.35,
        wspace=0.30,
        left=0.06, right=0.97,
        top=0.93, bottom=0.03,
    )

    # ─── Title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Tweede Kamer Open Data — Dashboard",
        fontsize=26, fontweight="bold", color=TEXT_COLOR,
        y=0.97,
    )
    fig.text(
        0.5, 0.952,
        f"15 entities  ·  {len(personen):,} persons  ·  {len(vergaderingen):,} meetings  ·  "
        f"{len(dossiers):,} dossiers  ·  {len(toezeggingen):,} promises  ·  "
        f"{len(geschenken):,} gifts  ·  {len(reizen):,} trips",
        ha="center", fontsize=12, color=MUTED,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # A — Dataset size overview (horizontal bar)
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 0])
    add_panel_label(ax, "A")

    datasets = {}
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("_"):
            continue
        df = pd.read_parquet(f)
        datasets[f.stem] = len(df)
    ds = pd.Series(datasets).sort_values()

    colors = [ACCENT if v > 1000 else MUTED for v in ds.values]
    bars = ax.barh(range(len(ds)), ds.values, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels(ds.index, fontsize=9)
    ax.set_xlabel("Records")
    ax.set_title("Dataset Sizes")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_k))
    ax.grid(axis="x", alpha=0.3)

    # Add count labels
    for i, (name, val) in enumerate(ds.items()):
        ax.text(val + max(ds.values) * 0.01, i, f"{val:,}",
                va="center", fontsize=8, color=MUTED)

    # ═══════════════════════════════════════════════════════════════════════
    # B — Current party seats (top 15)
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 1])
    add_panel_label(ax, "B")

    active = fracties[fracties["DatumInactief"].isna() & fracties["AantalZetels"].notna()]
    active = active[active["AantalZetels"] > 0].sort_values("AantalZetels", ascending=True).tail(15)

    # Party colors (approximate Dutch political palette)
    party_colors = {
        "VVD": "#FF6600", "D66": "#00AA55", "PVV": "#002F6C",
        "CDA": "#007B5F", "SP": "#FF0000", "GroenLinks-PvdA": "#CC0033",
        "ChristenUnie": "#00AEEF", "SGP": "#FF6600", "DENK": "#00B4D8",
        "PvdD": "#006B3F", "FVD": "#8B2252", "JA21": "#1B3A5C",
        "BBB": "#92C83E", "Volt": "#502379", "50PLUS": "#93328E",
        "Groep Markuszower": "#666666", "Nieuw Sociaal Contract": "#0066CC",
    }
    bar_colors = [party_colors.get(a, ACCENT) for a in active["Afkorting"]]

    ax.barh(range(len(active)), active["AantalZetels"].values,
            color=bar_colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(active)))
    ax.set_yticklabels(active["Afkorting"].values, fontsize=9)
    ax.set_xlabel("Seats")
    ax.set_title("Current Party Seats in Tweede Kamer")
    ax.grid(axis="x", alpha=0.3)

    for i, val in enumerate(active["AantalZetels"].values):
        ax.text(val + 0.3, i, f"{int(val)}", va="center", fontsize=9, color=TEXT_COLOR)

    # ═══════════════════════════════════════════════════════════════════════
    # C — Gender distribution
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 2])
    add_panel_label(ax, "C")

    gender = personen["Geslacht"].value_counts()
    wedge_colors = [ACCENT, ACCENT2, ACCENT4]
    wedges, texts, autotexts = ax.pie(
        gender.values,
        labels=gender.index,
        colors=wedge_colors[:len(gender)],
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(gender.values) / 100)):,})",
        startangle=90,
        textprops={"color": TEXT_COLOR, "fontsize": 11},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
        pctdistance=0.65,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(TEXT_COLOR)
    ax.set_title("Gender — All Parliamentarians")

    # ═══════════════════════════════════════════════════════════════════════
    # D — Meetings per year (stacked by Soort)
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, :2])
    add_panel_label(ax, "D")

    vj = vergaderingen.copy()
    vj["year"] = vj["Vergaderjaar"].str[:4].astype(float)
    vj = vj[vj["year"] >= 2013]
    pivot = vj.groupby(["Vergaderjaar", "Soort"]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()

    bottom = np.zeros(len(pivot))
    soort_colors = {"Commissie": ACCENT, "Plenair": ACCENT2}
    for col in pivot.columns:
        c = soort_colors.get(col, ACCENT4)
        ax.bar(range(len(pivot)), pivot[col].values, bottom=bottom,
               label=col, color=c, edgecolor="none", width=0.7)
        bottom += pivot[col].values

    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Meetings")
    ax.set_title("Parliamentary Meetings per Session Year")
    ax.legend(loc="upper left", framealpha=0.7,
              facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ═══════════════════════════════════════════════════════════════════════
    # E — Promises by status (donut)
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 2])
    add_panel_label(ax, "E")

    status = toezeggingen["Status"].value_counts()
    status_colors = [ACCENT3, ACCENT2, ACCENT4, MUTED]
    wedges, texts, autotexts = ax.pie(
        status.values,
        labels=status.index,
        colors=status_colors[:len(status)],
        autopct=lambda p: f"{p:.0f}%",
        startangle=140,
        textprops={"color": TEXT_COLOR, "fontsize": 10},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2, "width": 0.55},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(TEXT_COLOR)
    centre_circle = plt.Circle((0, 0), 0.45, fc=PANEL_BG)
    ax.add_artist(centre_circle)
    ax.text(0, 0, f"{len(toezeggingen):,}\npromises", ha="center", va="center",
            fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_title("Ministerial Promises — Status")

    # ═══════════════════════════════════════════════════════════════════════
    # F — Gifts per year
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[2, 0])
    add_panel_label(ax, "F")

    g = geschenken.copy()
    g["year"] = pd.to_datetime(g["Datum"], utc=True, errors="coerce").dt.year
    g = g[g["year"].between(2007, 2025)]
    yearly = g.groupby("year").size()

    ax.fill_between(yearly.index, yearly.values, alpha=0.3, color=ACCENT3)
    ax.plot(yearly.index, yearly.values, color=ACCENT3, linewidth=2.5, marker="o",
            markersize=5, markerfacecolor=ACCENT3, markeredgecolor=DARK_BG)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gifts Registered")
    ax.set_title("Gifts Received by MPs per Year")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(2007, 2025)

    # ═══════════════════════════════════════════════════════════════════════
    # G — Top 12 travel destinations
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[2, 1])
    add_panel_label(ax, "G")

    dest = reizen["Bestemming"].value_counts().head(12).sort_values()
    # Shorten labels for readability
    short = dest.index.str.replace(", België", " (BE)", regex=False)\
                        .str.replace(", Duitsland", " (DE)", regex=False)\
                        .str.replace(", Frankrijk", " (FR)", regex=False)\
                        .str.replace("Verenigde Staten", "USA", regex=False)\
                        .str.replace(", Oostenrijk", " (AT)", regex=False)\
                        .str.replace(", Verenigd Koninkrijk", " (UK)", regex=False)\
                        .str.replace(", Engeland", " (UK)", regex=False)\
                        .str.replace(", Italië", " (IT)", regex=False)\
                        .str.replace(", Spanje", " (ES)", regex=False)

    ax.barh(range(len(dest)), dest.values, color=ACCENT4, edgecolor="none", height=0.65)
    ax.set_yticks(range(len(dest)))
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Trips")
    ax.set_title("Top MP Travel Destinations")
    ax.grid(axis="x", alpha=0.3)

    for i, val in enumerate(dest.values):
        ax.text(val + 3, i, str(val), va="center", fontsize=8, color=MUTED)

    # ═══════════════════════════════════════════════════════════════════════
    # H — Promises by ministry (top 10)
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[2, 2])
    add_panel_label(ax, "H")

    ministry = toezeggingen["Ministerie"].value_counts().head(10).sort_values()
    # Shorten long ministry names
    short_names = ministry.index\
        .str.replace("Volksgezondheid, Welzijn en Sport", "VWS", regex=False)\
        .str.replace("Infrastructuur en Waterstaat", "I&W", regex=False)\
        .str.replace("Justitie en Veiligheid", "J&V", regex=False)\
        .str.replace("Binnenlandse Zaken en Koninkrijksrelaties", "BZK", regex=False)\
        .str.replace("Onderwijs, Cultuur en Wetenschap", "OCW", regex=False)\
        .str.replace("Sociale Zaken en Werkgelegenheid", "SZW", regex=False)\
        .str.replace("Financiën", "Fin", regex=False)\
        .str.replace("Buitenlandse Zaken", "BuZa", regex=False)\
        .str.replace("Economische Zaken en Klimaat", "EZK", regex=False)\
        .str.replace("Landbouw, Natuur en Voedselkwaliteit", "LNV", regex=False)

    colors_h = plt.cm.cool(np.linspace(0.2, 0.8, len(ministry)))
    ax.barh(range(len(ministry)), ministry.values, color=colors_h, edgecolor="none", height=0.65)
    ax.set_yticks(range(len(ministry)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Promises")
    ax.set_title("Ministerial Promises by Department")
    ax.grid(axis="x", alpha=0.3)

    for i, val in enumerate(ministry.values):
        ax.text(val + 8, i, str(val), va="center", fontsize=8, color=MUTED)

    # ═══════════════════════════════════════════════════════════════════════
    # I — Birth decade distribution of parliamentarians
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[3, 0])
    add_panel_label(ax, "I")

    p = personen.copy()
    p["birth_decade"] = (pd.to_datetime(p["Geboortedatum"], utc=True, errors="coerce").dt.year // 10 * 10)
    decades = p["birth_decade"].dropna().astype(int).value_counts().sort_index()
    decades = decades[decades.index >= 1850]

    ax.bar(decades.index.astype(str), decades.values, color=ACCENT, edgecolor="none", width=0.7)
    ax.set_xlabel("Birth Decade")
    ax.set_ylabel("Persons")
    ax.set_title("Parliamentarians by Birth Decade")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # ═══════════════════════════════════════════════════════════════════════
    # J — Dossiers by Kamer
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[3, 1])
    add_panel_label(ax, "J")

    kamer = dossiers["Kamer"].value_counts()
    kamer_colors = [ACCENT, ACCENT2, ACCENT4]
    bars = ax.bar(range(len(kamer)), kamer.values, color=kamer_colors[:len(kamer)],
                  edgecolor="none", width=0.5)
    ax.set_xticks(range(len(kamer)))
    ax.set_xticklabels(kamer.index, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Dossiers")
    ax.set_title("Parliamentary Dossiers by Chamber")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_k))

    for i, val in enumerate(kamer.values):
        ax.text(i, val + 50, f"{val:,}", ha="center", fontsize=10, color=TEXT_COLOR, fontweight="bold")

    # ═══════════════════════════════════════════════════════════════════════
    # K — Previous careers of parliamentarians (top 12)
    # ═══════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[3, 2])
    add_panel_label(ax, "K")

    careers = loopbaan["Functie"].value_counts().head(12).sort_values()
    ax.barh(range(len(careers)), careers.values, color=ACCENT5, edgecolor="none", height=0.65)
    ax.set_yticks(range(len(careers)))
    ax.set_yticklabels(careers.index, fontsize=9)
    ax.set_xlabel("Persons")
    ax.set_title("Previous Careers of MPs (Top 12)")
    ax.grid(axis="x", alpha=0.3)

    for i, val in enumerate(careers.values):
        ax.text(val + 0.3, i, str(val), va="center", fontsize=8, color=MUTED)

    # ─── Save ────────────────────────────────────────────────────────────
    print(f"Saving dashboard to {OUT_FILE}...")
    fig.savefig(OUT_FILE, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Done! Dashboard saved: {OUT_FILE.resolve()}")
    print(f"  Size: {OUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
