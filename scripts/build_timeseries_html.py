"""Generate interactive time-series HTML: politically meaningful plots about
Dutch parliamentary voting dynamics, deployed to deploy/2kmer/timeseries.html."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, numpy as np
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "data" / "analysis"
DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy" / "2kmer"

pairs = pd.read_parquet(ANALYSIS_DIR / "speech_vote_pairs.parquet")
pairs["datum"] = pd.to_datetime(pairs["datum"], errors="coerce")
pairs["quarter"] = pairs["datum"].dt.to_period("Q")

vt = pairs[pairs["vote"].isin(["Voor", "Tegen"])].copy()
vt["is_voor"] = (vt["vote"] == "Voor").astype(int)
vt["quarter"] = vt["datum"].dt.to_period("Q")

# ── Coalition definitions ────────────────────────────────────────────────────
COALITIONS = [
    ("2012-11-05", "2017-10-26", "Rutte II",  ["VVD", "PvdA"]),
    ("2017-10-26", "2022-01-10", "Rutte III", ["VVD", "CDA", "D66", "ChristenUnie"]),
    ("2022-01-10", "2023-07-07", "Rutte IV",  ["VVD", "CDA", "D66", "ChristenUnie"]),
    ("2023-07-07", "2024-07-02", "Demissionair", []),
    ("2024-07-02", "2027-01-01", "Schoof",    ["PVV", "VVD", "NSC", "BBB"]),
]

def get_coalition_parties(date):
    for start, end, name, parties in COALITIONS:
        if pd.Timestamp(start) <= date < pd.Timestamp(end):
            return name, set(parties)
    return "Unknown", set()

vt["coal_name"], vt["coal_parties"] = zip(*vt["datum"].map(get_coalition_parties))
vt["is_coalition"] = vt.apply(lambda r: r["fractie"] in r["coal_parties"], axis=1)

# ── theme ────────────────────────────────────────────────────────────────────
BG = "#0a0a1a"
PAPER = "#0e0e24"
GRID = "rgba(255,255,255,0.06)"
FONT = dict(family="Segoe UI, system-ui, sans-serif", color="#e0e0e0")
ACCENT = "#f9c74f"
C_COAL = "#60a5fa"
C_OPP  = "#f87171"

PARTY_COLORS = {
    "VVD": "#ff6f00", "PVV": "#1a1a2e", "CDA": "#2e7d32", "D66": "#00e676",
    "SP": "#d50000", "PvdA": "#e53935", "GroenLinks": "#43a047", "PvdD": "#1b5e20",
    "ChristenUnie": "#1565c0", "SGP": "#f57f17", "DENK": "#00bfa5",
    "BBB": "#8d6e63", "NSC": "#5c6bc0", "FVD": "#6d4c41",
    "GroenLinks-PvdA": "#66bb6a", "JA21": "#283593", "Volt": "#7b1fa2",
    "BIJ1": "#f9a825", "50PLUS": "#7c4dff",
}
FALLBACK_COLORS = [
    "#60a5fa", "#f87171", "#34d399", "#fbbf24", "#a78bfa",
    "#fb923c", "#38bdf8", "#f472b6", "#4ade80", "#e879f9",
]

def pcolor(party, idx=0):
    return PARTY_COLORS.get(party, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])

def base_layout(**kw):
    d = dict(
        template="plotly_dark", paper_bgcolor=PAPER, plot_bgcolor=BG, font=FONT,
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
        legend=dict(bgcolor="rgba(14,14,36,0.9)", bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1),
        hovermode="x unified",
    )
    d.update(kw)
    return d

def add_coalition_bands(fig, ymin=0, ymax=100):
    band_colors = {
        "Rutte II": "rgba(255,111,0,0.07)", "Rutte III": "rgba(0,230,118,0.07)",
        "Rutte IV": "rgba(0,230,118,0.04)", "Demissionair": "rgba(255,255,255,0.03)",
        "Schoof": "rgba(96,165,250,0.07)",
    }
    for start, end, name, _ in COALITIONS:
        s, e = pd.Timestamp(start), min(pd.Timestamp(end), pd.Timestamp("2026-03-01"))
        if e < pairs["datum"].min() or s > pairs["datum"].max():
            continue
        fig.add_vrect(x0=s, x1=e, fillcolor=band_colors.get(name, "rgba(255,255,255,0.03)"),
                      line_width=0, layer="below")
        fig.add_annotation(x=s + (e - s) / 2, y=ymax, text=f"<b>{name}</b>",
                           showarrow=False, font=dict(size=10, color="rgba(255,255,255,0.45)"),
                           yanchor="top")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Coalition vs Opposition "Voor" rate over time
# ═══════════════════════════════════════════════════════════════════════════
coal_q = vt.groupby(["quarter", "is_coalition"])["is_voor"].mean().unstack() * 100
coal_q.index = coal_q.index.to_timestamp()
coal_q.columns = ["Opposition", "Coalition"]
# 2-quarter rolling smooth
for c in coal_q.columns:
    coal_q[c + "_smooth"] = coal_q[c].rolling(2, center=True, min_periods=1).mean()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=coal_q.index, y=coal_q["Coalition"], mode="markers",
    marker=dict(color=C_COAL, size=4, opacity=0.3), showlegend=False,
    hovertemplate="%{y:.1f}%<extra></extra>",
))
fig1.add_trace(go.Scatter(
    x=coal_q.index, y=coal_q["Coalition_smooth"], name="Coalition parties",
    line=dict(color=C_COAL, width=3),
    hovertemplate="%{y:.1f}%<extra></extra>",
))
fig1.add_trace(go.Scatter(
    x=coal_q.index, y=coal_q["Opposition"], mode="markers",
    marker=dict(color=C_OPP, size=4, opacity=0.3), showlegend=False,
    hovertemplate="%{y:.1f}%<extra></extra>",
))
fig1.add_trace(go.Scatter(
    x=coal_q.index, y=coal_q["Opposition_smooth"], name="Opposition parties",
    line=dict(color=C_OPP, width=3),
    hovertemplate="%{y:.1f}%<extra></extra>",
))
fig1.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.15)")
add_coalition_bands(fig1, ymax=95)
fig1.update_layout(**base_layout(
    title="Coalition vs Opposition: % Voting 'Voor' (quarterly)",
    yaxis=dict(title="% Voor", gridcolor=GRID, range=[25, 95]),
    height=480,
))

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: Individual party "Voor" rate trajectories
# ═══════════════════════════════════════════════════════════════════════════
focus_parties = ["VVD", "PVV", "D66", "CDA", "SP", "GroenLinks", "PvdA",
                 "ChristenUnie", "PvdD", "BBB"]
focus_parties = [p for p in focus_parties if p in vt["fractie"].values]

pq = vt[vt["fractie"].isin(focus_parties)].groupby(["quarter", "fractie"])["is_voor"].mean().unstack() * 100
pq.index = pq.index.to_timestamp()
# rolling smooth
pq_s = pq.rolling(2, center=True, min_periods=1).mean()

fig2 = go.Figure()
for i, party in enumerate(focus_parties):
    if party not in pq_s.columns:
        continue
    fig2.add_trace(go.Scatter(
        x=pq_s.index, y=pq_s[party], name=party,
        line=dict(color=pcolor(party, i), width=2.5),
        hovertemplate="%{y:.1f}%<extra>" + party + "</extra>",
    ))
add_coalition_bands(fig2, ymax=95)
fig2.update_layout(**base_layout(
    title="Party 'Voor' Rate Over Time (quarterly, smoothed)",
    yaxis=dict(title="% Voor", gridcolor=GRID, range=[20, 95]),
    height=520,
    legend=dict(x=1.02, y=1, xanchor="left"),
))

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: Vote contentiousness over time
# ═══════════════════════════════════════════════════════════════════════════
besluit_info = vt.groupby("besluit_id").agg(
    voor_pct=("is_voor", "mean"),
    date=("datum", "first"),
).dropna()
besluit_info["quarter"] = besluit_info["date"].dt.to_period("Q")
besluit_info["contested"] = (besluit_info["voor_pct"] > 0.3) & (besluit_info["voor_pct"] < 0.7)
besluit_info["unanimous"] = (besluit_info["voor_pct"] < 0.05) | (besluit_info["voor_pct"] > 0.95)

contest_q = besluit_info.groupby("quarter").agg(
    pct_contested=("contested", "mean"),
    pct_unanimous=("unanimous", "mean"),
    n_votes=("contested", "size"),
).reset_index()
contest_q["quarter_ts"] = contest_q["quarter"].dt.to_timestamp()
contest_q["pct_contested"] *= 100
contest_q["pct_unanimous"] *= 100

fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Bar(
    x=contest_q["quarter_ts"], y=contest_q["pct_contested"], name="Contested (30-70%)",
    marker_color="rgba(251,191,36,0.6)", hovertemplate="%{y:.1f}%<extra></extra>",
), secondary_y=False)
fig3.add_trace(go.Bar(
    x=contest_q["quarter_ts"], y=contest_q["pct_unanimous"], name="Unanimous (>95% or <5%)",
    marker_color="rgba(96,165,250,0.4)", hovertemplate="%{y:.1f}%<extra></extra>",
), secondary_y=False)
fig3.add_trace(go.Scatter(
    x=contest_q["quarter_ts"], y=contest_q["n_votes"], name="Total votes (right axis)",
    line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dot"),
    hovertemplate="%{y}<extra></extra>",
), secondary_y=True)
add_coalition_bands(fig3, ymax=68)
fig3.update_layout(**base_layout(
    title="How Contested Are Votes? (quarterly)",
    height=450,
    barmode="group",
))
fig3.update_yaxes(title_text="% of voting moments", gridcolor=GRID, secondary_y=False)
fig3.update_yaxes(title_text="Total voting moments", gridcolor=GRID, secondary_y=True)

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: Cross-party agreement heatmap (overall)
# ═══════════════════════════════════════════════════════════════════════════
heat_parties = ["VVD", "PVV", "CDA", "D66", "BBB", "NSC", "ChristenUnie",
                "SGP", "PvdA", "GroenLinks", "SP", "PvdD", "DENK"]
heat_parties = [p for p in heat_parties if p in vt["fractie"].values]
pb = vt[vt["fractie"].isin(heat_parties)].groupby(["besluit_id", "fractie"])["vote"].first().unstack()

agree_mat = pd.DataFrame(index=heat_parties, columns=heat_parties, dtype=float)
for p1 in heat_parties:
    for p2 in heat_parties:
        if p1 in pb.columns and p2 in pb.columns:
            mask = pb[p1].notna() & pb[p2].notna()
            agree_mat.loc[p1, p2] = (pb.loc[mask, p1] == pb.loc[mask, p2]).mean() * 100

fig4 = go.Figure(go.Heatmap(
    z=agree_mat.values.astype(float),
    x=heat_parties, y=heat_parties,
    colorscale=[[0, "#0a0a1a"], [0.45, "#1e3a5f"], [0.65, "#2563eb"], [0.85, "#f9c74f"], [1, "#ffffff"]],
    zmin=20, zmax=100,
    text=np.where(pd.isna(agree_mat.values), "", np.nan_to_num(agree_mat.values.astype(float), nan=0).round(0).astype(int).astype(str)),
    texttemplate="%{text}%", textfont=dict(size=10),
    hovertemplate="<b>%{x}</b> & <b>%{y}</b><br>Agreement: %{z:.1f}%<extra></extra>",
    colorbar=dict(title="% agree"),
))
fig4.update_layout(**base_layout(
    title="Cross-Party Voting Agreement (all time)",
    height=520,
    yaxis=dict(autorange="reversed", gridcolor=GRID),
    xaxis=dict(gridcolor=GRID),
))

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5: Key alliance shifts over time (agreement between specific pairs)
# ═══════════════════════════════════════════════════════════════════════════
alliance_pairs = [
    ("VVD", "PVV",  "VVD ↔ PVV"),
    ("VVD", "D66",  "VVD ↔ D66"),
    ("VVD", "CDA",  "VVD ↔ CDA"),
    ("SP",  "PvdD", "SP ↔ PvdD"),
    ("PVV", "BBB",  "PVV ↔ BBB"),
]
# filter to pairs where both exist
alliance_pairs = [(a, b, l) for a, b, l in alliance_pairs
                  if a in vt["fractie"].values and b in vt["fractie"].values]

pb_q = vt[vt["fractie"].isin(heat_parties)].copy()
pb_q["quarter"] = pb_q["datum"].dt.to_period("Q")
pb_full = pb_q.groupby(["quarter", "besluit_id", "fractie"])["vote"].first().reset_index()

alliance_colors = ["#ff6f00", "#00e676", "#2e7d32", "#d50000", "#8d6e63"]

fig5 = go.Figure()
for idx, (p1, p2, label) in enumerate(alliance_pairs):
    d1 = pb_full[pb_full["fractie"] == p1][["quarter", "besluit_id", "vote"]].rename(columns={"vote": "v1"})
    d2 = pb_full[pb_full["fractie"] == p2][["quarter", "besluit_id", "vote"]].rename(columns={"vote": "v2"})
    merged = d1.merge(d2, on=["quarter", "besluit_id"])
    merged["agree"] = merged["v1"] == merged["v2"]
    qa = merged.groupby("quarter")["agree"].mean().reset_index()
    qa["quarter_ts"] = qa["quarter"].dt.to_timestamp()
    qa["agree"] *= 100
    qa["smooth"] = qa["agree"].rolling(2, center=True, min_periods=1).mean()
    fig5.add_trace(go.Scatter(
        x=qa["quarter_ts"], y=qa["smooth"], name=label,
        line=dict(color=alliance_colors[idx % len(alliance_colors)], width=2.5),
        hovertemplate="%{y:.1f}%<extra>" + label + "</extra>",
    ))
add_coalition_bands(fig5, ymax=95)
fig5.update_layout(**base_layout(
    title="Shifting Alliances: Pairwise Voting Agreement Over Time",
    yaxis=dict(title="% votes in agreement", gridcolor=GRID, range=[10, 95]),
    height=480,
))

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 6: Speech length for contested vs uncontested votes over time
# ═══════════════════════════════════════════════════════════════════════════
vt_c = vt.copy()
bp = vt_c.groupby("besluit_id")["is_voor"].mean().rename("bp")
vt_c = vt_c.merge(bp.reset_index(), on="besluit_id", how="left")
vt_c["contested"] = (vt_c["bp"] > 0.3) & (vt_c["bp"] < 0.7)

slen = vt_c.groupby(["quarter", "contested"])["speech_length"].median().unstack()
slen.index = slen.index.to_timestamp()
slen.columns = ["Uncontested", "Contested"]

fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=slen.index, y=slen["Contested"], name="Contested votes (30-70%)",
    line=dict(color="#fbbf24", width=2.5), fill="tozeroy",
    fillcolor="rgba(251,191,36,0.08)",
    hovertemplate="%{y:,.0f} chars<extra></extra>",
))
fig6.add_trace(go.Scatter(
    x=slen.index, y=slen["Uncontested"], name="Uncontested votes",
    line=dict(color="#60a5fa", width=2.5), fill="tozeroy",
    fillcolor="rgba(96,165,250,0.08)",
    hovertemplate="%{y:,.0f} chars<extra></extra>",
))
add_coalition_bands(fig6, ymax=slen.max().max() * 0.95)
fig6.update_layout(**base_layout(
    title="Do MPs Speak More Before Contested Votes? (median speech length)",
    yaxis=dict(title="Median speech length (chars)", gridcolor=GRID),
    height=430,
))

# ═══════════════════════════════════════════════════════════════════════════
# Assemble HTML
# ═══════════════════════════════════════════════════════════════════════════
plots = [
    ("coal-opp",    "Coalition vs Opposition",  fig1,
     "Coalition parties consistently vote 'Voor' more than opposition. Watch the gap shift as coalitions change."),
    ("party-lines", "Party Trajectories",       fig2,
     "Each party's 'Voor' rate over time. Parties entering government spike upward; leaving drops them."),
    ("contested",   "Vote Contentiousness",     fig3,
     "What fraction of votes are close calls vs unanimous? Contested votes are where speech text matters most."),
    ("agreement",   "Cross-Party Agreement",    fig4,
     "Which parties vote together? Coalition partners cluster; left-bloc (SP, PvdD, GroenLinks) forms a tight block."),
    ("alliances",   "Shifting Alliances",       fig5,
     "Track how specific party pairs move together or apart — especially around coalition changes."),
    ("speech-len",  "Speech Length & Contention", fig6,
     "Do MPs speak more before contested votes? Longer speeches signal harder-to-predict outcomes."),
]

# stats
n_pairs = f"{len(pairs):,}"
n_besluit = f"{pairs['besluit_id'].nunique():,}"
n_speakers = f"{pairs['persoon_id'].nunique():,}"
date_range = f"{pairs['datum'].min().strftime('%b %Y')} – {pairs['datum'].max().strftime('%b %Y')}"

fig_jsons = {pid: json.loads(fig.to_json()) for pid, _, fig, _ in plots}

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tweede Kamer — Voting Dynamics</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: {BG};
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    min-height: 100vh;
  }}
  header {{
    width: 100%;
    padding: 16px 28px;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
    background: {PAPER};
    border-bottom: 1px solid rgba(255,255,255,0.08);
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  header h1 {{
    font-size: 17px;
    font-weight: 700;
    color: #fff;
    white-space: nowrap;
  }}
  header h1 span {{ color: {ACCENT}; }}
  .nav-bar {{
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    flex: 1;
    justify-content: flex-end;
  }}
  .nav-btn {{
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(255,255,255,0.04);
    color: #a0a0c0;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .nav-btn:hover {{ background: rgba(255,255,255,0.08); color: #fff; }}
  .nav-btn.active {{
    background: rgba(249,199,79,0.12);
    color: {ACCENT};
    border-color: rgba(249,199,79,0.3);
  }}
  main {{
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    padding: 24px 20px 60px;
  }}
  .info-bar {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    padding: 14px 0 10px;
    font-size: 12px;
    color: #808098;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 18px;
  }}
  .info-bar strong {{ color: #c0c0d8; }}
  .plot-section {{
    display: none;
    margin-bottom: 20px;
  }}
  .plot-section.visible {{ display: block; }}
  .show-all .plot-section {{ display: block; }}
  .plot-caption {{
    font-size: 13px;
    color: #909098;
    padding: 6px 8px 14px;
    line-height: 1.5;
  }}
  .plot-container {{
    width: 100%;
    border-radius: 10px;
    overflow: hidden;
    background: {PAPER};
    border: 1px solid rgba(255,255,255,0.06);
  }}
  .toggle-all {{
    padding: 5px 12px;
    font-size: 11px;
    background: rgba(255,255,255,0.04);
    color: #808098;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 5px;
    cursor: pointer;
    margin-left: auto;
  }}
  .toggle-all:hover {{ color: #fff; }}
</style>
</head>
<body>

<header>
  <h1>Tweede Kamer <span>Voting Dynamics</span></h1>
  <nav class="nav-bar">
    {"".join(f'<button class="nav-btn{" active" if i==0 else ""}" data-target="{pid}">{label}</button>' for i,(pid,label,_,__) in enumerate(plots))}
  </nav>
</header>

<main id="main">
  <div class="info-bar">
    <span><strong>{n_pairs}</strong> speech-vote pairs</span>
    <span><strong>{n_besluit}</strong> voting moments</span>
    <span><strong>{n_speakers}</strong> speakers</span>
    <span><strong>{date_range}</strong></span>
    <button class="toggle-all" id="toggle-all">Show all</button>
  </div>
  {"".join(f'''
  <div class="plot-section{" visible" if i==0 else ""}" id="section-{pid}">
    <div class="plot-container" id="plot-{pid}"></div>
    <div class="plot-caption">{caption}</div>
  </div>''' for i,(pid,label,_,caption) in enumerate(plots))}
</main>

<script>
const FIGS = {json.dumps(fig_jsons)};
const rendered = new Set();

function renderPlot(id) {{
  if (rendered.has(id)) return;
  const spec = FIGS[id];
  Plotly.newPlot("plot-" + id, spec.data, spec.layout, {{responsive: true}});
  rendered.add(id);
}}

document.querySelectorAll(".nav-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    const target = btn.dataset.target;
    const main = document.getElementById("main");
    if (main.classList.contains("show-all")) {{
      main.classList.remove("show-all");
      document.getElementById("toggle-all").textContent = "Show all";
    }}
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".plot-section").forEach(s => s.classList.remove("visible"));
    document.getElementById("section-" + target).classList.add("visible");
    renderPlot(target);
    setTimeout(() => Plotly.Plots.resize("plot-" + target), 50);
  }});
}});

document.getElementById("toggle-all").addEventListener("click", function() {{
  const main = document.getElementById("main");
  const showAll = !main.classList.contains("show-all");
  main.classList.toggle("show-all", showAll);
  this.textContent = showAll ? "Show one" : "Show all";
  if (showAll) {{
    Object.keys(FIGS).forEach(id => {{
      renderPlot(id);
      setTimeout(() => Plotly.Plots.resize("plot-" + id), 100);
    }});
  }}
}});

renderPlot("{plots[0][0]}");
</script>

</body>
</html>"""

out_path = DEPLOY_DIR / "timeseries.html"
out_path.write_text(html, encoding="utf-8")
print(f"Written {out_path}  ({len(html):,} bytes)")
