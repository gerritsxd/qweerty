#!/usr/bin/env python3
"""
Streamlit dashboard for Speech-to-Vote project.

Pages:
- Overview: dataset stats, vote distribution, temporal coverage, party breakdown
- Speech Explorer: browse speeches with vote outcomes
- Model Results: baseline vs Model A comparison, confusion matrix
- Prediction Demo: live prediction with examples
- Methodology: pipeline and model descriptions
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@st.cache_data(ttl=3600)
def load_overview_data():
    """Load dataset for overview page (cached)."""
    analysis = ROOT / "data" / "analysis"
    if not (analysis / "speech_vote_pairs.parquet").exists():
        return None
    import pandas as pd
    pairs = pd.read_parquet(analysis / "speech_vote_pairs.parquet")
    sample = pairs.sample(200_000, random_state=42) if len(pairs) > 200_000 else pairs
    return {"pairs": pairs, "sample": sample}


def render_overview():
    """Overview page with KPI cards, vote distribution, temporal chart, party breakdown."""
    import pandas as pd
    data = load_overview_data()
    if data is None:
        st.warning("Run the pipeline first: `python -m src.build_speech_dataset`")
        return

    pairs = data["pairs"]
    sample = data["sample"]

    # KPI cards
    st.subheader("Dataset at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Speech-Vote Pairs", f"{len(pairs):,}")
    with col2:
        st.metric("Unique Speakers", f"{pairs['persoon_id'].nunique():,}")
    with col3:
        st.metric("Parties", pairs["fractie"].nunique())
    with col4:
        dates = pd.to_datetime(pairs["datum"], errors="coerce")
        valid_dates = dates.dropna()
        if len(valid_dates) > 0:
            date_range = f"{valid_dates.min().year}-{valid_dates.max().year}"
        else:
            date_range = "N/A"
        st.metric("Date Range", date_range)

    st.caption("Data source: Tweede Kamer Open Data API")

    # Vote distribution - pie chart
    st.subheader("Vote Distribution")
    vote_counts = pairs["vote"].value_counts()
    if len(vote_counts) > 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"Voor": "#2ecc71", "Tegen": "#e74c3c", "Niet deelgenomen": "#95a5a6"}
        plot_colors = [colors.get(v, "#3498db") for v in vote_counts.index]
        ax.pie(vote_counts.values, labels=vote_counts.index, autopct="%1.1f%%",
               colors=plot_colors, startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
        plt.close()

    # Temporal coverage
    st.subheader("Temporal Coverage")
    sample_copy = sample.copy()
    sample_copy["year"] = pd.to_datetime(sample_copy["datum"], errors="coerce").dt.year
    yearly = sample_copy["year"].dropna().astype(int).value_counts().sort_index()
    if len(yearly) > 0:
        st.bar_chart(yearly)

    # Party breakdown (pairs per party, colored by vote)
    st.subheader("Pairs per Party (top 15)")
    party_vote = sample.groupby(["fractie", "vote"]).size().unstack(fill_value=0)
    top_parties = party_vote.sum(axis=1).nlargest(15)
    party_vote_top = party_vote.loc[top_parties.index]
    st.bar_chart(party_vote_top)


def render_speech_explorer():
    """Speech Explorer with filters and vote-colored labels."""
    data = load_overview_data()
    if data is None:
        st.warning("No speech_vote_pairs.parquet found.")
        return

    import pandas as pd
    pairs = data["sample"]

    st.subheader("Filter")
    col1, col2, col3 = st.columns(3)
    with col1:
        parties = ["All"] + sorted(pairs["fractie"].dropna().unique().tolist())
        party = st.selectbox("Party", parties)
    with col2:
        votes = ["All", "Voor", "Tegen", "Niet deelgenomen"]
        vote_filter = st.selectbox("Vote outcome", votes)
    with col3:
        pairs_copy = pairs.copy()
        pairs_copy["_year"] = pd.to_datetime(pairs_copy["datum"], errors="coerce").dt.year
        years = sorted(pairs_copy["_year"].dropna().astype(int).unique().tolist())[:25]
        year_filter = st.selectbox("Year", ["All"] + years)

    filtered = pairs.copy()
    filtered["_year"] = pd.to_datetime(filtered["datum"], errors="coerce").dt.year
    if party != "All":
        filtered = filtered[filtered["fractie"] == party]
    if vote_filter != "All":
        filtered = filtered[filtered["vote"] == vote_filter]
    if year_filter != "All":
        filtered = filtered[filtered["_year"] == year_filter]

    keyword = st.text_input("Search in speech text (optional)", "")
    if keyword:
        filtered = filtered[filtered["speech_text"].fillna("").str.contains(keyword, case=False, na=False)]

    n = st.slider("Show", 1, min(50, len(filtered)), 10)

    st.subheader("Speeches")
    vote_colors = {"Voor": "green", "Tegen": "red", "Niet deelgenomen": "gray"}
    for _, row in filtered.head(n).iterrows():
        vote = row.get("vote", "?")
        color = vote_colors.get(vote, "gray")
        label = f"[{row.get('fractie', '?')}] {row.get('achternaam', '?')} ({vote})"
        text = str(row.get("speech_text", ""))
        word_count = len(text.split()) if text else 0
        topic = row.get("activiteit_onderwerp", "") or row.get("activiteithoofd_onderwerp", "")
        with st.expander(label):
            st.markdown(f"**Vote:** :{color}[{vote}]")
            if topic:
                st.caption(f"Topic: {str(topic)[:100]}")
            st.caption(f"Words: {word_count}")
            st.write(text[:600] + ("..." if len(text) > 600 else ""))


def render_model_results():
    """Model Results with accuracy chart, confusion matrix, per-party breakdown."""
    data = load_overview_data()
    if data is None:
        st.warning("No speech_vote_pairs.parquet found.")
        return

    import pandas as pd
    from src.ml.features import load_pairs, get_train_val_test, build_basic_features, add_enhanced_features
    from src.ml.models import (
        train_baseline_party, predict_baseline_party,
        train_model_a, predict_model_a,
        train_model_gb, train_model_rf, train_model_xgb,
        train_ensemble_two_stage, predict_ensemble_two_stage,
        load_model_robbert, predict_model_robbert,
        evaluate,
    )
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    MODEL_KW = dict(
        max_features=2000, ngram_range=(1, 1), min_df=1,
        use_besluit_tfidf=True, use_speech_position=True, use_speaker_loyalty=True,
        use_kabinetsappreciatie=True, use_zaak_soort=True, use_is_coalition=True,
    )

    @st.cache_resource
    def train_and_evaluate():
        df = load_pairs(sample=50000)
        df = df[df["datum"].notna()]
        df = build_basic_features(df)
        train, val, test = get_train_val_test(df)
        train = train[train["vote"].isin(["Voor", "Tegen"])]
        val = val[val["vote"].isin(["Voor", "Tegen"])]
        train, val, test = add_enhanced_features(train, val, test)
        if len(train) < 100 or len(val) < 20:
            return None
        model_b = train_baseline_party(train)
        pred_b = predict_baseline_party(model_b, val)
        r_b = evaluate(val["vote"].values, pred_b)
        model_a = train_model_a(train, **MODEL_KW)
        pred_a = predict_model_a(model_a, val)
        r_a = evaluate(val["vote"].values, pred_a)
        model_gb = train_model_gb(train, **MODEL_KW)
        pred_gb = predict_model_a(model_gb, val)
        r_gb = evaluate(val["vote"].values, pred_gb)
        model_rf = train_model_rf(train, **MODEL_KW)
        pred_rf = predict_model_a(model_rf, val)
        r_rf = evaluate(val["vote"].values, pred_rf)
        try:
            model_xgb = train_model_xgb(train, **MODEL_KW)
            pred_xgb = predict_model_a(model_xgb, val)
            r_xgb = evaluate(val["vote"].values, pred_xgb)
        except Exception:
            model_xgb = pred_xgb = r_xgb = None
        try:
            model_ens = train_ensemble_two_stage(train, confidence_threshold=0.7, **MODEL_KW)
            pred_ens = predict_ensemble_two_stage(model_ens, val)
            r_ens = evaluate(val["vote"].values, pred_ens)
        except Exception:
            model_ens = pred_ens = r_ens = None
        model_robbert = pred_robbert = r_robbert = None
        try:
            robbert_path = ROOT / "models" / "robbert_vote_classifier"
            if (robbert_path / "config.json").exists():
                model_robbert = load_model_robbert(str(robbert_path))
                pred_robbert = predict_model_robbert(model_robbert, val)
                r_robbert = evaluate(val["vote"].values, pred_robbert)
        except Exception:
            pass
        return {
            "model_b": model_b, "pred_b": pred_b, "r_b": r_b,
            "model_a": model_a, "pred_a": pred_a, "r_a": r_a,
            "model_gb": model_gb, "pred_gb": pred_gb, "r_gb": r_gb,
            "model_rf": model_rf, "pred_rf": pred_rf, "r_rf": r_rf,
            "model_xgb": model_xgb, "pred_xgb": pred_xgb, "r_xgb": r_xgb,
            "model_ens": model_ens, "pred_ens": pred_ens, "r_ens": r_ens,
            "model_robbert": model_robbert, "pred_robbert": pred_robbert, "r_robbert": r_robbert,
            "val": val, "train": train,
        }
    result = train_and_evaluate()
    if result is None:
        st.warning("Not enough data with dates for evaluation.")
        return

    r_b, r_a = result["r_b"], result["r_a"]
    r_gb, r_rf = result["r_gb"], result["r_rf"]
    r_xgb, r_ens = result.get("r_xgb"), result.get("r_ens")
    r_robbert = result.get("r_robbert")
    val = result["val"]

    models_list = [("Baseline", r_b), ("Model A", r_a), ("GradientBoosting", r_gb), ("RandomForest", r_rf)]
    if r_xgb:
        models_list.append(("XGBoost", r_xgb))
    if r_ens:
        models_list.append(("Ensemble", r_ens))
    if r_robbert:
        models_list.append(("RobBERT", r_robbert))
    best = max(models_list, key=lambda x: x[1]["accuracy"])
    st.info(
        f"**Party identity alone** predicts **{r_b['accuracy']*100:.1f}%**. "
        f"Best model: **{best[0]}** at **{best[1]['accuracy']*100:.1f}%** accuracy."
    )

    # Accuracy bar chart
    st.subheader("Model Comparison")
    comp_data = {"Model": [m[0] for m in models_list], "Accuracy": [m[1]["accuracy"] for m in models_list], "F1 (macro)": [m[1]["f1_macro"] for m in models_list]}
    comp_df = pd.DataFrame(comp_data)
    n_cols = len(models_list)
    cols = st.columns(min(n_cols, 6))
    for i, (name, r) in enumerate(models_list[:6]):
        with cols[i]:
            st.metric(name, f"{r['accuracy']*100:.1f}%", f"F1: {r['f1_macro']:.3f}")
    st.bar_chart(comp_df.set_index("Model")[["Accuracy", "F1 (macro)"]])

    # Confusion matrices
    st.subheader("Confusion Matrices")
    classes = ["Voor", "Tegen"]
    preds_list = [("Baseline", result["pred_b"]), ("Model A", result["pred_a"]), ("GradientBoosting", result["pred_gb"]), ("RandomForest", result["pred_rf"])]
    if result.get("pred_xgb") is not None:
        preds_list.append(("XGBoost", result["pred_xgb"]))
    if result.get("pred_ens") is not None:
        preds_list.append(("Ensemble", result["pred_ens"]))
    if result.get("pred_robbert") is not None:
        preds_list.append(("RobBERT", result["pred_robbert"]))
    pred_cols = st.columns(min(len(preds_list), 6))
    for i, (label, pred) in enumerate(preds_list[:6]):
        with pred_cols[i]:
            cm = confusion_matrix(val["vote"], pred, labels=classes)
            fig, ax = plt.subplots(figsize=(3, 2.5))
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, ax=ax, cmap="Blues")
            ax.set_title(label)
            ax.set_ylabel("True")
            ax.set_xlabel("Predicted")
            st.pyplot(fig)
            plt.close()

    # Per-party accuracy
    st.subheader("Per-Party Accuracy (Baseline)")
    val_copy = val.copy()
    val_copy["pred"] = result["pred_b"]
    val_copy["correct"] = val_copy["vote"] == val_copy["pred"]
    party_acc = val_copy.groupby("fractie").agg(
        total=("correct", "count"),
        correct=("correct", "sum"),
    )
    party_acc["accuracy"] = party_acc["correct"] / party_acc["total"]
    party_acc = party_acc[party_acc["total"] >= 20].sort_values("accuracy", ascending=False)
    st.dataframe(party_acc[["total", "correct", "accuracy"]].head(15), use_container_width=True)


def render_prediction_demo():
    """Prediction Demo with cached model, examples, confidence."""
    data = load_overview_data()
    if data is None:
        st.warning("No speech_vote_pairs.parquet found.")
        return

    import pandas as pd
    from src.ml.features import load_pairs, get_train_val_test
    from src.ml.models import train_model_a, predict_model_a, load_model_robbert, predict_model_robbert, predict_proba_model_robbert

    @st.cache_resource
    def get_model():
        robbert_path = ROOT / "models" / "robbert_vote_classifier"
        if (robbert_path / "config.json").exists():
            try:
                return (load_model_robbert(str(robbert_path)), "robbert")
            except Exception:
                pass
        from src.ml.features import build_basic_features, add_enhanced_features
        df = load_pairs(sample=20000)
        df = df[df["datum"].notna()]
        df = df[df["vote"].isin(["Voor", "Tegen"])]
        df = build_basic_features(df)
        train, val, test = get_train_val_test(df)
        train, val, test = add_enhanced_features(train, val, test)
        if len(train) < 100:
            return None
        return (train_model_a(
            train, max_features=2000, ngram_range=(1, 1), min_df=1,
            use_besluit_tfidf=True, use_speech_position=True, use_speaker_loyalty=True,
            use_kabinetsappreciatie=True, use_zaak_soort=True, use_is_coalition=True,
        ), "model_a")

    model_tuple = get_model()
    if model_tuple is None:
        model, model_type = None, None
    else:
        model, model_type = model_tuple
    if model is None:
        st.error("Not enough training data.")
        return

    st.caption(f"Using: **{model_type}**" + (" (run `python scripts/train_robbert.py` for RobBERT)" if model_type != "robbert" else ""))

    # Example snippets
    examples = [
        ("Supportive", "Voorzitter, wij steunen dit wetsvoorstel van harte. Het is een belangrijke stap voorwaarts."),
        ("Opposing", "Dit voorstel is onaanvaardbaar. Wij stemmen tegen."),
    ]
    st.subheader("Try it")
    for label, text in examples:
        if st.button(f"Use example: {label}", key=label):
            st.session_state["demo_text"] = text
    if "demo_text" in st.session_state:
        default_text = st.session_state["demo_text"]
    else:
        default_text = ""

    text = st.text_area("Speech text (Dutch)", height=150, placeholder="Voorzitter, dit wetsvoorstel...", value=default_text)
    party = st.selectbox("Party (fractie)", ["VVD", "PVV", "CDA", "D66", "GroenLinks-PvdA", "SP", "FvD", "ChristenUnie", "PvdD", "SGP", "DENK", "JA21", "BBB", "Other"])

    if st.button("Predict") and text.strip():
        demo_df = pd.DataFrame([{
            "speech_text": text, "fractie": party,
            "besluit_tekst": "", "speech_position": 0.5, "speaker_loyalty": 0.5,
            "kabinetsappreciatie": "Onbekend", "zaak_soort": "Onbekend", "is_coalition": 0,
            "agendapunt_onderwerp": "",
        }])
        if model_type == "robbert":
            pred = predict_model_robbert(model, demo_df)
            conf = predict_proba_model_robbert(model, demo_df)
        else:
            from src.ml.models import predict_proba_model_a
            pred = predict_model_a(model, demo_df)
            conf = predict_proba_model_a(model, demo_df)
        st.success(f"Predicted vote: **{pred[0]}**")
        st.caption(f"Confidence: Voor {conf.get('Voor', 0):.0%}, Tegen {conf.get('Tegen', 0):.0%}")

        # Top TF-IDF terms (Model A only)
        try:
            if model_type != "robbert" and "clf" in model:
                coef = model["clf"].coef_[0]
                tfidf_names = model["tfidf"].get_feature_names_out()
                n_party = len(model["party_enc"].get_feature_names_out())
                tfidf_coef = coef[n_party:]
                if len(tfidf_coef) == len(tfidf_names):
                    top_idx = tfidf_coef.argsort()[-5:][::-1]
                    top_terms = [tfidf_names[i] for i in top_idx]
                    st.caption(f"Top terms: {', '.join(top_terms)}")
        except Exception:
            pass


def render_methodology():
    """Methodology page with pipeline diagram and model descriptions."""
    st.subheader("Pipeline")
    st.markdown("""
    ```
    Raw Data (JSON)  -->  Process (parquet)  -->  Verslag XMLs
                                                      |
                                                      v
    Stemming, Besluit, Activiteit  <--  Parse speeches  <--  Download XMLs
              |                              |
              v                              v
         Link speeches to votes  -->  speech_vote_pairs.parquet
                      |
                      v
         Features (party, TF-IDF)  -->  Model  -->  Prediction (Voor/Tegen)
    ```
    """)
    st.subheader("Temporal Split")
    st.markdown("""
    - **Train**: data up to 2021
    - **Validation**: 2022
    - **Test**: 2023 and later

    This prevents leakage and simulates real-world deployment.
    """)
    st.subheader("Models")
    st.markdown("""
    - **Baseline (party only)**: Predicts the majority vote per party. No speech text used.
    - **Model A (LogReg)**: Party + TF-IDF of speech + Kabinetsappreciatie + Zaak.Soort + speech position + speaker loyalty + coalition flag.
    - **GradientBoosting / RandomForest / XGBoost**: Same features, different classifiers.
    - **Ensemble (two-stage)**: Party baseline + speech-based override only when confidence ≥ threshold (speech can help, never hurt).
    - **RobBERT**: Fine-tuned Dutch transformer (DTAI-KULeuven/robbert-2023-dutch-base). Train with `python scripts/train_robbert.py`.
    """)


def main():
    st.set_page_config(page_title="Speech-to-Vote", page_icon="🗳️", layout="wide")
    st.title("🗳️ Speech-to-Vote Prediction")
    st.caption("Tweede Kamer Open Data · Can we predict votes from debate speeches?")

    page = st.sidebar.radio(
        "Page",
        ["Overview", "Speech Explorer", "Model Results", "Prediction Demo", "Methodology"],
    )

    if page == "Overview":
        render_overview()
    elif page == "Speech Explorer":
        render_speech_explorer()
    elif page == "Model Results":
        render_model_results()
    elif page == "Prediction Demo":
        render_prediction_demo()
    elif page == "Methodology":
        render_methodology()


if __name__ == "__main__":
    main()
