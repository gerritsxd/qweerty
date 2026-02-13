#!/usr/bin/env python3
"""
Streamlit dashboard for Speech-to-Vote project.

Pages:
- Overview: dataset stats, speech-vote pair counts
- Speech explorer: browse speeches with vote outcomes
- Model results: (placeholder)
- Prediction demo: (placeholder)
"""

import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_stats():
    """Load dataset statistics."""
    analysis = ROOT / "data" / "analysis"
    if not (analysis / "speech_vote_pairs.parquet").exists():
        return None
    import pandas as pd
    pairs = pd.read_parquet(analysis / "speech_vote_pairs.parquet")
    return {
        "total_pairs": len(pairs),
        "unique_speakers": pairs["persoon_id"].nunique(),
        "unique_parties": pairs["fractie"].nunique(),
        "vote_dist": pairs["vote"].value_counts().to_dict(),
    }


def main():
    st.set_page_config(page_title="Speech-to-Vote", page_icon="🗳️", layout="wide")
    st.title("🗳️ Speech-to-Vote Prediction")
    st.caption("Tweede Kamer Open Data · Can we predict votes from debate speeches?")

    page = st.sidebar.radio(
        "Page",
        ["Overview", "Speech Explorer", "Model Results", "Prediction Demo"],
    )

    if page == "Overview":
        st.header("Dataset Overview")
        stats = load_stats()
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Speech-Vote Pairs", f"{stats['total_pairs']:,}")
            with col2:
                st.metric("Unique Speakers", f"{stats['unique_speakers']:,}")
            with col3:
                st.metric("Parties", stats["unique_parties"])
            st.subheader("Vote Distribution")
            st.bar_chart(stats["vote_dist"])
        else:
            st.warning("Run the pipeline first: `python -m src.build_speech_dataset`")

    elif page == "Speech Explorer":
        st.header("Speech Explorer")
        st.info("Browse speeches with their vote outcomes. Filter by party, topic.")
        analysis = ROOT / "data" / "analysis"
        if (analysis / "speech_vote_pairs.parquet").exists():
            import pandas as pd
            pairs = pd.read_parquet(analysis / "speech_vote_pairs.parquet")
            if len(pairs) > 100_000:
                pairs = pairs.sample(5000, random_state=42)
            party = st.selectbox("Filter by party", ["All"] + sorted(pairs["fractie"].dropna().unique().tolist()))
            if party != "All":
                pairs = pairs[pairs["fractie"] == party]
            n = st.slider("Show", 1, min(50, len(pairs)), 5)
            for _, row in pairs.head(n).iterrows():
                with st.expander(f"[{row.get('fractie','?')}] {row.get('achternaam','?')} ({row.get('vote','?')})"):
                    st.write(row.get("speech_text", "")[:500] + "..." if len(str(row.get("speech_text",""))) > 500 else row.get("speech_text", ""))
        else:
            st.warning("No speech_vote_pairs.parquet found.")

    elif page == "Model Results":
        st.header("Model Results")
        analysis = ROOT / "data" / "analysis"
        if (analysis / "speech_vote_pairs.parquet").exists():
            import pandas as pd
            from src.ml.features import load_pairs, get_train_val_test
            from src.ml.models import train_baseline_party, predict_baseline_party, train_model_a, predict_model_a, evaluate
            with st.spinner("Loading and evaluating models..."):
                df = load_pairs(sample=50000)
                df = df[df["datum"].notna()]
                train, val, test = get_train_val_test(df)
                train = train[train["vote"].isin(["Voor", "Tegen"])]
                val = val[val["vote"].isin(["Voor", "Tegen"])]
                if len(train) > 1000 and len(val) > 100:
                    model_b = train_baseline_party(train)
                    pred_b = predict_baseline_party(model_b, val)
                    r_b = evaluate(val["vote"].values, pred_b)
                    model_a = train_model_a(train, max_features=2000)
                    pred_a = predict_model_a(model_a, val)
                    r_a = evaluate(val["vote"].values, pred_a)
                    st.metric("Baseline (party only)", f"{r_b['accuracy']*100:.1f}% accuracy")
                    st.metric("Model A (party + TF-IDF)", f"{r_a['accuracy']*100:.1f}% accuracy")
                    st.dataframe(pd.DataFrame([
                        {"Model": "Baseline", "Accuracy": r_b["accuracy"], "F1 (macro)": r_b["f1_macro"]},
                        {"Model": "Model A (TF-IDF)", "Accuracy": r_a["accuracy"], "F1 (macro)": r_a["f1_macro"]},
                    ]))
                else:
                    st.warning("Not enough data with dates for evaluation.")
        else:
            st.warning("No speech_vote_pairs.parquet found.")

    elif page == "Prediction Demo":
        st.header("Prediction Demo")
        st.info("Paste a speech excerpt and select party to get a vote prediction.")
        analysis = ROOT / "data" / "analysis"
        if (analysis / "speech_vote_pairs.parquet").exists():
            import pandas as pd
            from src.ml.features import load_pairs, get_train_val_test
            from src.ml.models import train_model_a, predict_model_a
            text = st.text_area("Speech text (Dutch)", height=150, placeholder="Voorzitter, dit wetsvoorstel...")
            party = st.selectbox("Party (fractie)", ["VVD", "PVV", "CDA", "D66", "GroenLinks-PvdA", "SP", "FvD", "ChristenUnie", "PvdD", "SGP", "DENK", "JA21", "BBB", "Other"])
            if st.button("Predict") and text.strip():
                with st.spinner("Training model and predicting..."):
                    df = load_pairs(sample=20000)
                    df = df[df["datum"].notna()]
                    df = df[df["vote"].isin(["Voor", "Tegen"])]
                    train, _, _ = get_train_val_test(df)
                    if len(train) > 1000:
                        model = train_model_a(train, max_features=2000)
                        demo_df = pd.DataFrame([{"speech_text": text, "fractie": party}])
                        pred = predict_model_a(model, demo_df)
                        st.success(f"Predicted vote: **{pred[0]}**")
                    else:
                        st.error("Not enough training data.")
        else:
            st.warning("No speech_vote_pairs.parquet found.")


if __name__ == "__main__":
    main()
