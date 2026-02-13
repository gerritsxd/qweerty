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
        st.info("Placeholder: model comparison and accuracy charts will go here.")

    elif page == "Prediction Demo":
        st.header("Prediction Demo")
        st.info("Placeholder: paste a speech, get a vote prediction.")


if __name__ == "__main__":
    main()
