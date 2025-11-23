"""Interactive search and generation interface."""

import pandas as pd
import streamlit as st
from loguru import logger

from counter_speech_generation.config import get_config
from counter_speech_generation.data.loader import load_dataset
from counter_speech_generation.models.encoder import load_generator
from counter_speech_generation.utils import setup_logging

setup_logging()


@st.cache_resource
def load_model():
    """Load model with caching."""
    return load_generator()


@st.cache_data
def load_data(path: str):
    """Load dataset with caching."""
    return load_dataset(path)


def main() -> None:
    """Main search interface."""
    st.set_page_config(
        page_title="Counter Speech Search",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Counter Speech Search & Generation")

    # Load model
    try:
        with st.spinner("Loading model..."):
            generator = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Dataset selection
    st.sidebar.header("📊 Dataset")
    dataset_path = st.sidebar.text_input(
        "Dataset Path",
        value="dataset/indonesian_hate_speech_with_counter.csv",
    )

    if st.sidebar.button("Load Dataset"):
        try:
            with st.spinner("Loading dataset..."):
                df = load_data(dataset_path)
                st.session_state.dataset = df
                st.session_state.dataset_loaded = True
            st.sidebar.success(f"Loaded {len(df)} samples")
        except Exception as e:
            st.sidebar.error(f"Failed to load dataset: {e}")

    # Search interface
    if "dataset_loaded" in st.session_state and st.session_state.dataset_loaded:
        df = st.session_state.dataset

        st.header("🔎 Search Dataset")
        search_query = st.text_input("Search toxic texts:")

        if search_query:
            # Simple text search
            mask = df["text"].str.contains(search_query, case=False, na=False)
            results = df[mask].head(10)

            if len(results) > 0:
                st.write(f"Found {len(results)} results:")
                
                for idx, row in results.iterrows():
                    with st.expander(f"Text: {row['text'][:100]}..."):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Text:**")
                            st.write(row["text"])
                            
                            st.write("**Reference Counter Speech:**")
                            st.info(row["counter"])
                        
                        with col2:
                            st.write("**Generate New Counter Speech:**")
                            if st.button("Generate", key=f"gen_{idx}"):
                                with st.spinner("Generating..."):
                                    generated = generator.generate(row["text"])
                                    st.success(generated)
            else:
                st.warning("No results found")

        # Random sample
        st.header("🎲 Random Sample")
        if st.button("Show Random Sample"):
            sample = df.sample(1).iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Text:**")
                st.write(sample["text"])
                
                st.write("**Reference Counter Speech:**")
                st.info(sample["counter"])
            
            with col2:
                st.write("**Generate New Counter Speech:**")
                if st.button("Generate", key="gen_random"):
                    with st.spinner("Generating..."):
                        generated = generator.generate(sample["text"])
                        st.success(generated)


if __name__ == "__main__":
    main()

