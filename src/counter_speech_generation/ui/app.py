"""Streamlit app for interactive counter speech generation."""

import streamlit as st
from loguru import logger

from counter_speech_generation.config import get_config
from counter_speech_generation.models.encoder import load_generator
from counter_speech_generation.utils import setup_logging

setup_logging()


@st.cache_resource
def load_model():
    """Load model with caching."""
    return load_generator()


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Counter Speech Generation",
        page_icon="💬",
        layout="wide",
    )

    st.title("💬 Counter Speech Generation")
    st.markdown(
        "Generate counter speech for Indonesian toxic content using IndoT5"
    )

    # Load model
    try:
        with st.spinner("Loading model..."):
            generator = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        config = get_config().model
        
        max_length = st.slider(
            "Max Length",
            min_value=10,
            max_value=512,
            value=config.max_target_length,
            help="Maximum length of generated counter speech",
        )
        
        num_beams = st.slider(
            "Number of Beams",
            min_value=1,
            max_value=10,
            value=config.num_beams,
            help="Number of beams for beam search",
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=config.temperature,
            step=0.1,
            help="Sampling temperature (higher = more random)",
        )
        
        do_sample = st.checkbox(
            "Use Sampling",
            value=config.do_sample,
            help="Whether to use sampling instead of greedy/beam search",
        )

    # Main input area
    st.header("📝 Input")
    text_input = st.text_area(
        "Enter toxic text:",
        height=150,
        placeholder="Masukkan teks toxic di sini...",
    )

    if st.button("Generate Counter Speech", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Generating counter speech..."):
                try:
                    counter_speech = generator.generate(
                        text_input,
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        do_sample=do_sample,
                    )

                    st.header("✨ Generated Counter Speech")
                    st.info(counter_speech)

                    # Download button
                    st.download_button(
                        label="Download Result",
                        data=f"Original: {text_input}\n\nCounter Speech: {counter_speech}",
                        file_name="counter_speech.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    logger.exception("Generation error")

    # Example section
    with st.expander("📚 Example Inputs"):
        examples = [
            "Kamu bodoh sekali!",
            "Partai ini sampah!",
            "Kelompok ini tidak berguna!",
        ]
        
        for example in examples:
            if st.button(f"Use: {example}", key=f"example_{example}"):
                st.session_state.example_text = example
                st.rerun()
        
        if "example_text" in st.session_state:
            text_input = st.session_state.example_text
            del st.session_state.example_text


if __name__ == "__main__":
    main()

