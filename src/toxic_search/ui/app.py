"""Streamlit application entry point for toxic content exploration."""

from pathlib import Path

import streamlit as st
from loguru import logger
from qdrant_client import QdrantClient

from toxic_search.config import get_config
from toxic_search.models.encoder import load_encoder
from toxic_search.ui.search import render_search_interface
from toxic_search.ui.visualizer import render_embedding_visualization
from toxic_search.utils import setup_logging


@st.cache_resource
def get_encoder():
    """Cache encoder singleton."""
    return load_encoder()


@st.cache_resource
def get_client():
    """Cache Qdrant client singleton."""
    config = get_config().qdrant
    return QdrantClient(host=config.host, port=config.port)


def main() -> None:
    """Main Streamlit application."""
    setup_logging()
    
    st.set_page_config(
        page_title="Toxic Content Inspector",
        page_icon="🔍",
        layout="wide",
    )
    
    st.title("🔍 Toxic Content Inspector")
    st.markdown("Semantic retrieval and exploration for toxic content detection")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Initialize components
        try:
            encoder = get_encoder()
            client = get_client()
            st.success("✅ System ready")
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            logger.error(f"Initialization error: {e}")
            return
        
        st.divider()
        
        # Mode selection
        mode = st.radio(
            "Select mode:",
            ["Search", "Visualize"],
            index=0,
        )
        
        st.divider()
        
        st.markdown("### 📊 System Info")
        st.markdown(f"**Embedding dim:** {encoder.embedding_dim}")
        st.markdown(f"**Device:** {encoder.device}")
    
    # Main content
    if mode == "Search":
        render_search_interface(encoder, client)
    
    elif mode == "Visualize":
        st.header("📊 Embedding Visualization")
        
        # File upload for visualization
        uploaded_file = st.file_uploader(
            "Upload CSV file with 'text' and 'label' columns",
            type=["csv"],
        )
        
        if uploaded_file:
            import pandas as pd
            
            df = pd.read_csv(uploaded_file)
            
            if "text" not in df.columns:
                st.error("CSV must contain 'text' column")
                return
            
            texts = df["text"].tolist()
            labels = df["label"].tolist() if "label" in df.columns else None
            
            render_embedding_visualization(texts, labels, encoder)
        else:
            st.info("👆 Upload a CSV file to visualize embeddings")


if __name__ == "__main__":
    main()