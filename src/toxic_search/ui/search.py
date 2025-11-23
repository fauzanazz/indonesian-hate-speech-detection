"""Search interface component for Streamlit UI."""

import streamlit as st
from loguru import logger
from qdrant_client import QdrantClient

from toxic_search.config import get_config
from toxic_search.index.search import search_similar
from toxic_search.models.encoder import ToxicEncoder


def render_search_interface(
    encoder: ToxicEncoder,
    client: QdrantClient,
) -> None:
    """Render search interface with results."""
    config = get_config()
    
    st.header("🔍 Semantic Search")
    st.markdown("Search for semantically similar toxic content")
    
    # Search input
    query = st.text_area(
        "Enter your search query:",
        height=100,
        placeholder="Type your search query here...",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", 1, 50, config.api.top_k_results)
    with col2:
        score_threshold = st.slider(
            "Score threshold",
            0.0,
            1.0,
            config.api.score_threshold,
            0.05,
        )
    
    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query")
            return
        
        with st.spinner("Searching..."):
            try:
                results = search_similar(
                    query=query,
                    encoder=encoder,
                    client=client,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                
                st.success(f"Found {len(results)} results")
                
                # Display results
                for idx, result in enumerate(results, 1):
                    with st.expander(f"Result {idx} - Score: {result.score:.4f}"):
                        st.markdown(f"**Text:** {result.text}")
                        
                        if result.metadata:
                            st.markdown("**Metadata:**")
                            for key, value in result.metadata.items():
                                st.markdown(f"- {key}: {value}")
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                st.error(f"Search failed: {e}")