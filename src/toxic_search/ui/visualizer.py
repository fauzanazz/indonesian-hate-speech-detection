"""UMAP visualization for embedding space exploration."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from loguru import logger

from toxic_search.config import get_config
from toxic_search.models.encoder import ToxicEncoder


def render_embedding_visualization(
    texts: list[str],
    labels: list[int] | None,
    encoder: ToxicEncoder,
) -> None:
    """Render UMAP visualization of embeddings."""
    config = get_config().ui
    
    st.header("📊 Embedding Visualization")
    st.markdown("Explore the embedding space with UMAP dimensionality reduction")
    
    # Sample size control
    sample_size = min(len(texts), config.vis_sample_size)
    
    if len(texts) > config.vis_sample_size:
        st.info(f"Sampling {sample_size} points for visualization (max: {config.vis_sample_size})")
        indices = np.random.choice(len(texts), sample_size, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices] if labels else None
    
    with st.spinner("Generating embeddings..."):
        embeddings = encoder.encode(texts, show_progress=True)
        embeddings_np = embeddings.cpu().numpy()
    
    # UMAP reduction
    with st.spinner("Running UMAP reduction..."):
        try:
            from umap import UMAP
            
            reducer = UMAP(
                n_neighbors=config.umap_n_neighbors,
                min_dist=config.umap_min_dist,
                n_components=2,
                random_state=42,
            )
            
            embeddings_2d = reducer.fit_transform(embeddings_np)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "text": [text[:100] + "..." if len(text) > 100 else text for text in texts],
                "label": labels if labels else ["unknown"] * len(texts),
            })
            
            # Create interactive scatter plot
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="label",
                hover_data=["text"],
                title="UMAP Projection of Embeddings",
                labels={"x": "UMAP 1", "y": "UMAP 2"},
                height=600,
            )
            
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Points", len(texts))
            
            with col2:
                if labels:
                    st.metric("Unique Labels", len(set(labels)))
                else:
                    st.metric("Unique Labels", "N/A")
            
            with col3:
                st.metric("Embedding Dim", embeddings_np.shape[1])
            
        except ImportError:
            st.error("UMAP not installed. Install with: uv add umap-learn")
            logger.error("UMAP not available")
        except Exception as e:
            st.error(f"Visualization failed: {e}")
            logger.error(f"Visualization error: {e}")