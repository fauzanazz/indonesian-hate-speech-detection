"""Pure functions for querying Qdrant vector database."""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from toxic_search.config import get_config
from toxic_search.models.encoder import ToxicEncoder


@dataclass
class SearchResult:
    """Container for search result."""

    text: str
    score: float
    metadata: dict


def search_similar(
    query: str,
    encoder: ToxicEncoder,
    client: QdrantClient | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[SearchResult]:
    """Search for semantically similar texts in Qdrant.
    
    Pure function for vector similarity search.
    """
    config = get_config()
    
    client = client or QdrantClient(host=config.qdrant.host, port=config.qdrant.port)
    top_k = top_k or config.api.top_k_results
    score_threshold = score_threshold or config.api.score_threshold
    
    # Encode query
    query_embedding = encoder.encode([query], show_progress=False)[0].cpu().numpy()
    query_vector = query_embedding.tolist()
    
    # DEBUG: Log query vector details
    logger.debug(f"Query: '{query[:100]}'")
    logger.debug(f"Query vector shape: {query_embedding.shape}")
    logger.debug(f"Query vector norm: {np.linalg.norm(query_embedding):.4f}")
    logger.debug(f"Query vector first 5 dims: {query_embedding[:5].tolist()}")
    
    # DEBUG: Get collection info
    try:
        collection_info = client.get_collection(config.qdrant.collection_name)
        total_vectors = collection_info.points_count
        logger.debug(f"Collection '{config.qdrant.collection_name}' has {total_vectors} vectors")
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        total_vectors = "unknown"
    
    # DEBUG: First query WITHOUT score_threshold to see raw results
    logger.debug(f"Querying with top_k={top_k}, NO score_threshold (raw results)")
    raw_results = client.query_points(
        collection_name=config.qdrant.collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=None,  # Get all results
    ).points
    
    # DEBUG: Log raw result statistics
    if raw_results:
        raw_scores = [hit.score for hit in raw_results]
        logger.debug(f"Raw results count: {len(raw_results)}")
        logger.debug(f"Score range: min={min(raw_scores):.4f}, max={max(raw_scores):.4f}, mean={np.mean(raw_scores):.4f}")
        logger.debug(f"Top 5 scores: {[f'{s:.4f}' for s in raw_scores[:5]]}")
        logger.debug(f"Score threshold being applied: {score_threshold}")
    else:
        logger.warning(f"No raw results returned from Qdrant! Collection may be empty or query failed.")
    
    # Now apply score_threshold filtering
    results = client.query_points(
        collection_name=config.qdrant.collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    ).points
    
    # DEBUG: Log filtering impact
    if raw_results:
        filtered_count = len(raw_results) - len(results)
        logger.debug(f"Score threshold={score_threshold} filtered out {filtered_count}/{len(raw_results)} results")
    
    # Convert to SearchResult objects
    search_results = [
        SearchResult(
            text=hit.payload.get("text", ""),
            score=hit.score,
            metadata={k: v for k, v in hit.payload.items() if k != "text"},
        )
        for hit in results
    ]
    
    logger.info(f"Found {len(search_results)} results for query: {query[:50]}... (after score_threshold={score_threshold})")
    
    return search_results


def batch_search(
    queries: list[str],
    encoder: ToxicEncoder,
    client: QdrantClient | None = None,
    top_k: int | None = None,
) -> list[list[SearchResult]]:
    """Batch search for multiple queries.
    
    More efficient than calling search_similar in a loop.
    """
    config = get_config()
    
    client = client or QdrantClient(host=config.qdrant.host, port=config.qdrant.port)
    top_k = top_k or config.api.top_k_results
    
    # Encode all queries
    query_vectors = encoder.encode(queries, show_progress=True).cpu().numpy().tolist()
    
    # Batch search
    all_results = []
    for query, query_vector in zip(queries, query_vectors):
        results = client.query_points(
            collection_name=config.qdrant.collection_name,
            query=query_vector,
            limit=top_k,
        ).points
        
        search_results = [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
            )
            for hit in results
        ]
        
        all_results.append(search_results)
    
    logger.info(f"Batch search completed for {len(queries)} queries")
    
    return all_results