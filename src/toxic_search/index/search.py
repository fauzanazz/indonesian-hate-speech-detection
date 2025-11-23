"""Pure functions for querying Qdrant vector database."""

from dataclasses import dataclass

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
    query_vector = encoder.encode([query], show_progress=False)[0].cpu().numpy().tolist()
    
    # Search
    results = client.search(
        collection_name=config.qdrant.collection_name,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    )
    
    # Convert to SearchResult objects
    search_results = [
        SearchResult(
            text=hit.payload.get("text", ""),
            score=hit.score,
            metadata={k: v for k, v in hit.payload.items() if k != "text"},
        )
        for hit in results
    ]
    
    logger.info(f"Found {len(search_results)} results for query: {query[:50]}...")
    
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
        results = client.search(
            collection_name=config.qdrant.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        
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