"""Semantic search routes for unified API."""

import os

from fastapi import APIRouter, HTTPException
from loguru import logger
from qdrant_client import QdrantClient

from api.schemas import SearchRequest, SearchResponse, SearchResultItem
from toxic_search.index.search import search_similar
from toxic_search.models.encoder import ToxicEncoder, load_encoder

router = APIRouter(prefix="/search", tags=["Semantic Search"])

DEFAULT_QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
DEFAULT_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Global state for encoder and client
_encoder: ToxicEncoder | None = None
_client: QdrantClient | None = None


def get_encoder() -> ToxicEncoder:
    """Get or initialize encoder singleton."""
    global _encoder

    if _encoder is None:
        logger.info("Loading encoder...")
        _encoder = load_encoder()

    return _encoder


def get_client(host: str | None = None, port: int | None = None) -> QdrantClient:
    """Get or initialize Qdrant client singleton."""
    global _client

    resolved_host = host or DEFAULT_QDRANT_HOST
    resolved_port = port or DEFAULT_QDRANT_PORT

    if _client is None:
        logger.info(f"Connecting to Qdrant at {resolved_host}:{resolved_port}")
        _client = QdrantClient(host=resolved_host, port=resolved_port)

    return _client


def initialize_search_service(qdrant_host: str = DEFAULT_QDRANT_HOST, qdrant_port: int = DEFAULT_QDRANT_PORT) -> None:
    """Initialize search service (lazy loading pattern)."""
    logger.info("Search service configured for lazy loading")
    # Encoder and client will be initialized on first request


def get_search_service_health(
    qdrant_host: str = DEFAULT_QDRANT_HOST, qdrant_port: int = DEFAULT_QDRANT_PORT
) -> dict:
    """Get health status of search service."""
    global _encoder, _client

    encoder_loaded = _encoder is not None

    qdrant_connected = False
    try:
        if _client is None:
            test_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        else:
            test_client = _client
        test_client.get_collections()
        qdrant_connected = True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")

    return {
        "encoder_loaded": encoder_loaded,
        "qdrant_connected": qdrant_connected,
    }


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    qdrant_host: str = DEFAULT_QDRANT_HOST,
    qdrant_port: int = DEFAULT_QDRANT_PORT,
) -> SearchResponse:
    """Semantic search for similar toxic content."""
    try:
        encoder = get_encoder()
        client = get_client(qdrant_host, qdrant_port)

        # Perform search
        results = search_similar(
            query=request.query,
            encoder=encoder,
            client=client,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )

        # Convert to response schema
        result_items = [
            SearchResultItem(
                text=result.text,
                score=result.score,
                metadata=result.metadata,
            )
            for result in results
        ]

        return SearchResponse(
            query=request.query,
            results=result_items,
            count=len(result_items),
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))