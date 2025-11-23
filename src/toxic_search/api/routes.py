"""API endpoints for toxic content search."""

from fastapi import APIRouter, HTTPException
from loguru import logger
from qdrant_client import QdrantClient

from toxic_search import __version__
from toxic_search.api.schemas import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from toxic_search.config import get_config
from toxic_search.index.search import search_similar
from toxic_search.models.encoder import ToxicEncoder

router = APIRouter()

# Global state for encoder and client
_encoder: ToxicEncoder | None = None
_client: QdrantClient | None = None


def get_encoder() -> ToxicEncoder:
    """Get or initialize encoder singleton."""
    global _encoder
    
    if _encoder is None:
        from toxic_search.models.encoder import load_encoder
        logger.info("Loading encoder...")
        _encoder = load_encoder()
    
    return _encoder


def get_client() -> QdrantClient:
    """Get or initialize Qdrant client singleton."""
    global _client
    
    if _client is None:
        config = get_config().qdrant
        logger.info(f"Connecting to Qdrant at {config.host}:{config.port}")
        _client = QdrantClient(host=config.host, port=config.port)
    
    return _client


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        client = get_client()
        collections = client.get_collections()
        qdrant_connected = True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        qdrant_connected = False
    
    return HealthResponse(
        status="healthy" if qdrant_connected else "degraded",
        version=__version__,
        qdrant_connected=qdrant_connected,
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Semantic search for similar toxic content."""
    try:
        encoder = get_encoder()
        client = get_client()
        
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