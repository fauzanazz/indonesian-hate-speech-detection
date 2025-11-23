"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request schema for semantic search."""

    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int | None = Field(None, ge=1, le=100, description="Number of results to return")
    score_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class SearchResultItem(BaseModel):
    """Single search result item."""

    text: str = Field(..., description="Matched text")
    score: float = Field(..., description="Similarity score")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseModel):
    """Response schema for semantic search."""

    query: str = Field(..., description="Original query")
    results: list[SearchResultItem] = Field(..., description="Search results")
    count: int = Field(..., description="Number of results returned")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    qdrant_connected: bool = Field(..., description="Qdrant connection status")