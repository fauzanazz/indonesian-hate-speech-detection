"""Combined Pydantic models for unified API."""

from typing import Optional

from pydantic import BaseModel, Field


# Toxicity Detection Schemas
class ToxicityRequest(BaseModel):
    """Request model for toxicity detection."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    return_explanation: bool = Field(
        default=False,
        description="Whether to return explanation (feature importance/attention)",
    )


class ToxicityResponse(BaseModel):
    """Response model for toxicity detection."""

    text: str = Field(..., description="Input text")
    is_toxic: bool = Field(..., description="Whether text is toxic")
    toxicity_score: float = Field(..., ge=0.0, le=1.0, description="Toxicity probability (0-1)")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    tier: str = Field(..., description="BEAM tier (basic/contextual/sociolinguistic)")
    model_name: str = Field(..., description="Model used for prediction")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    trace_id: str = Field(..., description="Request trace ID")
    explanation: Optional[dict] = Field(None, description="Optional explanation data")


# Search Schemas
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


# Health Check Schema
class ServiceHealth(BaseModel):
    """Health status for a service component."""

    status: str = Field(..., description="Component status")
    details: dict = Field(default_factory=dict, description="Additional details")


class HealthResponse(BaseModel):
    """Combined health check response."""

    status: str = Field(..., description="Overall service status (healthy/degraded)")
    version: str = Field(..., description="API version")
    services: dict[str, ServiceHealth] = Field(..., description="Status of each service")