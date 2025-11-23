"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request schema for counter speech generation."""

    text: str = Field(..., min_length=1, description="Input toxic text")
    max_length: int | None = Field(
        None, ge=10, le=512, description="Maximum generation length"
    )
    num_beams: int | None = Field(
        None, ge=1, le=10, description="Number of beams for beam search"
    )
    temperature: float | None = Field(
        None, ge=0.1, le=2.0, description="Sampling temperature"
    )
    do_sample: bool | None = Field(None, description="Whether to use sampling")


class GenerateResponse(BaseModel):
    """Response schema for counter speech generation."""

    text: str = Field(..., description="Input toxic text")
    counter_speech: str = Field(..., description="Generated counter speech")
    model: str = Field(..., description="Model name used")
    generation_config: dict | None = Field(
        None, description="Generation parameters used (max_length, num_beams, etc.)"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Model loading status")

