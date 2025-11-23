"""API endpoints for counter speech generation."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from counter_speech_generation import __version__
from counter_speech_generation.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from counter_speech_generation.config import get_config
from counter_speech_generation.models.encoder import CounterSpeechGenerator, load_generator

router = APIRouter()

# Global state for generator
_generator: CounterSpeechGenerator | None = None


def get_generator() -> CounterSpeechGenerator:
    """Get or initialize generator singleton."""
    global _generator

    if _generator is None:
        logger.info("Loading generator...")
        _generator = load_generator()

    return _generator


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        generator = get_generator()
        model_loaded = generator is not None
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=__version__,
        model_loaded=model_loaded,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate counter speech for given toxic text."""
    try:
        generator = get_generator()
        config = get_config()

        # Generate counter speech
        counter_speech = generator.generate(
            request.text,
            max_length=request.max_length,
            num_beams=request.num_beams,
            temperature=request.temperature,
            do_sample=request.do_sample,
        )

        # Include generation parameters in response
        generation_config = {
            "max_length": request.max_length or config.model.max_target_length,
            "num_beams": request.num_beams or config.model.num_beams,
            "temperature": request.temperature or config.model.temperature,
            "do_sample": request.do_sample if request.do_sample is not None else config.model.do_sample,
            "length_penalty": config.model.length_penalty,
            "repetition_penalty": config.model.repetition_penalty,
        }
        
        response = GenerateResponse(
            text=request.text,
            counter_speech=counter_speech,
            model=config.model.base_model,
            generation_config=generation_config,
        )
        
        return response

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

