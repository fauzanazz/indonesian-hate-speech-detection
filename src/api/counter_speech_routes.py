"""API routes for counter speech generation."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas import CounterSpeechRequest, CounterSpeechResponse
from counter_speech_generation.models.encoder import CounterSpeechGenerator, load_generator
from counter_speech_generation.config import get_config as get_cs_config

router = APIRouter(tags=["counter-speech"])

# Global state for generator
_generator: CounterSpeechGenerator | None = None


def get_generator() -> CounterSpeechGenerator:
    """Get or initialize generator singleton."""
    global _generator

    if _generator is None:
        logger.info("Loading counter speech generator...")
        try:
            _generator = load_generator()
            logger.info("Counter speech generator loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load counter speech generator: {e}", exc_info=True)
            raise

    return _generator


def initialize_counter_speech_service() -> None:
    """Initialize counter speech generation service."""
    try:
        get_generator()
        logger.info("Counter speech service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize counter speech service: {e}", exc_info=True)
        raise


def get_counter_speech_service_health() -> dict:
    """Get health status of counter speech service."""
    try:
        generator = _generator
        model_loaded = generator is not None
        
        return {
            "model_loaded": model_loaded,
            "model_name": get_cs_config().model.base_model if model_loaded else None,
            "device": generator.device if model_loaded else None,
        }
    except Exception as e:
        logger.error(f"Error checking counter speech service health: {e}")
        return {
            "model_loaded": False,
            "model_name": None,
            "device": None,
            "error": str(e),
        }


@router.post("/counter-speech/generate", response_model=CounterSpeechResponse)
async def generate_counter_speech(request: CounterSpeechRequest) -> CounterSpeechResponse:
    """Generate counter speech for toxic text."""
    try:
        generator = get_generator()
        config = get_cs_config()

        # Generate counter speech
        counter_speech = generator.generate(
            request.text,
            max_length=request.max_length,
            num_beams=request.num_beams,
            temperature=request.temperature,
            do_sample=request.do_sample,
        )

        # Prepare generation config for response
        generation_config = {
            "max_length": request.max_length or config.model.max_target_length,
            "num_beams": request.num_beams or config.model.num_beams,
            "temperature": request.temperature or config.model.temperature,
            "do_sample": request.do_sample if request.do_sample is not None else config.model.do_sample,
            "length_penalty": config.model.length_penalty,
            "repetition_penalty": config.model.repetition_penalty,
        }

        return CounterSpeechResponse(
            text=request.text,
            counter_speech=counter_speech,
            model=config.model.base_model,
            generation_config=generation_config,
        )

    except Exception as e:
        logger.error(f"Counter speech generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


