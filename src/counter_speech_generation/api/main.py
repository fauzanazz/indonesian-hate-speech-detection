"""FastAPI application entry point."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from counter_speech_generation import __version__
from counter_speech_generation.api.routes import router
from counter_speech_generation.config import get_config
from counter_speech_generation.utils import setup_logging


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    setup_logging()

    app = FastAPI(
        title="Counter Speech Generation API",
        description="Generate counter speech for Indonesian toxic content using IndoT5",
        version=__version__,
    )

    # CORS configuration
    config = get_config().api
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include routes
    app.include_router(router, prefix="/api/v1")

    logger.info("FastAPI application created")

    return app


app = create_app()


def run() -> None:
    """CLI entry point for serve-api command."""
    config = get_config().api

    logger.info(f"Starting API server on {config.host}:{config.port}")

    uvicorn.run(
        "counter_speech_generation.api.main:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


if __name__ == "__main__":
    run()

