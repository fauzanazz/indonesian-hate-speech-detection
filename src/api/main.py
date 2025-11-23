"""Unified FastAPI application combining toxicity detection and search services."""

import sys
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api import __version__
from api.config import get_config
from api.schemas import HealthResponse, ServiceHealth
from api.search_routes import (
    get_search_service_health,
    initialize_search_service,
    router as search_router,
)
from api.toxicity_routes import (
    get_toxicity_service_health,
    initialize_toxicity_service,
    router as toxicity_router,
)


def setup_logging() -> None:
    """Configure loguru logging."""
    config = get_config().logging

    logger.remove()
    logger.add(
        sys.stderr,
        format=config.format,
        level=config.level,
        colorize=True,
    )
    logger.add(
        "logs/api_{time}.log",
        format=config.format,
        level=config.level,
        rotation=config.rotation,
        retention=config.retention,
        compression="zip",
    )


def create_app() -> FastAPI:
    """Create and configure unified FastAPI application."""
    setup_logging()

    app = FastAPI(
        title="Indonesian Toxicity Detection & Search API",
        description=(
            "Unified API for toxicity detection using BEAM architecture "
            "and semantic search for toxic content"
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    config = get_config()
    if config.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Trace ID middleware
    @app.middleware("http")
    async def trace_id_middleware(request: Request, call_next):
        """Add trace ID to all requests."""
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id

        with logger.contextualize(trace_id=trace_id):
            start_time = time.time()
            response = await call_next(request)
            latency = (time.time() - start_time) * 1000

            logger.info(
                f"Request completed | method={request.method} "
                f"path={request.url.path} status={response.status_code} "
                f"latency={latency:.2f}ms"
            )

            return response

    # Startup event
    @app.on_event("startup")
    async def startup_event() -> None:
        logger.info("Starting unified API server...")

        # Initialize toxicity detection service
        try:
            model_paths = {
                "basic": config.toxicity_models.basic_path,
                "contextual": config.toxicity_models.contextual_path,
                "sociolinguistic": config.toxicity_models.sociolinguistic_path,
            }
            ensemble_config_path = config.toxicity_models.ensemble_config_path

            initialize_toxicity_service(
                model_paths=model_paths,
                ensemble_config_path=ensemble_config_path,
            )
            logger.info("Toxicity detection service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize toxicity service: {e}", exc_info=True)

        # Initialize search service
        try:
            initialize_search_service(
                qdrant_host=config.qdrant.host,
                qdrant_port=config.qdrant.port,
            )
            logger.info("Search service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize search service: {e}", exc_info=True)

        logger.info("API server startup complete")

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Unified health check for all services."""
        config = get_config()

        # Get toxicity service health
        toxicity_health = get_toxicity_service_health()
        toxicity_status = "healthy" if any(toxicity_health["models_loaded"].values()) else "degraded"

        # Get search service health
        search_health = get_search_service_health(
            qdrant_host=config.qdrant.host,
            qdrant_port=config.qdrant.port,
        )
        search_status = "healthy" if search_health["qdrant_connected"] else "degraded"

        # Overall status
        overall_status = "healthy" if (toxicity_status == "healthy" or search_status == "healthy") else "degraded"

        return HealthResponse(
            status=overall_status,
            version=__version__,
            services={
                "toxicity_detection": ServiceHealth(
                    status=toxicity_status,
                    details=toxicity_health,
                ),
                "search": ServiceHealth(
                    status=search_status,
                    details=search_health,
                ),
            },
        )

    # Include routers
    app.include_router(toxicity_router, prefix="/api/v1")
    app.include_router(search_router, prefix="/api/v1")

    logger.info("FastAPI application created")

    return app


app = create_app()


def run() -> None:
    """CLI entry point for starting the API server."""
    config = get_config().api

    logger.info(f"Starting API server on {config.host}:{config.port}")

    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()