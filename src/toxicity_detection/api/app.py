import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel, Field

from toxicity_detection.models.base import BaseModel
from toxicity_detection.models.bilstm_model import BiLSTMModel
from toxicity_detection.models.tfidf_model import TFIDFModel
from toxicity_detection.models.transformer_model import TransformerModel
from toxicity_detection.utils.logging import generate_trace_id, setup_logger

logger = setup_logger(__name__)


class ToxicityRequest(PydanticBaseModel):
    """Request model for toxicity detection."""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    return_explanation: bool = Field(
        default=False,
        description="Whether to return explanation (feature importance/attention)",
    )


class ToxicityResponse(PydanticBaseModel):
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


class HealthResponse(PydanticBaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]


# Model registry
class ModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, Optional[BaseModel]] = {
            "basic": None,
            "contextual": None,
            "sociolinguistic": None,
        }

    def load_model(self, tier: str, model: BaseModel) -> None:
        if tier not in self.models:
            raise ValueError(f"Invalid tier: {tier}")
        self.models[tier] = model
        logger.info("Model loaded", tier=tier, model=model.name)

    def get_model(self, tier: str) -> BaseModel:
        model = self.models.get(tier)
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model for tier '{tier}' not loaded",
            )
        return model

    def is_loaded(self, tier: str) -> bool:
        return self.models.get(tier) is not None


# Global registry
registry = ModelRegistry()


def create_app(
    load_models: bool = True,
    model_paths: Optional[dict[str, Path]] = None,
) -> FastAPI:
    app = FastAPI(
        title="Indonesian Toxicity Detection API",
        description=(
            "API for detecting toxicity in Indonesian text "
            "using BEAM (Boxology Extended Annotation Model) architecture"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore
        """Log all requests with trace ID."""
        trace_id = generate_trace_id()
        request.state.trace_id = trace_id

        start_time = time.time()
        response = await call_next(request)
        latency = (time.time() - start_time) * 1000

        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=f"{latency:.2f}",
            trace_id=trace_id,
        )

        return response

    @app.on_event("startup")
    async def startup_event() -> None:
        logger.info("Starting API server...")

        if load_models:
            logger.info("Loading models...")
            try:
                # Tier 1: TF-IDF model
                basic_model = TFIDFModel(name="tfidf_lr")
                if model_paths and "basic" in model_paths:
                    basic_model.load(model_paths["basic"])
                registry.load_model("basic", basic_model)

                # Tier 2: BiLSTM model
                contextual_model = BiLSTMModel(name="bilstm")
                if model_paths and "contextual" in model_paths:
                    contextual_model.load(model_paths["contextual"])
                registry.load_model("contextual", contextual_model)

                # Tier 3: Transformer model
                sociolinguistic_model = TransformerModel(name="indobert")
                if model_paths and "sociolinguistic" in model_paths:
                    sociolinguistic_model.load(model_paths["sociolinguistic"])
                registry.load_model("sociolinguistic", sociolinguistic_model)

                logger.info("All models loaded successfully")
            except Exception as e:
                logger.error("Failed to load models", error=str(e))
                # Models will remain None, endpoints will return 503

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            models_loaded={
                "basic": registry.is_loaded("basic"),
                "contextual": registry.is_loaded("contextual"),
                "sociolinguistic": registry.is_loaded("sociolinguistic"),
            },
        )

    # Tier 1: Basic toxicity detection
    @app.post("/api/v1/basic", response_model=ToxicityResponse)
    async def predict_basic(request: ToxicityRequest, req: Request) -> ToxicityResponse:
        trace_id = req.state.trace_id
        logger.info("Basic toxicity detection", trace_id=trace_id)

        model = registry.get_model("basic")

        start = time.perf_counter()
        toxicity_score = float(model.predict_proba([request.text])[0])
        latency_ms = (time.perf_counter() - start) * 1000

        is_toxic = toxicity_score > 0.5
        confidence = _get_confidence_level(toxicity_score)

        explanation = None
        if request.return_explanation and isinstance(model, TFIDFModel):
            try:
                explanation = {"feature_importance": model.get_feature_importance(top_k=10)}
            except Exception as e:
                logger.warning("Failed to get explanation", error=str(e))

        return ToxicityResponse(
            text=request.text,
            is_toxic=is_toxic,
            toxicity_score=toxicity_score,
            confidence=confidence,
            tier="basic",
            model_name=model.name,
            latency_ms=latency_ms,
            trace_id=trace_id,
            explanation=explanation,
        )

    # Tier 2: Contextual toxicity detection
    @app.post("/api/v1/contextual", response_model=ToxicityResponse)
    async def predict_contextual(request: ToxicityRequest, req: Request) -> ToxicityResponse:
        trace_id = req.state.trace_id
        logger.info("Contextual toxicity detection", trace_id=trace_id)

        model = registry.get_model("contextual")

        start = time.perf_counter()
        toxicity_score = float(model.predict_proba([request.text])[0])
        latency_ms = (time.perf_counter() - start) * 1000

        is_toxic = toxicity_score > 0.5
        confidence = _get_confidence_level(toxicity_score)

        return ToxicityResponse(
            text=request.text,
            is_toxic=is_toxic,
            toxicity_score=toxicity_score,
            confidence=confidence,
            tier="contextual",
            model_name=model.name,
            latency_ms=latency_ms,
            trace_id=trace_id,
            explanation=None,
        )

    # Tier 3: Sociolinguistic toxicity detection
    @app.post("/api/v1/sociolinguistic", response_model=ToxicityResponse)
    async def predict_sociolinguistic(
        request: ToxicityRequest,
        req: Request,
    ) -> ToxicityResponse:
        trace_id = req.state.trace_id
        logger.info("Sociolinguistic toxicity detection", trace_id=trace_id)

        model = registry.get_model("sociolinguistic")

        start = time.perf_counter()
        toxicity_score = float(model.predict_proba([request.text])[0])
        latency_ms = (time.perf_counter() - start) * 1000

        is_toxic = toxicity_score > 0.5
        confidence = _get_confidence_level(toxicity_score)

        explanation = None
        if request.return_explanation and isinstance(model, TransformerModel):
            try:
                tokens, attention = model.get_attention_weights(request.text)
                avg_attention = attention.mean(axis=0).tolist()
                explanation = {
                    "tokens": tokens[:50], 
                    "attention_scores": avg_attention[:50],
                }
            except Exception as e:
                logger.warning("Failed to get attention", error=str(e))

        return ToxicityResponse(
            text=request.text,
            is_toxic=is_toxic,
            toxicity_score=toxicity_score,
            confidence=confidence,
            tier="sociolinguistic",
            model_name=model.name,
            latency_ms=latency_ms,
            trace_id=trace_id,
            explanation=explanation,
        )

    return app


def _get_confidence_level(score: float) -> str:
    distance_from_threshold = abs(score - 0.5)
    if distance_from_threshold > 0.3:
        return "high"
    elif distance_from_threshold > 0.15:
        return "medium"
    else:
        return "low"


if __name__ == "__main__":
    import uvicorn

    app = create_app(load_models=False)  
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
