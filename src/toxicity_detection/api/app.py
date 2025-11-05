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
            
            # Auto-detect default model paths if not provided
            resolved_model_paths = model_paths
            if resolved_model_paths is None:
                # Try to find project root (models/ directory should be at project root)
                # Strategy 1: Relative to this file
                project_root = Path(__file__).parent.parent.parent.parent
                # Strategy 2: Check if models/ exists in current working directory
                if not (project_root / "models").exists():
                    cwd = Path.cwd()
                    if (cwd / "models").exists():
                        project_root = cwd
                
                resolved_model_paths = {
                    "basic": project_root / "models" / "tfidf",
                    "contextual": project_root / "models" / "bilstm",
                    "sociolinguistic": project_root / "models" / "transformer",
                }
                logger.info("Using default model paths", project_root=str(project_root), paths={k: str(v) for k, v in resolved_model_paths.items()})
            
            # Load each model independently to avoid one failure blocking others
            # Tier 1: TF-IDF model
            try:
                basic_model = TFIDFModel(name="tfidf_lr")
                basic_path = resolved_model_paths.get("basic")
                if basic_path and basic_path.exists() and (basic_path / "tfidf_lr_pipeline.joblib").exists():
                    logger.info("Loading TF-IDF model from disk", path=str(basic_path))
                    basic_model.load(basic_path)
                else:
                    logger.warning("TF-IDF model not found, model will not be trained", path=str(basic_path) if basic_path else "None")
                registry.load_model("basic", basic_model)
            except Exception as e:
                logger.error("Failed to load TF-IDF model", error=str(e), exc_info=True)
                # Create untrained model instance so endpoint can return proper error
                basic_model = TFIDFModel(name="tfidf_lr")
                registry.load_model("basic", basic_model)

            # Tier 2: BiLSTM model
            try:
                contextual_model = BiLSTMModel(name="bilstm")
                contextual_path = resolved_model_paths.get("contextual")
                if contextual_path and contextual_path.exists() and (contextual_path / "bilstm_metadata.joblib").exists():
                    logger.info("Loading BiLSTM model from disk", path=str(contextual_path))
                    contextual_model.load(contextual_path)
                else:
                    logger.warning("BiLSTM model not found, model will not be trained", path=str(contextual_path) if contextual_path else "None")
                registry.load_model("contextual", contextual_model)
            except Exception as e:
                logger.error("Failed to load BiLSTM model", error=str(e), exc_info=True)
                contextual_model = BiLSTMModel(name="bilstm")
                registry.load_model("contextual", contextual_model)

            # Tier 3: Transformer model
            try:
                sociolinguistic_model = TransformerModel(name="indobert")
                sociolinguistic_path = resolved_model_paths.get("sociolinguistic")
                if sociolinguistic_path and sociolinguistic_path.exists() and (sociolinguistic_path / "indobert_metadata.joblib").exists():
                    logger.info("Loading Transformer model from disk", path=str(sociolinguistic_path))
                    sociolinguistic_model.load(sociolinguistic_path)
                else:
                    logger.warning("Transformer model not found, model will not be trained", path=str(sociolinguistic_path) if sociolinguistic_path else "None")
                registry.load_model("sociolinguistic", sociolinguistic_model)
            except Exception as e:
                logger.error("Failed to load Transformer model", error=str(e), exc_info=True)
                sociolinguistic_model = TransformerModel(name="indobert")
                registry.load_model("sociolinguistic", sociolinguistic_model)

            loaded_count = sum(1 for model in registry.models.values() if model is not None and model.is_trained)
            logger.info("Model loading complete", loaded_count=loaded_count, total_count=len(registry.models))

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        models_loaded = {}
        for tier in ["basic", "contextual", "sociolinguistic"]:
            model = registry.models.get(tier)
            models_loaded[tier] = model is not None and model.is_trained
        
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            models_loaded=models_loaded,
        )

    # Tier 1: Basic toxicity detection
    @app.post("/api/v1/basic", response_model=ToxicityResponse)
    async def predict_basic(request: ToxicityRequest, req: Request) -> ToxicityResponse:
        trace_id = req.state.trace_id
        logger.info("Basic toxicity detection", trace_id=trace_id)

        model = registry.get_model("basic")
        if not model.is_trained:
            raise HTTPException(
                status_code=503,
                detail="TF-IDF model is not trained. Please train the model first.",
            )

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
        if not model.is_trained:
            raise HTTPException(
                status_code=503,
                detail="BiLSTM model is not trained. Please train the model first.",
            )

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
        if not model.is_trained:
            raise HTTPException(
                status_code=503,
                detail="Transformer model is not trained. Please train the model first.",
            )

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
