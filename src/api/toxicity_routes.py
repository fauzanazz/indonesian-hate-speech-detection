"""Toxicity detection routes for unified API."""

import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from api.schemas import ToxicityRequest, ToxicityResponse
from toxicity_detection.models.base import BaseModel
from toxicity_detection.models.bilstm_model import BiLSTMModel
from toxicity_detection.models.tfidf_model import TFIDFModel
from toxicity_detection.models.ensemble import EnsembleService, load_ensemble_config
from toxicity_detection.models.transformer_model import TransformerModel

router = APIRouter(prefix="/toxicity", tags=["Toxicity Detection"])


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
        logger.info(f"Model loaded: tier={tier}, model={model.name}")

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


# Global registry and ensemble service
registry = ModelRegistry()
ensemble_service: Optional[EnsembleService] = None


def initialize_toxicity_service(
    model_paths: dict[str, Path] | None = None,
    ensemble_config_path: Path | None = None,
) -> None:
    """Initialize toxicity detection models and ensemble service."""
    logger.info("Initializing toxicity detection service...")

    # Auto-detect paths if not provided
    if model_paths is None:
        primary_root = Path(__file__).parent.parent.parent
        candidate_roots = [primary_root, Path.cwd()]
        project_root = next(
            (root for root in candidate_roots if (root / "configs").exists()),
            primary_root,
        )
        model_root = next(
            (root for root in candidate_roots if (root / "models").exists()),
            project_root,
        )
        model_paths = {
            "basic": model_root / "models" / "tfidf",
            "contextual": model_root / "models" / "bilstm",
            "sociolinguistic": model_root / "models" / "transformer",
        }
        logger.info(f"Using default model paths: {model_paths}")

    # Load Tier 1: TF-IDF model
    try:
        basic_model = TFIDFModel(name="tfidf_lr")
        basic_path = model_paths.get("basic")
        if basic_path and basic_path.exists() and (basic_path / "tfidf_lr_pipeline.joblib").exists():
            logger.info(f"Loading TF-IDF model from {basic_path}")
            basic_model.load(basic_path)
        else:
            logger.warning(f"TF-IDF model not found at {basic_path}")
        registry.load_model("basic", basic_model)
    except Exception as e:
        logger.error(f"Failed to load TF-IDF model: {e}", exc_info=True)
        basic_model = TFIDFModel(name="tfidf_lr")
        registry.load_model("basic", basic_model)

    # Load Tier 2: BiLSTM model
    try:
        contextual_model = BiLSTMModel(name="bilstm")
        contextual_path = model_paths.get("contextual")
        if contextual_path and contextual_path.exists() and (contextual_path / "bilstm_metadata.joblib").exists():
            logger.info(f"Loading BiLSTM model from {contextual_path}")
            contextual_model.load(contextual_path)
        else:
            logger.warning(f"BiLSTM model not found at {contextual_path}")
        registry.load_model("contextual", contextual_model)
    except Exception as e:
        logger.error(f"Failed to load BiLSTM model: {e}", exc_info=True)
        contextual_model = BiLSTMModel(name="bilstm")
        registry.load_model("contextual", contextual_model)

    # Load Tier 3: Transformer model
    try:
        sociolinguistic_model = TransformerModel(name="indobert")
        sociolinguistic_path = model_paths.get("sociolinguistic")
        if sociolinguistic_path and sociolinguistic_path.exists() and (sociolinguistic_path / "indobert_metadata.joblib").exists():
            logger.info(f"Loading Transformer model from {sociolinguistic_path}")
            sociolinguistic_model.load(sociolinguistic_path)
        else:
            logger.warning(f"Transformer model not found at {sociolinguistic_path}")
        registry.load_model("sociolinguistic", sociolinguistic_model)
    except Exception as e:
        logger.error(f"Failed to load Transformer model: {e}", exc_info=True)
        sociolinguistic_model = TransformerModel(name="indobert")
        registry.load_model("sociolinguistic", sociolinguistic_model)

    loaded_count = sum(1 for model in registry.models.values() if model is not None and model.is_trained)
    logger.info(f"Model loading complete: {loaded_count}/{len(registry.models)} models trained")

    # Initialize ensemble service
    if ensemble_config_path is None:
        primary_root = Path(__file__).parent.parent.parent
        ensemble_config_path = primary_root / "configs" / "models" / "ensemble.yaml"

    try:
        ensemble_config = load_ensemble_config(ensemble_config_path)
        global ensemble_service
        ensemble_service = EnsembleService(registry, ensemble_config)
        logger.info(f"Ensemble service initialized: threshold={ensemble_config.threshold:.2f}")
    except Exception as e:
        logger.error(f"Failed to initialize ensemble service: {e}", exc_info=True)


def get_toxicity_service_health() -> dict:
    """Get health status of toxicity detection service."""
    models_status = {}
    for tier in ["basic", "contextual", "sociolinguistic"]:
        model = registry.models.get(tier)
        models_status[tier] = model is not None and model.is_trained

    return {
        "models_loaded": models_status,
        "ensemble_ready": ensemble_service is not None,
    }


def _get_confidence_level(score: float, threshold: float = 0.5) -> str:
    """Calculate confidence level based on distance from threshold."""
    distance_from_threshold = abs(score - threshold)
    if distance_from_threshold > 0.3:
        return "high"
    elif distance_from_threshold > 0.15:
        return "medium"
    else:
        return "low"


@router.post("/basic", response_model=ToxicityResponse)
async def predict_basic(request: ToxicityRequest, req: Request) -> ToxicityResponse:
    """Tier 1: Basic toxicity detection using TF-IDF + Logistic Regression."""
    trace_id = req.state.trace_id
    logger.info(f"Basic toxicity detection | trace_id={trace_id}")

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
            logger.warning(f"Failed to get explanation: {e}")

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


@router.post("/contextual", response_model=ToxicityResponse)
async def predict_contextual(request: ToxicityRequest, req: Request) -> ToxicityResponse:
    """Tier 2: Contextual toxicity detection using BiLSTM."""
    trace_id = req.state.trace_id
    logger.info(f"Contextual toxicity detection | trace_id={trace_id}")

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


@router.post("/sociolinguistic", response_model=ToxicityResponse)
async def predict_sociolinguistic(
    request: ToxicityRequest,
    req: Request,
) -> ToxicityResponse:
    """Tier 3: Sociolinguistic toxicity detection using IndoBERT."""
    trace_id = req.state.trace_id
    logger.info(f"Sociolinguistic toxicity detection | trace_id={trace_id}")

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
            logger.warning(f"Failed to get attention: {e}")

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


@router.post("/ensemble", response_model=ToxicityResponse)
async def predict_ensemble(request: ToxicityRequest, req: Request) -> ToxicityResponse:
    """Ensemble prediction with weighted voting across all tiers."""
    trace_id = req.state.trace_id
    logger.info(f"Ensemble toxicity detection | trace_id={trace_id}")

    if ensemble_service is None:
        raise HTTPException(
            status_code=503,
            detail="Ensemble service is not initialized.",
        )

    try:
        result = ensemble_service.predict(
            request.text,
            return_explanation=request.return_explanation,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    confidence = _get_confidence_level(
        result.score,
        threshold=ensemble_service.config.threshold,
    )

    used_models = [
        tier
        for tier, prediction in result.per_model.items()
        if prediction.used
    ]
    model_name = "+".join(
        registry.models[tier].name  # type: ignore[index]
        for tier in used_models
        if registry.models.get(tier)
    )

    explanation: Optional[dict] = None
    if request.return_explanation:
        explanation = {
            "ensemble": {
                "weights": result.weights,
                "threshold": ensemble_service.config.threshold,
                "review_recommended": result.review_recommended,
            },
            "per_model": {},
        }

        for tier, prediction in result.per_model.items():
            if not prediction.used:
                continue
            explanation["per_model"][tier] = {
                "score": prediction.score,
                "latency_ms": prediction.latency_ms,
                "explanation": prediction.explanation,
            }

    return ToxicityResponse(
        text=request.text,
        is_toxic=result.is_toxic,
        toxicity_score=result.score,
        confidence=confidence,
        tier="ensemble",
        model_name=model_name or "ensemble",
        latency_ms=result.total_latency_ms,
        trace_id=trace_id,
        explanation=explanation,
    )