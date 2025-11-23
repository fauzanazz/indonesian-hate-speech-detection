"""Ensemble inference utilities for production inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Tuple

from omegaconf import OmegaConf

from toxicity_detection.models.base import BaseModel
from toxicity_detection.models.bilstm_model import BiLSTMModel
from toxicity_detection.models.tfidf_model import TFIDFModel
from toxicity_detection.models.transformer_model import TransformerModel

CascadeThresholds = Dict[str, float]


@dataclass(frozen=True)
class EnsembleConfig:
    weights: Dict[str, float]
    threshold: float = 0.5
    abstain_margin: float = 0.05
    cascade_thresholds: Optional[CascadeThresholds] = None

    def __post_init__(self) -> None:
        if self.cascade_thresholds is None:
            object.__setattr__(
                self,
                "cascade_thresholds",
                self._default_cascade_thresholds(),
            )

    @staticmethod
    def _default_cascade_thresholds() -> CascadeThresholds:
        return {
            "basic": 0.35,
            "contextual": 0.25,
        }

    @classmethod
    def default(cls) -> "EnsembleConfig":
        return cls(
            weights={"basic": 0.25, "contextual": 0.35, "sociolinguistic": 0.4},
            threshold=0.52,
            abstain_margin=0.05,
            cascade_thresholds=cls._default_cascade_thresholds(),
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "EnsembleConfig":
        weights = {
            key: float(value)
            for key, value in mapping.get("weights", {}).items()
        }

        threshold = float(mapping.get("threshold", 0.5))
        abstain_margin = float(mapping.get("abstain_margin", 0.05))

        raw_cascade = mapping.get("cascade_thresholds", mapping.get("cascade", {}))
        cascade_thresholds = {
            key: float(value)
            for key, value in raw_cascade.items()
        }

        if not cascade_thresholds:
            cascade_thresholds = cls._default_cascade_thresholds()

        config = cls(
            weights=weights or cls.default().weights,
            threshold=threshold,
            abstain_margin=abstain_margin,
            cascade_thresholds=cascade_thresholds,
        )
        return config


def load_ensemble_config(config_path: Path) -> EnsembleConfig:
    if not config_path.exists():
        return EnsembleConfig.default()

    config_dict = OmegaConf.to_container(
        OmegaConf.load(config_path),
        resolve=True,
    )

    ensemble_section = {}
    if isinstance(config_dict, Mapping):
        ensemble_section = config_dict.get("ensemble", config_dict)

    return EnsembleConfig.from_mapping(ensemble_section)


@dataclass
class ModelPrediction:
    tier: str
    score: Optional[float]
    latency_ms: float
    used: bool
    explanation: Optional[Dict[str, Any]] = None


@dataclass
class EnsemblePrediction:
    score: float
    is_toxic: bool
    weights: Dict[str, float]
    per_model: Dict[str, ModelPrediction]
    total_latency_ms: float
    review_recommended: bool


class EnsembleService:
    MODEL_ORDER: tuple[str, ...] = ("basic", "contextual", "sociolinguistic")

    def __init__(self, registry: Any, config: EnsembleConfig) -> None:
        self.registry = registry
        self.config = config

    def predict(self, text: str, return_explanation: bool = False) -> EnsemblePrediction:
        scores: Dict[str, float] = {}
        per_model: Dict[str, ModelPrediction] = {}

        for tier in self.MODEL_ORDER:
            model = self.registry.models.get(tier)
            if model is None or not getattr(model, "is_trained", False):
                per_model[tier] = ModelPrediction(tier, None, 0.0, False)
                continue

            score, latency_ms, explanation = self._run_model(
                model,
                text,
                return_explanation,
            )

            scores[tier] = score
            per_model[tier] = ModelPrediction(
                tier=tier,
                score=score,
                latency_ms=latency_ms,
                used=True,
                explanation=explanation,
            )

            if self._should_short_circuit(tier, score):
                break

        if not scores:
            raise RuntimeError("No trained models available for ensemble prediction")

        aggregated_score, normalized_weights = self._aggregate(scores)
        is_toxic = aggregated_score >= self.config.threshold
        review_recommended = (
            abs(aggregated_score - self.config.threshold)
            < self.config.abstain_margin
        )

        total_latency = sum(
            prediction.latency_ms
            for prediction in per_model.values()
            if prediction.used
        )

        return EnsemblePrediction(
            score=aggregated_score,
            is_toxic=is_toxic,
            weights=normalized_weights,
            per_model=per_model,
            total_latency_ms=total_latency,
            review_recommended=review_recommended,
        )

    def _should_short_circuit(self, tier: str, score: float) -> bool:
        cascade = self.config.cascade_thresholds or {}
        margin = cascade.get(tier)
        if margin is None:
            return False
        return abs(score - 0.5) >= margin

    def _aggregate(self, scores: Mapping[str, float]) -> Tuple[float, Dict[str, float]]:
        weights = {
            tier: self.config.weights.get(tier, 0.0)
            for tier in scores
        }

        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            normalized = {tier: 1.0 / len(scores) for tier in scores}
        else:
            normalized = {
                tier: weight / weight_sum
                for tier, weight in weights.items()
            }

        aggregated = sum(normalized[tier] * scores[tier] for tier in scores)
        return aggregated, normalized

    def _run_model(
        self,
        model: BaseModel,
        text: str,
        return_explanation: bool,
    ) -> tuple[float, float, Optional[Dict[str, Any]]]:
        start = perf_counter()
        proba = float(model.predict_proba([text])[0])
        latency_ms = (perf_counter() - start) * 1000

        explanation: Optional[Dict[str, Any]] = None
        if return_explanation:
            explanation = self._build_explanation(model, text)

        return proba, latency_ms, explanation

    def _build_explanation(
        self,
        model: BaseModel,
        text: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(model, TFIDFModel):
                return {
                    "feature_importance": model.get_feature_importance(top_k=10),
                }

            if isinstance(model, TransformerModel):
                tokens, attention = model.get_attention_weights(text)
                if attention.ndim == 2:
                    attention = attention.mean(axis=0)
                attention_scores = attention.tolist()
                return {
                    "tokens": tokens[:50],
                    "attention_scores": attention_scores[:50],
                }

            if isinstance(model, BiLSTMModel):
                return None
        except Exception:
            return None

        return None

