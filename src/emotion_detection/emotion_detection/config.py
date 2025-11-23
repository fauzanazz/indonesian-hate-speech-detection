"""Configuration utilities for the emotion detection package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, HttpUrl, PositiveFloat, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmotionSettings(BaseSettings):
    """Application settings with environment and YAML support."""

    model_config = SettingsConfigDict(env_prefix="EMOTION_", extra="ignore")

    # Data paths
    data_dir: Path = Field(
        default=Path("dataset") / "Emotion Dataset from Indonesian Public Opinion",
        description="Directory containing the emotion CSV files.",
    )
    output_dir: Path = Field(
        default=Path("artifacts") / "emotion",
        description="Base directory for model artifacts and reports.",
    )

    # General runtime
    model_name: str = Field(
        default="logreg",
        description="Model identifier registered in the model registry.",
    )
    random_state: PositiveInt = Field(default=42, description="Random seed.")
    test_size: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Fraction reserved for evaluation split."
    )

    # TF-IDF / classic ML params
    max_features: Optional[int] = Field(
        default=20000, description="Maximum vocabulary size for TF-IDF."
    )
    ngram_min: PositiveInt = Field(default=1, description="Minimum n-gram size.")
    ngram_max: PositiveInt = Field(default=2, description="Maximum n-gram size.")

    # Transformer defaults
    pretrained_model_name: str = Field(
        default="indobenchmark/indobert-base-p1",
        description="HuggingFace model identifier for IndoBERT.",
    )
    num_epochs: PositiveInt = Field(default=3, description="Training epochs.")
    learning_rate: PositiveFloat = Field(default=2e-5, description="Optimizer LR.")
    batch_size: PositiveInt = Field(default=16, description="Training batch size.")
    warmup_ratio: float = Field(
        default=0.06, ge=0.0, lt=1.0, description="Warmup ratio for scheduler."
    )
    weight_decay: float = Field(
        default=0.01, ge=0.0, description="Weight decay for optimizer."
    )

    # API server
    api_host: str = Field(default="0.0.0.0", description="API bind host.")
    api_port: PositiveInt = Field(default=8000, description="API port.")

    # Optional monitoring hook
    metrics_url: Optional[HttpUrl] = Field(
        default=None,
        description="Optional external metrics sink endpoint.",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Root log level.")

    @property
    def ngram_range(self) -> tuple[int, int]:
        """Return scikit-learn friendly n-gram tuple."""

        return (self.ngram_min, self.ngram_max)

    @model_validator(mode="after")
    def _check_ranges(self) -> "EmotionSettings":
        if self.ngram_max < self.ngram_min:
            raise ValueError("ngram_max must be >= ngram_min")
        return self


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def load_settings(
    config_path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None
) -> EmotionSettings:
    """Load settings from optional YAML + explicit overrides + environment."""

    merged: Dict[str, Any] = {}
    sources: List[Dict[str, Any]] = []

    if config_path is not None:
        sources.append(_read_yaml(config_path))

    if overrides:
        sources.append(overrides)

    for source in sources:
        merged.update(source)

    return EmotionSettings(**merged)


