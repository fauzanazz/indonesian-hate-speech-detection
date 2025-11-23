"""Unified configuration for combined API services."""

from pathlib import Path
from typing import Literal

import tomli
from pydantic import Field
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    enable_cors: bool = True


class ToxicityModelConfig(BaseSettings):
    """Toxicity detection model paths."""

    model_dir: Path = Field(default=Path("models"))
    basic_path: Path | None = None
    contextual_path: Path | None = None
    sociolinguistic_path: Path | None = None
    ensemble_config_path: Path = Field(default=Path("configs/models/ensemble.yaml"))

    def model_post_init(self, __context) -> None:
        """Set default paths if not provided."""
        if self.basic_path is None:
            self.basic_path = self.model_dir / "tfidf"
        if self.contextual_path is None:
            self.contextual_path = self.model_dir / "bilstm"
        if self.sociolinguistic_path is None:
            self.sociolinguistic_path = self.model_dir / "transformer"


class SearchConfig(BaseSettings):
    """Search service configuration."""

    encoder_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    max_seq_length: int = 128
    embedding_dim: int = 768
    top_k_results: int = 20
    score_threshold: float = 0.7


class QdrantConfig(BaseSettings):
    """Vector database configuration."""

    host: str = "localhost"
    port: int = 6333
    collection_name: str = "toxic_vectors"
    vector_size: int = 768
    distance: Literal["Cosine", "Euclid", "Dot"] = "Cosine"
    on_disk_payload: bool = True


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    )
    rotation: str = "100 MB"
    retention: str = "1 week"


class UnifiedConfig(BaseSettings):
    """Main configuration container."""

    api: APIConfig = Field(default_factory=APIConfig)
    toxicity_models: ToxicityModelConfig = Field(default_factory=ToxicityModelConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_from_pyproject() -> dict:
    """Load config from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        return {}

    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    # Try to load from tool.unified-api or tool.toxic-search
    config = data.get("tool", {}).get("unified-api", {})
    if not config:
        config = data.get("tool", {}).get("toxic-search", {})

    return config


_config_instance: UnifiedConfig | None = None


def get_config() -> UnifiedConfig:
    """Get singleton config instance."""
    global _config_instance

    if _config_instance is None:
        toml_config = _load_from_pyproject()
        _config_instance = UnifiedConfig(
            api=APIConfig(**toml_config.get("api", {})),
            toxicity_models=ToxicityModelConfig(**toml_config.get("toxicity_models", {})),
            search=SearchConfig(**toml_config.get("search", {})),
            qdrant=QdrantConfig(**toml_config.get("qdrant", {})),
            logging=LoggingConfig(**toml_config.get("logging", {})),
        )

    return _config_instance