"""Configuration management from pyproject.toml."""

from pathlib import Path
from typing import Literal

import tomli
from pydantic import Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    """Model configuration."""

    base_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    max_seq_length: int = 128
    embedding_dim: int = 768
    distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine"


class TrainingConfig(BaseSettings):
    """Training hyperparameters."""

    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    margin: float = 0.5
    evaluation_steps: int = 500
    save_steps: int = 1000
    fp16: bool = True


class MiningConfig(BaseSettings):
    """Hard negative mining configuration."""

    strategy: Literal["bm25", "semantic", "hybrid"] = "bm25"
    min_lexical_overlap: float = 0.3
    negatives_per_anchor: int = 5
    sampling_temperature: float = 0.7


class QdrantConfig(BaseSettings):
    """Vector database configuration."""

    host: str = "localhost"
    port: int = 6333
    collection_name: str = "toxic_vectors"
    vector_size: int = 768
    distance: Literal["Cosine", "Euclid", "Dot"] = "Cosine"
    on_disk_payload: bool = True


class APIConfig(BaseSettings):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    top_k_results: int = 20
    score_threshold: float = 0.7
    enable_cors: bool = True


class UIConfig(BaseSettings):
    """UI configuration."""

    port: int = 8501
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    vis_sample_size: int = 10000


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


class Config(BaseSettings):
    """Main configuration container."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mining: MiningConfig = Field(default_factory=MiningConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_from_pyproject() -> dict:
    """Load toxic-search config from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    
    if not pyproject_path.exists():
        return {}
    
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
    
    return data.get("tool", {}).get("toxic-search", {})


_config_instance: Config | None = None


def get_config() -> Config:
    """Get singleton config instance loaded from pyproject.toml."""
    global _config_instance
    
    if _config_instance is None:
        toml_config = _load_from_pyproject()
        _config_instance = Config(
            model=ModelConfig(**toml_config.get("model", {})),
            training=TrainingConfig(**toml_config.get("training", {})),
            mining=MiningConfig(**toml_config.get("mining", {})),
            qdrant=QdrantConfig(**toml_config.get("qdrant", {})),
            api=APIConfig(**toml_config.get("api", {})),
            ui=UIConfig(**toml_config.get("ui", {})),
            logging=LoggingConfig(**toml_config.get("logging", {})),
        )
    
    return _config_instance