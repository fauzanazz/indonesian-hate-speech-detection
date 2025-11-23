"""Configuration management for counter speech generation."""

from pathlib import Path
from typing import Literal

import tomli
from pydantic import Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    """Model configuration for IndoT5."""

    base_model: str = "Wikidepia/IndoT5-base"
    max_length: int = 512  # Can use full length with 8GB GPU
    max_target_length: int = 128  # Can use full length with 8GB GPU
    num_beams: int = 4
    length_penalty: float = 1.0
    repetition_penalty: float = 1.2
    temperature: float = 1.0
    do_sample: bool = False


class TrainingConfig(BaseSettings):
    """Training hyperparameters."""

    batch_size: int = 1  
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    evaluation_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    fp16: bool = True
    dataloader_num_workers: int = 2  # Can use workers with 8GB GPU
    gradient_checkpointing: bool = False  # Can disable with 8GB GPU for speed
    dataloader_pin_memory: bool = True  # Can enable with 8GB GPU
    use_cpu: bool = False  # Fallback to CPU if GPU OOM


class DataConfig(BaseSettings):
    """Data loading configuration."""

    text_column: str = "text"
    label_column: str = "labels"
    counter_column: str = "counter"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_state: int = 42


class APIConfig(BaseSettings):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8001
    enable_cors: bool = True
    max_length: int = 256
    max_target_length: int = 128


class UIConfig(BaseSettings):
    """UI configuration."""

    port: int = 8502
    max_examples: int = 100


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
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_from_pyproject() -> dict:
    """Load counter-speech-generation config from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        return {}

    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    return data.get("tool", {}).get("counter-speech-generation", {})


_config_instance: Config | None = None


def get_config() -> Config:
    """Get singleton config instance loaded from pyproject.toml."""
    global _config_instance

    if _config_instance is None:
        toml_config = _load_from_pyproject()
        _config_instance = Config(
            model=ModelConfig(**toml_config.get("model", {})),
            training=TrainingConfig(**toml_config.get("training", {})),
            data=DataConfig(**toml_config.get("data", {})),
            api=APIConfig(**toml_config.get("api", {})),
            ui=UIConfig(**toml_config.get("ui", {})),
            logging=LoggingConfig(**toml_config.get("logging", {})),
        )

    return _config_instance

