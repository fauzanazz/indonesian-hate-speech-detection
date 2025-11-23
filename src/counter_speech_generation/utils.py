"""Shared utilities for logging and metrics."""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from counter_speech_generation.config import get_config


def setup_logging() -> None:
    """Configure loguru logger from config."""
    config = get_config().logging

    logger.remove()
    logger.add(
        sys.stderr,
        format=config.format,
        level=config.level,
        colorize=True,
    )

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger.add(
        logs_dir / "counter_speech_generation.log",
        format=config.format,
        level=config.level,
        rotation=config.rotation,
        retention=config.retention,
        compression="zip",
    )


def log_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    """Log metrics in a structured format."""
    if prefix:
        logger.info(f"{prefix}:")

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


def clear_gpu_cache() -> None:
    """Clear GPU cache to free memory."""
    import torch
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")

