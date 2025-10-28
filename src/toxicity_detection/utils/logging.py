import sys
import uuid
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> logger:
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    return logger


def generate_trace_id() -> str:
    return str(uuid.uuid4())


class LogContext:
    def __init__(self, **kwargs: str) -> None:
        self.context = kwargs

    def __enter__(self) -> "LogContext":
        logger.configure(extra=self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        logger.configure(extra={})
