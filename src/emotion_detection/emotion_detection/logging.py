"""Rich logging helpers for the emotion detection package."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from rich.logging import RichHandler


def configure_logging(level: str = "INFO") -> None:
    """Configure global logging with Rich if not already initialised."""

    root = logging.getLogger()
    if root.handlers:
        return

    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        omit_repeated_times=False,
    )

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler],
    )


def get_logger(name: str = "emotion") -> logging.Logger:
    """Return a module-level logger using the configured root handlers."""

    return logging.getLogger(name)


@contextmanager
def log_section(message: str, *, level: int = logging.INFO, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Log a start/finish pair with elapsed time for a block of work."""

    active_logger = logger or get_logger()
    start = time.perf_counter()
    active_logger.log(level, f">> {message}")
    try:
        yield
    except Exception:
        active_logger.exception("xx %s failed", message)
        raise
    else:
        elapsed = time.perf_counter() - start
        active_logger.log(level, f"ok {message} (took {elapsed:.2f}s)")

