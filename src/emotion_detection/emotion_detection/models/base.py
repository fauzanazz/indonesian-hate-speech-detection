"""Base interfaces for emotion detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List


class BaseEmotionModel(ABC):
    """Common protocol for all emotion detection models."""

    @abstractmethod
    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> None:
        """Train the model on the provided corpus."""

    @abstractmethod
    def predict(self, texts: Iterable[str]) -> List[str]:
        """Return predicted labels for each text input."""

    @abstractmethod
    def predict_proba(self, texts: Iterable[str]) -> List[Dict[str, float]]:
        """Return per-label probability distributions."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artefacts to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseEmotionModel":
        """Load model artefacts from ``path``."""

