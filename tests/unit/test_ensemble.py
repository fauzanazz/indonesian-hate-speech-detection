import numpy as np
import pytest

from toxicity_detection.models.base import BaseModel
from toxicity_detection.models.ensemble import (
    EnsembleConfig,
    EnsembleService,
)


class DummyModel(BaseModel):
    def __init__(self, name: str, proba: float, trained: bool = True) -> None:
        super().__init__(name)
        self._proba = proba
        self.is_trained = trained

    def train(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def predict(self, texts):  # type: ignore[override]
        return np.array([int(self._proba >= 0.5)])

    def predict_proba(self, texts):  # type: ignore[override]
        return np.array([self._proba])

    def save(self, path):  # type: ignore[override]
        raise NotImplementedError

    def load(self, path):  # type: ignore[override]
        raise NotImplementedError


class DummyRegistry:
    def __init__(self, models):
        self.models = models


def test_ensemble_aggregates_with_normalized_weights():
    config = EnsembleConfig.from_mapping(
        {
            "weights": {"basic": 1.0, "contextual": 2.0, "sociolinguistic": 3.0},
            "threshold": 0.5,
        }
    )
    registry = DummyRegistry(
        {
            "basic": DummyModel("basic", 0.4),
            "contextual": DummyModel("contextual", 0.6),
            "sociolinguistic": DummyModel("socio", 0.9),
        }
    )

    service = EnsembleService(registry, config)
    result = service.predict("contoh teks")

    expected_score = (1 * 0.4 + 2 * 0.6 + 3 * 0.9) / 6
    assert pytest.approx(result.score, rel=1e-3) == expected_score
    assert pytest.approx(result.weights["basic"], rel=1e-3) == 1 / 6
    assert pytest.approx(result.weights["contextual"], rel=1e-3) == 2 / 6
    assert pytest.approx(result.weights["sociolinguistic"], rel=1e-3) == 3 / 6


def test_ensemble_short_circuits_when_confident():
    config = EnsembleConfig.from_mapping(
        {
            "weights": {"basic": 1.0, "contextual": 1.0, "sociolinguistic": 1.0},
            "threshold": 0.5,
            "cascade_thresholds": {"basic": 0.3},
        }
    )
    registry = DummyRegistry(
        {
            "basic": DummyModel("basic", 0.95),
            "contextual": DummyModel("contextual", 0.1),
            "sociolinguistic": DummyModel("socio", 0.1),
        }
    )

    service = EnsembleService(registry, config)
    result = service.predict("contoh teks")

    assert result.is_toxic is True
    assert result.per_model["basic"].used is True
    assert result.per_model["contextual"].used is False
    assert result.per_model["sociolinguistic"].used is False


def test_ensemble_requires_trained_model():
    config = EnsembleConfig.default()
    registry = DummyRegistry(
        {
            "basic": DummyModel("basic", 0.4, trained=False),
            "contextual": DummyModel("contextual", 0.6, trained=False),
            "sociolinguistic": DummyModel("socio", 0.9, trained=False),
        }
    )

    service = EnsembleService(registry, config)

    with pytest.raises(RuntimeError):
        service.predict("contoh teks")

