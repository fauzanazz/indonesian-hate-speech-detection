import numpy as np
import pytest
from toxicity_detection.utils.metrics import calculate_ece, get_calibration_curve


def test_calculate_ece_perfect_calibration() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0])
    ece = calculate_ece(y_true, y_proba, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_calculate_ece_poor_calibration() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.9, 0.9, 0.9, 0.9])
    ece = calculate_ece(y_true, y_proba, n_bins=2)
    assert ece > 0.3 


def test_get_calibration_curve() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
    mean_probs, frac_pos = get_calibration_curve(y_true, y_proba, n_bins=3)
    assert len(mean_probs) <= 3
    assert len(frac_pos) <= 3
    assert all(0 <= p <= 1 for p in mean_probs)
    assert all(0 <= f <= 1 for f in frac_pos)
