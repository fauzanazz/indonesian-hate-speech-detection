from typing import Any
import numpy as np
import numpy.typing as npt


def calculate_ece(
    y_true: npt.NDArray[Any],
    y_proba: npt.NDArray[Any],
    n_bins: int = 10,
) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece_value = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_proba[in_bin])
            ece_value += (bin_size / len(y_true)) * np.abs(accuracy_in_bin - avg_confidence_in_bin)

    return float(ece_value)


def get_calibration_curve(
    y_true: npt.NDArray[Any],
    y_proba: npt.NDArray[Any],
    n_bins: int = 10,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mean_predicted_probs = []
    fraction_positives = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            mean_predicted_probs.append(np.mean(y_proba[in_bin]))
            fraction_positives.append(np.mean(y_true[in_bin]))

    return np.array(mean_predicted_probs), np.array(fraction_positives)
