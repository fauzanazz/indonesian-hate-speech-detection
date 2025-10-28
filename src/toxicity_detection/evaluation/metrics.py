import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from toxicity_detection.utils.metrics import calculate_ece


@dataclass
class ClassificationMetrics:
    """
    BEAM Metrics:
    - accuracy: Overall accuracy
    - precision: Precision (TP / (TP + FP))
    - recall: Recall/Sensitivity (TP / (TP + FN))
    - f1: F1 score (harmonic mean of precision and recall)
    - roc_auc: Area under ROC curve
    - pr_auc: Area under Precision-Recall curve
    - ece: Expected Calibration Error
    - confusion_matrix: Confusion matrix
    - fpr: False positive rate
    - tpr: True positive rate
    - thresholds: Decision thresholds
    - avg_latency: Average inference latency (ms)
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    ece: float
    confusion_matrix: npt.NDArray[Any]
    fpr: npt.NDArray[Any]
    tpr: npt.NDArray[Any]
    thresholds: npt.NDArray[Any]
    avg_latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "roc_auc": float(self.roc_auc),
            "pr_auc": float(self.pr_auc),
            "ece": float(self.ece),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "avg_latency_ms": self.avg_latency_ms,
        }

    def __repr__(self) -> str:
        return (
            f"ClassificationMetrics(\n"
            f"  accuracy={self.accuracy:.4f},\n"
            f"  precision={self.precision:.4f},\n"
            f"  recall={self.recall:.4f},\n"
            f"  f1={self.f1:.4f},\n"
            f"  roc_auc={self.roc_auc:.4f},\n"
            f"  pr_auc={self.pr_auc:.4f},\n"
            f"  ece={self.ece:.4f},\n"
            f"  avg_latency_ms={self.avg_latency_ms:.2f if self.avg_latency_ms else 'N/A'}\n"
            f")"
        )


def compute_metrics(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    y_proba: npt.NDArray[Any],
    latencies: Optional[List[float]] = None,
) -> ClassificationMetrics:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recalls, precisions)
    ece = calculate_ece(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    avg_latency_ms = None
    if latencies:
        avg_latency_ms = np.mean(latencies) * 1000 

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        ece=ece,
        confusion_matrix=cm,
        fpr=fpr,
        tpr=tpr,
        thresholds=roc_thresholds,
        avg_latency_ms=avg_latency_ms,
    )


def compute_slice_metrics(
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[Any],
    y_proba: npt.NDArray[Any],
    slice_indices: Dict[str, npt.NDArray[np.bool_]],
) -> Dict[str, ClassificationMetrics]:
    slice_metrics = {}

    for slice_name, mask in slice_indices.items():
        if np.sum(mask) == 0:
            continue

        metrics = compute_metrics(
            y_true[mask],
            y_pred[mask],
            y_proba[mask],
        )
        slice_metrics[slice_name] = metrics

    return slice_metrics


def measure_inference_latency(
    model: Any,
    texts: List[str],
    n_runs: int = 100,
) -> tuple[float, float]:
    latencies = []

    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict_proba([texts[0]])  # Single sample
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    return np.mean(latencies), np.std(latencies)
