from typing import Any, Dict, List
import numpy as np
import numpy.typing as npt
from toxicity_detection.models.base import BaseModel
from toxicity_detection.utils.logging import setup_logger

logger = setup_logger(__name__)


class BiasAuditor:
    def __init__(self, model: BaseModel) -> None:
        self.model = model
        logger.info("Bias auditor initialized", model=model.name)

    def audit_slices(
        self,
        texts: List[str],
        labels: npt.NDArray[Any],
        slice_fn: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        logger.info("Auditing slices", n_slices=len(slice_fn))

        y_proba = self.model.predict_proba(texts)
        y_pred = self.model.predict(texts)

        results = {}

        for slice_name, slice_func in slice_fn.items():
            mask = np.array([slice_func(text) for text in texts])

            if np.sum(mask) == 0:
                logger.warning(f"Slice '{slice_name}' is empty, skipping")
                continue

            slice_labels = labels[mask]
            slice_pred = y_pred[mask]
            slice_proba = y_proba[mask]
            positive_rate = float(np.mean(slice_pred))
            tp = np.sum((slice_labels == 1) & (slice_pred == 1))
            fn = np.sum((slice_labels == 1) & (slice_pred == 0))
            fp = np.sum((slice_labels == 0) & (slice_pred == 1))
            tn = np.sum((slice_labels == 0) & (slice_pred == 0))
            tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
            accuracy = float(np.mean(slice_labels == slice_pred))
            avg_proba = float(np.mean(slice_proba))

            results[slice_name] = {
                "n_samples": int(np.sum(mask)),
                "positive_rate": positive_rate,
                "tpr": tpr,
                "fpr": fpr,
                "accuracy": accuracy,
                "avg_predicted_proba": avg_proba,
            }

            logger.info(
                f"Slice: {slice_name}",
                n_samples=int(np.sum(mask)),
                positive_rate=f"{positive_rate:.3f}",
                tpr=f"{tpr:.3f}",
                fpr=f"{fpr:.3f}",
            )

        return results

    def compute_fairness_gaps(
        self,
        slice_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        if len(slice_metrics) < 2:
            logger.warning("Need at least 2 slices to compute gaps")
            return {}

        positive_rates = [m["positive_rate"] for m in slice_metrics.values()]
        tprs = [m["tpr"] for m in slice_metrics.values()]
        fprs = [m["fpr"] for m in slice_metrics.values()]
        accuracies = [m["accuracy"] for m in slice_metrics.values()]

        gaps = {
            "max_positive_rate_gap": float(np.max(positive_rates) - np.min(positive_rates)),
            "max_tpr_gap": float(np.max(tprs) - np.min(tprs)),
            "max_fpr_gap": float(np.max(fprs) - np.min(fprs)),
            "max_accuracy_gap": float(np.max(accuracies) - np.min(accuracies)),
        }

        logger.info(
            "Fairness gaps computed",
            max_accuracy_gap=f"{gaps['max_accuracy_gap']:.3f}",
            max_tpr_gap=f"{gaps['max_tpr_gap']:.3f}",
        )

        return gaps

    def generate_fairness_report(
        self,
        texts: List[str],
        labels: npt.NDArray[Any],
    ) -> Dict[str, Any]:
        logger.info("Generating fairness report", model=self.model.name)

        slice_functions = {
            "very_short": lambda t: len(t.split()) < 20,
            "short": lambda t: 20 <= len(t.split()) < 40,
            "medium": lambda t: 40 <= len(t.split()) < 80,
            "long": lambda t: len(t.split()) >= 80,
        }

        slice_metrics = self.audit_slices(texts, labels, slice_functions)
        fairness_gaps = self.compute_fairness_gaps(slice_metrics)
        max_gap = fairness_gaps.get("max_accuracy_gap", 0)
        fairness_status = "PASS" if max_gap < 0.1 else "WARNING" if max_gap < 0.2 else "FAIL"

        report = {
            "model_name": self.model.name,
            "slice_metrics": slice_metrics,
            "fairness_gaps": fairness_gaps,
            "fairness_status": fairness_status,
            "recommendation": self._get_recommendation(fairness_status, fairness_gaps),
        }

        logger.info(
            "Fairness report generated",
            status=fairness_status,
            max_gap=f"{max_gap:.3f}",
        )

        return report

    def _get_recommendation(
        self,
        status: str,
        gaps: Dict[str, float],
    ) -> str:
        if status == "PASS":
            return "Model demonstrates acceptable fairness across slices."
        elif status == "WARNING":
            max_gap_metric = max(gaps.items(), key=lambda x: x[1])
            return (
                f"Model shows moderate fairness disparities. "
                f"Consider investigating {max_gap_metric[0]} (gap: {max_gap_metric[1]:.3f}). "
                f"Review training data balance and consider fairness constraints."
            )
        else:
            return (
                "Model exhibits significant fairness issues. "
                "Recommend: (1) Audit training data for bias, "
                "(2) Apply fairness-aware training techniques, "
                "(3) Consider post-processing calibration per slice."
            )
