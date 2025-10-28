from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix
from toxicity_detection.models.base import BaseModel
from toxicity_detection.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class ErrorSample:
    text: str
    true_label: int
    pred_label: int
    pred_proba: float
    error_type: str

    def __repr__(self) -> str:
        return (
            f"ErrorSample(\n"
            f"  error_type='{self.error_type}',\n"
            f"  text='{self.text[:100]}...',\n"
            f"  true_label={self.true_label},\n"
            f"  pred_label={self.pred_label},\n"
            f"  pred_proba={self.pred_proba:.4f}\n"
            f")"
        )


class ErrorAnalyzer:
    def __init__(self, model: BaseModel) -> None:
        self.model = model
        logger.info("Error analyzer initialized", model=model.name)

    def analyze(
        self,
        texts: List[str],
        labels: npt.NDArray[Any],
        top_k: int = 50,
    ) -> Dict[str, Any]:
        logger.info("Performing error analysis", n_samples=len(texts), model=self.model.name)
        y_proba = self.model.predict_proba(texts)
        y_pred = self.model.predict(texts)
        cm = confusion_matrix(labels, y_pred)
        tn, fp, fn, tp = cm.ravel()

        logger.info(
            "Confusion matrix",
            TP=int(tp),
            FP=int(fp),
            FN=int(fn),
            TN=int(tn),
        )

        errors = self._extract_errors(texts, labels, y_pred, y_proba, top_k)
        error_patterns = self._analyze_patterns(errors)

        results = {
            "confusion_matrix": cm,
            "n_errors": len(errors),
            "n_false_positives": int(fp),
            "n_false_negatives": int(fn),
            "errors": errors,
            "error_patterns": error_patterns,
        }

        logger.info(
            "Error analysis complete",
            n_errors=len(errors),
            n_fp=int(fp),
            n_fn=int(fn),
        )

        return results

    def _extract_errors(
        self,
        texts: List[str],
        y_true: npt.NDArray[Any],
        y_pred: npt.NDArray[Any],
        y_proba: npt.NDArray[Any],
        top_k: int,
    ) -> List[ErrorSample]:
        errors = []

        for text, true_label, pred_label, pred_proba in zip(texts, y_true, y_pred, y_proba):
            if true_label != pred_label:
                error_type = "FP" if pred_label == 1 else "FN"
                errors.append(
                    ErrorSample(
                        text=text,
                        true_label=int(true_label),
                        pred_label=int(pred_label),
                        pred_proba=float(pred_proba),
                        error_type=error_type,
                    )
                )

        errors.sort(key=lambda e: abs(e.pred_proba - 0.5), reverse=True)
        return errors[:top_k]

    def _analyze_patterns(self, errors: List[ErrorSample]) -> Dict[str, Any]:
        if not errors:
            return {}

        false_positives = [e for e in errors if e.error_type == "FP"]
        false_negatives = [e for e in errors if e.error_type == "FN"]
        fp_lengths = [len(e.text.split()) for e in false_positives]
        fn_lengths = [len(e.text.split()) for e in false_negatives]
        high_confidence_errors = [
            e for e in errors
            if (e.error_type == "FP" and e.pred_proba > 0.8)
            or (e.error_type == "FN" and e.pred_proba < 0.2)
        ]

        patterns = {
            "n_false_positives": len(false_positives),
            "n_false_negatives": len(false_negatives),
            "fp_avg_length": float(np.mean(fp_lengths)) if fp_lengths else 0,
            "fn_avg_length": float(np.mean(fn_lengths)) if fn_lengths else 0,
            "n_high_confidence_errors": len(high_confidence_errors),
            "high_confidence_error_rate": len(high_confidence_errors) / len(errors) if errors else 0,
        }

        return patterns

    def get_top_errors(
        self,
        texts: List[str],
        labels: npt.NDArray[Any],
        error_type: str = "both",
        top_k: int = 10,
    ) -> List[ErrorSample]:

        y_proba = self.model.predict_proba(texts)
        y_pred = self.model.predict(texts)
        errors = self._extract_errors(texts, labels, y_pred, y_proba, top_k=len(texts))

        if error_type == "FP":
            errors = [e for e in errors if e.error_type == "FP"]
        elif error_type == "FN":
            errors = [e for e in errors if e.error_type == "FN"]

        return errors[:top_k]

    def create_error_report(
        self,
        texts: List[str],
        labels: npt.NDArray[Any],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        y_proba = self.model.predict_proba(texts)
        y_pred = self.model.predict(texts)
        errors = self._extract_errors(texts, labels, y_pred, y_proba, top_k=len(texts))
        df = pd.DataFrame([
            {
                "text": e.text,
                "true_label": e.true_label,
                "pred_label": e.pred_label,
                "pred_proba": e.pred_proba,
                "error_type": e.error_type,
                "confidence": abs(e.pred_proba - 0.5),
                "text_length": len(e.text.split()),
            }
            for e in errors
        ])

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info("Error report saved", path=output_path)

        return df
