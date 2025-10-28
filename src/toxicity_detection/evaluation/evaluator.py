import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt

from toxicity_detection.evaluation.metrics import (
    ClassificationMetrics,
    compute_metrics,
    compute_slice_metrics,
)
from toxicity_detection.models.base import BaseModel
from toxicity_detection.utils.logging import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    def __init__(self, model: BaseModel) -> None:
        self.model = model
        logger.info("Evaluator initialized", model=model.name)

    def evaluate(
        self,
        texts: List[str],
        labels: npt.NDArray[Any],
        measure_latency: bool = True,
    ) -> ClassificationMetrics:
        logger.info("Evaluating model", n_samples=len(texts), model=self.model.name)

        latencies = []
        if measure_latency:
            logger.info("Measuring inference latency...")
            for text in texts[:100]: 
                start = time.perf_counter()
                self.model.predict_proba([text])
                latencies.append(time.perf_counter() - start)

        logger.info("Getting predictions...")
        start = time.perf_counter()
        y_proba = self.model.predict_proba(texts)
        y_pred = self.model.predict(texts)
        inference_time = time.perf_counter() - start

        logger.info(
            "Prediction complete",
            total_time=f"{inference_time:.2f}s",
            throughput=f"{len(texts)/inference_time:.2f} samples/s",
        )

        metrics = compute_metrics(labels, y_pred, y_proba, latencies if measure_latency else None)
        logger.info(
            "Evaluation complete",
            accuracy=f"{metrics.accuracy:.4f}",
            f1=f"{metrics.f1:.4f}",
            roc_auc=f"{metrics.roc_auc:.4f}",
            ece=f"{metrics.ece:.4f}",
        )

        return metrics

    def evaluate_slices(
        self,
        texts: List[str],
        labels: npt.NDArray[Any],
        create_slices: bool = True,
    ) -> Dict[str, ClassificationMetrics]:
        logger.info("Evaluating slices", model=self.model.name)

        y_proba = self.model.predict_proba(texts)
        y_pred = self.model.predict(texts)

        slices: Dict[str, npt.NDArray[np.bool_]] = {}

        if create_slices:
            text_lengths = np.array([len(text.split()) for text in texts])
            slices["short_texts"] = text_lengths < np.percentile(text_lengths, 33)
            slices["medium_texts"] = (text_lengths >= np.percentile(text_lengths, 33)) & (
                text_lengths < np.percentile(text_lengths, 67)
            )
            slices["long_texts"] = text_lengths >= np.percentile(text_lengths, 67)
            slices["high_confidence"] = (y_proba > 0.8) | (y_proba < 0.2)
            slices["low_confidence"] = (y_proba >= 0.2) & (y_proba <= 0.8)
            slices["toxic_samples"] = labels == 1
            slices["non_toxic_samples"] = labels == 0

        slice_metrics = compute_slice_metrics(labels, y_pred, y_proba, slices)

        for slice_name, metrics in slice_metrics.items():
            logger.info(
                f"Slice: {slice_name}",
                f1=f"{metrics.f1:.4f}",
                n_samples=int(np.sum(slices[slice_name])),
            )

        return slice_metrics

    def compare_models(
        self,
        models: List[BaseModel],
        texts: List[str],
        labels: npt.NDArray[Any],
    ) -> Dict[str, ClassificationMetrics]:
        logger.info("Comparing models", n_models=len(models))

        results = {}
        for model in models:
            evaluator = ModelEvaluator(model)
            metrics = evaluator.evaluate(texts, labels, measure_latency=True)
            results[model.name] = metrics

        logger.info("Model comparison:")
        for name, metrics in results.items():
            logger.info(
                f"  {name}",
                f1=f"{metrics.f1:.4f}",
                latency=f"{metrics.avg_latency_ms:.2f}ms" if metrics.avg_latency_ms else "N/A",
            )

        return results

    def save_results(self, metrics: ClassificationMetrics, output_path: Path) -> None:
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info("Results saved", path=str(output_path))
