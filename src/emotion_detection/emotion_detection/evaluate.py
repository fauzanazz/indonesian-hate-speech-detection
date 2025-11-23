"""Evaluation helpers for trained models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

from .models.base import BaseEmotionModel
from .utils.metrics import (
    classification_metrics,
    confusion_matrix_dataframe,
    save_confusion_matrix_plot,
)


def evaluate_model(
    model: BaseEmotionModel,
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    output_dir: Path | None = None,
) -> Dict[str, object]:
    predictions = model.predict(texts)
    metrics = classification_metrics(labels, predictions)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "classification_report.txt"
        report_path.write_text(metrics["classification_report"], encoding="utf-8")

        matrix = confusion_matrix_dataframe(labels, predictions, metrics["labels"])
        matrix.to_csv(output_dir / "confusion_matrix.csv", index=True)
        save_confusion_matrix_plot(matrix, output_path=output_dir / "confusion_matrix.png")
        metrics["confusion_matrix"] = matrix.values.tolist()

    return metrics


