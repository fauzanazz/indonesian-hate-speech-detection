"""Evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    labels = sorted({*y_true, *y_pred})
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    per_class = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }

    return metrics


def confusion_matrix_dataframe(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def save_confusion_matrix_plot(
    matrix: pd.DataFrame,
    *,
    title: str = "Confusion Matrix",
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("Blues")
    im = ax.imshow(matrix.values, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=matrix.columns,
        yticklabels=matrix.index,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.values.max() / 2.0 if matrix.values.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax.text(
                j,
                i,
                format(value, "d"),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

