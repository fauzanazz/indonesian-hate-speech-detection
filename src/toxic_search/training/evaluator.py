"""Evaluation metrics for retrieval quality."""

import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score

from toxic_search.utils import log_metrics


def evaluate_retrieval(
    query_labels: list[int],
    retrieved_labels: list[list[int]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate retrieval quality with ranking metrics.
    
    Computes Precision@K, Recall@K, and Mean Average Precision (MAP).
    """
    k_values = k_values or [1, 5, 10, 20]
    
    metrics = {}
    all_precisions = []
    
    for query_label, retrieved in zip(query_labels, retrieved_labels):
        # Binary relevance: 1 if same label, 0 otherwise
        relevance = np.array([1 if label == query_label else 0 for label in retrieved])
        
        # Precision@K for different K values
        for k in k_values:
            k_relevance = relevance[:k]
            precision_at_k = k_relevance.sum() / k if k > 0 else 0.0
            
            key = f"precision@{k}"
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(precision_at_k)
        
        # Recall@K
        total_relevant = sum(1 for label in retrieved_labels if query_label in label)
        for k in k_values:
            k_relevance = relevance[:k]
            recall_at_k = k_relevance.sum() / max(total_relevant, 1)
            
            key = f"recall@{k}"
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(recall_at_k)
        
        # Average Precision (for MAP calculation)
        if relevance.sum() > 0:
            ap = average_precision_score(relevance, np.arange(len(relevance), 0, -1))
            all_precisions.append(ap)
    
    # Compute mean metrics
    final_metrics = {}
    for key, values in metrics.items():
        final_metrics[key] = np.mean(values)
    
    # Mean Average Precision
    final_metrics["MAP"] = np.mean(all_precisions) if all_precisions else 0.0
    
    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for query_label, retrieved in zip(query_labels, retrieved_labels):
        for rank, label in enumerate(retrieved, start=1):
            if label == query_label:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    final_metrics["MRR"] = np.mean(reciprocal_ranks)
    
    log_metrics(final_metrics, prefix="Retrieval Metrics")
    
    return final_metrics


def evaluate_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    """Compute standard classification metrics for comparison.
    
    Used when treating retrieval as a classification problem.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    log_metrics(metrics, prefix="Classification Metrics")
    
    return metrics