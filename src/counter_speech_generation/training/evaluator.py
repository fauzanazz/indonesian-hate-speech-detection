"""Evaluation metrics for counter speech generation."""

from typing import Any

import numpy as np
from loguru import logger
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from counter_speech_generation.config import get_config
from counter_speech_generation.training.experiment import get_tracker
from counter_speech_generation.utils import log_metrics


def compute_bleu(references: list[str], predictions: list[str]) -> dict[str, float]:
    """Compute BLEU scores for generated counter speeches.
    
    Args:
        references: List of reference counter speeches
        predictions: List of generated counter speeches
    
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    smoothing = SmoothingFunction().method1
    
    bleu_scores = {f"bleu-{i}": [] for i in range(1, 5)}
    
    for ref, pred in zip(references, predictions):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        
        for n in range(1, 5):
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=tuple([1.0 / n] * n + [0.0] * (4 - n)),
                    smoothing_function=smoothing,
                )
                bleu_scores[f"bleu-{n}"].append(score)
            except Exception as e:
                logger.warning(f"BLEU-{n} computation failed: {e}")
                bleu_scores[f"bleu-{n}"].append(0.0)
    
    # Compute means
    return {k: np.mean(v) for k, v in bleu_scores.items()}


def compute_rouge(references: list[str], predictions: list[str]) -> dict[str, float]:
    """Compute ROUGE scores for generated counter speeches.
    
    Args:
        references: List of reference counter speeches
        predictions: List of generated counter speeches
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores (F1)
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougel"], use_stemmer=True)
    
    rouge_scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}
    
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge_scores["rouge-1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge-2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rouge-l"].append(scores["rougel"].fmeasure)
    
    # Compute means
    return {k: np.mean(v) for k, v in rouge_scores.items()}


def compute_meteor(references: list[str], predictions: list[str]) -> float:
    """Compute METEOR score (requires nltk).
    
    Args:
        references: List of reference counter speeches
        predictions: List of generated counter speeches
    
    Returns:
        Average METEOR score
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        
        scores = []
        for ref, pred in zip(references, predictions):
            try:
                score = meteor_score([ref.split()], pred.split())
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    except ImportError:
        logger.warning("METEOR requires nltk. Install with: pip install nltk")
        return 0.0


def evaluate_generation(
    references: list[str],
    predictions: list[str],
    compute_meteor_score: bool = False,
    texts: list[str] | None = None,
    save_results: bool = True,
) -> dict[str, float]:
    """Evaluate generated counter speeches using multiple metrics.
    
    Args:
        references: List of reference counter speeches
        predictions: List of generated counter speeches
        compute_meteor_score: Whether to compute METEOR (requires nltk)
        texts: Optional list of input texts (for saving predictions)
        save_results: Whether to save results with experiment tracker
    
    Returns:
        Dictionary of evaluation metrics
    """
    if len(references) != len(predictions):
        raise ValueError(f"References ({len(references)}) and predictions ({len(predictions)}) must have same length")
    
    logger.info(f"Evaluating {len(references)} samples...")
    
    metrics: dict[str, float] = {}
    
    # BLEU scores
    bleu_scores = compute_bleu(references, predictions)
    metrics.update(bleu_scores)
    
    # ROUGE scores
    rouge_scores = compute_rouge(references, predictions)
    metrics.update(rouge_scores)
    
    # METEOR (optional)
    if compute_meteor_score:
        meteor = compute_meteor(references, predictions)
        metrics["meteor"] = meteor
    
    log_metrics(metrics, prefix="Generation Metrics")
    
    # Save results with experiment tracker if available
    if save_results:
        try:
            tracker = get_tracker()
            if tracker.current_experiment:
                tracker.save_evaluation_results(
                    metrics=metrics,
                    predictions=predictions,
                    references=references,
                    texts=texts,
                )
        except Exception as e:
            logger.warning(f"Could not save evaluation results to experiment tracker: {e}")
    
    return metrics

