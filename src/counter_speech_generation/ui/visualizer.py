"""Visualization utilities for counter speech generation results."""

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger


def plot_length_distribution(
    texts: list[str],
    counters: list[str],
    save_path: str | None = None,
) -> None:
    """Plot distribution of text and counter speech lengths.
    
    Args:
        texts: List of input texts
        counters: List of counter speeches
        save_path: Optional path to save figure
    """
    text_lengths = [len(t.split()) for t in texts]
    counter_lengths = [len(c.split()) for c in counters]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(text_lengths, bins=50, alpha=0.7, color="skyblue")
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Input Text Length Distribution")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(counter_lengths, bins=50, alpha=0.7, color="lightcoral")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Counter Speech Length Distribution")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(
    metrics: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> None:
    """Plot comparison of evaluation metrics.
    
    Args:
        metrics: Dictionary mapping model names to metric dictionaries
        save_path: Optional path to save figure
    """
    df = pd.DataFrame(metrics).T
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # BLEU scores
    bleu_cols = [c for c in df.columns if c.startswith("bleu")]
    if bleu_cols:
        df[bleu_cols].plot(kind="bar", ax=axes[0], rot=0)
        axes[0].set_title("BLEU Scores")
        axes[0].set_ylabel("Score")
        axes[0].legend(title="BLEU-n")
        axes[0].grid(True, alpha=0.3)
    
    # ROUGE scores
    rouge_cols = [c for c in df.columns if c.startswith("rouge")]
    if rouge_cols:
        df[rouge_cols].plot(kind="bar", ax=axes[1], rot=0)
        axes[1].set_title("ROUGE Scores")
        axes[1].set_ylabel("Score")
        axes[1].legend(title="ROUGE")
        axes[1].grid(True, alpha=0.3)
    
    # METEOR if available
    if "meteor" in df.columns:
        df["meteor"].plot(kind="bar", ax=axes[2], rot=0, color="green")
        axes[2].set_title("METEOR Score")
        axes[2].set_ylabel("Score")
        axes[2].grid(True, alpha=0.3)
    
    # Overall comparison
    all_metrics = [c for c in df.columns if c in ["bleu-1", "rouge-1", "meteor"]]
    if all_metrics:
        df[all_metrics].plot(kind="bar", ax=axes[3], rot=0)
        axes[3].set_title("Overall Metrics Comparison")
        axes[3].set_ylabel("Score")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_examples_table(
    texts: list[str],
    references: list[str],
    predictions: list[str],
    max_examples: int = 10,
) -> pd.DataFrame:
    """Create a table of example generations.
    
    Args:
        texts: Input texts
        references: Reference counter speeches
        predictions: Generated counter speeches
        max_examples: Maximum number of examples to show
    
    Returns:
        DataFrame with examples
    """
    examples = []
    for i, (text, ref, pred) in enumerate(zip(texts, references, predictions)):
        if i >= max_examples:
            break
        
        examples.append({
            "Index": i + 1,
            "Input Text": text[:100] + "..." if len(text) > 100 else text,
            "Reference": ref[:100] + "..." if len(ref) > 100 else ref,
            "Generated": pred[:100] + "..." if len(pred) > 100 else pred,
        })
    
    return pd.DataFrame(examples)

