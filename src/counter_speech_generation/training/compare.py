"""Utility script for comparing experiments."""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from counter_speech_generation.training.experiment import ExperimentTracker
from counter_speech_generation.utils import setup_logging


def main() -> None:
    """CLI for comparing experiments."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Compare counter speech generation experiments")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default="experiments",
        help="Directory containing experiments",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Specific experiment IDs to compare (default: all)",
    )
    parser.add_argument(
        "--metric",
        default="rouge-1",
        help="Primary metric for comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file for comparison results",
    )
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ExperimentTracker(experiment_dir=args.experiment_dir)
    
    # Compare experiments
    logger.info("Comparing experiments...")
    comparison_df = tracker.compare_experiments(
        experiment_ids=args.experiments,
        metric=args.metric,
    )
    
    if comparison_df.empty:
        logger.warning("No experiments found to compare")
        return
    
    # Save to file if requested
    if args.output:
        comparison_df.to_csv(args.output, index=False)
        logger.info(f"Saved comparison to {args.output}")
    else:
        # Print to console
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPARISON")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)
    
    # Find best experiment
    metric_col = f"metric_{args.metric}"
    if metric_col in comparison_df.columns:
        best_idx = comparison_df[metric_col].idxmax()
        best_exp = comparison_df.loc[best_idx]
        logger.info(f"\nBest experiment by {args.metric}:")
        logger.info(f"  ID: {best_exp['experiment_id']}")
        logger.info(f"  {metric_col}: {best_exp[metric_col]:.4f}")
        logger.info(f"  Learning rate: {best_exp['learning_rate']}")
        logger.info(f"  Batch size: {best_exp['batch_size']}")


if __name__ == "__main__":
    main()

