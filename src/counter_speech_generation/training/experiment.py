"""Experiment tracking and parameter comparison utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from counter_speech_generation.config import get_config


class ExperimentTracker:
    """Track experiments with hyperparameters and metrics."""

    def __init__(self, experiment_dir: str | Path = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment: dict[str, Any] | None = None

    def start_experiment(
        self,
        experiment_name: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> str:
        """Start a new experiment and return experiment ID.
        
        Args:
            experiment_name: Optional name for the experiment
            config_overrides: Dictionary of config values to override
        
        Returns:
            Experiment ID (timestamp-based)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = experiment_name or f"exp_{timestamp}"
        
        # Get current config
        config = get_config()
        
        # Build experiment metadata
        experiment_data = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "config": {
                "model": config.model.dict(),
                "training": config.training.dict(),
                "data": config.data.dict(),
            },
        }
        
        # Apply overrides
        if config_overrides:
            experiment_data["config_overrides"] = config_overrides
            # Merge overrides into config
            for key, value in config_overrides.items():
                if "." in key:
                    # Nested config (e.g., "training.learning_rate")
                    parts = key.split(".")
                    if len(parts) == 2:
                        section, param = parts
                        if hasattr(config, section):
                            section_obj = getattr(config, section)
                            if hasattr(section_obj, param):
                                setattr(section_obj, param, value)
                                experiment_data["config"][section][param] = value
        
        self.current_experiment = experiment_data
        
        # Create experiment directory
        exp_dir = self.experiment_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        with open(exp_dir / "config.json", "w") as f:
            json.dump(experiment_data, f, indent=2)
        
        logger.info(f"Started experiment: {exp_id}")
        logger.info(f"Experiment directory: {exp_dir}")
        
        return exp_id

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        phase: str = "train",
    ) -> None:
        """Log metrics for current experiment.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step/epoch number
            phase: Phase name (train, val, test)
        """
        if self.current_experiment is None:
            logger.warning("No active experiment. Call start_experiment() first.")
            return
        
        exp_id = self.current_experiment["experiment_id"]
        exp_dir = self.experiment_dir / exp_id
        
        # Load existing metrics or create new
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        
        # Add new metrics
        metric_key = f"{phase}_{step}" if step is not None else phase
        all_metrics[metric_key] = {
            "step": step,
            "phase": phase,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info(f"Logged {len(metrics)} metrics for {phase} (step: {step})")

    def log_training_config(self, training_config: dict[str, Any]) -> None:
        """Log training configuration.
        
        Args:
            training_config: Dictionary of training hyperparameters
        """
        if self.current_experiment is None:
            logger.warning("No active experiment. Call start_experiment() first.")
            return
        
        exp_id = self.current_experiment["experiment_id"]
        exp_dir = self.experiment_dir / exp_id
        
        # Save training config
        with open(exp_dir / "training_config.json", "w") as f:
            json.dump(training_config, f, indent=2)

    def save_evaluation_results(
        self,
        metrics: dict[str, float],
        predictions: list[str] | None = None,
        references: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> None:
        """Save evaluation results with configuration.
        
        Args:
            metrics: Evaluation metrics
            predictions: Generated counter speeches
            references: Reference counter speeches
            texts: Input texts
        """
        if self.current_experiment is None:
            logger.warning("No active experiment. Call start_experiment() first.")
            return
        
        exp_id = self.current_experiment["experiment_id"]
        exp_dir = self.experiment_dir / exp_id
        
        # Save metrics with config
        results = {
            "experiment_id": exp_id,
            "config": self.current_experiment["config"],
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(exp_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save predictions if provided
        if predictions and references and texts:
            df = pd.DataFrame({
                "text": texts,
                "reference": references,
                "prediction": predictions,
            })
            df.to_csv(exp_dir / "predictions.csv", index=False)
            logger.info(f"Saved {len(df)} predictions to {exp_dir / 'predictions.csv'}")
        
        logger.info(f"Saved evaluation results to {exp_dir / 'evaluation_results.json'}")

    def compare_experiments(
        self,
        experiment_ids: list[str] | None = None,
        metric: str = "rouge-1",
    ) -> pd.DataFrame:
        """Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare. If None, compares all.
            metric: Metric to use for comparison
        
        Returns:
            DataFrame with comparison results
        """
        if experiment_ids is None:
            # Get all experiments
            experiment_ids = [d.name for d in self.experiment_dir.iterdir() if d.is_dir()]
        
        comparisons = []
        
        for exp_id in experiment_ids:
            exp_dir = self.experiment_dir / exp_id
            
            # Load config
            config_file = exp_dir / "config.json"
            if not config_file.exists():
                continue
            
            with open(config_file) as f:
                exp_data = json.load(f)
            
            # Load evaluation results
            eval_file = exp_dir / "evaluation_results.json"
            metrics = {}
            if eval_file.exists():
                with open(eval_file) as f:
                    eval_data = json.load(f)
                    metrics = eval_data.get("metrics", {})
            
            # Extract key config values
            training_config = exp_data.get("config", {}).get("training", {})
            model_config = exp_data.get("config", {}).get("model", {})
            
            comparison = {
                "experiment_id": exp_id,
                "timestamp": exp_data.get("timestamp", ""),
                "learning_rate": training_config.get("learning_rate"),
                "batch_size": training_config.get("batch_size"),
                "num_epochs": training_config.get("num_epochs"),
                "base_model": model_config.get("base_model"),
                f"metric_{metric}": metrics.get(metric, None),
            }
            
            # Add all metrics
            for key, value in metrics.items():
                comparison[f"metric_{key}"] = value
            
            comparisons.append(comparison)
        
        df = pd.DataFrame(comparisons)
        
        if not df.empty:
            logger.info(f"Compared {len(df)} experiments")
            logger.info(f"\n{df.to_string()}")
        
        return df

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any]:
        """Get summary of an experiment including config and metrics.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            Dictionary with experiment summary
        """
        exp_dir = self.experiment_dir / experiment_id
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        summary = {}
        
        # Load config
        config_file = exp_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                summary["config"] = json.load(f)
        
        # Load metrics
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                summary["metrics"] = json.load(f)
        
        # Load evaluation results
        eval_file = exp_dir / "evaluation_results.json"
        if eval_file.exists():
            with open(eval_file) as f:
                summary["evaluation"] = json.load(f)
        
        return summary


_tracker_instance: ExperimentTracker | None = None


def get_tracker(experiment_dir: str | Path = "experiments") -> ExperimentTracker:
    """Get or create experiment tracker singleton."""
    global _tracker_instance
    
    if _tracker_instance is None:
        _tracker_instance = ExperimentTracker(experiment_dir)
    
    return _tracker_instance

