"""Training and evaluation for counter speech generation."""

from counter_speech_generation.training.evaluator import evaluate_generation
from counter_speech_generation.training.experiment import ExperimentTracker, get_tracker
from counter_speech_generation.training.trainer import train

__all__ = ["train", "evaluate_generation", "ExperimentTracker", "get_tracker"]

