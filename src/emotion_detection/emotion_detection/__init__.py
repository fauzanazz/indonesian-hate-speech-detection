"""Emotion detection package exports."""

from .config import EmotionSettings, load_settings
from .train import train

__all__ = ["EmotionSettings", "load_settings", "train"]

