"""Model implementations and registry."""

from .base import BaseEmotionModel
from ..registry import create_model, get_model_cls, list_models, register_model

# Import model implementations to trigger registration
from .sklearn_models import LinearSVMModel, LogisticRegressionModel, MultinomialNBModel
from .transformer_model import IndoBERTEmotionModel

__all__ = [
    "BaseEmotionModel",
    "create_model",
    "get_model_cls",
    "list_models",
    "register_model",
    "LinearSVMModel",
    "LogisticRegressionModel",
    "MultinomialNBModel",
    "IndoBERTEmotionModel",
]

