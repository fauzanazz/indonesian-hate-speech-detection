"""Model registry for emotion detection."""

from __future__ import annotations

from typing import Dict, List, Type

from .models.base import BaseEmotionModel


_REGISTRY: Dict[str, Type[BaseEmotionModel]] = {}


def register_model(name: str):
    """Decorator to register a model class under ``name``."""

    def decorator(cls: Type[BaseEmotionModel]) -> Type[BaseEmotionModel]:
        if name in _REGISTRY:
            raise ValueError(f"Model name already registered: {name}")
        if not issubclass(cls, BaseEmotionModel):
            raise TypeError("Registered class must inherit from BaseEmotionModel")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_model_cls(name: str) -> Type[BaseEmotionModel]:
    try:
        return _REGISTRY[name]
    except KeyError as err:
        available = ", ".join(sorted(_REGISTRY)) or "<empty>"
        raise KeyError(f"Model '{name}' not registered. Available: {available}") from err


def create_model(name: str, **kwargs) -> BaseEmotionModel:
    cls = get_model_cls(name)
    return cls(**kwargs)


def list_models() -> List[str]:
    return sorted(_REGISTRY.keys())


