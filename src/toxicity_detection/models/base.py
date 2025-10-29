from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt


class BaseModel(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.model: Optional[Any] = None
        self.is_trained = False

    @abstractmethod
    def train(
        self,
        X_train: List[str],
        y_train: npt.NDArray[Any],
        X_val: Optional[List[str]] = None,
        y_val: Optional[npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> npt.NDArray[Any]:
        pass

    @abstractmethod
    def predict_proba(self, texts: List[str]) -> npt.NDArray[Any]:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass

    def get_params(self) -> Dict[str, Any]:
        return {"name": self.name, "is_trained": self.is_trained}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', is_trained={self.is_trained})"
