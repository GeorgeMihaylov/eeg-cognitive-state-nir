from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, Union, runtime_checkable


PathLike = Union[str, Path]


@runtime_checkable
class ModelLike(Protocol):
    """Smallest interface required by ``BenchmarkRunner``."""

    def fit(self, X: Any, y: Any) -> Any:
        ...

    def predict(self, X: Any) -> Any:
        ...


class BaseModelAdapter(ABC):
    """Base contract for future non-sklearn adapters, including PyTorch."""

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseModelAdapter":
        """Fit the wrapped model and return the adapter."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Return class labels or regression predictions."""

    def predict_proba(self, X: Any) -> Any:
        """Return class probabilities when the task supports them."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_proba"
        )

    @abstractmethod
    def save(self, path: PathLike) -> None:
        """Persist adapter state for later inference."""
