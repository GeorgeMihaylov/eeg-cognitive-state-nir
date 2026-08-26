from typing import Any, Mapping, Optional, Sequence, cast

from .base import ModelLike
from model_zoo.factory import (
    build_model as build_benchmark_model,
    model_requires_data_shape as benchmark_model_requires_data_shape,
    model_requires_sequences as benchmark_model_requires_sequences,
)


APPLICATION_MODEL_NAMES = frozenset({"torch_shallow_convnet_multitask"})


def model_requires_data_shape(model_name: str) -> bool:
    """Return whether a model must be built after dataset loading."""
    normalized = model_name.strip().lower()
    return (
        normalized == "torch_shallow_convnet_multitask"
        or benchmark_model_requires_data_shape(normalized)
    )


def model_requires_sequences(model_name: str) -> bool:
    """Return whether a model consumes [batch, sequence, features]."""
    normalized = model_name.strip().lower()
    if normalized == "torch_shallow_convnet_multitask":
        return False
    return benchmark_model_requires_sequences(normalized)


def build_model(
    model_name: str,
    task_type: str,
    input_shape: Optional[Sequence[int]],
    num_outputs: Optional[int],
    params: Optional[Mapping[str, Any]] = None,
) -> ModelLike:
    """Build a benchmark model through the shared model-zoo entry point."""
    normalized_name = model_name.strip().lower()

    if normalized_name not in APPLICATION_MODEL_NAMES:
        return cast(
            ModelLike,
            build_benchmark_model(
                model_name=model_name,
                task_type=task_type,
                input_shape=input_shape,
                num_outputs=num_outputs,
                params=params,
            ),
        )

    if task_type.strip().lower() not in {"classification", "classifier"}:
        raise ValueError(f"{normalized_name} currently supports classification only")
    if input_shape is None:
        raise ValueError(f"{normalized_name} requires input_shape from the training data")
    if num_outputs is None:
        raise ValueError(f"{normalized_name} requires num_outputs from the task labels")

    from .DL.shallow_convnet import build_torch_shallow_convnet_multitask

    return cast(
        ModelLike,
        build_torch_shallow_convnet_multitask(
            input_shape=input_shape,
            num_outputs=int(num_outputs),
            params=params,
        ),
    )
