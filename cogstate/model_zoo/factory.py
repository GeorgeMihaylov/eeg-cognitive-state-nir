from typing import Any, Mapping, Optional, Sequence, cast

from .base import ModelLike
from .ML.sklearn_models import SKLEARN_MODEL_NAMES, build_sklearn_model
from model_zoo.factory import (
    build_model as build_benchmark_model,
    model_requires_data_shape as benchmark_model_requires_data_shape,
    model_requires_sequences as benchmark_model_requires_sequences,
)


TORCH_MODEL_NAMES = frozenset(
    {
        "torch_mlp",
        "torch_lstm",
        "torch_bilstm",
        "torch_eegnet",
        "torch_shallow_convnet",
        "torch_shallow_convnet_multitask",
    }
)
SEQUENCE_MODEL_NAMES = frozenset({"torch_lstm", "torch_bilstm"})


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

    if normalized_name != "torch_shallow_convnet_multitask":
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

    if normalized_name in SKLEARN_MODEL_NAMES:
        model = build_sklearn_model(
            model_name=normalized_name,
            task_type=task_type,
            params=params,
        )
        return cast(ModelLike, model)

    if normalized_name in TORCH_MODEL_NAMES:
        if task_type.strip().lower() not in {"classification", "classifier"}:
            raise ValueError(f"{normalized_name} currently supports classification only")
        if input_shape is None:
            raise ValueError(
                f"{normalized_name} requires input_shape from the training data"
            )
        if num_outputs is None:
            raise ValueError(f"{normalized_name} requires num_outputs from the task labels")
        if normalized_name == "torch_mlp":
            from .DL.mlp import build_torch_mlp

            return cast(
                ModelLike,
                build_torch_mlp(
                    input_shape=input_shape,
                    num_outputs=int(num_outputs),
                    params=params,
                ),
            )

        if normalized_name == "torch_eegnet":
            from .DL.eegnet import build_torch_eegnet

            return cast(
                ModelLike,
                build_torch_eegnet(
                    input_shape=input_shape,
                    num_outputs=int(num_outputs),
                    params=params,
                ),
            )

        if normalized_name == "torch_shallow_convnet":
            from .DL.shallow_convnet import build_torch_shallow_convnet

            return cast(
                ModelLike,
                build_torch_shallow_convnet(
                    input_shape=input_shape,
                    num_outputs=int(num_outputs),
                    params=params,
                ),
            )

        if normalized_name == "torch_shallow_convnet_multitask":
            from .DL.shallow_convnet import build_torch_shallow_convnet_multitask

            return cast(
                ModelLike,
                build_torch_shallow_convnet_multitask(
                    input_shape=input_shape,
                    num_outputs=int(num_outputs),
                    params=params,
                ),
            )

        from .DL.lstm import build_torch_lstm

        return cast(
            ModelLike,
            build_torch_lstm(
                input_shape=input_shape,
                num_outputs=int(num_outputs),
                params=params,
                force_bidirectional=(
                    True if normalized_name == "torch_bilstm" else None
                ),
            ),
        )

    shape_hint = tuple(input_shape) if input_shape is not None else None
    raise ValueError(
        f"Unknown model {model_name!r}. Available models: "
        f"{sorted(SKLEARN_MODEL_NAMES | TORCH_MODEL_NAMES)}. "
        f"Received input_shape={shape_hint}, num_outputs={num_outputs}."
    )
