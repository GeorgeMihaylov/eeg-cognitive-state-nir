from typing import Any, Mapping, Optional, Sequence, cast

from .base import ModelLike
from .ML.sklearn_models import SKLEARN_MODEL_NAMES, build_sklearn_model


TORCH_MODEL_NAMES = frozenset(
    {
        "torch_mlp",
        "torch_lstm",
        "torch_bilstm",
        "torch_eegnet",
        "torch_shallow_convnet",
        "torch_shallow_fusion",
        "torch_transformer",
    }
)
SEQUENCE_MODEL_NAMES = frozenset(
    {"torch_lstm", "torch_bilstm", "torch_transformer"}
)


def model_requires_data_shape(model_name: str) -> bool:
    """Return whether a model must be built after dataset loading."""
    return model_name.strip().lower() in TORCH_MODEL_NAMES


def model_requires_sequences(model_name: str) -> bool:
    """Return whether a model consumes [batch, sequence, features]."""
    return model_name.strip().lower() in SEQUENCE_MODEL_NAMES


def build_model(
    model_name: str,
    task_type: str,
    input_shape: Optional[Sequence[int]],
    num_outputs: Optional[int],
    params: Optional[Mapping[str, Any]] = None,
) -> ModelLike:
    """Build a benchmark model through the shared model-zoo entry point."""
    normalized_name = model_name.strip().lower()

    if normalized_name in SKLEARN_MODEL_NAMES:
        model = build_sklearn_model(
            model_name=normalized_name,
            task_type=task_type,
            params=params,
        )
        return cast(ModelLike, model)

    if normalized_name in TORCH_MODEL_NAMES:
        normalized_task_type = task_type.strip().lower()
        is_classification = normalized_task_type in {
            "classification",
            "classifier",
        }
        is_regression = normalized_task_type in {"regression", "regressor"}
        regression_models = {"torch_mlp", "torch_shallow_convnet"}
        if not is_classification and not (
            normalized_name in regression_models and is_regression
        ):
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
                    task_type=(
                        "regression" if is_regression else "classification"
                    ),
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
                    task_type=(
                        "regression" if is_regression else "classification"
                    ),
                ),
            )

        if normalized_name == "torch_shallow_fusion":
            from .DL.shallow_fusion import build_torch_shallow_fusion

            return cast(
                ModelLike,
                build_torch_shallow_fusion(
                    input_shape=input_shape,
                    num_outputs=int(num_outputs),
                    params=params,
                ),
            )

        if normalized_name == "torch_transformer":
            from .DL.transformer import build_torch_transformer

            return cast(
                ModelLike,
                build_torch_transformer(
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
