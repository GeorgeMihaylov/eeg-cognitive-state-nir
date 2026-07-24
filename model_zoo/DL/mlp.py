from typing import Any, Mapping, Optional, Sequence

from torch import Tensor, nn

from .adapter import TorchClassificationAdapter, seed_torch


class TorchMLP(nn.Module):
    """Feature-based MLP that maps ``[batch, features]`` to class logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        hidden = tuple(int(dim) for dim in hidden_dims)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes/output dimension must be positive")
        if not hidden or any(dim <= 0 for dim in hidden):
            raise ValueError("hidden_dims must contain positive dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        activation_factories = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        normalized_activation = activation.strip().lower()
        if normalized_activation not in activation_factories:
            raise ValueError(
                f"Unsupported activation {activation!r}. "
                f"Available: {sorted(activation_factories)}"
            )

        layers = []
        previous_dim = self.input_dim
        for hidden_dim in hidden:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(activation_factories[normalized_activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, self.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            raise ValueError(
                f"TorchMLP expects [batch, features], got shape {tuple(X.shape)}"
            )
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"TorchMLP expects {self.input_dim} features, got {X.shape[1]}"
            )
        return self.network(X)


def build_torch_mlp(
    input_shape: Sequence[int],
    num_outputs: int,
    params: Optional[Mapping[str, Any]] = None,
    task_type: str = "classification",
) -> TorchClassificationAdapter:
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) != 1:
        raise ValueError(f"torch_mlp requires input_shape=(n_features,), got {shape}")
    model_params = dict(params or {})
    hidden_dims = model_params.pop("hidden_dims", [256, 128])
    dropout = model_params.pop("dropout", 0.3)
    activation = model_params.pop("activation", "relu")
    random_state = int(model_params.get("random_state", 42))
    regression_loss = model_params.pop("regression_loss", "mse")
    seed_torch(random_state)
    model = TorchMLP(
        input_dim=shape[0],
        num_classes=num_outputs,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    )
    try:
        return TorchClassificationAdapter(
            model=model,
            input_shape=shape,
            num_classes=num_outputs,
            task_type=task_type,
            regression_loss=regression_loss,
            model_metadata={
                "model_type": "torch_mlp",
                "task_type": str(task_type),
                "hidden_dims": list(hidden_dims),
                "dropout": float(dropout),
                "activation": str(activation),
                "regression_loss": (
                    str(regression_loss)
                    if str(task_type).strip().lower()
                    in {"regression", "regressor"}
                    else None
                ),
            },
            **model_params,
        )
    except TypeError as exc:
        raise ValueError(f"Unsupported torch_mlp parameter: {exc}") from exc
