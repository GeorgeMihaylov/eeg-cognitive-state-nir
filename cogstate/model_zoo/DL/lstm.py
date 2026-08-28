from typing import Any, Mapping, Optional, Sequence

from torch import Tensor, nn

from .adapter import TorchClassificationAdapter, seed_torch


class TorchLSTMClassifier(nn.Module):
    """Feature-sequence LSTM classifier using final recurrent hidden states."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.2,
        classifier_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if classifier_hidden <= 0:
            raise ValueError("classifier_hidden must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_directions, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, self.num_classes),
        )

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 3:
            raise ValueError(
                "TorchLSTMClassifier expects [batch, sequence_length, features], "
                f"got shape {tuple(X.shape)}"
            )
        if X.shape[2] != self.input_size:
            raise ValueError(
                f"TorchLSTMClassifier expects {self.input_size} features, got {X.shape[2]}"
            )
        _, (hidden, _) = self.lstm(X)
        hidden = hidden.reshape(
            self.num_layers,
            self.num_directions,
            X.shape[0],
            self.hidden_size,
        )
        final_layer = hidden[-1]
        if self.bidirectional:
            representation = final_layer.transpose(0, 1).reshape(X.shape[0], -1)
        else:
            representation = final_layer[0]
        return self.classifier(representation)


def build_torch_lstm(
    input_shape: Sequence[int],
    num_outputs: int,
    params: Optional[Mapping[str, Any]] = None,
    *,
    force_bidirectional: Optional[bool] = None,
) -> TorchClassificationAdapter:
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) != 2:
        raise ValueError(
            "torch_lstm/torch_bilstm require "
            f"input_shape=(sequence_length, n_features), got {shape}"
        )
    model_params = dict(params or {})
    hidden_size = int(model_params.pop("hidden_size", 128))
    num_layers = int(model_params.pop("num_layers", 1))
    configured_bidirectional = bool(model_params.pop("bidirectional", False))
    bidirectional = (
        configured_bidirectional
        if force_bidirectional is None
        else bool(force_bidirectional)
    )
    dropout = float(model_params.pop("dropout", 0.2))
    classifier_hidden = int(model_params.pop("classifier_hidden", 64))
    random_state = int(model_params.get("random_state", 42))
    seed_torch(random_state)
    model = TorchLSTMClassifier(
        input_size=shape[1],
        num_classes=num_outputs,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        classifier_hidden=classifier_hidden,
    )
    model_type = "torch_bilstm" if bidirectional else "torch_lstm"
    try:
        return TorchClassificationAdapter(
            model=model,
            input_shape=shape,
            num_classes=num_outputs,
            model_metadata={
                "model_type": model_type,
                "sequence_length": shape[0],
                "input_size": shape[1],
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "bidirectional": bidirectional,
                "dropout": dropout,
                "classifier_hidden": classifier_hidden,
            },
            **model_params,
        )
    except TypeError as exc:
        raise ValueError(f"Unsupported {model_type} parameter: {exc}") from exc
