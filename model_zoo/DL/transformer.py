"""Feature-sequence Transformer classifier for the canonical benchmark."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .adapter import TorchClassificationAdapter, seed_torch


class TransformerPositionalEncoding(nn.Module):
    """Learned or sinusoidal positional values with a fixed maximum length."""

    SUPPORTED = frozenset({"learned", "sinusoidal"})

    def __init__(self, d_model: int, max_length: int, kind: str) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_length = int(max_length)
        self.kind = str(kind).strip().lower()
        if self.kind not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported positional_encoding {kind!r}; "
                f"expected one of {sorted(self.SUPPORTED)}"
            )
        if self.d_model <= 0 or self.max_length <= 0:
            raise ValueError("d_model and max_length must be positive")

        if self.kind == "learned":
            self.encoding = nn.Parameter(
                torch.empty(1, self.max_length, self.d_model)
            )
            nn.init.normal_(self.encoding, mean=0.0, std=0.02)
        else:
            position = torch.arange(
                self.max_length, dtype=torch.float32
            ).unsqueeze(1)
            even_indices = torch.arange(0, self.d_model, 2, dtype=torch.float32)
            div_term = torch.exp(
                even_indices * (-math.log(10_000.0) / self.d_model)
            )
            values = torch.zeros(
                1, self.max_length, self.d_model, dtype=torch.float32
            )
            values[0, :, 0::2] = torch.sin(position * div_term)
            odd_width = values[0, :, 1::2].shape[1]
            values[0, :, 1::2] = torch.cos(
                position * div_term[:odd_width]
            )
            self.register_buffer("encoding", values, persistent=True)

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 3 or X.shape[2] != self.d_model:
            raise ValueError(
                "Positional encoding expects [batch, sequence, d_model], "
                f"got {tuple(X.shape)}"
            )
        if X.shape[1] > self.max_length:
            raise ValueError(
                f"Sequence length {X.shape[1]} exceeds configured maximum "
                f"{self.max_length}"
            )
        return X + self.encoding[:, : X.shape[1]].to(
            device=X.device, dtype=X.dtype
        )


class TorchFeatureTransformerClassifier(nn.Module):
    """TransformerEncoder classifier over fixed-width EEG/POW windows."""

    SUPPORTED_ACTIVATIONS = frozenset({"relu", "gelu"})
    SUPPORTED_POOLING = frozenset({"last", "mean", "cls"})

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        sequence_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        pooling: str = "last",
        positional_encoding: str = "learned",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.activation = str(activation).strip().lower()
        self.pooling = str(pooling).strip().lower()
        self.positional_encoding_kind = str(positional_encoding).strip().lower()
        self._validate_config()

        self.input_projection = nn.Linear(self.input_size, self.d_model)
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.empty(1, 1, self.d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        positional_length = self.sequence_length + int(self.pooling == "cls")
        self.positional_encoding = TransformerPositionalEncoding(
            d_model=self.d_model,
            max_length=positional_length,
            kind=self.positional_encoding_kind,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.num_classes),
        )

    def _validate_config(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.d_model <= 0 or self.nhead <= 0:
            raise ValueError("d_model and nhead must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {self.activation!r}; expected one of "
                f"{sorted(self.SUPPORTED_ACTIVATIONS)}"
            )
        if self.pooling not in self.SUPPORTED_POOLING:
            raise ValueError(
                f"Unsupported pooling {self.pooling!r}; expected one of "
                f"{sorted(self.SUPPORTED_POOLING)}"
            )

    @staticmethod
    def _validate_padding_mask(X: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        if padding_mask is None:
            return torch.zeros(
                X.shape[:2], dtype=torch.bool, device=X.device
            )
        mask = torch.as_tensor(padding_mask, device=X.device)
        if mask.ndim != 2 or tuple(mask.shape) != tuple(X.shape[:2]):
            raise ValueError(
                "padding_mask must have shape [batch, sequence_length], "
                f"got {tuple(mask.shape)} for input {tuple(X.shape)}"
            )
        mask = mask.to(dtype=torch.bool)
        if mask.all(dim=1).any():
            raise ValueError("Every sequence must contain at least one valid token")
        return mask

    @staticmethod
    def _last_valid(encoded: Tensor, padding_mask: Tensor) -> Tensor:
        positions = torch.arange(
            encoded.shape[1], device=encoded.device
        ).unsqueeze(0).expand(encoded.shape[0], -1)
        last_indices = positions.masked_fill(padding_mask, -1).max(dim=1).values
        batch_indices = torch.arange(encoded.shape[0], device=encoded.device)
        return encoded[batch_indices, last_indices]

    @staticmethod
    def _masked_mean(encoded: Tensor, padding_mask: Tensor) -> Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(dtype=encoded.dtype)
        return (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        X: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if X.ndim != 3:
            raise ValueError(
                "TorchFeatureTransformerClassifier expects "
                "[batch, sequence_length, features], "
                f"got shape {tuple(X.shape)}"
            )
        if X.shape[1] > self.sequence_length:
            raise ValueError(
                f"Expected at most {self.sequence_length} tokens, got {X.shape[1]}"
            )
        if X.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {X.shape[2]}"
            )
        mask = self._validate_padding_mask(X, padding_mask)
        encoded = self.input_projection(X)
        if self.pooling == "cls":
            cls = self.cls_token.expand(X.shape[0], -1, -1)
            encoded = torch.cat([cls, encoded], dim=1)
            cls_mask = torch.zeros(
                (X.shape[0], 1), dtype=torch.bool, device=X.device
            )
            mask = torch.cat([cls_mask, mask], dim=1)
        encoded = self.positional_encoding(encoded)
        encoded = self.encoder(encoded, src_key_padding_mask=mask)

        if self.pooling == "cls":
            pooled = encoded[:, 0]
        elif self.pooling == "mean":
            pooled = self._masked_mean(encoded, mask)
        else:
            pooled = self._last_valid(encoded, mask)
        return self.classifier(pooled)


def build_torch_transformer(
    input_shape: Sequence[int],
    num_outputs: int,
    params: Optional[Mapping[str, Any]] = None,
) -> TorchClassificationAdapter:
    """Build the Transformer module inside the shared classification adapter."""
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) != 2:
        raise ValueError(
            "torch_transformer requires "
            f"input_shape=(sequence_length, n_features), got {shape}"
        )
    model_params = dict(params or {})
    configured_length = int(model_params.pop("sequence_length", shape[0]))
    if configured_length != shape[0]:
        raise ValueError(
            "model.params.sequence_length must match sequences built by the "
            f"benchmark: {configured_length} != {shape[0]}"
        )
    d_model = int(model_params.pop("d_model", 128))
    nhead = int(model_params.pop("nhead", 4))
    num_layers = int(model_params.pop("num_layers", 2))
    dim_feedforward = int(model_params.pop("dim_feedforward", 256))
    dropout = float(model_params.pop("dropout", 0.1))
    activation = str(model_params.pop("activation", "gelu"))
    pooling = str(model_params.pop("pooling", "last"))
    positional_encoding = str(
        model_params.pop("positional_encoding", "learned")
    )
    random_state = int(model_params.get("random_state", 42))
    seed_torch(random_state)
    model = TorchFeatureTransformerClassifier(
        input_size=shape[1],
        num_classes=int(num_outputs),
        sequence_length=shape[0],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        pooling=pooling,
        positional_encoding=positional_encoding,
    )
    metadata = {
        "model_type": "torch_transformer",
        "sequence_length": shape[0],
        "input_size": shape[1],
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "activation": activation,
        "pooling": pooling,
        "positional_encoding": positional_encoding,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    try:
        return TorchClassificationAdapter(
            model=model,
            input_shape=shape,
            num_classes=int(num_outputs),
            model_metadata=metadata,
            **model_params,
        )
    except TypeError as exc:
        raise ValueError(f"Unsupported torch_transformer parameter: {exc}") from exc
