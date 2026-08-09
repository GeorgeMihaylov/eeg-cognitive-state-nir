"""Shallow ConvNet for classification or scalar raw-EEG regression."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .adapter import TorchClassificationAdapter, seed_torch
from .encoder import ENCODER_API_VERSION, SharedEncoderMixin


class SquareActivation(nn.Module):
    """Element-wise square used by shallow filter-bank ConvNets."""

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.square(inputs)


class SafeLog(nn.Module):
    """Numerically safe element-wise logarithm."""

    def __init__(self, minimum: float = 1e-6) -> None:
        super().__init__()
        self.minimum = float(minimum)
        if self.minimum <= 0:
            raise ValueError("SafeLog minimum must be positive")

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.log(torch.clamp(inputs, min=self.minimum))


class TorchShallowConvNetClassifier(nn.Module, SharedEncoderMixin):
    """Shallow temporal/spatial ConvNet for ``[B, 1, channels, time]``."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        num_classes: int,
        *,
        task_type: str = "classification",
        n_filters: int = 40,
        temporal_kernel_samples: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
        log_minimum: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.num_classes = int(num_classes)
        self.task_type = str(task_type).strip().lower()
        if self.task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.n_filters = int(n_filters)
        self.temporal_kernel_samples = int(temporal_kernel_samples)
        self.pool_size = int(pool_size)
        self.pool_stride = int(pool_stride)
        if min(
            self.n_channels,
            self.n_times,
            self.n_filters,
            self.temporal_kernel_samples,
            self.pool_size,
            self.pool_stride,
        ) <= 0:
            raise ValueError("ShallowConvNet dimensions and pooling sizes must be positive")
        if self.task_type == "classification" and self.num_classes < 2:
            raise ValueError("classification output width must be at least 2")
        if self.task_type == "regression" and self.num_classes != 1:
            raise ValueError("ShallowConvNet regression currently requires one output")
        if self.n_times < self.pool_size:
            raise ValueError(
                f"n_times={self.n_times} must be at least pool_size={self.pool_size}"
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        temporal_left = (self.temporal_kernel_samples - 1) // 2
        temporal_right = self.temporal_kernel_samples - 1 - temporal_left
        self.temporal = nn.Sequential(
            nn.ZeroPad2d((temporal_left, temporal_right, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.n_filters,
                kernel_size=(1, self.temporal_kernel_samples),
            ),
        )
        self.spatial = nn.Conv2d(
            in_channels=self.n_filters,
            out_channels=self.n_filters,
            kernel_size=(self.n_channels, 1),
            groups=self.n_filters,
        )
        self.features = nn.Sequential(
            nn.BatchNorm2d(self.n_filters),
            SquareActivation(),
            nn.AvgPool2d(
                kernel_size=(1, self.pool_size),
                stride=(1, self.pool_stride),
            ),
            SafeLog(minimum=log_minimum),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self._latent_dim = self.n_filters
        self.classifier = nn.Linear(self.latent_dim, self.num_classes)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_times)
            spatial = self.spatial(self.temporal(dummy))
            if spatial.shape[2] != 1:
                raise RuntimeError(
                    "Spatial convolution must collapse the EEG channel dimension "
                    f"to 1, got {tuple(spatial.shape)}"
                )
            representation = self.features(spatial)
            if tuple(representation.shape) != (1, self.n_filters):
                raise RuntimeError(
                    "Unexpected ShallowConvNet representation shape "
                    f"{tuple(representation.shape)}"
                )

    def encode(self, inputs: Tensor) -> Tensor:
        """Return shallow ConvNet features with shape ``[batch, latent_dim]``."""
        expected = (1, self.n_channels, self.n_times)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                "TorchShallowConvNetClassifier expects input tail "
                f"{expected}, got {tuple(inputs.shape)}"
            )
        spatial = self.spatial(self.temporal(inputs))
        if spatial.shape[2] != 1:
            raise RuntimeError(
                "Spatial convolution did not collapse the EEG channel dimension"
            )
        features = self.features(spatial)
        if features.ndim != 2 or features.shape[1] != self.latent_dim:
            raise RuntimeError(
                "ShallowConvNet encoder returned an unexpected representation "
                f"shape {tuple(features.shape)}"
            )
        return features

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_head(self.encode(inputs))


def build_torch_shallow_convnet(
    input_shape: Sequence[int],
    num_outputs: int,
    params: Optional[Mapping[str, Any]] = None,
    *,
    task_type: str = "classification",
) -> TorchClassificationAdapter:
    """Build the shallow raw-EEG module and shared PyTorch adapter."""
    shape = tuple(int(dimension) for dimension in input_shape)
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(
            "torch_shallow_convnet requires input_shape=(1, n_channels, n_times), "
            f"got {shape}"
        )
    model_params = dict(params or {})
    sampling_rate = float(model_params.pop("sampling_rate", 256.0))
    channel_names = list(model_params.pop(
        "channel_names", [f"channel_{index}" for index in range(shape[1])]
    ))
    if len(channel_names) != shape[1]:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries for {shape[1]} channels"
        )
    n_filters = int(model_params.pop("n_filters", 40))
    temporal_kernel_samples = int(model_params.pop("temporal_kernel_samples", 25))
    pool_size = int(model_params.pop("pool_size", 75))
    pool_stride = int(model_params.pop("pool_stride", 15))
    dropout = float(model_params.pop("dropout", 0.5))
    log_minimum = float(model_params.pop("log_minimum", 1e-6))
    random_state = int(model_params.get("random_state", 42))
    normalized_task_type = {
        "classifier": "classification",
        "regressor": "regression",
    }.get(str(task_type).strip().lower(), str(task_type).strip().lower())
    if normalized_task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'")
    if normalized_task_type == "classification" and int(num_outputs) < 2:
        raise ValueError("classification output width must be at least 2")
    if normalized_task_type == "regression" and int(num_outputs) != 1:
        raise ValueError("torch_shallow_convnet supports scalar regression only")
    seed_torch(random_state)
    module = TorchShallowConvNetClassifier(
        n_channels=shape[1],
        n_times=shape[2],
        num_classes=int(num_outputs),
        task_type=normalized_task_type,
        n_filters=n_filters,
        temporal_kernel_samples=temporal_kernel_samples,
        pool_size=pool_size,
        pool_stride=pool_stride,
        dropout=dropout,
        log_minimum=log_minimum,
    )
    try:
        return TorchClassificationAdapter(
            model=module,
            input_shape=shape,
            num_classes=int(num_outputs),
            task_type=normalized_task_type,
            model_metadata={
                "model_type": "torch_shallow_convnet",
                "input_layout": "batch,1,channels,time",
                "sampling_rate": sampling_rate,
                "channel_names": channel_names,
                "n_channels": shape[1],
                "n_times": shape[2],
                "n_filters": n_filters,
                "temporal_kernel_samples": temporal_kernel_samples,
                "pool_size": pool_size,
                "pool_stride": pool_stride,
                "dropout": dropout,
                "log_minimum": log_minimum,
                "latent_dim": module.latent_dim,
                "encoder_api_version": ENCODER_API_VERSION,
                "task_type": normalized_task_type,
                "num_outputs": int(num_outputs),
            },
            **model_params,
        )
    except TypeError as exc:
        raise ValueError(
            f"Unsupported torch_shallow_convnet parameter: {exc}"
        ) from exc
