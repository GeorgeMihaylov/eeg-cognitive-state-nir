"""Shallow ConvNet classifier for fixed-size raw EEG windows."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from cogstate.protocol import PM_METRICS

from .adapter import (
    TorchClassificationAdapter,
    TorchMultiTaskClassificationAdapter,
    seed_torch,
)


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


class TorchShallowConvNetClassifier(nn.Module):
    """Shallow temporal/spatial ConvNet for ``[B, 1, channels, time]``."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        num_classes: int,
        *,
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
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
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
        self.classifier = nn.Linear(self.n_filters, self.num_classes)

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

    def extract_features(self, inputs: Tensor) -> Tensor:
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
        return self.features(spatial)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.extract_features(inputs))


class TorchShallowConvNetMultiTaskClassifier(nn.Module):
    """One shared ShallowConvNet encoder and one classification head per PM metric."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        num_classes: int,
        *,
        metric_names: Sequence[str] = PM_METRICS,
        **architecture: Any,
    ) -> None:
        super().__init__()
        self.metric_names = tuple(metric_names)
        if not self.metric_names or len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names must contain unique names")
        self.encoder = TorchShallowConvNetClassifier(
            n_channels=n_channels,
            n_times=n_times,
            num_classes=num_classes,
            **architecture,
        )
        self.encoder.classifier = nn.Identity()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(self.encoder.n_filters, num_classes) for name in self.metric_names}
        )

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        representation = self.encoder.extract_features(inputs)
        return {name: head(representation) for name, head in self.heads.items()}


def build_torch_shallow_convnet(
    input_shape: Sequence[int],
    num_outputs: int,
    params: Optional[Mapping[str, Any]] = None,
) -> TorchClassificationAdapter:
    """Build the shallow raw-EEG module and shared classification adapter."""
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
    seed_torch(random_state)
    module = TorchShallowConvNetClassifier(
        n_channels=shape[1],
        n_times=shape[2],
        num_classes=int(num_outputs),
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
            },
            **model_params,
        )
    except TypeError as exc:
        raise ValueError(
            f"Unsupported torch_shallow_convnet parameter: {exc}"
        ) from exc


def build_torch_shallow_convnet_multitask(
    input_shape: Sequence[int],
    num_outputs: int,
    params: Optional[Mapping[str, Any]] = None,
) -> TorchMultiTaskClassificationAdapter:
    """Build a raw-EEG ShallowConvNet predicting all seven PM metrics."""
    shape = tuple(int(dimension) for dimension in input_shape)
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(
            "torch_shallow_convnet_multitask requires input_shape=(1, n_channels, n_times), "
            f"got {shape}"
        )
    model_params = dict(params or {})
    sampling_rate = float(model_params.pop("sampling_rate", 256.0))
    channel_names = list(model_params.pop(
        "channel_names", [f"channel_{index}" for index in range(shape[1])]
    ))
    metric_names = tuple(model_params.pop("metric_names", PM_METRICS))
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
    seed_torch(random_state)
    architecture = {
        "n_filters": n_filters,
        "temporal_kernel_samples": temporal_kernel_samples,
        "pool_size": pool_size,
        "pool_stride": pool_stride,
        "dropout": dropout,
        "log_minimum": log_minimum,
    }
    module = TorchShallowConvNetMultiTaskClassifier(
        n_channels=shape[1],
        n_times=shape[2],
        num_classes=int(num_outputs),
        metric_names=metric_names,
        **architecture,
    )
    try:
        return TorchMultiTaskClassificationAdapter(
            model=module,
            input_shape=shape,
            num_classes=int(num_outputs),
            metric_names=metric_names,
            model_metadata={
                "model_type": "torch_shallow_convnet_multitask",
                "input_layout": "batch,1,channels,time",
                "sampling_rate": sampling_rate,
                "channel_names": channel_names,
                "metric_names": list(metric_names),
                "n_channels": shape[1],
                "n_times": shape[2],
                **architecture,
            },
            **model_params,
        )
    except TypeError as exc:
        raise ValueError(
            f"Unsupported torch_shallow_convnet_multitask parameter: {exc}"
        ) from exc
