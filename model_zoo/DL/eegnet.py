"""A compact EEGNet-style classifier for real raw EEG windows."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .adapter import TorchClassificationAdapter, seed_torch
from .encoder import ENCODER_API_VERSION, SharedEncoderMixin


class TorchEEGNetClassifier(nn.Module, SharedEncoderMixin):
    """EEGNet-inspired temporal/spatial/separable convolutional classifier."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        num_classes: int,
        *,
        temporal_kernel_samples: int,
        separable_kernel_samples: int,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.num_classes = int(num_classes)
        self.temporal_kernel_samples = int(temporal_kernel_samples)
        self.separable_kernel_samples = int(separable_kernel_samples)
        if self.n_channels <= 0 or self.n_times <= 0:
            raise ValueError("n_channels and n_times must be positive")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if min(f1, depth_multiplier, f2, pool1, pool2) <= 0:
            raise ValueError("EEGNet filter, depth, and pooling sizes must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        spatial_filters = int(f1 * depth_multiplier)
        temporal_left = (self.temporal_kernel_samples - 1) // 2
        temporal_right = self.temporal_kernel_samples - 1 - temporal_left
        separable_left = (self.separable_kernel_samples - 1) // 2
        separable_right = self.separable_kernel_samples - 1 - separable_left
        self.features = nn.Sequential(
            nn.ZeroPad2d((temporal_left, temporal_right, 0, 0)),
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, self.temporal_kernel_samples),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                spatial_filters,
                kernel_size=(self.n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(dropout),
            nn.ZeroPad2d((separable_left, separable_right, 0, 0)),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, self.separable_kernel_samples),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_times)
            flattened = int(self.features(dummy).numel())
        if flattened <= 0:
            raise ValueError(
                "EEGNet pooling/kernel configuration produced an empty representation"
            )
        self._latent_dim = flattened
        self.classifier = nn.Linear(flattened, self.num_classes)

    def encode(self, X: Tensor) -> Tensor:
        """Return flattened EEGNet features with shape ``[batch, latent_dim]``."""
        expected = (1, self.n_channels, self.n_times)
        if X.ndim != 4:
            raise ValueError(
                "TorchEEGNetClassifier expects [batch, 1, channels, time], "
                f"got shape {tuple(X.shape)}"
            )
        if tuple(X.shape[1:]) != expected:
            raise ValueError(
                f"TorchEEGNetClassifier expects input tail {expected}, "
                f"got {tuple(X.shape[1:])}"
            )
        features = self.features(X).flatten(start_dim=1)
        if features.shape[1] != self.latent_dim:
            raise RuntimeError(
                "EEGNet encoder returned an unexpected representation width: "
                f"{features.shape[1]} != {self.latent_dim}"
            )
        return features

    def forward(self, X: Tensor) -> Tensor:
        return self.forward_head(self.encode(X))


def _seconds_to_samples(seconds: float, sampling_rate: float, name: str) -> int:
    if seconds <= 0:
        raise ValueError(f"{name} must be positive")
    samples = int(round(seconds * sampling_rate))
    if samples < 2:
        raise ValueError(
            f"{name}={seconds} s gives only {samples} sample(s) at {sampling_rate} Hz"
        )
    return samples


def build_torch_eegnet(
    input_shape: Sequence[int],
    num_outputs: int,
    params: Optional[Mapping[str, Any]] = None,
) -> TorchClassificationAdapter:
    """Build EEGNet and the shared sklearn-like classification adapter."""
    shape = tuple(int(dimension) for dimension in input_shape)
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(
            "torch_eegnet requires input_shape=(1, n_channels, n_times), "
            f"got {shape}"
        )
    model_params = dict(params or {})
    sampling_rate = float(model_params.pop("sampling_rate", 256.0))
    channel_names = list(
        model_params.pop(
            "channel_names", [f"channel_{index}" for index in range(shape[1])]
        )
    )
    if len(channel_names) != shape[1]:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries for {shape[1]} channels"
        )
    temporal_kernel_seconds = float(
        model_params.pop("temporal_kernel_seconds", 0.5)
    )
    separable_kernel_seconds = float(
        model_params.pop("separable_kernel_seconds", 0.125)
    )
    temporal_kernel_samples = _seconds_to_samples(
        temporal_kernel_seconds, sampling_rate, "temporal_kernel_seconds"
    )
    separable_kernel_samples = _seconds_to_samples(
        separable_kernel_seconds, sampling_rate, "separable_kernel_seconds"
    )
    f1 = int(model_params.pop("f1", 8))
    depth_multiplier = int(model_params.pop("depth_multiplier", 2))
    f2 = int(model_params.pop("f2", f1 * depth_multiplier))
    pool1 = int(model_params.pop("pool1", 4))
    pool2 = int(model_params.pop("pool2", 8))
    dropout = float(model_params.pop("dropout", 0.5))
    random_state = int(model_params.get("random_state", 42))
    seed_torch(random_state)
    model = TorchEEGNetClassifier(
        n_channels=shape[1],
        n_times=shape[2],
        num_classes=int(num_outputs),
        temporal_kernel_samples=temporal_kernel_samples,
        separable_kernel_samples=separable_kernel_samples,
        f1=f1,
        depth_multiplier=depth_multiplier,
        f2=f2,
        pool1=pool1,
        pool2=pool2,
        dropout=dropout,
    )
    try:
        return TorchClassificationAdapter(
            model=model,
            input_shape=shape,
            num_classes=int(num_outputs),
            model_metadata={
                "model_type": "torch_eegnet",
                "input_layout": "batch,1,channels,time",
                "sampling_rate": sampling_rate,
                "channel_names": channel_names,
                "n_channels": shape[1],
                "n_times": shape[2],
                "temporal_kernel_seconds": temporal_kernel_seconds,
                "temporal_kernel_samples": temporal_kernel_samples,
                "separable_kernel_seconds": separable_kernel_seconds,
                "separable_kernel_samples": separable_kernel_samples,
                "f1": f1,
                "depth_multiplier": depth_multiplier,
                "f2": f2,
                "pool1": pool1,
                "pool2": pool2,
                "dropout": dropout,
                "latent_dim": model.latent_dim,
                "encoder_api_version": ENCODER_API_VERSION,
            },
            **model_params,
        )
    except TypeError as exc:
        raise ValueError(f"Unsupported torch_eegnet parameter: {exc}") from exc
