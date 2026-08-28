"""Minimal ShallowConvNet plus peripheral-feature fusion classifier."""

from __future__ import annotations

from math import prod
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .adapter import TorchClassificationAdapter, seed_torch
from .encoder import ENCODER_API_VERSION, SharedEncoderMixin
from .shallow_convnet import TorchShallowConvNetClassifier


class TorchShallowFusionClassifier(nn.Module, SharedEncoderMixin):
    """Two-branch model transported as one explicitly packed feature vector.

    The first ``prod(eeg_input_shape)`` values are reshaped to raw EEG
    ``[B, 1, channels, time]``.  The remaining values are passed only to the
    peripheral MLP.  Peripheral features are therefore never interpreted as
    EEG channels or time samples.
    """

    def __init__(
        self,
        *,
        eeg_input_shape: Sequence[int],
        peripheral_dim: int,
        num_classes: int,
        n_filters: int = 40,
        temporal_kernel_samples: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        eeg_dropout: float = 0.5,
        peripheral_hidden_dim: int = 32,
        peripheral_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.eeg_input_shape = tuple(int(value) for value in eeg_input_shape)
        if len(self.eeg_input_shape) != 3 or self.eeg_input_shape[0] != 1:
            raise ValueError("eeg_input_shape must be (1, channels, time)")
        self.peripheral_dim = int(peripheral_dim)
        self.num_classes = int(num_classes)
        if self.peripheral_dim <= 0:
            raise ValueError("peripheral_dim must be positive")
        if peripheral_hidden_dim <= 0:
            raise ValueError("peripheral_hidden_dim must be positive")
        if not 0 <= peripheral_dropout < 1:
            raise ValueError("peripheral_dropout must be in [0, 1)")
        self.eeg_flat_dim = int(prod(self.eeg_input_shape))
        self.packed_dim = self.eeg_flat_dim + self.peripheral_dim
        self.eeg_encoder = TorchShallowConvNetClassifier(
            n_channels=self.eeg_input_shape[1],
            n_times=self.eeg_input_shape[2],
            num_classes=self.num_classes,
            n_filters=n_filters,
            temporal_kernel_samples=temporal_kernel_samples,
            pool_size=pool_size,
            pool_stride=pool_stride,
            dropout=eeg_dropout,
        )
        # The fusion classifier owns the only output head.  The shallow module
        # is retained strictly as the same tested EEG encoder implementation.
        self.eeg_encoder.classifier = nn.Identity()
        self.peripheral_encoder = nn.Sequential(
            nn.Linear(self.peripheral_dim, int(peripheral_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(peripheral_dropout)),
        )
        self._latent_dim = self.eeg_encoder.latent_dim + int(peripheral_hidden_dim)
        self.classifier = nn.Linear(self.latent_dim, self.num_classes)

    def split_inputs(self, packed: Tensor) -> tuple[Tensor, Tensor]:
        if packed.ndim != 2 or packed.shape[1] != self.packed_dim:
            raise ValueError(
                f"Shallow fusion expects [batch, {self.packed_dim}], got {tuple(packed.shape)}"
            )
        if not torch.isfinite(packed).all():
            raise ValueError("Packed multimodal input contains NaN or Inf")
        eeg = packed[:, : self.eeg_flat_dim].reshape(
            packed.shape[0], *self.eeg_input_shape
        )
        peripheral = packed[:, self.eeg_flat_dim :]
        return eeg, peripheral

    def encode(self, packed: Tensor) -> Tensor:
        eeg, peripheral = self.split_inputs(packed)
        eeg_features = self.eeg_encoder.encode(eeg)
        peripheral_features = self.peripheral_encoder(peripheral)
        features = torch.cat((eeg_features, peripheral_features), dim=1)
        if tuple(features.shape[1:]) != (self.latent_dim,):
            raise RuntimeError(f"Unexpected fusion representation {tuple(features.shape)}")
        return features

    def forward(self, packed: Tensor) -> Tensor:
        return self.forward_head(self.encode(packed))


class TorchShallowFusionAdapter(TorchClassificationAdapter):
    """Shared training loop with branch-aware, train-only normalization."""

    def _fit_standardizer(self, X_train: Any) -> None:
        strategy = str(
            self.feature_scaling_config_.get(
                "strategy", "standard" if self.standardize else "none"
            )
        ).strip().lower()
        if strategy == "none":
            self.feature_mean_ = None
            self.feature_scale_ = None
            self.feature_preprocessor_ = None
            return
        if strategy != "standard":
            raise ValueError(
                "torch_shallow_fusion supports only standard or none feature scaling"
            )
        values = np.asarray(X_train, dtype=np.float32)
        module = self.model
        if not isinstance(module, TorchShallowFusionClassifier):
            raise TypeError("TorchShallowFusionAdapter requires its matching module")
        eeg = values[:, : module.eeg_flat_dim].reshape(
            len(values), *module.eeg_input_shape
        )
        peripheral = values[:, module.eeg_flat_dim :]
        channel_mean = eeg.mean(axis=(0, 1, 3), dtype=np.float64)
        channel_scale = eeg.std(axis=(0, 1, 3), dtype=np.float64)
        channel_scale = np.where(channel_scale < 1e-8, 1.0, channel_scale)
        peripheral_mean = peripheral.mean(axis=0, dtype=np.float64)
        peripheral_scale = peripheral.std(axis=0, dtype=np.float64)
        peripheral_scale = np.where(peripheral_scale < 1e-8, 1.0, peripheral_scale)
        eeg_mean = np.broadcast_to(
            channel_mean[None, :, None], module.eeg_input_shape
        ).reshape(-1)
        eeg_scale = np.broadcast_to(
            channel_scale[None, :, None], module.eeg_input_shape
        ).reshape(-1)
        self.feature_mean_ = np.concatenate((eeg_mean, peripheral_mean)).astype(
            np.float32
        )
        self.feature_scale_ = np.concatenate((eeg_scale, peripheral_scale)).astype(
            np.float32
        )
        self.feature_preprocessor_ = None


def build_torch_shallow_fusion(
    input_shape: Sequence[int],
    num_outputs: int,
    params: Optional[Mapping[str, Any]] = None,
) -> TorchClassificationAdapter:
    """Build the two-branch module through the existing Torch adapter."""
    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 1:
        raise ValueError("torch_shallow_fusion requires packed input_shape=(features,)")
    model_params = dict(params or {})
    eeg_input_shape = tuple(
        int(value) for value in model_params.pop("eeg_input_shape", (1, 4, 2560))
    )
    peripheral_dim = int(model_params.pop("peripheral_dim", 28))
    sampling_rate = float(model_params.pop("sampling_rate", 256.0))
    channel_names = tuple(
        str(value)
        for value in model_params.pop("channel_names", ("TP9", "AF7", "AF8", "TP10"))
    )
    n_filters = int(model_params.pop("n_filters", 40))
    temporal_kernel_samples = int(model_params.pop("temporal_kernel_samples", 25))
    pool_size = int(model_params.pop("pool_size", 75))
    pool_stride = int(model_params.pop("pool_stride", 15))
    eeg_dropout = float(model_params.pop("eeg_dropout", 0.5))
    peripheral_hidden_dim = int(model_params.pop("peripheral_hidden_dim", 32))
    peripheral_dropout = float(model_params.pop("peripheral_dropout", 0.2))
    expected = int(prod(eeg_input_shape)) + peripheral_dim
    if shape != (expected,):
        raise ValueError(f"Packed input shape must be ({expected},), got {shape}")
    if len(channel_names) != eeg_input_shape[1]:
        raise ValueError("channel_names must match the raw EEG channel dimension")
    random_state = int(model_params.get("random_state", 42))
    seed_torch(random_state)
    module = TorchShallowFusionClassifier(
        eeg_input_shape=eeg_input_shape,
        peripheral_dim=peripheral_dim,
        num_classes=int(num_outputs),
        n_filters=n_filters,
        temporal_kernel_samples=temporal_kernel_samples,
        pool_size=pool_size,
        pool_stride=pool_stride,
        eeg_dropout=eeg_dropout,
        peripheral_hidden_dim=peripheral_hidden_dim,
        peripheral_dropout=peripheral_dropout,
    )
    return TorchShallowFusionAdapter(
        model=module,
        input_shape=shape,
        num_classes=int(num_outputs),
        model_metadata={
            "model_type": "torch_shallow_fusion",
            "input_layout": "packed(raw_eeg_flat,peripheral_features)",
            "eeg_input_shape": list(eeg_input_shape),
            "peripheral_dim": peripheral_dim,
            "sampling_rate": sampling_rate,
            "channel_names": list(channel_names),
            "eeg_latent_dim": module.eeg_encoder.latent_dim,
            "peripheral_latent_dim": peripheral_hidden_dim,
            "latent_dim": module.latent_dim,
            "encoder_api_version": ENCODER_API_VERSION,
        },
        **model_params,
    )


__all__ = [
    "TorchShallowFusionAdapter",
    "TorchShallowFusionClassifier",
    "build_torch_shallow_fusion",
]
