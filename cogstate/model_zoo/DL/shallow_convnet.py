"""Shallow ConvNet classifier for fixed-size raw EEG windows."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from torch import Tensor, nn

from cogstate.protocol import PM_METRICS
from model_zoo.DL.shallow_convnet import (
    SafeLog,
    SquareActivation,
    TorchShallowConvNetClassifier,
    build_torch_shallow_convnet,
)

from .adapter import TorchMultiTaskClassificationAdapter, seed_torch


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
        representation = self.encoder.encode(inputs)
        return {name: head(representation) for name, head in self.heads.items()}


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
