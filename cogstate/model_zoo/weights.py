"""Persistence helpers for trained neural models from the model zoo."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .factory import build_model

_TRAINING_KEYS = {"batch_size", "max_epochs", "learning_rate", "weight_decay", "validation_size", "early_stopping_patience", "random_state", "standardize"}
_ARCHITECTURE_KEYS = {
    "torch_eegnet": {"sampling_rate", "channel_names", "temporal_kernel_seconds", "separable_kernel_seconds", "f1", "depth_multiplier", "f2", "pool1", "pool2", "dropout"},
    "torch_mlp": {"hidden_dims", "dropout", "activation"},
    "torch_lstm": {"hidden_size", "num_layers", "bidirectional", "dropout", "classifier_hidden"},
    "torch_bilstm": {"hidden_size", "num_layers", "bidirectional", "dropout", "classifier_hidden"},
    "torch_shallow_convnet": {"sampling_rate", "channel_names", "n_filters", "temporal_kernel_samples", "pool_size", "pool_stride", "dropout", "log_minimum"},
}


def load_torch_weights(path: str | Path, *, device: str = "auto") -> Any:
    """Restore a model-zoo PyTorch adapter saved with ``adapter.save``."""
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = dict(payload["model_metadata"])
    model_name = metadata["model_type"]
    params = {key: value for key, value in metadata.items() if key in _ARCHITECTURE_KEYS.get(model_name, set())}
    params.update({key: value for key, value in payload.get("training_config", {}).items() if key in _TRAINING_KEYS})
    params["device"] = device
    adapter = build_model(model_name, "classification", payload["input_shape"], payload["num_classes"], params)
    adapter.model.load_state_dict(payload["model_state_dict"])
    mean, scale = payload.get("feature_mean"), payload.get("feature_scale")
    adapter.feature_mean_ = None if mean is None else np.asarray(mean, dtype=np.float32)
    adapter.feature_scale_ = None if scale is None else np.asarray(scale, dtype=np.float32)
    adapter.is_fitted_ = True
    return adapter
