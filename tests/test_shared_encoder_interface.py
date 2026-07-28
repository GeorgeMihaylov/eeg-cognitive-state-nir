from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from model_zoo import build_model
from model_zoo.DL.adapter import TorchClassificationAdapter
from model_zoo.DL.eegnet import TorchEEGNetClassifier
from model_zoo.DL.encoder import ENCODER_API_VERSION, require_encoder_model
from model_zoo.DL.shallow_convnet import TorchShallowConvNetClassifier
from model_zoo.DL.transformer import TorchFeatureTransformerClassifier


def _eegnet(*, n_times: int = 64) -> TorchEEGNetClassifier:
    return TorchEEGNetClassifier(
        n_channels=4,
        n_times=n_times,
        num_classes=5,
        temporal_kernel_samples=16,
        separable_kernel_samples=8,
        f1=2,
        depth_multiplier=1,
        f2=2,
        pool1=2,
        pool2=2,
        dropout=0.0,
    )


def _shallow(*, n_filters: int = 4) -> TorchShallowConvNetClassifier:
    return TorchShallowConvNetClassifier(
        n_channels=4,
        n_times=128,
        num_classes=5,
        n_filters=n_filters,
        temporal_kernel_samples=9,
        pool_size=16,
        pool_stride=4,
        dropout=0.0,
    )


@pytest.mark.parametrize(
    ("model", "inputs"),
    [
        (_eegnet(), torch.randn(3, 1, 4, 64)),
        (_shallow(), torch.randn(3, 1, 4, 128)),
    ],
)
def test_forward_encode_and_head_are_equivalent(
    model: nn.Module,
    inputs: torch.Tensor,
) -> None:
    encoder = require_encoder_model(model)
    model.eval()
    with torch.no_grad():
        features = encoder.encode(inputs)
        direct = model(inputs)
        composed = encoder.forward_head(features)

    assert direct.shape == (3, 5)
    assert features.shape == (3, encoder.latent_dim)
    assert torch.equal(direct, composed)


def test_latent_dim_is_derived_by_each_model() -> None:
    short = _eegnet(n_times=64)
    long = _eegnet(n_times=128)
    shallow_small = _shallow(n_filters=3)
    shallow_large = _shallow(n_filters=6)

    assert short.latent_dim == short.classifier.in_features
    assert long.latent_dim == long.classifier.in_features
    assert short.latent_dim != long.latent_dim
    assert shallow_small.latent_dim == 3
    assert shallow_large.latent_dim == 6


@pytest.mark.parametrize(
    ("model", "inputs"),
    [
        (_eegnet(), torch.randn(2, 1, 4, 64)),
        (_shallow(), torch.randn(2, 1, 4, 128)),
    ],
)
def test_head_can_be_replaced_for_classification_and_seven_output_regression(
    model: nn.Module,
    inputs: torch.Tensor,
) -> None:
    encoder = require_encoder_model(model)
    model.eval()
    encoder_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("classifier.")
    }

    encoder.replace_output_head(3)
    assert model(inputs).shape == (2, 3)
    encoder.replace_output_head(7)
    assert model(inputs).shape == (2, 7)
    assert encoder.get_output_head().in_features == encoder.latent_dim
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in encoder_state.items()
    )


@pytest.mark.parametrize("model", [_eegnet(), _shallow()])
def test_freeze_and_unfreeze_encoder(model: nn.Module) -> None:
    encoder = require_encoder_model(model)
    encoder.freeze_encoder()

    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("classifier.")
    )

    encoder.unfreeze_encoder()
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(
    ("model", "inputs"),
    [
        (_eegnet(), torch.randn(4, 1, 4, 64)),
        (_shallow(), torch.randn(4, 1, 4, 128)),
    ],
)
def test_diagnostic_head_step_does_not_change_encoder(
    model: nn.Module,
    inputs: torch.Tensor,
) -> None:
    encoder = require_encoder_model(model)
    encoder.freeze_encoder()
    model.eval()
    before_encoder = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    }
    before_head = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("classifier.")
    }
    optimizer = torch.optim.AdamW(encoder.get_output_head().parameters(), lr=0.01)
    targets = torch.tensor([0, 1, 2, 3])

    optimizer.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(inputs), targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert all(
        torch.equal(before_encoder[name], parameter)
        for name, parameter in model.named_parameters()
        if name in before_encoder
    )
    assert any(
        not torch.equal(before_head[name], parameter)
        for name, parameter in model.named_parameters()
        if name in before_head
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_encoder_preserves_execution_device(device: str) -> None:
    model = _shallow().to(device).eval()
    inputs = torch.randn(2, 1, 4, 128, device=device)
    with torch.no_grad():
        encoded = model.encode(inputs)
        outputs = model.forward_head(encoded)
    assert encoded.device.type == device
    assert outputs.device.type == device


def _raw_adapter(*, num_outputs: int = 5) -> TorchClassificationAdapter:
    adapter = build_model(
        model_name="torch_shallow_convnet",
        task_type="classification",
        input_shape=(1, 4, 128),
        num_outputs=num_outputs,
        params={
            "n_filters": 4,
            "temporal_kernel_samples": 9,
            "pool_size": 16,
            "pool_stride": 4,
            "dropout": 0.0,
            "batch_size": 8,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "standardize": False,
            "device": "cpu",
            "random_state": 42,
        },
    )
    assert isinstance(adapter, TorchClassificationAdapter)
    return adapter


def test_adapter_exposes_encoder_and_explicit_feature_extraction() -> None:
    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(5, 1, 4, 128)).astype(np.float32)
    adapter = _raw_adapter()

    encoded = adapter.encode(inputs)

    assert adapter.get_encoder() is adapter.model
    assert adapter.get_output_head() is adapter.model.classifier
    assert encoded.shape == (5, adapter.model.latent_dim)
    assert encoded.device.type == "cpu"


def test_adapter_replaces_proxy_head_with_seven_output_regression(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(30, 1, 4, 128)).astype(np.float32)
    targets = rng.normal(size=(30, 7)).astype(np.float32)
    adapter = _raw_adapter()
    encoder_before = {
        name: parameter.detach().clone()
        for name, parameter in adapter.model.named_parameters()
        if not name.startswith("classifier.")
    }

    adapter.replace_output_head(7, task_type="regression")
    assert adapter.task_type == "regression"
    assert adapter.num_outputs == 7
    assert adapter.model(torch.from_numpy(inputs[:2])).shape == (2, 7)
    assert all(
        torch.equal(encoder_before[name], parameter)
        for name, parameter in adapter.model.named_parameters()
        if name in encoder_before
    )

    adapter.fit(inputs, targets)
    expected = adapter.predict(inputs[:4])
    checkpoint = tmp_path / "seven-output.pt"
    adapter.save(checkpoint)

    restored = _raw_adapter()
    restored.replace_output_head(7, task_type="regression")
    restored.load(checkpoint)
    np.testing.assert_allclose(restored.predict(inputs[:4]), expected, atol=1e-7)


def test_existing_classification_adapter_fit_predict_still_works() -> None:
    rng = np.random.default_rng(11)
    inputs = rng.normal(size=(30, 1, 4, 128)).astype(np.float32)
    labels = np.tile(np.arange(5), 6)
    adapter = _raw_adapter()

    adapter.fit(inputs, labels)

    assert adapter.predict(inputs[:4]).shape == (4,)
    assert adapter.predict_proba(inputs[:4]).shape == (4, 5)


@pytest.mark.parametrize("model", [_eegnet(), _shallow()])
def test_legacy_state_dict_names_and_strict_loading_are_preserved(
    model: nn.Module,
) -> None:
    state = deepcopy(model.state_dict())
    assert any(name.startswith("classifier.") for name in state)
    assert not any(name.startswith("encoder.") for name in state)
    assert not any("_latent_dim" in name for name in state)
    model.load_state_dict(state, strict=True)


@pytest.mark.parametrize("model_class", [TorchEEGNetClassifier, TorchShallowConvNetClassifier])
def test_encoder_api_has_no_dataset_fold_or_label_arguments(model_class: type[nn.Module]) -> None:
    signature = inspect.signature(model_class.encode)
    assert list(signature.parameters) in (["self", "X"], ["self", "inputs"])
    source = inspect.getsource(model_class.encode)
    for forbidden in ("self.data", "dataset", "fold", "subject_id", "labels", "y_true"):
        assert forbidden not in source


def test_feature_transformer_is_not_forced_into_raw_encoder_contract() -> None:
    transformer = TorchFeatureTransformerClassifier(
        input_size=6,
        num_classes=5,
        sequence_length=4,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )
    with pytest.raises(TypeError, match="shared encoder contract"):
        require_encoder_model(transformer)


def test_factory_metadata_records_encoder_contract() -> None:
    adapter = _raw_adapter()
    assert adapter.model_metadata["encoder_api_version"] == ENCODER_API_VERSION
    assert adapter.model_metadata["latent_dim"] == adapter.model.latent_dim
