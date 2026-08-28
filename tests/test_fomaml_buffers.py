from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import pytest
import torch
from torch import nn

from bench.meta import (
    BufferPolicy,
    FunctionalStateError,
    FirstOrderMAML,
    FOMAMLConfig,
    architecture_schema_signature,
    batchnorm_inventory,
    create_functional_state,
    functional_forward,
    model_state_hash,
    validate_functional_state,
)
from cogstate.model_zoo.DL.eegnet import TorchEEGNetClassifier
from cogstate.model_zoo.DL.shallow_convnet import TorchShallowConvNetClassifier


class TinyBatchNormClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(0.9)
        self.classifier = nn.Linear(3, 2)
        self._latent_dim = 3

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def get_output_head(self) -> nn.Linear:
        return self.classifier

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.normalization(features)))


def _mini_models() -> list[nn.Module]:
    return [
        TorchEEGNetClassifier(
            4, 128, 5, temporal_kernel_samples=16,
            separable_kernel_samples=8, f1=2, depth_multiplier=2,
            f2=4, pool1=2, pool2=2, dropout=0.1,
        ),
        TorchShallowConvNetClassifier(
            4, 128, 5, n_filters=4, temporal_kernel_samples=9,
            pool_size=15, pool_stride=5, dropout=0.1,
        ),
    ]


def test_functional_state_is_independent_and_schema_checked() -> None:
    model = TinyBatchNormClassifier()
    state = create_functional_state(model, "frozen_global")
    assert state.buffer_policy is BufferPolicy.FROZEN_GLOBAL
    assert all(
        state.parameters[name].data_ptr() != value.data_ptr()
        for name, value in model.named_parameters()
    )
    assert all(
        state.buffers[name].data_ptr() != value.data_ptr()
        for name, value in model.named_buffers()
    )

    missing = OrderedDict(state.buffers)
    missing.pop(next(iter(missing)))
    with pytest.raises(FunctionalStateError, match="missing"):
        validate_functional_state(model, replace(state, buffers=missing))
    extra = OrderedDict(state.buffers)
    extra["unexpected"] = torch.zeros(1)
    with pytest.raises(FunctionalStateError, match="extra"):
        validate_functional_state(model, replace(state, buffers=extra))
    shaped = OrderedDict(state.buffers)
    key = next(iter(shaped))
    shaped[key] = torch.zeros(4)
    with pytest.raises(FunctionalStateError, match="buffer shape mismatch"):
        validate_functional_state(model, replace(state, buffers=shaped))
    parameters = OrderedDict(state.parameters)
    key = next(iter(parameters))
    parameters[key] = torch.zeros(parameters[key].numel() + 1)
    with pytest.raises(FunctionalStateError, match="parameter shape mismatch"):
        validate_functional_state(model, replace(state, parameters=parameters))


def test_frozen_global_is_deterministic_and_never_updates_statistics() -> None:
    model = TinyBatchNormClassifier()
    before = model_state_hash(model)
    state = create_functional_state(model, "frozen_global")
    buffer_snapshot = {name: value.clone() for name, value in state.buffers.items()}
    features = torch.randn(4, 3)
    first, audit = functional_forward(model, state, features, phase="support")
    second, _ = functional_forward(model, state, features, phase="support")
    assert torch.equal(first, second)
    assert not audit.batchnorm_training
    assert not audit.dropout_active
    assert not audit.buffers_changed
    assert all(torch.equal(buffer_snapshot[name], value) for name, value in state.buffers.items())
    assert model_state_hash(model) == before


def test_support_local_updates_only_episode_buffers_and_query_cannot_update() -> None:
    model = TinyBatchNormClassifier()
    before = model_state_hash(model)
    state = create_functional_state(model, "support_local")
    initial = {name: value.clone() for name, value in state.buffers.items()}
    _, support_audit = functional_forward(
        model, state, torch.randn(4, 3) + 2.0, phase="support"
    )
    assert support_audit.batchnorm_training
    assert support_audit.buffers_changed
    assert any(not torch.equal(initial[name], value) for name, value in state.buffers.items())
    support_hash = {name: value.clone() for name, value in state.buffers.items()}
    _, query_audit = functional_forward(
        model, state, torch.randn(4, 3) - 50.0, phase="query"
    )
    assert not query_audit.buffers_changed
    assert all(torch.equal(support_hash[name], value) for name, value in state.buffers.items())
    assert model_state_hash(model) == before


def test_support_local_rejects_single_sample_and_support_controls_buffers() -> None:
    model = TinyBatchNormClassifier()
    with pytest.raises(FunctionalStateError, match="at least two"):
        functional_forward(
            model,
            create_functional_state(model, "support_local"),
            torch.randn(1, 3),
            phase="support",
        )
    first = create_functional_state(model, "support_local")
    second = create_functional_state(model, "support_local")
    functional_forward(model, first, torch.randn(4, 3), phase="support")
    functional_forward(model, second, torch.randn(4, 3) + 10.0, phase="support")
    assert any(
        not torch.equal(first.buffers[name], second.buffers[name])
        for name in first.buffers
    )


@pytest.mark.parametrize("policy", ["frozen_global", "support_local"])
def test_production_model_shapes_one_step_and_original_state(policy: str) -> None:
    torch.manual_seed(42)
    features = torch.randn(4, 1, 4, 128)
    labels = torch.tensor([0, 1, 2, 3])
    for model in _mini_models():
        model.eval()
        before = model_state_hash(model)
        encoded = model.encode(features)
        assert encoded.shape == (4, model.latent_dim)
        assert model.get_output_head().in_features == model.latent_dim
        signature = architecture_schema_signature(model)
        assert signature == architecture_schema_signature(model)
        learner = FirstOrderMAML(
            model,
            FOMAMLConfig(
                inner_steps=1,
                buffer_policy=policy,
                inner_learning_rate=0.01,
                meta_learning_rate=0.001,
            ),
        )
        adapted = learner.adapt(model, (features, labels))
        loss, accuracy, gradients = learner.evaluate(adapted, (features * -1, labels))
        assert torch.isfinite(torch.tensor(loss))
        assert 0.0 <= accuracy <= 1.0
        assert all(torch.isfinite(value).all() for value in gradients.values())
        assert model(features).shape == (4, 5)
        assert model_state_hash(model) == before


def test_architecture_signatures_distinguish_configurations_and_bn_inventory() -> None:
    eegnet, shallow = _mini_models()
    larger_eegnet = TorchEEGNetClassifier(
        4, 128, 5, temporal_kernel_samples=16,
        separable_kernel_samples=8, f1=4, depth_multiplier=2,
        f2=8, pool1=2, pool2=2, dropout=0.1,
    )
    assert architecture_schema_signature(eegnet) != architecture_schema_signature(larger_eegnet)
    eeg_inventory = batchnorm_inventory(eegnet)
    shallow_inventory = batchnorm_inventory(shallow)
    assert len(eeg_inventory) == 3
    assert len(shallow_inventory) == 1
    for row in eeg_inventory + shallow_inventory:
        assert row["running_mean"]
        assert row["running_var"]
        assert row["num_batches_tracked"]
        assert row["minimum_support_batch_size"] == 2
