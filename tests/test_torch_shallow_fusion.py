from __future__ import annotations

import numpy as np
import pytest
import torch

from cogstate.model_zoo.DL.shallow_fusion import (
    TorchShallowFusionAdapter,
    TorchShallowFusionClassifier,
)
from cogstate.model_zoo.factory import build_model


def test_shallow_fusion_branch_shapes_and_output() -> None:
    model = TorchShallowFusionClassifier(
        eeg_input_shape=(1, 4, 2560),
        peripheral_dim=28,
        num_classes=3,
        peripheral_hidden_dim=32,
    )
    packed = torch.randn(2, 4 * 2560 + 28)
    eeg, peripheral = model.split_inputs(packed)
    assert eeg.shape == (2, 1, 4, 2560)
    assert peripheral.shape == (2, 28)
    assert model.eeg_encoder.encode(eeg).shape == (2, 40)
    assert model.peripheral_encoder(peripheral).shape == (2, 32)
    assert model(packed).shape == (2, 3)
    assert model.latent_dim == 72


def test_factory_builds_shallow_fusion_with_shared_adapter() -> None:
    adapter = build_model(
        "torch_shallow_fusion",
        "classification",
        (10268,),
        3,
        {
            "eeg_input_shape": [1, 4, 2560],
            "peripheral_dim": 28,
            "batch_size": 2,
            "max_epochs": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )
    assert adapter.input_shape == (10268,)
    assert isinstance(adapter, TorchShallowFusionAdapter)
    assert adapter.model_metadata["eeg_input_shape"] == [1, 4, 2560]
    assert adapter.model_metadata["peripheral_dim"] == 28
    with torch.no_grad():
        assert adapter.model(torch.zeros(2, 10268)).shape == (2, 3)


def test_shallow_fusion_rejects_bad_shape_and_nonfinite_values() -> None:
    model = TorchShallowFusionClassifier(
        eeg_input_shape=(1, 4, 2560), peripheral_dim=28, num_classes=3
    )
    with pytest.raises(ValueError, match="expects"):
        model(torch.zeros(2, 100))
    invalid = np.zeros((2, 10268), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model(torch.from_numpy(invalid))


def test_peripheral_features_never_enter_eeg_branch() -> None:
    model = TorchShallowFusionClassifier(
        eeg_input_shape=(1, 4, 2560), peripheral_dim=28, num_classes=3
    )
    packed = torch.zeros(1, 10268)
    packed[:, -28:] = 7.0
    eeg, peripheral = model.split_inputs(packed)
    assert torch.count_nonzero(eeg) == 0
    assert torch.all(peripheral == 7.0)


def test_branch_aware_normalization_uses_channel_and_peripheral_statistics() -> None:
    adapter = build_model(
        "torch_shallow_fusion",
        "classification",
        (10268,),
        3,
        {
            "eeg_input_shape": [1, 4, 2560],
            "peripheral_dim": 28,
            "batch_size": 2,
            "max_epochs": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )
    eeg = np.arange(4 * 4 * 2560, dtype=np.float32).reshape(4, 1, 4, 2560)
    peripheral = np.arange(4 * 28, dtype=np.float32).reshape(4, 28)
    packed = np.column_stack((eeg.reshape(4, -1), peripheral))
    adapter._fit_standardizer(packed[:3])
    transformed = adapter._transform_features(packed[:3])
    transformed_eeg = transformed[:, :10240].reshape(3, 1, 4, 2560)
    transformed_peripheral = transformed[:, 10240:]
    np.testing.assert_allclose(
        transformed_eeg.mean(axis=(0, 1, 3)), np.zeros(4), atol=1e-6
    )
    np.testing.assert_allclose(
        transformed_eeg.std(axis=(0, 1, 3)), np.ones(4), atol=1e-6
    )
    np.testing.assert_allclose(transformed_peripheral.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(transformed_peripheral.std(axis=0), 1.0, atol=1e-6)


def test_shallow_fusion_synthetic_fit_predict_is_group_aware() -> None:
    rng = np.random.default_rng(42)
    adapter = build_model(
        "torch_shallow_fusion",
        "classification",
        (10268,),
        3,
        {
            "eeg_input_shape": [1, 4, 2560],
            "peripheral_dim": 28,
            "batch_size": 6,
            "max_epochs": 1,
            "validation_size": 0.34,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )
    values = rng.normal(size=(18, 10268)).astype(np.float32)
    labels = np.tile(np.arange(3, dtype=np.int64), 6)
    groups = np.repeat([f"sub-{index}" for index in range(6)], 3)
    records = np.repeat([f"record-{index}" for index in range(6)], 3)
    adapter.set_validation_groups(
        groups,
        subject_ids=groups,
        record_ids=records,
        outer_test_group_ids=np.asarray(["outer-test"]),
        outer_test_record_ids=np.asarray(["outer-record"]),
        strategy="group_holdout",
        group_column="participant_id",
    )
    adapter.fit(values, labels)
    probabilities = adapter.predict_proba(values[:4])
    assert probabilities.shape == (4, 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert adapter.validation_split_["strategy"] == "group_holdout"
    assert not set(adapter.validation_split_["inner_train_group_ids"]) & set(
        adapter.validation_split_["inner_validation_group_ids"]
    )
