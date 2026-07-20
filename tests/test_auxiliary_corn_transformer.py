from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from bench.bench_runner import benchmark_config_hash
from model_zoo import build_model
from model_zoo.DL.ordinal import CategoricalCornOutput


def _adapter(weight: float = 0.5):
    return build_model(
        "torch_transformer",
        "classification",
        input_shape=(8, 6),
        num_outputs=5,
        params={
            "head_type": "categorical_corn",
            "auxiliary_weight": weight,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "batch_size": 64,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )


def test_factory_builds_two_head_transformer_with_expected_state_keys() -> None:
    adapter = _adapter()
    model = adapter.model
    outputs = model(torch.randn(4, 8, 6))
    assert isinstance(outputs, CategoricalCornOutput)
    assert outputs.categorical_logits.shape == (4, 5)
    assert outputs.ordinal_logits.shape == (4, 4)
    keys = set(model.state_dict())
    assert "classifier.0.weight" in keys
    assert "classifier.4.weight" in keys
    assert any(key.startswith("auxiliary_ordinal_head.") for key in keys)
    assert not any(key.startswith("ordinal_head.") for key in keys)


def test_two_head_forward_calls_encode_once(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _adapter().model
    calls = 0
    original = model.encode

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "encode", counted)
    model(torch.randn(3, 8, 6))
    assert calls == 1


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_invalid_auxiliary_weight_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="auxiliary_weight"):
        _adapter(bad)


def test_auxiliary_weight_is_rejected_for_single_heads() -> None:
    with pytest.raises(ValueError, match="only valid"):
        build_model(
            "torch_transformer",
            "classification",
            input_shape=(8, 6),
            num_outputs=5,
            params={
                "head_type": "categorical",
                "auxiliary_weight": 0.5,
                "device": "cpu",
            },
        )


def test_config_hash_changes_with_auxiliary_weight() -> None:
    base = {
        "models": {
            "transformer": {
                "type": "torch_transformer",
                "params": {
                    "head_type": "categorical_corn",
                    "auxiliary_weight": 0.25,
                },
            }
        }
    }
    changed = deepcopy(base)
    changed["models"]["transformer"]["params"]["auxiliary_weight"] = 0.5
    assert benchmark_config_hash(base) != benchmark_config_hash(changed)


def test_single_head_configs_without_auxiliary_weight_remain_supported() -> None:
    for head_type in ("categorical", "coral", "corn"):
        adapter = build_model(
            "torch_transformer",
            "classification",
            (8, 6),
            5,
            {
                "head_type": head_type,
                "d_model": 8,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 16,
                "device": "cpu",
            },
        )
        assert adapter.head_type == head_type


def test_metadata_describes_primary_and_auxiliary_semantics() -> None:
    adapter = _adapter(0.5)
    metadata = adapter.objective_handler.to_metadata()
    assert metadata["head_type"] == "categorical_corn"
    assert metadata["auxiliary_weight"] == 0.5
    assert metadata["early_stopping_monitor"] == "validation_categorical_loss"
    assert metadata["primary_prediction_rule"] == "argmax_softmax"
    assert metadata["auxiliary_prediction_rule"].startswith("count_")
