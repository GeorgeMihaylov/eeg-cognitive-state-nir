from __future__ import annotations

import pytest

from bench.automl.search_space import SearchSpaceSpec


def _space() -> SearchSpaceSpec:
    return SearchSpaceSpec.from_dict({
        "model.params.d_model": {
            "type": "categorical",
            "choices": [64, 128],
        },
        "model.params.nhead": {
            "type": "categorical",
            "choices": [2, 4],
        },
        "model.params.num_layers": {"type": "integer", "low": 1, "high": 3},
        "model.params.dim_feedforward": {
            "type": "categorical",
            "choices": [128, 256],
        },
        "model.params.dropout": {"type": "float", "low": 0.05, "high": 0.4},
        "training.learning_rate": {
            "type": "log_float",
            "low": 0.00005,
            "high": 0.002,
        },
        "training.weight_decay": {
            "type": "log_float",
            "low": 0.000001,
            "high": 0.01,
        },
        "training.batch_size": {
            "type": "categorical",
            "choices": [64, 128],
        },
    })


def _parameters() -> dict[str, object]:
    return {
        "model.params.d_model": 128,
        "model.params.nhead": 4,
        "model.params.num_layers": 2,
        "model.params.dim_feedforward": 256,
        "model.params.dropout": 0.2,
        "training.learning_rate": 0.0005,
        "training.weight_decay": 0.0001,
        "training.batch_size": 128,
    }


def test_search_space_serializes_and_hashes_deterministically() -> None:
    first = _space()
    second = SearchSpaceSpec.from_dict(
        first.to_dict()["parameters"],
        constraints=tuple(first.to_dict()["constraints"]),
    )
    assert first.to_dict() == second.to_dict()
    assert first.config_hash() == second.config_hash()


def test_unknown_parameter_path_is_rejected() -> None:
    values = _parameters()
    values["training.momentum"] = 0.9
    with pytest.raises(ValueError, match="Unknown search parameter"):
        _space().validate_parameters(values)


def test_invalid_transformer_shape_is_rejected_before_model_build() -> None:
    space = SearchSpaceSpec.from_dict({
        "model.params.d_model": {
            "type": "categorical",
            "choices": [65],
        },
        "model.params.nhead": {
            "type": "categorical",
            "choices": [4],
        },
    })
    with pytest.raises(ValueError, match="divisible"):
        space.validate_parameters({
            "model.params.d_model": 65,
            "model.params.nhead": 4,
        })


def test_numeric_range_is_validated() -> None:
    values = _parameters()
    values["model.params.dropout"] = 0.8
    with pytest.raises(ValueError, match="outside"):
        _space().validate_parameters(values)
