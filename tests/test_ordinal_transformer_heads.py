from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from bench.bench_runner import benchmark_config_hash
from cogstate.model_zoo import build_model
from cogstate.model_zoo.DL import TorchFeatureTransformerClassifier
from cogstate.model_zoo.DL.ordinal import CoralOrdinalHead, CornOrdinalHead


def _model(head_type: str = "categorical") -> TorchFeatureTransformerClassifier:
    return TorchFeatureTransformerClassifier(
        input_size=6,
        num_classes=5,
        sequence_length=8,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        head_type=head_type,
    )


def test_coral_head_has_four_logits_and_strict_cutpoints() -> None:
    head = CoralOrdinalHead(8, 5, dropout=0.0)
    logits = head(torch.randn(7, 8))
    cutpoints = head.cutpoints()
    assert logits.shape == (7, 4)
    assert cutpoints.shape == (4,)
    assert torch.all(cutpoints[1:] > cutpoints[:-1])
    torch.testing.assert_close(
        cutpoints.detach(),
        torch.tensor([-1.5, -0.5, 0.5, 1.5]),
        atol=1e-6,
        rtol=0,
    )


def test_coral_head_probabilities_are_monotone_by_construction() -> None:
    logits = CoralOrdinalHead(8, 5, dropout=0.0)(torch.randn(20, 8))
    probabilities = torch.sigmoid(logits)
    assert torch.all(probabilities[:, :-1] >= probabilities[:, 1:])


def test_coral_optimizer_step_changes_parameters_with_finite_gradients() -> None:
    torch.manual_seed(42)
    head = CoralOrdinalHead(8, 5, dropout=0.0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
    before = {name: value.detach().clone() for name, value in head.named_parameters()}
    logits = head(torch.randn(10, 8))
    loss = logits.square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )
    optimizer.step()
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in head.named_parameters()
    )


def test_corn_head_has_four_finite_logits_and_gradients() -> None:
    head = CornOrdinalHead(8, 5, dropout=0.0)
    logits = head(torch.randn(11, 8))
    assert logits.shape == (11, 4)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


@pytest.mark.parametrize(
    "head_type, width",
    [("categorical", 5), ("coral", 4), ("corn", 4)],
)
def test_transformer_forward_uses_selected_head(head_type: str, width: int) -> None:
    model = _model(head_type)
    outputs = model(torch.randn(4, 8, 6))
    assert outputs.shape == (4, width)
    assert model.encode(torch.randn(4, 8, 6)).shape == (4, 8)


def test_default_categorical_state_dict_keys_remain_legacy_compatible() -> None:
    keys = set(_model().state_dict())
    assert "classifier.0.weight" in keys
    assert "classifier.0.bias" in keys
    assert "classifier.1.weight" in keys
    assert "classifier.1.bias" in keys
    assert "classifier.4.weight" in keys
    assert "classifier.4.bias" in keys
    assert not any(key.startswith("ordinal_head.") for key in keys)


@pytest.mark.parametrize("head_type", ["categorical", "coral", "corn"])
def test_factory_builds_all_transformer_heads(head_type: str) -> None:
    adapter = build_model(
        "torch_transformer",
        "classification",
        input_shape=(8, 6),
        num_outputs=5,
        params={
            "head_type": head_type,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "max_epochs": 1,
            "device": "cpu",
        },
    )
    assert adapter.head_type == head_type
    assert adapter.model.head_type == head_type
    assert adapter.model_metadata["head_type"] == head_type
    assert adapter.model_metadata["num_thresholds"] == (
        None if head_type == "categorical" else 4
    )


def test_factory_defaults_to_categorical_and_rejects_bad_head() -> None:
    adapter = build_model(
        "torch_transformer",
        "classification",
        (8, 6),
        5,
        {"d_model": 8, "nhead": 2, "device": "cpu"},
    )
    assert adapter.head_type == "categorical"
    with pytest.raises(ValueError, match="Unsupported head_type"):
        build_model(
            "torch_transformer",
            "classification",
            (8, 6),
            5,
            {"head_type": "unknown", "device": "cpu"},
        )


def test_factory_rejects_mismatched_num_classes() -> None:
    with pytest.raises(ValueError, match="must match num_outputs"):
        build_model(
            "torch_transformer",
            "classification",
            (8, 6),
            5,
            {"head_type": "coral", "num_classes": 4, "device": "cpu"},
        )


def test_config_hash_changes_with_head_type() -> None:
    base = {
        "models": {
            "transformer": {
                "type": "torch_transformer",
                "params": {"head_type": "categorical", "d_model": 8},
            }
        }
    }
    coral = deepcopy(base)
    corn = deepcopy(base)
    coral["models"]["transformer"]["params"]["head_type"] = "coral"
    corn["models"]["transformer"]["params"]["head_type"] = "corn"
    hashes = {
        benchmark_config_hash(base),
        benchmark_config_hash(coral),
        benchmark_config_hash(corn),
    }
    assert len(hashes) == 3
