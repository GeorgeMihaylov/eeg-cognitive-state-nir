from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from model_zoo import build_model


LEGACY_RUN = Path(
    "benchmark_results/groupkfold_torch_transformer_label_q5/20260716_191246"
)


def _legacy_checkpoint() -> Path:
    matches = sorted(LEGACY_RUN.rglob("fold_01/model.pt"))
    if not matches:
        pytest.skip("Published categorical Transformer checkpoint is unavailable")
    return matches[0]


def _legacy_adapter():
    config = yaml.safe_load(
        Path("configs/groupkfold_torch_transformer_label_q5.yaml").read_text(
            encoding="utf-8"
        )
    )
    parameters = dict(config["models"]["torch_transformer"]["params"])
    assert "head_type" not in parameters
    parameters["device"] = "cpu"
    return build_model(
        "torch_transformer",
        "classification",
        input_shape=(8, 448),
        num_outputs=5,
        params=parameters,
    )


def test_real_legacy_checkpoint_loads_strict_and_keeps_classifier_keys() -> None:
    adapter = _legacy_adapter()
    adapter.load(_legacy_checkpoint())
    keys = set(adapter.model.state_dict())
    expected = {
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.1.weight",
        "classifier.1.bias",
        "classifier.4.weight",
        "classifier.4.bias",
    }
    assert expected <= keys
    assert adapter.head_type == "categorical"
    assert not any(key.startswith("ordinal_head.") for key in keys)


def test_refactored_forward_matches_manual_legacy_computation() -> None:
    adapter = _legacy_adapter()
    adapter.load(_legacy_checkpoint())
    model = adapter.model.eval()
    torch.manual_seed(42)
    features = torch.randn(2, 8, 448)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    with torch.no_grad():
        encoded = model.input_projection(features)
        encoded = model.positional_encoding(encoded)
        encoded = model.encoder(encoded, src_key_padding_mask=mask)
        pooled = model._last_valid(encoded, mask)
        legacy_output = model.classifier(pooled)
        refactored_output = model(features)
        encoded_output = model.encode(features)
    torch.testing.assert_close(refactored_output, legacy_output, atol=0, rtol=0)
    torch.testing.assert_close(encoded_output, pooled, atol=0, rtol=0)


def test_legacy_checkpoint_probabilities_are_softmax_of_same_logits() -> None:
    adapter = _legacy_adapter()
    adapter.load(_legacy_checkpoint())
    features = torch.randn(3, 8, 448)
    transformed = adapter._transform_features(features.numpy())
    with torch.no_grad():
        logits = adapter.model(torch.from_numpy(transformed))
        expected = torch.softmax(logits, dim=1).numpy()
    detailed = adapter.predict_detailed(features.numpy())
    torch.testing.assert_close(
        torch.from_numpy(detailed["class_probabilities"]),
        torch.from_numpy(expected),
        atol=0,
        rtol=0,
    )
