from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from bench.experiments.fomaml_label_q5_diagnostic import (
    apply_decision_rule,
    audit_raw_episode_alignment,
    paired_subject_comparison,
    prepare_preregistration,
    select_buffer_policy,
    validate_diagnostic_config,
    validate_episode_protocol,
)
from bench.meta import FOMAMLConfig, FirstOrderMAML, model_state_hash
from bench.meta.production import audit_architectures


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/meta_learning/fomaml_label_q5_diagnostic.json"
PROTOCOL_PATH = ROOT / "benchmark_results/meta_learning_fomaml_production_contract/meta_validation_protocol.json"
EPISODE_PATH = ROOT / "benchmark_results/meta_learning_fomaml_production_contract/meta_validation_episode_index.parquet"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _protocol() -> tuple[dict, pd.DataFrame]:
    return (
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")),
        pd.read_parquet(EPISODE_PATH),
    )


def test_config_is_one_fold_one_seed_one_model_and_explicitly_enabled() -> None:
    config = _config()
    validate_diagnostic_config(config)
    assert config["execution_enabled"] is True
    assert config["seed"] == 42
    assert config["protocol"]["outer_fold"] == 1
    assert config["model"]["name"] == "torch_eegnet"
    assert config["fomaml"]["inner_steps"] == 1
    assert config["protocol"]["support_budget"] == 32


def test_preregistration_is_written_once_and_then_immutable(tmp_path: Path) -> None:
    path = tmp_path / "experiment_preregistration.json"
    payload = {"seed": 42, "outer_fold": 1, "execution_enabled": True}
    digest = prepare_preregistration(path, payload)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert prepare_preregistration(path, payload) == digest
    with pytest.raises(RuntimeError, match="differs"):
        prepare_preregistration(path, {**payload, "seed": 7})


def test_task8f_protocol_is_reused_without_subject_sample_or_record_leakage() -> None:
    protocol, episodes = _protocol()
    before = hashlib.sha256(EPISODE_PATH.read_bytes()).hexdigest()
    audit = validate_episode_protocol(
        protocol,
        episodes,
        expected_hash=_config()["protocol"]["expected_hash"],
        support_budget=32,
        query_budget=64,
    )
    assert audit["valid"]
    assert audit["scope_counts"] == {
        "meta_train": 23, "meta_validation": 9, "outer_test": 8
    }
    assert audit["subject_overlap"] == 0
    assert audit["support_query_sample_overlap"] == 0
    assert audit["support_query_record_overlap"] == 0
    assert not audit["unsafe_fallback_used"]
    assert hashlib.sha256(EPISODE_PATH.read_bytes()).hexdigest() == before


def test_raw_alignment_fails_without_remapping_or_episode_rebuild() -> None:
    metadata = pd.DataFrame({
        "sample_id": ["1", "2", "3"],
        "subject_id": ["s1", "s1", "s1"],
        "label_q5": [0, 1, 2],
    })
    episodes = pd.DataFrame([{
        "scope": "meta_train",
        "subject_id": "s1",
        "episode_id": "episode-1",
        "support_sample_ids": np.asarray(["1", "2"]),
        "query_sample_ids": np.asarray(["3", "4"]),
        "support_targets": np.asarray([0, 1]),
        "query_targets": np.asarray([2, 3]),
    }])
    audit = audit_raw_episode_alignment(metadata, episodes)
    assert not audit["valid"]
    assert audit["missing_episode_samples"] == 1
    assert audit["semantic_mismatch_count_for_present_ids"] == 0
    assert not audit["safe_remapping_applied"]
    assert not audit["episode_protocol_rebuilt"]


def test_protocol_hash_and_production_architecture_signature_match() -> None:
    config = _config()
    protocol, _ = _protocol()
    assert protocol["protocol_hash"] == config["protocol"]["expected_hash"]
    production = json.loads(
        (ROOT / "experiments/meta_learning/fomaml_production_contract.json")
        .read_text(encoding="utf-8")
    )
    rows, _, _ = audit_architectures(production, repository_root=ROOT)
    eegnet = next(row for row in rows if row["model_id"] == "torch_eegnet:canonical")
    assert eegnet["architecture_signature"] == config["model"]["architecture_signature"]
    assert eegnet["latent_dim"] == 1280
    assert eegnet["parameter_count"] == 8501


def test_skipped_participants_and_reasons_are_preserved() -> None:
    errors = pd.read_csv(
        ROOT / "benchmark_results/meta_learning_fomaml_production_contract/errors.csv"
    )
    assert len(errors) == 14
    assert errors["entity_id"].astype(str).nunique() == 14
    assert (errors["message"] == "requires at least 2 records").sum() == 13
    assert (errors["message"] == "support requires 32 samples, found 3").sum() == 1


def test_policy_selection_uses_meta_validation_and_frozen_tie_break() -> None:
    tied = select_buffer_policy(
        {
            "frozen_global": {"macro_f1": 0.300, "balanced_accuracy": 0.31},
            "support_local": {"macro_f1": 0.304, "balanced_accuracy": 0.40},
        },
        tie_threshold=0.005,
    )
    assert tied["selected_policy"] == "frozen_global"
    assert not tied["outer_test_used"]
    better = select_buffer_policy(
        {
            "frozen_global": {"macro_f1": 0.300, "balanced_accuracy": 0.31},
            "support_local": {"macro_f1": 0.306, "balanced_accuracy": 0.32},
        },
        tie_threshold=0.005,
    )
    assert better["selected_policy"] == "support_local"


class _TinyBN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.head = nn.Linear(3, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.bn(features))


@pytest.mark.parametrize("policy", ["frozen_global", "support_local"])
def test_adapted_prediction_never_updates_query_buffers_or_base(policy: str) -> None:
    model = _TinyBN()
    before = model_state_hash(model)
    learner = FirstOrderMAML(
        model,
        FOMAMLConfig(
            inner_steps=1,
            inner_learning_rate=0.01,
            buffer_policy=policy,
        ),
    )
    support = torch.randn(4, 3)
    adapted = learner.adapt(model, (support, torch.tensor([0, 1, 0, 1])))
    buffer_before = {
        name: value.clone() for name, value in adapted.buffers.items()
    }
    logits, audit = learner.predict_adapted(adapted, torch.randn(3, 3))
    probabilities = torch.softmax(logits, dim=1)
    assert logits.shape == (3, 2)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-6)
    assert not audit.buffers_changed
    assert all(torch.equal(buffer_before[name], value) for name, value in adapted.buffers.items())
    assert model_state_hash(model) == before


def _subject_metrics() -> pd.DataFrame:
    rows = []
    for subject, base, candidate in (("s1", 0.2, 0.3), ("s2", 0.4, 0.35)):
        rows.extend([
            {
                "subject_id": subject, "mode": "supervised_full_model",
                "macro_f1": base, "balanced_accuracy": base,
                "ordinal_mae": 1.0,
            },
            {
                "subject_id": subject, "mode": "selected_fomaml",
                "macro_f1": candidate, "balanced_accuracy": candidate,
                "ordinal_mae": 0.8,
            },
        ])
    return pd.DataFrame(rows)


def test_paired_comparison_uses_subjects_not_windows() -> None:
    comparison = paired_subject_comparison(
        _subject_metrics(), "selected_fomaml", "supervised_full_model"
    )
    assert comparison["n_subjects"] == 2
    assert comparison["comparison_unit"] == "subject"
    assert comparison["macro_f1_wins"] == 1
    assert comparison["macro_f1_losses"] == 1


@pytest.mark.parametrize(
    ("macro", "balanced", "wins", "expected"),
    [
        (0.021, 0.011, 6, "strong_proceed"),
        (0.011, -0.004, 5, "proceed"),
        (0.005, 0.001, 4, "inconclusive"),
        (-0.01, -0.01, 4, "do_not_proceed"),
        (0.01, 0.01, 2, "do_not_proceed"),
    ],
)
def test_decision_rule_is_deterministic(
    macro: float, balanced: float, wins: int, expected: str
) -> None:
    rule = _config()["decision_rule"]
    comparison = {
        "mean_delta_macro_f1": macro,
        "mean_delta_balanced_accuracy": balanced,
        "macro_f1_wins": wins,
    }
    assert apply_decision_rule(comparison, rule)["status"] == expected
    assert apply_decision_rule(comparison, rule) == apply_decision_rule(comparison, rule)


def test_config_contains_no_absolute_paths_and_fair_adaptation_contract() -> None:
    config = _config()
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert "F:\\" not in text and "C:\\Users" not in text
    assert config["protocol"]["training_sample_contract"] == (
        "union_of_materialized_meta_train_support_and_query_ids"
    )
    assert config["fomaml"]["inner_steps"] == 1
    assert config["fomaml"]["inner_learning_rate"] == 0.01
    assert config["protocol"]["support_budget"] == 32
