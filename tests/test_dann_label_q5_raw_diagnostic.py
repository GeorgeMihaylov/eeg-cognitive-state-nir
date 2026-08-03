from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bench.experiments.dann_label_q5_raw_diagnostic import (
    TargetTestLock,
    apply_dann_decision_rule,
    batch_plan_hash,
    deterministic_batch_plan,
    enforce_target_batch_firewall,
    logistic_grl_alpha,
    paired_dann_comparison,
    validate_dann_diagnostic_config,
)
from bench.experiments.fomaml_label_q5_diagnostic import _sha256_file
from model_zoo.DL.dann import DANNModule, DANNObjective
from model_zoo.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/domain_adaptation/dann_label_q5_raw_diagnostic.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_fixed_protocol_hashes_and_relative_config() -> None:
    config = _config()
    validate_dann_diagnostic_config(config)
    hashes = json.loads((ROOT / config["protocol"]["protocol_hash_file"]).read_text(encoding="utf-8"))
    assert hashes["protocol_hash"] == config["protocol"]["expected_protocol_hash"]
    assert hashes["primary_candidate_hash"] == config["protocol"]["expected_candidate_hash"]
    assert _sha256_file(ROOT / config["protocol"]["disabled_preregistration"]) == config["protocol"]["disabled_preregistration_sha256"]
    assert _sha256_file(ROOT / config["protocol"]["architecture_audit"]) == config["protocol"]["architecture_audit_sha256"]
    serialized = json.dumps(config)
    assert "F:\\" not in serialized and "C:\\" not in serialized


def test_schedule_is_fixed_and_finite() -> None:
    values = np.asarray([logistic_grl_alpha(value) for value in np.linspace(0, 1, 101)])
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(0.9999092043)
    assert np.isfinite(values).all()
    assert np.all(np.diff(values) >= 0)
    with pytest.raises(ValueError):
        logistic_grl_alpha(1.1)


def test_target_firewall_and_lock() -> None:
    safe = {"eeg": torch.zeros(2, 1, 14, 64), "domain_label": torch.zeros(2), "sample_id": ["a", "b"], "subject_id": ["s1", "s2"], "record_group_id": ["r1", "r2"]}
    enforce_target_batch_firewall(safe)
    with pytest.raises(RuntimeError, match="firewall"):
        enforce_target_batch_firewall({**safe, "task_label": torch.ones(2)})
    lock = TargetTestLock()
    with pytest.raises(RuntimeError, match="locked"):
        lock.require_access()
    assert lock.reads_before_unlock == 1
    lock.unlock("a" * 64)
    lock.require_access()
    assert lock.reads_after_unlock == 1


def test_matched_batch_plan_and_update_budget() -> None:
    ids = [f"sample-{index}" for index in range(17)]
    left = deterministic_batch_plan(17, 4, 19, 42)
    right = deterministic_batch_plan(17, 4, 19, 42)
    assert left == right and len(left) == 19
    assert batch_plan_hash(left, ids) == batch_plan_hash(right, ids)
    assert sum(len(batch) for batch in left) > len(ids)


def test_objective_uses_source_task_and_both_domains_with_finite_gradients() -> None:
    params = {"sampling_rate": 64, "temporal_kernel_seconds": 0.125, "separable_kernel_seconds": 0.125, "f1": 2, "depth_multiplier": 1, "f2": 2, "pool1": 2, "pool2": 2, "dropout": 0.0, "device": "cpu", "standardize": False}
    adapter = build_model("torch_eegnet", "classification", (1, 3, 64), 5, params)
    module = DANNModule(adapter.model, n_domains=2, domain_hidden_dims=(8,), domain_dropout=0.0)
    source = torch.randn(4, 1, 3, 64)
    target = torch.randn(5, 1, 3, 64)
    source_labels = torch.tensor([0, 1, 2, 3])
    domains = torch.tensor([1] * 4 + [0] * 5)
    module.eval()
    before = {name: value.clone() for name, value in module.state_dict().items()}
    output = module(source, target, gradient_reversal_alpha=logistic_grl_alpha(0.5))
    assert not hasattr(output, "target_task_outputs")
    loss = DANNObjective(task_type="classification", lambda_domain=1.0)(output, source_labels, domains)
    loss.total_loss.backward()
    assert torch.isfinite(loss.total_loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    assert all(torch.equal(before[name], value) for name, value in module.state_dict().items())


def test_paired_metrics_and_decision_are_subject_level_and_deterministic() -> None:
    rows = []
    for subject in range(8):
        for mode, gain in (("source_only_matched", 0.0), ("dann", 0.02)):
            rows.append({"subject_id": f"s{subject}", "mode": mode, "macro_f1": 0.3 + gain, "balanced_accuracy": 0.31 + gain, "ordinal_mae": 1.2 - gain})
    frame = pd.DataFrame(rows)
    comparison = paired_dann_comparison(frame)
    assert comparison["n_subjects"] == 8
    assert comparison["macro_f1_wins"] == 8
    rule = _config()["decision_rule"]
    assert apply_dann_decision_rule(comparison, rule)["status"] == "strong_proceed"
    assert paired_dann_comparison(frame)["bootstrap_macro_f1_mean_95_ci"] == comparison["bootstrap_macro_f1_mean_95_ci"]


def test_real_protocol_partitions_have_no_ids_or_subject_overlap() -> None:
    config = _config()
    source = json.loads((ROOT / config["protocol"]["source_validation_manifest"]).read_text(encoding="utf-8"))
    target = json.loads((ROOT / config["protocol"]["target_unlabeled_manifest"]).read_text(encoding="utf-8"))
    test = json.loads((ROOT / config["protocol"]["target_test_reference"]).read_text(encoding="utf-8"))
    partitions = [source["source_task_train"], source["source_validation"], target, test]
    assert [part["samples"] for part in partitions] == [3753, 1456, 18555, 4973]
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            assert set(left["sample_ids"]).isdisjoint(right["sample_ids"])
            assert set(left["subject_ids"]).isdisjoint(right["subject_ids"])
