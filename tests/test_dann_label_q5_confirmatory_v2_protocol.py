from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pandas as pd
import pytest
import torch

from bench.experiments.dann_label_q5_confirmatory_v2_protocol import (
    MODES,
    PRIMARY_SEEDS,
    PROTOCOL_ID,
    SECONDARY_SEEDS,
    aggregate_participant_deltas,
    apply_primary_decision_rule,
    build_dann_label_q5_confirmatory_v2_protocol,
    build_run_matrix,
    compute_confirmatory_v2_protocol_hash,
    refuse_observed_diagnostic_rerun,
    validate_confirmatory_v2_config,
)
from bench.experiments.dann_label_q5_raw_protocol import _sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments/domain_adaptation/dann_label_q5_confirmatory_v2_protocol.json"
)
V1_OUTPUT = ROOT / "benchmark_results/domain_adaptation_dann_confirmatory_protocol"
EXPECTED_FOLD_HASHES = {
    "1": "b8591f6a0ff5a8abc2f99a9358629117583e1c219662ddc444a99cba473a6041",
    "2": "c3062fa8f721d6c7796924d0b330d099e1d48f9b05c7fdd9211aefbf51882eca",
    "3": "adc7b66241efd12ed44e51e195c45d0ffde0ebf3ea53c11a6aeb6fbe1ba818e4",
    "4": "e6117577010d16271e2e1e6f29132bbcad5493c183766288ea921cc1f658891c",
    "5": "a11dbdf7654f13fd3a4db3bd644b4855b9a35ee01e1f931368783a541008711f",
}


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def built_v2(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("dann-confirmatory-v2")
    result = build_dann_label_q5_confirmatory_v2_protocol(
        _config(), repository_root=ROOT, output_dir=output
    )
    return output, result


def test_config_has_separate_primary_secondary_and_observed_groups() -> None:
    config = _config()
    validate_confirmatory_v2_config(config)
    assert config["protocol_id"] == PROTOCOL_ID
    assert config["execution_enabled"] is False
    assert config["rerun_observed_diagnostic"] is False
    assert tuple(config["primary_seeds"]) == PRIMARY_SEEDS == (123, 2026)
    assert tuple(config["secondary_seeds"]) == SECONDARY_SEEDS == (42,)
    assert tuple(config["modes"]) == MODES
    assert config["observed_diagnostic_cell"]["analysis_group"] == (
        "previously_observed_diagnostic"
    )
    assert not re.search(r"[A-Za-z]:[\\/]", CONFIG_PATH.read_text(encoding="utf-8"))


def test_run_matrices_have_exact_sizes_cells_and_modes() -> None:
    matrix = build_run_matrix(_config())
    primary = matrix[matrix["analysis_group"].eq("primary_confirmatory")]
    secondary = matrix[matrix["analysis_group"].eq("secondary_sensitivity")]
    completed = matrix[
        matrix["analysis_group"].eq("previously_observed_diagnostic")
    ]
    assert len(matrix) == 30
    assert len(primary) == 20
    assert len(secondary) == 8
    assert len(completed) == 2
    assert sorted(primary["fold"].unique()) == [1, 2, 3, 4, 5]
    assert sorted(primary["seed"].unique()) == [123, 2026]
    assert sorted(secondary["fold"].unique()) == [2, 3, 4, 5]
    assert secondary["seed"].unique().tolist() == [42]
    assert set(completed["execution_status"]) == {"already_completed"}
    assert set(completed["mode"]) == set(MODES)
    for _, group in primary.groupby(["fold", "seed"]):
        assert set(group["mode"]) == set(MODES)
    observed = matrix["fold"].eq(1) & matrix["seed"].eq(42)
    assert not observed.loc[primary.index].any()
    assert not observed.loc[secondary.index].any()
    assert observed.loc[completed.index].all()


def test_observed_diagnostic_rerun_is_rejected_by_default() -> None:
    with pytest.raises(RuntimeError, match="previously observed"):
        refuse_observed_diagnostic_rerun(1, 42)
    refuse_observed_diagnostic_rerun(1, 42, allow_technical_reproduction=True)
    refuse_observed_diagnostic_rerun(2, 42)
    preregistration = _config()
    assert "allow_technical_reproduction" not in preregistration


def test_v1_partitions_fold_hashes_and_firewall_are_unchanged(built_v2) -> None:
    output, result = built_v2
    summary = result.summary
    assert summary["confirmatory_v1_protocol_hash"] == (
        "a261d6081b4924af82752021fa24bbd50a75ed83ac3672db1e691709ad2cad71"
    )
    assert summary["confirmatory_v1_preregistration_hash"] == (
        "f4862dbf09d6eccd04438eebd5bbd99899dc4f1530ba3554c87a45a13dba59c4"
    )
    assert summary["outer_fold_hashes"] == EXPECTED_FOLD_HASHES
    expected_partition_hash = (
        "94f7c06fecb9122ba296e8a94ac3405fc743c90f470c90da10b7f6743b96feb6"
    )
    assert summary["fold_partitions_sha256"] == expected_partition_hash
    assert _sha256_file(output / "fold_partitions.parquet") == expected_partition_hash
    assert _sha256_file(V1_OUTPUT / "fold_subject_partitions.parquet") == expected_partition_hash
    assert sorted(result.fold_partitions["fold"].unique()) == [1, 2, 3, 4, 5]
    firewall = summary["target_label_firewall"]
    assert firewall["all_target_manifests_label_free"] is True
    assert firewall["target_train_task_labels_accessible"] is False
    assert firewall["target_test_tensor_values_read"] == 0


def test_hyperparameters_source_splits_and_budgets_match_task_8e(built_v2) -> None:
    _, result = built_v2
    summary = result.summary
    v1 = _load(V1_OUTPUT / "protocol_manifest.json")
    for key, value in summary["training"].items():
        if key in v1["training_contract"]:
            assert v1["training_contract"][key] == value
    assert summary["training"]["source_validation_split_seed"] == 42
    assert summary["training"]["source_validation_shared_across_model_seeds"] is True
    expected_steps = {
        1: (139, 580, 580), 2: (149, 602, 602), 3: (157, 569, 569),
        4: (143, 606, 606), 5: (150, 586, 586),
    }
    for row in summary["matched_update_budgets"]:
        assert (
            row["source_natural_steps"], row["target_natural_steps"],
            row["matched_steps_per_epoch"],
        ) == expected_steps[row["fold"]]
        source = _load(V1_OUTPUT / f"batching_contracts/fold_{row['fold']:02d}.json")
        assert row["matched_steps_per_epoch"] == source["matched_steps_per_epoch"]
    assert len(set(summary["source_validation_partition_hashes"].values())) == 5


def test_diagnostic_provenance_hashes_are_verified_and_unchanged(built_v2) -> None:
    output, result = built_v2
    reference = _load(output / "diagnostic_cell_reference.json")
    expected = {
        "source_only_checkpoint": "5b371b8da06088f0f386aa82c6848cd9d473e7845b2f1655339587acc72e11f3",
        "dann_checkpoint": "0fa4900e166c2ce2a6f51bbd3c79871409ba47d20df4e649146684d925e1ada2",
        "target_test_predictions": "00c7abbb4037973b7482d611a162f32ff9ba4c1914e6bfefa817bd275191e218",
        "participant_metrics": "dabb732aa01703500aac51450b9221b14d98cd9d4ee9306ff2c8f78a53203104",
        "diagnostic_summary": "fc52fe7f72f2c2d8b9d8ed6ec8ec2218d2dcbe47be18278fab9b13ea0143bd3d",
    }
    assert reference["reference_hashes"] == expected
    assert reference["fold"] == 1 and reference["seed"] == 42
    assert reference["rerun"] is False
    config_reference = _config()["diagnostic_reference"]
    for name, digest in expected.items():
        assert _sha256_file(ROOT / config_reference[name]) == digest
    assert result.summary["diagnostic_protocol_hash"] == (
        "7f5642109e1ed26dd6de96aa88fe0711bfa08e8f3a58422b17364301d693f7c5"
    )
    assert result.summary["diagnostic_preregistration_hash"] == (
        "f5e7cd962cc361b36e74b073c8532e2af5f4a94a36831f21531d6db36b54a817"
    )


def _synthetic_results(include_secondary: bool = False) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for seed in PRIMARY_SEEDS:
            for subject in (f"subject-{fold}-a", f"subject-{fold}-b"):
                for mode, gain in (("source_only_matched", 0.0), ("dann", 0.02)):
                    rows.append({
                        "analysis_group": "primary_confirmatory",
                        "fold": fold,
                        "seed": seed,
                        "subject_id": subject,
                        "mode": mode,
                        "macro_f1": 0.25 + gain,
                        "balanced_accuracy": 0.26 + gain,
                        "ordinal_mae": 1.2 - gain,
                        "windows": 1 if subject.endswith("a") else 10_000,
                    })
    if include_secondary:
        for mode, value in (("source_only_matched", 0.9), ("dann", 0.0)):
            rows.append({
                "analysis_group": "secondary_sensitivity",
                "fold": 2,
                "seed": 42,
                "subject_id": "secondary-only",
                "mode": mode,
                "macro_f1": value,
                "balanced_accuracy": value,
                "ordinal_mae": value,
                "windows": 1_000_000,
            })
    return pd.DataFrame(rows)


def test_primary_aggregation_filters_groups_averages_seeds_and_weights_people_equally() -> None:
    paired, participants = aggregate_participant_deltas(
        _synthetic_results(include_secondary=True)
    )
    assert set(paired["seed"]) == set(PRIMARY_SEEDS)
    assert "secondary-only" not in set(participants["subject_id"])
    assert participants["seed_count"].eq(2).all()
    assert participants["participant_weight"].eq(1.0).all()
    assert all(
        value == pytest.approx(0.02)
        for value in participants["delta_macro_f1"]
    )
    assert len(participants) == 10


def test_secondary_results_cannot_change_primary_decision() -> None:
    primary_paired, primary_participants = aggregate_participant_deltas(
        _synthetic_results(False)
    )
    combined_paired, combined_participants = aggregate_participant_deltas(
        _synthetic_results(True)
    )
    rule = _config()["primary_decision_rule"]
    before = apply_primary_decision_rule(primary_paired, primary_participants, rule)
    after = apply_primary_decision_rule(combined_paired, combined_participants, rule)
    assert before == after
    assert before["status"] == "confirmed"
    assert _config()["secondary_aggregation"]["may_change_primary_decision"] is False


def test_primary_decision_rule_is_deterministic_and_seed_aware() -> None:
    paired, participants = aggregate_participant_deltas(_synthetic_results())
    rule = _config()["primary_decision_rule"]
    first = apply_primary_decision_rule(paired, participants, rule)
    second = apply_primary_decision_rule(
        paired.sample(frac=1, random_state=7),
        participants.sample(frac=1, random_state=123),
        copy.deepcopy(rule),
    )
    assert first == second
    assert first["nonnegative_primary_seed_count"] == 2
    assert apply_primary_decision_rule(
        paired, participants, rule, protocol_valid=False
    )["status"] == "blocked"


def test_protocol_hash_is_deterministic_and_sensitive_to_seed_and_diagnostic_reference() -> None:
    matrix = build_run_matrix(_config()).to_dict("records")
    contract = {
        "primary_seeds": [123, 2026],
        "diagnostic_reference": {"protocol_hash": "diagnostic-a"},
    }
    first = compute_confirmatory_v2_protocol_hash(
        run_matrix=matrix, scientific_contract=contract
    )
    second = compute_confirmatory_v2_protocol_hash(
        run_matrix=copy.deepcopy(matrix), scientific_contract=copy.deepcopy(contract)
    )
    changed_seed = copy.deepcopy(matrix)
    for row in changed_seed:
        if row["seed"] == 2026:
            row["seed"] = 2027
            row["run_id"] = row["run_id"].replace("2026", "2027")
    changed_reference = copy.deepcopy(contract)
    changed_reference["diagnostic_reference"]["protocol_hash"] = "diagnostic-b"
    assert first == second
    assert first != compute_confirmatory_v2_protocol_hash(
        run_matrix=changed_seed, scientific_contract=contract
    )
    assert first != compute_confirmatory_v2_protocol_hash(
        run_matrix=matrix, scientific_contract=changed_reference
    )


def test_each_new_fold_seed_pair_has_a_separate_locked_unlock_contract(built_v2) -> None:
    output, result = built_v2
    paths = sorted((output / "target_test_unlock_contracts").glob("*.json"))
    assert len(paths) == 14
    pairs = set()
    for path in paths:
        lock = _load(path)
        pairs.add((lock["fold"], lock["seed"]))
        assert lock["status"] == "locked"
        assert lock["target_test_opened"] is False
        assert lock["diagnostic_unlock_reused"] is False
        assert lock["source_only_checkpoint_hash"] is None
        assert lock["dann_checkpoint_hash"] is None
        assert lock["protocol_hash"] == result.summary["protocol_hash"]
        assert lock["preregistration_hash"] == result.summary["preregistration_hash"]
    assert (1, 42) not in pairs
    assert {(fold, seed) for fold in range(1, 6) for seed in PRIMARY_SEEDS} <= pairs


def test_builder_does_not_load_arrays_train_backpropagate_or_use_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden execution path called")

    monkeypatch.setattr("numpy.load", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden)
    result = build_dann_label_q5_confirmatory_v2_protocol(
        _config(), repository_root=ROOT, output_dir=tmp_path / "guarded"
    )
    summary = result.summary
    assert summary["execution_enabled"] is False
    assert summary["optimizer_created"] is False
    assert summary["backward_called"] is False
    assert summary["training_performed"] is False
    assert summary["cuda_tensor_created"] is False
    assert summary["target_test_tensor_values_read"] == 0
    assert summary["target_test_inference_performed"] is False


def test_runtime_artifacts_are_complete_portable_and_ready(built_v2) -> None:
    output, result = built_v2
    required = {
        "protocol_reference.json", "diagnostic_cell_reference.json",
        "run_matrix.csv", "primary_run_matrix.csv", "secondary_run_matrix.csv",
        "completed_run_matrix.csv", "fold_partitions.parquet",
        "model_seed_manifest.json", "aggregation_contract.json",
        "primary_decision_rule.json", "secondary_sensitivity_rule.json",
        "protocol_manifest.json", "protocol_hash.json",
        "preregistration/experiment_preregistration.json",
        "preregistration/preregistration_hash.json", "readiness_decision.json",
        "errors.csv", "protocol_report.md",
    }
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*") if path.is_file()
    }
    assert required <= observed
    for path in output.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:[\\/]", text)
    preregistration = _load(output / "preregistration/experiment_preregistration.json")
    assert preregistration["execution_enabled"] is False
    assert preregistration["rerun_observed_diagnostic"] is False
    assert "allow_technical_reproduction" not in preregistration
    assert result.summary["status"] == "confirmatory_v2_protocol_ready"
    assert result.summary["primary_runs"] == 20
    assert result.summary["new_secondary_runs"] == 8
    assert result.summary["completed_diagnostic_results"] == 2
