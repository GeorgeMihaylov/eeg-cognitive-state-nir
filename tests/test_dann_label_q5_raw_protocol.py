from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from bench.experiments.dann_label_q5_raw_protocol import (
    PROTOCOL_ID,
    build_dann_label_q5_raw_protocol,
    build_direction_candidate,
    select_primary_direction,
    validate_dann_raw_protocol_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "experiments/domain_adaptation/dann_label_q5_raw_protocol.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def built_protocol(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("dann-raw-protocol")
    result = build_dann_label_q5_raw_protocol(
        _config(), repository_root=ROOT, output_dir=output
    )
    return output, result


def test_config_is_disabled_canonical_and_portable() -> None:
    config = _config()
    validate_dann_raw_protocol_config(config)
    assert config["protocol_id"] == PROTOCOL_ID
    assert config["execution_enabled"] is False
    assert config["seed"] == 42
    assert config["outer_fold"] == 1
    assert config["domains"] == {"gpn_data": 0, "Old_EEG": 1}
    assert config["future_training"]["target_train_task_labels_accessible"] is False
    assert config["future_training"]["outer_test_selection_accessible"] is False
    assert config["architecture"]["device"] == "cpu"
    assert not re.search(r"[A-Za-z]:[\\/]", CONFIG_PATH.read_text(encoding="utf-8"))


def test_invalid_execution_domain_and_target_label_contracts_fail() -> None:
    for path, value in (
        (("execution_enabled",), True),
        (("domains",), {"gpn_data": 1, "Old_EEG": 0}),
        (("future_training", "target_train_task_labels_accessible"), True),
        (("future_training", "outer_test_selection_accessible"), True),
        (("architecture", "device"), "cuda"),
    ):
        config = _config()
        target = config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            validate_dann_raw_protocol_config(config)


def test_canonical_raw_universe_and_outer_fold_are_reused(built_protocol) -> None:
    output, result = built_protocol
    summary = result.summary
    universe = summary["raw_universe"]
    outer = json.loads((output / "outer_fold_audit.json").read_text())
    assert universe["raw_universe_hash"] == (
        "308fdd96523565417d0fc2f3b3bfdea74639224ce1ee44f5e87af6cc0b7e94cf"
    )
    assert universe["sample_count"] == 30_958
    assert universe["subject_count"] == 54
    assert universe["record_count"] == 86
    assert universe["logical_record_count"] == 86
    assert universe["channels"] == 14
    assert universe["samples_per_window"] == 2560
    assert universe["tensor_values_read_this_stage"] == 0
    assert universe["target_test_tensor_values_read_this_stage"] == 0
    assert outer["source_fold_assignments_sha256"] == (
        "41ec5a244e11b5dd4ff25faa7361f2bca302dd719612fea8cbc54a55b6ff3341"
    )
    assert outer["outer_train_count"] == 43
    assert outer["outer_test_count"] == 11
    assert outer["outer_subject_overlap"] == 0
    assert result.raw_inventory["source"].notna().all()
    assert set(result.raw_inventory["source"]) == {"gpn_data", "Old_EEG"}
    assert result.raw_inventory["sample_id"].is_unique


def test_domain_inventory_and_class_counts_cover_the_universe(built_protocol) -> None:
    _, result = built_protocol
    inventory = result.domain_inventory
    assert set(inventory["source"]) == {"gpn_data", "Old_EEG"}
    assert set(inventory["domain_id"]) == {0, 1}
    assert int(inventory["samples"].sum()) == 30_958
    assert int(inventory.loc[inventory["source"].eq("gpn_data"), "samples"].sum()) == 23_791
    assert int(inventory.loc[inventory["source"].eq("Old_EEG"), "samples"].sum()) == 7_167
    assert int(inventory[[f"class_{i}" for i in range(5)]].to_numpy().sum()) == 30_958


def test_participant_domain_matrix_identifies_the_single_shared_subject(
    built_protocol,
) -> None:
    _, result = built_protocol
    matrix = result.participant_domain_matrix
    source_counts = matrix.groupby("subject_id")["source"].nunique()
    assert source_counts[source_counts.gt(1)].index.tolist() == ["a02151ac"]
    assert matrix["samples"].sum() == 30_958
    assert set(matrix["outer_partition"]) == {"outer_train", "outer_test"}


def test_logical_deduplication_is_explicit_and_leakage_safe(built_protocol) -> None:
    output, result = built_protocol
    audit = pd.read_csv(output / "logical_record_domain_audit.csv")
    summary = result.summary["logical_deduplication"]
    assert len(audit) == 86
    assert int(audit["present_in_both_sources"].sum()) == 33
    assert audit["deduplication_reason"].astype(str).str.len().gt(0).all()
    assert summary["original_cross_source_logical_records"] == 33
    assert summary["retained_cross_source_logical_records"] == 0
    assert summary["duplicate_sample_ids"] == 0
    assert summary["duplicate_logical_window_keys"] == 0
    assert summary["one_source_per_logical_record"]


def test_both_directions_and_both_subject_policies_are_audited(
    built_protocol,
) -> None:
    _, result = built_protocol
    audit = result.direction_audit
    assert len(audit) == 4
    assert set(audit["direction_id"]) == {
        "gpn_data_to_Old_EEG",
        "Old_EEG_to_gpn_data",
    }
    assert set(audit["subject_policy"]) == {
        "allow_cross_domain_train_subjects",
        "strict_cross_domain_subject_disjoint",
    }
    allow = audit[audit["subject_policy"].eq("allow_cross_domain_train_subjects")]
    strict = audit[
        audit["subject_policy"].eq("strict_cross_domain_subject_disjoint")
    ]
    assert set(allow["cross_domain_training_subjects"]) == {1}
    assert set(strict["cross_domain_training_subjects"]) == {0}


def test_strict_policy_is_feasible_only_for_old_to_gpn(built_protocol) -> None:
    _, result = built_protocol
    audit = result.direction_audit
    strict = audit[
        audit["subject_policy"].eq("strict_cross_domain_subject_disjoint")
    ]
    by_direction = strict.set_index("direction_id")
    old_to_gpn = by_direction.loc["Old_EEG_to_gpn_data"]
    gpn_to_old = by_direction.loc["gpn_data_to_Old_EEG"]
    assert bool(old_to_gpn["feasible"])
    assert old_to_gpn["source_outer_train_subjects"] == 10
    assert old_to_gpn["target_outer_train_subjects"] == 33
    assert old_to_gpn["target_outer_test_subjects"] == 8
    assert not bool(gpn_to_old["feasible"])
    assert gpn_to_old["target_outer_test_subjects"] == 3


def test_primary_selection_uses_counts_only_and_is_deterministic(
    built_protocol,
) -> None:
    _, result = built_protocol
    strict = result.direction_audit.loc[
        result.direction_audit["subject_policy"].eq(
            "strict_cross_domain_subject_disjoint"
        )
    ].to_dict("records")
    selected = dict(select_primary_direction(strict))
    repeated = dict(select_primary_direction(copy.deepcopy(strict)))
    assert selected == repeated
    assert selected["direction_id"] == "Old_EEG_to_gpn_data"
    assert result.summary["primary_selection_uses_target_labels"] is False


def test_target_label_mutation_cannot_change_direction_selection(built_protocol) -> None:
    _, result = built_protocol
    raw = result.raw_inventory
    mutated = raw.copy()
    mutated.loc[mutated["source"].eq("gpn_data"), "label_q5"] = 0
    rows = []
    for direction in _config()["directions"]:
        _, row = build_direction_candidate(
            mutated,
            direction,
            policy="strict_cross_domain_subject_disjoint",
            config=_config(),
        )
        rows.append(row)
    assert select_primary_direction(rows)["direction_id"] == (
        result.summary["primary_direction"]["direction_id"]
    )


def test_source_validation_is_grouped_deterministic_and_source_only(
    built_protocol,
) -> None:
    output, result = built_protocol
    manifest = json.loads((output / "source_validation_manifest.json").read_text())
    train = manifest["source_task_train"]
    validation = manifest["source_validation"]
    assert len(train["subject_ids"]) == 7
    assert len(validation["subject_ids"]) == 3
    assert set(train["subject_ids"]).isdisjoint(validation["subject_ids"])
    assert set(train["sample_ids"]).isdisjoint(validation["sample_ids"])
    assert set(train["record_group_ids"]).isdisjoint(validation["record_group_ids"])
    assert result.summary["primary_direction"]["source_domain"] == "Old_EEG"


def test_target_train_is_unlabelled_and_outer_test_is_reference_only(
    built_protocol,
) -> None:
    output, _ = built_protocol
    target = json.loads((output / "target_unlabeled_manifest.json").read_text())
    outer = json.loads((output / "target_test_reference.json").read_text())
    assert target["task_labels_exposed"] is False
    assert "class_counts" not in target
    assert "label_q5" not in target
    assert outer["task_labels_exposed"] is False
    assert outer["selection_accessible"] is False
    assert "class_counts" not in outer
    assert "label_q5" not in outer
    assert set(target["sample_ids"]).isdisjoint(outer["sample_ids"])
    assert set(target["subject_ids"]).isdisjoint(outer["subject_ids"])
    assert set(target["record_group_ids"]).isdisjoint(outer["record_group_ids"])


def test_primary_overlap_audit_is_zero_where_required(built_protocol) -> None:
    _, result = built_protocol
    primary = result.summary["primary_direction"]
    overlaps = primary["overlaps"]
    assert all(value == 0 for value in overlaps.values())
    assert primary["overlap_safe"]


def test_batching_and_schedule_contracts_are_fixed(built_protocol) -> None:
    output, _ = built_protocol
    batching = json.loads((output / "batching_contract.json").read_text())
    schedule = json.loads((output / "schedule_contract.json").read_text())
    assert batching["source_batch_size"] == 32
    assert batching["target_batch_size"] == 32
    assert batching["smaller_loader_policy"] == "deterministic_cycle"
    assert batching["drop_last"] is False
    assert batching["class_weighting"] == "none"
    assert batching["domain_weighting"] == "constant_lambda_1.0"
    assert schedule["gradient_reversal"]["name"] == "logistic"
    assert "2/(1+exp(-10*p))-1" in schedule["gradient_reversal"]["formula"]
    assert schedule["domain_loss"] == {"name": "constant", "lambda_domain": 1.0}


def test_objective_and_checkpoint_contract_use_no_target_task_labels(
    built_protocol,
) -> None:
    output, _ = built_protocol
    objective = json.loads((output / "objective_audit.json").read_text())
    prereg = json.loads(
        (output / "preregistration/experiment_preregistration.json").read_text()
    )
    assert objective["source_task_loss_uses_source_labels_only"]
    assert objective["target_task_logits_in_objective"] is False
    assert objective["target_task_labels_in_objective"] is False
    assert objective["domain_loss_uses_source_and_target_domains"]
    assert objective["domain_labels"] == {"gpn_data": 0, "Old_EEG": 1}
    assert prereg["future_training"]["checkpoint_criterion"] == (
        "source_validation_macro_f1"
    )
    assert prereg["future_training"]["early_stopping_partition"] == (
        "source_validation"
    )


def test_cpu_forward_reuses_production_encoder_and_separate_heads(
    built_protocol,
) -> None:
    output, result = built_protocol
    audit = json.loads((output / "dann_architecture_audit.json").read_text())
    assert audit == result.summary["cpu_forward_audit"]
    assert audit["device"] == "cpu"
    assert audit["input_shape"] == [1, 14, 2560]
    assert audit["source_task_output_shape"] == [2, 5]
    assert audit["domain_output_shape"] == [5, 2]
    assert audit["source_latent_shape"] == [2, 1280]
    assert audit["target_latent_shape"] == [3, 1280]
    assert audit["task_domain_heads_separate"]
    assert audit["architecture_signature"] == (
        "248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7"
    )
    assert audit["source_task_loss_finite"]
    assert audit["domain_loss_finite"]
    assert audit["encoder_state_unchanged"]


def test_no_training_optimizer_backward_cuda_predictions_or_metrics(
    built_protocol,
) -> None:
    _, result = built_protocol
    summary = result.summary
    audit = summary["cpu_forward_audit"]
    assert summary["execution_enabled"] is False
    assert summary["training_performed"] is False
    assert summary["optimizer_created"] is False
    assert summary["backward_called"] is False
    assert summary["cuda_tensor_created"] is False
    assert summary["predictions_computed"] is False
    assert summary["metrics_computed"] is False
    assert summary["checkpoint_created"] is False
    assert audit["target_task_output_absent"]
    assert audit["training_batch_target_label_absent"]
    assert audit["all_parameter_gradients_absent"]


def test_protocol_hashes_and_preregistration_are_stable_and_bound_to_ids(
    built_protocol,
) -> None:
    output, result = built_protocol
    prereg = output / "preregistration/experiment_preregistration.json"
    first = result.summary["protocol_hash"]
    primary = result.summary["primary_direction"]
    assert len(first) == 64
    assert len(primary["candidate_protocol_hash"]) == 64
    assert result.summary["preregistration_hash"] == hashlib.sha256(
        prereg.read_bytes()
    ).hexdigest()
    raw = result.raw_inventory
    direction = next(
        row
        for row in _config()["directions"]
        if row["direction_id"] == "Old_EEG_to_gpn_data"
    )
    repeated, _ = build_direction_candidate(
        raw,
        direction,
        policy="strict_cross_domain_subject_disjoint",
        config=_config(),
    )
    modified_raw = raw.copy()
    source_index = modified_raw.index[modified_raw["source"].eq("Old_EEG")][0]
    modified_raw.loc[source_index, "sample_id"] += "-changed"
    modified, _ = build_direction_candidate(
        modified_raw,
        direction,
        policy="strict_cross_domain_subject_disjoint",
        config=_config(),
    )
    assert repeated["candidate_protocol_hash"] == primary[
        "candidate_protocol_hash"
    ]
    assert modified["candidate_protocol_hash"] != repeated[
        "candidate_protocol_hash"
    ]


def test_direction_and_subject_policy_change_candidate_hash(built_protocol) -> None:
    _, result = built_protocol
    audit = result.direction_audit
    assert audit["candidate_protocol_hash"].is_unique
    per_direction = audit.groupby("direction_id")["candidate_protocol_hash"].nunique()
    per_policy = audit.groupby("subject_policy")["candidate_protocol_hash"].nunique()
    assert (per_direction == 2).all()
    assert (per_policy == 2).all()


def test_tracked_and_runtime_protocol_metadata_have_no_absolute_paths(
    built_protocol,
) -> None:
    output, _ = built_protocol
    pattern = re.compile(r"[A-Za-z]:[\\/]")
    for path in output.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md"}:
            assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_required_artifacts_are_complete_and_error_free(built_protocol) -> None:
    output, result = built_protocol
    required = {
        "raw_universe_reference.json",
        "domain_inventory.csv",
        "subject_domain_matrix.csv",
        "logical_record_domain_audit.csv",
        "direction_audit.csv",
        "source_target_overlap_audit.json",
        "source_validation_manifest.json",
        "target_unlabeled_manifest.json",
        "target_test_reference.json",
        "batching_contract.json",
        "dann_architecture_audit.json",
        "objective_audit.json",
        "protocol_manifest.json",
        "protocol_hash.json",
        "experiment_preregistration.json",
        "preregistration_hash.json",
        "readiness_decision.json",
        "errors.csv",
        "protocol_report.md",
    }
    actual = {path.name for path in output.iterdir()}
    actual.update(path.name for path in (output / "preregistration").iterdir())
    assert required.issubset(actual)
    assert result.summary["status"] == "dann_protocol_ready"
    assert result.summary["errors"] == []
    assert result.summary["canonical_inputs_unchanged"]


def test_builder_is_deterministic_and_immutable_preregistration_reusable(
    built_protocol,
) -> None:
    output, first = built_protocol
    second = build_dann_label_q5_raw_protocol(
        _config(), repository_root=ROOT, output_dir=output
    )
    assert first.summary["protocol_hash"] == second.summary["protocol_hash"]
    assert first.summary["preregistration_hash"] == second.summary[
        "preregistration_hash"
    ]
    assert first.direction_audit.to_dict("records") == second.direction_audit.to_dict(
        "records"
    )
