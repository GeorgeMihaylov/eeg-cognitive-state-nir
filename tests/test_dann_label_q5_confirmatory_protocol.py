from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pandas as pd
import pytest
import torch

from bench.experiments.dann_label_q5_confirmatory_protocol import (
    EXPECTED_FOLDS,
    EXPECTED_MODEL_SEEDS,
    PROTOCOL_ID,
    build_dann_label_q5_confirmatory_protocol,
    compute_confirmatory_protocol_hash,
    domain_head_signature,
    validate_confirmatory_config,
)
from bench.experiments.dann_label_q5_raw_protocol import _sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments/domain_adaptation/dann_label_q5_confirmatory_protocol.json"
)
EXPECTED_RAW_HASH = (
    "308fdd96523565417d0fc2f3b3bfdea74639224ce1ee44f5e87af6cc0b7e94cf"
)
EXPECTED_OUTER_HASHES = {
    1: "b8591f6a0ff5a8abc2f99a9358629117583e1c219662ddc444a99cba473a6041",
    2: "c3062fa8f721d6c7796924d0b330d099e1d48f9b05c7fdd9211aefbf51882eca",
    3: "adc7b66241efd12ed44e51e195c45d0ffde0ebf3ea53c11a6aeb6fbe1ba818e4",
    4: "e6117577010d16271e2e1e6f29132bbcad5493c183766288ea921cc1f658891c",
    5: "a11dbdf7654f13fd3a4db3bd644b4855b9a35ee01e1f931368783a541008711f",
}


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def built_confirmatory_protocol(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("dann-confirmatory-protocol")
    result = build_dann_label_q5_confirmatory_protocol(
        _config(), repository_root=ROOT, output_dir=output
    )
    return output, result


def test_config_is_disabled_fixed_direction_and_portable() -> None:
    config = _config()
    validate_confirmatory_config(config)
    assert config["protocol_id"] == PROTOCOL_ID
    assert config["execution_enabled"] is False
    assert tuple(config["outer_folds"]) == EXPECTED_FOLDS
    assert tuple(config["model_seeds"]) == EXPECTED_MODEL_SEEDS
    assert config["direction"] == {
        "direction_id": "Old_EEG_to_gpn_data",
        "source_domain": "Old_EEG",
        "target_domain": "gpn_data",
        "subject_policy": "strict_cross_domain_subject_disjoint",
        "strict_shared_subject_rule": (
            "retain_in_source_loader_exclude_from_target_loader"
        ),
    }
    serialized = CONFIG_PATH.read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:[\\/]", serialized)
    assert "gpn_data_to_Old_EEG" not in serialized


def test_fixed_raw_universe_and_all_outer_fold_hashes(
    built_confirmatory_protocol,
) -> None:
    _, result = built_confirmatory_protocol
    summary = result.summary
    assert summary["raw_universe_hash"] == EXPECTED_RAW_HASH
    assert summary["outer_fold_artifact_sha256"] == (
        "41ec5a244e11b5dd4ff25faa7361f2bca302dd719612fea8cbc54a55b6ff3341"
    )
    assert {int(key): value for key, value in summary["outer_fold_hashes"].items()} == (
        EXPECTED_OUTER_HASHES
    )
    assert [row["fold"] for row in summary["folds"]] == list(EXPECTED_FOLDS)


def test_all_folds_are_eligible_with_expected_partition_counts(
    built_confirmatory_protocol,
) -> None:
    _, result = built_confirmatory_protocol
    expected = {
        1: (8, 4433, 2, 776, 33, 18555, 8, 4973),
        2: (8, 4749, 2, 776, 33, 19241, 9, 4550),
        3: (9, 4995, 3, 1456, 31, 18207, 10, 5321),
        4: (8, 4557, 2, 985, 34, 19377, 7, 4151),
        5: (8, 4790, 2, 1151, 33, 18732, 8, 4796),
    }
    assert result.summary["eligible_folds"] == list(EXPECTED_FOLDS)
    assert result.summary["excluded_folds"] == []
    for row in result.summary["folds"]:
        observed = (
            row["source_train_subjects"], row["source_train_samples"],
            row["source_validation_subjects"], row["source_validation_samples"],
            row["target_train_subjects"], row["target_train_samples"],
            row["target_test_subjects"], row["target_test_samples"],
        )
        assert observed == expected[row["fold"]]
        assert row["eligible"] is True
        assert set(json.loads(row["source_train_class_counts"])) == set("01234")
        assert set(json.loads(row["source_validation_class_counts"])) == set("01234")


def test_subject_sample_and_logical_record_partitions_are_disjoint(
    built_confirmatory_protocol,
) -> None:
    output, result = built_confirmatory_protocol
    overlaps = pd.read_csv(output / "logical_record_overlap_audit.csv")
    assert overlaps["passed"].all()
    assert (overlaps["overlap_count"] == 0).all()
    partitions = result.fold_partitions
    for fold, frame in partitions.groupby("fold"):
        parts = {
            row.partition: {
                "subjects": set(row.subject_ids),
                "samples": set(row.sample_ids),
                "records": set(row.record_group_ids),
            }
            for row in frame.itertuples()
        }
        source_subjects = (
            parts["source_task_train"]["subjects"]
            | parts["source_validation"]["subjects"]
        )
        assert source_subjects.isdisjoint(parts["target_train_unlabelled"]["subjects"])
        training_subjects = source_subjects | parts["target_train_unlabelled"]["subjects"]
        assert training_subjects.isdisjoint(
            parts["target_outer_test_reference"]["subjects"]
        )
        names = list(parts)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                assert parts[left]["samples"].isdisjoint(parts[right]["samples"]), fold
                assert parts[left]["records"].isdisjoint(parts[right]["records"]), fold


def test_shared_participant_policy_is_deterministic(
    built_confirmatory_protocol,
) -> None:
    _, result = built_confirmatory_protocol
    audit = result.shared_subject_audit.sort_values("fold")
    assert audit["subject_id"].unique().tolist() == ["a02151ac"]
    assert audit["target_metrics_used"].eq(False).all()
    assert audit["deterministic"].eq(True).all()
    assert audit.loc[audit["fold"].eq(2), "resolution"].item() == (
        "outer_test_excluded_from_all_training"
    )
    retained = audit.loc[audit["fold"].ne(2)]
    assert retained["resolution"].eq("retained_source_excluded_target").all()
    assert retained["present_in_target_train_after_policy"].eq(False).all()


def test_source_validation_seed_and_split_are_shared_across_model_seeds(
    built_confirmatory_protocol,
) -> None:
    output, _ = built_confirmatory_protocol
    hashes = []
    for fold in EXPECTED_FOLDS:
        manifest = _load(output / f"source_validation_manifests/fold_{fold:02d}.json")
        contract = _load(output / f"batching_contracts/fold_{fold:02d}.json")
        assert manifest["shared_across_model_seeds"] is True
        assert manifest["model_seeds"] == list(EXPECTED_MODEL_SEEDS)
        assert set(manifest["source_task_train"]["subject_ids"]).isdisjoint(
            manifest["source_validation"]["subject_ids"]
        )
        per_seed = contract["model_seed_contracts"]
        assert [item["model_seed"] for item in per_seed] == list(EXPECTED_MODEL_SEEDS)
        assert {item["source_validation_split_seed"] for item in per_seed} == {42}
        assert len({item["source_validation_partition_hash"] for item in per_seed}) == 1
        hashes.append(per_seed[0]["source_validation_partition_hash"])
    assert len(hashes) == 5


def test_target_train_is_unlabelled_and_target_test_is_locked(
    built_confirmatory_protocol,
) -> None:
    output, result = built_confirmatory_protocol
    forbidden = {"label_q5", "target", "task_label", "y"}
    for fold in EXPECTED_FOLDS:
        target = _load(output / f"target_unlabeled_manifests/fold_{fold:02d}.json")
        test = _load(output / f"target_test_references/fold_{fold:02d}.json")
        assert target["task_labels_exposed"] is False
        assert test["task_labels_exposed"] is False
        assert test["selection_accessible"] is False
        assert forbidden.isdisjoint(target)
        assert forbidden.isdisjoint(test)
        assert forbidden.isdisjoint(target["future_batch_fields"])
        assert set(target["forbidden_batch_fields"]) == forbidden
        assert set(target["sample_ids"]).isdisjoint(test["sample_ids"])
    firewall = result.summary["target_label_firewall"]
    assert firewall["all_target_manifests_label_free"] is True
    assert firewall["target_test_tensor_values_read"] == 0
    assert firewall["target_test_predictions_computed"] is False
    assert firewall["target_test_metrics_computed"] is False


def test_architecture_signatures_and_cpu_no_grad_audit(
    built_confirmatory_protocol,
) -> None:
    _, result = built_confirmatory_protocol
    config = _config()
    architecture = result.summary["architecture_audit"]
    assert architecture["architecture_signature"] == (
        "248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7"
    )
    assert architecture["domain_head_signature"] == domain_head_signature(
        config["architecture"]
    )
    assert architecture["task_model_parameter_count"] == 8501
    assert architecture["latent_dim"] == 1280
    assert architecture["domain_head_parameter_count"] == 172354
    assert architecture["device"] == "cpu"
    assert architecture["all_parameter_gradients_absent"] is True
    assert architecture["encoder_state_unchanged"] is True


def test_hyperparameters_and_checkpoint_policy_match_diagnostic_preregistration(
    built_confirmatory_protocol,
) -> None:
    _, result = built_confirmatory_protocol
    config = _config()
    diagnostic = _load(ROOT / config["diagnostic_provenance"]["executable_preregistration"])
    training = result.summary["training_contract"]
    assert training["optimizer"] == diagnostic["optimizer"] == "AdamW"
    assert training["learning_rate"] == diagnostic["learning_rate"] == 0.001
    assert training["weight_decay"] == diagnostic["weight_decay"] == 0.0001
    assert training["source_batch_size"] == diagnostic["batch_sizes"]["source"] == 32
    assert training["target_batch_size"] == diagnostic["batch_sizes"]["target"] == 32
    assert training["maximum_epochs"] == diagnostic["maximum_epochs"] == 12
    assert training["early_stopping_patience"] == diagnostic["early_stopping"]["patience"] == 3
    assert training["checkpoint_partition"] == "source_validation"
    assert training["checkpoint_primary"] == "source_validation_macro_f1"
    assert training["checkpoint_secondary"] == "source_validation_balanced_accuracy"
    assert training["domain_accuracy_selects_checkpoint"] is False


def test_matched_update_budgets_and_source_batch_hashes(
    built_confirmatory_protocol,
) -> None:
    output, _ = built_confirmatory_protocol
    expected_steps = {1: (139, 580, 580), 2: (149, 602, 602), 3: (157, 569, 569), 4: (143, 606, 606), 5: (150, 586, 586)}
    for fold, steps in expected_steps.items():
        contract = _load(output / f"batching_contracts/fold_{fold:02d}.json")
        assert (
            contract["source_natural_steps"],
            contract["target_natural_steps"],
            contract["matched_steps_per_epoch"],
        ) == steps
        assert contract["same_source_sequence_required_between_modes"] is True
        assert contract["same_source_optimizer_updates_required_between_modes"] is True
        for seed_contract in contract["model_seed_contracts"]:
            assert seed_contract["maximum_source_optimizer_updates_per_mode"] == steps[2] * 12
            for epoch in seed_contract["epoch_source_batch_hashes"]:
                assert epoch["hashes_match"] is True
                assert epoch["source_only_matched_source_batch_hash"] == epoch["dann_source_batch_hash"]


def test_protocol_hash_is_deterministic_and_sensitive_to_partitions_and_seeds() -> None:
    folds = [{"fold": 1, "source_subjects": ["s1"], "target_subjects": ["t1"]}]
    contract = {"direction": "Old_EEG_to_gpn_data", "execution_enabled": False}
    first = compute_confirmatory_protocol_hash(
        folds, model_seeds=EXPECTED_MODEL_SEEDS, scientific_contract=contract
    )
    second = compute_confirmatory_protocol_hash(
        copy.deepcopy(folds), model_seeds=EXPECTED_MODEL_SEEDS,
        scientific_contract=copy.deepcopy(contract),
    )
    changed_fold = copy.deepcopy(folds)
    changed_fold[0]["target_subjects"].append("t2")
    changed_seed = (42, 123, 2027)
    assert first == second
    assert first != compute_confirmatory_protocol_hash(
        changed_fold, model_seeds=EXPECTED_MODEL_SEEDS, scientific_contract=contract
    )
    assert first != compute_confirmatory_protocol_hash(
        folds, model_seeds=changed_seed, scientific_contract=contract
    )


def test_builder_never_loads_arrays_trains_or_uses_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden execution path called")

    monkeypatch.setattr("numpy.load", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.Tensor, "cuda", forbidden)
    result = build_dann_label_q5_confirmatory_protocol(
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


def test_required_artifacts_are_complete_and_diagnostic_artifacts_unchanged(
    built_confirmatory_protocol,
) -> None:
    output, result = built_confirmatory_protocol
    required = {
        "raw_universe_reference.json", "outer_fold_inventory.csv",
        "fold_eligibility.csv", "fold_subject_partitions.parquet",
        "shared_subject_audit.csv", "logical_record_overlap_audit.csv",
        "model_seed_manifest.json", "architecture_audit.json",
        "target_label_firewall_audit.json", "protocol_manifest.json",
        "protocol_hash.json", "readiness_decision.json", "errors.csv",
        "protocol_report.md", "preregistration/experiment_preregistration.json",
        "preregistration/preregistration_hash.json",
    }
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*") if path.is_file()
    }
    assert required <= observed
    for directory in (
        "source_validation_manifests", "target_unlabeled_manifests",
        "target_test_references", "batching_contracts",
    ):
        assert len(list((output / directory).glob("fold_*.json"))) == 5
    for path in output.rglob("*.json"):
        assert not re.search(r"[A-Za-z]:[\\/]", path.read_text(encoding="utf-8"))
    assert result.summary["status"] == "confirmatory_protocol_ready"
    assert result.summary["eligible_fold_count"] == 5
    config = _config()["diagnostic_provenance"]
    for name in (
        "executable_preregistration", "source_only_checkpoint",
        "dann_checkpoint", "diagnostic_summary",
    ):
        assert _sha256_file(ROOT / config[name]) == config[f"{name}_sha256"]
