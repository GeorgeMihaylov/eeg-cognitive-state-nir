from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from bench.experiments.fomaml_label_q5_raw_protocol import (
    PROTOCOL_ID,
    audit_support_budget,
    build_fomaml_label_q5_raw_protocol,
    choose_meta_validation_subjects,
    compute_protocol_hash,
    episode_identifier,
    validate_raw_protocol_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/meta_learning/fomaml_label_q5_raw_protocol.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def built_protocol(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("fomaml-raw-protocol")
    result = build_fomaml_label_q5_raw_protocol(
        _config(), repository_root=ROOT, output_dir=output
    )
    return output, result


def test_config_is_disabled_new_protocol_without_absolute_paths() -> None:
    config = _config()
    validate_raw_protocol_config(config)
    assert config["protocol_id"] == PROTOCOL_ID
    assert config["execution_enabled"] is False
    assert config["outer_fold"] == 1
    assert config["seed"] == 42
    assert config["episode_contract"]["allow_window_level_fallback"] is False
    assert config["episode_contract"]["sampling_with_replacement"] is False
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:[\\/]", text)


def test_raw_universe_is_canonical_unique_valid_and_resolvable(built_protocol) -> None:
    output, result = built_protocol
    frame = result.raw_inventory
    manifest = json.loads((output / "raw_universe_manifest.json").read_text())
    assert len(frame) == 30_958
    assert frame["sample_id"].is_unique
    assert set(frame["label_q5"]) == {0, 1, 2, 3, 4}
    assert frame["subject_id"].notna().all()
    assert frame["record_group_id"].notna().all()
    assert manifest["subject_count"] == 54
    assert manifest["record_count"] == 86
    assert manifest["logical_record_count"] == 86
    assert manifest["channels"] == 14
    assert manifest["samples_per_window"] == 2560
    assert manifest["sampling_rate"] == 256.0
    assert manifest["tensor_audit"]["referenced_samples"] == 30_958
    assert manifest["tensor_audit"]["all_tensor_references_resolved"]
    assert manifest["tensor_audit"]["all_shard_shapes_valid"]


def test_outer_fold_is_exactly_reused_without_subject_overlap(built_protocol) -> None:
    output, _ = built_protocol
    audit = json.loads((output / "outer_split_audit.json").read_text())
    assert audit["source_fold_assignments_sha256"] == (
        "41ec5a244e11b5dd4ff25faa7361f2bca302dd719612fea8cbc54a55b6ff3341"
    )
    assert audit["source_hash_matches_blocked_protocol"]
    assert audit["outer_train_count"] == 43
    assert audit["outer_test_count"] == 11
    assert audit["outer_subject_overlap"] == 0
    assert audit["raw_subject_universe_matches"]


def test_eligibility_class_policy_and_skips_are_explicit(built_protocol) -> None:
    output, result = built_protocol
    eligibility = result.eligibility
    train = eligibility[eligibility["outer_partition"] == "outer_train"]
    test = eligibility[eligibility["outer_partition"] == "outer_test"]
    assert int(train["base_eligible"].sum()) == 20
    assert int(test["base_eligible"].sum()) == 6
    assert int(train["at_least_one_per_class_eligible"].sum()) == 16
    assert int(test["at_least_one_per_class_eligible"].sum()) == 5
    assert (~eligibility["at_least_one_per_class_eligible"]).sum() == 33
    skipped = pd.read_csv(output / "errors.csv")
    assert len(skipped) == 33
    assert skipped["reason"].astype(str).str.len().gt(0).all()
    policy = pd.read_csv(output / "class_policy_audit.csv")
    selected = policy[(policy["outer_partition"] == "outer_train") & policy["selected"]]
    assert selected.iloc[0]["policy"] == "at_least_one_per_class"
    assert selected.iloc[0]["selection_scope"] == "outer_train_only"


def test_support_budget_uses_complete_records_and_outer_train_only(built_protocol) -> None:
    output, result = built_protocol
    spec = json.loads((output / "episode_spec.json").read_text())
    prereg = json.loads(
        (output / "preregistration/experiment_preregistration.json").read_text()
    )
    assert spec["support_record_count"] == 1
    assert spec["query_record_count"] == "all_remaining_records"
    assert spec["window_level_fallback"] is False
    assert spec["sampling_with_replacement"] is False
    budget = prereg["support_budget"]
    assert budget["selection_scope"] == "outer_train_only"
    assert budget["outer_test_used"] is False
    assert budget["fixed_window_cap"] is None
    assert budget["support_windows"] == {
        "minimum": 128, "median": 400.5, "maximum": 622
    }
    assert budget["query_windows"] == {
        "minimum": 97, "median": 200.0, "maximum": 427
    }
    modified = result.eligibility.copy()
    modified.loc[modified["outer_partition"] == "outer_test", "support_count"] = 1
    _, original_summary = audit_support_budget(
        result.eligibility, _config()["episode_contract"]
    )
    _, modified_summary = audit_support_budget(
        modified, _config()["episode_contract"]
    )
    assert original_summary == modified_summary


def test_meta_split_is_deterministic_balanced_and_outer_test_independent(
    built_protocol,
) -> None:
    output, result = built_protocol
    manifest = json.loads((output / "meta_split_manifest.json").read_text())
    eligible_train = sorted(
        result.eligibility.loc[
            result.eligibility["outer_partition"].eq("outer_train")
            & result.eligibility["at_least_one_per_class_eligible"],
            "subject_id",
        ]
    )
    repeated = choose_meta_validation_subjects(
        result.raw_inventory, eligible_train, fraction=0.2,
        minimum_subjects=5, seed=42,
    )
    altered = result.raw_inventory.copy()
    outer_test_subjects = set(
        result.eligibility.loc[
            result.eligibility["outer_partition"].eq("outer_test"), "subject_id"
        ]
    )
    altered.loc[altered["subject_id"].isin(outer_test_subjects), "label_q5"] = 0
    altered_result = choose_meta_validation_subjects(
        altered, eligible_train, fraction=0.2, minimum_subjects=5, seed=42
    )
    assert repeated == manifest
    assert altered_result == manifest
    assert len(manifest["meta_train_subjects"]) == 11
    assert len(manifest["meta_validation_subjects"]) == 5
    assert not set(manifest["meta_train_subjects"]) & set(
        manifest["meta_validation_subjects"]
    )
    assert manifest["outer_test_used"] is False


def test_episode_index_is_record_disjoint_complete_and_leakage_safe(
    built_protocol,
) -> None:
    output, result = built_protocol
    episodes = result.episode_index
    audit = json.loads((output / "episode_leakage_audit.json").read_text())
    assert episodes["scope"].value_counts().to_dict() == {
        "meta_train": 11, "meta_validation": 5, "outer_test": 5
    }
    assert episodes["episode_id"].is_unique
    assert audit["valid"]
    assert audit["missing_raw_ids"] == 0
    assert audit["duplicate_episode_sample_references"] == 0
    assert audit["support_query_sample_overlap"] == 0
    assert audit["support_query_record_overlap"] == 0
    assert audit["episode_subject_mismatches"] == 0
    assert audit["chronology_failures"] == 0
    assert audit["within_record_fallbacks"] == 0
    for episode in episodes.itertuples():
        assert set(episode.support_targets) == {0, 1, 2, 3, 4}
        assert set(episode.query_targets) == {0, 1, 2, 3, 4}
        assert set(episode.support_sample_ids).isdisjoint(episode.query_sample_ids)
        assert set(episode.support_record_ids).isdisjoint(episode.query_record_ids)


def test_episode_and_protocol_hashes_change_with_samples_or_meta_split() -> None:
    base = {
        "sample_ids": ["1", "2"],
        "subject": "s1",
        "meta_split_hash": "split-a",
    }
    assert episode_identifier(base) == episode_identifier(dict(base))
    assert episode_identifier(base) != episode_identifier(
        {**base, "sample_ids": ["1", "3"]}
    )
    protocol = {"raw": "u", "meta_split_hash": "split-a", "episodes": ["e"]}
    assert compute_protocol_hash(protocol) == compute_protocol_hash(dict(protocol))
    assert compute_protocol_hash(protocol) != compute_protocol_hash(
        {**protocol, "meta_split_hash": "split-b"}
    )


def test_old_protocol_is_audited_not_changed_remapped_or_reused(built_protocol) -> None:
    output, result = built_protocol
    comparison = pd.read_csv(output / "old_new_protocol_comparison.csv")
    assert result.summary["old_runtime_artifacts_unchanged"]
    assert result.summary["comparison"]["old_episodes"] == 40
    assert result.summary["comparison"]["old_episodes_fully_raw_compatible"] == 23
    assert result.summary["comparison"]["old_missing_ids"] == 901
    assert result.summary["comparison"]["old_episode_ids_reused"] == 0
    assert result.summary["comparison"]["remapping_performed"] is False
    assert not set(comparison["old_feature_level_episode"]) & set(
        result.episode_index["episode_id"]
    )


def test_preregistration_follows_audit_and_matches_production_architecture(
    built_protocol,
) -> None:
    output, result = built_protocol
    path = output / "preregistration/experiment_preregistration.json"
    prereg = json.loads(path.read_text())
    sidecar = json.loads(
        (output / "preregistration/preregistration_hash.json").read_text()
    )
    assert sidecar["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.summary["preregistration_hash"] == sidecar["sha256"]
    assert prereg["protocol_audit_completed_before_preregistration"]
    assert prereg["execution_enabled"] is False
    assert prereg["architecture_signature"] == (
        "248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7"
    )
    assert prereg["relation_to_blocked_8X"]["protocol_hash"] == (
        "a3e6ff5ee2dbfa1638ffee9180ddff582dbab8aa6186e164320dd92f082871e8"
    )
    assert prereg["relation_to_blocked_8X"]["preregistration_hash"] == (
        "54f21e907ff1a414d45c1594e422c4caede0a449ca9acf02374bb50502122754"
    )


def test_protocol_builder_never_trains_uses_cuda_optimizer_or_checkpoint(
    built_protocol,
) -> None:
    output, result = built_protocol
    decision = json.loads((output / "readiness_decision.json").read_text())
    assert result.summary["readiness_status"] == "raw_protocol_ready"
    assert decision["execution_enabled"] is False
    assert decision["training_performed"] is False
    assert decision["optimizer_created"] is False
    assert decision["cuda_tensors_created"] is False
    assert decision["checkpoint_created"] is False
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob("*.ckpt"))


def test_runtime_artifacts_are_complete_and_contain_no_absolute_paths(
    built_protocol,
) -> None:
    output, _ = built_protocol
    required = {
        "raw_universe_manifest.json", "raw_sample_inventory.parquet",
        "outer_split_audit.json", "participant_eligibility.csv",
        "class_policy_audit.csv", "support_budget_audit.csv",
        "meta_split_manifest.json", "episode_spec.json", "episode_index.parquet",
        "episode_manifest.json", "episode_balance.csv",
        "episode_leakage_audit.json", "old_new_protocol_comparison.csv",
        "protocol_manifest.json", "protocol_hash.json",
        "preregistration/experiment_preregistration.json",
        "preregistration/preregistration_hash.json", "readiness_decision.json",
        "errors.csv", "protocol_report.md",
    }
    found = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert required <= found
    absolute = re.compile(r"[A-Za-z]:[\\/]")
    for path in output.rglob("*"):
        if path.suffix in {".json", ".csv", ".md"}:
            assert not absolute.search(path.read_text(encoding="utf-8")), path
    for path in output.rglob("*.parquet"):
        frame = pd.read_parquet(path)
        for column in frame.select_dtypes(include="str"):
            assert not frame[column].fillna("").astype(str).str.contains(absolute).any(), path
