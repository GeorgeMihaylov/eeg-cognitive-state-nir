from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bench.meta import (
    audit_architectures,
    build_meta_validation_protocol,
    run_fomaml_production_contract_audit,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/meta_learning/fomaml_production_contract.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _synthetic_assignments(path: Path) -> None:
    rows = []
    sample = 0
    for subject_number in range(6):
        fold = 1 if subject_number < 2 else 2
        for record_number in range(2):
            for offset in range(4):
                rows.append({
                    "sample_id": str(sample),
                    "sample_index": sample,
                    "subject_id": f"subject-{subject_number}",
                    "record_id": f"record-{subject_number}-{record_number}",
                    "fold": fold,
                    "y_true": (subject_number + offset) % 5,
                })
                sample += 1
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_meta_validation_protocol_is_deterministic_and_leakage_safe(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.parquet"
    _synthetic_assignments(assignments)
    config = {
        "fold_assignments": assignments.name,
        "outer_fold": 1,
        "meta_validation_fraction": 0.25,
        "support_budget": 2,
        "query_budget": 2,
        "seed": 42,
    }
    first = build_meta_validation_protocol(config, repository_root=tmp_path)
    second = build_meta_validation_protocol(config, repository_root=tmp_path)
    assert first.protocol == second.protocol
    assert set(first.protocol["outer_train_subjects"]).isdisjoint(
        first.protocol["outer_test_subjects"]
    )
    assert set(first.protocol["meta_train_subjects"]).isdisjoint(
        first.protocol["meta_validation_subjects"]
    )
    assert set(first.protocol["meta_validation_subjects"]).isdisjoint(
        first.protocol["outer_test_subjects"]
    )
    assert not first.protocol["outer_test_in_meta_validation"]
    for row in first.episode_index.itertuples():
        assert set(row.support_sample_ids).isdisjoint(row.query_sample_ids)
        assert set(row.support_record_ids).isdisjoint(row.query_record_ids)


def test_production_architectures_match_latent_head_and_checkpoints() -> None:
    rows, schemas, batchnorm = audit_architectures(_config(), repository_root=ROOT)
    assert len(rows) == 8
    canonical = {row["model_id"]: row for row in rows if row["model_id"].endswith(":canonical")}
    assert canonical["torch_eegnet:canonical"]["latent_dim"] == 1280
    assert canonical["torch_shallow_convnet:canonical"]["latent_dim"] == 40
    assert all(row["latent_dim"] == row["output_head_input_dimension"] for row in rows)
    assert all(row["architecture_signature"] for row in rows)
    assert all(schemas[name]["checkpoint_signature"] for name in schemas)
    assert len(batchnorm) == 4


def test_disabled_production_audit_writes_safe_runtime_contract() -> None:
    config = _config()
    assert config["execution_enabled"] is False
    summary = run_fomaml_production_contract_audit(config, repository_root=ROOT)
    assert summary["status"] == "production_contract_ready"
    assert summary["both_buffer_policies_safe"]
    assert summary["query_leakage_absent"]
    output = ROOT / config["output_dir"]
    expected = {
        "architecture_audit.json", "latent_dim_audit.csv",
        "parameter_buffer_schema.json", "batchnorm_inventory.csv",
        "buffer_policy_audit.json", "production_compatibility.csv",
        "functional_state_audit.json", "query_leakage_audit.json",
        "meta_validation_protocol.json", "meta_validation_episode_index.parquet",
        "future_experiment_config.json", "decision.json", "errors.csv",
        "contract_report.md",
    }
    assert expected == {path.name for path in output.iterdir() if path.is_file()}
    for path in output.iterdir():
        if path.suffix.lower() in {".json", ".csv", ".md"}:
            text = path.read_text(encoding="utf-8")
            assert "F:\\EEG" not in text
            assert "C:\\Users" not in text
