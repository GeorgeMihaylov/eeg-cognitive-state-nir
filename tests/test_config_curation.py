from __future__ import annotations

import csv
import hashlib
import io
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from bench.analysis import experiment_config_audit as audit


ROOT = Path(__file__).resolve().parents[1]
CURATION = ROOT / "reports" / "summary" / "config_curation.yaml"


def make_record(
    path: str,
    *,
    loader_type: str = "benchmark_config",
    role: str = "unknown",
    status: str = "unclassified",
) -> object:
    record = audit.ConfigRecord(
        path=path,
        document={"datasets": {}, "models": {}, "tasks": []},
        loader_type=loader_type,
        role=role,
        status=status,
        schema_valid=True,
    )
    record.extracted = audit.extract_fields(record.document)
    return record


def decision(path: str, **overrides: object) -> dict:
    item = {
        "config_path": path,
        "review_status": "reviewed",
        "decision": "keep",
        "decision_reason": "Verified by repository history.",
        "canonical_config": path,
        "safe_to_move": False,
        "safe_to_edit": False,
        "evidence": ["commit:946126c"],
    }
    item.update(overrides)
    return item


def curation_document(items: list[dict], *, canonical: str = "configs/a.yaml") -> dict:
    return {
        "schema_version": 1,
        "families": {
            "example": {
                "canonical_config": canonical,
                "canonical_smoke_config": None,
                "base_configs": [],
                "legacy_configs": [],
                "diagnostic_configs": [],
                "protected_configs": [],
                "decision_reason": "Verified family entry point.",
                "evidence": ["commit:946126c"],
            }
        },
        "configs": items,
        "duplicate_groups": [],
        "seed_provenance": [],
        "normalization_plan": [],
    }


def write_curation(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "curation.yaml"
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def validate(
    tmp_path: Path,
    document: dict,
    records: dict[str, object],
) -> tuple[dict, list[str]]:
    return audit.load_and_validate_curation(
        ROOT,
        write_curation(tmp_path, document),
        records,
    )


def source_config_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in audit.discover_config_paths(ROOT)
    }


@pytest.fixture(scope="session")
def project_curation() -> object:
    return audit.audit_repository(ROOT, curation_path=CURATION)


def test_01_valid_curation_yaml_loads(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document, _ = validate(
        tmp_path,
        curation_document([decision("configs/a.yaml")]),
        records,
    )
    assert document["configs"][0]["config_path"] == "configs/a.yaml"


def test_02_absent_curation_preserves_old_behavior(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "a.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("raw_preprocessing:\n  bandpass:\n    enabled: false\n", encoding="utf-8")
    result = audit.audit_repository(tmp_path)
    assert result.curation == {}
    assert result.curation_warnings == []
    assert all(not record.curation for record in result.records)
    inventory_header = audit.render_inventory_csv(result).splitlines()[0]
    registry = yaml.safe_load(audit.render_config_registry(result))
    assert "review_status" not in inventory_header
    assert "automatic_config_role" not in inventory_header
    assert "curation" not in registry["configs"][0]
    assert "automatic_role" not in registry["configs"][0]


def test_03_unknown_config_path_is_rejected(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document = curation_document([decision("configs/missing.yaml")])
    with pytest.raises(audit.CurationValidationError, match="unknown config_path"):
        validate(tmp_path, document, records)


def test_04_duplicate_config_path_is_rejected(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    item = decision("configs/a.yaml")
    document = curation_document([item, deepcopy(item)])
    with pytest.raises(audit.CurationValidationError, match="duplicate curation"):
        validate(tmp_path, document, records)


def test_05_unknown_decision_is_rejected(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document = curation_document([decision("configs/a.yaml", decision="delete")])
    with pytest.raises(audit.CurationValidationError, match="unknown decision"):
        validate(tmp_path, document, records)


def test_06_unknown_review_status_is_rejected(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document = curation_document(
        [decision("configs/a.yaml", review_status="approved")]
    )
    with pytest.raises(audit.CurationValidationError, match="unknown review_status"):
        validate(tmp_path, document, records)


def test_07_reviewed_requires_evidence(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document = curation_document([decision("configs/a.yaml", evidence=[])])
    with pytest.raises(
        audit.CurationValidationError,
        match="must contain at least one evidence item",
    ):
        validate(tmp_path, document, records)


def test_08_reviewed_requires_decision_reason(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document = curation_document([decision("configs/a.yaml", decision_reason="")])
    with pytest.raises(audit.CurationValidationError, match="requires decision_reason"):
        validate(tmp_path, document, records)


def test_09_superseded_requires_target(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document = curation_document([decision("configs/a.yaml", decision="superseded")])
    with pytest.raises(audit.CurationValidationError, match="requires superseded_by"):
        validate(tmp_path, document, records)


def test_10_unknown_superseded_target_is_rejected(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document = curation_document(
        [
            decision(
                "configs/a.yaml",
                decision="superseded",
                superseded_by="configs/missing.yaml",
            )
        ]
    )
    with pytest.raises(audit.CurationValidationError, match="unknown superseded_by"):
        validate(tmp_path, document, records)


def test_11_superseded_cycle_is_rejected(tmp_path: Path) -> None:
    records = {
        "configs/a.yaml": make_record("configs/a.yaml"),
        "configs/b.yaml": make_record("configs/b.yaml"),
    }
    document = curation_document(
        [
            decision(
                "configs/a.yaml",
                decision="superseded",
                superseded_by="configs/b.yaml",
            ),
            decision(
                "configs/b.yaml",
                decision="superseded",
                superseded_by="configs/a.yaml",
                canonical_config="configs/b.yaml",
            ),
        ]
    )
    with pytest.raises(audit.CurationValidationError, match="superseded_by cycle"):
        validate(tmp_path, document, records)


def test_12_unknown_canonical_config_is_rejected(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document = curation_document(
        [decision("configs/a.yaml", canonical_config="configs/missing.yaml")]
    )
    with pytest.raises(audit.CurationValidationError, match="unknown canonical_config"):
        validate(tmp_path, document, records)


def test_13_canonical_self_reference_is_allowed(tmp_path: Path) -> None:
    records = {"configs/a.yaml": make_record("configs/a.yaml")}
    document, _ = validate(
        tmp_path,
        curation_document([decision("configs/a.yaml")]),
        records,
    )
    assert document["configs"][0]["canonical_config"] == "configs/a.yaml"


def test_14_incompatible_canonical_loader_is_rejected(tmp_path: Path) -> None:
    records = {
        "configs/a.yaml": make_record("configs/a.yaml"),
        "configs/raw.yaml": make_record(
            "configs/raw.yaml", loader_type="raw_preprocessing_fragment"
        ),
    }
    document = curation_document(
        [
            decision("configs/a.yaml", canonical_config="configs/raw.yaml"),
            decision("configs/raw.yaml"),
        ],
        canonical="configs/a.yaml",
    )
    with pytest.raises(audit.CurationValidationError, match="incompatible loader_type"):
        validate(tmp_path, document, records)


def test_15_safe_to_move_rejects_detected_references(tmp_path: Path) -> None:
    record = make_record("configs/a.yaml")
    record.registry_ids = ["experiment"]
    records = {record.path: record}
    document = curation_document(
        [decision("configs/a.yaml", safe_to_move=True)]
    )
    with pytest.raises(audit.CurationValidationError, match="safe_to_move=true"):
        validate(tmp_path, document, records)


def test_16_safe_to_edit_is_merged() -> None:
    record = make_record("configs/a.yaml")
    item = decision("configs/a.yaml", safe_to_edit=True)
    audit.apply_curation({record.path: record}, {"configs": [item]})
    assert record.curation["safe_to_edit"] is True


def test_17_manual_role_overrides_unknown() -> None:
    record = make_record("configs/a.yaml", role="unknown")
    record.automatic_role = "unknown"
    item = decision("configs/a.yaml", config_role="full")
    audit.apply_curation({record.path: record}, {"configs": [item]})
    assert record.role == "full"


def test_18_manual_result_status_is_merged() -> None:
    record = make_record("configs/a.yaml", status="unclassified")
    item = decision("configs/a.yaml", result_status="baseline")
    audit.apply_curation({record.path: record}, {"configs": [item]})
    assert record.status == "baseline"


def test_19_automatic_fields_are_preserved() -> None:
    record = make_record("configs/a.yaml", role="unknown", status="unclassified")
    record.automatic_role = record.role
    record.automatic_status = record.status
    item = decision(
        "configs/a.yaml",
        config_role="full",
        result_status="baseline",
    )
    audit.apply_curation({record.path: record}, {"configs": [item]})
    assert (record.automatic_role, record.automatic_status) == (
        "unknown",
        "unclassified",
    )


def test_20_manual_fields_appear_in_registry(project_curation: object) -> None:
    document = yaml.safe_load(audit.render_config_registry(project_curation))
    item = next(
        value
        for value in document["configs"]
        if value["config_path"] == "configs/groupkfold_rf_label_q5.yaml"
    )
    assert item["curation"]["decision"] == "keep"
    assert item["automatic_status"] == "baseline"


def test_21_manual_fields_appear_in_inventory(project_curation: object) -> None:
    rows = list(
        csv.DictReader(io.StringIO(audit.render_inventory_csv(project_curation)))
    )
    item = next(
        row for row in rows if row["config_path"] == "configs/groupkfold_rf_label_q5.yaml"
    )
    assert item["review_status"] == "reviewed"
    assert item["decision"] == "keep"
    assert item["automatic_result_status"] == "baseline"


def test_22_curation_outputs_are_deterministic(project_curation: object) -> None:
    first = (
        audit.render_inventory_csv(project_curation),
        audit.render_config_registry(project_curation),
        audit.render_markdown(project_curation),
        audit.render_curation_markdown(project_curation),
    )
    second = (
        audit.render_inventory_csv(project_curation),
        audit.render_config_registry(project_curation),
        audit.render_markdown(project_curation),
        audit.render_curation_markdown(project_curation),
    )
    assert first == second


def test_23_automatically_unclassified_configs_are_reviewed_and_links_resolve(
    project_curation: object,
) -> None:
    automatically_unclassified = [
        record
        for record in project_curation.records
        if record.automatic_status == "unclassified"
    ]
    assert automatically_unclassified
    assert all(
        record.curation.get("review_status") in audit.VALID_REVIEW_STATUSES
        for record in automatically_unclassified
    )
    assert not project_curation.registry_missing_configs
    assert not [
        mismatch
        for mismatch in project_curation.registry_consistency
        if mismatch["severity"] == "error"
    ]
    assert all(
        (ROOT / report_path).is_file()
        for record in project_curation.records
        for report_path in record.report_links
    )


def test_24_all_reviewed_configs_have_evidence(project_curation: object) -> None:
    reviewed = [
        record
        for record in project_curation.records
        if record.curation.get("review_status") == "reviewed"
    ]
    assert reviewed
    assert all(record.curation.get("evidence") for record in reviewed)


def test_25_families_have_canonical_or_reason(project_curation: object) -> None:
    families = project_curation.curation["families"]
    assert families
    assert all(
        value.get("canonical_config") or value.get("no_canonical_reason")
        for value in families.values()
    )
    assert all(value.get("decision_reason") for value in families.values())


def test_26_two_gapaware_legacy_sequence_configs_are_preserved(
    project_curation: object,
) -> None:
    expected = {
        "configs/groupkfold_torch_lstm_gapaware_label_q5.yaml",
        "configs/groupkfold_torch_bilstm_gapaware_label_q5.yaml",
    }
    actual = {
        record.path
        for record in project_curation.records
        if record.path in expected
        and record.curation.get("decision") == "keep_as_legacy"
    }
    assert actual == expected


def test_27_raw_fragments_are_not_full_runs(project_curation: object) -> None:
    fragments = [
        record
        for record in project_curation.records
        if record.loader_type == "raw_preprocessing_fragment"
    ]
    assert len(fragments) == 4
    assert all(record.role != "full" and record.status != "final" for record in fragments)
    assert all(
        record.curation.get("review_status") == "not_applicable"
        for record in fragments
    )


def test_28_base_configs_are_not_final_runs(project_curation: object) -> None:
    bases = [
        record
        for record in project_curation.records
        if record.curation.get("decision") == "keep_as_base"
    ]
    assert len(bases) == 12
    assert all(record.role == "base" and record.status != "final" for record in bases)


def test_29_seed_provenance_is_preserved(project_curation: object) -> None:
    provenance = project_curation.curation["seed_provenance"]
    assert len(provenance) == 4
    assert {item["name"] for item in provenance} == {
        "Transformer",
        "EEGNet",
        "ShallowConvNet",
        "preprocessing_ablation",
    }
    assert all(item["registry_seeds"] == [7, 42, 123] for item in provenance)


def test_30_scientific_duplicate_groups_are_documented(
    project_curation: object,
) -> None:
    groups = project_curation.curation["duplicate_groups"]
    assert len(groups) == 4
    assert all(len(group["configs"]) >= 2 for group in groups)
    assert all(group["relationship"] for group in groups)
    assert all("keep_separate" in group for group in groups)


def test_31_audit_does_not_modify_source_experiment_yaml() -> None:
    before = source_config_hashes()
    audit.audit_repository(ROOT, curation_path=CURATION)
    assert source_config_hashes() == before


def test_32_current_curation_passes_strict_validation(
    project_curation: object,
) -> None:
    records = {record.path: record for record in project_curation.records}
    document, warnings = audit.load_and_validate_curation(ROOT, CURATION, records)
    assert len(document["configs"]) == 76
    assert len(document["duplicate_groups"]) == 4
    assert len(document["seed_provenance"]) == 4
    assert all(isinstance(value, str) for value in warnings)
