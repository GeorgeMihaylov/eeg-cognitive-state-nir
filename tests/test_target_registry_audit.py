from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.analysis.target_registry_audit import (
    OUTPUT_FILENAMES,
    PM_METRICS,
    REGISTRY_REQUIRED_FIELDS,
    TARGET_COLUMNS,
    REPO_ROOT,
    run_target_registry_audit,
)
from bench.datasets.base_eeg_data_loader import resolve_feature_columns


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    n_rows = 40
    focus = np.linspace(0.01, 0.99, n_rows)
    frame: dict[str, object] = {
        "record_id": [f"record_{index // 10}" for index in range(n_rows)],
        "source": ["Old_EEG"] * 20 + ["gpn_data"] * 20,
        "subject_id": [f"subject_{index // 5}" for index in range(n_rows)],
        "EEG.AF3__mean": np.arange(n_rows, dtype=float),
        "POW.AF3.Alpha__mean": np.arange(n_rows, dtype=float) / 10,
    }
    for metric_index, metric in enumerate(PM_METRICS):
        title = metric.title()
        values = focus if metric == "focus" else np.clip(
            focus + metric_index * 0.001, 0, 1
        )
        frame[f"PM.{title}.Scaled__mean"] = values
        frame[f"PM.{title}.Scaled__std"] = np.zeros(n_rows)
        frame[f"PM.{title}.Scaled__min"] = values
        frame[f"PM.{title}.Scaled__max"] = values
        frame[f"PM.{title}.Scaled__last"] = values
        frame[f"PM.{title}.IsActive__mean"] = (np.arange(n_rows) % 2).astype(float)
        frame[f"target_{metric}"] = values.copy()
    data = pd.DataFrame(frame)
    data["target_main"] = data["target_focus"]
    data["label_q5"] = pd.qcut(
        data["target_main"], q=5, labels=False, duplicates="drop"
    ).astype(float)
    dataset = tmp_path / "fixture.parquet"
    data.to_parquet(dataset, index=False)

    source_pm_columns = []
    for metric in PM_METRICS:
        title = metric.title()
        source_pm_columns.extend(
            [
                f"PM.{title}.Raw",
                f"PM.{title}.Scaled",
                f"PM.{title}.Min",
                f"PM.{title}.Max",
                f"PM.{title}.IsActive",
            ]
        )
    source_pm_columns.append("PM.LongTermExcitement")
    validated = {
        "by_field": {
            "pm_columns": {
                "by_source": {
                    source: {"union": source_pm_columns, "common": source_pm_columns}
                    for source in ("Old_EEG", "gpn_data")
                }
            }
        }
    }
    validated_path = tmp_path / "validated_columns.json"
    validated_path.write_text(json.dumps(validated), encoding="utf-8")
    return dataset, validated_path


def _run(tmp_path: Path):
    dataset, validated = _write_fixture(tmp_path)
    output = tmp_path / "output"
    report = tmp_path / "full_target_registry_audit.md"
    result = run_target_registry_audit(
        dataset,
        validated,
        output,
        repo_root=REPO_ROOT,
        report_path=report,
        logical_map_path=tmp_path / "missing_logical_map.parquet",
        raw_manifest_path=tmp_path / "missing_raw_manifest.parquet",
    )
    registry = yaml.safe_load((output / "target_registry.yaml").read_text(encoding="utf-8"))
    return dataset, output, report, result, registry


def test_target_registry_has_canonical_pm_order_and_required_fields(tmp_path: Path):
    _, _, _, result, registry = _run(tmp_path)

    assert result.status == "target_registry_ready"
    assert tuple(registry["canonical_pm_order"]) == PM_METRICS
    continuous = [
        target
        for target in registry["targets"]
        if target["target_family"] == "continuous_pm"
    ]
    assert [target["target_id"] for target in continuous] == list(TARGET_COLUMNS)
    assert len(continuous) == 7
    for target in registry["targets"]:
        assert not (set(REGISTRY_REQUIRED_FIELDS) - set(target))


def test_feature_contract_excludes_targets_and_pm_columns(tmp_path: Path):
    _, _, _, _, registry = _run(tmp_path)
    columns = [
        "EEG.AF3__mean",
        "POW.AF3.Alpha__mean",
        "target_focus",
        "target_main",
        "PM.Focus.Scaled__mean",
        "label_q5",
    ]

    selected = resolve_feature_columns(columns, "eeg_pow")

    assert selected == ["EEG.AF3__mean", "POW.AF3.Alpha__mean"]
    assert registry["feature_contract"]["target_or_pm_feature_count"] == 0


def test_focus_label_aliases_candidates_and_risks_are_explicit(tmp_path: Path):
    _, output, _, _, registry = _run(tmp_path)
    targets = {target["target_id"]: target for target in registry["targets"]}

    label = targets["label_q5"]
    assert label["device_metric"] == "PM.Focus.Scaled"
    assert label["canonical_display_id"] == "label_focus_q5"
    assert label["status"] == "legacy"
    assert "global" in label["leakage_risk"]
    assert targets["target_main"]["status"] == "legacy"
    assert targets["pm_long_term_excitement_candidate"]["target_family"] == "candidate_additional_pm"
    assert "PM.LongTermExcitement" not in [
        target["device_metric"]
        for target in registry["targets"]
        if target["target_family"] == "continuous_pm"
    ]
    ordinal_candidates = [
        target
        for target in registry["targets"]
        if target["target_id"].endswith("_candidate")
        and target["target_family"] == "derived_ordinal_proxy"
    ]
    assert len(ordinal_candidates) == 14
    assert all(target["processed_column"] is None for target in ordinal_candidates)

    risks = pd.read_csv(output / "target_leakage_risk.csv")
    assert "global_label_q5_quantiles" in set(risks["risk_id"])
    assert "raw_loader_target_lock" in set(risks["risk_id"])


def test_derivations_availability_cohorts_and_activity_are_audited(tmp_path: Path):
    _, output, _, _, _ = _run(tmp_path)
    derivation = pd.read_csv(output / "target_derivation_audit.csv")
    target_rows = derivation[derivation["output_column"].isin(TARGET_COLUMNS)]
    assert len(target_rows) == 7
    assert target_rows["verified_equivalence"].all()
    assert (target_rows["max_abs_difference"] == 0).all()
    assert (target_rows["mismatch_count"] == 0).all()

    availability = pd.read_csv(output / "target_availability_by_source.csv")
    for target in TARGET_COLUMNS:
        source_rows = availability[
            (availability["target_id"] == target)
            & availability["source"].isin(["Old_EEG", "gpn_data"])
        ]
        assert len(source_rows) == 2
        assert source_rows["upstream_field_available"].all()

    cohorts = pd.read_csv(output / "target_cohort_counts.csv")
    assert set(TARGET_COLUMNS).issubset(set(cohorts["cohort_id"]))
    complete = cohorts[
        (cohorts["cohort_id"] == "seven_pm_complete_case")
        & (cohorts["source"] == "all")
    ].iloc[0]
    assert complete["n_windows"] == 40

    proxies = pd.read_csv(output / "target_proxy_candidates.csv")
    activity = proxies[proxies["candidate_family"] == "device_activity_proxy"]
    assert len(activity) == 7
    assert (activity["n_intermediate"] == 0).all()
    assert set(activity["status"]) == {"requires_semantic_validation"}


def test_code_report_links_and_tracked_outputs_are_relative(tmp_path: Path):
    _, output, report, _, registry = _run(tmp_path)
    for target in registry["targets"]:
        code_path = target["derivation_file"]
        if code_path:
            assert (REPO_ROOT / code_path).is_file()
    for report_path in registry["report_references"]:
        assert (REPO_ROOT / report_path).is_file()

    for path in [*(output / name for name in OUTPUT_FILENAMES), report]:
        text = path.read_text(encoding="utf-8")
        assert "F:\\" not in text
        assert str(tmp_path) not in text
    assert "target_registry_ready" in report.read_text(encoding="utf-8")


def test_audit_is_deterministic_read_only_and_training_free(tmp_path: Path):
    dataset, output, report, _, _ = _run(tmp_path)
    dataset_before = _sha(dataset)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    first[report.name] = report.read_bytes()

    run_target_registry_audit(
        dataset,
        tmp_path / "validated_columns.json",
        output,
        repo_root=REPO_ROOT,
        report_path=report,
        logical_map_path=tmp_path / "missing_logical_map.parquet",
        raw_manifest_path=tmp_path / "missing_raw_manifest.parquet",
    )

    second = {path.name: path.read_bytes() for path in output.iterdir()}
    second[report.name] = report.read_bytes()
    assert first == second
    assert _sha(dataset) == dataset_before
    assert not (tmp_path / "benchmark_results").exists()
    module_source = (REPO_ROOT / "bench/analysis/target_registry_audit.py").read_text(
        encoding="utf-8"
    )
    assert "bench_runner" not in module_source
    assert "torch.optim" not in module_source


@pytest.mark.integration
def test_canonical_data_contract_and_raw_compatible_cohorts(tmp_path: Path):
    dataset = REPO_ROOT / "data/processed/windowed_eeg_pm_dataset_w10.parquet"
    validated = REPO_ROOT / "data/interim/validated_columns.json"
    logical_map = REPO_ROOT / "data/interim/logical_recording_map.parquet"
    raw_manifest = REPO_ROOT / "data/interim/raw_eeg_window_index_w10_raw_v3.parquet"
    if not all(path.is_file() for path in (dataset, validated, logical_map, raw_manifest)):
        pytest.skip("Canonical local data artifacts are unavailable")
    before = _sha(dataset)
    output = tmp_path / "canonical"

    result = run_target_registry_audit(
        dataset,
        validated,
        output,
        repo_root=REPO_ROOT,
        report_path=tmp_path / "canonical_report.md",
        logical_map_path=logical_map,
        raw_manifest_path=raw_manifest,
    )

    assert (result.dataset_rows, result.dataset_columns) == (51308, 508)
    assert dict(result.feature_counts) == {"eeg": 168, "pow": 280, "eeg_pow": 448}
    registry = yaml.safe_load((output / "target_registry.yaml").read_text(encoding="utf-8"))
    assert registry["feature_contract"]["eeg_pow_feature_list_sha256"] == (
        "8cd5d70faa8ff30fb4290dd9d9a2dde0e81f50e7682d05668b5fb47df511fd51"
    )
    cohorts = pd.read_csv(output / "target_cohort_counts.csv")
    complete = cohorts[cohorts["cohort_id"] == "seven_pm_complete_case"].set_index("source")
    assert int(complete.loc["all", "n_windows"]) == 43174
    assert int(complete.loc["all", "n_subjects"]) == 53
    assert int(complete.loc["all", "n_source_records"]) == 117
    assert int(complete.loc["gpn_data", "n_windows"]) == 22808
    assert int(complete.loc["Old_EEG", "n_windows"]) == 20366
    label = cohorts[(cohorts["cohort_id"] == "label_q5") & (cohorts["source"] == "all")].iloc[0]
    assert int(label["n_windows"]) == 45384
    assert int(label["n_subjects"]) == 54
    raw = cohorts[cohorts["cohort_id"] == "raw_deduplicated_label_q5_compatible"].iloc[0]
    assert int(raw["n_windows"]) == 30958
    assert int(raw["n_logical_records"]) == 86
    assert _sha(dataset) == before
