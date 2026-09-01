from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.analysis.pm_quality_downstream import (
    build_downstream_manifest,
    build_downstream_plan,
    plan_downstream,
)
from bench.analysis.pm_temporal_quality import (
    PM_METRICS,
    VARIANT_ORDER,
    build_variants,
    calculate_q3_stability,
    causal_transform_1d,
    load_config,
    plan_analysis,
    prepare_pm_frame,
    stable_hash,
)
from bench.analysis.pm_quality_result_status_migration import migrate_result_status


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "pm_quality" / "pm_temporal_quality_v1.json"
FINAL_CONFIG = (
    ROOT / "experiments" / "pm_quality" / "pm_temporal_quality_rf_final_v1.json"
)


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    sample_id = 0
    for subject_index in range(10):
        subject = f"s{subject_index:02d}"
        for time_index, t_start in enumerate((0.0, 10.0, 20.0, 30.0)):
            base = subject_index + time_index / 10.0
            row = {
                "sample_id": sample_id,
                "source": "gpn_data" if subject_index % 2 == 0 else "Old_EEG",
                "subject_id": subject,
                "record_id": f"{'gpn_data' if subject_index % 2 == 0 else 'Old_EEG'}__{subject}__record",
                "record_group_id": f"{subject}__record",
                "t_start": t_start,
            }
            row.update({f"target_{metric}": base for metric in PM_METRICS})
            rows.append(row)
            sample_id += 1
    return pd.DataFrame(rows)


def test_config_fixes_all_preregistered_variants() -> None:
    config = load_config(CONFIG)
    assert tuple(config["variants"]) == VARIANT_ORDER
    assert config["temporal"]["missing_policy"] == "preserve_nan_and_reset_state"
    assert stable_hash(config) == stable_hash(load_config(CONFIG))


def test_median_and_ema_use_only_current_and_past() -> None:
    values = np.asarray([1.0, 9.0, 3.0, 100.0])
    median, _ = causal_transform_1d(values, method="trailing_median", window=3)
    ema, _ = causal_transform_1d(values, method="causal_ema", alpha=0.5)
    np.testing.assert_allclose(median, [1.0, 5.0, 3.0, 9.0])
    np.testing.assert_allclose(ema, [1.0, 5.0, 4.0, 52.0])

    changed_future = values.copy()
    changed_future[-1] = -500.0
    median_changed, _ = causal_transform_1d(
        changed_future, method="trailing_median", window=3
    )
    ema_changed, _ = causal_transform_1d(
        changed_future, method="causal_ema", alpha=0.5
    )
    np.testing.assert_allclose(median_changed[:-1], median[:-1])
    np.testing.assert_allclose(ema_changed[:-1], ema[:-1])


def test_nan_is_preserved_and_resets_state() -> None:
    values = np.asarray([1.0, 2.0, np.nan, 10.0, 12.0])
    median, _ = causal_transform_1d(values, method="trailing_median", window=3)
    ema, _ = causal_transform_1d(values, method="causal_ema", alpha=0.5)
    np.testing.assert_allclose(median, [1.0, 1.5, np.nan, 10.0, 11.0], equal_nan=True)
    np.testing.assert_allclose(ema, [1.0, 1.5, np.nan, 10.0, 11.0], equal_nan=True)


def test_hampel_replaces_only_robust_statistical_anomaly() -> None:
    values = np.asarray([1.0, 1.1, 0.9, 1.0, 10.0])
    transformed, audit = causal_transform_1d(
        values,
        method="causal_hampel",
        window=5,
        threshold=3.0,
        mad_scale=1.4826,
    )
    assert audit["outlier_flag"].tolist() == [False, False, False, False, True]
    assert transformed[-1] == 1.0
    constant, constant_audit = causal_transform_1d(
        np.ones(6),
        method="causal_hampel",
        window=5,
        threshold=3.0,
        mad_scale=1.4826,
    )
    np.testing.assert_allclose(constant, np.ones(6))
    assert not constant_audit["outlier_flag"].any()


def test_subject_record_and_gap_boundaries_reset_state() -> None:
    config = load_config(CONFIG)
    frame = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "source": ["gpn_data"] * 4,
            "subject_id": ["a", "a", "a", "b"],
            "record_id": ["gpn_data__a__r", "gpn_data__a__r", "gpn_data__a__r", "gpn_data__b__r"],
            "record_group_id": ["a__r", "a__r", "a__r", "b__r"],
            "t_start": [0.0, 10.0, 30.0, 0.0],
            **{f"target_{metric}": [1.0, 3.0, 100.0, 50.0] for metric in PM_METRICS},
        }
    )
    prepared = prepare_pm_frame(frame, config)
    variants = build_variants(prepared, config)
    median = variants.values["focus"]["causal_median_w3"]
    np.testing.assert_allclose(median, [1.0, 2.0, 100.0, 50.0])


def test_all_variants_preserve_identical_sample_and_missing_universe() -> None:
    config = load_config(CONFIG)
    frame = _synthetic_frame()
    frame.loc[[2, 9], "target_focus"] = np.nan
    prepared = prepare_pm_frame(frame, config)
    variants = build_variants(prepared, config)
    baseline_missing = np.isnan(variants.values["focus"]["baseline_raw"])
    for variant in VARIANT_ORDER:
        np.testing.assert_array_equal(
            np.isnan(variants.values["focus"][variant]), baseline_missing
        )


def test_q3_thresholds_are_fit_without_outer_test_targets() -> None:
    config = load_config(CONFIG)
    frame = prepare_pm_frame(_synthetic_frame(), config)
    folds = {f"s{subject:02d}": subject % 5 + 1 for subject in range(10)}
    first = calculate_q3_stability(frame, build_variants(frame, config), folds)[0]

    changed = frame.copy()
    test_subjects = {subject for subject, fold in folds.items() if fold == 1}
    test_mask = changed["subject_id"].isin(test_subjects)
    for column in (f"target_{metric}" for metric in PM_METRICS):
        changed.loc[test_mask, column] += 1000.0
    second = calculate_q3_stability(changed, build_variants(changed, config), folds)[0]
    columns = ["pm", "variant", "outer_fold", "q1", "q2", "transform_hash"]
    left = first.loc[first["outer_fold"].eq(1), columns].reset_index(drop=True)
    right = second.loc[second["outer_fold"].eq(1), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_plan_only_is_deterministic_and_does_not_write(tmp_path: Path) -> None:
    first = plan_analysis(CONFIG, output_dir=tmp_path / "not-created")
    second = plan_analysis(CONFIG, output_dir=tmp_path / "not-created")
    assert first == second
    assert first["writes_performed"] is False
    assert not (tmp_path / "not-created").exists()


def test_downstream_plan_supports_one_outer_fold_without_training() -> None:
    plan = plan_downstream(CONFIG, outer_folds=[1])
    assert plan["outer_folds"] == [1]
    assert plan["run_count"] == 7 * 4 * 1 * 4
    assert plan["models_trained"] == 0
    assert plan["hyperparameter_search"] is False


def test_downstream_manifest_propagates_config_result_status() -> None:
    config = load_config(FINAL_CONFIG)
    matrix = build_downstream_plan(config, outer_folds=[1])
    manifest = build_downstream_manifest(
        config,
        matrix,
        completed_run_count=len(matrix),
    )
    assert config["result_status"] == "confirmatory"
    assert manifest["result_status"] == config["result_status"]
    assert manifest["run_count"] == 56


def test_fold_specific_protocol_hash_is_expected() -> None:
    config = load_config(FINAL_CONFIG)
    first_matrix = build_downstream_plan(config, outer_folds=[1])
    second_matrix = build_downstream_plan(config, outer_folds=[2])
    first = build_downstream_manifest(
        config, first_matrix, completed_run_count=len(first_matrix)
    )
    second = build_downstream_manifest(
        config, second_matrix, completed_run_count=len(second_matrix)
    )
    assert first["fixed_outer_folds"] == [1]
    assert second["fixed_outer_folds"] == [2]
    assert first["run_matrix_hash"] != second["run_matrix_hash"]
    assert first["protocol_hash"] != second["protocol_hash"]


def _write_synthetic_completed_results(results_root: Path) -> None:
    config = load_config(FINAL_CONFIG)
    legacy_config = dict(config)
    legacy_config["result_status"] = "diagnostic"
    for fold in config["folds"]["fold_ids"]:
        fold = int(fold)
        fold_dir = results_root / f"fold{fold:02d}"
        fold_dir.mkdir(parents=True)
        matrix = build_downstream_plan(config, outer_folds=[fold])
        matrix.to_csv(fold_dir / "run_matrix.csv", index=False)
        matrix.loc[:, ["run_id", "specification_hash"]].to_csv(
            fold_dir / "summary.csv", index=False
        )
        manifest = build_downstream_manifest(
            legacy_config,
            matrix,
            completed_run_count=len(matrix),
        )
        (fold_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for row in matrix.to_dict(orient="records"):
            run_dir = fold_dir / str(row["run_id"])
            run_dir.mkdir()
            for artifact in (
                "metrics.json",
                "normalization_stats.json",
                "predictions.parquet",
                "split.json",
            ):
                (run_dir / artifact).touch()
            if row["task_type"] == "classification":
                (run_dir / "target_transform.json").touch()


def test_result_status_migration_changes_only_metadata_on_temp_copies(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    audit_path = tmp_path / "migration_audit.json"
    _write_synthetic_completed_results(results_root)
    before = {
        fold: (results_root / f"fold{fold:02d}" / "manifest.json").read_bytes()
        for fold in range(1, 6)
    }

    dry_run = migrate_result_status(
        config_path=FINAL_CONFIG,
        results_root=results_root,
        audit_path=audit_path,
        apply=False,
    )
    assert dry_run["writes_performed"] is False
    assert not audit_path.exists()
    assert all(
        (results_root / f"fold{fold:02d}" / "manifest.json").read_bytes()
        == before[fold]
        for fold in range(1, 6)
    )

    audit = migrate_result_status(
        config_path=FINAL_CONFIG,
        results_root=results_root,
        audit_path=audit_path,
        apply=True,
    )
    assert audit["writes_performed"] is True
    assert audit["changed_manifest_count"] == 5
    assert audit_path.is_file()
    for file_audit in audit["files"]:
        fold = int(file_audit["fold"])
        original = json.loads(before[fold])
        migrated = json.loads(
            (results_root / f"fold{fold:02d}" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        changed = {
            key
            for key in set(original) | set(migrated)
            if original.get(key) != migrated.get(key)
        }
        assert changed == {"result_status"}
        assert migrated["result_status"] == "confirmatory"
        assert migrated["protocol_hash"] == original["protocol_hash"]
        assert file_audit["original_sha256"] == hashlib.sha256(before[fold]).hexdigest()

    repeated = migrate_result_status(
        config_path=FINAL_CONFIG,
        results_root=results_root,
        apply=False,
    )
    assert repeated["writes_performed"] is False
    assert all(not item["changed"] for item in repeated["files"])
    assert all(
        item["protocol_hash_validation"] == "pre_migration_manifest"
        for item in repeated["files"]
    )


def test_result_status_migration_validates_all_folds_before_writing(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    _write_synthetic_completed_results(results_root)
    bad_manifest_path = results_root / "fold03" / "manifest.json"
    bad_manifest = json.loads(bad_manifest_path.read_text(encoding="utf-8"))
    bad_manifest["run_count"] = 55
    bad_manifest_path.write_text(
        json.dumps(bad_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run_count must be 56"):
        migrate_result_status(
            config_path=FINAL_CONFIG,
            results_root=results_root,
            audit_path=tmp_path / "audit.json",
            apply=True,
        )
    for fold in range(1, 6):
        manifest = json.loads(
            (results_root / f"fold{fold:02d}" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["result_status"] == "diagnostic"
