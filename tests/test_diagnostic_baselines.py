from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

from bench.analysis.diagnostic_baselines import (
    DIAGNOSTIC_FEATURES,
    FORBIDDEN_FEATURE_COLUMNS,
    align_with_canonical_predictions,
    assign_subject_folds,
    run_diagnostic_baselines,
)
from bench.analysis.temporal_target_structure import (
    TemporalTargetAudit,
    prepare_temporal_frame,
)


def _source_frame(n_subjects: int = 10, windows: int = 25) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_id = 0
    for subject_index in range(n_subjects):
        source = "gpn_data" if subject_index % 2 == 0 else "Old_EEG"
        subject = f"subject_{subject_index:02d}"
        for window in range(windows):
            label = (window + subject_index) % 5
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "subject_id": subject,
                    "record_id": f"record_{subject}",
                    "t_start": float(window * 10),
                    "t_end": float((window + 1) * 10),
                    "target_focus": float(label) / 5.0 + window * 0.0001,
                    "label_q5": label,
                }
            )
            sample_id += 1
    return pd.DataFrame(rows)


def _prepared() -> pd.DataFrame:
    prepared = prepare_temporal_frame(_source_frame())
    return assign_subject_folds(prepared, n_splits=5)[0]


def _small_spec() -> dict[str, object]:
    return {
        "n_splits": 5,
        "random_state": 42,
        "logistic_regression": {"max_iter": 100},
        "random_forest": {
            "n_estimators": 3,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "n_jobs": 1,
        },
    }


def test_group_folds_are_deterministic_and_subject_disjoint() -> None:
    prepared = prepare_temporal_frame(_source_frame())
    first, metadata = assign_subject_folds(prepared, n_splits=5)
    second, _ = assign_subject_folds(prepared, n_splits=5)
    np.testing.assert_array_equal(first["outer_fold"], second["outer_fold"])
    assert sorted(first["outer_fold"].unique().tolist()) == [1, 2, 3, 4, 5]
    for fold in metadata["folds"].values():
        assert fold["subject_overlap"] == []
        assert set(fold["train_subjects"]).isdisjoint(fold["test_subjects"])


def test_canonical_alignment_checks_ids_folds_subjects_and_targets(tmp_path: Path) -> None:
    folded = _prepared()
    reference = folded[
        ["sample_id", "outer_fold", "subject_id", "label_q5"]
    ].rename(columns={"outer_fold": "fold", "label_q5": "y_true"})
    path = tmp_path / "reference.parquet"
    reference.to_parquet(path, index=False)
    result = align_with_canonical_predictions(folded, path)
    assert result["exact_match"] is True
    assert result["mismatches"] == {
        "sample_id": 0,
        "fold": 0,
        "subject_id": 0,
        "y_true": 0,
    }

    broken = reference.copy()
    broken.loc[0, "fold"] = 5 if broken.loc[0, "fold"] != 5 else 4
    broken.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Canonical alignment failed"):
        align_with_canonical_predictions(folded, path)


def test_baselines_use_only_safe_features_and_train_partition_statistics() -> None:
    folded = _prepared()
    original = folded.copy(deep=True)
    fold_one_test = folded["outer_fold"] == 1
    folded.loc[fold_one_test, "record_duration"] = 10000.0
    predictions, metrics = run_diagnostic_baselines(
        folded, spec=_small_spec()
    )

    assert predictions["prediction_id"].is_unique
    assert len(metrics["overall"]) == 7
    assert metrics["identity_alignment"]["exact_match"] is True
    assert metrics["identity_alignment"]["rows_per_variant"] == len(folded)
    assert metrics["feature_policy"]["eeg_features"] == []
    assert metrics["feature_policy"]["pow_features"] == []
    source_rows = metrics["by_source"]
    assert {row["source"] for row in source_rows} == {"Old_EEG", "gpn_data"}
    for feature_columns in DIAGNOSTIC_FEATURES.values():
        assert set(feature_columns).isdisjoint(FORBIDDEN_FEATURE_COLUMNS)
        assert "subject_id" not in feature_columns
        assert "record_id" not in feature_columns

    audit = next(
        row
        for row in metrics["fit_audit"]
        if row["fold"] == 1
        and row["diagnostic_set"] == "D2"
        and row["model"] == "logistic_regression"
    )
    outer_train = folded.loc[~fold_one_test]
    assert audit["subject_overlap"] == []
    assert audit["train_rows_fit"] == len(outer_train)
    assert audit["preprocessor_fit_partition"] == "outer_train"
    for column in DIAGNOSTIC_FEATURES["D2"]:
        assert audit["scaler_mean"][column] == pytest.approx(
            outer_train[column].mean()
        )
    assert audit["scaler_mean"]["record_duration"] != pytest.approx(
        folded["record_duration"].mean()
    )
    # The caller's input is unchanged except for the deliberate test-only mutation.
    expected = original.copy()
    expected.loc[fold_one_test, "record_duration"] = 10000.0
    pdt.assert_frame_equal(folded, expected)


def test_outer_train_majority_is_recomputed_for_each_fold() -> None:
    folded = _prepared()
    predictions, metrics = run_diagnostic_baselines(folded, spec=_small_spec())
    majority = predictions.loc[predictions["diagnostic_set"] == "D0"]
    for fold in range(1, 6):
        train = folded.loc[folded["outer_fold"] != fold, "label_q5"].to_numpy(dtype=int)
        expected = int(np.argmax(np.bincount(train, minlength=5)))
        fold_predictions = majority.loc[majority["fold"] == fold, "y_pred"]
        assert fold_predictions.nunique() == 1
        assert int(fold_predictions.iloc[0]) == expected
    assert all(
        row["majority_from_outer_train_only"]
        for row in metrics["fit_audit"]
        if row["diagnostic_set"] == "D0"
    )


def test_full_temporal_audit_writes_expected_artifacts_without_mutating_input(
    tmp_path: Path,
) -> None:
    source = _source_frame()
    # Omit sample_id to exercise the same positional convention as the real Parquet.
    source_without_id = source.drop(columns="sample_id")
    data_path = tmp_path / "dataset.parquet"
    source_without_id.to_parquet(data_path, index=False)
    before = sha256(data_path.read_bytes()).hexdigest()
    prepared = prepare_temporal_frame(source_without_id)
    folded, _ = assign_subject_folds(prepared, n_splits=5)
    reference = folded[
        ["sample_id", "outer_fold", "subject_id", "label_q5"]
    ].rename(columns={"outer_fold": "fold", "label_q5": "y_true"})
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)

    output_dir = tmp_path / "generated"
    report_dir = tmp_path / "reports"
    spec_path = tmp_path / "audit.yaml"
    spec = {
        "audit": {
            "name": "synthetic_temporal_audit",
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "temporal_report": str(report_dir / "temporal.md"),
            "diagnostic_report": str(report_dir / "diagnostic.md"),
            "summary_path": str(report_dir / "summary.json"),
            "expected_supervised_rows": len(source),
            "canonical_reference_predictions": str(reference_path),
            "lags": [1, 2, 3, 5, 10, 20],
        },
        "blocked_time": {"early_end": 0.4, "late_start": 0.6},
        "diagnostic_baselines": _small_spec(),
    }
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    result = TemporalTargetAudit(spec_path).execute()

    assert result["deep_models_trained"] == 0
    assert result["eeg_or_pow_features_used"] is False
    assert result["canonical_alignment"]["exact_match"] is True
    assert result["input_modified"] is False
    assert result["input_sha256"] == result["input_sha256_after"] == before
    assert (output_dir / "temporal_statistics.json").is_file()
    predictions = pd.read_parquet(output_dir / "diagnostic_predictions.parquet")
    assert predictions["prediction_id"].is_unique
    assert (output_dir / "diagnostic_metrics.json").is_file()
    assert (report_dir / "temporal.md").is_file()
    assert (report_dir / "diagnostic.md").is_file()
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["supervised_rows"] == len(source)
    assert sha256(data_path.read_bytes()).hexdigest() == before
