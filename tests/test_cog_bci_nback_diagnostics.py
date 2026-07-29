from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.analysis.cog_bci_nback_diagnostics import (
    _runtime_report,
    aggregate_spectral_records,
    analyze_existing_predictions,
    build_duration_audit,
    build_task_boundary_masks,
    build_within_subject_rotations,
    evaluate_subject_disjoint,
    evaluate_within_subject,
    manifest_hashes,
    spectral_diagnostic_features,
    spectral_features,
)
from bench.datasets.cog_bci_baseline_dataset import (
    PerWindowCenteredRawEEGWindowArrayView,
)


def _boundary_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted_rows = []
    all_rows = []
    events = []
    for record_index, (record_id, target) in enumerate((("r0", 0), ("r1", 1))):
        for window_index in range(4):
            row = {
                "sample_id": f"{record_id}-w{window_index}",
                "record_id": record_id,
                "record_group_id": record_id,
                "subject_id": f"s{record_index}",
                "session_id": "ses-01",
                "task_variant": f"class-{target}",
                "target": target,
                "class_name": f"class-{target}",
                "outer_fold": record_index + 1,
                "window_index": window_index,
                "start_time_seconds": window_index * 5.12,
                "stop_time_seconds": (window_index + 1) * 5.12,
                "start_sample": window_index * 2560,
                "stop_sample": (window_index + 1) * 2560,
                "valid_stop_sample": (window_index + 1) * 2560,
                "sampling_rate_hz": 500.0,
                "status": "accepted",
            }
            accepted_rows.append(row)
            all_rows.append(row)
        rejected = dict(all_rows[-1])
        rejected.update(
            {
                "sample_id": f"{record_id}-tail",
                "window_index": 4,
                "start_time_seconds": 20.48,
                "stop_time_seconds": 25.60,
                "start_sample": 10240,
                "stop_sample": 12800,
                "valid_stop_sample": 11250,
                "status": "rejected",
            }
        )
        all_rows.append(rejected)
        events.extend(
            [
                {
                    "record_id": record_id,
                    "subject_id": f"s{record_index}",
                    "session_id": "ses-01",
                    "task_variant": f"class-{target}",
                    "event_index": 0,
                    "onset_seconds": 0.0,
                    "duration_seconds": 0.0,
                    "description": "boundary",
                    "is_boundary": True,
                    "is_task_start": False,
                    "is_task_end": False,
                },
                {
                    "record_id": record_id,
                    "subject_id": f"s{record_index}",
                    "session_id": "ses-01",
                    "task_variant": f"class-{target}",
                    "event_index": 1,
                    "onset_seconds": 2.0,
                    "duration_seconds": 0.0,
                    "description": "trial",
                    "is_boundary": False,
                    "is_task_start": False,
                    "is_task_end": False,
                },
                {
                    "record_id": record_id,
                    "subject_id": f"s{record_index}",
                    "session_id": "ses-01",
                    "task_variant": f"class-{target}",
                    "event_index": 2,
                    "onset_seconds": 21.0,
                    "duration_seconds": 0.0,
                    "description": f"end-{target}",
                    "is_boundary": False,
                    "is_task_start": False,
                    "is_task_end": True,
                },
            ]
        )
    return (
        pd.DataFrame(accepted_rows),
        pd.DataFrame(all_rows),
        pd.DataFrame(events),
    )


def _record_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for subject_index in range(15):
        subject = f"sub-{subject_index:02d}"
        outer_fold = subject_index % 5 + 1
        for session_index in range(3):
            for target in range(3):
                record_id = f"{subject}-ses-{session_index}-class-{target}"
                rows.append(
                    {
                        "record_id": record_id,
                        "subject_id": subject,
                        "session_id": f"ses-{session_index + 1:02d}",
                        "target": target,
                        "class_name": str(target),
                        "outer_fold": outer_fold,
                        "mean__f0": target + subject_index * 0.01,
                        "median__f0": target + session_index * 0.02,
                        "std__f0": 0.1 + target * 0.01,
                    }
                )
    records = pd.DataFrame(rows)
    inner_rows = []
    for fold in range(1, 6):
        outer_train_subjects = sorted(
            set(records.loc[records.outer_fold.ne(fold), "subject_id"])
        )
        validation_subjects = set(outer_train_subjects[:3])
        for row in records.loc[records.outer_fold.ne(fold)].itertuples():
            inner_rows.append(
                {
                    "outer_fold": fold,
                    "record_id": row.record_id,
                    "subject_id": row.subject_id,
                    "partition": (
                        "inner_validation"
                        if row.subject_id in validation_subjects
                        else "inner_train"
                    ),
                }
            )
    return records, pd.DataFrame(inner_rows)


def test_boundary_masks_are_record_and_class_local() -> None:
    accepted, all_windows, events = _boundary_frames()
    masks, audit, summary = build_task_boundary_masks(
        accepted, all_windows, events
    )
    assert len(masks) == len(accepted)
    assert masks.groupby("record_id")["target"].nunique().eq(1).all()
    assert masks["record_full"].all()
    assert masks["to_end_marker"].all()
    assert masks.groupby("record_id")["exclude_first_5s_to_end"].sum().eq(3).all()
    assert masks.groupby("record_id")["exclude_first_10s_to_end"].sum().eq(2).all()
    assert audit["task_end_marker_count"].eq(1).all()
    assert set(summary["target"]) == {0, 1}


def test_ambiguous_end_marker_fails() -> None:
    accepted, all_windows, events = _boundary_frames()
    duplicate = events.loc[events["is_task_end"]].iloc[[0]].copy()
    duplicate["event_index"] = 3
    with pytest.raises(ValueError, match="exactly one"):
        build_task_boundary_masks(
            accepted,
            all_windows,
            pd.concat([events, duplicate], ignore_index=True),
        )


def test_spectral_features_are_finite_and_deterministic() -> None:
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(5, 2, 1000)).astype(np.float32)
    first, names = spectral_features(
        windows, sampling_rate=250.0, channel_names=["C1", "C2"]
    )
    second, second_names = spectral_features(
        windows, sampling_rate=250.0, channel_names=["C1", "C2"]
    )
    assert first.shape == (5, 16)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    assert names == second_names
    diagnostics = spectral_diagnostic_features(
        windows, sampling_rate=250.0
    )
    assert diagnostics.shape == (5, 5)
    assert np.isfinite(diagnostics.to_numpy()).all()


def test_record_aggregation_does_not_mix_records() -> None:
    rows = []
    for record_id, target in (("r0", 0), ("r1", 1)):
        for window in range(2):
            rows.append(
                {
                    "record_id": record_id,
                    "subject_id": record_id,
                    "session_id": "ses-01",
                    "target": target,
                    "class_name": str(target),
                    "outer_fold": target + 1,
                    "f": float(target + window),
                }
            )
    result = aggregate_spectral_records(pd.DataFrame(rows), ["f"])
    assert len(result) == 2
    assert result["record_id"].is_unique
    assert result.set_index("record_id").loc["r0", "mean__f"] == 0.5
    assert result.set_index("record_id").loc["r1", "mean__f"] == 1.5


def test_subject_disjoint_uses_exact_inner_train_scaler() -> None:
    records, inner = _record_features()
    predictions, folds, audit = evaluate_subject_disjoint(
        records, inner, seed=42
    )
    assert predictions.groupby("model").size().eq(len(records)).all()
    assert predictions.groupby("model")["record_id"].nunique().eq(len(records)).all()
    assert set(folds["fold"]) == {1, 2, 3, 4, 5}
    assert all(
        row["fit_partition"] == "inner_train"
        and not row["outer_test_used_for_fit"]
        for row in audit["scaler_audit"]
    )


def test_within_subject_rotations_have_only_intentional_subject_overlap() -> None:
    records, _ = _record_features()
    rotations = build_within_subject_rotations(records)
    assert [row["held_out_session"] for row in rotations] == [
        "ses-01",
        "ses-02",
        "ses-03",
    ]
    assert all(row["subject_overlap"] == 15 for row in rotations)
    assert all(row["record_overlap"] == 0 for row in rotations)
    predictions, metrics, summary = evaluate_within_subject(records, seed=42)
    assert predictions.groupby("model").size().eq(len(records)).all()
    assert metrics["record_overlap"].eq(0).all()
    assert metrics["sample_overlap"].eq(0).all()
    assert len(summary["rotations"]) == 3


def test_duration_audit_is_deterministic() -> None:
    accepted, all_windows, events = _boundary_frames()
    _, boundary, _ = build_task_boundary_masks(accepted, all_windows, events)
    first, first_summary = build_duration_audit(accepted, boundary)
    second, second_summary = build_duration_audit(accepted, boundary)
    pd.testing.assert_frame_equal(first, second)
    assert json.dumps(first_summary, sort_keys=True) == json.dumps(
        second_summary, sort_keys=True
    )


def _write_prediction_fixture(root: Path) -> pd.DataFrame:
    records, _ = _record_features()
    records = records.head(9).copy()
    audit = records[
        ["record_id", "subject_id", "session_id", "target"]
    ].copy()
    audit["duration_seconds"] = np.linspace(100, 180, len(audit))
    audit["duration_group"] = ["short"] * 3 + ["medium"] * 3 + ["long"] * 3
    for directory, offset in (
        ("eegnet_seed42", 0),
        ("shallowconvnet_seed42", 1),
    ):
        probabilities = np.full((len(records), 3), 0.2)
        predicted = (records["target"].to_numpy() + offset) % 3
        probabilities[np.arange(len(records)), predicted] = 0.6
        frame = pd.DataFrame(
            {
                "record_id": records["record_id"],
                "subject_id": records["subject_id"],
                "session_id": records["session_id"],
                "true_class": records["target"],
                "predicted_class": predicted,
                "window_count": 2,
                "fold_id": 1,
            }
        )
        for class_id in range(3):
            frame[f"mean_probability_class_{class_id}"] = probabilities[:, class_id]
        path = root / directory
        path.mkdir(parents=True)
        frame.to_parquet(path / "record_predictions.parquet", index=False)
    return audit


def test_prediction_audit_preserves_artifacts_and_checks_probabilities(
    tmp_path: Path,
) -> None:
    audit = _write_prediction_fixture(tmp_path)
    paths = list(tmp_path.rglob("*.parquet"))
    before = manifest_hashes({str(index): path for index, path in enumerate(paths)})
    confidence, agreement, summary = analyze_existing_predictions(
        tmp_path, audit
    )
    after = manifest_hashes({str(index): path for index, path in enumerate(paths)})
    assert before == after
    assert len(confidence) == 18
    assert len(agreement) == 9
    assert confidence["record_id"].notna().all()
    assert 0 <= summary["agreement_fraction"] <= 1


def test_manifest_hashes_detect_no_read_side_effect(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")
    first = manifest_hashes({"manifest": path})
    second = manifest_hashes({"manifest": path})
    assert first == second


def test_runtime_report_contains_no_local_absolute_path() -> None:
    metrics = {
        "multinomial_logistic_regression": {
            "balanced_accuracy": 0.4,
            "macro_f1": 0.39,
            "ordinal_mae": 0.8,
        }
    }
    report = _runtime_report(
        {
            "result_status": "diagnostic",
            "unit_audit": {
                "physical_unit": "unresolved",
                "mne_applied_factor": 1e-6,
            },
            "dataset": {"windows": 9, "records": 3},
            "subject_disjoint": {"aggregate_metrics": metrics},
            "within_subject": {"aggregate_metrics": metrics},
        }
    )
    assert "F:\\" not in report
    assert "C:\\" not in report


def test_per_window_centering_is_lazy_and_preserves_record_identity(
    tmp_path: Path,
) -> None:
    array = np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8)
    path = tmp_path / "record.npy"
    np.save(path, array)
    manifest = pd.DataFrame(
        {
            "sample_id": ["w0", "w1"],
            "record_id": ["record", "record"],
            "cache_file": [str(path), str(path)],
            "cache_offset": [0, 1],
            "n_channels": [2, 2],
            "n_samples_expected": [8, 8],
            "status": ["ok", "ok"],
        }
    )
    view = PerWindowCenteredRawEEGWindowArrayView(manifest)
    assert np.allclose(view[0].mean(axis=-1), 0.0)
    subset = view[[1]]
    assert isinstance(subset, PerWindowCenteredRawEEGWindowArrayView)
    assert subset.manifest["record_id"].tolist() == ["record"]
    assert np.allclose(subset[0].mean(axis=-1), 0.0)
