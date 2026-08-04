from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.core.abstract_dataset import EEGData
from bench.datasets.emotiv_loader import EmotivDataset
from bench.datasets.raw_eeg_window_dataset import RawEEGWindowDataset
from bench.datasets.target_view import (
    attach_targets_by_sample_id,
    build_feature_target_view,
    build_target_view,
    target_cohort_manifest,
)
from bench.tasks.target_registry import (
    PM_METRICS,
    PM_TARGET_COLUMNS,
    LegacyTargetConfigWarning,
    get_target_spec,
    list_target_specs,
    resolve_target_spec,
)
from bench.tasks.target_spec import TargetSpec
from bench.tasks.target_transforms import FoldLocalQuantileTargetTransform
from bench.tasks.tasks_registry import TASK_REGISTRY, get_task
from scripts.build_target_contract import build_contract


EXECUTABLE_IDS = {
    *(f"pm_{metric}_regression" for metric in PM_METRICS),
    "pm_multioutput_regression_7",
    "label_focus_q5_legacy",
}


def _canonical_frame(n: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": np.arange(n),
            "subject_id": np.repeat(["s1", "s2"], n // 2),
            "record_id": np.repeat(["r1", "r2"], n // 2),
            "source": "synthetic",
            "EEG.AF3.mean": np.arange(n, dtype=float),
            "EEG.AF4.std": np.arange(n, dtype=float) + 0.5,
            "POW.AF3.alpha": np.arange(n, dtype=float) + 10,
            "PM.Focus.Scaled": np.arange(n, dtype=float) + 100,
            "target_main": np.linspace(0.1, 0.9, n),
            "label_q5": np.arange(n) % 5,
        }
    )
    for offset, metric in enumerate(PM_METRICS):
        frame[f"target_{metric}"] = np.linspace(0.0, 1.0, n) + offset
    return frame


def _raw_fixture(tmp_path: Path, n: int = 10) -> tuple[Path, Path]:
    cache = tmp_path / "windows.npy"
    np.save(cache, np.zeros((n, 14, 2560), dtype=np.float32))
    frame = _canonical_frame(n)
    manifest = pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "source": "synthetic",
            "subject_id": frame["subject_id"],
            "record_id": frame["record_id"],
            "record_group_id": frame["record_id"],
            "t_start": np.arange(n, dtype=float),
            "t_end": np.arange(n, dtype=float) + 10.0,
            "label_q5": frame["label_q5"],
            "sfreq_original": 256.0,
            "sfreq_target": 256.0,
            "n_channels": 14,
            "n_samples_expected": 2560,
            "outer_fold": np.where(frame["subject_id"] == "s1", 1, 2),
            "status": "ok",
            "rejection_reason": "",
            "cache_file": str(cache),
            "cache_offset": np.arange(n),
            "missing_fraction": 0.0,
        }
    )
    manifest_path = tmp_path / "manifest.parquet"
    target_path = tmp_path / "targets.parquet"
    manifest.to_parquet(manifest_path, index=False)
    frame.to_parquet(target_path, index=False)
    return manifest_path, target_path


def test_registry_has_exact_executable_target_ids() -> None:
    assert {spec.target_id for spec in list_target_specs(executable_only=True)} == (
        EXECUTABLE_IDS
    )


@pytest.mark.parametrize("metric", PM_METRICS)
def test_registry_has_scalar_pm_contract(metric: str) -> None:
    spec = get_target_spec(f"pm_{metric}_regression")
    assert spec.processed_columns == (f"target_{metric}",)
    assert spec.output_dim == 1
    assert spec.raw_input_supported
    assert spec.task_type == "regression"


def test_registry_has_fixed_multioutput_order() -> None:
    spec = get_target_spec("pm_multioutput_regression_7")
    assert spec.processed_columns == PM_TARGET_COLUMNS
    assert spec.output_names == PM_METRICS
    assert spec.output_dim == 7


def test_legacy_label_has_explicit_status_and_physical_column() -> None:
    spec = get_target_spec("label_focus_q5_legacy")
    assert spec.processed_columns == ("label_q5",)
    assert spec.registry_status == "legacy_global_benchmark_label"
    assert spec.target_type == "legacy_classification"


@pytest.mark.parametrize(
    "target_id",
    [
        "pm_attention_active_proxy",
        "pm_focus_q3_fold_local",
        "pm_focus_q5_fold_local",
        "pm_activity_multilabel_7",
        "pm_long_term_excitement_regression",
    ],
)
def test_candidates_are_registered_but_disabled(target_id: str) -> None:
    spec = get_target_spec(target_id, require_executable=False)
    assert not spec.is_executable
    with pytest.raises(ValueError, match="registered but disabled"):
        get_target_spec(target_id)


def test_target_spec_rejects_output_contract_mismatch() -> None:
    payload = get_target_spec("pm_focus_regression").to_dict()
    payload["output_dim"] = 2
    payload["processed_columns"] = tuple(payload["processed_columns"])
    payload["output_names"] = tuple(payload["output_names"])
    payload["recommended_metrics"] = tuple(payload["recommended_metrics"])
    payload["allowed_feature_inputs"] = tuple(payload["allowed_feature_inputs"])
    with pytest.raises(ValueError, match="output_dim"):
        TargetSpec(**payload)


@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("label_q5", "label_focus_q5_legacy"),
        ("target_focus", "pm_focus_regression"),
        ("target_main", "pm_focus_regression"),
    ],
)
def test_legacy_aliases_warn_and_resolve(legacy: str, canonical: str) -> None:
    with pytest.warns(LegacyTargetConfigWarning):
        spec = resolve_target_spec({"target_col": legacy})
    assert spec.target_id == canonical


def test_multioutput_legacy_alias_warns_and_resolves() -> None:
    with pytest.warns(LegacyTargetConfigWarning):
        spec = resolve_target_spec({"target_cols": list(PM_TARGET_COLUMNS)})
    assert spec.target_id == "pm_multioutput_regression_7"


def test_no_implicit_target_main_fallback() -> None:
    with pytest.raises(ValueError, match="explicit target_id"):
        resolve_target_spec({})


def test_target_id_cannot_be_combined_with_legacy_fields() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_target_spec(
            {"target_id": "pm_focus_regression", "target_col": "target_focus"}
        )


def test_fold_local_transform_requires_fit_and_valid_shape() -> None:
    transform = FoldLocalQuantileTargetTransform(5)
    with pytest.raises(RuntimeError, match="fit before transform"):
        transform.transform(np.arange(5))
    with pytest.raises(ValueError, match="shape"):
        transform.fit(np.zeros((5, 1)))


@pytest.mark.parametrize("q", [3, 5])
def test_fold_local_transform_is_deterministic_and_outer_train_only(q: int) -> None:
    train = np.linspace(0.0, 1.0, 101)
    first = FoldLocalQuantileTargetTransform(q).fit(train)
    second = FoldLocalQuantileTargetTransform(q).fit(train.copy())
    assert first.manifest() == second.manifest()
    assert first.manifest()["fit_scope"] == "outer_train_only"
    assert first.manifest()["fit_sample_count"] == 101
    np.testing.assert_array_equal(first.transform(train), second.transform(train))


def test_fold_local_transform_reports_duplicate_boundaries() -> None:
    transform = FoldLocalQuantileTargetTransform(5, duplicates="drop").fit(
        np.asarray([0.0] * 8 + [1.0] * 2)
    )
    assert transform.actual_class_count < 5
    assert transform.manifest()["duplicates"] == "drop"
    with pytest.raises(ValueError, match="not unique"):
        FoldLocalQuantileTargetTransform(5, duplicates="raise").fit(
            np.asarray([0.0] * 8 + [1.0] * 2)
        )


def test_target_view_filters_missing_and_preserves_sample_order() -> None:
    frame = _canonical_frame()
    frame.loc[[2, 7], "target_focus"] = np.nan
    view = build_target_view(frame, get_target_spec("pm_focus_regression"))
    assert view.cohort.availability_mask.tolist() == [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
    ]
    assert view.sample_ids.tolist() == [0, 1, 3, 4, 5, 6, 8, 9]
    assert view.targets.dtype == np.float32


def test_multioutput_view_uses_complete_cases_and_fixed_order() -> None:
    frame = _canonical_frame()
    frame.loc[3, "target_stress"] = np.nan
    view = build_target_view(frame, get_target_spec("pm_multioutput_regression_7"))
    assert view.targets.shape == (9, 7)
    assert view.targets.dtype == np.float32
    assert view.sample_ids.tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 9]


def test_classification_view_is_integer_and_rejects_fractional_labels() -> None:
    frame = _canonical_frame()
    spec = get_target_spec("label_focus_q5_legacy")
    view = build_target_view(frame, spec)
    assert view.targets.dtype == np.int64
    frame["label_q5"] = frame["label_q5"].astype(float)
    frame.loc[0, "label_q5"] = 0.5
    with pytest.raises(ValueError, match="non-integer"):
        build_target_view(frame, spec)


@pytest.mark.parametrize(
    ("feature_set", "expected_width"),
    [("eeg", 2), ("pow", 1), ("eeg_pow", 3), ("pow_plus_eeg", 3)],
)
def test_feature_view_selects_only_approved_inputs(
    feature_set: str, expected_width: int
) -> None:
    view = build_feature_target_view(
        _canonical_frame(), get_target_spec("pm_focus_regression"), feature_set
    )
    assert view.features.shape == (10, expected_width)
    assert not any(
        name.startswith(("PM.", "target_", "label_"))
        for name in view.feature_names
    )


def test_feature_view_combines_feature_and_target_availability() -> None:
    frame = _canonical_frame()
    frame.loc[1, "EEG.AF3.mean"] = np.nan
    frame.loc[2, "target_focus"] = np.nan
    view = build_feature_target_view(
        frame, get_target_spec("pm_focus_regression"), "eeg_pow"
    )
    assert view.availability_mask.sum() == 8
    assert view.target_view.sample_ids.tolist() == [0, 3, 4, 5, 6, 7, 8, 9]


def test_attach_targets_preserves_order_and_checks_identifiers() -> None:
    frame = _canonical_frame(10)
    windows = frame[["sample_id", "subject_id", "record_id"]].iloc[::-1].copy()
    attached = attach_targets_by_sample_id(
        windows, frame, get_target_spec("pm_focus_regression")
    )
    assert attached["sample_id"].tolist() == windows["sample_id"].tolist()
    bad = frame.copy()
    bad.loc[0, "subject_id"] = "wrong"
    with pytest.raises(ValueError, match="identifiers disagree"):
        attach_targets_by_sample_id(
            windows, bad, get_target_spec("pm_focus_regression")
        )


def test_target_cohort_uses_fixed_outer_folds() -> None:
    frame = _canonical_frame()
    frame.loc[0, "target_focus"] = np.nan
    manifest = target_cohort_manifest(
        frame,
        get_target_spec("pm_focus_regression"),
        {"s1": 1, "s2": 2},
    )
    assert manifest["outer_fold"].tolist() == [1, 2]
    assert manifest["n_samples"].tolist() == [4, 5]


def test_emotiv_loader_accepts_explicit_scalar_target_id(tmp_path: Path) -> None:
    path = tmp_path / "feature.parquet"
    _canonical_frame().to_parquet(path, index=False)
    data = EmotivDataset(
        {
            "data_path": str(path),
            "target_id": "pm_attention_regression",
            "feature_set": "eeg_pow",
            "max_features": None,
        }
    ).load()
    assert data.labels.shape == (10,)
    assert data.labels.dtype == np.float32
    assert data.metadata["target_id"] == "pm_attention_regression"


def test_emotiv_loader_accepts_explicit_multioutput_target_id(tmp_path: Path) -> None:
    path = tmp_path / "feature.parquet"
    _canonical_frame().to_parquet(path, index=False)
    data = EmotivDataset(
        {
            "data_path": str(path),
            "target_id": "pm_multioutput_regression_7",
            "feature_set": "eeg_pow",
            "max_features": None,
        }
    ).load()
    assert data.labels.shape == (10, 7)
    assert data.metadata["target_output_names"] == list(PM_METRICS)


@pytest.mark.parametrize(
    "target_id",
    [
        *(f"pm_{metric}_regression" for metric in PM_METRICS),
        "pm_multioutput_regression_7",
        "label_focus_q5_legacy",
    ],
)
def test_raw_loader_supports_every_executable_target(
    tmp_path: Path, target_id: str
) -> None:
    manifest_path, target_path = _raw_fixture(tmp_path)
    data = RawEEGWindowDataset(
        {
            "data_path": str(manifest_path),
            "target_data_path": str(target_path),
            "target_id": target_id,
        }
    ).load()
    assert data.data.shape == (10, 1, 14, 2560)
    assert data.metadata["target_id"] == target_id
    if target_id == "pm_multioutput_regression_7":
        assert data.labels.shape == (10, 7)
        assert data.labels.dtype == np.float32
    elif target_id == "label_focus_q5_legacy":
        assert data.labels.shape == (10,)
        assert data.labels.dtype == np.int64
    else:
        assert data.labels.shape == (10,)
        assert data.labels.dtype == np.float32


def test_raw_loader_filters_missing_target_without_zero_fill(tmp_path: Path) -> None:
    manifest_path, target_path = _raw_fixture(tmp_path)
    target_frame = pd.read_parquet(target_path)
    target_frame.loc[4, "target_focus"] = np.nan
    target_frame.to_parquet(target_path, index=False)
    data = RawEEGWindowDataset(
        {
            "data_path": str(manifest_path),
            "target_data_path": str(target_path),
            "target_id": "pm_focus_regression",
        }
    ).load()
    assert data.n_samples == 9
    assert 4 not in data.sample_ids
    assert np.isfinite(data.labels).all()
    assert data.metadata["dropped_target_rows"] == 1


def test_raw_legacy_default_warns_instead_of_target_main_fallback(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _raw_fixture(tmp_path)
    with pytest.warns(LegacyTargetConfigWarning):
        data = RawEEGWindowDataset({"data_path": str(manifest_path)}).load()
    assert data.metadata["target_id"] == "label_focus_q5_legacy"


def test_new_task_ids_use_existing_task_implementations() -> None:
    for target_id in EXECUTABLE_IDS:
        assert target_id in TASK_REGISTRY
    labels = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    data = EEGData(
        data=np.zeros((10, 2), dtype=np.float32),
        labels=labels,
        subject_ids=np.repeat(["s1", "s2"], 5),
        feature_names=["a", "b"],
        metadata={"target_id": "pm_attention_regression"},
    )
    task = get_task("pm_attention_regression", data, {})
    assert task.task_type == "regression"


def test_task_registry_rejects_target_mismatch() -> None:
    data = EEGData(
        data=np.zeros((10, 2), dtype=np.float32),
        labels=np.linspace(0.0, 1.0, 10),
        subject_ids=np.repeat(["s1", "s2"], 5),
        feature_names=["a", "b"],
        metadata={"target_id": "pm_focus_regression"},
    )
    with pytest.raises(ValueError, match="requires target_id"):
        get_task("pm_attention_regression", data, {})


def test_plan_only_writes_nothing(tmp_path: Path) -> None:
    source_path = Path("experiments/targets/canonical_target_contract.yaml")
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["registry_path"] = str(Path(config["registry_path"]).resolve())
    config["feature_dataset_path"] = str(
        Path(config["feature_dataset_path"]).resolve()
    )
    config["raw_manifest_path"] = str(Path(config["raw_manifest_path"]).resolve())
    config["logical_recording_map_path"] = str(
        Path(config["logical_recording_map_path"]).resolve()
    )
    config["output_dir"] = str(tmp_path / "must_not_exist")
    config["report_path"] = str(tmp_path / "must_not_exist.md")
    config_path = tmp_path / "contract.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    manifest = build_contract(config_path, plan_only=True)

    assert manifest["decision"] == "canonical_target_contract_ready"
    assert not (tmp_path / "must_not_exist").exists()
    assert not (tmp_path / "must_not_exist.md").exists()
    assert "target_contract_manifest.json" in manifest["planned_artifacts"]
