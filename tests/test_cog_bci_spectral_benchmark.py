from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.datasets.channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
from bench.experiments.cog_bci_spectral_benchmark import (
    _metrics,
    _model_grid,
    _relative_path,
    _validate_window_identity,
    input_hashes,
    resolve_paths,
    run_nested_benchmark,
    subject_bootstrap_differences,
    subject_bootstrap_indices,
)
from bench.features.cog_bci_spectral_features import (
    NUISANCE_FEATURE_TYPES,
    SpectralFeatureSpec,
    aggregate_record_features,
    extract_spectral_feature_bundle,
    feature_columns_for,
)


def _windows(channels: int, *, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(512, dtype=np.float64) / 500.0
    base = np.sin(2 * np.pi * 10.0 * time) + 0.2 * np.sin(
        2 * np.pi * 50.0 * time
    )
    return (
        base[None, None, :]
        + rng.normal(scale=0.1, size=(4, channels, len(time)))
        + np.arange(channels)[None, :, None] * 0.01
    ).astype(np.float32)


def _window_metadata(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s-{index}" for index in range(rows)],
            "record_id": ["r-1"] * (rows // 2) + ["r-2"] * (rows - rows // 2),
            "subject_id": ["u-1"] * (rows // 2) + ["u-2"] * (rows - rows // 2),
            "session_id": ["a"] * rows,
            "target": [0] * (rows // 2) + [1] * (rows - rows // 2),
            "class_name": ["zero"] * (rows // 2) + ["one"] * (rows - rows // 2),
            "outer_fold": [1] * (rows // 2) + [2] * (rows - rows // 2),
        }
    )


def _synthetic_records() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rows = []
    inner_rows = []
    for subject_index in range(15):
        subject = f"sub-{subject_index:02d}"
        outer_fold = subject_index % 5 + 1
        for target in range(3):
            record = f"{subject}-class-{target}"
            signal = target * 0.8 + subject_index * 0.01
            rows.append(
                {
                    "record_id": record,
                    "subject_id": subject,
                    "session_id": f"ses-{target + 1}",
                    "target": target,
                    "class_name": str(target),
                    "outer_fold": outer_fold,
                    "window_count": 2,
                    "record_mean__cw__C1__log_power_delta": signal,
                    "record_mean__cw__C1__dc_magnitude": subject_index,
                    "record_mean__gs__log_power_delta__mean": signal,
                    "record_mean__gs__dc_magnitude__mean": subject_index,
                }
            )
    frame = pd.DataFrame(rows)
    for fold in range(1, 6):
        test_subjects = set(frame.loc[frame["outer_fold"].eq(fold), "subject_id"])
        remaining = sorted(set(frame["subject_id"]) - test_subjects)
        validation_subjects = {remaining[0]}
        for row in frame.itertuples(index=False):
            if row.subject_id in test_subjects:
                partition = "outer_test_excluded"
            elif row.subject_id in validation_subjects:
                partition = "inner_validation"
            else:
                partition = "inner_train"
            inner_rows.append(
                {
                    "outer_fold": fold,
                    "record_id": row.record_id,
                    "subject_id": row.subject_id,
                    "partition": partition,
                }
            )
    second = frame.copy()
    second["record_mean__cw__C1__log_power_delta"] += 0.001
    second["record_mean__gs__log_power_delta__mean"] += 0.001
    return (
        {"emotiv_common": frame, "cog_bci_common": second},
        pd.DataFrame(inner_rows),
    )


def _small_config() -> dict:
    return {
        "seed": 42,
        "feature_sets": ["spectral_only", "spectral_plus_nuisance"],
        "representations": ["channel_wise"],
        "models": {
            "enabled": ["multinomial_logistic_regression"],
            "logistic_regression": {"C": [0.01, 0.1]},
            "hist_gradient_boosting": {
                "seeds": [42],
                "grid": [
                    {
                        "learning_rate": 0.05,
                        "max_iter": 5,
                        "max_leaf_nodes": 7,
                        "l2_regularization": 0.001,
                    }
                ],
            },
        },
    }


def test_feature_bundle_supports_14_and_62_channels_and_is_finite() -> None:
    spec = SpectralFeatureSpec(nperseg=256, noverlap=128)
    for channels in (14, 62):
        names = [f"C{index}" for index in range(channels)]
        bundle = extract_spectral_feature_bundle(
            _windows(channels),
            channel_names=names,
            spec=spec,
        )
        assert bundle.channel_wise.shape == (4, channels * 16)
        assert bundle.global_summary.shape == (4, 16 * 5)
        assert np.isfinite(bundle.channel_wise).all()
        assert np.isfinite(bundle.global_summary).all()


def test_channel_contracts_exclude_ecg1_and_cz_and_preserve_order() -> None:
    common = json.loads(
        Path(
            "benchmark_results/cog_bci_channel_audit/cog_bci_common_channels.json"
        ).read_text(encoding="utf-8")
    )["common_eeg_channel_order"]
    assert len(PROJECT_EMOTIV_CHANNEL_ORDER) == 14
    assert len(common) == 62
    assert "ECG1" not in common
    assert "Cz" not in common
    assert common[:5] == ["Fp1", "Fz", "F3", "F7", "FT9"]


def test_spectral_extraction_is_deterministic_and_order_sensitive() -> None:
    spec = SpectralFeatureSpec(nperseg=256, noverlap=128)
    windows = _windows(14)
    first = extract_spectral_feature_bundle(
        windows, channel_names=PROJECT_EMOTIV_CHANNEL_ORDER, spec=spec
    )
    second = extract_spectral_feature_bundle(
        windows, channel_names=PROJECT_EMOTIV_CHANNEL_ORDER, spec=spec
    )
    np.testing.assert_array_equal(first.channel_wise, second.channel_wise)
    assert first.channel_wise_columns == second.channel_wise_columns
    reversed_bundle = extract_spectral_feature_bundle(
        windows[:, ::-1],
        channel_names=tuple(reversed(PROJECT_EMOTIV_CHANNEL_ORDER)),
        spec=spec,
    )
    assert first.channel_wise_columns != reversed_bundle.channel_wise_columns


def test_feature_extraction_rejects_nonfinite_values() -> None:
    windows = _windows(14)
    windows[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        extract_spectral_feature_bundle(
            windows,
            channel_names=PROJECT_EMOTIV_CHANNEL_ORDER,
            spec=SpectralFeatureSpec(nperseg=256, noverlap=128),
        )


def test_window_identity_requires_same_bounds_record_and_target_id() -> None:
    expected = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "record_id": ["r", "r"],
            "window_index": [0, 1],
            "start_sample": [0, 2560],
            "stop_sample": [2560, 5120],
        }
    )
    _validate_window_identity(expected, expected.copy(), label="same")
    changed = expected.copy()
    changed.loc[1, "stop_sample"] += 1
    with pytest.raises(RuntimeError, match="window identity"):
        _validate_window_identity(expected, changed, label="changed")


def test_record_aggregation_is_record_safe_and_includes_iqr() -> None:
    metadata = _window_metadata(4)
    metadata["cw__C1__log_power_delta"] = [1.0, 3.0, 10.0, 14.0]
    result = aggregate_record_features(
        metadata,
        feature_columns=["cw__C1__log_power_delta"],
    )
    assert len(result) == 2
    assert result["record_id"].is_unique
    first = result.set_index("record_id").loc["r-1"]
    assert first["record_mean__cw__C1__log_power_delta"] == 2.0
    assert first["record_iqr__cw__C1__log_power_delta"] == 1.0


def test_record_aggregation_rejects_mixed_record_identity() -> None:
    metadata = _window_metadata(4)
    metadata["cw__C1__log_power_delta"] = 1.0
    metadata.loc[1, "target"] = 2
    with pytest.raises(ValueError, match="identity changed"):
        aggregate_record_features(
            metadata,
            feature_columns=["cw__C1__log_power_delta"],
        )


def test_feature_sets_exclude_nuisance_until_explicitly_requested() -> None:
    records, _ = _synthetic_records()
    frame = records["emotiv_common"]
    spectral = feature_columns_for(
        frame, representation="channel_wise", feature_set="spectral_only"
    )
    with_nuisance = feature_columns_for(
        frame,
        representation="channel_wise",
        feature_set="spectral_plus_nuisance",
    )
    assert len(spectral) == 1
    assert len(with_nuisance) == 2
    assert not any(
        column.endswith(tuple(f"__{name}" for name in NUISANCE_FEATURE_TYPES))
        for column in spectral
    )
    global_spectral = feature_columns_for(
        frame, representation="global_summary", feature_set="spectral_only"
    )
    global_with_nuisance = feature_columns_for(
        frame,
        representation="global_summary",
        feature_set="spectral_plus_nuisance",
    )
    assert len(global_spectral) == 1
    assert len(global_with_nuisance) == 2


def test_nested_selection_uses_inner_train_scaler_and_unique_oof_records() -> None:
    records, inner = _synthetic_records()
    predictions, folds, selection = run_nested_benchmark(
        records, inner, _small_config()
    )
    assert selection["scaler_fit_partition"].eq("inner_train").all()
    assert not selection["outer_test_used_for_selection"].any()
    assert folds["test_records"].eq(9).all()
    keys = [
        "channel_policy",
        "representation",
        "model",
        "seed",
        "record_id",
    ]
    assert not predictions.duplicated(keys).any()
    assert predictions.groupby(keys[:-1]).size().eq(45).all()


def test_outer_test_values_do_not_change_inner_selection() -> None:
    records, inner = _synthetic_records()
    _, _, first = run_nested_benchmark(records, inner, _small_config())
    changed = {key: frame.copy() for key, frame in records.items()}
    for frame in changed.values():
        frame.loc[
            frame["outer_fold"].eq(1),
            "record_mean__cw__C1__log_power_delta",
        ] = 1e9
    _, _, second = run_nested_benchmark(changed, inner, _small_config())
    columns = [
        "channel_policy",
        "representation",
        "fold",
        "model",
        "seed",
        "feature_set",
        "grid_index",
        "selected",
        "inner_macro_f1",
    ]
    first_fold = first.loc[first["fold"].eq(1), columns].reset_index(drop=True)
    second_fold = second.loc[second["fold"].eq(1), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(first_fold, second_fold)


def test_small_hyperparameter_grid_is_enforced() -> None:
    config = _small_config()
    assert len(_model_grid(config, "multinomial_logistic_regression")) == 2
    config["models"]["logistic_regression"]["C"] = [0.01, 0.1, 1.0, 10.0, 100.0]
    with pytest.raises(ValueError, match="no more than four"):
        _model_grid(config, "multinomial_logistic_regression")


def test_ordinal_mae_qwk_and_severe_errors_are_correct() -> None:
    truth = np.array([0, 1, 2])
    perfect = np.eye(3)
    result = _metrics(truth, perfect)
    assert result["ordinal_mae"] == 0.0
    assert result["quadratic_weighted_kappa"] == pytest.approx(1.0)
    assert result["severe_0_to_2_errors"] == 0
    reversed_probabilities = np.eye(3)[[2, 1, 0]]
    severe = _metrics(truth, reversed_probabilities)
    assert severe["severe_0_to_2_errors"] == 2


def test_subject_bootstrap_samples_whole_subject_units() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": np.repeat(["a", "b", "c"], 3),
            "y_true": np.tile([0, 1, 2], 3),
            "y_pred_emotiv_common": np.tile([0, 1, 2], 3),
            "y_pred_cog_bci_common": np.tile([0, 1, 1], 3),
        }
    )
    sampled = subject_bootstrap_indices(
        frame["subject_id"], rng=np.random.default_rng(42)
    )
    assert len(sampled) == 3
    result = subject_bootstrap_differences(frame, repeats=20, seed=42)
    assert result["sampled_subjects"].eq(3).all()
    assert result["sampled_records"].eq(9).all()
    assert len(result) == 20


def test_config_paths_reject_absolute_and_parent_paths() -> None:
    with pytest.raises(ValueError, match="relative"):
        _relative_path(r"F:\EEG\data", label="data")
    with pytest.raises(ValueError, match="escape"):
        _relative_path("../outside", label="data")


def test_input_hash_audit_is_read_only_and_deterministic() -> None:
    config = json.loads(
        Path("experiments/cog_bci/nback_spectral_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    paths = resolve_paths(config, Path.cwd())
    before = input_hashes(paths)
    after = input_hashes(paths)
    assert before == after
    assert len(before) == 10
    serialized = json.dumps(config, sort_keys=True)
    assert "F:\\EEG" not in serialized
    assert "F:/EEG" not in serialized
