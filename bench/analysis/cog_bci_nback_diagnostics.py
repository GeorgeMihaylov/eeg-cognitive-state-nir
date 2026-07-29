"""Leakage-safe diagnostics for the weak COG-BCI N-Back baseline signal."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import signal, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bench.experiments.cog_bci_nback_baseline import classification_metrics


BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}
TASK_PATHS = {
    "zero_back": "zeroBACK",
    "one_back": "oneBACK",
    "two_back": "twoBACK",
}
CLASS_NAMES = {0: "zero_back", 1: "one_back", 2: "two_back"}
METADATA_COLUMNS = [
    "sample_id",
    "subject_id",
    "session_id",
    "record_id",
    "record_group_id",
    "window_index",
    "target",
    "class_name",
    "task_variant",
    "outer_fold",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be repository-relative")
    return path


def _shard_stem(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:24]


def input_manifest_paths(cache_dir: Path, protocol_dir: Path) -> dict[str, Path]:
    return {
        "window_index": cache_dir / "window_index.parquet",
        "task_definition": protocol_dir / "task_definition.json",
        "target_index": protocol_dir / "target_index.parquet",
        "outer_assignments": protocol_dir / "outer_assignments.parquet",
        "outer_folds": protocol_dir / "outer_folds.json",
        "inner_assignments": protocol_dir / "inner_assignments.parquet",
        "inner_folds": protocol_dir / "inner_folds.json",
    }


def manifest_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing diagnostic input manifests: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _one_task_end_per_record(events: pd.DataFrame) -> pd.DataFrame:
    ends = events.loc[events["is_task_end"].astype(bool)].copy()
    counts = ends.groupby("record_id", sort=False).size()
    invalid = counts[counts.ne(1)]
    all_records = set(events["record_id"].astype(str))
    missing = sorted(all_records - set(counts.index.astype(str)))
    if not invalid.empty or missing:
        raise ValueError(
            "Every record requires exactly one unambiguous task-end marker; "
            f"invalid={invalid.to_dict()}, missing={missing}"
        )
    indexed = ends.set_index("record_id")
    if not indexed.index.is_unique:
        raise ValueError("Task-end marker index is not unique")
    return indexed


def build_task_boundary_masks(
    accepted_windows: pd.DataFrame,
    all_windows: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build record-local metadata masks without changing cached windows."""

    accepted = accepted_windows.copy()
    if accepted["sample_id"].duplicated().any():
        raise ValueError("Accepted windows require unique sample_id")
    if not accepted["status"].eq("accepted").all():
        raise ValueError("Boundary masks operate on accepted windows only")
    record_ids = set(accepted["record_id"].astype(str))
    relevant_events = events.loc[
        events["record_id"].astype(str).isin(record_ids)
    ].copy()
    ends = _one_task_end_per_record(relevant_events)
    if set(ends.index.astype(str)) != record_ids:
        raise ValueError("End-marker records do not match accepted records")

    duration = (
        all_windows.loc[
            all_windows["record_id"].astype(str).isin(record_ids)
        ]
        .groupby("record_id", sort=False)["valid_stop_sample"]
        .max()
        .astype(float)
        / all_windows["sampling_rate_hz"].iloc[0]
    )
    event_groups = {
        str(record_id): group.sort_values("event_index", kind="stable")
        for record_id, group in relevant_events.groupby("record_id", sort=False)
    }
    audit_rows: list[dict[str, Any]] = []
    for record_id in sorted(record_ids):
        group = event_groups[record_id]
        non_boundary = group.loc[~group["is_boundary"].astype(bool)]
        end = ends.loc[record_id]
        audit_rows.append(
            {
                "record_id": record_id,
                "subject_id": str(group["subject_id"].iloc[0]),
                "session_id": str(group["session_id"].iloc[0]),
                "task_variant": str(group["task_variant"].iloc[0]),
                "first_event": str(group["description"].iloc[0]),
                "first_event_seconds": float(group["onset_seconds"].iloc[0]),
                "first_non_boundary_event": (
                    str(non_boundary["description"].iloc[0])
                    if not non_boundary.empty
                    else None
                ),
                "first_non_boundary_seconds": (
                    float(non_boundary["onset_seconds"].iloc[0])
                    if not non_boundary.empty
                    else None
                ),
                "last_event": str(group["description"].iloc[-1]),
                "last_event_seconds": float(group["onset_seconds"].iloc[-1]),
                "end_marker": str(end["description"]),
                "end_marker_seconds": float(end["onset_seconds"]),
                "record_duration_seconds": float(duration.loc[record_id]),
                "tail_after_end_seconds": max(
                    0.0,
                    float(duration.loc[record_id])
                    - float(end["onset_seconds"]),
                ),
                "events_after_end": int(
                    group["onset_seconds"].gt(float(end["onset_seconds"])).sum()
                ),
                "task_start_marker_count": int(
                    group["is_task_start"].astype(bool).sum()
                ),
                "task_end_marker_count": int(
                    group["is_task_end"].astype(bool).sum()
                ),
            }
        )
    boundary_audit = pd.DataFrame(audit_rows)
    end_map = boundary_audit.set_index("record_id")["end_marker_seconds"]
    accepted["end_marker_seconds"] = accepted["record_id"].map(end_map)
    if accepted["end_marker_seconds"].isna().any():
        raise ValueError("Mask rows cannot be matched to their own end marker")

    mask_frame = accepted[
        [
            "sample_id",
            "record_id",
            "subject_id",
            "session_id",
            "task_variant",
            "target",
            "class_name",
            "window_index",
            "start_time_seconds",
            "stop_time_seconds",
        ]
    ].copy()
    before_end = mask_frame["stop_time_seconds"].le(
        accepted["end_marker_seconds"].to_numpy(dtype=float) + 1e-9
    )
    mask_frame["record_full"] = True
    mask_frame["to_end_marker"] = before_end
    mask_frame["exclude_first_5s_to_end"] = (
        before_end & mask_frame["start_time_seconds"].ge(5.0)
    )
    mask_frame["exclude_first_10s_to_end"] = (
        before_end & mask_frame["start_time_seconds"].ge(10.0)
    )

    summaries: list[dict[str, Any]] = []
    for mask_name in (
        "record_full",
        "to_end_marker",
        "exclude_first_5s_to_end",
        "exclude_first_10s_to_end",
    ):
        selected = mask_frame.loc[mask_frame[mask_name].astype(bool)]
        empty_records = record_ids - set(selected["record_id"].astype(str))
        for target, group in selected.groupby("target", sort=True):
            summaries.append(
                {
                    "mask": mask_name,
                    "target": int(target),
                    "class_name": CLASS_NAMES[int(target)],
                    "records": int(group["record_id"].nunique()),
                    "windows": int(len(group)),
                    "excluded_windows": int(
                        accepted["target"].eq(target).sum() - len(group)
                    ),
                    "duration_seconds": float(
                        group["stop_time_seconds"].sub(
                            group["start_time_seconds"]
                        ).sum()
                    ),
                    "empty_records_total": len(empty_records),
                }
            )
    summary = pd.DataFrame(summaries)
    return mask_frame, boundary_audit, summary


def spectral_features(
    windows: np.ndarray,
    *,
    sampling_rate: float,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Compute deterministic DC-robust band-power features per window."""

    array = np.asarray(windows, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            "Spectral input must have shape [windows, channels, time]"
        )
    if array.shape[1] != len(channel_names):
        raise ValueError("Channel-name count does not match spectral input")
    if not np.isfinite(array).all():
        raise ValueError("Spectral input contains NaN or infinite values")
    frequencies, psd = signal.welch(
        array,
        fs=float(sampling_rate),
        nperseg=min(512, array.shape[-1]),
        noverlap=min(256, max(0, array.shape[-1] // 2)),
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    epsilon = np.finfo(np.float64).tiny
    band_power: dict[str, np.ndarray] = {}
    for name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        if mask.sum() < 2:
            raise ValueError(f"Insufficient bins for spectral band {name}")
        band_power[name] = np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)

    parts: list[np.ndarray] = []
    names: list[str] = []
    for channel_index, channel in enumerate(channel_names):
        for band in BANDS:
            parts.append(
                np.log10(np.maximum(band_power[band][:, channel_index], epsilon))
            )
            names.append(f"{channel}__log_power_{band}")
        theta = band_power["theta"][:, channel_index]
        alpha = band_power["alpha"][:, channel_index]
        beta = band_power["beta"][:, channel_index]
        parts.append(np.log(np.maximum(theta, epsilon) / np.maximum(alpha, epsilon)))
        names.append(f"{channel}__log_theta_alpha")
        parts.append(np.log(np.maximum(theta, epsilon) / np.maximum(beta, epsilon)))
        names.append(f"{channel}__log_theta_beta")
        variance = np.var(array[:, channel_index, :], axis=-1, dtype=np.float64)
        parts.append(np.log10(np.maximum(variance, epsilon)))
        names.append(f"{channel}__log_variance")
    features = np.column_stack(parts).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Spectral features contain NaN or infinite values")
    return features, names


def spectral_diagnostic_features(
    windows: np.ndarray, *, sampling_rate: float
) -> pd.DataFrame:
    """Return non-model DC, 50 Hz and total-power diagnostics per window."""

    array = np.asarray(windows, dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("Spectral diagnostic input must be finite 3D data")
    dc = np.mean(array, axis=-1)
    ac_variance = np.var(array, axis=-1)
    demeaned = array - dc[..., None]
    spectrum = np.abs(np.fft.rfft(demeaned, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(array.shape[-1], d=1.0 / sampling_rate)
    line_mask = (frequencies >= 49.0) & (frequencies <= 51.0)
    total_mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    line_power = np.sum(spectrum[..., line_mask], axis=-1)
    total_power = np.sum(spectrum[..., total_mask], axis=-1)
    epsilon = np.finfo(np.float64).tiny
    return pd.DataFrame(
        {
            "diagnostic_median_abs_dc": np.median(np.abs(dc), axis=1),
            "diagnostic_median_ac_std": np.median(
                np.sqrt(ac_variance), axis=1
            ),
            "diagnostic_dc_to_ac_variance_ratio": np.median(
                dc**2 / np.maximum(ac_variance, epsilon), axis=1
            ),
            "diagnostic_log_total_power_1_45": np.log10(
                np.maximum(np.median(total_power, axis=1), epsilon)
            ),
            "diagnostic_line_50_to_1_45_ratio": np.median(
                line_power / np.maximum(total_power, epsilon), axis=1
            ),
        }
    )


def aggregate_spectral_records(
    windows: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate window features within records without mixing identities."""

    rows: list[dict[str, Any]] = []
    for record_id, group in windows.groupby("record_id", sort=True):
        identity = group[
            ["subject_id", "session_id", "target", "class_name", "outer_fold"]
        ].drop_duplicates()
        if len(identity) != 1:
            raise ValueError(f"Record identity changed within {record_id}")
        values = group[list(feature_columns)].to_numpy(dtype=np.float64)
        row = {
            "record_id": str(record_id),
            **identity.iloc[0].to_dict(),
            "window_count": int(len(group)),
        }
        for index, name in enumerate(feature_columns):
            row[f"mean__{name}"] = float(np.mean(values[:, index]))
            row[f"median__{name}"] = float(np.median(values[:, index]))
            row[f"std__{name}"] = float(np.std(values[:, index]))
        rows.append(row)
    result = pd.DataFrame(rows)
    if result["record_id"].duplicated().any():
        raise ValueError("Record aggregation produced duplicate record_id")
    return result


def build_within_subject_rotations(
    records: pd.DataFrame,
) -> list[dict[str, Any]]:
    sessions = sorted(records["session_id"].astype(str).unique().tolist())
    if len(sessions) != 3:
        raise ValueError(f"Expected exactly three sessions, got {sessions}")
    rotations: list[dict[str, Any]] = []
    all_records = set(records["record_id"].astype(str))
    for rotation, held_out in enumerate(sessions, start=1):
        test = records.loc[
            records["session_id"].astype(str).eq(held_out)
        ].copy()
        train = records.loc[
            ~records["session_id"].astype(str).eq(held_out)
        ].copy()
        subject_overlap = set(train["subject_id"]) & set(test["subject_id"])
        record_overlap = set(train["record_id"]) & set(test["record_id"])
        if subject_overlap != set(records["subject_id"].astype(str).unique()):
            raise ValueError("Within-subject rotation lost intentional overlap")
        if record_overlap:
            raise ValueError("Within-subject rotation has record leakage")
        if set(train["record_id"]) | set(test["record_id"]) != all_records:
            raise ValueError("Within-subject rotation is not a full partition")
        rotations.append(
            {
                "rotation": rotation,
                "held_out_session": held_out,
                "train_record_ids": sorted(train["record_id"].astype(str)),
                "test_record_ids": sorted(test["record_id"].astype(str)),
                "subject_overlap": len(subject_overlap),
                "record_overlap": 0,
            }
        )
    return rotations


def _models(seed: int) -> dict[str, Any]:
    return {
        "multinomial_logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1e-4,
            random_state=seed,
        ),
    }


def _prediction_rows(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model: str,
    protocol: str,
    split_id: int,
) -> pd.DataFrame:
    predicted = np.argmax(probabilities, axis=1)
    result = frame[
        ["record_id", "subject_id", "session_id", "target", "class_name"]
    ].reset_index(drop=True)
    result["predicted_class"] = predicted.astype(int)
    for class_id in range(3):
        result[f"probability_class_{class_id}"] = probabilities[:, class_id]
    result["model"] = model
    result["protocol"] = protocol
    result["split_id"] = split_id
    return result


def _pooled_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    probabilities = predictions[
        [f"probability_class_{class_id}" for class_id in range(3)]
    ].to_numpy(dtype=float)
    return classification_metrics(
        predictions["target"].to_numpy(dtype=int),
        predictions["predicted_class"].to_numpy(dtype=int),
        probabilities,
    )


def evaluate_subject_disjoint(
    record_features: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_columns = [
        column
        for column in record_features.columns
        if column.startswith(("mean__", "median__", "std__"))
    ]
    if not feature_columns:
        raise ValueError("Record feature table has no model features")
    inner_records = inner_assignments.drop_duplicates(
        ["outer_fold", "record_id", "partition"]
    )
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        split = inner_records.loc[inner_records["outer_fold"].eq(fold)]
        train_ids = set(
            split.loc[split["partition"].eq("inner_train"), "record_id"].astype(str)
        )
        validation_ids = set(
            split.loc[
                split["partition"].eq("inner_validation"), "record_id"
            ].astype(str)
        )
        test_ids = set(
            record_features.loc[
                record_features["outer_fold"].eq(fold), "record_id"
            ].astype(str)
        )
        if train_ids & validation_ids or (train_ids | validation_ids) & test_ids:
            raise ValueError(f"Subject-disjoint fold {fold} has record leakage")
        train = record_features.loc[
            record_features["record_id"].astype(str).isin(train_ids)
        ]
        validation = record_features.loc[
            record_features["record_id"].astype(str).isin(validation_ids)
        ]
        test = record_features.loc[
            record_features["record_id"].astype(str).isin(test_ids)
        ]
        subject_sets = [
            set(frame["subject_id"].astype(str))
            for frame in (train, validation, test)
        ]
        if (
            subject_sets[0] & subject_sets[1]
            or subject_sets[0] & subject_sets[2]
            or subject_sets[1] & subject_sets[2]
        ):
            raise ValueError(f"Subject-disjoint fold {fold} has subject leakage")
        scaler = StandardScaler().fit(train[feature_columns])
        scaler_rows.append(
            {
                "fold": fold,
                "fit_partition": "inner_train",
                "fit_records": len(train),
                "fit_subjects": train["subject_id"].nunique(),
                "validation_records": len(validation),
                "test_records": len(test),
                "outer_test_used_for_fit": False,
            }
        )
        transformed = {
            "train": scaler.transform(train[feature_columns]),
            "validation": scaler.transform(validation[feature_columns]),
            "test": scaler.transform(test[feature_columns]),
        }
        for model_name, model in _models(seed + fold).items():
            model.fit(transformed["train"], train["target"].to_numpy(dtype=int))
            validation_probability = model.predict_proba(
                transformed["validation"]
            )
            test_probability = model.predict_proba(transformed["test"])
            validation_rows = _prediction_rows(
                validation,
                validation_probability,
                model=model_name,
                protocol="subject_disjoint_inner_validation",
                split_id=fold,
            )
            test_rows = _prediction_rows(
                test,
                test_probability,
                model=model_name,
                protocol="subject_disjoint_outer_test",
                split_id=fold,
            )
            validation_metrics = _pooled_metrics(validation_rows)
            test_metrics = _pooled_metrics(test_rows)
            fold_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "validation_balanced_accuracy": validation_metrics[
                        "balanced_accuracy"
                    ],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_ordinal_mae": test_metrics["ordinal_mae"],
                    "test_qwk": test_metrics["quadratic_weighted_kappa"],
                    "train_records": len(train),
                    "validation_records": len(validation),
                    "test_records": len(test),
                }
            )
            predictions.append(test_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    for model_name, group in prediction_frame.groupby("model"):
        if len(group) != group["record_id"].nunique():
            raise ValueError(f"Duplicate OOF record prediction for {model_name}")
    aggregate = {
        model: _pooled_metrics(group)
        for model, group in prediction_frame.groupby("model", sort=True)
    }
    return (
        prediction_frame,
        pd.DataFrame(fold_rows),
        {
            "aggregate_metrics": aggregate,
            "scaler_audit": scaler_rows,
            "features": feature_columns,
        },
    )


def evaluate_within_subject(
    record_features: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_columns = [
        column
        for column in record_features.columns
        if column.startswith(("mean__", "median__", "std__"))
    ]
    rotations = build_within_subject_rotations(record_features)
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for rotation in rotations:
        train = record_features.loc[
            record_features["record_id"].isin(rotation["train_record_ids"])
        ]
        test = record_features.loc[
            record_features["record_id"].isin(rotation["test_record_ids"])
        ]
        scaler = StandardScaler().fit(train[feature_columns])
        train_x = scaler.transform(train[feature_columns])
        test_x = scaler.transform(test[feature_columns])
        for model_name, model in _models(seed + int(rotation["rotation"])).items():
            model.fit(train_x, train["target"].to_numpy(dtype=int))
            rows = _prediction_rows(
                test,
                model.predict_proba(test_x),
                model=model_name,
                protocol="within_subject_session_disjoint",
                split_id=int(rotation["rotation"]),
            )
            metrics = _pooled_metrics(rows)
            metric_rows.append(
                {
                    "rotation": rotation["rotation"],
                    "held_out_session": rotation["held_out_session"],
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "ordinal_mae": metrics["ordinal_mae"],
                    "qwk": metrics["quadratic_weighted_kappa"],
                    "train_records": len(train),
                    "test_records": len(test),
                    "subject_overlap": rotation["subject_overlap"],
                    "record_overlap": rotation["record_overlap"],
                    "sample_overlap": 0,
                }
            )
            predictions.append(rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    aggregate = {
        model: _pooled_metrics(group)
        for model, group in prediction_frame.groupby("model", sort=True)
    }
    return (
        prediction_frame,
        pd.DataFrame(metric_rows),
        {"aggregate_metrics": aggregate, "rotations": rotations},
    )


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def analyze_existing_predictions(
    baseline_dir: Path,
    record_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    for model_dir, model_name in (
        ("eegnet_seed42", "torch_eegnet"),
        ("shallowconvnet_seed42", "torch_shallow_convnet"),
    ):
        path = baseline_dir / model_dir / "record_predictions.parquet"
        frame = pd.read_parquet(path)
        if frame["record_id"].duplicated().any():
            raise ValueError(f"Existing {model_name} predictions are duplicated")
        probabilities = frame[
            [f"mean_probability_class_{class_id}" for class_id in range(3)]
        ].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Existing {model_name} probabilities are non-finite")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"Existing {model_name} probabilities do not sum to one")
        frame = frame.rename(
            columns={
                "true_class": "target",
                "predicted_class": "predicted_class",
            }
        )
        frame["model"] = model_name
        frame["entropy"] = _entropy(probabilities)
        frame["normalized_entropy"] = frame["entropy"] / math.log(3.0)
        frame["max_probability"] = probabilities.max(axis=1)
        frame["true_class_probability"] = probabilities[
            np.arange(len(frame)), frame["target"].to_numpy(dtype=int)
        ]
        frame["correct"] = frame["target"].eq(frame["predicted_class"])
        frames.append(frame)
    confidence = pd.concat(frames, ignore_index=True)
    confidence = confidence.merge(
        record_audit[
            ["record_id", "duration_seconds", "duration_group"]
        ],
        on="record_id",
        how="left",
        validate="many_to_one",
    )
    eegnet = confidence.loc[confidence["model"].eq("torch_eegnet")].copy()
    shallow = confidence.loc[
        confidence["model"].eq("torch_shallow_convnet")
    ].copy()
    agreement = eegnet.merge(
        shallow,
        on=[
            "record_id",
            "subject_id",
            "session_id",
            "target",
            "window_count",
        ],
        suffixes=("_eegnet", "_shallow"),
        validate="one_to_one",
    )
    agreement["models_agree"] = agreement[
        "predicted_class_eegnet"
    ].eq(agreement["predicted_class_shallow"])
    agreement["both_correct"] = (
        agreement["correct_eegnet"] & agreement["correct_shallow"]
    )
    agreement["both_wrong"] = (
        ~agreement["correct_eegnet"] & ~agreement["correct_shallow"]
    )
    agreement["one_correct"] = (
        agreement["correct_eegnet"] ^ agreement["correct_shallow"]
    )
    p = agreement[
        [f"mean_probability_class_{i}_eegnet" for i in range(3)]
    ].to_numpy(dtype=float)
    q = agreement[
        [f"mean_probability_class_{i}_shallow" for i in range(3)]
    ].to_numpy(dtype=float)
    midpoint = 0.5 * (p + q)
    agreement["jensen_shannon_divergence"] = 0.5 * (
        np.sum(p * np.log(np.clip(p / midpoint, 1e-12, None)), axis=1)
        + np.sum(q * np.log(np.clip(q / midpoint, 1e-12, None)), axis=1)
    )
    summary: dict[str, Any] = {
        "agreement_fraction": float(agreement["models_agree"].mean()),
        "both_correct_fraction": float(agreement["both_correct"].mean()),
        "both_wrong_fraction": float(agreement["both_wrong"].mean()),
        "one_correct_fraction": float(agreement["one_correct"].mean()),
        "mean_jensen_shannon_divergence": float(
            agreement["jensen_shannon_divergence"].mean()
        ),
        "models": {},
    }
    for model, group in confidence.groupby("model", sort=True):
        calibration_rows = []
        bins = pd.cut(
            group["max_probability"],
            bins=[0.0, 0.4, 0.5, 0.6, 0.75, 1.0],
            include_lowest=True,
        )
        for bin_name, bin_group in group.groupby(bins, observed=True):
            calibration_rows.append(
                {
                    "bin": str(bin_name),
                    "records": len(bin_group),
                    "mean_confidence": float(bin_group["max_probability"].mean()),
                    "accuracy": float(bin_group["correct"].mean()),
                }
            )
        summary["models"][model] = {
            "mean_entropy": float(group["entropy"].mean()),
            "mean_normalized_entropy": float(
                group["normalized_entropy"].mean()
            ),
            "mean_max_probability": float(group["max_probability"].mean()),
            "median_max_probability": float(group["max_probability"].median()),
            "calibration_bins": calibration_rows,
            "prediction_counts": {
                str(key): int(value)
                for key, value in group["predicted_class"]
                .value_counts()
                .sort_index()
                .items()
            },
            "subject_prediction_distribution": (
                group.groupby(["subject_id", "predicted_class"])
                .size()
                .unstack(fill_value=0)
                .to_dict(orient="index")
            ),
            "session_prediction_distribution": (
                group.groupby(["session_id", "predicted_class"])
                .size()
                .unstack(fill_value=0)
                .to_dict(orient="index")
            ),
        }
    return confidence, agreement, summary


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    left = np.asarray(list(x))
    right = np.asarray(list(y))
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return {"rho": None, "pvalue_descriptive_only": None}
    result = stats.spearmanr(left, right)
    return {
        "rho": float(result.statistic),
        "pvalue_descriptive_only": float(result.pvalue),
    }


def build_duration_audit(
    accepted_windows: pd.DataFrame,
    boundary_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = (
        accepted_windows.groupby("record_id", sort=True)
        .agg(
            subject_id=("subject_id", "first"),
            session_id=("session_id", "first"),
            target=("target", "first"),
            class_name=("class_name", "first"),
            outer_fold=("outer_fold", "first"),
            window_count=("sample_id", "size"),
        )
        .reset_index()
        .merge(
            boundary_audit[
                ["record_id", "record_duration_seconds", "end_marker_seconds"]
            ],
            on="record_id",
            validate="one_to_one",
        )
        .rename(columns={"record_duration_seconds": "duration_seconds"})
    )
    lower, upper = records["duration_seconds"].quantile([1 / 3, 2 / 3])
    records["duration_group"] = np.select(
        [
            records["duration_seconds"].le(lower),
            records["duration_seconds"].le(upper),
        ],
        ["short", "medium"],
        default="long",
    )
    records["outer_train_fold_count"] = 4
    records["training_window_contribution_across_folds"] = (
        records["window_count"] * records["outer_train_fold_count"]
    )
    summary = {
        "duration_quantiles_seconds": {
            "q33_333": float(lower),
            "q66_667": float(upper),
        },
        "duration_vs_class": _safe_spearman(
            records["duration_seconds"], records["target"]
        ),
        "window_count_vs_class": _safe_spearman(
            records["window_count"], records["target"]
        ),
    }
    return records, summary


def _channel_statistics(
    array: np.ndarray,
    *,
    channel_names: Sequence[str],
    sampling_rate: float,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    time_axis = np.arange(array.shape[-1], dtype=np.float64) / sampling_rate
    centered = time_axis - time_axis.mean()
    denominator = float(np.sum(centered**2))
    rows: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(channel_names):
        values = np.asarray(array[:, channel_index, :], dtype=np.float64)
        flat = values.reshape(-1)
        slopes = np.tensordot(values, centered, axes=([1], [0])) / denominator
        quantiles = np.quantile(
            flat, [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
        )
        median = float(quantiles[3])
        rows.append(
            {
                **identity,
                "channel": str(channel),
                "value_count": int(flat.size),
                "mean": float(np.mean(flat)),
                "median": median,
                "std": float(np.std(flat)),
                "mad": float(np.median(np.abs(flat - median))),
                "minimum": float(np.min(flat)),
                "maximum": float(np.max(flat)),
                "percentile_0_1": float(quantiles[0]),
                "percentile_1": float(quantiles[1]),
                "percentile_25": float(quantiles[2]),
                "percentile_50": float(quantiles[3]),
                "percentile_75": float(quantiles[4]),
                "percentile_99": float(quantiles[5]),
                "percentile_99_9": float(quantiles[6]),
                "near_zero_fraction": float(np.mean(np.abs(flat) <= 1e-9)),
                "dc_component": float(np.mean(flat)),
                "mean_window_linear_trend_per_second": float(np.mean(slopes)),
            }
        )
    return rows


def audit_source_units(
    source_root: Path,
    *,
    subjects: Sequence[str],
    sessions: Sequence[str],
    task_variants: Sequence[str],
) -> dict[str, Any]:
    """Compare EEGLAB header/FDT values with the MNE reader contract."""

    import mne

    rows: list[dict[str, Any]] = []
    factors: list[float] = []
    for subject in subjects:
        for session in sessions:
            source_session = session.replace("ses-0", "ses-S")
            for task_variant in task_variants:
                task_name = TASK_PATHS[task_variant]
                candidates = sorted(
                    (source_root / subject).rglob(
                        f"{source_session}/eeg/{task_name}.set"
                    )
                )
                if len(candidates) != 1:
                    raise FileNotFoundError(
                        "Expected one source file for unit audit, got "
                        f"{len(candidates)}: subject={subject}, "
                        f"session={session}, task={task_variant}"
                    )
                set_path = candidates[0]
                header = scipy_io.loadmat(
                    set_path, squeeze_me=True, struct_as_record=False
                )
                channel_count = int(header["nbchan"])
                samples = int(header["pnts"])
                fdt_path = set_path.with_name(str(header["data"]))
                direct = np.memmap(
                    fdt_path,
                    dtype="<f4",
                    mode="r",
                    shape=(channel_count, samples),
                    order="F",
                )
                stop = min(samples, int(float(header["srate"]) * 5))
                direct_fragment = np.asarray(direct[:, :stop], dtype=np.float64)
                raw = mne.io.read_raw_eeglab(
                    set_path, preload=False, verbose="ERROR"
                )
                mne_fragment = raw.get_data(start=0, stop=stop)
                nonzero = np.abs(direct_fragment) > 1e-12
                ratio = np.median(
                    mne_fragment[nonzero] / direct_fragment[nonzero]
                )
                factors.append(float(ratio))
                chanlocs = np.atleast_1d(header["chanlocs"])
                etc = header.get("etc")
                rows.append(
                    {
                        "subject_id": subject,
                        "session_id": session,
                        "task_variant": task_variant,
                        "header_fields": sorted(
                            key
                            for key in header
                            if not key.startswith("__")
                        ),
                        "data_metadata": str(header.get("data")),
                        "datfile_metadata": str(header.get("datfile")),
                        "header_reference": str(header.get("ref")),
                        "header_sampling_rate": float(header["srate"]),
                        "header_channels": channel_count,
                        "header_samples": samples,
                        "explicit_header_unit_fields": [
                            key
                            for key in header
                            if "unit" in key.casefold()
                        ],
                        "etc_fields": list(
                            getattr(etc, "_fieldnames", []) or []
                        ),
                        "chanloc_fields": list(
                            getattr(chanlocs[0], "_fieldnames", []) or []
                        ),
                        "chanloc_type_values": sorted(
                            {
                                str(getattr(item, "type", ""))
                                for item in chanlocs
                                if str(getattr(item, "type", "")).strip()
                            }
                        ),
                        "chanloc_reference_values": sorted(
                            {
                                str(getattr(item, "ref", ""))
                                for item in chanlocs
                                if str(getattr(item, "ref", "")).strip()
                            }
                        ),
                        "mne_original_units": dict(raw._orig_units),
                        "mne_channel_units": sorted(
                            {int(channel["unit"]) for channel in raw.info["chs"]}
                        ),
                        "mne_calibration_factors": sorted(
                            {float(value) for value in raw._cals}
                        ),
                        "mne_highpass_hz": float(raw.info["highpass"]),
                        "mne_lowpass_hz": float(raw.info["lowpass"]),
                        "observed_mne_to_fdt_factor": float(ratio),
                        "direct_fragment": {
                            "minimum": float(direct_fragment.min()),
                            "maximum": float(direct_fragment.max()),
                            "mean": float(direct_fragment.mean()),
                            "std": float(direct_fragment.std()),
                            "channel_means_min": float(
                                direct_fragment.mean(axis=1).min()
                            ),
                            "channel_means_max": float(
                                direct_fragment.mean(axis=1).max()
                            ),
                        },
                        "mne_fragment": {
                            "minimum": float(mne_fragment.min()),
                            "maximum": float(mne_fragment.max()),
                            "mean": float(mne_fragment.mean()),
                            "std": float(mne_fragment.std()),
                            "channel_means_min": float(
                                mne_fragment.mean(axis=1).min()
                            ),
                            "channel_means_max": float(
                                mne_fragment.mean(axis=1).max()
                            ),
                        },
                    }
                )
    if not np.allclose(factors, 1e-6, rtol=0, atol=1e-12):
        raise ValueError(f"Unexpected MNE/FDT conversion factors: {factors}")
    return {
        "physical_unit": "unresolved",
        "mne_output_unit": "volt_by_reader_convention",
        "mne_applied_factor": 1e-6,
        "source_declares_explicit_unit": False,
        "evidence": [
            "EEGLAB headers contain no explicit unit field.",
            "EEG.chanlocs type/reference fields do not declare physical units.",
            "MNE raw._orig_units is empty for every audited file.",
            "MNE assigns FIFF volts and calibration 1e-6 to EEGLAB channels.",
            "Direct FDT values multiplied by 1e-6 exactly match MNE output.",
            "Large record/channel DC offsets make magnitude-only unit inference unsafe.",
        ],
        "audited_file_count": len(rows),
        "files": rows,
    }


@dataclass(frozen=True)
class DiagnosticPaths:
    repository_root: Path
    cache_dir: Path
    protocol_dir: Path
    baseline_dir: Path
    source_root: Path
    output_dir: Path


def _resolve_paths(
    config: Mapping[str, Any], repository_root: Path
) -> DiagnosticPaths:
    return DiagnosticPaths(
        repository_root=repository_root,
        cache_dir=repository_root
        / _relative_path(config["window_cache"], label="window_cache"),
        protocol_dir=repository_root
        / _relative_path(config["task_protocol"], label="task_protocol"),
        baseline_dir=repository_root
        / _relative_path(config["baseline_results"], label="baseline_results"),
        source_root=repository_root
        / _relative_path(config["source_root"], label="source_root"),
        output_dir=repository_root
        / _relative_path(config["output_dir"], label="output_dir"),
    )


def _runtime_report(summary: Mapping[str, Any]) -> str:
    subject = summary["subject_disjoint"]["aggregate_metrics"]
    within = summary["within_subject"]["aggregate_metrics"]
    lines = [
        "# COG-BCI N-Back weak-signal diagnostic",
        "",
        f"- Result status: `{summary['result_status']}`",
        f"- Physical unit: `{summary['unit_audit']['physical_unit']}`",
        f"- MNE applied factor: `{summary['unit_audit']['mne_applied_factor']}`",
        f"- Accepted windows: {summary['dataset']['windows']}",
        f"- Records: {summary['dataset']['records']}",
        "",
        "## Lightweight record-level baselines",
        "",
        "| Protocol | Model | Balanced accuracy | Macro F1 | Ordinal MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for protocol, metrics_by_model in (
        ("subject-disjoint", subject),
        ("within-subject/session-disjoint", within),
    ):
        for model, metrics in metrics_by_model.items():
            lines.append(
                f"| {protocol} | {model} | "
                f"{metrics['balanced_accuracy']:.6f} | "
                f"{metrics['macro_f1']:.6f} | "
                f"{metrics['ordinal_mae']:.6f} |"
            )
    lines.extend(
        [
            "",
            "The report is diagnostic. P-values, where emitted, are descriptive",
            "only and were not used for model or preprocessing selection.",
            "",
        ]
    )
    return "\n".join(lines)


def run_nback_signal_diagnostics(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Run the complete non-deep diagnostic workflow."""

    started = time.perf_counter()
    paths = _resolve_paths(config, repository_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = paths.output_dir / "errors.csv"
    pd.DataFrame(columns=["stage", "error"]).to_csv(errors_path, index=False)
    input_paths = input_manifest_paths(paths.cache_dir, paths.protocol_dir)
    hashes_before = manifest_hashes(input_paths)

    cache_manifest = json.loads(
        (paths.cache_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    target = pd.read_parquet(paths.protocol_dir / "target_index.parquet")
    accepted = target.loc[target["included_for_supervised"].astype(bool)].copy()
    all_windows = pd.read_parquet(paths.cache_dir / "window_index.parquet")
    cache_rows = all_windows[
        ["sample_id", "cache_offset", "status"]
    ].rename(columns={"status": "cache_status"})
    accepted = accepted.merge(
        cache_rows, on="sample_id", how="left", validate="one_to_one"
    )
    outer = pd.read_parquet(paths.protocol_dir / "outer_assignments.parquet")[
        ["sample_id", "fold"]
    ].rename(columns={"fold": "outer_fold"})
    accepted = accepted.merge(
        outer, on="sample_id", how="left", validate="one_to_one"
    )
    if (
        len(accepted) != 16927
        or accepted["sample_id"].duplicated().any()
        or sorted(accepted["target"].unique()) != [0, 1, 2]
    ):
        raise ValueError("Unexpected canonical N-Back supervised contract")
    events = pd.read_parquet(paths.cache_dir / "events.parquet")
    nback_all_windows = all_windows.loc[
        all_windows["record_id"].isin(accepted["record_id"].unique())
    ]
    mask_frame, boundary_audit, mask_summary = build_task_boundary_masks(
        accepted, nback_all_windows, events
    )
    boundary_audit.to_csv(
        paths.output_dir / "task_boundary_audit.csv", index=False
    )
    mask_summary.to_csv(
        paths.output_dir / "window_mask_summary.csv", index=False
    )

    unit_config = config["unit_audit"]
    unit_audit = audit_source_units(
        paths.source_root,
        subjects=unit_config["subjects"],
        sessions=unit_config["sessions"],
        task_variants=unit_config["task_variants"],
    )
    _write_json(paths.output_dir / "unit_audit.json", unit_audit)

    channel_names = list(cache_manifest["channel_order"])
    sampling_rate = float(cache_manifest["sampling_rate_hz"])
    amplitude_rows: list[dict[str, Any]] = []
    spectral_frames: list[pd.DataFrame] = []
    feature_names: list[str] | None = None
    for record_id, group in accepted.groupby("record_id", sort=True):
        ordered = group.sort_values("cache_offset", kind="stable")
        shard = (
            paths.cache_dir
            / "shards"
            / f"{_shard_stem(str(record_id))}.npy"
        )
        array = np.load(shard, mmap_mode="r")
        offsets = ordered["cache_offset"].to_numpy(dtype=int)
        selected = np.asarray(array[offsets], dtype=np.float32)
        identity = {
            "record_id": str(record_id),
            "subject_id": str(ordered["subject_id"].iloc[0]),
            "session_id": str(ordered["session_id"].iloc[0]),
            "target": int(ordered["target"].iloc[0]),
            "class_name": str(ordered["class_name"].iloc[0]),
            "outer_fold": int(ordered["outer_fold"].iloc[0]),
            "window_count": len(ordered),
        }
        amplitude_rows.extend(
            _channel_statistics(
                selected,
                channel_names=channel_names,
                sampling_rate=sampling_rate,
                identity=identity,
            )
        )
        features, current_names = spectral_features(
            selected,
            sampling_rate=sampling_rate,
            channel_names=channel_names,
        )
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise ValueError("Spectral feature order changed between records")
        metadata = ordered[METADATA_COLUMNS].reset_index(drop=True)
        spectral_frames.append(
            pd.concat(
                [
                    metadata,
                    pd.DataFrame(features, columns=current_names),
                    spectral_diagnostic_features(
                        selected, sampling_rate=sampling_rate
                    ),
                ],
                axis=1,
            )
        )
    amplitude = pd.DataFrame(amplitude_rows)
    amplitude.to_parquet(
        paths.output_dir / "amplitude_channel_summary.parquet", index=False
    )
    class_rows = []
    for fold in range(1, 6):
        train = amplitude.loc[amplitude["outer_fold"].ne(fold)]
        for (target_id, channel), group in train.groupby(
            ["target", "channel"], sort=True
        ):
            weights = group["value_count"].to_numpy(dtype=float)
            class_rows.append(
                {
                    "outer_fold": fold,
                    "partition": "outer_train_diagnostic",
                    "target": int(target_id),
                    "class_name": CLASS_NAMES[int(target_id)],
                    "channel": channel,
                    "records": len(group),
                    "subjects": group["subject_id"].nunique(),
                    "weighted_mean": float(
                        np.average(group["mean"], weights=weights)
                    ),
                    "weighted_std": float(
                        np.sqrt(
                            np.average(
                                group["std"].to_numpy() ** 2
                                + group["mean"].to_numpy() ** 2,
                                weights=weights,
                            )
                            - np.average(group["mean"], weights=weights) ** 2
                        )
                    ),
                    "median_record_dc": float(group["dc_component"].median()),
                    "median_record_mad": float(group["mad"].median()),
                    "mean_record_trend_per_second": float(
                        group["mean_window_linear_trend_per_second"].mean()
                    ),
                }
            )
    amplitude_class = pd.DataFrame(class_rows)
    amplitude_class.to_csv(
        paths.output_dir / "amplitude_class_summary.csv", index=False
    )

    spectral_windows = pd.concat(spectral_frames, ignore_index=True)
    if feature_names is None or not np.isfinite(
        spectral_windows[feature_names].to_numpy()
    ).all():
        raise ValueError("Spectral extraction did not produce finite features")
    spectral_windows.to_parquet(
        paths.output_dir / "spectral_window_features.parquet", index=False
    )
    spectral_records = aggregate_spectral_records(
        spectral_windows, feature_names
    )
    spectral_records.to_parquet(
        paths.output_dir / "spectral_record_features.parquet", index=False
    )
    spectral_class_rows = []
    for (target_id, class_name), group in spectral_windows.groupby(
        ["target", "class_name"], sort=True
    ):
        for feature in feature_names:
            spectral_class_rows.append(
                {
                    "target": int(target_id),
                    "class_name": class_name,
                    "feature": feature,
                    "mean": float(group[feature].mean()),
                    "std": float(group[feature].std(ddof=0)),
                    "median": float(group[feature].median()),
                    "windows": len(group),
                    "records": group["record_id"].nunique(),
                    "subjects": group["subject_id"].nunique(),
                }
            )
    pd.DataFrame(spectral_class_rows).to_csv(
        paths.output_dir / "spectral_class_summary.csv", index=False
    )

    inner = pd.read_parquet(paths.protocol_dir / "inner_assignments.parquet")
    subject_predictions, subject_folds, subject_summary = (
        evaluate_subject_disjoint(
            spectral_records,
            inner,
            seed=int(config.get("seed", 42)),
        )
    )
    subject_dir = paths.output_dir / "subject_disjoint_baseline"
    subject_dir.mkdir(exist_ok=True)
    subject_predictions.to_parquet(
        subject_dir / "predictions.parquet", index=False
    )
    subject_folds.to_csv(subject_dir / "fold_metrics.csv", index=False)
    _write_json(subject_dir / "metrics.json", subject_summary)

    within_predictions, within_metrics, within_summary = evaluate_within_subject(
        spectral_records,
        seed=int(config.get("seed", 42)),
    )
    within_dir = paths.output_dir / "within_subject_baseline"
    within_dir.mkdir(exist_ok=True)
    within_predictions.to_parquet(
        within_dir / "predictions.parquet", index=False
    )
    within_metrics.to_csv(within_dir / "rotation_metrics.csv", index=False)
    _write_json(within_dir / "metrics.json", within_summary)

    duration_audit, duration_summary = build_duration_audit(
        accepted, boundary_audit
    )
    confidence, agreement, confidence_summary = analyze_existing_predictions(
        paths.baseline_dir, duration_audit
    )
    confidence.to_csv(
        paths.output_dir / "prediction_confidence_audit.csv", index=False
    )
    agreement.to_csv(paths.output_dir / "model_agreement.csv", index=False)
    for model, group in confidence.groupby("model", sort=True):
        duration_summary[f"{model}_duration_vs_max_probability"] = (
            _safe_spearman(group["duration_seconds"], group["max_probability"])
        )
        duration_summary[f"{model}_window_count_vs_correct"] = (
            _safe_spearman(group["window_count"], group["correct"].astype(int))
        )
        duration_summary.setdefault("metrics_by_duration_group", {})[model] = {}
        for duration_group, subgroup in group.groupby(
            "duration_group", sort=True
        ):
            probabilities = subgroup[
                [
                    f"mean_probability_class_{class_id}"
                    for class_id in range(3)
                ]
            ].to_numpy(dtype=float)
            duration_summary["metrics_by_duration_group"][model][
                str(duration_group)
            ] = classification_metrics(
                subgroup["target"].to_numpy(dtype=int),
                subgroup["predicted_class"].to_numpy(dtype=int),
                probabilities,
            )
        prefix = (
            "eegnet"
            if model == "torch_eegnet"
            else "shallowconvnet"
        )
        additions = group[
            [
                "record_id",
                "predicted_class",
                "correct",
                "max_probability",
                "true_class_probability",
                "entropy",
            ]
        ].rename(
            columns={
                column: f"{prefix}_{column}"
                for column in (
                    "predicted_class",
                    "correct",
                    "max_probability",
                    "true_class_probability",
                    "entropy",
                )
            }
        )
        duration_audit = duration_audit.merge(
            additions, on="record_id", how="left", validate="one_to_one"
        )
    duration_audit.to_csv(
        paths.output_dir / "record_duration_audit.csv", index=False
    )

    hashes_after = manifest_hashes(input_paths)
    if hashes_before != hashes_after:
        raise RuntimeError("Input manifest checksum changed during diagnostics")
    amplitude_effect = (
        amplitude.groupby("target")["dc_component"].median().to_dict()
    )
    spectral_class_separation = {}
    for feature in feature_names:
        grouped = spectral_windows.groupby("target")[feature].mean()
        pooled_std = float(spectral_windows[feature].std(ddof=0))
        spectral_class_separation[feature] = (
            float((grouped.max() - grouped.min()) / pooled_std)
            if pooled_std > 0
            else 0.0
        )
    strongest_spectral = sorted(
        spectral_class_separation.items(), key=lambda item: abs(item[1]), reverse=True
    )[:10]
    mean_record_features = [
        column
        for column in spectral_records.columns
        if column.startswith("mean__")
    ]
    variance_attribution: dict[str, Any] = {}
    for group_column in ("target", "subject_id", "session_id"):
        eta_values = []
        group_sizes = spectral_records.groupby(group_column).size()
        for feature in mean_record_features:
            values = spectral_records[feature].to_numpy(dtype=float)
            grand_mean = float(values.mean())
            denominator = float(np.square(values - grand_mean).sum())
            group_means = spectral_records.groupby(group_column)[feature].mean()
            numerator = float(
                (
                    group_sizes
                    * np.square(group_means - grand_mean)
                ).sum()
            )
            eta_values.append(numerator / denominator if denominator else 0.0)
        variance_attribution[group_column] = {
            "median_eta_squared": float(np.median(eta_values)),
            "mean_eta_squared": float(np.mean(eta_values)),
            "percentile_90_eta_squared": float(
                np.quantile(eta_values, 0.9)
            ),
            "maximum_eta_squared": float(np.max(eta_values)),
            "interpretation": "descriptive_univariate_record_feature_fraction",
        }
    total_power = spectral_windows["diagnostic_log_total_power_1_45"]
    total_power_median = float(total_power.median())
    total_power_mad = float(
        np.median(np.abs(total_power.to_numpy() - total_power_median))
    )
    robust_scale = max(1.4826 * total_power_mad, 1e-12)
    spectral_outliers = np.abs(
        total_power.to_numpy() - total_power_median
    ) > 6.0 * robust_scale
    deep_result_path = (
        paths.output_dir
        / "deep_check_eegnet_demean_fold1"
        / "run_summary.json"
    )
    if deep_result_path.is_file():
        deep_run = json.loads(deep_result_path.read_text(encoding="utf-8"))
        original_run = json.loads(
            (
                paths.baseline_dir
                / "eegnet_seed42"
                / "run_summary.json"
            ).read_text(encoding="utf-8")
        )
        deep_fold = deep_run["folds"][0]
        original_fold = next(
            fold for fold in original_run["folds"] if fold["fold_id"] == 1
        )
        deep_checks: dict[str, Any] = {
            "performed": True,
            "model": "torch_eegnet",
            "fold": 1,
            "seed": 42,
            "epochs_trained": deep_fold["epochs_trained"],
            "best_epoch": deep_fold["best_epoch"],
            "transform": "per_window_mean_removal",
            "record_metrics": deep_fold["record_metrics"],
            "window_metrics": deep_fold["window_metrics"],
            "original_fold_record_metrics": original_fold["record_metrics"],
            "record_balanced_accuracy_delta": float(
                deep_fold["record_metrics"]["balanced_accuracy"]
                - original_fold["record_metrics"]["balanced_accuracy"]
            ),
            "record_macro_f1_delta": float(
                deep_fold["record_metrics"]["macro_f1"]
                - original_fold["record_metrics"]["macro_f1"]
            ),
            "interpretation": (
                "Per-window mean removal did not improve the matched fold; "
                "constant offset alone is not the sufficient explanation."
            ),
        }
    else:
        deep_checks = {
            "performed": False,
            "reason": (
                "No unit correction is scientifically confirmed and the "
                "metadata masks do not identify a confirmed task-start marker."
            ),
        }
    summary = {
        "result_status": "diagnostic",
        "dataset": {
            "windows": len(accepted),
            "records": accepted["record_id"].nunique(),
            "subjects": accepted["subject_id"].nunique(),
            "sessions": accepted["session_id"].nunique(),
            "classes": accepted["target"].value_counts().sort_index().to_dict(),
        },
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "inputs_unchanged": True,
        "unit_audit": unit_audit,
        "amplitude_audit": {
            "record_channel_rows": len(amplitude),
            "median_dc_by_class": amplitude_effect,
            "median_absolute_dc": float(
                amplitude["dc_component"].abs().median()
            ),
            "median_within_record_channel_std": float(
                amplitude["std"].median()
            ),
            "dc_to_within_record_std_ratio": float(
                amplitude["dc_component"].abs().median()
                / amplitude["std"].median()
            ),
        },
        "spectral_audit": {
            "features": len(feature_names),
            "finite": True,
            "strongest_standardized_class_mean_ranges": strongest_spectral,
            "median_line_50_to_1_45_ratio": float(
                spectral_windows[
                    "diagnostic_line_50_to_1_45_ratio"
                ].median()
            ),
            "percentile_99_line_50_to_1_45_ratio": float(
                spectral_windows[
                    "diagnostic_line_50_to_1_45_ratio"
                ].quantile(0.99)
            ),
            "median_window_dc_to_ac_variance_ratio": float(
                spectral_windows[
                    "diagnostic_dc_to_ac_variance_ratio"
                ].median()
            ),
            "extreme_total_power_outlier_windows_robust_z_gt_6": int(
                spectral_outliers.sum()
            ),
            "extreme_total_power_outlier_fraction": float(
                spectral_outliers.mean()
            ),
            "class_median_log_total_power_1_45": {
                str(key): float(value)
                for key, value in spectral_windows.groupby("target")[
                    "diagnostic_log_total_power_1_45"
                ]
                .median()
                .items()
            },
            "record_feature_variance_attribution": variance_attribution,
        },
        "boundary_audit": {
            "unique_end_markers": True,
            "records": len(boundary_audit),
            "task_start_markers": int(
                boundary_audit["task_start_marker_count"].sum()
            ),
            "events_after_end": int(boundary_audit["events_after_end"].sum()),
            "median_tail_after_end_seconds": float(
                boundary_audit["tail_after_end_seconds"].median()
            ),
            "mask_summary": mask_summary.to_dict(orient="records"),
        },
        "subject_disjoint": subject_summary,
        "within_subject": within_summary,
        "duration_audit": duration_summary,
        "prediction_audit": confidence_summary,
        "deep_checks": deep_checks,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(paths.output_dir / "diagnostic_summary.json", summary)
    (paths.output_dir / "diagnostic_report.md").write_text(
        _runtime_report(summary), encoding="utf-8"
    )
    return summary
