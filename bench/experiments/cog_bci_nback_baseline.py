"""Diagnostic EEGNet/ShallowConvNet baselines for native COG-BCI N-Back."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from bench.datasets.datasets_registry import get_dataset
from bench.validation.metrics import MetricsCalculator
from model_zoo.factory import build_model


CLASS_NAMES = ("zero_back", "one_back", "two_back")
EXPECTED_INPUT_SHAPE = (1, 14, 2560)
EXPECTED_TASK_ID = "cog_bci_nback_3class"
RESULT_STATUS = "diagnostic"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str | Path, *, label: str) -> Path:
    text = str(value).replace("\\", "/")
    if Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise ValueError(f"{label} must be relative, got {value!r}")
    path = Path(text)
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{label} must not escape repository root")
    return path


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, Any]:
    """Return the complete categorical plus ordinal baseline metric contract."""
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    probability = np.asarray(y_probability, dtype=np.float64)
    if truth.shape != prediction.shape or probability.shape != (len(truth), 3):
        raise ValueError("Classification metric inputs have incompatible shapes")
    if not np.isfinite(probability).all():
        raise ValueError("Classification probabilities must be finite")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Classification probabilities must sum to one")
    labels = np.arange(3, dtype=np.int64)
    result = MetricsCalculator.calculate_all_metrics(
        truth,
        prediction,
        probability,
        labels=labels,
    )
    result.update({
        "macro_precision": float(
            precision_score(
                truth,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                truth,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "class_wise_recall": {
            str(index): float(value)
            for index, value in enumerate(
                recall_score(
                    truth,
                    prediction,
                    labels=labels,
                    average=None,
                    zero_division=0,
                )
            )
        },
        "within_one_class_accuracy": float(
            np.mean(np.abs(truth - prediction) <= 1)
        ),
    })
    return _jsonable(result)


def aggregate_record_predictions(windows: pd.DataFrame) -> pd.DataFrame:
    """Average accepted-window probabilities without crossing records."""
    required = {
        "record_id",
        "subject_id",
        "session_id",
        "true_class",
        "probability_class_0",
        "probability_class_1",
        "probability_class_2",
        "fold_id",
    }
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"Window predictions are missing columns: {missing}")
    probability_columns = [
        "probability_class_0",
        "probability_class_1",
        "probability_class_2",
    ]
    rows: list[dict[str, Any]] = []
    for record_id, group in windows.groupby("record_id", sort=True):
        for column in (
            "subject_id",
            "session_id",
            "true_class",
            "fold_id",
        ):
            if group[column].nunique() != 1:
                raise ValueError(
                    f"Record {record_id} has multiple values for {column}"
                )
        mean_probability = group[probability_columns].mean().to_numpy()
        rows.append({
            "record_id": str(record_id),
            "subject_id": str(group["subject_id"].iloc[0]),
            "session_id": str(group["session_id"].iloc[0]),
            "true_class": int(group["true_class"].iloc[0]),
            "predicted_class": int(mean_probability.argmax()),
            "mean_probability_class_0": float(mean_probability[0]),
            "mean_probability_class_1": float(mean_probability[1]),
            "mean_probability_class_2": float(mean_probability[2]),
            "window_count": int(len(group)),
            "fold_id": int(group["fold_id"].iloc[0]),
            "model": str(group["model"].iloc[0]),
            "seed": int(group["seed"].iloc[0]),
        })
    result = pd.DataFrame(rows)
    if result["record_id"].duplicated().any():
        raise RuntimeError("Record aggregation produced duplicate record_id")
    return result


def calculate_record_subject_metrics(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate per-subject record metrics and distribution summaries."""
    rows: list[dict[str, Any]] = []
    probability_columns = [
        "mean_probability_class_0",
        "mean_probability_class_1",
        "mean_probability_class_2",
    ]
    for subject_id, group in records.groupby("subject_id", sort=True):
        metrics = classification_metrics(
            group["true_class"].to_numpy(dtype=np.int64),
            group["predicted_class"].to_numpy(dtype=np.int64),
            group[probability_columns].to_numpy(dtype=np.float64),
        )
        rows.append({
            "subject_id": str(subject_id),
            "fold_id": int(group["fold_id"].iloc[0]),
            "records": int(len(group)),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "ordinal_mae": metrics["ordinal_mae"],
        })
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "ordinal_mae"):
        values = frame[metric].to_numpy(dtype=float)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return frame, summary


def audit_input_scale(
    data: Any,
    train_indices: np.ndarray,
    *,
    sample_windows: int = 256,
) -> dict[str, Any]:
    """Audit a deterministic inner-train sample without changing the cache."""
    ordered = sorted(
        np.asarray(train_indices, dtype=np.int64),
        key=lambda index: str(data.sample_ids[index]),
    )[:sample_windows]
    values = np.stack([data.data[index] for index in ordered]).astype(
        np.float32, copy=False
    )
    flattened = values.astype(np.float64, copy=False).reshape(-1)
    percentiles = np.percentile(flattened, [0.1, 1, 50, 99, 99.9])
    return {
        "sample_selection": "first_sample_id_sorted_inner_train_fold_1",
        "sample_windows": int(len(values)),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "minimum": float(flattened.min()),
        "maximum": float(flattened.max()),
        "mean": float(flattened.mean()),
        "standard_deviation": float(flattened.std()),
        "median_absolute_value": float(np.median(np.abs(flattened))),
        "percentiles": {
            name: float(value)
            for name, value in zip(
                ("0.1", "1", "50", "99", "99.9"), percentiles
            )
        },
        "finite": bool(np.isfinite(flattened).all()),
        "input_unit_before": data.metadata["input_unit"],
        "source_physical_unit_status": data.metadata[
            "source_physical_unit_status"
        ],
        "reader_contract": (
            "MNE Raw.get_data() SI representation; original source unit "
            "metadata is not exposed"
        ),
        "cache_scaling_detected": False,
        "loader_scale_factor": 1.0,
        "model_input_scale_factor": 1.0,
        "normalization": "inner_train_channel_standardization",
        "input_unit_after": "standardized_dimensionless",
        "double_scaling": False,
    }


@dataclass
class BaselineRunOptions:
    smoke: bool = False
    fold: Optional[int] = None
    resume: bool = False


class COGBCINBackBaselineRunner:
    """Coordinate shared model/adapter components over immutable manifests."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        repository_root: Path,
        options: Optional[BaselineRunOptions] = None,
    ) -> None:
        self.config = deepcopy(dict(config))
        self.repository_root = repository_root
        self.options = options or BaselineRunOptions()
        self._validate_config()

    def _validate_config(self) -> None:
        required = {
            "dataset",
            "window_cache",
            "task_protocol",
            "model",
            "seed",
            "device",
            "epochs",
            "batch_size",
            "optimizer",
            "early_stopping",
            "input_scale",
            "metrics",
            "output_dir",
            "hashes",
        }
        missing = sorted(required - set(self.config))
        if missing:
            raise ValueError(f"Baseline config is missing fields: {missing}")
        if self.config["dataset"] != "cog_bci_nback_raw":
            raise ValueError("dataset must be 'cog_bci_nback_raw'")
        model = self.config["model"]
        if model.get("type") not in {
            "torch_eegnet",
            "torch_shallow_convnet",
        }:
            raise ValueError("Only EEGNet and ShallowConvNet are supported")
        if self.config["optimizer"].get("type") != "adamw":
            raise ValueError("The shared baseline optimizer must be AdamW")
        if (
            self.config["early_stopping"].get("monitor")
            != "validation_record_macro_f1"
        ):
            raise ValueError(
                "Baseline checkpoint selection must use record macro F1"
            )
        scale = self.config["input_scale"]
        if (
            float(scale.get("scale_factor", 0.0)) != 1.0
            or scale.get("normalization")
            != "inner_train_channel_standardization"
        ):
            raise ValueError(
                "COG-BCI baseline uses one explicit unit-preserving pass "
                "followed by train-only channel standardization"
            )
        for field in ("window_cache", "task_protocol", "output_dir"):
            _relative_path(self.config[field], label=field)
        if int(self.config["seed"]) != 42:
            raise ValueError("The first diagnostic baseline uses seed 42 only")
        if not 1 <= int(self.config["epochs"]) <= 50:
            raise ValueError("epochs must be between 1 and 50")

    def _dataset(self) -> Any:
        hashes = self.config["hashes"]
        dataset = get_dataset(
            self.config["dataset"],
            {
                "data_path": self.repository_root
                / _relative_path(
                    self.config["window_cache"], label="window_cache"
                ),
                "task_protocol_path": self.repository_root
                / _relative_path(
                    self.config["task_protocol"], label="task_protocol"
                ),
                "window_cache_config_hash": hashes[
                    "window_cache_config_hash"
                ],
                "task_protocol_hash": hashes["task_protocol_hash"],
                "window_transform": self.config.get(
                    "window_transform", "none"
                ),
            },
        )
        return dataset.load()

    def _output_dir(self) -> Path:
        base = self.repository_root / _relative_path(
            self.config["output_dir"], label="output_dir"
        )
        return base / "smoke" if self.options.smoke else base

    @staticmethod
    def _split_indices(
        data: Any, fold: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outer_fold = np.asarray(data.row_metadata["outer_fold"], dtype=int)
        test_indices = np.flatnonzero(outer_fold == fold)
        outer_train_indices = np.flatnonzero(outer_fold != fold)
        assignments = data.metadata["inner_assignments"]
        partition = assignments.loc[
            assignments["outer_fold"].eq(fold),
            ["sample_id", "partition"],
        ]
        partition_by_id = dict(
            zip(partition["sample_id"].astype(str), partition["partition"])
        )
        local_train: list[int] = []
        local_validation: list[int] = []
        for local_index, global_index in enumerate(outer_train_indices):
            sample_id = str(data.sample_ids[global_index])
            value = partition_by_id.get(sample_id)
            if value == "inner_train":
                local_train.append(local_index)
            elif value == "inner_validation":
                local_validation.append(local_index)
            else:
                raise ValueError(
                    f"Outer-train sample {sample_id} has invalid inner partition "
                    f"{value!r}"
                )
        return (
            outer_train_indices,
            test_indices,
            np.asarray(local_train, dtype=np.int64),
            np.asarray(local_validation, dtype=np.int64),
        )

    def _build_model(self, data: Any, fold: int, *, smoke: bool) -> Any:
        model_config = self.config["model"]
        params = deepcopy(dict(model_config.get("params", {})))
        params.update({
            "sampling_rate": float(data.sampling_rate),
            "channel_names": list(data.feature_names),
            "batch_size": int(self.config["batch_size"]),
            "max_epochs": 2 if smoke else int(self.config["epochs"]),
            "learning_rate": float(
                self.config["optimizer"]["learning_rate"]
            ),
            "weight_decay": float(
                self.config["optimizer"]["weight_decay"]
            ),
            "early_stopping_patience": int(
                self.config["early_stopping"]["patience"]
            ),
            "early_stopping_monitor": self.config["early_stopping"][
                "monitor"
            ],
            "device": self.config["device"],
            "random_state": int(self.config["seed"]),
            "standardize": True,
            "num_workers": int(self.config.get("num_workers", 0)),
        })
        if smoke:
            params["max_train_batches_per_epoch"] = int(
                self.config.get("smoke", {}).get(
                    "max_train_batches_per_epoch", 4
                )
            )
        adapter = build_model(
            model_name=model_config["type"],
            task_type="classification",
            input_shape=EXPECTED_INPUT_SHAPE,
            num_outputs=3,
            params=params,
        )
        adapter.model_metadata.update({
            "dataset": "cog_bci",
            "task_id": EXPECTED_TASK_ID,
            "target_name": "n_back_level",
            "class_names": list(CLASS_NAMES),
            "seed": int(self.config["seed"]),
            "fold_id": int(fold),
            "input_shape": list(EXPECTED_INPUT_SHAPE),
            "channel_order": list(data.feature_names),
            "input_scale": deepcopy(self.config["input_scale"]),
            **deepcopy(self.config["hashes"]),
        })
        return adapter

    @staticmethod
    def _leakage_row(
        data: Any,
        fold: int,
        outer_train: np.ndarray,
        outer_test: np.ndarray,
        inner_train_global: np.ndarray,
        inner_validation_global: np.ndarray,
    ) -> dict[str, Any]:
        def overlap(
            left: np.ndarray, right: np.ndarray, values: np.ndarray
        ) -> int:
            return len(
                set(values[left].astype(str)) & set(values[right].astype(str))
            )

        subjects = np.asarray(data.subject_ids)
        records = np.asarray(data.record_ids)
        samples = np.asarray(data.sample_ids)
        record_groups = np.asarray(data.row_metadata["record_group_id"])
        row = {
            "fold_id": fold,
            "outer_subject_overlap": overlap(
                outer_train, outer_test, subjects
            ),
            "outer_record_overlap": overlap(
                outer_train, outer_test, records
            ),
            "outer_record_group_overlap": overlap(
                outer_train, outer_test, record_groups
            ),
            "outer_sample_overlap": overlap(
                outer_train, outer_test, samples
            ),
            "inner_subject_overlap": overlap(
                inner_train_global, inner_validation_global, subjects
            ),
            "inner_record_overlap": overlap(
                inner_train_global, inner_validation_global, records
            ),
            "inner_record_group_overlap": overlap(
                inner_train_global,
                inner_validation_global,
                record_groups,
            ),
            "inner_sample_overlap": overlap(
                inner_train_global, inner_validation_global, samples
            ),
            "inner_outer_test_subject_overlap": len(
                (
                    set(subjects[inner_train_global].astype(str))
                    | set(subjects[inner_validation_global].astype(str))
                )
                & set(subjects[outer_test].astype(str))
            ),
            "inner_outer_test_record_overlap": len(
                (
                    set(records[inner_train_global].astype(str))
                    | set(records[inner_validation_global].astype(str))
                )
                & set(records[outer_test].astype(str))
            ),
            "inner_outer_test_sample_overlap": len(
                (
                    set(samples[inner_train_global].astype(str))
                    | set(samples[inner_validation_global].astype(str))
                )
                & set(samples[outer_test].astype(str))
            ),
        }
        row["leakage_safe"] = not any(
            value for key, value in row.items() if key.endswith("_overlap")
        )
        return row

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        data = self._dataset()
        if tuple(data.data.shape[1:]) != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"Unexpected COG-BCI input shape {data.data.shape[1:]}"
            )
        output_dir = self._output_dir()
        checkpoint_dir = output_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resolved_config = deepcopy(self.config)
        resolved_config["run_mode"] = (
            "smoke"
            if self.options.smoke
            else (
                "single_fold_diagnostic"
                if self.options.fold is not None
                else "full_diagnostic"
            )
        )
        _write_json(output_dir / "resolved_config.json", resolved_config)
        folds = (
            [int(self.options.fold)]
            if self.options.fold is not None
            else [1] if self.options.smoke else [1, 2, 3, 4, 5]
        )
        if any(fold not in {1, 2, 3, 4, 5} for fold in folds):
            raise ValueError("fold must be in 1..5")

        first_split = self._split_indices(data, folds[0])
        first_inner_train_global = first_split[0][first_split[2]]
        scale_audit = audit_input_scale(data, first_inner_train_global)
        _write_json(output_dir / "input_scale_audit.json", scale_audit)

        window_frames: list[pd.DataFrame] = []
        record_frames: list[pd.DataFrame] = []
        subject_frames: list[pd.DataFrame] = []
        fold_metric_rows: list[dict[str, Any]] = []
        history_frames: list[pd.DataFrame] = []
        leakage_rows: list[dict[str, Any]] = []
        fold_summaries: list[dict[str, Any]] = []
        model_name = str(self.config["model"]["type"])
        parameter_count: Optional[int] = None
        for fold in folds:
            (
                outer_train,
                outer_test,
                inner_train_local,
                inner_validation_local,
            ) = self._split_indices(data, fold)
            inner_train_global = outer_train[inner_train_local]
            inner_validation_global = outer_train[inner_validation_local]
            leakage = self._leakage_row(
                data,
                fold,
                outer_train,
                outer_test,
                inner_train_global,
                inner_validation_global,
            )
            if not leakage["leakage_safe"]:
                raise RuntimeError(f"Fold {fold} failed leakage audit")
            leakage_rows.append(leakage)
            adapter = self._build_model(data, fold, smoke=self.options.smoke)
            parameter_count = sum(
                parameter.numel() for parameter in adapter.model.parameters()
            )
            outer_train_view = data.data[outer_train]
            outer_train_labels = np.asarray(data.labels)[outer_train]
            adapter.set_validation_indices(
                inner_train_local,
                inner_validation_local,
                subject_ids=np.asarray(data.subject_ids)[outer_train],
                record_ids=np.asarray(data.record_ids)[outer_train],
                group_ids=np.asarray(data.subject_ids)[outer_train],
                outer_test_record_ids=np.asarray(data.record_ids)[outer_test],
                outer_test_group_ids=np.asarray(data.subject_ids)[outer_test],
                group_column="subject_id",
            )
            checkpoint = checkpoint_dir / f"fold_{fold:02d}.pt"
            fold_state_path = checkpoint_dir / f"fold_{fold:02d}.json"
            fold_started = time.perf_counter()
            resumed = (
                self.options.resume
                and checkpoint.is_file()
                and fold_state_path.is_file()
            )
            if resumed:
                adapter.load(checkpoint)
                stored = json.loads(
                    fold_state_path.read_text(encoding="utf-8")
                )
                training_seconds = float(stored["training_time_seconds"])
            else:
                initial = {
                    key: value.detach().cpu().clone()
                    for key, value in adapter.model.state_dict().items()
                }
                adapter.fit(outer_train_view, outer_train_labels)
                parameters_changed = any(
                    not torch.equal(initial[key], value.detach().cpu())
                    for key, value in adapter.model.state_dict().items()
                )
                if not parameters_changed:
                    raise RuntimeError("Model parameters did not change")
                adapter.save(checkpoint)
                training_seconds = time.perf_counter() - fold_started
                _write_json(
                    fold_state_path,
                    {
                        "fold_id": fold,
                        "training_time_seconds": training_seconds,
                        "best_epoch": adapter.best_epoch_,
                        "best_monitor_value": adapter.best_monitor_value_,
                        "best_validation_loss": adapter.best_validation_loss_,
                        "epochs_trained": adapter.n_epochs_trained_,
                    },
                )
            fresh = self._build_model(data, fold, smoke=self.options.smoke)
            fresh.load(checkpoint)
            probe = data.data[outer_test[: min(8, len(outer_test))]]
            np.testing.assert_allclose(
                adapter.predict_proba(probe),
                fresh.predict_proba(probe),
                atol=1e-7,
                rtol=1e-6,
            )
            adapter = fresh
            probabilities = adapter.predict_proba(data.data[outer_test])
            if probabilities.shape != (len(outer_test), 3):
                raise RuntimeError("Model returned invalid probability shape")
            if not np.isfinite(probabilities).all() or not np.allclose(
                probabilities.sum(axis=1), 1.0, atol=1e-5
            ):
                raise RuntimeError("Model returned invalid probabilities")
            prediction = probabilities.argmax(axis=1).astype(np.int64)
            frame = data.metadata["frame"].iloc[outer_test].reset_index(
                drop=True
            )
            windows = pd.DataFrame({
                "sample_id": frame["sample_id"].astype(str),
                "subject_id": frame["subject_id"].astype(str),
                "session_id": frame["session_id"].astype(str),
                "record_id": frame["record_id"].astype(str),
                "record_group_id": frame["record_group_id"].astype(str),
                "window_index": frame["window_index"].astype(int),
                "true_class": np.asarray(data.labels)[outer_test].astype(int),
                "predicted_class": prediction,
                "probability_class_0": probabilities[:, 0],
                "probability_class_1": probabilities[:, 1],
                "probability_class_2": probabilities[:, 2],
                "fold_id": fold,
                "model": model_name,
                "seed": int(self.config["seed"]),
            })
            if windows["sample_id"].duplicated().any():
                raise RuntimeError("Fold predictions duplicate sample_id")
            records = aggregate_record_predictions(windows)
            window_metrics = classification_metrics(
                windows["true_class"].to_numpy(),
                windows["predicted_class"].to_numpy(),
                windows[[
                    "probability_class_0",
                    "probability_class_1",
                    "probability_class_2",
                ]].to_numpy(),
            )
            record_metrics = classification_metrics(
                records["true_class"].to_numpy(),
                records["predicted_class"].to_numpy(),
                records[[
                    "mean_probability_class_0",
                    "mean_probability_class_1",
                    "mean_probability_class_2",
                ]].to_numpy(),
            )
            subjects, _ = calculate_record_subject_metrics(records)
            subjects["model"] = model_name
            subjects["seed"] = int(self.config["seed"])
            subject_frames.append(subjects)
            for level, metrics in (
                ("window", window_metrics),
                ("record", record_metrics),
            ):
                fold_metric_rows.append({
                    "fold_id": fold,
                    "level": level,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    },
                })
            history = pd.DataFrame(adapter.training_log_)
            history.insert(0, "fold_id", fold)
            history.insert(1, "model", model_name)
            history_frames.append(history)
            fold_summaries.append({
                "fold_id": fold,
                "train_windows": int(len(outer_train)),
                "test_windows": int(len(outer_test)),
                "test_records": int(len(records)),
                "train_subjects": int(
                    len(np.unique(np.asarray(data.subject_ids)[outer_train]))
                ),
                "test_subjects": int(
                    len(np.unique(np.asarray(data.subject_ids)[outer_test]))
                ),
                "epochs_trained": adapter.n_epochs_trained_,
                "best_epoch": adapter.best_epoch_,
                "best_validation_record_macro_f1": (
                    adapter.best_monitor_value_
                ),
                "best_validation_loss": adapter.best_validation_loss_,
                "training_time_seconds": training_seconds,
                "peak_gpu_memory_bytes": adapter.peak_gpu_memory_bytes_,
                "checkpoint": f"checkpoints/fold_{fold:02d}.pt",
                "checkpoint_sha256": _sha256_file(checkpoint),
                "resumed": resumed,
                "window_metrics": window_metrics,
                "record_metrics": record_metrics,
            })
            window_frames.append(windows)
            record_frames.append(records)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        windows_all = pd.concat(window_frames, ignore_index=True)
        records_all = pd.concat(record_frames, ignore_index=True)
        subjects_all = pd.concat(subject_frames, ignore_index=True)
        histories = pd.concat(history_frames, ignore_index=True)
        fold_metrics = pd.DataFrame(fold_metric_rows)
        leakage_frame = pd.DataFrame(leakage_rows)
        if windows_all["sample_id"].duplicated().any():
            raise RuntimeError("Unified predictions duplicate sample_id")
        if records_all["record_id"].duplicated().any():
            raise RuntimeError("Unified predictions duplicate record_id")
        if not self.options.smoke and self.options.fold is None:
            if len(windows_all) != 16927 or len(records_all) != 261:
                raise RuntimeError(
                    "Full baseline did not predict every accepted N-Back row"
                )
            if set(windows_all["sample_id"]) != set(data.sample_ids.astype(str)):
                raise RuntimeError("Full baseline sample coverage mismatch")

        window_metrics_all = classification_metrics(
            windows_all["true_class"].to_numpy(),
            windows_all["predicted_class"].to_numpy(),
            windows_all[[
                "probability_class_0",
                "probability_class_1",
                "probability_class_2",
            ]].to_numpy(),
        )
        record_metrics_all = classification_metrics(
            records_all["true_class"].to_numpy(),
            records_all["predicted_class"].to_numpy(),
            records_all[[
                "mean_probability_class_0",
                "mean_probability_class_1",
                "mean_probability_class_2",
            ]].to_numpy(),
        )
        _, subject_summary = calculate_record_subject_metrics(records_all)
        aggregate_by_level: dict[str, Any] = {}
        metric_columns = [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
            "ordinal_mae",
            "within_one_class_accuracy",
            "quadratic_weighted_kappa",
        ]
        for level, group in fold_metrics.groupby("level", sort=True):
            aggregate_by_level[level] = {
                metric: {
                    "mean": float(group[metric].mean()),
                    "std": float(group[metric].std(ddof=0)),
                }
                for metric in metric_columns
            }
        aggregate = {
            "result_status": RESULT_STATUS,
            "model": model_name,
            "seed": int(self.config["seed"]),
            "folds_completed": len(folds),
            "fold_metrics_mean_std": aggregate_by_level,
            "pooled_window_metrics": window_metrics_all,
            "pooled_record_metrics": record_metrics_all,
            "subject_record_metric_summary": subject_summary,
        }
        windows_all.to_parquet(
            output_dir / "window_predictions.parquet", index=False
        )
        records_all.to_parquet(
            output_dir / "record_predictions.parquet", index=False
        )
        subjects_all.to_csv(output_dir / "subject_metrics.csv", index=False)
        fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
        histories.to_csv(output_dir / "training_history.csv", index=False)
        pd.DataFrame(window_metrics_all["confusion_matrix"]).to_csv(
            output_dir / "confusion_matrix_window.csv", index=False
        )
        pd.DataFrame(record_metrics_all["confusion_matrix"]).to_csv(
            output_dir / "confusion_matrix_record.csv", index=False
        )
        _write_json(output_dir / "aggregate_metrics.json", aggregate)
        leakage_document = {
            "all_folds_leakage_safe": bool(
                leakage_frame["leakage_safe"].all()
            ),
            "folds": leakage_rows,
            **deepcopy(self.config["hashes"]),
        }
        _write_json(output_dir / "leakage_audit.json", leakage_document)
        pd.DataFrame(
            columns=["fold_id", "stage", "error_type", "message"]
        ).to_csv(output_dir / "errors.csv", index=False)
        completed_at = datetime.now(timezone.utc).isoformat()
        total_seconds = time.perf_counter() - started
        summary = {
            "result_status": RESULT_STATUS,
            "run_mode": resolved_config["run_mode"],
            "task_id": EXPECTED_TASK_ID,
            "model": model_name,
            "seed": int(self.config["seed"]),
            "device": str(
                fold_summaries and adapter.device_ or self.config["device"]
            ),
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else platform.processor() or "CPU"
            ),
            "parameter_count": parameter_count,
            "input_shape": list(EXPECTED_INPUT_SHAPE),
            "folds_requested": folds,
            "folds_completed": len(fold_summaries),
            "windows_predicted": int(len(windows_all)),
            "records_predicted": int(len(records_all)),
            "subjects_predicted": int(records_all["subject_id"].nunique()),
            "started_at": started_at,
            "completed_at": completed_at,
            "total_time_seconds": total_seconds,
            "folds": fold_summaries,
            "aggregate_metrics": aggregate,
            "checkpoint_verification": "factory_load_and_probability_match",
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            },
            "hashes": deepcopy(self.config["hashes"]),
        }
        _write_json(output_dir / "run_summary.json", summary)
        report = (
            f"# COG-BCI N-Back {model_name} {resolved_config['run_mode']}\n\n"
            f"- Status: `{RESULT_STATUS}`\n"
            f"- Folds: {len(folds)}\n"
            f"- Windows: {len(windows_all)}\n"
            f"- Records: {len(records_all)}\n"
            f"- Device: `{summary['device_name']}`\n"
            f"- Parameters: {parameter_count}\n"
            f"- Record balanced accuracy: "
            f"{record_metrics_all['balanced_accuracy']:.6f}\n"
            f"- Record macro F1: {record_metrics_all['macro_f1']:.6f}\n"
            f"- Record ordinal MAE: {record_metrics_all['ordinal_mae']:.6f}\n"
            f"- Leakage-safe: `{leakage_document['all_folds_leakage_safe']}`\n\n"
            "Checkpoint selection used inner-validation record-level macro F1. "
            "No outer-test value affected fitting or epoch selection.\n"
        )
        (output_dir / "run_report.md").write_text(
            report, encoding="utf-8"
        )
        return summary
