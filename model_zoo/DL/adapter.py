import random
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..base import BaseModelAdapter, PathLike
from .ordinal import ClassificationObjectiveHandler
from .regression import RegressionObjectiveHandler


def seed_torch(random_state: int) -> None:
    """Seed Python, NumPy and PyTorch without assuming CUDA is available."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _aggregate_loss_component_values(
    numerators: Mapping[str, float],
    denominators: Mapping[str, float],
) -> Dict[str, float]:
    """Return one dataset-level mean for each loss component."""
    if set(numerators) != set(denominators) or not numerators:
        raise RuntimeError("Objective loss components are inconsistent")
    values: Dict[str, float] = {}
    for name, numerator in numerators.items():
        denominator = float(denominators[name])
        if denominator <= 0:
            raise RuntimeError(
                f"Loss denominator for {name!r} must be positive"
            )
        values[name] = float(numerator) / denominator
    return values


class _LazyArrayDataset(Dataset):
    """Pair a NumPy-shaped lazy feature view with optional labels."""

    def __init__(self, features: Any, labels: Optional[np.ndarray]) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Any:
        features = torch.from_numpy(
            np.ascontiguousarray(self.features[index], dtype=np.float32)
        )
        if self.labels is None:
            return (features,)
        return features, torch.as_tensor(self.labels[index])


class TorchClassificationAdapter(BaseModelAdapter):
    """Sklearn-like trainer for a PyTorch classification module."""

    def __init__(
        self,
        model: nn.Module,
        input_shape: Sequence[int],
        num_classes: int,
        *,
        task_type: str = "classification",
        regression_loss: str = "mse",
        batch_size: int = 256,
        max_epochs: int = 30,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_size: float = 0.15,
        early_stopping_patience: int = 5,
        device: str = "auto",
        random_state: int = 42,
        standardize: bool = True,
        num_workers: int = 0,
        model_metadata: Optional[Mapping[str, Any]] = None,
        objective_handler: Optional[Any] = None,
    ) -> None:
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.num_classes = int(num_classes)
        self.num_outputs = self.num_classes
        normalized_task_type = str(task_type).strip().lower()
        self.task_type = {
            "classifier": "classification",
            "regressor": "regression",
        }.get(normalized_task_type, normalized_task_type)
        if self.task_type not in {"classification", "regression"}:
            raise ValueError(
                "task_type must be 'classification' or 'regression'"
            )
        self.regression_loss = str(regression_loss).strip().lower()
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.validation_size = float(validation_size)
        self.early_stopping_patience = int(early_stopping_patience)
        self.requested_device = str(device)
        self.random_state = int(random_state)
        self.standardize = bool(standardize)
        self.num_workers = int(num_workers)
        self.model_metadata = dict(model_metadata or {})
        self.objective_handler = (
            (
                ClassificationObjectiveHandler("categorical", self.num_classes)
                if self.task_type == "classification"
                else RegressionObjectiveHandler(
                    self.num_outputs,
                    self.regression_loss,
                )
            )
            if objective_handler is None
            else objective_handler
        )
        if self.objective_handler.num_classes != self.num_classes:
            raise ValueError(
                "Objective handler num_classes does not match adapter: "
                f"{self.objective_handler.num_classes} != {self.num_classes}"
            )
        self.head_type = self.objective_handler.head_type

        self._validate_config()
        self.device_ = self._resolve_device(self.requested_device)
        self.model = model.to(self.device_)
        self._initial_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

        self.training_log_: List[Dict[str, Any]] = []
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_scale_: Optional[np.ndarray] = None
        self.best_epoch_: Optional[int] = None
        self.best_validation_loss_: Optional[float] = None
        self.best_validation_components_: Dict[str, float] = {}
        self.early_stopping_monitor_: str = "validation_loss"
        self.n_epochs_trained_: int = 0
        self.stopping_reason_: Optional[str] = None
        self.peak_gpu_memory_bytes_: int = 0
        self.is_fitted_: bool = False
        self.validation_strategy_ = (
            "stratified_random"
            if self.task_type == "classification"
            else "random"
        )
        self.validation_group_column_: Optional[str] = None
        self.validation_random_state_ = self.random_state
        self.validation_split_: Optional[Dict[str, Any]] = None
        self.objective_training_diagnostics_: Dict[str, Any] = {}
        self.inner_train_indices_: Optional[np.ndarray] = None
        self.inner_validation_indices_: Optional[np.ndarray] = None
        self._validation_groups: Optional[np.ndarray] = None
        self._validation_subject_ids: Optional[np.ndarray] = None
        self._validation_record_ids: Optional[np.ndarray] = None
        self._outer_test_record_ids: Optional[np.ndarray] = None
        self._outer_test_group_ids: Optional[np.ndarray] = None
        self._validation_warning: Optional[str] = None
        self.validation_fallback_reason_: Optional[str] = None

    def _validate_config(self) -> None:
        if not self.input_shape or any(dim <= 0 for dim in self.input_shape):
            raise ValueError(f"input_shape must contain positive dimensions, got {self.input_shape}")
        if len(self.input_shape) not in {1, 2, 3}:
            raise ValueError(
                "TorchClassificationAdapter supports feature matrices, sequences, "
                "or image-like EEG windows: input_shape=(features,), "
                "(timesteps, features), or (1, channels, time), "
                f"got {self.input_shape}"
            )
        if len(self.input_shape) == 3 and self.input_shape[0] != 1:
            raise ValueError(
                "Four-dimensional EEG inputs must use input_shape=(1, channels, time), "
                f"got {self.input_shape}"
            )
        if self.task_type == "classification" and self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
        if self.task_type == "regression" and self.num_outputs <= 0:
            raise ValueError(f"num_outputs must be positive, got {self.num_outputs}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        normalized = device.strip().lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            resolved = torch.device(normalized)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid PyTorch device {device!r}") from exc
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return resolved

    def _validate_features(self, X: Any) -> Any:
        if getattr(X, "is_lazy_raw_eeg", False):
            shape = tuple(int(value) for value in getattr(X, "shape", ()))
            expected_shape = (len(X), *self.input_shape)
            if shape != expected_shape:
                raise ValueError(
                    f"Expected lazy X with shape {expected_shape}, got {shape}"
                )
            if len(X) == 0:
                raise ValueError("X cannot be empty")
            first = np.asarray(X[0], dtype=np.float32)
            if first.shape != self.input_shape or not np.isfinite(first).all():
                raise ValueError(
                    "Lazy raw EEG dataset returned an invalid first window: "
                    f"shape={first.shape}, finite={np.isfinite(first).all()}"
                )
            return X
        try:
            array = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be convertible to a numeric NumPy array") from exc
        expected_ndim = len(self.input_shape) + 1
        if array.ndim != expected_ndim:
            raise ValueError(
                f"Expected X with shape [batch, {', '.join(map(str, self.input_shape))}], "
                f"got {array.shape}"
            )
        if tuple(array.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected input_shape {self.input_shape}, got {tuple(array.shape[1:])}"
            )
        if len(array) == 0:
            raise ValueError("X cannot be empty")
        if not np.isfinite(array).all():
            raise ValueError("X contains NaN or infinite values")
        return np.ascontiguousarray(array)

    def _validate_labels(self, y: Any, n_samples: int) -> np.ndarray:
        array = np.asarray(y)
        if self.task_type == "regression":
            if self.num_outputs == 1 and array.ndim == 1:
                pass
            elif array.ndim != 2 or array.shape[1] != self.num_outputs:
                raise ValueError(
                    "Expected scalar regression y with shape [batch] or "
                    f"multi-output y with shape [batch, {self.num_outputs}], "
                    f"got {array.shape}"
                )
            if len(array) != n_samples:
                raise ValueError(
                    f"X and y have different lengths: {n_samples} and {len(array)}"
                )
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError("Regression y must contain numeric values")
            targets = np.asarray(array, dtype=np.float32)
            if not np.isfinite(targets).all():
                raise ValueError("y contains NaN or infinite values")
            return np.ascontiguousarray(targets)
        if array.ndim != 1:
            raise ValueError(f"Expected one-dimensional y, got shape {array.shape}")
        if len(array) != n_samples:
            raise ValueError(f"X and y have different lengths: {n_samples} and {len(array)}")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("y must contain numeric class labels")
        numeric = array.astype(np.float64, copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError("y contains NaN or infinite values")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("y must contain integer class labels")
        labels = numeric.astype(np.int64)
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            raise ValueError("Classification training requires at least two classes")
        if unique.min() < 0 or unique.max() >= self.num_classes:
            raise ValueError(
                f"Class labels must be in [0, {self.num_classes - 1}], got {unique.tolist()}"
            )
        if counts.min() < 2:
            raise ValueError("Each class needs at least two samples for validation splitting")
        return labels

    def _validate_calibration_labels(
        self,
        y: Any,
        n_samples: int,
    ) -> np.ndarray:
        """Validate calibration labels without requiring class coverage."""
        array = np.asarray(y)
        if array.ndim != 1:
            raise ValueError(f"Expected one-dimensional y, got shape {array.shape}")
        if len(array) != n_samples:
            raise ValueError(
                f"X and y have different lengths: {n_samples} and {len(array)}"
            )
        if len(array) == 0:
            raise ValueError("Calibration labels cannot be empty")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("y must contain numeric class labels")
        numeric = array.astype(np.float64, copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError("y contains NaN or infinite values")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("y must contain integer class labels")
        labels = numeric.astype(np.int64)
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError(
                f"Class labels must be in [0, {self.num_classes - 1}]"
            )
        return labels

    def _fit_standardizer(self, X_train: Any) -> None:
        if not self.standardize:
            self.feature_mean_ = None
            self.feature_scale_ = None
            return
        if getattr(X_train, "is_lazy_raw_eeg", False):
            mean, scale = X_train.compute_channel_statistics()
            self.feature_mean_ = np.asarray(mean, dtype=np.float32)
            self.feature_scale_ = np.asarray(scale, dtype=np.float32)
            return
        if X_train.ndim == 4:
            mean = X_train.mean(axis=(0, 1, 3), dtype=np.float64)
            scale = X_train.std(axis=(0, 1, 3), dtype=np.float64)
        else:
            statistics_rows = (
                X_train.reshape(-1, X_train.shape[-1])
                if X_train.ndim == 3
                else X_train
            )
            mean = statistics_rows.mean(axis=0, dtype=np.float64)
            scale = statistics_rows.std(axis=0, dtype=np.float64)
        scale = np.where(scale < 1e-8, 1.0, scale)
        self.feature_mean_ = mean.astype(np.float32)
        self.feature_scale_ = scale.astype(np.float32)

    def set_validation_groups(
        self,
        groups: Any,
        *,
        subject_ids: Any,
        record_ids: Any,
        outer_test_record_ids: Optional[Any] = None,
        outer_test_group_ids: Optional[Any] = None,
        strategy: str = "group_record",
        group_column: str = "record_id",
        validation_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> "TorchClassificationAdapter":
        """Configure record-disjoint validation for the next ``fit`` call."""
        normalized_strategy = strategy.strip().lower()
        if normalized_strategy not in {"group_record", "group_holdout"}:
            raise ValueError(
                "Group validation strategy must be 'group_record' or "
                "'group_holdout'"
            )
        group_values = np.asarray(groups)
        subject_values = np.asarray(subject_ids)
        record_values = np.asarray(record_ids)
        arrays = {
            "groups": group_values,
            "subject_ids": subject_values,
            "record_ids": record_values,
        }
        invalid_dimensions = {
            name: values.shape
            for name, values in arrays.items()
            if values.ndim != 1
        }
        if invalid_dimensions:
            raise ValueError(
                "Validation metadata must be one-dimensional, got "
                f"{invalid_dimensions}"
            )
        lengths = {name: len(values) for name, values in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Validation metadata must have equal lengths, got {lengths}"
            )
        if validation_size is not None:
            validation_size = float(validation_size)
            if not 0 < validation_size < 1:
                raise ValueError("validation_size must be between 0 and 1")
            self.validation_size = validation_size
        self.validation_strategy_ = normalized_strategy
        self.validation_group_column_ = str(group_column)
        self.validation_random_state_ = (
            self.random_state if random_state is None else int(random_state)
        )
        self._validation_groups = group_values.astype(str)
        self._validation_subject_ids = subject_values.astype(str)
        self._validation_record_ids = record_values.astype(str)
        self._outer_test_record_ids = (
            None
            if outer_test_record_ids is None
            else np.asarray(outer_test_record_ids).astype(str)
        )
        self._outer_test_group_ids = (
            None
            if outer_test_group_ids is None
            else np.asarray(outer_test_group_ids).astype(str)
        )
        if self._outer_test_group_ids is not None:
            overlap = np.intersect1d(
                np.unique(self._validation_groups),
                np.unique(self._outer_test_group_ids),
            )
            if len(overlap):
                raise ValueError(
                    "Outer training validation groups overlap outer test groups: "
                    f"{overlap.astype(str).tolist()}"
                )
        self.validation_fallback_reason_ = None
        self.validation_split_ = None
        return self

    def set_random_validation(
        self,
        *,
        subject_ids: Optional[Any] = None,
        record_ids: Optional[Any] = None,
        outer_test_group_ids: Optional[Any] = None,
        validation_size: Optional[float] = None,
        random_state: Optional[int] = None,
        fallback_reason: Optional[str] = None,
    ) -> "TorchClassificationAdapter":
        """Configure the explicit legacy random-window validation strategy."""
        if validation_size is not None:
            validation_size = float(validation_size)
            if not 0 < validation_size < 1:
                raise ValueError("validation_size must be between 0 and 1")
            self.validation_size = validation_size
        self.validation_strategy_ = "random_holdout"
        self.validation_group_column_ = None
        self.validation_random_state_ = (
            self.random_state if random_state is None else int(random_state)
        )
        self._validation_subject_ids = (
            None
            if subject_ids is None
            else np.asarray(subject_ids).astype(str)
        )
        self._validation_record_ids = (
            None
            if record_ids is None
            else np.asarray(record_ids).astype(str)
        )
        for name, values in {
            "subject_ids": self._validation_subject_ids,
            "record_ids": self._validation_record_ids,
        }.items():
            if values is not None and values.ndim != 1:
                raise ValueError(
                    f"{name} must be one-dimensional, got {values.shape}"
                )
        lengths = {
            len(values)
            for values in (
                self._validation_subject_ids,
                self._validation_record_ids,
            )
            if values is not None
        }
        if len(lengths) > 1:
            raise ValueError("Random validation metadata lengths must match")
        self._validation_groups = None
        self._outer_test_record_ids = None
        self._outer_test_group_ids = (
            None
            if outer_test_group_ids is None
            else np.asarray(outer_test_group_ids).astype(str)
        )
        self.validation_fallback_reason_ = fallback_reason
        self.validation_split_ = None
        return self

    @staticmethod
    def _class_distribution(labels: np.ndarray) -> Dict[str, int]:
        classes, counts = np.unique(labels, return_counts=True)
        return {
            str(class_label): int(count)
            for class_label, count in zip(classes, counts)
        }

    def _group_validation_indices(
        self,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._validation_groups is None:
            raise ValueError(
                "Grouped validation requires validation groups before fit"
            )
        if len(self._validation_groups) != len(labels):
            raise ValueError(
                "Validation groups must match training rows: "
                f"{len(self._validation_groups)} != {len(labels)}"
            )
        unique_groups = np.unique(self._validation_groups)
        if len(unique_groups) < 2:
            raise ValueError(
                "Group-aware validation requires at least two unique "
                "records/groups, "
                f"got {len(unique_groups)}"
            )

        n_candidates = min(128, max(32, len(unique_groups) * 4))
        splitter = GroupShuffleSplit(
            n_splits=n_candidates,
            test_size=self.validation_size,
            random_state=self.validation_random_state_,
        )
        if self.task_type == "classification":
            all_classes = np.unique(labels)
            overall_counts = np.asarray([
                np.sum(labels == class_label)
                for class_label in all_classes
            ], dtype=np.float64)
            overall_distribution = overall_counts / overall_counts.sum()
        else:
            overall_mean = np.mean(labels, axis=0, dtype=np.float64)
            overall_scale = np.std(labels, axis=0, dtype=np.float64)
            overall_scale = np.where(overall_scale < 1e-12, 1.0, overall_scale)
        best_indices: Optional[tuple[np.ndarray, np.ndarray]] = None
        best_score: Optional[tuple[int, float, float]] = None

        for train_idx, validation_idx in splitter.split(
            np.zeros(len(labels)), labels, self._validation_groups
        ):
            actual_fraction = len(validation_idx) / len(labels)
            size_error = abs(actual_fraction - self.validation_size)
            if self.task_type == "classification":
                train_classes = np.unique(labels[train_idx])
                validation_classes = np.unique(labels[validation_idx])
                missing_classes = (
                    len(np.setdiff1d(all_classes, train_classes))
                    + len(np.setdiff1d(all_classes, validation_classes))
                )
                train_distribution = np.asarray([
                    np.mean(labels[train_idx] == class_label)
                    for class_label in all_classes
                ])
                validation_distribution = np.asarray([
                    np.mean(labels[validation_idx] == class_label)
                    for class_label in all_classes
                ])
                balance_error = float(
                    np.abs(train_distribution - overall_distribution).sum()
                    + np.abs(validation_distribution - overall_distribution).sum()
                )
            else:
                missing_classes = 0
                balance_error = float(
                    np.abs(
                        (np.mean(labels[train_idx], axis=0) - overall_mean)
                        / overall_scale
                    ).sum()
                    + np.abs(
                        (
                            np.mean(labels[validation_idx], axis=0)
                            - overall_mean
                        )
                        / overall_scale
                    ).sum()
                )
            score = (missing_classes, size_error, balance_error)
            if best_score is None or score < best_score:
                best_score = score
                best_indices = (train_idx, validation_idx)

        if best_indices is None or best_score is None:
            raise ValueError("Could not create a group-aware validation split")
        if (
            self.task_type == "classification"
            and best_score[0] > 0
            and self.validation_strategy_ == "group_holdout"
        ):
            distribution_by_group = {
                str(group): self._class_distribution(
                    labels[self._validation_groups == group]
                )
                for group in unique_groups
            }
            raise ValueError(
                "Could not create a class-complete group_holdout split. "
                f"Missing class partitions={best_score[0]}; "
                f"class_distribution_by_group={distribution_by_group}"
            )
        if self.task_type == "classification" and best_score[0] > 0:
            self._validation_warning = (
                "Perfect class coverage is impossible for the selected record-level "
                f"validation split; missing class partitions={best_score[0]}"
            )
            warnings.warn(self._validation_warning, RuntimeWarning, stacklevel=2)
        else:
            self._validation_warning = None
        train_idx, validation_idx = best_indices
        group_overlap = np.intersect1d(
            self._validation_groups[train_idx],
            self._validation_groups[validation_idx],
        )
        if len(group_overlap):
            raise RuntimeError(
                "Group-aware validation produced overlapping records: "
                f"{group_overlap.astype(str).tolist()}"
            )
        return train_idx, validation_idx

    def _validation_indices(
        self,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.validation_strategy_ in {"group_record", "group_holdout"}:
            return self._group_validation_indices(labels)
        indices = np.arange(len(labels), dtype=np.int64)
        try:
            train_idx, validation_idx = train_test_split(
                indices,
                test_size=self.validation_size,
                random_state=self.validation_random_state_,
                shuffle=True,
                stratify=(
                    labels if self.task_type == "classification" else None
                ),
            )
        except ValueError as exc:
            raise ValueError(
                "Could not create a validation split inside training data"
            ) from exc
        return train_idx, validation_idx

    def _validation_summary(
        self,
        labels: np.ndarray,
        train_idx: np.ndarray,
        validation_idx: np.ndarray,
    ) -> Dict[str, Any]:
        train_subjects: list[str] = []
        validation_subjects: list[str] = []
        train_records: list[str] = []
        validation_records: list[str] = []
        train_groups: list[str] = []
        validation_groups: list[str] = []
        if self._validation_subject_ids is not None:
            train_subjects = sorted(
                np.unique(self._validation_subject_ids[train_idx]).tolist()
            )
            validation_subjects = sorted(
                np.unique(self._validation_subject_ids[validation_idx]).tolist()
            )
        if self._validation_record_ids is not None:
            train_records = sorted(
                np.unique(self._validation_record_ids[train_idx]).tolist()
            )
            validation_records = sorted(
                np.unique(self._validation_record_ids[validation_idx]).tolist()
            )
        if self._validation_groups is not None:
            train_groups = sorted(
                np.unique(self._validation_groups[train_idx]).tolist()
            )
            validation_groups = sorted(
                np.unique(self._validation_groups[validation_idx]).tolist()
            )
        subject_overlap = sorted(set(train_subjects) & set(validation_subjects))
        record_overlap = sorted(set(train_records) & set(validation_records))
        group_overlap = sorted(set(train_groups) & set(validation_groups))
        outer_records = set(
            []
            if self._outer_test_record_ids is None
            else np.unique(self._outer_test_record_ids).tolist()
        )
        outer_test_record_overlap = sorted(
            (set(train_records) | set(validation_records)) & outer_records
        )
        outer_groups = set(
            []
            if self._outer_test_group_ids is None
            else np.unique(self._outer_test_group_ids).tolist()
        )
        train_outer_group_overlap = sorted(set(train_groups) & outer_groups)
        validation_outer_group_overlap = sorted(
            set(validation_groups) & outer_groups
        )
        summary = {
            "strategy": self.validation_strategy_,
            "random_state": self.validation_random_state_,
            "group_column": self.validation_group_column_,
            "validation_size_requested": self.validation_size,
            "validation_fraction_actual": len(validation_idx) / len(labels),
            "inner_train_size": int(len(train_idx)),
            "inner_validation_size": int(len(validation_idx)),
            "inner_train_sequences": int(len(train_idx)),
            "inner_val_sequences": int(len(validation_idx)),
            "inner_train_subject_ids": train_subjects,
            "inner_validation_subject_ids": validation_subjects,
            "inner_train_record_ids": train_records,
            "inner_validation_record_ids": validation_records,
            "inner_train_group_ids": train_groups,
            "inner_validation_group_ids": validation_groups,
            "inner_train_records": len(train_records),
            "inner_val_records": len(validation_records),
            "subject_overlap": subject_overlap,
            "record_overlap": record_overlap,
            "group_overlap": group_overlap,
            "inner_record_overlap": len(record_overlap),
            "outer_test_record_overlap": outer_test_record_overlap,
            "validation_strategy": self.validation_strategy_,
            "validation_group_column": self.validation_group_column_,
            "validation_fraction": self.validation_size,
            "validation_random_state": self.validation_random_state_,
            "inner_train_samples": int(len(train_idx)),
            "inner_validation_samples": int(len(validation_idx)),
            "inner_train_groups": int(len(train_groups)),
            "inner_validation_groups": int(len(validation_groups)),
            "inner_group_overlap": int(len(group_overlap)),
            "outer_test_group_overlap": int(
                len(
                    set(train_outer_group_overlap)
                    | set(validation_outer_group_overlap)
                )
            ),
            "train_outer_test_overlap_count": int(
                len(train_outer_group_overlap)
            ),
            "validation_outer_test_overlap_count": int(
                len(validation_outer_group_overlap)
            ),
            "train_outer_test_group_overlap": train_outer_group_overlap,
            "validation_outer_test_group_overlap": (
                validation_outer_group_overlap
            ),
            "fallback_reason": self.validation_fallback_reason_,
            "class_balance_warning": self._validation_warning,
        }
        if self.task_type == "classification":
            summary.update({
                "class_distribution_train": self._class_distribution(
                    labels[train_idx]
                ),
                "class_distribution_validation": self._class_distribution(
                    labels[validation_idx]
                ),
                "inner_train_class_distribution": self._class_distribution(
                    labels[train_idx]
                ),
                "inner_val_class_distribution": self._class_distribution(
                    labels[validation_idx]
                ),
            })
        else:
            summary.update({
                "target_mean_train": np.mean(
                    labels[train_idx], axis=0
                ).astype(float).tolist(),
                "target_mean_validation": np.mean(
                    labels[validation_idx], axis=0
                ).astype(float).tolist(),
            })
        return summary

    def _transform_features(self, X: Any) -> Any:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            return X
        if getattr(X, "is_lazy_raw_eeg", False):
            return X.with_channel_normalization(
                self.feature_mean_, self.feature_scale_
            )
        if X.ndim == 4:
            mean = self.feature_mean_[None, None, :, None]
            scale = self.feature_scale_[None, None, :, None]
            transformed = (X - mean) / scale
        else:
            transformed = (X - self.feature_mean_) / self.feature_scale_
        if not np.isfinite(transformed).all():
            raise ValueError("Feature standardization produced NaN or infinite values")
        return np.ascontiguousarray(transformed, dtype=np.float32)

    def set_feature_normalization(
        self,
        mean: Any,
        scale: Any,
    ) -> "TorchClassificationAdapter":
        """Replace transform statistics on an independent fitted adapter."""
        mean_array = np.asarray(mean, dtype=np.float32)
        scale_array = np.asarray(scale, dtype=np.float32)
        expected_shape = (
            (self.input_shape[1],)
            if len(self.input_shape) == 3
            else (self.input_shape[-1],)
        )
        if mean_array.shape != expected_shape or scale_array.shape != expected_shape:
            raise ValueError(
                "Normalization statistics must have shape "
                f"{expected_shape}, got mean={mean_array.shape}, "
                f"scale={scale_array.shape}"
            )
        if not np.isfinite(mean_array).all() or not np.isfinite(scale_array).all():
            raise ValueError("Normalization statistics must be finite")
        if np.any(scale_array <= 0):
            raise ValueError("Normalization scale must be positive")
        self.feature_mean_ = mean_array.copy()
        self.feature_scale_ = scale_array.copy()
        return self

    def _make_loader(
        self,
        X: Any,
        y: Optional[np.ndarray] = None,
        *,
        shuffle: bool = False,
    ) -> DataLoader:
        dataset: Dataset
        if getattr(X, "is_lazy_raw_eeg", False):
            dataset = _LazyArrayDataset(X, y)
        else:
            features = torch.from_numpy(X)
            if y is None:
                dataset = TensorDataset(features)
            else:
                dataset = TensorDataset(features, torch.from_numpy(y))
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.device_.type == "cuda",
            generator=generator if shuffle else None,
        )

    def fit(self, X: Any, y: Any) -> "TorchClassificationAdapter":
        features = self._validate_features(X)
        labels = self._validate_labels(y, len(features))
        seed_torch(self.random_state)
        self.model.load_state_dict(self._initial_state)
        self.model.to(self.device_)
        if self.device_.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device_)

        train_idx, validation_idx = self._validation_indices(labels)
        self.inner_train_indices_ = np.asarray(train_idx, dtype=np.int64)
        self.inner_validation_indices_ = np.asarray(
            validation_idx, dtype=np.int64
        )
        self.validation_split_ = self._validation_summary(
            labels, self.inner_train_indices_, self.inner_validation_indices_
        )
        X_train = features[self.inner_train_indices_]
        X_validation = features[self.inner_validation_indices_]
        y_train = labels[self.inner_train_indices_]
        y_validation = labels[self.inner_validation_indices_]
        self.objective_training_diagnostics_ = (
            self.objective_handler.training_diagnostics(
                torch.from_numpy(y_train)
            )
        )

        self._fit_standardizer(X_train)
        X_train = self._transform_features(X_train)
        X_validation = self._transform_features(X_validation)
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        validation_loader = self._make_loader(X_validation, y_validation)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_monitor = float("inf")
        best_validation_components: Dict[str, float] = {}
        epochs_without_improvement = 0
        self.training_log_ = []
        self.best_epoch_ = None
        self.best_validation_loss_ = None
        self.best_validation_components_ = {}
        monitor_component = str(
            getattr(self.objective_handler, "early_stopping_component", "objective")
        )
        self.early_stopping_monitor_ = (
            "validation_categorical_loss"
            if monitor_component == "categorical"
            else "validation_loss"
        )
        self.stopping_reason_ = None

        for epoch in range(1, self.max_epochs + 1):
            epoch_started = time.perf_counter()
            self.model.train()
            train_numerators: Dict[str, float] = {}
            train_denominators: Dict[str, float] = {}
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device_, non_blocking=True)
                batch_labels = batch_labels.to(self.device_, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                raw_outputs = self.model(batch_features)
                component_parts = self.objective_handler.loss_component_parts(
                    raw_outputs, batch_labels
                )
                component_means = {
                    name: parts.mean for name, parts in component_parts.items()
                }
                loss = self.objective_handler.combine_component_means(
                    component_means
                )
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became NaN or infinite")
                loss.backward()
                optimizer.step()
                for name, parts in component_parts.items():
                    train_numerators[name] = train_numerators.get(name, 0.0) + float(
                        parts.numerator.detach().item()
                    )
                    train_denominators[name] = train_denominators.get(name, 0.0) + float(
                        parts.denominator.detach().item()
                    )

            self.model.eval()
            validation_numerators: Dict[str, float] = {}
            validation_denominators: Dict[str, float] = {}
            validation_correct = 0
            validation_count = 0
            with torch.no_grad():
                for batch_features, batch_labels in validation_loader:
                    batch_features = batch_features.to(self.device_, non_blocking=True)
                    batch_labels = batch_labels.to(self.device_, non_blocking=True)
                    raw_outputs = self.model(batch_features)
                    component_parts = self.objective_handler.loss_component_parts(
                        raw_outputs, batch_labels
                    )
                    for name, parts in component_parts.items():
                        validation_numerators[name] = (
                            validation_numerators.get(name, 0.0)
                            + float(parts.numerator.detach().item())
                        )
                        validation_denominators[name] = (
                            validation_denominators.get(name, 0.0)
                            + float(parts.denominator.detach().item())
                        )
                    if self.task_type == "classification":
                        decoded = self.objective_handler.decode(raw_outputs)
                        validation_correct += int(
                            (decoded.y_pred == batch_labels).sum().item()
                        )
                        validation_count += len(batch_labels)

            train_components = _aggregate_loss_component_values(
                train_numerators, train_denominators
            )
            validation_components = _aggregate_loss_component_values(
                validation_numerators, validation_denominators
            )
            train_loss = float(
                self.objective_handler.combine_component_means(train_components)
            )
            validation_loss = float(
                self.objective_handler.combine_component_means(validation_components)
            )
            if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
                raise ValueError("Aggregated training loss became non-finite")
            if monitor_component not in validation_components:
                raise RuntimeError(
                    f"Early-stopping component {monitor_component!r} is unavailable"
                )
            monitor_value = float(validation_components[monitor_component])
            improved = monitor_value < best_monitor
            if improved:
                best_monitor = monitor_value
                best_validation_components = dict(validation_components)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_diagnostics = dict(self.objective_training_diagnostics_)
            head_diagnostics = getattr(
                self.model, "output_head_diagnostics", None
            )
            if callable(head_diagnostics):
                epoch_diagnostics.update(head_diagnostics())
            log_row: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": (
                    validation_correct / validation_count
                    if validation_count
                    else None
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "is_best": improved,
                "epoch_time_seconds": time.perf_counter() - epoch_started,
                "early_stopping_metric": self.early_stopping_monitor_,
                **epoch_diagnostics,
            }
            if self.head_type == "categorical_corn":
                log_row.update({
                    "train_total_loss": train_loss,
                    "train_categorical_loss": train_components["categorical"],
                    "train_ordinal_loss": train_components["ordinal"],
                    "validation_total_loss": validation_loss,
                    "validation_categorical_loss": validation_components["categorical"],
                    "validation_ordinal_loss": validation_components["ordinal"],
                })
            self.training_log_.append(log_row)
            if epochs_without_improvement >= self.early_stopping_patience:
                self.stopping_reason_ = "early_stopping_patience"
                break

        if best_state is None or self.best_epoch_ is None:
            raise RuntimeError("Training finished without a valid model state")
        self.model.load_state_dict(best_state)
        self.model.to(self.device_)
        self.model.eval()
        self.best_validation_loss_ = best_monitor
        self.best_validation_components_ = best_validation_components
        self.n_epochs_trained_ = len(self.training_log_)
        if self.stopping_reason_ is None:
            self.stopping_reason_ = "max_epochs"
        if self.device_.type == "cuda":
            self.peak_gpu_memory_bytes_ = int(
                torch.cuda.max_memory_allocated(self.device_)
            )
        self.is_fitted_ = True
        return self

    def resolve_validation_indices(
        self,
        y: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rebuild the deterministic inner split without fitting the model.

        Group metadata must be configured through :meth:`set_validation_groups`
        before this method is called. The method is used to reconstruct the
        exact validation partition of an already completed checkpoint.
        """
        raw_labels = np.asarray(y)
        if raw_labels.ndim != 1:
            raise ValueError(
                f"Validation labels must be one-dimensional, got {raw_labels.shape}"
            )
        labels = self._validate_labels(raw_labels, len(raw_labels))
        train_idx, validation_idx = self._validation_indices(labels)
        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(validation_idx, dtype=np.int64),
        )

    def validation_partition_detailed(
        self,
        X: Any,
        y: Any,
        *,
        validation_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Predict the inner-validation partition with the fitted checkpoint.

        The outer-test partition is never accessed. Explicit indices are useful
        when reconstructing validation predictions for legacy checkpoints that
        did not persist ``inner_validation_indices_``.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "The model must be fitted or loaded before validation prediction"
            )
        features = self._validate_features(X)
        labels = self._validate_labels(y, len(features))
        if validation_indices is None:
            if self.inner_validation_indices_ is not None:
                indices = np.asarray(
                    self.inner_validation_indices_, dtype=np.int64
                )
            else:
                _, indices = self.resolve_validation_indices(labels)
        else:
            indices = np.asarray(validation_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(
                "validation_indices must be a non-empty one-dimensional array"
            )
        if np.any(indices < 0) or np.any(indices >= len(features)):
            raise IndexError(
                "validation_indices contain values outside the training partition"
            )
        if getattr(features, "is_lazy_raw_eeg", False):
            validation_features = np.stack(
                [np.asarray(features[int(index)]) for index in indices], axis=0
            )
        else:
            validation_features = features[indices]
        detailed = self.predict_detailed(validation_features)
        return {
            "indices": indices.copy(),
            "y_true": labels[indices].copy(),
            **detailed,
        }

    def fine_tune(
        self,
        X_train: Any,
        y_train: Any,
        *,
        X_validation: Optional[Any] = None,
        y_validation: Optional[Any] = None,
        trainable_parameter_prefixes: Optional[Sequence[str]] = None,
        max_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
    ) -> "TorchClassificationAdapter":
        """Fine-tune loaded weights on explicit calibration-only partitions."""
        if not self.is_fitted_:
            raise RuntimeError("A fitted or loaded base model is required")
        if self.head_type != "categorical":
            message = (
                "Auxiliary CORN calibration is not supported yet."
                if self.head_type == "categorical_corn"
                else (
                    "Ordinal calibration is not supported yet; "
                    f"received head_type={self.head_type!r}"
                )
            )
            raise NotImplementedError(message)
        features = self._validate_features(X_train)
        labels = self._validate_calibration_labels(y_train, len(features))
        has_validation = X_validation is not None or y_validation is not None
        if (X_validation is None) != (y_validation is None):
            raise ValueError(
                "X_validation and y_validation must be provided together"
            )
        validation_features: Optional[Any] = None
        validation_labels: Optional[np.ndarray] = None
        if has_validation:
            validation_features = self._validate_features(X_validation)
            validation_labels = self._validate_calibration_labels(
                y_validation, len(validation_features)
            )

        epochs = self.max_epochs if max_epochs is None else int(max_epochs)
        lr = self.learning_rate if learning_rate is None else float(learning_rate)
        decay = self.weight_decay if weight_decay is None else float(weight_decay)
        patience = (
            self.early_stopping_patience
            if early_stopping_patience is None
            else int(early_stopping_patience)
        )
        if epochs <= 0 or lr <= 0 or decay < 0 or patience <= 0:
            raise ValueError(
                "Fine-tuning epochs, learning rate, patience must be positive "
                "and weight decay non-negative"
            )

        prefixes = (
            None
            if trainable_parameter_prefixes is None
            else tuple(str(value) for value in trainable_parameter_prefixes)
        )
        trainable_names: list[str] = []
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = (
                prefixes is None or any(name.startswith(prefix) for prefix in prefixes)
            )
            if parameter.requires_grad:
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError("No model parameters were selected for fine-tuning")

        seed_torch(self.random_state)
        self.model.to(self.device_)
        if self.device_.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device_)
        transformed_train = self._transform_features(features)
        train_loader = self._make_loader(
            transformed_train, labels, shuffle=True
        )
        validation_loader: Optional[DataLoader] = None
        if validation_features is not None and validation_labels is not None:
            validation_loader = self._make_loader(
                self._transform_features(validation_features), validation_labels
            )

        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=lr,
            weight_decay=decay,
        )
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_loss = float("inf")
        epochs_without_improvement = 0
        self.training_log_ = []
        self.best_epoch_ = None
        self.best_validation_loss_ = None

        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            self.model.train()
            train_loss_numerator = 0.0
            train_loss_denominator = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device_, non_blocking=True)
                batch_labels = batch_labels.to(self.device_, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                raw_outputs = self.model(batch_features)
                loss_parts = self.objective_handler.loss_parts(
                    raw_outputs,
                    batch_labels,
                )
                loss = loss_parts.mean
                if not torch.isfinite(loss):
                    raise ValueError("Fine-tuning loss became NaN or infinite")
                loss.backward()
                optimizer.step()
                train_loss_numerator += float(
                    loss_parts.numerator.detach().item()
                )
                train_loss_denominator += float(
                    loss_parts.denominator.detach().item()
                )

            validation_loss: Optional[float] = None
            validation_accuracy: Optional[float] = None
            improved = False
            if validation_loader is not None:
                self.model.eval()
                validation_loss_numerator = 0.0
                validation_loss_denominator = 0.0
                validation_correct = 0
                validation_count = 0
                with torch.no_grad():
                    for batch_features, batch_labels in validation_loader:
                        batch_features = batch_features.to(
                            self.device_, non_blocking=True
                        )
                        batch_labels = batch_labels.to(
                            self.device_, non_blocking=True
                        )
                        raw_outputs = self.model(batch_features)
                        loss_parts = self.objective_handler.loss_parts(
                            raw_outputs,
                            batch_labels,
                        )
                        validation_loss_numerator += float(
                            loss_parts.numerator.detach().item()
                        )
                        validation_loss_denominator += float(
                            loss_parts.denominator.detach().item()
                        )
                        decoded = self.objective_handler.decode(raw_outputs)
                        validation_correct += int(
                            (decoded.y_pred == batch_labels).sum().item()
                        )
                        validation_count += len(batch_labels)
                if validation_loss_denominator <= 0:
                    raise RuntimeError(
                        "Validation loss denominator must be positive"
                    )
                validation_loss = (
                    validation_loss_numerator / validation_loss_denominator
                )
                validation_accuracy = validation_correct / validation_count
                improved = validation_loss < best_loss
                if improved:
                    best_loss = validation_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    self.best_epoch_ = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.training_log_.append({
                "epoch": epoch,
                "train_loss": train_loss_numerator / train_loss_denominator,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "is_best": improved,
                "epoch_time_seconds": time.perf_counter() - epoch_started,
            })
            if validation_loader is not None and (
                epochs_without_improvement >= patience
            ):
                break

        if validation_loader is not None:
            if best_state is None or self.best_epoch_ is None:
                raise RuntimeError("Fine-tuning finished without a valid state")
            self.model.load_state_dict(best_state)
            self.best_validation_loss_ = best_loss
        else:
            self.best_epoch_ = len(self.training_log_)
            self.best_validation_loss_ = None
        self.n_epochs_trained_ = len(self.training_log_)
        self.model.to(self.device_)
        self.model.eval()
        if self.device_.type == "cuda":
            self.peak_gpu_memory_bytes_ = int(
                torch.cuda.max_memory_allocated(self.device_)
            )
        self.is_fitted_ = True
        self.model_metadata["fine_tune_trainable_parameters"] = trainable_names
        return self

    def predict_detailed(self, X: Any) -> Dict[str, Any]:
        """Return canonical predictions plus optional ordinal diagnostics."""
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction")
        features = self._transform_features(self._validate_features(X))
        loader = self._make_loader(features)
        if self.task_type == "regression":
            predictions: list[np.ndarray] = []
            self.model.eval()
            with torch.no_grad():
                for (batch_features,) in loader:
                    outputs = self.model(
                        batch_features.to(self.device_, non_blocking=True)
                    )
                    if (
                        outputs.ndim != 2
                        or outputs.shape[1] != self.num_outputs
                        or not torch.isfinite(outputs).all()
                    ):
                        raise ValueError(
                            "Regression model returned invalid outputs with shape "
                            f"{tuple(outputs.shape)}"
                        )
                    predictions.append(outputs.cpu().numpy())
            values = np.concatenate(predictions, axis=0)
            y_pred = values[:, 0] if self.num_outputs == 1 else values
            return {
                "y_pred": y_pred,
                "raw_outputs": values,
                "head_type": "regression",
            }
        values: Dict[str, list[np.ndarray]] = {
            "class_probabilities": [],
            "y_pred": [],
            "raw_outputs": [],
        }
        optional_names = (
            "threshold_probabilities",
            "expected_rank",
            "ordinal_argmax",
            "conditional_probabilities",
            "categorical_expected_rank",
            "auxiliary_raw_outputs",
            "aux_threshold_probabilities",
            "aux_class_probabilities",
            "aux_expected_rank",
            "aux_ordinal_prediction",
            "aux_ordinal_argmax",
            "aux_conditional_probabilities",
        )
        optional_values: Dict[str, list[np.ndarray]] = {
            name: [] for name in optional_names
        }
        self.model.eval()
        with torch.no_grad():
            for (batch_features,) in loader:
                raw_outputs = self.model(
                    batch_features.to(self.device_, non_blocking=True)
                )
                decoded = self.objective_handler.decode(raw_outputs)
                values["class_probabilities"].append(
                    decoded.class_probabilities.cpu().numpy()
                )
                values["y_pred"].append(decoded.y_pred.cpu().numpy())
                values["raw_outputs"].append(decoded.raw_outputs.cpu().numpy())
                for name in optional_names:
                    tensor = getattr(decoded, name)
                    if tensor is not None:
                        optional_values[name].append(tensor.cpu().numpy())

        result: Dict[str, Any] = {
            name: np.concatenate(parts, axis=0)
            for name, parts in values.items()
        }
        result["head_type"] = self.head_type
        for name, parts in optional_values.items():
            if parts:
                result[name] = np.concatenate(parts, axis=0)
        if self.head_type in {"coral", "corn"}:
            result["threshold_logits"] = result["raw_outputs"]
            result["y_pred_argmax"] = result["ordinal_argmax"]
        elif self.head_type == "categorical_corn":
            result["categorical_logits"] = result["raw_outputs"]
            result["aux_ordinal_logits"] = result["auxiliary_raw_outputs"]
            result["auxiliary_weight"] = float(
                self.objective_handler.auxiliary_weight
            )
        return result

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task_type == "regression":
            raise AttributeError("predict_proba is unavailable for regression")
        return np.asarray(self.predict_detailed(X)["class_probabilities"])

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.predict_detailed(X)["y_pred"])

    def get_training_summary(self) -> Dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("The model has not been fitted")
        device_name = (
            torch.cuda.get_device_name(self.device_)
            if self.device_.type == "cuda"
            else "CPU"
        )
        head_diagnostics = getattr(self.model, "output_head_diagnostics", None)
        return {
            "input_shape": list(self.input_shape),
            "num_outputs": self.num_outputs,
            "task_type": self.task_type,
            "regression_loss": (
                self.regression_loss if self.task_type == "regression" else None
            ),
            "head_type": self.head_type,
            "device": str(self.device_),
            "device_name": device_name,
            "epochs_trained": self.n_epochs_trained_,
            "best_epoch": self.best_epoch_,
            "best_validation_loss": self.best_validation_loss_,
            "best_monitor_value": self.best_validation_loss_,
            "best_validation_components": dict(self.best_validation_components_),
            "early_stopping_monitor": self.early_stopping_monitor_,
            "stopping_reason": self.stopping_reason_,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes_,
            "validation_size": self.validation_size,
            "standardize": self.standardize,
            "validation_strategy": self.validation_strategy_,
            "validation_split": self.validation_split_,
            "validation_fallback_reason": self.validation_fallback_reason_,
            "objective_training_diagnostics": dict(
                self.objective_training_diagnostics_
            ),
            "head_diagnostics": (
                head_diagnostics() if callable(head_diagnostics) else {}
            ),
        }

    def save(self, path: PathLike) -> None:
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before it can be saved")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
            "model_metadata": self.model_metadata,
            "head_type": self.head_type,
            "task_type": self.task_type,
            "objective": self.objective_handler.to_metadata(),
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "num_outputs": self.num_outputs,
            "training_config": {
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "validation_size": self.validation_size,
                "early_stopping_patience": self.early_stopping_patience,
                "random_state": self.random_state,
                "standardize": self.standardize,
                "head_type": self.head_type,
                "task_type": self.task_type,
                "regression_loss": (
                    self.regression_loss
                    if self.task_type == "regression"
                    else None
                ),
                "auxiliary_weight": getattr(
                    self.objective_handler, "auxiliary_weight", None
                ),
            },
            "training_summary": self.get_training_summary(),
            "training_log": self.training_log_,
            "validation_split": self.validation_split_,
            "inner_train_indices": (
                None
                if self.inner_train_indices_ is None
                else torch.from_numpy(
                    np.asarray(self.inner_train_indices_, dtype=np.int64)
                )
            ),
            "inner_validation_indices": (
                None
                if self.inner_validation_indices_ is None
                else torch.from_numpy(
                    np.asarray(self.inner_validation_indices_, dtype=np.int64)
                )
            ),
            "feature_mean": (
                torch.from_numpy(self.feature_mean_)
                if self.feature_mean_ is not None
                else None
            ),
            "feature_scale": (
                torch.from_numpy(self.feature_scale_)
                if self.feature_scale_ is not None
                else None
            ),
        }
        torch.save(payload, output_path)

    def load(self, path: PathLike) -> "TorchClassificationAdapter":
        """Load a canonical adapter checkpoint into a factory-built instance."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        try:
            payload = torch.load(
                checkpoint_path, map_location=self.device_, weights_only=False
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=self.device_)
        stored_shape = tuple(int(value) for value in payload.get("input_shape", ()))
        stored_classes = int(payload.get("num_classes", -1))
        stored_task_type = str(
            payload.get(
                "task_type",
                payload.get("training_config", {}).get(
                    "task_type",
                    "classification",
                ),
            )
        ).strip().lower()
        stored_metadata = dict(payload.get("model_metadata", {}))
        stored_objective = dict(payload.get("objective", {}))
        stored_head_type = str(
            payload.get(
                "head_type",
                stored_objective.get(
                    "head_type",
                    stored_metadata.get("head_type", "categorical"),
                ),
            )
        ).strip().lower()
        if stored_shape != self.input_shape:
            raise ValueError(
                f"Checkpoint input_shape {stored_shape} does not match "
                f"adapter {self.input_shape}"
            )
        if stored_classes != self.num_classes:
            raise ValueError(
                f"Checkpoint num_classes {stored_classes} does not match "
                f"adapter {self.num_classes}"
            )
        if stored_task_type != self.task_type:
            raise ValueError(
                "Checkpoint task_type does not match the factory-built model: "
                f"{stored_task_type!r} != {self.task_type!r}"
            )
        if stored_head_type != self.head_type:
            raise ValueError(
                "Checkpoint head_type does not match the factory-built model: "
                f"{stored_head_type!r} != {self.head_type!r}. Automatic head "
                "weight conversion is not supported."
            )
        if self.head_type == "categorical_corn":
            stored_weight = float(stored_objective.get(
                "auxiliary_weight",
                payload.get("training_config", {}).get("auxiliary_weight", float("nan")),
            ))
            expected_weight = float(self.objective_handler.auxiliary_weight)
            if not np.isfinite(stored_weight) or stored_weight != expected_weight:
                raise ValueError(
                    "Checkpoint auxiliary_weight does not match the factory-built "
                    f"model: {stored_weight!r} != {expected_weight!r}"
                )
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device_)
        self.model.eval()
        self._initial_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        mean = payload.get("feature_mean")
        scale = payload.get("feature_scale")
        self.feature_mean_ = (
            None if mean is None else np.asarray(mean.cpu(), dtype=np.float32)
        )
        self.feature_scale_ = (
            None if scale is None else np.asarray(scale.cpu(), dtype=np.float32)
        )
        self.model_metadata = dict(payload.get("model_metadata", self.model_metadata))
        self.training_log_ = list(payload.get("training_log", []))
        self.validation_split_ = payload.get("validation_split")
        stored_train_indices = payload.get("inner_train_indices")
        stored_validation_indices = payload.get("inner_validation_indices")
        self.inner_train_indices_ = (
            None
            if stored_train_indices is None
            else np.asarray(
                stored_train_indices.cpu()
                if isinstance(stored_train_indices, torch.Tensor)
                else stored_train_indices,
                dtype=np.int64,
            )
        )
        self.inner_validation_indices_ = (
            None
            if stored_validation_indices is None
            else np.asarray(
                stored_validation_indices.cpu()
                if isinstance(stored_validation_indices, torch.Tensor)
                else stored_validation_indices,
                dtype=np.int64,
            )
        )
        summary = payload.get("training_summary", {})
        self.validation_fallback_reason_ = summary.get(
            "validation_fallback_reason"
        )
        self.best_epoch_ = summary.get("best_epoch")
        self.best_validation_loss_ = summary.get("best_validation_loss")
        self.best_validation_components_ = dict(
            summary.get("best_validation_components", {})
        )
        self.early_stopping_monitor_ = str(
            summary.get("early_stopping_monitor", "validation_loss")
        )
        self.n_epochs_trained_ = int(summary.get("epochs_trained", 0))
        self.stopping_reason_ = summary.get("stopping_reason")
        self.peak_gpu_memory_bytes_ = int(
            summary.get("peak_gpu_memory_bytes", 0)
        )
        self.objective_training_diagnostics_ = dict(
            summary.get("objective_training_diagnostics", {})
        )
        self.is_fitted_ = True
        return self

    def clone(self) -> "TorchClassificationAdapter":
        """Create an independent fitted adapter for one calibration subject."""
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted or loaded before cloning")
        cloned = deepcopy(self)
        cloned.model.to(cloned.device_)
        return cloned
