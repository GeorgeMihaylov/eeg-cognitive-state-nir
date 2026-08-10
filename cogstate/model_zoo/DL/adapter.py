import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..base import BaseModelAdapter, PathLike


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
        return features, torch.as_tensor(self.labels[index], dtype=torch.int64)


class TorchClassificationAdapter(BaseModelAdapter):
    """Sklearn-like trainer for a PyTorch classification module."""

    def __init__(
        self,
        model: nn.Module,
        input_shape: Sequence[int],
        num_classes: int,
        *,
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
    ) -> None:
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.num_classes = int(num_classes)
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
        self.n_epochs_trained_: int = 0
        self.is_fitted_: bool = False
        self.validation_strategy_ = "stratified_random"
        self.validation_group_column_: Optional[str] = None
        self.validation_random_state_ = self.random_state
        self.validation_split_: Optional[Dict[str, Any]] = None
        self.inner_train_indices_: Optional[np.ndarray] = None
        self.inner_validation_indices_: Optional[np.ndarray] = None
        self._validation_groups: Optional[np.ndarray] = None
        self._validation_subject_ids: Optional[np.ndarray] = None
        self._validation_record_ids: Optional[np.ndarray] = None
        self._outer_test_record_ids: Optional[np.ndarray] = None
        self._validation_warning: Optional[str] = None

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
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
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
        strategy: str = "group_record",
        group_column: str = "record_id",
        validation_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> "TorchClassificationAdapter":
        """Configure record-disjoint validation for the next ``fit`` call."""
        normalized_strategy = strategy.strip().lower()
        if normalized_strategy != "group_record":
            raise ValueError(
                "Group validation currently supports strategy='group_record' only"
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
                "strategy='group_record' requires validation groups before fit"
            )
        if len(self._validation_groups) != len(labels):
            raise ValueError(
                "Validation groups must match training rows: "
                f"{len(self._validation_groups)} != {len(labels)}"
            )
        unique_groups = np.unique(self._validation_groups)
        if len(unique_groups) < 2:
            raise ValueError(
                "Group-aware validation requires at least two unique records, "
                f"got {len(unique_groups)}"
            )

        n_candidates = min(128, max(32, len(unique_groups) * 4))
        splitter = GroupShuffleSplit(
            n_splits=n_candidates,
            test_size=self.validation_size,
            random_state=self.validation_random_state_,
        )
        all_classes = np.unique(labels)
        overall_counts = np.asarray([
            np.sum(labels == class_label)
            for class_label in all_classes
        ], dtype=np.float64)
        overall_distribution = overall_counts / overall_counts.sum()
        best_indices: Optional[tuple[np.ndarray, np.ndarray]] = None
        best_score: Optional[tuple[int, float, float]] = None

        for train_idx, validation_idx in splitter.split(
            np.zeros(len(labels)), labels, self._validation_groups
        ):
            train_classes = np.unique(labels[train_idx])
            validation_classes = np.unique(labels[validation_idx])
            missing_classes = (
                len(np.setdiff1d(all_classes, train_classes))
                + len(np.setdiff1d(all_classes, validation_classes))
            )
            actual_fraction = len(validation_idx) / len(labels)
            size_error = abs(actual_fraction - self.validation_size)
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
            score = (missing_classes, size_error, balance_error)
            if best_score is None or score < best_score:
                best_score = score
                best_indices = (train_idx, validation_idx)

        if best_indices is None or best_score is None:
            raise ValueError("Could not create a group-aware validation split")
        if best_score[0] > 0:
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
        if self.validation_strategy_ == "group_record":
            return self._group_validation_indices(labels)
        indices = np.arange(len(labels), dtype=np.int64)
        try:
            train_idx, validation_idx = train_test_split(
                indices,
                test_size=self.validation_size,
                random_state=self.random_state,
                shuffle=True,
                stratify=labels,
            )
        except ValueError as exc:
            raise ValueError(
                "Could not create a stratified validation split inside training data"
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
        return {
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
            "class_balance_warning": self._validation_warning,
        }

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
        criterion = nn.CrossEntropyLoss()
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_loss = float("inf")
        epochs_without_improvement = 0
        self.training_log_ = []
        self.best_epoch_ = None
        self.best_validation_loss_ = None

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            train_count = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device_, non_blocking=True)
                batch_labels = batch_labels.to(self.device_, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_features)
                loss = criterion(logits, batch_labels)
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became NaN or infinite")
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item()) * len(batch_labels)
                train_count += len(batch_labels)

            self.model.eval()
            validation_loss_sum = 0.0
            validation_correct = 0
            validation_count = 0
            with torch.no_grad():
                for batch_features, batch_labels in validation_loader:
                    batch_features = batch_features.to(self.device_, non_blocking=True)
                    batch_labels = batch_labels.to(self.device_, non_blocking=True)
                    logits = self.model(batch_features)
                    loss = criterion(logits, batch_labels)
                    validation_loss_sum += float(loss.item()) * len(batch_labels)
                    validation_correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
                    validation_count += len(batch_labels)

            train_loss = train_loss_sum / train_count
            validation_loss = validation_loss_sum / validation_count
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

            self.training_log_.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_correct / validation_count,
                    "is_best": improved,
                }
            )
            if epochs_without_improvement >= self.early_stopping_patience:
                break

        if best_state is None or self.best_epoch_ is None:
            raise RuntimeError("Training finished without a valid model state")
        self.model.load_state_dict(best_state)
        self.model.to(self.device_)
        self.model.eval()
        self.best_validation_loss_ = best_loss
        self.n_epochs_trained_ = len(self.training_log_)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction")
        features = self._transform_features(self._validate_features(X))
        loader = self._make_loader(features)
        probabilities = []
        self.model.eval()
        with torch.no_grad():
            for (batch_features,) in loader:
                logits = self.model(batch_features.to(self.device_, non_blocking=True))
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    def predict(self, X: Any) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def get_training_summary(self) -> Dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("The model has not been fitted")
        device_name = (
            torch.cuda.get_device_name(self.device_)
            if self.device_.type == "cuda"
            else "CPU"
        )
        return {
            "input_shape": list(self.input_shape),
            "num_outputs": self.num_classes,
            "device": str(self.device_),
            "device_name": device_name,
            "epochs_trained": self.n_epochs_trained_,
            "best_epoch": self.best_epoch_,
            "best_validation_loss": self.best_validation_loss_,
            "validation_size": self.validation_size,
            "standardize": self.standardize,
            "validation_strategy": self.validation_strategy_,
            "validation_split": self.validation_split_,
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
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "training_config": {
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "validation_size": self.validation_size,
                "early_stopping_patience": self.early_stopping_patience,
                "random_state": self.random_state,
                "standardize": self.standardize,
            },
            "training_summary": self.get_training_summary(),
            "training_log": self.training_log_,
            "validation_split": self.validation_split_,
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
