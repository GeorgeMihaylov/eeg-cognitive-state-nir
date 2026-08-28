"""Masked-label multitask extension of the canonical PyTorch adapter."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from .adapter import TorchClassificationAdapter, seed_torch


class TorchMultiTaskClassificationAdapter(TorchClassificationAdapter):
    """Shared PyTorch encoder with one masked classification head per target."""

    def __init__(
        self,
        model: nn.Module,
        input_shape: Sequence[int],
        num_classes: int,
        *,
        metric_names: Sequence[str],
        **kwargs: Any,
    ) -> None:
        self.metric_names = tuple(str(name) for name in metric_names)
        if not self.metric_names or len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names must contain unique target names")
        super().__init__(model, input_shape, num_classes, **kwargs)

    def _validate_multitask_labels(self, y: Any, n_samples: int) -> np.ndarray:
        labels = np.asarray(y)
        expected_shape = (n_samples, len(self.metric_names))
        if labels.shape != expected_shape:
            raise ValueError(
                f"Expected multitask y with shape {expected_shape}, got {labels.shape}"
            )
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError("Multitask labels must be numeric")
        numeric = labels.astype(np.float64, copy=False)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("Multitask labels must contain finite integers")
        normalized = numeric.astype(np.int64)
        if normalized.min() < -1 or normalized.max() >= self.num_classes:
            raise ValueError(f"Labels must be -1 or in [0, {self.num_classes - 1}]")
        for index, metric in enumerate(self.metric_names):
            valid = normalized[:, index] >= 0
            classes, counts = np.unique(normalized[valid, index], return_counts=True)
            if len(classes) < 2:
                raise ValueError(f"Target {metric!r} needs at least two classes")
            if counts.min() < 2:
                raise ValueError(
                    f"Every observed class of target {metric!r} needs at least two rows"
                )
        return normalized

    @staticmethod
    def _masked_loss_and_accuracy(
        logits: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        metric_names: Sequence[str],
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, int, int]:
        weighted_losses: list[torch.Tensor] = []
        total_valid = 0
        total_correct = 0
        for index, metric in enumerate(metric_names):
            valid = labels[:, index] >= 0
            valid_count = int(valid.sum().item())
            if not valid_count:
                continue
            target_logits = logits[metric][valid]
            target_labels = labels[valid, index]
            weighted_losses.append(criterion(target_logits, target_labels) * valid_count)
            total_valid += valid_count
            total_correct += int(
                (target_logits.argmax(dim=1) == target_labels).sum().item()
            )
        if not weighted_losses or not total_valid:
            raise ValueError("A training batch contains no valid multitask labels")
        return torch.stack(weighted_losses).sum() / total_valid, total_correct, total_valid

    def fit(self, X: Any, y: Any) -> "TorchMultiTaskClassificationAdapter":
        features = self._validate_features(X)
        labels = self._validate_multitask_labels(y, len(features))
        seed_torch(self.random_state)
        self.model.load_state_dict(self._initial_state)
        self.model.to(self.device_)

        split_labels = labels[:, 0]
        try:
            train_idx, validation_idx = self._validation_indices(split_labels)
        except ValueError:
            if self.validation_strategy_ == "group_record":
                raise
            indices = np.arange(len(labels), dtype=np.int64)
            train_idx, validation_idx = train_test_split(
                indices,
                test_size=self.validation_size,
                random_state=self.random_state,
                shuffle=True,
            )
        self.inner_train_indices_ = np.asarray(train_idx, dtype=np.int64)
        self.inner_validation_indices_ = np.asarray(validation_idx, dtype=np.int64)
        self.validation_split_ = self._validation_summary(
            split_labels,
            self.inner_train_indices_,
            self.inner_validation_indices_,
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
            train_valid = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device_, non_blocking=True)
                batch_labels = batch_labels.to(self.device_, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, _, valid_count = self._masked_loss_and_accuracy(
                    self.model(batch_features),
                    batch_labels,
                    self.metric_names,
                    criterion,
                )
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became NaN or infinite")
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item()) * valid_count
                train_valid += valid_count

            self.model.eval()
            validation_loss_sum = 0.0
            validation_correct = 0
            validation_valid = 0
            with torch.no_grad():
                for batch_features, batch_labels in validation_loader:
                    batch_features = batch_features.to(self.device_, non_blocking=True)
                    batch_labels = batch_labels.to(self.device_, non_blocking=True)
                    loss, correct, valid_count = self._masked_loss_and_accuracy(
                        self.model(batch_features),
                        batch_labels,
                        self.metric_names,
                        criterion,
                    )
                    validation_loss_sum += float(loss.item()) * valid_count
                    validation_correct += correct
                    validation_valid += valid_count

            train_loss = train_loss_sum / train_valid
            validation_loss = validation_loss_sum / validation_valid
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
                    "validation_accuracy": validation_correct / validation_valid,
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

    def predict_proba(self, X: Any) -> dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("The model must be fitted before prediction")
        features = self._transform_features(self._validate_features(X))
        loader = self._make_loader(features)
        collected: dict[str, list[np.ndarray]] = {
            metric: [] for metric in self.metric_names
        }
        self.model.eval()
        with torch.no_grad():
            for (batch_features,) in loader:
                logits = self.model(batch_features.to(self.device_, non_blocking=True))
                for metric in self.metric_names:
                    collected[metric].append(
                        torch.softmax(logits[metric], dim=1).cpu().numpy()
                    )
        return {
            metric: np.concatenate(probabilities, axis=0)
            for metric, probabilities in collected.items()
        }

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.column_stack(
            [probabilities[metric].argmax(axis=1) for metric in self.metric_names]
        )

    def get_training_summary(self) -> Dict[str, Any]:
        summary = super().get_training_summary()
        summary["metric_names"] = list(self.metric_names)
        summary["n_targets"] = len(self.metric_names)
        return summary


__all__ = [
    "TorchClassificationAdapter",
    "TorchMultiTaskClassificationAdapter",
    "seed_torch",
]
