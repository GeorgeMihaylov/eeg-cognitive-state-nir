from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn
from xgboost import XGBClassifier


def _softmax_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def xgboost_state_sha256(model: XGBClassifier) -> str:
    """Return a deterministic hash of the fitted XGBoost booster."""
    raw = model.get_booster().save_raw(raw_format="ubj")
    return hashlib.sha256(bytes(raw)).hexdigest()


def _torch_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


class XGBoostMarginHeadAdapter:
    """
    Personalization adapter over a frozen multiclass XGBoost classifier.

    The global XGBoost model is never fitted or mutated by this adapter.
    Personalization operates only on a small linear layer over raw
    multiclass margins.

    Before adaptation the head is initialized as

        W = I
        b = 0

    so softmax(head(margins)) reproduces the global XGBoost probabilities
    up to numerical precision.
    """

    def __init__(
        self,
        global_model: XGBClassifier,
        *,
        device: str = "cpu",
    ) -> None:
        self.global_model = global_model
        self.device = torch.device(device)

        classes = np.asarray(global_model.classes_)
        if classes.ndim != 1:
            raise ValueError("XGBoost classes_ must be one-dimensional")
        if len(classes) < 2:
            raise ValueError("At least two classes are required")

        self.classes_ = classes.copy()
        self.n_classes = int(len(classes))

        # Fail immediately if the estimator is not fitted.
        global_model.get_booster()

        self._global_model_hash = xgboost_state_sha256(global_model)

        self.head = nn.Linear(
            self.n_classes,
            self.n_classes,
            bias=True,
        ).to(self.device)

        self.reset_head()

    @property
    def global_model_hash(self) -> str:
        return self._global_model_hash

    @property
    def head_hash(self) -> str:
        return _torch_state_sha256(self.head)

    def reset_head(self) -> None:
        """Reset the personalization head to the identity transform."""
        with torch.no_grad():
            self.head.weight.copy_(
                torch.eye(
                    self.n_classes,
                    dtype=self.head.weight.dtype,
                    device=self.device,
                )
            )
            self.head.bias.zero_()

    def clone_for_participant(self) -> "XGBoostMarginHeadAdapter":
        """
        Create an independent participant adapter.

        The fitted XGBoost estimator remains shared and frozen.
        The personalization head is a new identity-initialized module.
        """
        return XGBoostMarginHeadAdapter(
            self.global_model,
            device=str(self.device),
        )

    def predict_margin(self, X: Any) -> np.ndarray:
        margins = np.asarray(
            self.global_model.predict(
                X,
                output_margin=True,
            )
        )

        if margins.ndim != 2:
            raise ValueError(
                "Expected multiclass XGBoost margins with shape "
                f"[n_samples, n_classes], got {margins.shape}"
            )

        if margins.shape[1] != self.n_classes:
            raise ValueError(
                "XGBoost margin dimension does not match classes_: "
                f"{margins.shape[1]} != {self.n_classes}"
            )

        return margins.astype(np.float32, copy=False)

    def zero_shot_predict_proba(self, X: Any) -> np.ndarray:
        """Return probabilities from the untouched global XGBoost."""
        return np.asarray(
            self.global_model.predict_proba(X),
            dtype=np.float64,
        )

    def identity_predict_proba(self, X: Any) -> np.ndarray:
        """
        Apply softmax directly to raw margins.

        Useful for verifying that the identity head reproduces XGBoost.
        """
        return _softmax_numpy(self.predict_margin(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        margins = torch.as_tensor(
            self.predict_margin(X),
            dtype=torch.float32,
            device=self.device,
        )

        self.head.eval()
        with torch.no_grad():
            logits = self.head(margins)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def identity_probability_error(self, X: Any) -> float:
        """
        Maximum absolute difference between the original XGBoost
        probabilities and probabilities reconstructed from raw margins.
        """
        baseline = self.zero_shot_predict_proba(X)
        reconstructed = self.identity_predict_proba(X)

        if baseline.shape != reconstructed.shape:
            raise ValueError(
                "Probability shape mismatch: "
                f"{baseline.shape} != {reconstructed.shape}"
            )

        return float(
            np.max(np.abs(baseline - reconstructed))
        )

    def fit_head(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_validation: Any,
        y_validation: np.ndarray,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        patience: int = 5,
    ) -> list[dict[str, float | int]]:
        """
        Fit only the participant-specific calibration head.

        The underlying XGBoost booster must remain byte-identical.
        """
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if patience <= 0:
            raise ValueError("patience must be positive")

        booster_hash_before = xgboost_state_sha256(
            self.global_model
        )

        train_margins = torch.as_tensor(
            self.predict_margin(X_train),
            dtype=torch.float32,
            device=self.device,
        )
        validation_margins = torch.as_tensor(
            self.predict_margin(X_validation),
            dtype=torch.float32,
            device=self.device,
        )

        y_train_indices = self._encode_labels(y_train)
        y_validation_indices = self._encode_labels(y_validation)

        train_targets = torch.as_tensor(
            y_train_indices,
            dtype=torch.long,
            device=self.device,
        )
        validation_targets = torch.as_tensor(
            y_validation_indices,
            dtype=torch.long,
            device=self.device,
        )

        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_state = deepcopy(self.head.state_dict())
        epochs_without_improvement = 0
        training_log: list[dict[str, float | int]] = []

        for epoch in range(1, int(max_epochs) + 1):
            self.head.train()

            optimizer.zero_grad(set_to_none=True)

            train_logits = self.head(train_margins)
            train_loss = criterion(
                train_logits,
                train_targets,
            )
            train_loss.backward()
            optimizer.step()

            self.head.eval()
            with torch.no_grad():
                validation_logits = self.head(
                    validation_margins
                )
                validation_loss = criterion(
                    validation_logits,
                    validation_targets,
                )

            train_loss_value = float(
                train_loss.detach().cpu().item()
            )
            validation_loss_value = float(
                validation_loss.detach().cpu().item()
            )

            training_log.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_value,
                    "validation_loss": validation_loss_value,
                }
            )

            if validation_loss_value < best_loss:
                best_loss = validation_loss_value
                best_state = deepcopy(
                    self.head.state_dict()
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        self.head.load_state_dict(best_state)
        self.head.eval()

        booster_hash_after = xgboost_state_sha256(
            self.global_model
        )

        if booster_hash_before != booster_hash_after:
            raise RuntimeError(
                "Frozen XGBoost booster changed during "
                "participant adaptation"
            )

        if booster_hash_after != self._global_model_hash:
            raise RuntimeError(
                "Frozen XGBoost booster differs from the "
                "state registered when the adapter was created"
            )

        return training_log

    def _encode_labels(
        self,
        labels: np.ndarray,
    ) -> np.ndarray:
        labels = np.asarray(labels)

        mapping = {
            label: index
            for index, label in enumerate(self.classes_.tolist())
        }

        try:
            encoded = np.asarray(
                [mapping[label] for label in labels.tolist()],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                f"Unknown calibration class: {exc.args[0]!r}"
            ) from exc

        return encoded