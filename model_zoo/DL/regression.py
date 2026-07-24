"""Regression objective used by the shared PyTorch training adapter."""

from typing import Any, Dict, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .ordinal import LossParts


class RegressionObjectiveHandler:
    """Multi-output MSE or Smooth-L1 objective with a linear output head."""

    head_type = "regression"

    def __init__(self, num_outputs: int, loss: str = "mse") -> None:
        self.num_outputs = int(num_outputs)
        self.num_classes = self.num_outputs
        self.loss_name = str(loss).strip().lower()
        if self.num_outputs <= 0:
            raise ValueError("num_outputs must be positive")
        if self.loss_name not in {"mse", "smooth_l1"}:
            raise ValueError(
                "regression_loss must be 'mse' or 'smooth_l1', "
                f"got {loss!r}"
            )

    @property
    def early_stopping_component(self) -> str:
        return "objective"

    def loss_component_parts(
        self,
        raw_outputs: Tensor,
        targets: Tensor,
    ) -> Dict[str, LossParts]:
        if raw_outputs.ndim != 2 or raw_outputs.shape[1] != self.num_outputs:
            raise ValueError(
                f"Regression outputs must have shape [batch, {self.num_outputs}], "
                f"got {tuple(raw_outputs.shape)}"
            )
        if targets.shape != raw_outputs.shape:
            raise ValueError(
                "Regression targets and outputs must have equal shape: "
                f"{tuple(targets.shape)} != {tuple(raw_outputs.shape)}"
            )
        if not torch.isfinite(raw_outputs).all() or not torch.isfinite(targets).all():
            raise ValueError("Regression outputs and targets must be finite")
        if self.loss_name == "mse":
            numerator = F.mse_loss(raw_outputs, targets, reduction="sum")
        else:
            numerator = F.smooth_l1_loss(
                raw_outputs,
                targets,
                reduction="sum",
            )
        return {
            "objective": LossParts(
                numerator=numerator,
                denominator=raw_outputs.new_tensor(raw_outputs.numel()),
            )
        }

    @staticmethod
    def combine_component_means(component_means: Mapping[str, Any]) -> Any:
        return component_means["objective"]

    @staticmethod
    def training_diagnostics(targets: Tensor) -> Dict[str, Any]:
        return {}

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "head_type": self.head_type,
            "task_type": "regression",
            "num_outputs": self.num_outputs,
            "output_semantics": "continuous_values",
            "loss": self.loss_name,
            "loss_normalization": "mean_over_samples_and_outputs",
        }
