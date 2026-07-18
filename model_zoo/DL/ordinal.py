"""Architecture-neutral objectives and output heads for ordinal classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SUPPORTED_HEAD_TYPES = frozenset({"categorical", "coral", "corn"})
DEFAULT_PROBABILITY_TOLERANCE = 1e-7


def normalize_head_type(head_type: str) -> str:
    """Return a validated lower-case output-head name."""
    normalized = str(head_type).strip().lower()
    if normalized not in SUPPORTED_HEAD_TYPES:
        raise ValueError(
            f"Unsupported head_type {head_type!r}; expected one of "
            f"{sorted(SUPPORTED_HEAD_TYPES)}"
        )
    return normalized


def _validate_num_classes(num_classes: int) -> int:
    value = int(num_classes)
    if value < 2:
        raise ValueError(f"num_classes must be at least 2, got {value}")
    return value


def _validate_class_labels(labels: Tensor, num_classes: int) -> Tensor:
    classes = _validate_num_classes(num_classes)
    if not isinstance(labels, Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if labels.ndim != 1:
        raise ValueError(
            f"Ordinal labels must have shape [batch], got {tuple(labels.shape)}"
        )
    if labels.numel() == 0:
        raise ValueError("Ordinal labels cannot be empty")
    if labels.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("Ordinal labels must use an integer tensor dtype")
    integer_labels = labels.to(dtype=torch.int64)
    minimum = int(integer_labels.min().item())
    maximum = int(integer_labels.max().item())
    if minimum < 0 or maximum >= classes:
        raise ValueError(
            f"Ordinal labels must be in [0, {classes - 1}], "
            f"got minimum={minimum}, maximum={maximum}"
        )
    return integer_labels


def build_cumulative_targets(
    labels: Tensor,
    num_classes: int,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Encode class ``y`` as ``1[y > k]`` for ``k=0..K-2``."""
    integer_labels = _validate_class_labels(labels, num_classes)
    thresholds = torch.arange(
        int(num_classes) - 1,
        device=integer_labels.device,
        dtype=integer_labels.dtype,
    )
    target_dtype = torch.float32 if dtype is None else dtype
    if not torch.empty((), dtype=target_dtype).is_floating_point():
        raise ValueError("Cumulative targets require a floating-point dtype")
    return (integer_labels.unsqueeze(1) > thresholds).to(dtype=target_dtype)


def build_corn_targets_and_masks(
    labels: Tensor,
    num_classes: int,
    *,
    dtype: Optional[torch.dtype] = None,
) -> tuple[Tensor, Tensor]:
    """Return CORN binary targets and the conditional risk-set mask."""
    integer_labels = _validate_class_labels(labels, num_classes)
    target_dtype = torch.float32 if dtype is None else dtype
    targets = build_cumulative_targets(
        integer_labels,
        num_classes,
        dtype=target_dtype,
    )
    thresholds = torch.arange(
        int(num_classes) - 1,
        device=integer_labels.device,
        dtype=integer_labels.dtype,
    )
    masks = (integer_labels.unsqueeze(1) >= thresholds).to(dtype=target_dtype)
    return targets, masks


def _validate_raw_outputs(
    raw_outputs: Tensor,
    *,
    expected_width: int,
) -> Tensor:
    if not isinstance(raw_outputs, Tensor):
        raise TypeError("raw_outputs must be a torch.Tensor")
    if raw_outputs.ndim != 2 or raw_outputs.shape[1] != expected_width:
        raise ValueError(
            f"Expected raw_outputs with shape [batch, {expected_width}], "
            f"got {tuple(raw_outputs.shape)}"
        )
    if raw_outputs.shape[0] == 0:
        raise ValueError("raw_outputs cannot be empty")
    if not raw_outputs.is_floating_point():
        raise ValueError("raw_outputs must use a floating-point dtype")
    if not torch.isfinite(raw_outputs).all():
        raise ValueError("raw_outputs contain NaN or infinite values")
    return raw_outputs


@dataclass(frozen=True)
class LossParts:
    """A differentiable loss numerator and its aggregation denominator."""

    numerator: Tensor
    denominator: Tensor

    @property
    def mean(self) -> Tensor:
        if not torch.isfinite(self.numerator) or not torch.isfinite(self.denominator):
            raise ValueError("Loss numerator and denominator must be finite")
        if float(self.denominator.detach().item()) <= 0:
            raise ValueError("Loss denominator must be positive")
        return self.numerator / self.denominator


def coral_loss_parts(
    threshold_logits: Tensor,
    labels: Tensor,
    num_classes: int,
) -> LossParts:
    """Return the unweighted BCE sum and ``N*(K-1)`` denominator."""
    logits = _validate_raw_outputs(
        threshold_logits,
        expected_width=int(num_classes) - 1,
    )
    targets = build_cumulative_targets(
        labels.to(device=logits.device),
        num_classes,
        dtype=logits.dtype,
    )
    if targets.shape != logits.shape:
        raise ValueError(
            f"Targets and logits must have equal shape: "
            f"{tuple(targets.shape)} != {tuple(logits.shape)}"
        )
    numerator = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="sum",
    )
    denominator = logits.new_tensor(logits.numel())
    return LossParts(numerator=numerator, denominator=denominator)


def coral_loss(
    threshold_logits: Tensor,
    labels: Tensor,
    num_classes: int,
) -> Tensor:
    """Return mean unweighted CORAL BCE over samples and thresholds."""
    return coral_loss_parts(threshold_logits, labels, num_classes).mean


def corn_loss_parts(
    conditional_logits: Tensor,
    labels: Tensor,
    num_classes: int,
) -> LossParts:
    """Return masked CORN BCE sum and total valid risk-set elements."""
    logits = _validate_raw_outputs(
        conditional_logits,
        expected_width=int(num_classes) - 1,
    )
    targets, masks = build_corn_targets_and_masks(
        labels.to(device=logits.device),
        num_classes,
        dtype=logits.dtype,
    )
    if targets.shape != logits.shape or masks.shape != logits.shape:
        raise ValueError("CORN targets, masks and logits must have equal shapes")
    element_losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    numerator = (element_losses * masks).sum()
    denominator = masks.sum()
    if float(denominator.detach().item()) <= 0:
        raise ValueError("CORN risk-set mask contains no valid elements")
    return LossParts(numerator=numerator, denominator=denominator)


def corn_loss(
    conditional_logits: Tensor,
    labels: Tensor,
    num_classes: int,
) -> Tensor:
    """Return CORN BCE averaged over all valid risk-set elements."""
    return corn_loss_parts(conditional_logits, labels, num_classes).mean


def threshold_logits_to_cumulative_probabilities(
    threshold_logits: Tensor,
    head_type: str,
) -> Tensor:
    """Convert CORAL or CORN raw outputs to cumulative ``P(y > k)``."""
    normalized = normalize_head_type(head_type)
    if normalized == "categorical":
        raise ValueError("Categorical logits do not represent ordinal thresholds")
    if not isinstance(threshold_logits, Tensor) or threshold_logits.ndim != 2:
        shape = getattr(threshold_logits, "shape", None)
        raise ValueError(
            "threshold_logits must have shape [batch, num_classes - 1], "
            f"got {shape}"
        )
    logits = _validate_raw_outputs(
        threshold_logits,
        expected_width=threshold_logits.shape[1],
    )
    probabilities = torch.sigmoid(logits)
    cumulative = (
        probabilities
        if normalized == "coral"
        else torch.cumprod(probabilities, dim=1)
    )
    if not torch.isfinite(cumulative).all():
        raise ValueError("Cumulative probabilities contain NaN or infinite values")
    return cumulative


def cumulative_to_class_probabilities(
    cumulative_probabilities: Tensor,
    *,
    tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
) -> Tensor:
    """Convert monotone cumulative probabilities to a valid class distribution."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if not isinstance(cumulative_probabilities, Tensor):
        raise TypeError("cumulative_probabilities must be a torch.Tensor")
    q = cumulative_probabilities
    if q.ndim != 2 or q.shape[1] < 1:
        raise ValueError(
            "cumulative_probabilities must have shape [batch, num_classes - 1]"
        )
    if q.shape[0] == 0 or not q.is_floating_point():
        raise ValueError("cumulative_probabilities must be non-empty and floating-point")
    if not torch.isfinite(q).all():
        raise ValueError("Cumulative probabilities contain NaN or infinite values")
    if bool(((q < -tolerance) | (q > 1.0 + tolerance)).any()):
        raise ValueError("Cumulative probabilities fall outside [0, 1]")
    monotonic_violations = q[:, 1:] - q[:, :-1]
    if monotonic_violations.numel() and bool(
        (monotonic_violations > tolerance).any()
    ):
        maximum = float(monotonic_violations.max().detach().item())
        raise ValueError(
            "Cumulative probabilities are not monotone; "
            f"maximum violation={maximum:.8g}, tolerance={tolerance:.8g}"
        )

    bounded = q.clamp(min=0.0, max=1.0)
    class_probabilities = torch.cat(
        [
            1.0 - bounded[:, :1],
            bounded[:, :-1] - bounded[:, 1:],
            bounded[:, -1:],
        ],
        dim=1,
    )
    if bool((class_probabilities < -tolerance).any()):
        minimum = float(class_probabilities.min().detach().item())
        raise ValueError(
            "Class probability conversion produced a material negative value: "
            f"{minimum:.8g}"
        )
    class_probabilities = class_probabilities.clamp_min(0.0)
    row_sums = class_probabilities.sum(dim=1, keepdim=True)
    if not torch.isfinite(row_sums).all() or bool((row_sums <= 0).any()):
        raise ValueError("Class probability rows must have positive finite sums")
    maximum_sum_error = float(
        torch.abs(row_sums - 1.0).max().detach().item()
    )
    normalization_tolerance = max(
        10.0 * tolerance,
        10.0 * torch.finfo(class_probabilities.dtype).eps,
    )
    if maximum_sum_error > normalization_tolerance:
        raise ValueError(
            "Class probability row sum differs materially from one: "
            f"maximum error={maximum_sum_error:.8g}"
        )
    normalized = class_probabilities / row_sums
    if not torch.isfinite(normalized).all():
        raise ValueError("Normalized class probabilities are not finite")
    return normalized


def decode_ordinal_prediction(cumulative_probabilities: Tensor) -> Tensor:
    """Decode the primary ordinal class using the fixed ``q >= 0.5`` rule."""
    if cumulative_probabilities.ndim != 2:
        raise ValueError("cumulative_probabilities must be two-dimensional")
    if not torch.isfinite(cumulative_probabilities).all():
        raise ValueError("Cumulative probabilities must be finite")
    return (cumulative_probabilities >= 0.5).sum(dim=1).to(dtype=torch.int64)


def expected_rank(cumulative_probabilities: Tensor) -> Tensor:
    """Return ``E[y] = sum_k P(y > k)`` for ordinal classes ``0..K-1``."""
    if cumulative_probabilities.ndim != 2:
        raise ValueError("cumulative_probabilities must be two-dimensional")
    if not torch.isfinite(cumulative_probabilities).all():
        raise ValueError("Cumulative probabilities must be finite")
    ranks = cumulative_probabilities.sum(dim=1)
    upper = float(cumulative_probabilities.shape[1])
    if bool(((ranks < -1e-7) | (ranks > upper + 1e-7)).any()):
        raise ValueError("Expected ranks fall outside the valid class range")
    return ranks


@dataclass(frozen=True)
class DecodedClassificationOutput:
    """Unified decoded outputs consumed by the adapter and artifact writer."""

    class_probabilities: Tensor
    y_pred: Tensor
    raw_outputs: Tensor
    head_type: str
    threshold_probabilities: Optional[Tensor] = None
    expected_rank: Optional[Tensor] = None
    ordinal_argmax: Optional[Tensor] = None
    conditional_probabilities: Optional[Tensor] = None


class ClassificationObjectiveHandler:
    """Centralize categorical, CORAL and CORN loss/output semantics."""

    def __init__(
        self,
        head_type: str,
        num_classes: int,
        *,
        tolerance: float = DEFAULT_PROBABILITY_TOLERANCE,
    ) -> None:
        self.head_type = normalize_head_type(head_type)
        self.num_classes = _validate_num_classes(num_classes)
        self.tolerance = float(tolerance)
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")

    def loss_parts(self, raw_outputs: Tensor, targets: Tensor) -> LossParts:
        if self.head_type == "categorical":
            logits = _validate_raw_outputs(
                raw_outputs,
                expected_width=self.num_classes,
            )
            labels = _validate_class_labels(
                targets.to(device=logits.device),
                self.num_classes,
            )
            numerator = F.cross_entropy(logits, labels, reduction="sum")
            denominator = logits.new_tensor(logits.shape[0])
            return LossParts(numerator=numerator, denominator=denominator)
        if self.head_type == "coral":
            return coral_loss_parts(raw_outputs, targets, self.num_classes)
        return corn_loss_parts(raw_outputs, targets, self.num_classes)

    def compute_loss(self, raw_outputs: Tensor, targets: Tensor) -> Tensor:
        return self.loss_parts(raw_outputs, targets).mean

    def training_diagnostics(self, targets: Tensor) -> Dict[str, int]:
        """Return objective-specific counts for a complete training partition."""
        if self.head_type != "corn":
            return {}
        _, masks = build_corn_targets_and_masks(
            targets,
            self.num_classes,
            dtype=torch.float64,
        )
        counts = masks.sum(dim=0).to(dtype=torch.int64).cpu().tolist()
        return {
            f"risk_count_{threshold}": int(count)
            for threshold, count in enumerate(counts)
        }

    def decode(self, raw_outputs: Tensor) -> DecodedClassificationOutput:
        expected_width = (
            self.num_classes
            if self.head_type == "categorical"
            else self.num_classes - 1
        )
        outputs = _validate_raw_outputs(
            raw_outputs,
            expected_width=expected_width,
        )
        if self.head_type == "categorical":
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)
            return DecodedClassificationOutput(
                class_probabilities=probabilities,
                y_pred=predictions,
                raw_outputs=outputs,
                head_type=self.head_type,
            )

        cumulative = threshold_logits_to_cumulative_probabilities(
            outputs,
            self.head_type,
        )
        probabilities = cumulative_to_class_probabilities(
            cumulative,
            tolerance=self.tolerance,
        )
        return DecodedClassificationOutput(
            class_probabilities=probabilities,
            y_pred=decode_ordinal_prediction(cumulative),
            raw_outputs=outputs,
            head_type=self.head_type,
            threshold_probabilities=cumulative,
            expected_rank=expected_rank(cumulative),
            ordinal_argmax=probabilities.argmax(dim=1),
            conditional_probabilities=(
                torch.sigmoid(outputs) if self.head_type == "corn" else None
            ),
        )

    def to_metadata(self) -> Dict[str, Any]:
        semantics = {
            "categorical": "class_logits",
            "coral": "cumulative_threshold_logits",
            "corn": "conditional_threshold_logits",
        }
        losses = {
            "categorical": "cross_entropy_mean_over_samples",
            "coral": "unweighted_bce_sum_over_n_times_k_minus_one",
            "corn": "unweighted_masked_bce_sum_over_valid_risk_elements",
        }
        return {
            "head_type": self.head_type,
            "num_classes": self.num_classes,
            "num_thresholds": (
                None if self.head_type == "categorical" else self.num_classes - 1
            ),
            "output_semantics": semantics[self.head_type],
            "loss_normalization": losses[self.head_type],
            "probability_tolerance": self.tolerance,
            "prediction_rule": (
                "argmax_softmax"
                if self.head_type == "categorical"
                else "count_cumulative_probability_ge_0.5"
            ),
        }


class _OrdinalHeadFeatures(nn.Module):
    """Match the existing categorical pre-logit head capacity."""

    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class CoralOrdinalHead(nn.Module):
    """Rank-consistent CORAL head with shared score and ordered cutpoints."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.1,
        cutpoint_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = _validate_num_classes(num_classes)
        self.num_thresholds = self.num_classes - 1
        self.cutpoint_epsilon = float(cutpoint_epsilon)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.cutpoint_epsilon <= 0:
            raise ValueError("cutpoint_epsilon must be positive")

        self.features = _OrdinalHeadFeatures(self.input_dim, dropout)
        self.score = nn.Linear(self.input_dim, 1, bias=True)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        if self.num_thresholds > 1:
            desired_increment = 1.0 - self.cutpoint_epsilon
            raw_increment = math.log(math.expm1(desired_increment))
            self.raw_cutpoint_deltas = nn.Parameter(
                torch.full((self.num_thresholds - 1,), raw_increment)
            )
        else:
            self.register_parameter("raw_cutpoint_deltas", None)

    def cutpoints(self) -> Tensor:
        first = self.first_cutpoint.reshape(1)
        if self.raw_cutpoint_deltas is None:
            return first
        increments = F.softplus(self.raw_cutpoint_deltas) + self.cutpoint_epsilon
        return torch.cat([first, first + torch.cumsum(increments, dim=0)])

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"CoralOrdinalHead expects [batch, {self.input_dim}], "
                f"got {tuple(values.shape)}"
            )
        scores = self.score(self.features(values))
        return scores - self.cutpoints().to(
            device=scores.device,
            dtype=scores.dtype,
        ).unsqueeze(0)


class CornOrdinalHead(nn.Module):
    """Small CORN head producing one conditional logit per threshold."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = _validate_num_classes(num_classes)
        self.num_thresholds = self.num_classes - 1
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.features = _OrdinalHeadFeatures(self.input_dim, dropout)
        self.output = nn.Linear(self.input_dim, self.num_thresholds)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"CornOrdinalHead expects [batch, {self.input_dim}], "
                f"got {tuple(values.shape)}"
            )
        return self.output(self.features(values))


__all__ = [
    "ClassificationObjectiveHandler",
    "CoralOrdinalHead",
    "CornOrdinalHead",
    "DecodedClassificationOutput",
    "LossParts",
    "SUPPORTED_HEAD_TYPES",
    "build_corn_targets_and_masks",
    "build_cumulative_targets",
    "coral_loss",
    "coral_loss_parts",
    "corn_loss",
    "corn_loss_parts",
    "cumulative_to_class_probabilities",
    "decode_ordinal_prediction",
    "expected_rank",
    "normalize_head_type",
    "threshold_logits_to_cumulative_probabilities",
]
