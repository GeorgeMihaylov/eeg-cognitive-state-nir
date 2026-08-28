"""Fold-scoped building blocks for domain-adversarial encoder training.

This module intentionally provides infrastructure rather than a second
training loop.  It composes a model implementing the shared encoder contract
with a separate domain head, an explicit objective, and a training-only data
view that cannot expose outer-test samples or target task labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset

from .encoder import EncoderModelProtocol, require_encoder_model
from .ordinal import LossParts


DANN_CHECKPOINT_SCHEMA_VERSION = 1


def _validate_nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, alpha: float) -> Tensor:
        ctx.alpha = float(alpha)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.alpha * gradient, None


class GradientReversal(nn.Module):
    """Identity on the forward pass and ``-alpha`` on backward gradients."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = _validate_nonnegative_finite(
            alpha, "gradient_reversal_alpha"
        )

    def forward(self, inputs: Tensor, alpha: Optional[float] = None) -> Tensor:
        if not inputs.is_floating_point():
            raise ValueError("GradientReversal expects floating-point features")
        coefficient = self.alpha if alpha is None else _validate_nonnegative_finite(
            alpha, "gradient_reversal_alpha"
        )
        return _GradientReversalFunction.apply(inputs, coefficient)


class DomainDiscriminator(nn.Module):
    """Configurable domain head whose input width comes from ``latent_dim``."""

    def __init__(
        self,
        input_dim: int,
        n_domains: int,
        *,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_domains = int(n_domains)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.dropout = float(dropout)
        if self.input_dim <= 0:
            raise ValueError("DomainDiscriminator input_dim must be positive")
        if self.n_domains < 2:
            raise ValueError("DomainDiscriminator n_domains must be at least 2")
        if any(width <= 0 for width in self.hidden_dims):
            raise ValueError("DomainDiscriminator hidden_dims must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("DomainDiscriminator dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        current_width = self.input_dim
        for width in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_width, width),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ]
            )
            current_width = width
        layers.append(nn.Linear(current_width, self.n_domains))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "DomainDiscriminator expects [batch, input_dim] with input_dim="
                f"{self.input_dim}, got {tuple(features.shape)}"
            )
        if features.shape[0] == 0:
            raise ValueError("DomainDiscriminator batch cannot be empty")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError(
                "DomainDiscriminator features must be finite floating-point values"
            )
        return self.network(features)


@dataclass(frozen=True)
class DANNForwardResult:
    """Outputs from one source/target DANN forward pass."""

    source_task_outputs: Tensor
    domain_outputs: Tensor
    source_latent: Tensor
    target_latent: Tensor

    @property
    def combined_latent(self) -> Tensor:
        return torch.cat((self.source_latent, self.target_latent), dim=0)


@dataclass(frozen=True)
class DANNLossResult:
    """Differentiable DANN losses with aggregation-safe components."""

    task: LossParts
    domain: LossParts
    total_loss: Tensor
    domain_correct: Tensor
    domain_count: Tensor
    lambda_domain: float

    @property
    def task_loss(self) -> Tensor:
        return self.task.mean

    @property
    def domain_loss(self) -> Tensor:
        return self.domain.mean

    @property
    def domain_accuracy(self) -> Tensor:
        if float(self.domain_count.detach().item()) <= 0:
            raise ValueError("Domain accuracy denominator must be positive")
        return self.domain_correct / self.domain_count

    def detached_metrics(self) -> dict[str, float]:
        return {
            "task_loss": float(self.task_loss.detach().cpu().item()),
            "domain_loss": float(self.domain_loss.detach().cpu().item()),
            "total_loss": float(self.total_loss.detach().cpu().item()),
            "domain_accuracy": float(
                self.domain_accuracy.detach().cpu().item()
            ),
        }


class DANNObjective:
    """Joint task/domain loss with independent domain weighting."""

    def __init__(
        self,
        *,
        task_type: str = "classification",
        lambda_domain: float = 1.0,
    ) -> None:
        normalized_task = str(task_type).strip().lower()
        if normalized_task not in {"classification", "regression"}:
            raise ValueError(
                "DANNObjective task_type must be classification or regression"
            )
        self.task_type = normalized_task
        self.lambda_domain = _validate_nonnegative_finite(
            lambda_domain, "lambda_domain"
        )

    def _task_parts(self, outputs: Tensor, targets: Tensor) -> LossParts:
        if outputs.shape[0] == 0:
            raise ValueError("Source task batch cannot be empty")
        if not outputs.is_floating_point() or not torch.isfinite(outputs).all():
            raise ValueError("Task outputs must be finite floating-point values")
        if self.task_type == "classification":
            if outputs.ndim != 2 or outputs.shape[1] < 2:
                raise ValueError(
                    "Classification task outputs must have shape "
                    "[batch, num_classes]"
                )
            labels = targets.to(device=outputs.device, dtype=torch.long)
            if labels.ndim != 1 or labels.shape[0] != outputs.shape[0]:
                raise ValueError(
                    "Classification task targets must have shape [source_batch]"
                )
            if int(labels.min().item()) < 0 or int(labels.max().item()) >= outputs.shape[1]:
                raise ValueError("Classification task targets are outside class range")
            return LossParts(
                numerator=F.cross_entropy(outputs, labels, reduction="sum"),
                denominator=outputs.new_tensor(outputs.shape[0]),
            )

        values = targets.to(device=outputs.device, dtype=outputs.dtype)
        if values.shape != outputs.shape:
            raise ValueError(
                "Regression task targets must match task outputs exactly: "
                f"{tuple(values.shape)} != {tuple(outputs.shape)}"
            )
        if not torch.isfinite(values).all():
            raise ValueError("Regression task targets contain NaN or infinite values")
        return LossParts(
            numerator=F.mse_loss(outputs, values, reduction="sum"),
            denominator=outputs.new_tensor(outputs.numel()),
        )

    def __call__(
        self,
        outputs: DANNForwardResult,
        source_task_targets: Tensor,
        domain_ids: Tensor,
    ) -> DANNLossResult:
        task = self._task_parts(
            outputs.source_task_outputs, source_task_targets
        )
        domain_logits = outputs.domain_outputs
        if domain_logits.ndim != 2 or domain_logits.shape[1] < 2:
            raise ValueError(
                "Domain outputs must have shape [source+target, n_domains]"
            )
        labels = domain_ids.to(device=domain_logits.device, dtype=torch.long)
        if labels.ndim != 1 or labels.shape[0] != domain_logits.shape[0]:
            raise ValueError(
                "domain_ids must have shape [source_batch + target_batch]"
            )
        if int(labels.min().item()) < 0 or int(labels.max().item()) >= domain_logits.shape[1]:
            raise ValueError("domain_ids are outside configured domain range")
        domain = LossParts(
            numerator=F.cross_entropy(domain_logits, labels, reduction="sum"),
            denominator=domain_logits.new_tensor(domain_logits.shape[0]),
        )
        total_loss = task.mean + self.lambda_domain * domain.mean
        predicted_domains = domain_logits.argmax(dim=1)
        return DANNLossResult(
            task=task,
            domain=domain,
            total_loss=total_loss,
            domain_correct=(predicted_domains == labels).sum().to(domain_logits.dtype),
            domain_count=domain_logits.new_tensor(labels.numel()),
            lambda_domain=self.lambda_domain,
        )


def aggregate_dann_loss_results(
    results: Iterable[DANNLossResult],
) -> dict[str, float]:
    """Aggregate batch results by summed numerators and denominators."""

    items = list(results)
    if not items:
        raise ValueError("At least one DANNLossResult is required")
    weights = {item.lambda_domain for item in items}
    if len(weights) != 1:
        raise ValueError("Cannot aggregate different lambda_domain values")
    task_numerator = sum(float(item.task.numerator.detach().item()) for item in items)
    task_denominator = sum(
        float(item.task.denominator.detach().item()) for item in items
    )
    domain_numerator = sum(
        float(item.domain.numerator.detach().item()) for item in items
    )
    domain_denominator = sum(
        float(item.domain.denominator.detach().item()) for item in items
    )
    domain_correct = sum(
        float(item.domain_correct.detach().item()) for item in items
    )
    domain_count = sum(float(item.domain_count.detach().item()) for item in items)
    if min(task_denominator, domain_denominator, domain_count) <= 0:
        raise ValueError("DANN aggregate denominators must be positive")
    task_loss = task_numerator / task_denominator
    domain_loss = domain_numerator / domain_denominator
    lambda_domain = weights.pop()
    return {
        "task_loss": task_loss,
        "domain_loss": domain_loss,
        "total_loss": task_loss + lambda_domain * domain_loss,
        "domain_accuracy": domain_correct / domain_count,
    }


class DANNModule(nn.Module):
    """Compose an encoder-compatible task model with a separate domain head."""

    def __init__(
        self,
        task_model: nn.Module,
        *,
        n_domains: int,
        gradient_reversal_alpha: float = 1.0,
        domain_hidden_dims: Sequence[int] = (128, 64),
        domain_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        encoder = require_encoder_model(task_model)
        self.task_model = task_model
        self.n_domains = int(n_domains)
        self.gradient_reversal = GradientReversal(gradient_reversal_alpha)
        self.domain_discriminator = DomainDiscriminator(
            encoder.latent_dim,
            self.n_domains,
            hidden_dims=domain_hidden_dims,
            dropout=domain_dropout,
        )

    @property
    def encoder(self) -> EncoderModelProtocol:
        return require_encoder_model(self.task_model)

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def gradient_reversal_alpha(self) -> float:
        return self.gradient_reversal.alpha

    def forward(
        self,
        source_inputs: Tensor,
        target_inputs: Tensor,
        *,
        gradient_reversal_alpha: Optional[float] = None,
    ) -> DANNForwardResult:
        if source_inputs.shape[0] == 0 or target_inputs.shape[0] == 0:
            raise ValueError("DANN source and target batches must be non-empty")
        source_latent = self.encoder.encode(source_inputs)
        target_latent = self.encoder.encode(target_inputs)
        for name, features in (
            ("source", source_latent),
            ("target", target_latent),
        ):
            if (
                features.ndim != 2
                or features.shape[1] != self.latent_dim
                or not torch.isfinite(features).all()
            ):
                raise ValueError(
                    f"DANN {name} latent features must be finite with shape "
                    f"[batch, {self.latent_dim}], got {tuple(features.shape)}"
                )
        task_outputs = self.encoder.forward_head(source_latent)
        combined = torch.cat((source_latent, target_latent), dim=0)
        reversed_features = self.gradient_reversal(
            combined, alpha=gradient_reversal_alpha
        )
        domain_outputs = self.domain_discriminator(reversed_features)
        return DANNForwardResult(
            source_task_outputs=task_outputs,
            domain_outputs=domain_outputs,
            source_latent=source_latent,
            target_latent=target_latent,
        )

    def checkpoint_payload(
        self,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": DANN_CHECKPOINT_SCHEMA_VERSION,
            "task_model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.task_model.state_dict().items()
            },
            "domain_discriminator_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.domain_discriminator.state_dict().items()
            },
            "dann": {
                "latent_dim": self.latent_dim,
                "n_domains": self.n_domains,
                "gradient_reversal_alpha": self.gradient_reversal_alpha,
                "domain_hidden_dims": list(
                    self.domain_discriminator.hidden_dims
                ),
                "domain_dropout": self.domain_discriminator.dropout,
            },
            "metadata": dict(metadata or {}),
        }

    def save(
        self,
        path: str | Path,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(metadata=metadata), output_path)
        return output_path

    def load(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        payload = torch.load(
            Path(path), map_location=map_location, weights_only=False
        )
        if payload.get("schema_version") != DANN_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported DANN checkpoint schema_version")
        configuration = payload.get("dann", {})
        expected = (self.latent_dim, self.n_domains)
        actual = (
            int(configuration.get("latent_dim", -1)),
            int(configuration.get("n_domains", -1)),
        )
        if actual != expected:
            raise ValueError(
                "DANN checkpoint dimensions do not match current module: "
                f"{actual} != {expected}"
            )
        self.task_model.load_state_dict(
            payload["task_model_state_dict"], strict=True
        )
        self.domain_discriminator.load_state_dict(
            payload["domain_discriminator_state_dict"], strict=True
        )
        return dict(payload.get("metadata", {}))


def _length(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.shape[0])
    return int(np.asarray(value).shape[0])


def _finite_features(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(value.is_floating_point() and torch.isfinite(value).all())
    array = np.asarray(value)
    return bool(np.issubdtype(array.dtype, np.number) and np.isfinite(array).all())


def _row_tensor(
    value: Any,
    index: int,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if isinstance(value, Tensor):
        result = value[index].detach().clone()
        return result if dtype is None else result.to(dtype=dtype)
    return torch.as_tensor(np.asarray(value)[index], dtype=dtype)


def _scalar_int(value: Any, index: int) -> int:
    if isinstance(value, Tensor):
        return int(value[index].detach().cpu().item())
    return int(np.asarray(value)[index])


@dataclass(frozen=True)
class DANNPartition:
    """One explicitly scoped fold partition with provenance identifiers."""

    name: str
    features: Any
    domain_ids: Any
    sample_ids: Sequence[str]
    record_group_ids: Sequence[str]
    subject_ids: Sequence[str]
    task_labels: Optional[Any] = None

    def __post_init__(self) -> None:
        count = _length(self.features)
        if count <= 0:
            raise ValueError(f"{self.name}: partition cannot be empty")
        feature_shape = tuple(np.shape(self.features))
        if len(feature_shape) < 2:
            raise ValueError(
                f"{self.name}: features must include batch and input dimensions"
            )
        if not _finite_features(self.features):
            raise ValueError(
                f"{self.name}: features must be finite floating-point values"
            )
        metadata = {
            "domain_ids": self.domain_ids,
            "sample_ids": self.sample_ids,
            "record_group_ids": self.record_group_ids,
            "subject_ids": self.subject_ids,
        }
        for field_name, values in metadata.items():
            if _length(values) != count:
                raise ValueError(
                    f"{self.name}: {field_name} length does not match features"
                )
        domain_array = (
            self.domain_ids.detach().cpu().numpy()
            if isinstance(self.domain_ids, Tensor)
            else np.asarray(self.domain_ids)
        )
        if domain_array.ndim != 1 or not np.issubdtype(
            domain_array.dtype, np.integer
        ):
            raise ValueError(f"{self.name}: domain_ids must be a 1D integer array")
        if int(domain_array.min()) < 0:
            raise ValueError(f"{self.name}: domain_ids must be non-negative")
        sample_ids = tuple(str(value) for value in self.sample_ids)
        record_group_ids = tuple(str(value) for value in self.record_group_ids)
        subject_ids = tuple(str(value) for value in self.subject_ids)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"{self.name}: sample_ids must be unique")
        if any(not value for value in (*sample_ids, *record_group_ids, *subject_ids)):
            raise ValueError(f"{self.name}: provenance identifiers cannot be empty")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "record_group_ids", record_group_ids)
        object.__setattr__(self, "subject_ids", subject_ids)
        if self.task_labels is not None and _length(self.task_labels) != count:
            raise ValueError(
                f"{self.name}: task_labels length does not match features"
            )

    def __len__(self) -> int:
        return _length(self.features)


@dataclass(frozen=True)
class DANNTrainingBatch:
    """A DANN batch that deliberately has no target task-label field."""

    source_inputs: Tensor
    source_task_labels: Tensor
    target_inputs: Tensor
    source_domain_ids: Tensor
    target_domain_ids: Tensor
    source_sample_ids: tuple[str, ...]
    target_sample_ids: tuple[str, ...]
    source_subject_ids: tuple[str, ...]
    target_subject_ids: tuple[str, ...]

    @property
    def domain_ids(self) -> Tensor:
        return torch.cat(
            (self.source_domain_ids, self.target_domain_ids), dim=0
        )

    def to(self, device: str | torch.device) -> "DANNTrainingBatch":
        return DANNTrainingBatch(
            source_inputs=self.source_inputs.to(device),
            source_task_labels=self.source_task_labels.to(device),
            target_inputs=self.target_inputs.to(device),
            source_domain_ids=self.source_domain_ids.to(device),
            target_domain_ids=self.target_domain_ids.to(device),
            source_sample_ids=self.source_sample_ids,
            target_sample_ids=self.target_sample_ids,
            source_subject_ids=self.source_subject_ids,
            target_subject_ids=self.target_subject_ids,
        )


class _DANNFoldTrainingDataset(Dataset[dict[str, Any]]):
    def __init__(self, source: DANNPartition, target: DANNPartition) -> None:
        if source.task_labels is None:
            raise ValueError("source_train requires authorized task_labels")
        if tuple(np.shape(source.features)[1:]) != tuple(
            np.shape(target.features)[1:]
        ):
            raise ValueError(
                "source_train and target training inputs must have equal shapes"
            )
        self.source = source
        self.target = target

    def __len__(self) -> int:
        return max(len(self.source), len(self.target))

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = index % len(self.source)
        target_index = index % len(self.target)
        return {
            "source_inputs": _row_tensor(
                self.source.features, source_index, dtype=torch.float32
            ),
            "source_task_labels": _row_tensor(
                self.source.task_labels, source_index
            ),
            "target_inputs": _row_tensor(
                self.target.features, target_index, dtype=torch.float32
            ),
            "source_domain_id": _scalar_int(
                self.source.domain_ids, source_index
            ),
            "target_domain_id": _scalar_int(
                self.target.domain_ids, target_index
            ),
            "source_sample_id": self.source.sample_ids[source_index],
            "target_sample_id": self.target.sample_ids[target_index],
            "source_subject_id": self.source.subject_ids[source_index],
            "target_subject_id": self.target.subject_ids[target_index],
        }


def _collate_dann_examples(
    examples: Sequence[Mapping[str, Any]],
) -> DANNTrainingBatch:
    return DANNTrainingBatch(
        source_inputs=torch.stack(
            [item["source_inputs"] for item in examples]
        ),
        source_task_labels=torch.stack(
            [item["source_task_labels"] for item in examples]
        ),
        target_inputs=torch.stack(
            [item["target_inputs"] for item in examples]
        ),
        source_domain_ids=torch.tensor(
            [item["source_domain_id"] for item in examples],
            dtype=torch.long,
        ),
        target_domain_ids=torch.tensor(
            [item["target_domain_id"] for item in examples],
            dtype=torch.long,
        ),
        source_sample_ids=tuple(
            str(item["source_sample_id"]) for item in examples
        ),
        target_sample_ids=tuple(
            str(item["target_sample_id"]) for item in examples
        ),
        source_subject_ids=tuple(
            str(item["source_subject_id"]) for item in examples
        ),
        target_subject_ids=tuple(
            str(item["target_subject_id"]) for item in examples
        ),
    )


@dataclass(frozen=True)
class DANNFoldData:
    """Leakage-checked fold partitions used to create DANN training loaders."""

    source_train: DANNPartition
    target_unlabelled_or_calibration: DANNPartition
    inner_validation: DANNPartition
    outer_test: DANNPartition

    def __post_init__(self) -> None:
        if self.source_train.task_labels is None:
            raise ValueError("source_train requires authorized task_labels")
        source_samples = set(self.source_train.sample_ids)
        source_records = set(self.source_train.record_group_ids)
        outer_samples = set(self.outer_test.sample_ids)
        outer_records = set(self.outer_test.record_group_ids)
        sample_overlap = source_samples & outer_samples
        record_overlap = source_records & outer_records
        if sample_overlap or record_overlap:
            raise ValueError(
                "source_train and outer_test provenance overlap: "
                f"sample_ids={sorted(sample_overlap)}, "
                f"record_group_ids={sorted(record_overlap)}"
            )
        target_samples = set(
            self.target_unlabelled_or_calibration.sample_ids
        )
        if target_samples & outer_samples:
            raise ValueError(
                "target training/calibration and outer_test sample_ids overlap"
            )
        validation_samples = set(self.inner_validation.sample_ids)
        if validation_samples & (
            source_samples | target_samples | outer_samples
        ):
            raise ValueError(
                "inner_validation sample_ids must be disjoint from train and test"
            )

    def training_loader(
        self,
        *,
        batch_size: int,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> DataLoader[DANNTrainingBatch]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        dataset = _DANNFoldTrainingDataset(
            self.source_train, self.target_unlabelled_or_calibration
        )
        generator = torch.Generator()
        generator.manual_seed(int(random_state))
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            generator=generator,
            collate_fn=_collate_dann_examples,
        )


__all__ = [
    "DANN_CHECKPOINT_SCHEMA_VERSION",
    "DANNFoldData",
    "DANNForwardResult",
    "DANNLossResult",
    "DANNModule",
    "DANNObjective",
    "DANNPartition",
    "DANNTrainingBatch",
    "DomainDiscriminator",
    "GradientReversal",
    "aggregate_dann_loss_results",
]
