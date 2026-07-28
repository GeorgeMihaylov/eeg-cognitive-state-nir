"""Leakage-safe contrastive building blocks for shared raw-EEG encoders.

The module deliberately contains no standalone training loop.  Experiment
orchestration supplies one authorized outer-train fold, builds a loader from
its indices, and performs optimizer steps with the components defined here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .encoder import EncoderModelProtocol, require_encoder_model
from .ordinal import LossParts


CONTRASTIVE_CHECKPOINT_SCHEMA_VERSION = 1
ENCODER_CHECKPOINT_SCHEMA_VERSION = 1


def _validate_probability(value: float, name: str = "probability") -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _validate_raw_eeg(inputs: Tensor) -> Tensor:
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        raise ValueError(
            "Raw EEG augmentations expect [batch, 1, channels, time], "
            f"got {tuple(inputs.shape)}"
        )
    if inputs.shape[0] == 0 or inputs.shape[2] == 0 or inputs.shape[3] == 0:
        raise ValueError("Raw EEG inputs cannot contain an empty dimension")
    if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
        raise ValueError("Raw EEG inputs must be finite floating-point values")
    return inputs


def _random_apply_mask(
    batch_size: int,
    probability: float,
    *,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> Tensor:
    if probability == 0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if probability == 1:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    return (
        torch.rand(batch_size, device=device, generator=generator)
        < probability
    )


class _RawEEGTransform(nn.Module):
    transform_name = "transform"

    def __init__(self, *, enabled: bool, probability: float) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.probability = _validate_probability(probability)

    def _base_configuration(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "probability": self.probability,
        }

    def configuration(self) -> dict[str, Any]:
        return self._base_configuration()


class GaussianNoise(_RawEEGTransform):
    transform_name = "gaussian_noise"

    def __init__(
        self,
        *,
        enabled: bool = False,
        probability: float = 0.5,
        std: float = 0.01,
    ) -> None:
        super().__init__(enabled=enabled, probability=probability)
        self.std = float(std)
        if not math.isfinite(self.std) or self.std < 0:
            raise ValueError("Gaussian noise std must be finite and non-negative")

    def forward(
        self,
        inputs: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        result = inputs.clone()
        if not self.enabled or self.probability == 0 or self.std == 0:
            return result
        apply = _random_apply_mask(
            len(result),
            self.probability,
            device=result.device,
            generator=generator,
        ).view(-1, 1, 1, 1)
        noise = torch.randn(
            result.shape,
            dtype=result.dtype,
            device=result.device,
            generator=generator,
        )
        return result + apply.to(result.dtype) * self.std * noise

    def configuration(self) -> dict[str, Any]:
        return {**self._base_configuration(), "std": self.std}


class AmplitudeScaling(_RawEEGTransform):
    transform_name = "amplitude_scaling"

    def __init__(
        self,
        *,
        enabled: bool = False,
        probability: float = 0.5,
        minimum: float = 0.9,
        maximum: float = 1.1,
    ) -> None:
        super().__init__(enabled=enabled, probability=probability)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        if (
            not math.isfinite(self.minimum)
            or not math.isfinite(self.maximum)
            or self.minimum <= 0
            or self.maximum < self.minimum
        ):
            raise ValueError(
                "Amplitude scale bounds must be finite, positive, and ordered"
            )

    def forward(
        self,
        inputs: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        result = inputs.clone()
        if (
            not self.enabled
            or self.probability == 0
            or self.minimum == self.maximum == 1
        ):
            return result
        apply = _random_apply_mask(
            len(result),
            self.probability,
            device=result.device,
            generator=generator,
        )
        scales = self.minimum + (
            self.maximum - self.minimum
        ) * torch.rand(
            len(result),
            dtype=result.dtype,
            device=result.device,
            generator=generator,
        )
        scales = torch.where(apply, scales, torch.ones_like(scales))
        return result * scales.view(-1, 1, 1, 1)

    def configuration(self) -> dict[str, Any]:
        return {
            **self._base_configuration(),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


class TimeMasking(_RawEEGTransform):
    transform_name = "time_masking"

    def __init__(
        self,
        *,
        enabled: bool = False,
        probability: float = 0.5,
        maximum_fraction: float = 0.1,
    ) -> None:
        super().__init__(enabled=enabled, probability=probability)
        self.maximum_fraction = float(maximum_fraction)
        if (
            not math.isfinite(self.maximum_fraction)
            or not 0 <= self.maximum_fraction <= 1
        ):
            raise ValueError("maximum_fraction must be finite and in [0, 1]")

    def forward(
        self,
        inputs: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        result = inputs.clone()
        if (
            not self.enabled
            or self.probability == 0
            or self.maximum_fraction == 0
        ):
            return result
        maximum_width = max(
            1, min(result.shape[-1], round(result.shape[-1] * self.maximum_fraction))
        )
        apply = _random_apply_mask(
            len(result),
            self.probability,
            device=result.device,
            generator=generator,
        )
        for index in torch.nonzero(apply, as_tuple=False).flatten().tolist():
            width = int(
                torch.randint(
                    1,
                    maximum_width + 1,
                    (1,),
                    device=result.device,
                    generator=generator,
                ).item()
            )
            start = int(
                torch.randint(
                    0,
                    result.shape[-1] - width + 1,
                    (1,),
                    device=result.device,
                    generator=generator,
                ).item()
            )
            result[index, :, :, start : start + width] = 0
        return result

    def configuration(self) -> dict[str, Any]:
        return {
            **self._base_configuration(),
            "maximum_fraction": self.maximum_fraction,
        }


class ChannelMasking(_RawEEGTransform):
    transform_name = "channel_masking"

    def __init__(
        self,
        *,
        enabled: bool = False,
        probability: float = 0.3,
        maximum_channels: int = 1,
    ) -> None:
        super().__init__(enabled=enabled, probability=probability)
        self.maximum_channels = int(maximum_channels)
        if self.maximum_channels < 0:
            raise ValueError("maximum_channels must be non-negative")

    def forward(
        self,
        inputs: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        result = inputs.clone()
        if (
            not self.enabled
            or self.probability == 0
            or self.maximum_channels == 0
        ):
            return result
        maximum = min(self.maximum_channels, result.shape[2])
        apply = _random_apply_mask(
            len(result),
            self.probability,
            device=result.device,
            generator=generator,
        )
        for index in torch.nonzero(apply, as_tuple=False).flatten().tolist():
            count = int(
                torch.randint(
                    1,
                    maximum + 1,
                    (1,),
                    device=result.device,
                    generator=generator,
                ).item()
            )
            channels = torch.randperm(
                result.shape[2],
                device=result.device,
                generator=generator,
            )[:count]
            result[index, :, channels, :] = 0
        return result

    def configuration(self) -> dict[str, Any]:
        return {
            **self._base_configuration(),
            "maximum_channels": self.maximum_channels,
        }


class TemporalShift(_RawEEGTransform):
    transform_name = "temporal_shift"

    def __init__(
        self,
        *,
        enabled: bool = False,
        probability: float = 0.5,
        maximum_fraction: float = 0.05,
    ) -> None:
        super().__init__(enabled=enabled, probability=probability)
        self.maximum_fraction = float(maximum_fraction)
        if (
            not math.isfinite(self.maximum_fraction)
            or not 0 <= self.maximum_fraction <= 1
        ):
            raise ValueError("maximum_fraction must be finite and in [0, 1]")

    def forward(
        self,
        inputs: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        result = inputs.clone()
        if (
            not self.enabled
            or self.probability == 0
            or self.maximum_fraction == 0
        ):
            return result
        maximum = max(1, round(result.shape[-1] * self.maximum_fraction))
        apply = _random_apply_mask(
            len(result),
            self.probability,
            device=result.device,
            generator=generator,
        )
        for index in torch.nonzero(apply, as_tuple=False).flatten().tolist():
            shift = int(
                torch.randint(
                    -maximum,
                    maximum + 1,
                    (1,),
                    device=result.device,
                    generator=generator,
                ).item()
            )
            result[index] = torch.roll(result[index], shifts=shift, dims=-1)
        return result

    def configuration(self) -> dict[str, Any]:
        return {
            **self._base_configuration(),
            "maximum_fraction": self.maximum_fraction,
        }


def _build_transform(
    transform_type: type[_RawEEGTransform],
    configuration: Optional[Mapping[str, Any]],
) -> _RawEEGTransform:
    try:
        return transform_type(**dict(configuration or {}))
    except TypeError as exc:
        raise ValueError(
            f"Unsupported {transform_type.transform_name} parameter: {exc}"
        ) from exc


class EEGAugmentationPipeline(nn.Module):
    """Config-driven, label-free transformations for raw EEG tensors."""

    def __init__(
        self,
        *,
        gaussian_noise: Optional[Mapping[str, Any]] = None,
        amplitude_scaling: Optional[Mapping[str, Any]] = None,
        time_masking: Optional[Mapping[str, Any]] = None,
        channel_masking: Optional[Mapping[str, Any]] = None,
        temporal_shift: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(
            [
                _build_transform(GaussianNoise, gaussian_noise),
                _build_transform(AmplitudeScaling, amplitude_scaling),
                _build_transform(TimeMasking, time_masking),
                _build_transform(ChannelMasking, channel_masking),
                _build_transform(TemporalShift, temporal_shift),
            ]
        )

    @classmethod
    def from_config(
        cls, configuration: Optional[Mapping[str, Any]]
    ) -> "EEGAugmentationPipeline":
        return cls(**dict(configuration or {}))

    @staticmethod
    def make_generator(
        seed: int,
        *,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(int(seed))
        return generator

    def configuration(self) -> dict[str, Any]:
        return {
            transform.transform_name: transform.configuration()
            for transform in self.transforms
        }

    def forward(
        self,
        inputs: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        result = _validate_raw_eeg(inputs).clone()
        if generator is not None and torch.device(generator.device) != result.device:
            raise ValueError(
                "Augmentation generator device must match input tensor device"
            )
        for transform in self.transforms:
            result = transform(result, generator=generator)
        if result.shape != inputs.shape or not torch.isfinite(result).all():
            raise RuntimeError(
                "EEG augmentation changed shape or produced non-finite values"
            )
        return result

    def two_views(
        self,
        inputs: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor]:
        return (
            self(inputs, generator=generator),
            self(inputs, generator=generator),
        )


class ProjectionHead(nn.Module):
    """Projection-only MLP attached to shared encoder latent features."""

    def __init__(
        self,
        latent_dim: int,
        projection_dim: int,
        *,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = (
            self.latent_dim if hidden_dim is None else int(hidden_dim)
        )
        if min(self.latent_dim, self.projection_dim, self.hidden_dim) <= 0:
            raise ValueError("Projection dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.projection_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                "ProjectionHead expects [batch, latent_dim] with latent_dim="
                f"{self.latent_dim}, got {tuple(latent.shape)}"
            )
        if not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError("ProjectionHead input must be finite floating-point")
        projected = self.network(latent)
        normalized = F.normalize(projected, p=2, dim=1, eps=1e-12)
        if not torch.isfinite(normalized).all():
            raise ValueError("ProjectionHead produced non-finite embeddings")
        return normalized


@dataclass(frozen=True)
class ContrastiveForwardResult:
    first_projection: Tensor
    second_projection: Tensor
    first_latent: Tensor
    second_latent: Tensor


class ContrastiveModule(nn.Module):
    """Compose an existing raw-EEG encoder with a separate projection head."""

    def __init__(
        self,
        encoder_model: nn.Module,
        *,
        projection_dim: int,
        projection_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        encoder = require_encoder_model(encoder_model)
        self.encoder_model = encoder_model
        self.projection_head = ProjectionHead(
            encoder.latent_dim,
            projection_dim,
            hidden_dim=projection_hidden_dim,
        )

    @property
    def encoder(self) -> EncoderModelProtocol:
        return require_encoder_model(self.encoder_model)

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def projection_dim(self) -> int:
        return self.projection_head.projection_dim

    def forward(
        self, first_view: Tensor, second_view: Tensor
    ) -> ContrastiveForwardResult:
        if first_view.shape != second_view.shape:
            raise ValueError(
                "Contrastive views must have identical shapes: "
                f"{tuple(first_view.shape)} != {tuple(second_view.shape)}"
            )
        first_latent = self.encoder.encode(first_view)
        second_latent = self.encoder.encode(second_view)
        return ContrastiveForwardResult(
            first_projection=self.projection_head(first_latent),
            second_projection=self.projection_head(second_latent),
            first_latent=first_latent,
            second_latent=second_latent,
        )

    def augmented_forward(
        self,
        inputs: Tensor,
        augmentations: EEGAugmentationPipeline,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> ContrastiveForwardResult:
        first, second = augmentations.two_views(
            inputs, generator=generator
        )
        return self(first, second)

    def checkpoint_payload(
        self,
        *,
        optimizer: Optional[torch.optim.Optimizer],
        configuration: Mapping[str, Any],
        augmentation_configuration: Mapping[str, Any],
        seed: int,
        epoch: int,
        training_provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        if int(epoch) < 0:
            raise ValueError("Contrastive checkpoint epoch must be non-negative")
        signature = encoder_architecture_signature(self.encoder_model)
        return {
            "schema_version": CONTRASTIVE_CHECKPOINT_SCHEMA_VERSION,
            "encoder_state_dict": _encoder_state_dict(self.encoder_model),
            "projection_head_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.projection_head.state_dict().items()
            },
            "optimizer_state_dict": (
                None if optimizer is None else optimizer.state_dict()
            ),
            "encoder_architecture": signature,
            "configuration": dict(configuration),
            "latent_dim": self.latent_dim,
            "projection_dim": self.projection_dim,
            "projection_hidden_dim": self.projection_head.hidden_dim,
            "augmentation_configuration": dict(augmentation_configuration),
            "seed": int(seed),
            "epoch": int(epoch),
            "training_provenance": dict(training_provenance),
        }

    def save(
        self,
        path: str | Path,
        *,
        optimizer: Optional[torch.optim.Optimizer],
        configuration: Mapping[str, Any],
        augmentation_configuration: Mapping[str, Any],
        seed: int,
        epoch: int,
        training_provenance: Mapping[str, Any],
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.checkpoint_payload(
                optimizer=optimizer,
                configuration=configuration,
                augmentation_configuration=augmentation_configuration,
                seed=seed,
                epoch=epoch,
                training_provenance=training_provenance,
            ),
            output_path,
        )
        return output_path

    def load(
        self,
        path: str | Path,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        payload = torch.load(
            Path(path), map_location=map_location, weights_only=False
        )
        if payload.get("schema_version") != CONTRASTIVE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported contrastive checkpoint schema_version")
        if (
            int(payload.get("latent_dim", -1)) != self.latent_dim
            or int(payload.get("projection_dim", -1)) != self.projection_dim
            or int(payload.get("projection_hidden_dim", -1))
            != self.projection_head.hidden_dim
        ):
            raise ValueError(
                "Contrastive checkpoint dimensions do not match current module"
            )
        _load_encoder_payload(
            self.encoder_model,
            payload["encoder_state_dict"],
            payload["encoder_architecture"],
        )
        self.projection_head.load_state_dict(
            payload["projection_head_state_dict"], strict=True
        )
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        return {
            "configuration": dict(payload.get("configuration", {})),
            "augmentation_configuration": dict(
                payload.get("augmentation_configuration", {})
            ),
            "seed": int(payload.get("seed", 0)),
            "epoch": int(payload.get("epoch", 0)),
            "training_provenance": dict(
                payload.get("training_provenance", {})
            ),
            "optimizer_state_available": optimizer_state is not None,
        }


@dataclass(frozen=True)
class ContrastiveLossResult:
    loss: LossParts
    positive_similarity: Tensor
    negative_similarity: Tensor
    embedding_norm: Tensor

    @property
    def contrastive_loss(self) -> Tensor:
        return self.loss.mean

    def detached_metrics(self) -> dict[str, float]:
        return {
            "contrastive_loss": float(
                self.contrastive_loss.detach().cpu().item()
            ),
            "positive_similarity": float(
                self.positive_similarity.detach().cpu().item()
            ),
            "negative_similarity": float(
                self.negative_similarity.detach().cpu().item()
            ),
            "embedding_norm": float(
                self.embedding_norm.detach().cpu().item()
            ),
        }


def nt_xent_logits(
    first: Tensor,
    second: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return masked NT-Xent logits, positive indices, and normalized views."""

    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Contrastive temperature must be finite and positive")
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(
            "Contrastive projections must have equal [batch, projection_dim] shapes"
        )
    if first.shape[0] < 2:
        raise ValueError("Contrastive batches require at least two windows")
    if (
        not first.is_floating_point()
        or not second.is_floating_point()
        or not torch.isfinite(first).all()
        or not torch.isfinite(second).all()
    ):
        raise ValueError("Contrastive projections must be finite floating-point")
    normalized = torch.cat(
        (
            F.normalize(first, p=2, dim=1, eps=1e-12),
            F.normalize(second, p=2, dim=1, eps=1e-12),
        ),
        dim=0,
    )
    similarities = normalized @ normalized.T
    count = similarities.shape[0]
    self_mask = torch.eye(count, dtype=torch.bool, device=similarities.device)
    logits = (similarities / value).masked_fill(
        self_mask, torch.finfo(similarities.dtype).min
    )
    batch_size = first.shape[0]
    positives = (
        torch.arange(count, device=similarities.device) + batch_size
    ) % count
    return logits, positives, normalized


class ContrastiveObjective:
    """Stable in-batch NT-Xent objective with diagnostic components."""

    def __init__(self, *, temperature: float = 0.1) -> None:
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Contrastive temperature must be finite and positive")

    def __call__(
        self, outputs: ContrastiveForwardResult
    ) -> ContrastiveLossResult:
        first = outputs.first_projection
        second = outputs.second_projection
        logits, positives, normalized = nt_xent_logits(
            first, second, temperature=self.temperature
        )
        numerator = F.cross_entropy(logits, positives, reduction="sum")
        denominator = logits.new_tensor(logits.shape[0])
        similarities = normalized @ normalized.T
        row_indices = torch.arange(
            similarities.shape[0], device=similarities.device
        )
        positive_similarity = similarities[row_indices, positives].mean()
        self_mask = torch.eye(
            similarities.shape[0],
            dtype=torch.bool,
            device=similarities.device,
        )
        positive_mask = torch.zeros_like(self_mask)
        positive_mask[row_indices, positives] = True
        negatives = similarities[~self_mask & ~positive_mask]
        if negatives.numel() == 0:
            raise ValueError("Contrastive batch does not contain negative pairs")
        result = ContrastiveLossResult(
            loss=LossParts(numerator=numerator, denominator=denominator),
            positive_similarity=positive_similarity,
            negative_similarity=negatives.mean(),
            embedding_norm=torch.cat((first, second), dim=0).norm(
                p=2, dim=1
            ).mean(),
        )
        if not all(
            math.isfinite(value)
            for value in result.detached_metrics().values()
        ):
            raise ValueError("Contrastive objective produced non-finite values")
        return result


def aggregate_contrastive_loss_results(
    results: Iterable[ContrastiveLossResult],
) -> dict[str, float]:
    items = list(results)
    if not items:
        raise ValueError("At least one ContrastiveLossResult is required")
    numerator = sum(float(item.loss.numerator.detach().item()) for item in items)
    denominator = sum(
        float(item.loss.denominator.detach().item()) for item in items
    )
    if denominator <= 0:
        raise ValueError("Contrastive aggregate denominator must be positive")
    weights = [
        float(item.loss.denominator.detach().item()) for item in items
    ]
    total_weight = sum(weights)
    return {
        "contrastive_loss": numerator / denominator,
        "positive_similarity": sum(
            weight * float(item.positive_similarity.detach().item())
            for weight, item in zip(weights, items)
        )
        / total_weight,
        "negative_similarity": sum(
            weight * float(item.negative_similarity.detach().item())
            for weight, item in zip(weights, items)
        )
        / total_weight,
        "embedding_norm": sum(
            weight * float(item.embedding_norm.detach().item())
            for weight, item in zip(weights, items)
        )
        / total_weight,
    }


def _head_prefixes(model: nn.Module) -> tuple[str, ...]:
    encoder = require_encoder_model(model)
    method = getattr(encoder, "output_head_parameter_prefixes", None)
    if not callable(method):
        raise TypeError(
            f"{model.__class__.__name__} does not expose output head prefixes"
        )
    prefixes = tuple(str(prefix) for prefix in method())
    if not prefixes:
        raise ValueError("Encoder output head prefixes cannot be empty")
    return prefixes


def _encoder_state_dict(model: nn.Module) -> dict[str, Tensor]:
    prefixes = _head_prefixes(model)
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not any(key.startswith(prefix) for prefix in prefixes)
    }


def encoder_architecture_signature(model: nn.Module) -> dict[str, Any]:
    encoder = require_encoder_model(model)
    state = _encoder_state_dict(model)
    input_metadata = {
        name: int(getattr(model, name))
        for name in ("n_channels", "n_times")
        if hasattr(model, name)
    }
    return {
        "model_class": (
            f"{model.__class__.__module__}.{model.__class__.__qualname__}"
        ),
        "latent_dim": encoder.latent_dim,
        "input_metadata": input_metadata,
        "state_shapes": {
            key: list(value.shape) for key, value in sorted(state.items())
        },
    }


def _load_encoder_payload(
    model: nn.Module,
    encoder_state: Mapping[str, Tensor],
    saved_signature: Mapping[str, Any],
) -> None:
    current_signature = encoder_architecture_signature(model)
    if dict(saved_signature) != current_signature:
        raise ValueError(
            "Encoder checkpoint architecture is incompatible with current model"
        )
    current_encoder_keys = set(_encoder_state_dict(model))
    saved_keys = set(encoder_state)
    if saved_keys != current_encoder_keys:
        raise ValueError(
            "Encoder checkpoint keys are incompatible with current model: "
            f"missing={sorted(current_encoder_keys - saved_keys)}, "
            f"unexpected={sorted(saved_keys - current_encoder_keys)}"
        )
    merged = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    for key, value in encoder_state.items():
        if merged[key].shape != value.shape:
            raise ValueError(
                f"Encoder checkpoint tensor shape mismatch for {key}: "
                f"{tuple(value.shape)} != {tuple(merged[key].shape)}"
            )
        merged[key] = value.to(
            device=merged[key].device, dtype=merged[key].dtype
        )
    model.load_state_dict(merged, strict=True)


def export_encoder_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ENCODER_CHECKPOINT_SCHEMA_VERSION,
        "encoder_state_dict": _encoder_state_dict(model),
        "encoder_architecture": encoder_architecture_signature(model),
        "latent_dim": require_encoder_model(model).latent_dim,
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, output_path)
    return output_path


def load_encoder_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    if payload.get("schema_version") != ENCODER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported encoder checkpoint schema_version")
    if int(payload.get("latent_dim", -1)) != require_encoder_model(model).latent_dim:
        raise ValueError("Encoder checkpoint latent_dim is incompatible")
    _load_encoder_payload(
        model,
        payload["encoder_state_dict"],
        payload["encoder_architecture"],
    )
    return dict(payload.get("metadata", {}))


def _stable_hash(values: Sequence[Any]) -> str:
    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_indices(
    values: Sequence[int],
    *,
    name: str,
    size: int,
    allow_empty: bool,
) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    indices = tuple(int(value) for value in array.tolist())
    if not indices and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate indices")
    if indices and (min(indices) < 0 or max(indices) >= size):
        raise ValueError(f"{name} contains an out-of-range index")
    return indices


@dataclass(frozen=True)
class ContrastiveTrainingBatch:
    inputs: Tensor
    source_indices: Tensor
    sample_ids: tuple[str, ...]
    record_group_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]

    def to(self, device: str | torch.device) -> "ContrastiveTrainingBatch":
        return ContrastiveTrainingBatch(
            inputs=self.inputs.to(device),
            source_indices=self.source_indices.to(device),
            sample_ids=self.sample_ids,
            record_group_ids=self.record_group_ids,
            subject_ids=self.subject_ids,
        )


class _AuthorizedContrastiveDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        features: Any,
        indices: tuple[int, ...],
        sample_ids: tuple[str, ...],
        record_group_ids: tuple[str, ...],
        subject_ids: tuple[str, ...],
    ) -> None:
        self.features = features
        self.indices = indices
        self.sample_ids = sample_ids
        self.record_group_ids = record_group_ids
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> dict[str, Any]:
        source_index = self.indices[local_index]
        window = np.ascontiguousarray(
            self.features[source_index], dtype=np.float32
        )
        if window.ndim != 3 or window.shape[0] != 1:
            raise ValueError(
                "Contrastive raw EEG window must have shape "
                f"[1, channels, time], got {window.shape}"
            )
        if not np.isfinite(window).all():
            raise ValueError("Contrastive raw EEG window contains NaN or Inf")
        return {
            "inputs": torch.from_numpy(window),
            "source_index": source_index,
            "sample_id": self.sample_ids[source_index],
            "record_group_id": self.record_group_ids[source_index],
            "subject_id": self.subject_ids[source_index],
        }


def _collate_contrastive(
    examples: Sequence[Mapping[str, Any]],
) -> ContrastiveTrainingBatch:
    return ContrastiveTrainingBatch(
        inputs=torch.stack([item["inputs"] for item in examples]),
        source_indices=torch.tensor(
            [item["source_index"] for item in examples], dtype=torch.long
        ),
        sample_ids=tuple(str(item["sample_id"]) for item in examples),
        record_group_ids=tuple(
            str(item["record_group_id"]) for item in examples
        ),
        subject_ids=tuple(str(item["subject_id"]) for item in examples),
    )


@dataclass(frozen=True)
class ContrastiveFoldData:
    """An indexed outer-train view with explicit forbidden partitions."""

    features: Any
    sample_ids: tuple[str, ...]
    record_group_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    training_indices: tuple[int, ...]
    inner_validation_indices: tuple[int, ...]
    outer_test_indices: tuple[int, ...]
    target_final_evaluation_indices: tuple[int, ...]
    fold_id: str

    @classmethod
    def from_indexed_source(
        cls,
        *,
        features: Any,
        sample_ids: Sequence[str],
        record_group_ids: Sequence[str],
        subject_ids: Sequence[str],
        training_indices: Sequence[int],
        outer_test_indices: Sequence[int],
        inner_validation_indices: Sequence[int] = (),
        target_final_evaluation_indices: Sequence[int] = (),
        fold_id: str | int,
    ) -> "ContrastiveFoldData":
        size = len(features)
        identifiers = {
            "sample_ids": tuple(str(value) for value in sample_ids),
            "record_group_ids": tuple(
                str(value) for value in record_group_ids
            ),
            "subject_ids": tuple(str(value) for value in subject_ids),
        }
        if any(len(values) != size for values in identifiers.values()):
            raise ValueError("Contrastive provenance lengths must match features")
        if any(not value for values in identifiers.values() for value in values):
            raise ValueError("Contrastive provenance identifiers cannot be empty")
        if len(set(identifiers["sample_ids"])) != size:
            raise ValueError("Contrastive sample_ids must be globally unique")
        shape = tuple(int(value) for value in getattr(features, "shape", ()))
        if len(shape) != 4 or shape[0] != size or shape[1] != 1:
            raise ValueError(
                "Contrastive features must expose [samples, 1, channels, time], "
                f"got {shape}"
            )
        partitions = {
            "training_indices": _validated_indices(
                training_indices,
                name="training_indices",
                size=size,
                allow_empty=False,
            ),
            "inner_validation_indices": _validated_indices(
                inner_validation_indices,
                name="inner_validation_indices",
                size=size,
                allow_empty=True,
            ),
            "outer_test_indices": _validated_indices(
                outer_test_indices,
                name="outer_test_indices",
                size=size,
                allow_empty=False,
            ),
            "target_final_evaluation_indices": _validated_indices(
                target_final_evaluation_indices,
                name="target_final_evaluation_indices",
                size=size,
                allow_empty=True,
            ),
        }
        training = set(partitions["training_indices"])
        forbidden = set().union(
            partitions["inner_validation_indices"],
            partitions["outer_test_indices"],
            partitions["target_final_evaluation_indices"],
        )
        overlap = training & forbidden
        if overlap:
            raise ValueError(
                "Authorized contrastive training indices overlap forbidden "
                f"validation/test indices: {sorted(overlap)}"
            )
        train_records = {
            identifiers["record_group_ids"][index] for index in training
        }
        forbidden_records = {
            identifiers["record_group_ids"][index] for index in forbidden
        }
        record_overlap = train_records & forbidden_records
        if record_overlap:
            raise ValueError(
                "Contrastive train/test record_group_id overlap: "
                f"{sorted(record_overlap)}"
            )
        train_subjects = {
            identifiers["subject_ids"][index] for index in training
        }
        forbidden_subjects = {
            identifiers["subject_ids"][index] for index in forbidden
        }
        subject_overlap = train_subjects & forbidden_subjects
        if subject_overlap:
            raise ValueError(
                "Contrastive train/test subject_id overlap: "
                f"{sorted(subject_overlap)}"
            )
        return cls(
            features=features,
            sample_ids=identifiers["sample_ids"],
            record_group_ids=identifiers["record_group_ids"],
            subject_ids=identifiers["subject_ids"],
            training_indices=partitions["training_indices"],
            inner_validation_indices=partitions["inner_validation_indices"],
            outer_test_indices=partitions["outer_test_indices"],
            target_final_evaluation_indices=partitions[
                "target_final_evaluation_indices"
            ],
            fold_id=str(fold_id),
        )

    def training_loader(
        self,
        *,
        batch_size: int,
        shuffle: bool = True,
        random_state: int = 42,
        drop_last: bool = False,
    ) -> DataLoader[ContrastiveTrainingBatch]:
        if int(batch_size) < 2:
            raise ValueError("Contrastive batch_size must be at least 2")
        dataset = _AuthorizedContrastiveDataset(
            self.features,
            self.training_indices,
            self.sample_ids,
            self.record_group_ids,
            self.subject_ids,
        )
        generator = torch.Generator()
        generator.manual_seed(int(random_state))
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
            generator=generator,
            collate_fn=_collate_contrastive,
        )

    def training_provenance(self) -> dict[str, Any]:
        indices = self.training_indices
        samples = [self.sample_ids[index] for index in indices]
        records = [self.record_group_ids[index] for index in indices]
        subjects = [self.subject_ids[index] for index in indices]
        return {
            "fold_id": self.fold_id,
            "training_sample_count": len(indices),
            "training_indices_sha256": _stable_hash(indices),
            "training_sample_ids_sha256": _stable_hash(samples),
            "training_record_group_ids": sorted(set(records)),
            "training_subject_ids": sorted(set(subjects)),
            "inner_validation_count": len(self.inner_validation_indices),
            "outer_test_count": len(self.outer_test_indices),
            "target_final_evaluation_count": len(
                self.target_final_evaluation_indices
            ),
        }


__all__ = [
    "AmplitudeScaling",
    "CONTRASTIVE_CHECKPOINT_SCHEMA_VERSION",
    "ChannelMasking",
    "ContrastiveFoldData",
    "ContrastiveForwardResult",
    "ContrastiveLossResult",
    "ContrastiveModule",
    "ContrastiveObjective",
    "ContrastiveTrainingBatch",
    "EEGAugmentationPipeline",
    "ENCODER_CHECKPOINT_SCHEMA_VERSION",
    "GaussianNoise",
    "ProjectionHead",
    "TemporalShift",
    "TimeMasking",
    "aggregate_contrastive_loss_results",
    "encoder_architecture_signature",
    "export_encoder_checkpoint",
    "load_encoder_checkpoint",
    "nt_xent_logits",
]
