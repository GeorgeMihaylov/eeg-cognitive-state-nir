"""Shared latent-feature contract for compatible PyTorch EEG models."""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


ENCODER_API_VERSION = 1


@runtime_checkable
class EncoderModelProtocol(Protocol):
    """Minimal model surface needed by future transfer objectives."""

    @property
    def latent_dim(self) -> int:
        """Width of the two-dimensional representation returned by ``encode``."""

    def encode(self, inputs: Tensor) -> Tensor:
        """Return latent features with shape ``[batch, latent_dim]``."""

    def forward_head(self, features: Tensor) -> Tensor:
        """Map latent features to the current task outputs."""

    def get_output_head(self) -> nn.Module:
        """Return the current output head."""

    def replace_output_head(self, num_outputs: int) -> nn.Module:
        """Install and return a new linear output head."""

    def freeze_encoder(self) -> None:
        """Freeze every parameter except the output head."""

    def unfreeze_encoder(self) -> None:
        """Make encoder and output-head parameters trainable."""


class SharedEncoderMixin:
    """Reusable head and freezing behavior for an ``nn.Module`` encoder.

    Subclasses implement ``encode`` and keep their output module under
    ``output_head_attribute``.  The mixin registers no parameters or buffers,
    so adding it does not change legacy state-dict keys.
    """

    output_head_attribute = "classifier"
    _latent_dim: int

    @property
    def latent_dim(self) -> int:
        value = int(self._latent_dim)
        if value <= 0:
            raise RuntimeError("latent_dim must be positive")
        return value

    def _module(self) -> nn.Module:
        if not isinstance(self, nn.Module):
            raise TypeError("SharedEncoderMixin requires an nn.Module subclass")
        return self

    def get_output_head(self) -> nn.Module:
        head = getattr(self, self.output_head_attribute, None)
        if not isinstance(head, nn.Module):
            raise RuntimeError(
                f"Output head {self.output_head_attribute!r} is not an nn.Module"
            )
        return head

    def forward_head(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.latent_dim:
            raise ValueError(
                "Encoder head expects [batch, latent_dim] with latent_dim="
                f"{self.latent_dim}, got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError("Latent features contain NaN or infinite values")
        return self.get_output_head()(features)

    def replace_output_head(self, num_outputs: int) -> nn.Module:
        output_width = int(num_outputs)
        if output_width <= 0:
            raise ValueError("num_outputs must be positive")
        old_head = self.get_output_head()
        reference = next(old_head.parameters(), None)
        new_head = nn.Linear(self.latent_dim, output_width)
        if reference is not None:
            new_head = new_head.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        new_head.train(old_head.training)
        setattr(self, self.output_head_attribute, new_head)
        if hasattr(self, "num_classes"):
            self.num_classes = output_width
        return new_head

    def output_head_parameter_prefixes(self) -> tuple[str, ...]:
        """Return stable prefixes used by the existing fine-tuning adapter."""
        return (f"{self.output_head_attribute}.",)

    def named_encoder_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        prefixes = self.output_head_parameter_prefixes()
        for name, parameter in self._module().named_parameters():
            if not any(name.startswith(prefix) for prefix in prefixes):
                yield name, parameter

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_encoder_parameters():
            yield parameter

    def freeze_encoder(self) -> None:
        for parameter in self.encoder_parameters():
            parameter.requires_grad = False
        for parameter in self.get_output_head().parameters():
            parameter.requires_grad = True

    def unfreeze_encoder(self) -> None:
        for parameter in self._module().parameters():
            parameter.requires_grad = True


def require_encoder_model(model: nn.Module) -> EncoderModelProtocol:
    """Validate and return a model implementing the shared encoder contract."""
    required = (
        "latent_dim",
        "encode",
        "forward_head",
        "get_output_head",
        "replace_output_head",
        "freeze_encoder",
        "unfreeze_encoder",
    )
    missing = [
        name
        for name in required
        if not hasattr(model, name)
        or (name != "latent_dim" and not callable(getattr(model, name)))
    ]
    if missing:
        raise TypeError(
            f"{model.__class__.__name__} does not implement the shared encoder "
            f"contract; missing: {missing}"
        )
    latent_dim = int(getattr(model, "latent_dim"))
    if latent_dim <= 0:
        raise ValueError("Encoder model latent_dim must be positive")
    return model  # type: ignore[return-value]
