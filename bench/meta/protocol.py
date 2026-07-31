"""Contracts for future meta-learners; no learning algorithm lives here."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable

import torch
from torch import nn


@dataclass(frozen=True)
class ModelCloneValidation:
    valid: bool
    architecture_matches: bool
    state_matches: bool
    independent_storage: bool
    output_shape_matches: bool
    device: str


def validate_model_clone(
    original: nn.Module,
    clone: nn.Module,
    *,
    example_input: torch.Tensor | None = None,
) -> ModelCloneValidation:
    """Verify state equality, independent storage, and optional head shape."""
    original_state = original.state_dict()
    clone_state = clone.state_dict()
    architecture_matches = type(original) is type(clone)
    state_matches = (
        original_state.keys() == clone_state.keys()
        and all(
            torch.equal(original_state[name].detach().cpu(), clone_state[name].detach().cpu())
            for name in original_state
        )
    )
    original_parameters = dict(original.named_parameters())
    clone_parameters = dict(clone.named_parameters())
    independent_storage = original_parameters.keys() == clone_parameters.keys() and all(
        original_parameters[name].data_ptr() != clone_parameters[name].data_ptr()
        for name in original_parameters
    )
    output_shape_matches = True
    if example_input is not None:
        original_device = next(original.parameters()).device
        clone_device = next(clone.parameters()).device
        original_mode, clone_mode = original.training, clone.training
        original.eval()
        clone.eval()
        with torch.no_grad():
            original_output = original(example_input.to(original_device))
            clone_output = clone(example_input.to(clone_device))
        output_shape_matches = tuple(original_output.shape) == tuple(clone_output.shape)
        original.train(original_mode)
        clone.train(clone_mode)
    device = str(next(clone.parameters()).device)
    valid = (
        architecture_matches
        and state_matches
        and independent_storage
        and output_shape_matches
    )
    return ModelCloneValidation(
        valid=valid,
        architecture_matches=architecture_matches,
        state_matches=state_matches,
        independent_storage=independent_storage,
        output_shape_matches=output_shape_matches,
        device=device,
    )


def clone_model_for_episode(
    model: nn.Module,
    *,
    device: str | torch.device,
    example_input: torch.Tensor | None = None,
) -> nn.Module:
    """Deep-copy a production module and prove that its parameters are isolated."""
    clone = deepcopy(model).to(torch.device(device))
    result = validate_model_clone(model, clone, example_input=example_input)
    if not result.valid:
        raise RuntimeError(f"Unsafe episode model clone: {result}")
    return clone


@runtime_checkable
class MetaLearnerProtocol(Protocol):
    """Algorithm-neutral interface reserved for a later approved experiment."""

    def meta_train_step(self, episodes: Iterable[Any]) -> dict[str, float]:
        ...

    def adapt(self, model: nn.Module, support_batch: Any) -> nn.Module:
        ...

    def evaluate(self, adapted_model: nn.Module, query_batch: Any) -> dict[str, float]:
        ...
