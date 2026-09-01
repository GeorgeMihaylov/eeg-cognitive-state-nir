"""Episode-local parameter and BatchNorm buffer policies for functional calls."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor, nn


class BufferPolicy(str, Enum):
    FROZEN_GLOBAL = "frozen_global"
    SUPPORT_LOCAL = "support_local"

    @classmethod
    def normalize(cls, value: str) -> "BufferPolicy":
        if value == "frozen":
            return cls.FROZEN_GLOBAL
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                "buffer_policy must be frozen_global or support_local"
            ) from exc


class FunctionalStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class BufferAuditResult:
    policy: str
    phase: str
    module_training: bool
    batchnorm_training: bool
    dropout_active: bool
    running_statistics_may_update: bool
    buffers_changed: bool


@dataclass(frozen=True)
class FunctionalModelState:
    parameters: OrderedDict[str, Tensor]
    buffers: OrderedDict[str, Tensor]
    training_mode: bool
    architecture_signature: str
    buffer_policy: BufferPolicy

    def with_parameters(
        self, parameters: Mapping[str, Tensor]
    ) -> "FunctionalModelState":
        return replace(self, parameters=OrderedDict(parameters))


def tensor_mapping_hash(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in values.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def stable_model_class_path(model: nn.Module) -> str:
    """Return the pre-relocation logical identity used by protocol hashes.

    Historical protocol manifests hashed classes below ``model_zoo``.  The
    implementation now lives below ``cogstate.model_zoo``; normalizing only
    this approved relocation preserves architecture identity without exposing
    an import shim at the old package path.
    """
    path = f"{model.__class__.__module__}.{model.__class__.__name__}"
    prefix = "cogstate.model_zoo."
    if path.startswith(prefix):
        return path.removeprefix("cogstate.")
    return path


def architecture_schema_signature(model: nn.Module) -> str:
    payload = {
        "class": stable_model_class_path(model),
        "parameters": [
            [name, list(value.shape), str(value.dtype)]
            for name, value in model.named_parameters()
        ],
        "buffers": [
            [name, list(value.shape), str(value.dtype)]
            for name, value in model.named_buffers()
        ],
        "latent_dim": int(getattr(model, "latent_dim", 0)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_functional_state(
    model: nn.Module, state: FunctionalModelState
) -> None:
    expected_parameters = OrderedDict(model.named_parameters())
    expected_buffers = OrderedDict(model.named_buffers())
    for label, expected, actual in (
        ("parameter", expected_parameters, state.parameters),
        ("buffer", expected_buffers, state.buffers),
    ):
        if tuple(expected) != tuple(actual):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise FunctionalStateError(
                f"{label} key mismatch; missing={missing}, extra={extra}"
            )
        for name, reference in expected.items():
            value = actual[name]
            if value.shape != reference.shape:
                raise FunctionalStateError(
                    f"{label} shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(reference.shape)}"
                )
            if value.dtype != reference.dtype:
                raise FunctionalStateError(f"{label} dtype mismatch for {name}")
            if value.device != reference.device:
                raise FunctionalStateError(f"{label} device mismatch for {name}")
    if state.architecture_signature != architecture_schema_signature(model):
        raise FunctionalStateError("architecture signature mismatch")


def create_functional_state(
    model: nn.Module, buffer_policy: str | BufferPolicy
) -> FunctionalModelState:
    policy = (
        buffer_policy
        if isinstance(buffer_policy, BufferPolicy)
        else BufferPolicy.normalize(buffer_policy)
    )
    parameters = OrderedDict(
        (name, value.detach().clone().requires_grad_(value.requires_grad))
        for name, value in model.named_parameters()
    )
    buffers = OrderedDict(
        (name, value.detach().clone()) for name, value in model.named_buffers()
    )
    state = FunctionalModelState(
        parameters=parameters,
        buffers=buffers,
        training_mode=bool(model.training),
        architecture_signature=architecture_schema_signature(model),
        buffer_policy=policy,
    )
    validate_functional_state(model, state)
    for name, value in model.named_parameters():
        if state.parameters[name].data_ptr() == value.data_ptr():
            raise FunctionalStateError(f"parameter storage is shared for {name}")
    for name, value in model.named_buffers():
        if state.buffers[name].data_ptr() == value.data_ptr():
            raise FunctionalStateError(f"buffer storage is shared for {name}")
    return state


@contextmanager
def _controlled_module_modes(
    model: nn.Module, *, batchnorm_training: bool
) -> Iterator[None]:
    modes = {module: bool(module.training) for module in model.modules()}
    try:
        model.eval()
        if batchnorm_training:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.train(True)
        yield
    finally:
        for module, training in modes.items():
            module.training = training


def functional_forward(
    model: nn.Module,
    state: FunctionalModelState,
    features: Tensor,
    *,
    phase: str,
    minimum_support_batch_size: int = 2,
) -> tuple[Tensor, BufferAuditResult]:
    if phase not in {"support", "query"}:
        raise ValueError("phase must be support or query")
    validate_functional_state(model, state)
    if phase == "support" and state.buffer_policy is BufferPolicy.SUPPORT_LOCAL:
        if features.ndim < 2 or len(features) < minimum_support_batch_size:
            raise FunctionalStateError(
                "support_local requires at least two support samples per batch"
            )
    batchnorm_training = (
        phase == "support" and state.buffer_policy is BufferPolicy.SUPPORT_LOCAL
    )
    before = tensor_mapping_hash(state.buffers)
    with _controlled_module_modes(model, batchnorm_training=batchnorm_training):
        output = torch.func.functional_call(
            model,
            (dict(state.parameters), dict(state.buffers)),
            (features,),
            strict=True,
        )
    after = tensor_mapping_hash(state.buffers)
    changed = before != after
    expected_change = batchnorm_training and bool(state.buffers)
    if changed and not expected_change:
        raise FunctionalStateError(
            f"{phase} forward changed buffers under {state.buffer_policy.value}"
        )
    return output, BufferAuditResult(
        policy=state.buffer_policy.value,
        phase=phase,
        module_training=False,
        batchnorm_training=batchnorm_training,
        dropout_active=False,
        running_statistics_may_update=batchnorm_training,
        buffers_changed=changed,
    )


def batchnorm_inventory(model: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = f"{name}." if name else ""
            rows.append({
                "module": name,
                "class": module.__class__.__name__,
                "num_features": int(module.num_features),
                "eps": float(module.eps),
                "momentum": None if module.momentum is None else float(module.momentum),
                "affine": bool(module.affine),
                "track_running_stats": bool(module.track_running_stats),
                "running_mean": f"{prefix}running_mean",
                "running_var": f"{prefix}running_var",
                "num_batches_tracked": f"{prefix}num_batches_tracked",
                "minimum_support_batch_size": 2,
                "minimum_samples_per_channel": 2,
            })
    return rows
