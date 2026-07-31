"""First-order MAML mechanics for approved synthetic CPU diagnostics only."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .buffers import (
    BufferAuditResult,
    FunctionalModelState,
    create_functional_state,
    functional_forward,
)
from .protocol import MetaLearnerProtocol


class FOMAMLError(RuntimeError):
    """Raised when a mathematical or safety invariant is violated."""


@dataclass(frozen=True)
class FOMAMLConfig:
    algorithm_id: str = "fomaml"
    task_type: str = "classification"
    inner_steps: int = 2
    inner_learning_rate: float = 0.1
    meta_learning_rate: float = 0.01
    episodes_per_meta_batch: int = 4
    maximum_meta_steps: int = 20
    loss_name: str = "cross_entropy"
    gradient_clip_norm: float = 5.0
    buffer_policy: str = "frozen"
    device: str = "cpu"
    seed: int = 42
    finite_check: bool = True

    def __post_init__(self) -> None:
        if self.algorithm_id != "fomaml":
            raise ValueError("algorithm_id must be 'fomaml'")
        if self.task_type != "classification":
            raise ValueError("Synthetic FOMAML supports classification only")
        if self.loss_name != "cross_entropy":
            raise ValueError("Synthetic FOMAML supports cross_entropy only")
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid FOMAML device {self.device!r}") from exc
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("FOMAML supports CPU or CUDA devices only")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if self.buffer_policy not in {"frozen", "frozen_global", "support_local"}:
            raise ValueError(
                "buffer_policy must be frozen, frozen_global, or support_local"
            )
        if min(self.inner_steps, self.episodes_per_meta_batch, self.maximum_meta_steps) <= 0:
            raise ValueError("Step and episode counts must be positive")
        if min(self.inner_learning_rate, self.meta_learning_rate) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FOMAMLConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class FOMAMLAdaptationResult:
    fast_weights: Mapping[str, Tensor] = field(repr=False)
    buffers: Mapping[str, Tensor] = field(repr=False)
    buffer_policy: str
    buffer_audits: tuple[BufferAuditResult, ...]
    support_losses: tuple[float, ...]
    support_accuracy_before: float
    support_accuracy_after: float
    gradient_norm_per_step: tuple[float, ...]
    create_graph: bool
    base_unchanged: bool
    storage_independent: bool


@dataclass(frozen=True)
class FOMAMLEpisodeResult:
    episode_id: str
    support_losses: tuple[float, ...]
    inner_gradient_norms: tuple[float, ...]
    support_loss_before: float
    support_loss_after: float
    support_accuracy_before: float
    support_accuracy_after: float
    query_loss: float
    query_accuracy: float
    query_gradient_norm: float
    query_gradients: Mapping[str, Tensor] = field(repr=False)


@dataclass(frozen=True)
class FOMAMLBatchResult:
    episodes: tuple[FOMAMLEpisodeResult, ...]
    mean_gradients: Mapping[str, Tensor] = field(repr=False)
    meta_gradient_norm: float


@dataclass(frozen=True)
class FOMAMLStepResult:
    step: int
    episode_count: int
    support_loss_before: float
    support_loss_after: float
    query_loss: float
    meta_gradient_norm_before_clip: float
    meta_gradient_norm_after_clip: float
    parameters_updated: int
    base_unchanged_before_step: bool
    optimizer_state_finite: bool


def model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _global_norm(values: Iterable[Tensor]) -> float:
    tensors = [value.detach().float() for value in values]
    if not tensors:
        return 0.0
    return float(torch.sqrt(sum(torch.sum(value * value) for value in tensors)).item())


def _finite(values: Iterable[Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def validate_parameter_mapping(
    model: nn.Module,
    parameters: Mapping[str, Tensor],
) -> None:
    expected = OrderedDict(model.named_parameters())
    if tuple(parameters) != tuple(expected):
        missing = sorted(set(expected) - set(parameters))
        extra = sorted(set(parameters) - set(expected))
        raise FOMAMLError(f"Parameter-name mismatch; missing={missing}, extra={extra}")
    for name, base in expected.items():
        if tuple(parameters[name].shape) != tuple(base.shape):
            raise FOMAMLError(
                f"Parameter-shape mismatch for {name}: "
                f"{tuple(parameters[name].shape)} != {tuple(base.shape)}"
            )


def _has_stateful_buffers(model: nn.Module) -> bool:
    return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())


class FirstOrderMAML(MetaLearnerProtocol):
    """FOMAML with detached inner updates and query-gradient transfer."""

    def __init__(self, model: nn.Module, config: FOMAMLConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        if not tuple(self.model.named_parameters()):
            raise ValueError("FOMAML requires trainable named parameters")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.meta_learning_rate
        )
        self.meta_step_index = 0

    def create_fast_weights(self, model: nn.Module | None = None) -> OrderedDict[str, Tensor]:
        module = self.model if model is None else model
        fast = OrderedDict(
            (name, parameter.detach().clone().requires_grad_(True))
            for name, parameter in module.named_parameters()
        )
        validate_parameter_mapping(module, fast)
        if any(
            fast[name].data_ptr() == parameter.data_ptr()
            for name, parameter in module.named_parameters()
        ):
            raise FOMAMLError("Fast weights share storage with base parameters")
        return fast

    def _forward(
        self,
        model: nn.Module,
        state: FunctionalModelState,
        features: Tensor,
        *,
        phase: str,
    ) -> tuple[Tensor, BufferAuditResult]:
        validate_parameter_mapping(model, state.parameters)
        return functional_forward(
            model, state, features.to(self.device), phase=phase
        )

    @staticmethod
    def _accuracy(logits: Tensor, targets: Tensor) -> float:
        return float((logits.argmax(dim=1) == targets).float().mean().item())

    def adapt(
        self,
        model: nn.Module,
        support_batch: tuple[Tensor, Tensor],
    ) -> FOMAMLAdaptationResult:
        if model is not self.model:
            raise FOMAMLError("FirstOrderMAML can adapt only its bound base model")
        features, targets = support_batch
        features = features.to(self.device)
        targets = targets.to(self.device, dtype=torch.long)
        if not bool(torch.isfinite(features).all()):
            raise FOMAMLError("Support features contain NaN or Inf")
        base_hash = model_state_hash(model)
        fast = self.create_fast_weights(model)
        state = create_functional_state(model, self.config.buffer_policy)
        state = state.with_parameters(fast)
        losses: list[float] = []
        gradient_norms: list[float] = []
        buffer_audits: list[BufferAuditResult] = []
        logits, buffer_audit = self._forward(
            model, state, features, phase="support"
        )
        buffer_audits.append(buffer_audit)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        losses.append(float(loss.item()))
        accuracy_before = self._accuracy(logits, targets)
        for _ in range(self.config.inner_steps):
            gradients = torch.autograd.grad(
                loss,
                tuple(fast.values()),
                create_graph=False,
                retain_graph=False,
            )
            if self.config.finite_check and not _finite(gradients):
                raise FOMAMLError("Inner gradients contain NaN or Inf")
            gradient_norms.append(_global_norm(gradients))
            fast = OrderedDict(
                (
                    name,
                    (parameter - self.config.inner_learning_rate * gradient)
                    .detach()
                    .requires_grad_(True),
                )
                for (name, parameter), gradient in zip(fast.items(), gradients)
            )
            validate_parameter_mapping(model, fast)
            state = state.with_parameters(fast)
            logits, buffer_audit = self._forward(
                model, state, features, phase="support"
            )
            buffer_audits.append(buffer_audit)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            losses.append(float(loss.item()))
        if self.config.finite_check and not all(torch.isfinite(torch.tensor(losses))):
            raise FOMAMLError("Support loss contains NaN or Inf")
        base_unchanged = model_state_hash(model) == base_hash
        if not base_unchanged:
            raise FOMAMLError("Base model changed during inner adaptation")
        independent = all(
            fast[name].data_ptr() != parameter.data_ptr()
            for name, parameter in model.named_parameters()
        )
        return FOMAMLAdaptationResult(
            fast_weights=fast,
            buffers=OrderedDict(
                (name, value.detach().clone())
                for name, value in state.buffers.items()
            ),
            buffer_policy=state.buffer_policy.value,
            buffer_audits=tuple(buffer_audits),
            support_losses=tuple(losses),
            support_accuracy_before=accuracy_before,
            support_accuracy_after=self._accuracy(logits, targets),
            gradient_norm_per_step=tuple(gradient_norms),
            create_graph=False,
            base_unchanged=base_unchanged,
            storage_independent=independent,
        )

    def evaluate(
        self,
        adapted_model: FOMAMLAdaptationResult,
        query_batch: tuple[Tensor, Tensor],
    ) -> tuple[float, float, OrderedDict[str, Tensor]]:
        features, targets = query_batch
        features = features.to(self.device)
        targets = targets.to(self.device, dtype=torch.long)
        if not bool(torch.isfinite(features).all()):
            raise FOMAMLError("Query features contain NaN or Inf")
        state = create_functional_state(
            self.model, adapted_model.buffer_policy
        ).with_parameters(adapted_model.fast_weights)
        state = FunctionalModelState(
            parameters=state.parameters,
            buffers=OrderedDict(
                (name, value.detach().clone())
                for name, value in adapted_model.buffers.items()
            ),
            training_mode=state.training_mode,
            architecture_signature=state.architecture_signature,
            buffer_policy=state.buffer_policy,
        )
        buffers_before = {
            name: value.detach().clone() for name, value in state.buffers.items()
        }
        logits, _ = self._forward(
            self.model, state, features, phase="query"
        )
        if any(
            not torch.equal(buffers_before[name], value)
            for name, value in state.buffers.items()
        ):
            raise FOMAMLError("Query changed episode-local buffers")
        loss = torch.nn.functional.cross_entropy(logits, targets)
        gradients = torch.autograd.grad(
            loss, tuple(adapted_model.fast_weights.values()), create_graph=False
        )
        if self.config.finite_check and (
            not bool(torch.isfinite(loss)) or not _finite(gradients)
        ):
            raise FOMAMLError("Query loss or gradients contain NaN or Inf")
        mapped = OrderedDict(
            (name, gradient.detach().clone())
            for name, gradient in zip(adapted_model.fast_weights, gradients)
        )
        validate_parameter_mapping(self.model, mapped)
        return float(loss.item()), self._accuracy(logits, targets), mapped

    def predict_adapted(
        self,
        adapted_model: FOMAMLAdaptationResult,
        features: Tensor,
    ) -> tuple[Tensor, BufferAuditResult]:
        """Predict from frozen episode state without labels or buffer updates."""
        if not bool(torch.isfinite(features).all()):
            raise FOMAMLError("Prediction features contain NaN or Inf")
        state = create_functional_state(
            self.model, adapted_model.buffer_policy
        ).with_parameters(adapted_model.fast_weights)
        state = FunctionalModelState(
            parameters=state.parameters,
            buffers=OrderedDict(
                (name, value.detach().clone())
                for name, value in adapted_model.buffers.items()
            ),
            training_mode=state.training_mode,
            architecture_signature=state.architecture_signature,
            buffer_policy=state.buffer_policy,
        )
        buffers_before = {
            name: value.detach().clone() for name, value in state.buffers.items()
        }
        with torch.no_grad():
            logits, audit = self._forward(
                self.model, state, features, phase="query"
            )
        if any(
            not torch.equal(buffers_before[name], value)
            for name, value in state.buffers.items()
        ):
            raise FOMAMLError("Prediction query changed episode-local buffers")
        if not bool(torch.isfinite(logits).all()):
            raise FOMAMLError("Prediction logits contain NaN or Inf")
        return logits.detach().cpu(), audit

    def episode_result(self, episode: Any) -> FOMAMLEpisodeResult:
        adaptation = self.adapt(
            self.model, (episode.support_features, episode.support_targets)
        )
        query_loss, query_accuracy, gradients = self.evaluate(
            adaptation, (episode.query_features, episode.query_targets)
        )
        return FOMAMLEpisodeResult(
            episode_id=episode.episode.episode_id,
            support_losses=adaptation.support_losses,
            inner_gradient_norms=adaptation.gradient_norm_per_step,
            support_loss_before=adaptation.support_losses[0],
            support_loss_after=adaptation.support_losses[-1],
            support_accuracy_before=adaptation.support_accuracy_before,
            support_accuracy_after=adaptation.support_accuracy_after,
            query_loss=query_loss,
            query_accuracy=query_accuracy,
            query_gradient_norm=_global_norm(gradients.values()),
            query_gradients=gradients,
        )

    def compute_meta_batch_gradients(
        self, episodes: Sequence[Any]
    ) -> FOMAMLBatchResult:
        if not episodes:
            raise ValueError("A meta-batch cannot be empty")
        base_hash = model_state_hash(self.model)
        results = tuple(self.episode_result(episode) for episode in episodes)
        if model_state_hash(self.model) != base_hash:
            raise FOMAMLError("Base model changed before meta-optimizer step")
        names = tuple(name for name, _ in self.model.named_parameters())
        means = OrderedDict(
            (
                name,
                torch.stack([result.query_gradients[name] for result in results]).mean(dim=0),
            )
            for name in names
        )
        validate_parameter_mapping(self.model, means)
        if self.config.finite_check and not _finite(means.values()):
            raise FOMAMLError("Averaged meta-gradients contain NaN or Inf")
        norm = _global_norm(means.values())
        if norm <= torch.finfo(torch.float32).eps:
            raise FOMAMLError("Meta-gradient norm is numerically zero")
        return FOMAMLBatchResult(results, means, norm)

    def meta_train_step(self, episodes: Iterable[Any]) -> FOMAMLStepResult:
        batch = self.compute_meta_batch_gradients(tuple(episodes))
        base_before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            parameter.grad = batch.mean_gradients[name].detach().clone()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        clipped_norm = _global_norm(
            parameter.grad for parameter in self.model.parameters() if parameter.grad is not None
        )
        self.optimizer.step()
        updated = sum(
            not torch.equal(base_before[name], parameter.detach())
            for name, parameter in self.model.named_parameters()
        )
        if updated == 0:
            raise FOMAMLError("Meta-optimizer step changed no base parameter")
        optimizer_finite = all(
            not torch.is_tensor(value) or bool(torch.isfinite(value).all())
            for state in self.optimizer.state.values()
            for value in state.values()
        )
        if not optimizer_finite:
            raise FOMAMLError("Optimizer state contains NaN or Inf")
        self.meta_step_index += 1
        return FOMAMLStepResult(
            step=self.meta_step_index,
            episode_count=len(batch.episodes),
            support_loss_before=sum(r.support_loss_before for r in batch.episodes) / len(batch.episodes),
            support_loss_after=sum(r.support_loss_after for r in batch.episodes) / len(batch.episodes),
            query_loss=sum(r.query_loss for r in batch.episodes) / len(batch.episodes),
            meta_gradient_norm_before_clip=batch.meta_gradient_norm,
            meta_gradient_norm_after_clip=clipped_norm,
            parameters_updated=updated,
            base_unchanged_before_step=True,
            optimizer_state_finite=optimizer_finite,
        )


def audit_production_model_compatibility(
    model: nn.Module,
    example_input: Tensor,
) -> dict[str, Any]:
    """Read-only functional-forward audit for stateful production EEG models."""
    before = model_state_hash(model)
    params = OrderedDict(
        (name, value.detach().clone().requires_grad_(value.requires_grad))
        for name, value in model.named_parameters()
    )
    buffers = OrderedDict(
        (name, value.detach().clone()) for name, value in model.named_buffers()
    )
    validate_parameter_mapping(model, params)
    original_mode = model.training
    model.eval()
    with torch.no_grad():
        output = torch.func.functional_call(
            model, (dict(params), dict(buffers)), (example_input,), strict=True
        )
    model.train(original_mode)
    unchanged = model_state_hash(model) == before
    stateful = _has_stateful_buffers(model)
    return {
        "model": model.__class__.__name__,
        "parameter_names_available": bool(params),
        "parameter_shapes_valid": all(
            params[name].shape == value.shape for name, value in model.named_parameters()
        ),
        "functional_eval_forward": True,
        "output_shape": list(output.shape),
        "latent_dim": int(getattr(model, "latent_dim")),
        "output_head_width": int(model.get_output_head().out_features),
        "state_dict_unchanged": unchanged,
        "stateful_buffers_present": stateful,
        "supported_buffer_policies": ["frozen_global", "support_local"],
        "adaptation_supported": True,
        "status": "compatible_with_explicit_buffer_policy",
    }
