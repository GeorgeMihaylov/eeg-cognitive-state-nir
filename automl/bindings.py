from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from personal_automl import (
    AdaptationMethod,
    CandidateRegistry,
    CandidateSpec,
    CandidateStatus,
)

from ..model_factory import build_sklearn_model
from ..torch_lstm import build_torch_lstm
from ..torch_mlp import build_torch_mlp
from ..torch_transformer import build_torch_transformer
from ..dann import DANNFoldData, DANNModule, DANNObjective


REQUIRED_CHANNEL_LAYOUT = "emotiv_common_14"


class HeadFreezableTorchAdapter:
    """Wraps a project TorchClassificationAdapter to satisfy ModelAdapter
    and add head-only-fine-tuning support for HEAD_ONLY candidates.
    """

    def __init__(self, inner_adapter: Any) -> None:
        self._inner = inner_adapter

    @property
    def model(self) -> torch.nn.Module:
        return self._inner.model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "HeadFreezableTorchAdapter":
        self._inner.fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._inner.predict(X)

    def freeze_all_but_head(self) -> None:
        if not hasattr(self.model, "output_head_parameter_prefixes"):
            raise ValueError(
                f"{type(self.model).__name__} does not expose "
                "output_head_parameter_prefixes(); it cannot support "
                "HEAD_ONLY fine-tuning."
            )
        prefixes = self.model.output_head_parameter_prefixes()
        for name, param in self.model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)

    def unfreeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def __deepcopy__(self, memo: dict) -> "HeadFreezableTorchAdapter":
        import copy

        return HeadFreezableTorchAdapter(copy.deepcopy(self._inner, memo))


def _wrapped_torch_builder(raw_builder):
    def _build(input_shape: Sequence[int], num_outputs: int, params: Mapping[str, Any]):
        return HeadFreezableTorchAdapter(raw_builder(input_shape, num_outputs, params))

    return _build


def _lstm_builder(bidirectional: bool):
    def _build(input_shape: Sequence[int], num_outputs: int, params: Mapping[str, Any]):
        adapter = build_torch_lstm(
            input_shape, num_outputs, params, force_bidirectional=bidirectional
        )
        return HeadFreezableTorchAdapter(adapter)

    return _build


def _sklearn_random_forest_builder(input_shape: Sequence[int], num_outputs: int, params: Mapping[str, Any]):
    return build_sklearn_model("random_forest", "classification", params)

def build_eeg_registry() -> CandidateRegistry:
    return CandidateRegistry(
        [
            CandidateSpec(
                name="transformer_head_only",
                builder=_wrapped_torch_builder(build_torch_transformer),
                builder_params={"head_type": "categorical"},
                adaptation_method=AdaptationMethod.HEAD_ONLY,
                status=CandidateStatus.APPROVED_DEFAULT,
                required_tags={"channel_layout": REQUIRED_CHANNEL_LAYOUT},
                notes="Best zero-shot and calibrated result in baseline v1.",
            ),
            CandidateSpec(
                name="transformer_full_finetune",
                builder=_wrapped_torch_builder(build_torch_transformer),
                builder_params={"head_type": "categorical"},
                adaptation_method=AdaptationMethod.FULL_FINETUNE,
                status=CandidateStatus.EXPERIMENTAL_CONDITIONAL,
                min_calibration_samples=200,
                required_tags={"channel_layout": REQUIRED_CHANNEL_LAYOUT},
                notes="Full fine-tune isn't universally better than head-only; "
                "gated on calibration volume.",
            ),
            CandidateSpec(
                name="bilstm_head_only",
                builder=_lstm_builder(bidirectional=True),
                builder_params={},
                adaptation_method=AdaptationMethod.HEAD_ONLY,
                status=CandidateStatus.APPROVED_FALLBACK,
                required_tags={"channel_layout": REQUIRED_CHANNEL_LAYOUT},
                notes="Comparable macro F1 (~0.36) to Transformer on label_q5.",
            ),
            CandidateSpec(
                name="lstm_head_only",
                builder=_lstm_builder(bidirectional=False),
                builder_params={},
                adaptation_method=AdaptationMethod.HEAD_ONLY,
                status=CandidateStatus.APPROVED_FALLBACK,
                required_tags={"channel_layout": REQUIRED_CHANNEL_LAYOUT},
            ),
            CandidateSpec(
                name="random_forest",
                builder=_sklearn_random_forest_builder,
                builder_params={},
                adaptation_method=AdaptationMethod.REFIT,
                status=CandidateStatus.APPROVED_FALLBACK,
                notes="Strong low-data fallback; beats mean baseline in PM regression.",
            ),
            CandidateSpec(
                name="mlp_pooled",
                builder=_wrapped_torch_builder(build_torch_mlp),
                builder_params={"hidden_dims": [128, 64]},
                adaptation_method=AdaptationMethod.HEAD_ONLY,
                status=CandidateStatus.DEPRIORITIZED,
                required_tags={"channel_layout": REQUIRED_CHANNEL_LAYOUT},
                notes="Consistently worse than Transformer/RNNs; kept for diagnostics.",
            ),
            CandidateSpec(
                name="dann_transformer_shadow",
                builder=_wrapped_torch_builder(build_torch_transformer),
                builder_params={"head_type": "categorical"},
                adaptation_method=AdaptationMethod.SHADOW_ONLY,
                status=CandidateStatus.EXPERIMENTAL_GATED,
                required_tags={"channel_layout": REQUIRED_CHANNEL_LAYOUT},
                condition=lambda meta: meta.tag_shift("source_domain"),
                notes="One-fold diagnostic gave 'proceed' but the participant "
                "bootstrap CI includes zero; confirmatory 5-fold x 3-seed "
                "protocol has not run. Shadow mode only until that clears.",
            ),
        ]
    )

def run_dann_shadow_diagnostic(
    candidate: CandidateSpec,
    dann_fold_data: "DANNFoldData",
    *,
    n_domains: int = 2,
    lambda_domain: float = 1.0,
    max_steps: int = 200,
) -> Mapping[str, Any]:
    num_outputs = int(np.max(np.asarray(dann_fold_data.source_train.task_labels)) + 1)
    task_model = candidate.build(
        input_shape=tuple(np.shape(dann_fold_data.source_train.features)[1:]),
        num_outputs=num_outputs,
    ).model
    dann_module = DANNModule(task_model, n_domains=n_domains)
    objective = DANNObjective(task_type="classification", lambda_domain=lambda_domain)
    loader = dann_fold_data.training_loader(batch_size=32)
    optimizer = torch.optim.Adam(dann_module.parameters(), lr=1e-4)

    dann_module.train()
    step = 0
    for batch in loader:
        if step >= max_steps:
            break
        result = dann_module(batch.source_inputs, batch.target_inputs)
        loss = objective(result, batch.source_task_labels, batch.domain_ids)
        optimizer.zero_grad()
        loss.total_loss.backward()
        optimizer.step()
        step += 1
    return {"steps_run": step}


SHADOW_RUNNERS = {"dann_transformer_shadow": run_dann_shadow_diagnostic}
