from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from model_zoo.DL.dann import (
    DANNFoldData,
    DANNModule,
    DANNObjective,
    DANNPartition,
    DomainDiscriminator,
    GradientReversal,
    aggregate_dann_loss_results,
)
from model_zoo.DL.eegnet import TorchEEGNetClassifier
from model_zoo.DL.shallow_convnet import TorchShallowConvNetClassifier


def _eegnet() -> TorchEEGNetClassifier:
    return TorchEEGNetClassifier(
        n_channels=4,
        n_times=64,
        num_classes=5,
        temporal_kernel_samples=16,
        separable_kernel_samples=8,
        f1=2,
        depth_multiplier=1,
        f2=2,
        pool1=2,
        pool2=2,
        dropout=0.0,
    )


def _shallow() -> TorchShallowConvNetClassifier:
    return TorchShallowConvNetClassifier(
        n_channels=4,
        n_times=128,
        num_classes=5,
        n_filters=4,
        temporal_kernel_samples=9,
        pool_size=16,
        pool_stride=4,
        dropout=0.0,
    )


def _partition(
    name: str,
    *,
    prefix: str,
    count: int = 4,
    domain_id: int = 0,
    task_labels: object | None = None,
    record_prefix: str | None = None,
) -> DANNPartition:
    rng = np.random.default_rng(sum(ord(character) for character in prefix))
    return DANNPartition(
        name=name,
        features=rng.normal(size=(count, 1, 4, 128)).astype(np.float32),
        domain_ids=np.full(count, domain_id, dtype=np.int64),
        sample_ids=[f"{prefix}-sample-{index}" for index in range(count)],
        record_group_ids=[
            f"{record_prefix or prefix}-record-{index}" for index in range(count)
        ],
        subject_ids=[f"subject-{prefix}-{index % 2}" for index in range(count)],
        task_labels=task_labels,
    )


def _fold_data(*, target_labels: object | None = None) -> DANNFoldData:
    return DANNFoldData(
        source_train=_partition(
            "source_train",
            prefix="source",
            domain_id=0,
            task_labels=np.arange(4, dtype=np.int64),
        ),
        target_unlabelled_or_calibration=_partition(
            "target_unlabelled_or_calibration",
            prefix="target",
            domain_id=1,
            task_labels=target_labels,
        ),
        inner_validation=_partition(
            "inner_validation",
            prefix="validation",
            domain_id=0,
            task_labels=np.arange(4, dtype=np.int64),
        ),
        outer_test=_partition(
            "outer_test",
            prefix="outer",
            domain_id=1,
            task_labels=np.arange(4, dtype=np.int64),
        ),
    )


def _batch() -> object:
    return next(iter(_fold_data().training_loader(batch_size=4, shuffle=False)))


def test_gradient_reversal_preserves_forward_values() -> None:
    values = torch.randn(3, 5, requires_grad=True)
    reversed_values = GradientReversal(alpha=0.7)(values)
    torch.testing.assert_close(reversed_values, values)


def test_gradient_reversal_changes_gradient_sign() -> None:
    values = torch.tensor([1.0, -2.0], requires_grad=True)
    GradientReversal(alpha=1.0)(values).sum().backward()
    torch.testing.assert_close(values.grad, -torch.ones_like(values))


def test_gradient_reversal_scales_gradient() -> None:
    values = torch.tensor([1.0, -2.0], requires_grad=True)
    GradientReversal(alpha=0.25)(values).sum().backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, -0.25))


@pytest.mark.parametrize("latent_dim", [4, 11])
def test_domain_discriminator_uses_dynamic_latent_dim(latent_dim: int) -> None:
    discriminator = DomainDiscriminator(
        latent_dim, 3, hidden_dims=(7,), dropout=0.0
    )
    assert discriminator(torch.randn(5, latent_dim)).shape == (5, 3)
    assert discriminator.input_dim == latent_dim


@pytest.mark.parametrize(
    ("model", "source", "target"),
    [
        (_eegnet(), torch.randn(3, 1, 4, 64), torch.randn(2, 1, 4, 64)),
        (_shallow(), torch.randn(3, 1, 4, 128), torch.randn(2, 1, 4, 128)),
    ],
)
def test_raw_eeg_encoders_integrate_with_dann(
    model: nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
) -> None:
    dann = DANNModule(
        model,
        n_domains=2,
        domain_hidden_dims=(8,),
        domain_dropout=0.0,
    ).eval()
    with torch.no_grad():
        outputs = dann(source, target)
    assert outputs.source_task_outputs.shape == (3, 5)
    assert outputs.domain_outputs.shape == (5, 2)
    assert outputs.combined_latent.shape == (5, model.latent_dim)


def test_wrapping_does_not_change_ordinary_model_forward() -> None:
    model = _shallow().eval()
    inputs = torch.randn(3, 1, 4, 128)
    with torch.no_grad():
        before = model(inputs)
        DANNModule(model, n_domains=2, domain_hidden_dims=(), domain_dropout=0.0)
        after = model(inputs)
    torch.testing.assert_close(after, before)


def test_task_and_domain_heads_receive_same_source_latent() -> None:
    model = _shallow().eval()
    dann = DANNModule(
        model,
        n_domains=2,
        domain_hidden_dims=(),
        domain_dropout=0.0,
    ).eval()
    captured: list[torch.Tensor] = []
    handle = dann.domain_discriminator.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )
    with torch.no_grad():
        outputs = dann(
            torch.randn(3, 1, 4, 128),
            torch.randn(2, 1, 4, 128),
        )
    handle.remove()
    torch.testing.assert_close(
        outputs.source_task_outputs,
        model.forward_head(outputs.source_latent),
    )
    torch.testing.assert_close(captured[0], outputs.combined_latent)


def test_domain_head_is_not_in_ordinary_model_checkpoint() -> None:
    model = _shallow()
    ordinary_keys = set(model.state_dict())
    dann = DANNModule(model, n_domains=2)
    payload = dann.checkpoint_payload()

    assert ordinary_keys == set(payload["task_model_state_dict"])
    assert not any("domain" in key for key in ordinary_keys)
    assert payload["domain_discriminator_state_dict"]


def test_dann_checkpoint_restores_task_and_domain_states(tmp_path: Path) -> None:
    torch.manual_seed(42)
    dann = DANNModule(
        _shallow(),
        n_domains=2,
        domain_hidden_dims=(8,),
        domain_dropout=0.0,
    ).eval()
    source = torch.randn(3, 1, 4, 128)
    target = torch.randn(2, 1, 4, 128)
    with torch.no_grad():
        expected = dann(source, target)
    checkpoint = dann.save(tmp_path / "dann.pt", metadata={"fold": 1})

    restored = DANNModule(
        _shallow(),
        n_domains=2,
        domain_hidden_dims=(8,),
        domain_dropout=0.0,
    ).eval()
    metadata = restored.load(checkpoint)
    with torch.no_grad():
        actual = restored(source, target)

    assert metadata == {"fold": 1}
    torch.testing.assert_close(
        actual.source_task_outputs, expected.source_task_outputs
    )
    torch.testing.assert_close(actual.domain_outputs, expected.domain_outputs)


def test_target_task_labels_are_not_exposed_to_dann_step() -> None:
    sentinel_labels = np.full(4, 99, dtype=np.int64)
    batch = next(
        iter(
            _fold_data(target_labels=sentinel_labels).training_loader(
                batch_size=4, shuffle=False
            )
        )
    )
    assert not hasattr(batch, "target_task_labels")
    assert 99 not in batch.source_task_labels.tolist()


def test_outer_test_samples_are_not_in_training_loader() -> None:
    fold = _fold_data()
    observed: set[str] = set()
    for batch in fold.training_loader(batch_size=2, shuffle=False):
        observed.update(batch.source_sample_ids)
        observed.update(batch.target_sample_ids)

    assert observed == (
        set(fold.source_train.sample_ids)
        | set(fold.target_unlabelled_or_calibration.sample_ids)
    )
    assert observed.isdisjoint(fold.outer_test.sample_ids)
    assert observed.isdisjoint(fold.inner_validation.sample_ids)


def test_string_subject_ids_survive_loader_without_numeric_conversion() -> None:
    batch = _batch()
    assert all(
        isinstance(subject_id, str)
        for subject_id in (
            *batch.source_subject_ids,
            *batch.target_subject_ids,
        )
    )
    assert batch.source_subject_ids[0].startswith("subject-source")


def test_cpu_optimizer_step_updates_encoder_task_and_domain_heads() -> None:
    torch.manual_seed(7)
    batch = _batch().to("cpu")
    dann = DANNModule(
        _shallow(),
        n_domains=2,
        gradient_reversal_alpha=0.5,
        domain_hidden_dims=(8,),
        domain_dropout=0.0,
    ).to("cpu")
    objective = DANNObjective(task_type="classification", lambda_domain=0.5)
    before_encoder = {
        name: parameter.detach().clone()
        for name, parameter in dann.task_model.named_parameters()
        if not name.startswith("classifier.")
    }
    before_task = {
        name: parameter.detach().clone()
        for name, parameter in dann.task_model.named_parameters()
        if name.startswith("classifier.")
    }
    before_domain = {
        name: parameter.detach().clone()
        for name, parameter in dann.domain_discriminator.named_parameters()
    }
    optimizer = torch.optim.AdamW(dann.parameters(), lr=0.01)

    optimizer.zero_grad(set_to_none=True)
    losses = objective(
        dann(batch.source_inputs, batch.target_inputs),
        batch.source_task_labels,
        batch.domain_ids,
    )
    losses.total_loss.backward()
    optimizer.step()

    assert all(np.isfinite(list(losses.detached_metrics().values())))
    assert any(
        not torch.equal(before_encoder[name], parameter)
        for name, parameter in dann.task_model.named_parameters()
        if name in before_encoder
    )
    assert any(
        not torch.equal(before_task[name], parameter)
        for name, parameter in dann.task_model.named_parameters()
        if name in before_task
    )
    assert any(
        not torch.equal(before_domain[name], parameter)
        for name, parameter in dann.domain_discriminator.named_parameters()
    )


def _task_gradients(
    model: TorchShallowConvNetClassifier,
    batch: object,
    *,
    lambda_domain: float,
    alpha: float,
) -> tuple[dict[str, torch.Tensor], object]:
    dann = DANNModule(
        model,
        n_domains=2,
        gradient_reversal_alpha=alpha,
        domain_hidden_dims=(),
        domain_dropout=0.0,
    ).eval()
    outputs = dann(batch.source_inputs, batch.target_inputs)
    losses = DANNObjective(
        task_type="classification", lambda_domain=lambda_domain
    )(outputs, batch.source_task_labels, batch.domain_ids)
    losses.total_loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return gradients, losses


def test_lambda_domain_zero_is_equivalent_to_task_only_path() -> None:
    batch = _batch()
    model = _shallow().eval()
    ordinary = deepcopy(model)
    dann_gradients, losses = _task_gradients(
        model, batch, lambda_domain=0.0, alpha=1.0
    )

    ordinary_loss = nn.CrossEntropyLoss()(
        ordinary(batch.source_inputs), batch.source_task_labels
    )
    ordinary_loss.backward()

    torch.testing.assert_close(losses.total_loss, losses.task_loss)
    for name, parameter in ordinary.named_parameters():
        torch.testing.assert_close(dann_gradients[name], parameter.grad)


def test_alpha_zero_keeps_task_model_gradients_unchanged() -> None:
    batch = _batch()
    baseline_model = _shallow().eval()
    domain_model = deepcopy(baseline_model)
    baseline_gradients, _ = _task_gradients(
        baseline_model, batch, lambda_domain=0.0, alpha=0.0
    )
    domain_gradients, _ = _task_gradients(
        domain_model, batch, lambda_domain=1.0, alpha=0.0
    )
    for name in baseline_gradients:
        torch.testing.assert_close(
            domain_gradients[name], baseline_gradients[name]
        )


def test_dimension_errors_are_explicit() -> None:
    discriminator = DomainDiscriminator(4, 2)
    with pytest.raises(ValueError, match="input_dim"):
        discriminator(torch.randn(2, 5))

    dann = DANNModule(_shallow(), n_domains=2)
    with pytest.raises(ValueError, match="expects input tail"):
        dann(
            torch.randn(2, 1, 3, 128),
            torch.randn(2, 1, 4, 128),
        )


def test_loss_aggregation_uses_component_numerators_and_denominators() -> None:
    torch.manual_seed(3)
    dann = DANNModule(
        _shallow(),
        n_domains=2,
        domain_hidden_dims=(),
        domain_dropout=0.0,
    ).eval()
    objective = DANNObjective(
        task_type="classification", lambda_domain=0.25
    )
    results = []
    for source_size, target_size in ((2, 3), (4, 1)):
        outputs = dann(
            torch.randn(source_size, 1, 4, 128),
            torch.randn(target_size, 1, 4, 128),
        )
        results.append(
            objective(
                outputs,
                torch.arange(source_size) % 5,
                torch.tensor([0] * source_size + [1] * target_size),
            )
        )
    aggregated = aggregate_dann_loss_results(results)
    expected_task = sum(
        float(result.task.numerator.detach()) for result in results
    ) / sum(float(result.task.denominator.detach()) for result in results)
    expected_domain = sum(
        float(result.domain.numerator.detach()) for result in results
    ) / sum(float(result.domain.denominator.detach()) for result in results)
    assert aggregated["task_loss"] == pytest.approx(expected_task)
    assert aggregated["domain_loss"] == pytest.approx(expected_domain)
    assert aggregated["total_loss"] == pytest.approx(
        expected_task + 0.25 * expected_domain
    )


@pytest.mark.parametrize("overlap_kind", ["sample", "record"])
def test_fold_contract_rejects_source_outer_test_overlap(
    overlap_kind: str,
) -> None:
    source = _partition(
        "source_train",
        prefix="source",
        domain_id=0,
        task_labels=np.arange(4, dtype=np.int64),
    )
    outer = _partition(
        "outer_test",
        prefix="outer",
        record_prefix="outer",
        domain_id=1,
        task_labels=np.arange(4, dtype=np.int64),
    )
    if overlap_kind == "sample":
        object.__setattr__(
            outer,
            "sample_ids",
            (source.sample_ids[0], *outer.sample_ids[1:]),
        )
    else:
        object.__setattr__(
            outer,
            "record_group_ids",
            (source.record_group_ids[0], *outer.record_group_ids[1:]),
        )

    with pytest.raises(ValueError, match="provenance overlap"):
        DANNFoldData(
            source_train=source,
            target_unlabelled_or_calibration=_partition(
                "target", prefix="target", domain_id=1
            ),
            inner_validation=_partition(
                "validation", prefix="validation", domain_id=0
            ),
            outer_test=outer,
        )
