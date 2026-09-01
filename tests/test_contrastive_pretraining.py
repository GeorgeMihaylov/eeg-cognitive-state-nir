from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cogstate.model_zoo.DL.contrastive import (
    AmplitudeScaling,
    ChannelMasking,
    ContrastiveFoldData,
    ContrastiveModule,
    ContrastiveObjective,
    EEGAugmentationPipeline,
    GaussianNoise,
    ProjectionHead,
    TemporalShift,
    TimeMasking,
    aggregate_contrastive_loss_results,
    export_encoder_checkpoint,
    load_encoder_checkpoint,
    nt_xent_logits,
)
from cogstate.model_zoo.DL.eegnet import TorchEEGNetClassifier
from cogstate.model_zoo.DL.shallow_convnet import TorchShallowConvNetClassifier


def _eegnet(*, num_classes: int = 5) -> TorchEEGNetClassifier:
    return TorchEEGNetClassifier(
        n_channels=4,
        n_times=64,
        num_classes=num_classes,
        temporal_kernel_samples=16,
        separable_kernel_samples=8,
        f1=2,
        depth_multiplier=1,
        f2=2,
        pool1=2,
        pool2=2,
        dropout=0.0,
    )


def _shallow(
    *,
    num_classes: int = 5,
    n_filters: int = 4,
) -> TorchShallowConvNetClassifier:
    return TorchShallowConvNetClassifier(
        n_channels=4,
        n_times=128,
        num_classes=num_classes,
        n_filters=n_filters,
        temporal_kernel_samples=9,
        pool_size=16,
        pool_stride=4,
        dropout=0.0,
    )


def _raw_batch(batch_size: int = 4, *, n_times: int = 128) -> torch.Tensor:
    generator = torch.Generator().manual_seed(42)
    return torch.randn(
        batch_size, 1, 4, n_times, generator=generator
    )


@pytest.mark.parametrize(
    "transform",
    [
        GaussianNoise(enabled=True, probability=1.0, std=0.05),
        AmplitudeScaling(
            enabled=True, probability=1.0, minimum=0.8, maximum=1.2
        ),
        TimeMasking(
            enabled=True, probability=1.0, maximum_fraction=0.2
        ),
        ChannelMasking(
            enabled=True, probability=1.0, maximum_channels=2
        ),
        TemporalShift(
            enabled=True, probability=1.0, maximum_fraction=0.1
        ),
    ],
)
def test_each_augmentation_preserves_shape_and_input(
    transform: nn.Module,
) -> None:
    inputs = _raw_batch()
    original = inputs.clone()
    output = transform(
        inputs, generator=torch.Generator().manual_seed(7)
    )
    assert output.shape == inputs.shape
    assert torch.equal(inputs, original)
    assert torch.isfinite(output).all()


def _active_pipeline() -> EEGAugmentationPipeline:
    return EEGAugmentationPipeline(
        gaussian_noise={"enabled": True, "probability": 1.0, "std": 0.02},
        amplitude_scaling={
            "enabled": True,
            "probability": 1.0,
            "minimum": 0.9,
            "maximum": 1.1,
        },
        time_masking={
            "enabled": True,
            "probability": 1.0,
            "maximum_fraction": 0.1,
        },
        channel_masking={
            "enabled": True,
            "probability": 1.0,
            "maximum_channels": 1,
        },
        temporal_shift={
            "enabled": True,
            "probability": 1.0,
            "maximum_fraction": 0.05,
        },
    )


def test_fixed_augmentation_seed_is_deterministic() -> None:
    inputs = _raw_batch()
    pipeline = _active_pipeline()
    first = pipeline(
        inputs, generator=EEGAugmentationPipeline.make_generator(19)
    )
    second = pipeline(
        inputs, generator=EEGAugmentationPipeline.make_generator(19)
    )
    torch.testing.assert_close(first, second)


def test_different_augmentation_seeds_can_change_views() -> None:
    inputs = _raw_batch()
    pipeline = _active_pipeline()
    first = pipeline(
        inputs, generator=EEGAugmentationPipeline.make_generator(1)
    )
    second = pipeline(
        inputs, generator=EEGAugmentationPipeline.make_generator(2)
    )
    assert not torch.equal(first, second)


def test_disabled_augmentations_are_neutral() -> None:
    inputs = _raw_batch()
    output = EEGAugmentationPipeline()(inputs)
    torch.testing.assert_close(output, inputs)
    assert output.data_ptr() != inputs.data_ptr()


def test_two_views_are_finite_and_do_not_modify_input() -> None:
    inputs = _raw_batch()
    original = inputs.clone()
    first, second = _active_pipeline().two_views(
        inputs, generator=EEGAugmentationPipeline.make_generator(23)
    )
    assert torch.equal(inputs, original)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert first.shape == second.shape == inputs.shape
    assert not torch.equal(first, second)


@pytest.mark.parametrize("latent_dim", [4, 13])
def test_projection_head_uses_dynamic_latent_dim(latent_dim: int) -> None:
    head = ProjectionHead(latent_dim, 7, hidden_dim=9)
    output = head(torch.randn(3, latent_dim))
    assert output.shape == (3, 7)
    torch.testing.assert_close(output.norm(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)


def test_nt_xent_is_finite_and_exposes_aggregation_parts() -> None:
    model = ContrastiveModule(
        _shallow(), projection_dim=8, projection_hidden_dim=6
    ).eval()
    views = _active_pipeline().two_views(
        _raw_batch(), generator=EEGAugmentationPipeline.make_generator(5)
    )
    result = ContrastiveObjective(temperature=0.2)(model(*views))
    assert torch.isfinite(result.contrastive_loss)
    assert result.loss.denominator.item() == 8
    assert all(np.isfinite(list(result.detached_metrics().values())))


def test_positive_pairs_are_aligned_and_self_comparisons_are_masked() -> None:
    first = torch.eye(3)
    second = first.clone()
    logits, positives, _ = nt_xent_logits(
        first, second, temperature=0.1
    )
    assert positives.tolist() == [3, 4, 5, 0, 1, 2]
    assert torch.isfinite(logits).all()
    assert torch.equal(
        logits.diag(),
        torch.full((6,), torch.finfo(logits.dtype).min),
    )
    assert logits.argmax(dim=1).tolist() == positives.tolist()


def test_batch_size_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        nt_xent_logits(
            torch.randn(1, 4),
            torch.randn(1, 4),
            temperature=0.1,
        )


@pytest.mark.parametrize(
    ("model", "inputs"),
    [
        (_eegnet(), _raw_batch(n_times=64)),
        (_shallow(), _raw_batch()),
    ],
)
def test_eegnet_and_shallowconvnet_support_contrastive_module(
    model: nn.Module,
    inputs: torch.Tensor,
) -> None:
    module = ContrastiveModule(model, projection_dim=6).eval()
    with torch.no_grad():
        output = module(inputs, inputs)
    assert output.first_latent.shape == (4, model.latent_dim)
    assert output.first_projection.shape == (4, 6)


def test_contrastive_wrapper_does_not_change_ordinary_forward_or_state() -> None:
    model = _shallow().eval()
    inputs = _raw_batch()
    state_keys = set(model.state_dict())
    with torch.no_grad():
        expected = model(inputs)
        module = ContrastiveModule(model, projection_dim=8)
        actual = model(inputs)
    torch.testing.assert_close(actual, expected)
    assert set(model.state_dict()) == state_keys
    assert not any("projection" in key for key in model.state_dict())
    assert module.projection_head.state_dict()


def _fold_source(
    *,
    record_overlap: bool = False,
    sample_overlap: bool = False,
    subject_overlap: bool = False,
    training_indices: tuple[int, ...] = (0, 1, 2, 3),
) -> ContrastiveFoldData:
    features = np.random.default_rng(42).normal(
        size=(8, 1, 4, 128)
    ).astype(np.float32)
    records = [f"record-{index}" for index in range(8)]
    if record_overlap:
        records[5] = records[0]
    samples = [f"sample-{index}" for index in range(8)]
    if sample_overlap:
        samples[5] = samples[0]
    subjects = [
        "train-a",
        "train-a",
        "train-b",
        "train-b",
        "validation",
        "outer-a",
        "outer-b",
        "target-final",
    ]
    if subject_overlap:
        subjects[5] = subjects[0]
    return ContrastiveFoldData.from_indexed_source(
        features=features,
        sample_ids=samples,
        record_group_ids=records,
        subject_ids=subjects,
        training_indices=training_indices,
        inner_validation_indices=(4,),
        outer_test_indices=(5, 6),
        target_final_evaluation_indices=(7,),
        fold_id="fold-01",
    )


def test_fold_loader_contains_only_authorized_training_indices() -> None:
    fold = _fold_source()
    observed_indices: set[int] = set()
    observed_samples: set[str] = set()
    for batch in fold.training_loader(batch_size=2, shuffle=False):
        observed_indices.update(batch.source_indices.tolist())
        observed_samples.update(batch.sample_ids)
    assert observed_indices == {0, 1, 2, 3}
    assert observed_samples == {f"sample-{index}" for index in range(4)}
    assert observed_indices.isdisjoint({4, 5, 6, 7})


def test_outer_test_indices_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap forbidden"):
        _fold_source(training_indices=(0, 1, 2, 5))


def test_record_group_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="record_group_id overlap"):
        _fold_source(record_overlap=True)


def test_outer_test_sample_id_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_ids must be globally unique"):
        _fold_source(sample_overlap=True)


def test_outer_test_subject_id_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="subject_id overlap"):
        _fold_source(subject_overlap=True)


def test_string_subject_ids_and_fold_provenance_are_preserved() -> None:
    fold = _fold_source()
    batch = next(
        iter(fold.training_loader(batch_size=2, shuffle=False))
    )
    assert all(isinstance(value, str) for value in batch.subject_ids)
    provenance = fold.training_provenance()
    assert provenance["fold_id"] == "fold-01"
    assert provenance["training_sample_count"] == 4
    assert provenance["training_subject_ids"] == ["train-a", "train-b"]
    assert len(provenance["training_sample_ids_sha256"]) == 64


def _one_contrastive_step(
    module: ContrastiveModule,
    inputs: torch.Tensor,
) -> tuple[object, torch.optim.Optimizer]:
    pipeline = _active_pipeline()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    outputs = module.augmented_forward(
        inputs,
        pipeline,
        generator=EEGAugmentationPipeline.make_generator(42),
    )
    result = ContrastiveObjective(temperature=0.2)(outputs)
    result.contrastive_loss.backward()
    optimizer.step()
    return result, optimizer


def test_cpu_contrastive_step_updates_encoder_and_projection() -> None:
    module = ContrastiveModule(
        _shallow(), projection_dim=8, projection_hidden_dim=6
    ).to("cpu")
    before_encoder = {
        name: value.detach().clone()
        for name, value in module.encoder_model.named_parameters()
        if not name.startswith("classifier.")
    }
    before_projection = {
        name: value.detach().clone()
        for name, value in module.projection_head.named_parameters()
    }
    result, _ = _one_contrastive_step(module, _raw_batch())
    assert torch.isfinite(result.contrastive_loss)
    assert any(
        not torch.equal(before_encoder[name], parameter)
        for name, parameter in module.encoder_model.named_parameters()
        if name in before_encoder
    )
    assert any(
        not torch.equal(before_projection[name], parameter)
        for name, parameter in module.projection_head.named_parameters()
    )


def test_contrastive_checkpoint_restores_encoder_projection_and_optimizer(
    tmp_path: Path,
) -> None:
    torch.manual_seed(9)
    module = ContrastiveModule(
        _shallow(), projection_dim=8, projection_hidden_dim=6
    )
    _, optimizer = _one_contrastive_step(module, _raw_batch())
    checkpoint = module.save(
        tmp_path / "contrastive.pt",
        optimizer=optimizer,
        configuration={"temperature": 0.2},
        augmentation_configuration=_active_pipeline().configuration(),
        seed=42,
        epoch=1,
        training_provenance=_fold_source().training_provenance(),
    )
    module.eval()
    with torch.no_grad():
        expected = module(_raw_batch(), _raw_batch())

    restored = ContrastiveModule(
        _shallow(), projection_dim=8, projection_hidden_dim=6
    )
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    metadata = restored.load(checkpoint, optimizer=restored_optimizer)
    restored.eval()
    with torch.no_grad():
        actual = restored(_raw_batch(), _raw_batch())

    assert metadata["seed"] == 42
    assert metadata["epoch"] == 1
    assert metadata["optimizer_state_available"] is True
    torch.testing.assert_close(
        actual.first_latent, expected.first_latent
    )
    torch.testing.assert_close(
        actual.first_projection, expected.first_projection
    )


def test_encoder_only_export_loads_into_new_model_with_different_head(
    tmp_path: Path,
) -> None:
    source = _shallow(num_classes=5)
    module = ContrastiveModule(source, projection_dim=8)
    _one_contrastive_step(module, _raw_batch())
    checkpoint = export_encoder_checkpoint(
        source, tmp_path / "encoder.pt", metadata={"source": "contrastive"}
    )
    expected_encoder = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
        if not name.startswith("classifier.")
    }

    target = _shallow(num_classes=7)
    target_head_before = deepcopy(target.classifier.state_dict())
    metadata = load_encoder_checkpoint(target, checkpoint)

    assert metadata == {"source": "contrastive"}
    assert all(
        torch.equal(value, target.state_dict()[name])
        for name, value in expected_encoder.items()
    )
    assert all(
        torch.equal(value, target.classifier.state_dict()[name])
        for name, value in target_head_before.items()
    )
    target.eval()
    assert target(_raw_batch(2)).shape == (2, 7)


def test_incompatible_encoder_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = export_encoder_checkpoint(
        _shallow(), tmp_path / "encoder.pt"
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_encoder_checkpoint(_eegnet(), checkpoint)
    with pytest.raises(ValueError, match="incompatible"):
        load_encoder_checkpoint(
            _shallow(n_filters=6), checkpoint
        )


def _encoder_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    }


def test_head_only_step_preserves_loaded_encoder() -> None:
    model = _shallow(num_classes=7)
    model.freeze_encoder()
    before = _encoder_parameters(model)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.MSELoss()(model(_raw_batch()), torch.randn(4, 7))
    loss.backward()
    optimizer.step()
    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in before
    )


def test_full_model_step_gives_encoder_gradients_and_changes_encoder() -> None:
    model = _shallow(num_classes=7)
    model.unfreeze_encoder()
    before = _encoder_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.MSELoss()(model(_raw_batch()), torch.randn(4, 7))
    loss.backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for name, parameter in model.named_parameters()
        if name in before
    )
    optimizer.step()
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in before
    )


def test_loss_aggregation_uses_numerators_and_denominators() -> None:
    module = ContrastiveModule(_shallow(), projection_dim=8).eval()
    objective = ContrastiveObjective(temperature=0.2)
    results = [
        objective(module(_raw_batch(size), _raw_batch(size)))
        for size in (2, 4)
    ]
    aggregate = aggregate_contrastive_loss_results(results)
    expected = sum(
        float(result.loss.numerator.detach()) for result in results
    ) / sum(float(result.loss.denominator.detach()) for result in results)
    assert aggregate["contrastive_loss"] == pytest.approx(expected)


def test_contrastive_contract_has_no_labels_or_full_data_lookup() -> None:
    signatures = (
        inspect.signature(ContrastiveModule.forward),
        inspect.signature(EEGAugmentationPipeline.forward),
        inspect.signature(ContrastiveFoldData.from_indexed_source),
    )
    assert all(
        "labels" not in signature.parameters
        and "y" not in signature.parameters
        for signature in signatures
    )
    source = inspect.getsource(ContrastiveFoldData)
    assert "self.data" not in source
