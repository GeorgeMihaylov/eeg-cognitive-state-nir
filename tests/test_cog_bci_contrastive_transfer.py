from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from bench.experiments.cog_bci_contrastive_transfer import (
    COGBCIContrastiveTransferRunner,
    EXPECTED_TASK_FAMILIES,
    assess_collapse,
    classification_metrics,
    create_pretraining_split,
    embedding_diagnostics,
    transfer_decision,
    validate_encoder_manifest_for_downstream,
    validate_unlabelled_pretraining_columns,
)
from cogstate.model_zoo.DL.adapter import seed_torch
from cogstate.model_zoo.DL.contrastive import (
    ContrastiveFoldData,
    ContrastiveModule,
    ContrastiveObjective,
    EEGAugmentationPipeline,
    export_encoder_checkpoint,
    load_encoder_checkpoint,
)
from cogstate.model_zoo.DL.eegnet import TorchEEGNetClassifier


def _frame() -> pd.DataFrame:
    rows = []
    families = sorted(EXPECTED_TASK_FAMILIES)
    for subject_index in range(29):
        for window_index, family in enumerate(families):
            rows.append({
                "sample_id": f"sample-{subject_index:02d}-{window_index}",
                "subject_id": f"sub-{subject_index + 1:02d}",
                "record_id": f"record-{subject_index:02d}-{window_index}",
                "record_group_id": f"record-{subject_index:02d}-{window_index}",
                "task_family": family,
            })
    return pd.DataFrame(rows)


def _eegnet(num_classes: int = 5) -> TorchEEGNetClassifier:
    return TorchEEGNetClassifier(
        n_channels=4,
        n_times=64,
        num_classes=num_classes,
        temporal_kernel_samples=15,
        separable_kernel_samples=7,
        f1=2,
        depth_multiplier=1,
        f2=2,
        pool1=2,
        pool2=2,
        dropout=0.0,
    )


def _encoder_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    }


def _head_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("classifier.")
    }


def _changed(
    before: dict[str, torch.Tensor], model: nn.Module
) -> bool:
    current = dict(model.named_parameters())
    return any(not torch.equal(value, current[name]) for name, value in before.items())


def test_pretraining_rejects_target_and_behavioural_columns() -> None:
    for column in ("label_q5", "target", "KSS", "RSME", "n_back_level"):
        frame = _frame()
        frame[column] = 0
        with pytest.raises(ValueError, match="target columns"):
            validate_unlabelled_pretraining_columns(frame)


def test_all_task_families_are_valid_unlabelled_metadata() -> None:
    frame = _frame()
    validate_unlabelled_pretraining_columns(frame)
    assert set(frame["task_family"]) == EXPECTED_TASK_FAMILIES


def test_pretraining_split_is_deterministic_24_by_5_and_subject_disjoint() -> None:
    first = create_pretraining_split(_frame(), seed=42, validation_subjects=5)
    second = create_pretraining_split(_frame(), seed=42, validation_subjects=5)
    assert first == second
    assert first["training_subject_count"] == 24
    assert first["validation_subject_count"] == 5
    assert first["subject_overlap_count"] == 0
    assert not (
        set(first["training_subject_ids"])
        & set(first["validation_subject_ids"])
    )
    assert len(first["split_hash"]) == 64


def test_pretraining_split_keeps_every_subject_window_together() -> None:
    frame = _frame()
    split = create_pretraining_split(frame, seed=42, validation_subjects=5)
    train_subjects = set(
        frame.iloc[split["training_indices"]]["subject_id"]
    )
    validation_subjects = set(
        frame.iloc[split["validation_indices"]]["subject_id"]
    )
    assert train_subjects == set(split["training_subject_ids"])
    assert validation_subjects == set(split["validation_subject_ids"])
    assert train_subjects.isdisjoint(validation_subjects)


def test_contrastive_scope_accepts_empty_optional_forbidden_partitions() -> None:
    features = np.zeros((5, 1, 4, 64), dtype=np.float32)
    scope = ContrastiveFoldData.from_indexed_source(
        features=features,
        sample_ids=[f"sample-{index}" for index in range(5)],
        record_group_ids=[f"record-{index}" for index in range(5)],
        subject_ids=["validation"] * 4 + ["forbidden"],
        training_indices=(0, 1, 2, 3),
        inner_validation_indices=(),
        outer_test_indices=(4,),
        target_final_evaluation_indices=(),
        fold_id="validation-only",
    )
    batch = next(iter(scope.training_loader(batch_size=2, shuffle=False)))
    assert batch.inputs.shape == (2, 1, 4, 64)


def test_two_views_are_distinct_storage_from_same_source_batch() -> None:
    batch = torch.randn(4, 1, 4, 64)
    pipeline = EEGAugmentationPipeline.from_config({
        "gaussian_noise": {
            "enabled": True,
            "probability": 1.0,
            "std": 0.01,
        }
    })
    first, second = pipeline.two_views(
        batch, generator=torch.Generator().manual_seed(42)
    )
    assert first.data_ptr() != second.data_ptr()
    assert first.shape == second.shape == batch.shape
    assert not torch.equal(first, second)


def test_augmentations_are_deterministic_for_fixed_seed() -> None:
    batch = torch.randn(4, 1, 4, 64)
    pipeline = EEGAugmentationPipeline.from_config({
        "temporal_shift": {
            "enabled": True,
            "probability": 1.0,
            "maximum_fraction": 0.1,
        }
    })
    first = pipeline(
        batch, generator=torch.Generator().manual_seed(11)
    )
    second = pipeline(
        batch, generator=torch.Generator().manual_seed(11)
    )
    torch.testing.assert_close(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_generator_matches_cuda_zero_inputs() -> None:
    batch = torch.randn(2, 1, 4, 64, device="cuda:0")
    pipeline = EEGAugmentationPipeline.from_config({
        "gaussian_noise": {
            "enabled": True,
            "probability": 1.0,
            "std": 0.01,
        }
    }).to("cuda:0")
    output = pipeline(
        batch,
        generator=EEGAugmentationPipeline.make_generator(42, device="cuda:0"),
    )
    assert output.device == batch.device
    assert torch.isfinite(output).all()


def test_projection_and_nt_xent_are_finite_with_encoder_gradients() -> None:
    module = ContrastiveModule(
        _eegnet(), projection_dim=8, projection_hidden_dim=6
    )
    batch = torch.randn(4, 1, 4, 64)
    result = ContrastiveObjective(temperature=0.1)(module(batch, batch))
    assert torch.isfinite(result.contrastive_loss)
    result.contrastive_loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in module.encoder_model.parameters()
        if parameter.requires_grad
    )
    assert all(
        parameter.grad is not None
        for parameter in module.projection_head.parameters()
    )


def test_projection_shape_is_model_derived() -> None:
    module = ContrastiveModule(_eegnet(), projection_dim=9)
    output = module(
        torch.randn(3, 1, 4, 64),
        torch.randn(3, 1, 4, 64),
    )
    assert output.first_latent.shape == (3, module.latent_dim)
    assert output.first_projection.shape == (3, 9)


def test_embedding_diagnostics_are_finite_and_detect_noncollapse() -> None:
    embeddings = torch.nn.functional.normalize(
        torch.randn(32, 8), dim=1
    )
    result = embedding_diagnostics(
        embeddings,
        positive_similarity=0.7,
        negative_similarity=0.1,
    )
    assert result["feature_std_mean"] > 0
    assert result["effective_rank"] > 1
    assert result["positive_negative_gap"] == pytest.approx(0.6)
    assert all(np.isfinite(list(result.values())))


def test_collapse_audit_detects_zero_variance() -> None:
    row = {
        "validation_feature_std_mean": 0.0,
        "validation_identical_embedding_fraction": 1.0,
        "validation_embedding_norm_mean": 1.0,
        "validation_positive_negative_gap": 0.0,
    }
    result = assess_collapse([row])
    assert result["fatal"] is True
    assert "zero_embedding_variance" in result["reasons"]


def test_encoder_only_checkpoint_excludes_head_and_loads(tmp_path: Path) -> None:
    source = _eegnet(num_classes=3)
    checkpoint = export_encoder_checkpoint(
        source,
        tmp_path / "encoder.pt",
        metadata={
            "input_shape": [1, 4, 64],
            "channel_order": ["a", "b", "c", "d"],
        },
    )
    payload = torch.load(checkpoint, weights_only=False)
    assert not any(
        key.startswith("classifier.")
        for key in payload["encoder_state_dict"]
    )
    target = _eegnet(num_classes=5)
    head_before = deepcopy(target.classifier.state_dict())
    load_encoder_checkpoint(target, checkpoint)
    assert target(torch.randn(2, 1, 4, 64)).shape == (2, 5)
    assert all(
        torch.equal(value, target.classifier.state_dict()[name])
        for name, value in head_before.items()
    )


def test_incompatible_encoder_architecture_is_rejected(tmp_path: Path) -> None:
    checkpoint = export_encoder_checkpoint(
        _eegnet(), tmp_path / "encoder.pt"
    )
    incompatible = TorchEEGNetClassifier(
        n_channels=4,
        n_times=128,
        num_classes=5,
        temporal_kernel_samples=15,
        separable_kernel_samples=7,
        f1=2,
        depth_multiplier=1,
        f2=2,
        pool1=2,
        pool2=2,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_encoder_checkpoint(incompatible, checkpoint)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_shape", [1, 4, 65], "input shape"),
        ("channel_order", ["a", "b", "d", "c"], "channel-order"),
        ("latent_dim", 0, "latent_dim"),
    ],
)
def test_encoder_manifest_contract_rejects_mismatch(
    field: str, value: object, message: str
) -> None:
    manifest = {
        "input_shape": [1, 4, 64],
        "channel_order": ["a", "b", "c", "d"],
        "latent_dim": 32,
    }
    manifest[field] = value
    with pytest.raises(ValueError, match=message):
        validate_encoder_manifest_for_downstream(
            manifest,
            input_shape=(1, 4, 64),
            channel_order=("a", "b", "c", "d"),
        )


def test_head_only_updates_only_five_output_head() -> None:
    seed_torch(42)
    model = _eegnet(num_classes=5)
    model.freeze_encoder()
    encoder_before = _encoder_parameters(model)
    head_before = _head_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    optimizer.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(
        model(torch.randn(8, 1, 4, 64)),
        torch.arange(8) % 5,
    ).backward()
    optimizer.step()
    assert not _changed(encoder_before, model)
    assert _changed(head_before, model)
    assert model(torch.randn(2, 1, 4, 64)).shape == (2, 5)


def test_full_model_updates_encoder_and_head() -> None:
    seed_torch(42)
    model = _eegnet(num_classes=5)
    model.unfreeze_encoder()
    encoder_before = _encoder_parameters(model)
    head_before = _head_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(
        model(torch.randn(8, 1, 4, 64)),
        torch.arange(8) % 5,
    ).backward()
    optimizer.step()
    assert _changed(encoder_before, model)
    assert _changed(head_before, model)


def test_random_initialization_path_does_not_load_checkpoint() -> None:
    source = inspect.getsource(
        COGBCIContrastiveTransferRunner._run_downstream
    )
    assert 'if mode != "random_init":' in source
    assert source.index('if mode != "random_init":') < source.index(
        "load_encoder_checkpoint"
    )


def test_probabilities_and_five_class_metrics_are_valid() -> None:
    probabilities = np.asarray([
        [0.7, 0.1, 0.1, 0.05, 0.05],
        [0.1, 0.7, 0.1, 0.05, 0.05],
        [0.1, 0.1, 0.7, 0.05, 0.05],
        [0.1, 0.1, 0.1, 0.65, 0.05],
        [0.1, 0.1, 0.1, 0.05, 0.65],
    ])
    truth = np.arange(5)
    metrics = classification_metrics(
        truth, probabilities.argmax(axis=1), probabilities
    )
    assert probabilities.shape == (5, 5)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert metrics["accuracy"] == 1.0


def test_preregistration_is_written_before_outer_test_inference() -> None:
    source = inspect.getsource(
        COGBCIContrastiveTransferRunner._run_downstream
    )
    assert source.index("transfer_screening_preregistration.json") < source.index(
        "adapter.predict_proba"
    )


def test_downstream_resume_precedes_retraining() -> None:
    source = inspect.getsource(
        COGBCIContrastiveTransferRunner._run_downstream
    )
    assert "resumed = model_path.is_file()" in source
    assert "if resumed:" in source
    assert source.index("adapter.load(model_path)") < source.index(
        "adapter.fit("
    )


def test_pretraining_resume_validates_split_and_checkpoint_hash() -> None:
    source = inspect.getsource(
        COGBCIContrastiveTransferRunner._run_pretraining
    )
    assert "pretraining_subject_split_hash" in source
    assert "_sha256_file(checkpoint_path)" in source
    assert 'summary["resumed"] = True' in source


def test_decision_rule_is_deterministic_and_can_proceed() -> None:
    metrics = {
        "random_init": {"macro_f1": 0.20, "balanced_accuracy": 0.22},
        "head_only": {"macro_f1": 0.211, "balanced_accuracy": 0.216},
        "full_model": {"macro_f1": 0.205, "balanced_accuracy": 0.23},
    }
    kwargs = {
        "collapse_fatal": False,
        "checkpoint_valid": True,
        "leakage_safe": True,
        "macro_f1_gain": 0.01,
        "balanced_accuracy_tolerance": 0.005,
        "strong_macro_f1_gain": 0.02,
        "strong_balanced_accuracy_gain": 0.01,
    }
    first = transfer_decision(metrics, **kwargs)
    second = transfer_decision(metrics, **kwargs)
    assert first == second
    assert first["decision"] == "proceed"
    assert first["qualifying_modes"] == ["head_only"]


def test_decision_rule_blocks_collapse() -> None:
    result = transfer_decision(
        {
            "random_init": {"macro_f1": 0.2, "balanced_accuracy": 0.2},
            "full_model": {"macro_f1": 0.4, "balanced_accuracy": 0.4},
        },
        collapse_fatal=True,
        checkpoint_valid=True,
        leakage_safe=True,
        macro_f1_gain=0.01,
        balanced_accuracy_tolerance=0.005,
        strong_macro_f1_gain=0.02,
        strong_balanced_accuracy_gain=0.01,
    )
    assert result["decision"] == "do_not_proceed"


def test_config_is_relative_and_fixed_to_one_fold_seed(tmp_path: Path) -> None:
    config = json.loads(
        Path(
            "experiments/cog_bci/contrastive_eegnet_transfer_screening.json"
        ).read_text(encoding="utf-8")
    )
    runner = COGBCIContrastiveTransferRunner(
        config, repository_root=tmp_path
    )
    assert runner.config["downstream"]["fold"] == 1
    assert runner.config["downstream"]["seed"] == 42
    invalid = deepcopy(config)
    invalid["output_dir"] = "F:/absolute/output"
    with pytest.raises(ValueError, match="relative"):
        COGBCIContrastiveTransferRunner(
            invalid, repository_root=tmp_path
        )
