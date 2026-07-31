from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import pytest
import torch

from bench.meta import (
    FOMAMLConfig,
    FOMAMLError,
    FirstOrderMAML,
    SyntheticClassifier,
    SyntheticEpisodeData,
    audit_production_model_compatibility,
    generate_synthetic_episodes,
    model_state_hash,
    validate_parameter_mapping,
)
from model_zoo.DL.eegnet import TorchEEGNetClassifier
from model_zoo.DL.shallow_convnet import TorchShallowConvNetClassifier


def _generator_config() -> dict[str, int]:
    return {
        "classes": 3,
        "support_per_class": 5,
        "query_per_class": 10,
        "meta_train_episodes": 4,
        "meta_validation_episodes": 2,
        "seed": 42,
    }


def _config(**overrides: object) -> FOMAMLConfig:
    values = {
        "inner_steps": 2,
        "inner_learning_rate": 0.1,
        "meta_learning_rate": 0.01,
        "episodes_per_meta_batch": 2,
        "maximum_meta_steps": 2,
        "gradient_clip_norm": 5.0,
        "device": "cpu",
        "seed": 42,
    }
    values.update(overrides)
    return FOMAMLConfig(**values)


def _learner(**overrides: object):
    torch.manual_seed(42)
    model = SyntheticClassifier()
    episodes, _ = generate_synthetic_episodes(_generator_config())
    return FirstOrderMAML(model, _config(**overrides)), episodes


def test_fast_weights_are_independent_aligned_and_first_order() -> None:
    learner, episodes = _learner()
    base_hash = model_state_hash(learner.model)
    fast = learner.create_fast_weights()
    validate_parameter_mapping(learner.model, fast)
    assert model_state_hash(learner.model) == base_hash
    for name, parameter in learner.model.named_parameters():
        assert fast[name].shape == parameter.shape
        assert fast[name].data_ptr() != parameter.data_ptr()
    adapted = learner.adapt(
        learner.model,
        (episodes[0].support_features, episodes[0].support_targets),
    )
    assert adapted.create_graph is False
    assert adapted.base_unchanged and adapted.storage_independent
    assert all(value.grad_fn is None for value in adapted.fast_weights.values())
    assert model_state_hash(learner.model) == base_hash


def test_inner_adaptation_is_finite_reduces_loss_and_support_changes_weights() -> None:
    learner, episodes = _learner()
    first = learner.adapt(
        learner.model,
        (episodes[0].support_features, episodes[0].support_targets),
    )
    changed_targets = torch.roll(episodes[0].support_targets, shifts=1)
    second = learner.adapt(
        learner.model, (episodes[0].support_features, changed_targets)
    )
    assert all(torch.isfinite(torch.tensor(first.support_losses)))
    assert first.support_losses[-1] < first.support_losses[0]
    assert all(value > 0 for value in first.gradient_norm_per_step)
    assert any(
        not torch.equal(first.fast_weights[name], second.fast_weights[name])
        for name in first.fast_weights
    )


def test_query_cannot_affect_inner_fast_weights() -> None:
    learner, episodes = _learner()
    episode = episodes[0]
    changed_query = replace(
        episode,
        query_features=episode.query_features * -7.0,
        query_targets=torch.roll(episode.query_targets, 1),
    )
    first = learner.adapt(
        learner.model, (episode.support_features, episode.support_targets)
    )
    second = learner.adapt(
        learner.model,
        (changed_query.support_features, changed_query.support_targets),
    )
    assert all(
        torch.equal(first.fast_weights[name], second.fast_weights[name])
        for name in first.fast_weights
    )
    first_query = learner.evaluate(
        first, (episode.query_features, episode.query_targets)
    )[2]
    second_query = learner.evaluate(
        second, (changed_query.query_features, changed_query.query_targets)
    )[2]
    assert any(
        not torch.equal(first_query[name], second_query[name])
        for name in first_query
    )


def test_query_gradient_and_meta_step_are_finite_nonzero_and_delayed() -> None:
    learner, episodes = _learner()
    before = model_state_hash(learner.model)
    batch = learner.compute_meta_batch_gradients(episodes[:2])
    assert batch.meta_gradient_norm > 0
    assert all(torch.isfinite(value).all() for value in batch.mean_gradients.values())
    assert model_state_hash(learner.model) == before
    result = learner.meta_train_step(episodes[:2])
    assert result.parameters_updated > 0
    assert result.base_unchanged_before_step
    assert result.optimizer_state_finite
    assert model_state_hash(learner.model) != before


def test_episode_gradient_aggregation_is_exact_mean() -> None:
    learner, episodes = _learner()
    first = learner.compute_meta_batch_gradients([episodes[0]])
    second = learner.compute_meta_batch_gradients([episodes[1]])
    combined = learner.compute_meta_batch_gradients(episodes[:2])
    for name in combined.mean_gradients:
        expected = (first.mean_gradients[name] + second.mean_gradients[name]) / 2
        assert torch.equal(combined.mean_gradients[name], expected)


def test_gradient_clipping_applies_after_averaging() -> None:
    learner, episodes = _learner(gradient_clip_norm=0.01)
    result = learner.meta_train_step(episodes[:2])
    assert result.meta_gradient_norm_before_clip > 0.01
    assert result.meta_gradient_norm_after_clip <= 0.010001


def test_invalid_parameter_mappings_are_rejected() -> None:
    learner, _ = _learner()
    fast = learner.create_fast_weights()
    missing = OrderedDict(list(fast.items())[1:])
    with pytest.raises(FOMAMLError, match="missing"):
        validate_parameter_mapping(learner.model, missing)
    extra = OrderedDict(fast)
    extra["unexpected"] = torch.zeros(1)
    with pytest.raises(FOMAMLError, match="extra"):
        validate_parameter_mapping(learner.model, extra)
    wrong = OrderedDict(fast)
    first_name = next(iter(wrong))
    wrong[first_name] = wrong[first_name].flatten()[:1]
    with pytest.raises(FOMAMLError, match="shape"):
        validate_parameter_mapping(learner.model, wrong)


def test_unsupported_tasks_device_and_failed_episode_are_explicit() -> None:
    with pytest.raises(ValueError, match="classification only"):
        _config(task_type="regression")
    with pytest.raises(ValueError, match="CPU-only"):
        _config(device="cuda")
    learner, episodes = _learner()
    bad = replace(
        episodes[0],
        support_features=torch.full_like(episodes[0].support_features, float("nan")),
    )
    before = model_state_hash(learner.model)
    with pytest.raises(FOMAMLError, match="NaN"):
        learner.meta_train_step([bad])
    assert model_state_hash(learner.model) == before


def test_zero_meta_gradient_is_rejected() -> None:
    learner, episodes = _learner()
    with torch.no_grad():
        for parameter in learner.model.parameters():
            parameter.zero_()
    zero = replace(
        episodes[0],
        support_features=torch.zeros_like(episodes[0].support_features),
        query_features=torch.zeros_like(episodes[0].query_features),
    )
    with pytest.raises(FOMAMLError, match="zero"):
        learner.compute_meta_batch_gradients([zero])


def test_seed_is_deterministic_for_episodes_and_model_updates() -> None:
    first, episodes_a = _learner()
    second, episodes_b = _learner()
    assert [x.episode.episode_id for x in episodes_a] == [
        x.episode.episode_id for x in episodes_b
    ]
    first.meta_train_step(episodes_a[:2])
    second.meta_train_step(episodes_b[:2])
    assert model_state_hash(first.model) == model_state_hash(second.model)


def _production_models():
    return [
        TorchEEGNetClassifier(
            4, 128, 3, temporal_kernel_samples=16,
            separable_kernel_samples=8, f1=2, depth_multiplier=2,
            f2=4, pool1=2, pool2=2, dropout=0.1,
        ),
        TorchShallowConvNetClassifier(
            4, 128, 3, n_filters=4, temporal_kernel_samples=9,
            pool_size=15, pool_stride=5, dropout=0.1,
        ),
    ]


def test_production_models_are_read_only_audited_and_adaptation_is_blocked() -> None:
    example = torch.zeros(2, 1, 4, 128)
    for model in _production_models():
        before = model_state_hash(model)
        audit = audit_production_model_compatibility(model, example)
        assert audit["functional_eval_forward"]
        assert audit["state_dict_unchanged"]
        assert audit["stateful_buffers_present"]
        assert not audit["adaptation_supported"]
        assert audit["output_shape"] == [2, 3]
        assert model_state_hash(model) == before
        learner = FirstOrderMAML(model, _config())
        with pytest.raises(FOMAMLError, match="buffer policy"):
            learner.adapt(
                model,
                (example, torch.tensor([0, 1])),
            )
