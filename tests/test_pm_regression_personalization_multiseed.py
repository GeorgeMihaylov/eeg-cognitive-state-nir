from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from bench.experiments.pm_regression_personalization import (
    CANONICAL_TARGETS,
    PMRegressionPersonalizationExperiment,
)
from bench.experiments.pm_regression_personalization_multiseed import (
    METRICS,
    METHODS,
    PMRegressionPersonalizationMultiseedExperiment,
    _canonical_hash,
    _paired_comparisons,
    build_multiseed_aggregates,
    build_stability_summary,
    build_target_summary,
    load_pm_multiseed_spec,
    resolve_seed_base_config,
    resolve_seed_personalization_config,
)
from bench.experiments.user_calibration import _state_digest
from cogstate.model_zoo.DL.adapter import TorchClassificationAdapter
from cogstate.model_zoo.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = (
    ROOT
    / "experiments/calibration/"
    "pm_regression_personalization_multiseed_20pct.yaml"
)
SMOKE_CONFIG = (
    ROOT
    / "experiments/calibration/"
    "pm_regression_personalization_multiseed_cuda_smoke.yaml"
)
BASE_CONFIG = (
    ROOT
    / "experiments/calibration/"
    "pm_regression_personalization_20pct_base.yaml"
)


def _subject_frame() -> pd.DataFrame:
    rows = []
    for subject_index, source in enumerate(("gpn_data", "Old_EEG")):
        for seed_index, seed in enumerate((7, 42, 2026)):
            for method_index, method in enumerate(METHODS):
                gain = (
                    0.0
                    if method == "zero_shot"
                    else 0.001 * (method_index + seed_index + subject_index)
                )
                row = {
                    "subject_id": f"s{subject_index}",
                    "source": source,
                    "outer_fold": f"fold_0{subject_index + 1}",
                    "split_seed": 42,
                    "model_seed": seed,
                    "method": method,
                    "status": "completed",
                    "targets_mae_improved_count": 0 if method == "zero_shot" else 5,
                    "targets_r2_improved_count": 0 if method == "zero_shot" else 4,
                    "targets_spearman_improved_count": (
                        0 if method == "zero_shot" else 4
                    ),
                }
                for metric in METRICS:
                    before = 0.2
                    row[f"{metric}_before"] = before
                    row[f"{metric}_gain"] = gain
                    if metric in {
                        "macro_mae", "macro_rmse", "macro_abs_bias"
                    }:
                        row[f"{metric}_after"] = before - gain
                    else:
                        row[f"{metric}_after"] = before + gain
                for target in CANONICAL_TARGETS:
                    for metric in (
                        "mae", "rmse", "r2", "pearson",
                        "spearman", "abs_bias",
                    ):
                        row[f"{target}_{metric}_before"] = 0.2
                        row[f"{target}_{metric}_gain"] = gain
                        if metric in {"mae", "rmse", "abs_bias"}:
                            row[f"{target}_{metric}_after"] = 0.2 - gain
                        else:
                            row[f"{target}_{metric}_after"] = 0.2 + gain
                rows.append(row)
    return pd.DataFrame(rows)


def _split_frames() -> dict[int, dict[str, pd.DataFrame]]:
    frames = {}
    for seed in (7, 42, 2026):
        rows = []
        for subject in ("s0", "s1"):
            for method in METHODS:
                rows.append({
                    "outer_fold": "fold_01",
                    "subject_id": subject,
                    "method": method,
                    "outer_train_subject_hash": "outer",
                    "inner_train_subject_hash": "inner-train",
                    "inner_validation_subject_hash": "inner-validation",
                    "calibration_sample_hash": f"cal-{subject}",
                    "adaptation_train_sample_hash": f"train-{subject}",
                    "adaptation_validation_sample_hash": f"val-{subject}",
                    "evaluation_sample_hash": f"eval-{subject}",
                    "preprocessor_hash": "preprocessor",
                    "calibration_evaluation_overlap": 0,
                    "adaptation_train_validation_overlap": 0,
                    "adaptation_evaluation_overlap": 0,
                    "duplicate_sample_ids": 0,
                    "target_in_global_inner_train": 0,
                    "target_in_global_inner_validation": 0,
                })
        frames[seed] = {"splits": pd.DataFrame(rows)}
    return frames


def _adapter(seed: int) -> TorchClassificationAdapter:
    adapter = build_model(
        model_name="torch_mlp",
        task_type="regression",
        input_shape=(6,),
        num_outputs=7,
        params={
            "hidden_dims": [8],
            "dropout": 0.0,
            "regression_loss": "mse",
            "batch_size": 16,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": seed,
        },
    )
    assert isinstance(adapter, TorchClassificationAdapter)
    return adapter


def test_main_config_has_fixed_split_and_three_model_seeds() -> None:
    spec = load_pm_multiseed_spec(MAIN_CONFIG)
    assert spec["experiment"]["split_seed"] == 42
    assert spec["experiment"]["model_seeds"] == [7, 42, 2026]


def test_smoke_config_uses_only_new_seeds() -> None:
    spec = load_pm_multiseed_spec(SMOKE_CONFIG)
    assert spec["experiment"]["model_seeds"] == [7, 2026]


def test_multiseed_methods_exclude_simple_calibrators() -> None:
    spec = load_pm_multiseed_spec(MAIN_CONFIG)
    assert tuple(spec["calibration"]["methods"]) == METHODS
    assert "bias_correction" not in METHODS
    assert "affine_calibration" not in METHODS


def test_canonical_target_order_is_preserved() -> None:
    spec = load_pm_multiseed_spec(MAIN_CONFIG)
    assert tuple(spec["targets"]) == CANONICAL_TARGETS


def test_resolved_base_separates_model_and_split_seeds(tmp_path) -> None:
    template = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    resolved = resolve_seed_base_config(
        template,
        model_seed=7,
        split_seed=42,
        output_dir=tmp_path,
    )
    assert resolved["models"]["torch_mlp"]["params"]["random_state"] == 7
    assert resolved["validation"]["random_state"] == 42
    assert resolved["evaluation"]["random_state"] == 42
    assert resolved["task_config"]["random_state"] == 42


def test_resolved_base_does_not_change_architecture(tmp_path) -> None:
    template = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    resolved = resolve_seed_base_config(
        template, model_seed=2026, split_seed=42, output_dir=tmp_path
    )
    original = template["models"]["torch_mlp"]["params"]
    current = resolved["models"]["torch_mlp"]["params"]
    for key in (
        "hidden_dims", "dropout", "activation", "regression_loss",
        "batch_size", "max_epochs", "learning_rate", "weight_decay",
    ):
        assert current[key] == original[key]


def test_smoke_epoch_override_is_local(tmp_path) -> None:
    template = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    resolved = resolve_seed_base_config(
        template,
        model_seed=7,
        split_seed=42,
        output_dir=tmp_path,
        maximum_epochs=2,
        outer_folds=[1],
    )
    assert resolved["models"]["torch_mlp"]["params"]["max_epochs"] == 2
    assert resolved["evaluation"]["folds"] == [1]
    assert template["models"]["torch_mlp"]["params"]["max_epochs"] == 8


def test_seed_personalization_config_passes_both_seeds(tmp_path) -> None:
    document = load_pm_multiseed_spec(MAIN_CONFIG)
    resolved = resolve_seed_personalization_config(
        document,
        model_seed=2026,
        split_seed=42,
        base_config_path=tmp_path / "base.yaml",
        output_dir=tmp_path / "out",
    )
    assert resolved["experiment"]["model_seed"] == 2026
    assert resolved["experiment"]["split_seed"] == 42


def test_seed_configs_have_distinct_hashes(tmp_path) -> None:
    document = load_pm_multiseed_spec(MAIN_CONFIG)
    first = resolve_seed_personalization_config(
        document, model_seed=7, split_seed=42,
        base_config_path=tmp_path / "7.yaml", output_dir=tmp_path / "7",
    )
    second = resolve_seed_personalization_config(
        document, model_seed=42, split_seed=42,
        base_config_path=tmp_path / "42.yaml", output_dir=tmp_path / "42",
    )
    assert _canonical_hash(first) != _canonical_hash(second)


def test_same_model_seed_reproduces_initial_checkpoint() -> None:
    assert _state_digest(_adapter(7)) == _state_digest(_adapter(7))


def test_different_model_seeds_change_initial_checkpoint() -> None:
    assert _state_digest(_adapter(7)) != _state_digest(_adapter(2026))


def test_same_model_seed_reproduces_predictions() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    y = rng.normal(size=(40, 7)).astype(np.float32)
    predictions = []
    for _ in range(2):
        adapter = _adapter(7)
        adapter.fit(X, y)
        predictions.append(adapter.predict(X[:8]))
    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_regression_output_shape_is_n_by_seven() -> None:
    adapter = _adapter(7)
    assert tuple(adapter.model(torch.zeros(4, 6)).shape) == (4, 7)


def test_per_seed_aggregation_keeps_three_seed_rows() -> None:
    per_seed, _, _, _ = build_multiseed_aggregates(
        _subject_frame(), model_seeds=(7, 42, 2026), bootstrap_samples=20
    )
    assert set(per_seed["model_seed"]) == {7, 42, 2026}
    assert set(per_seed["method"]) == set(METHODS)


def test_multiseed_aggregation_uses_two_subjects_not_six() -> None:
    _, multiseed, _, means = build_multiseed_aggregates(
        _subject_frame(), model_seeds=(7, 42, 2026), bootstrap_samples=20
    )
    assert (multiseed["n_subjects"] == 2).all()
    assert means.groupby("method")["subject_id"].nunique().eq(2).all()


def test_bootstrap_is_after_seed_averaging() -> None:
    _, multiseed, _, _ = build_multiseed_aggregates(
        _subject_frame(), model_seeds=(7, 42, 2026), bootstrap_samples=37
    )
    assert (multiseed["bootstrap_resamples"] == 37).all()
    assert (multiseed["n_subjects"] == 2).all()


def test_subject_incomplete_seed_set_is_excluded() -> None:
    frame = _subject_frame()
    frame = frame.loc[
        ~(
            (frame["subject_id"] == "s0")
            & (frame["method"] == "full_model")
            & (frame["model_seed"] == 2026)
        )
    ]
    _, multiseed, _, _ = build_multiseed_aggregates(
        frame, model_seeds=(7, 42, 2026), bootstrap_samples=20
    )
    full = multiseed.loc[multiseed["method"] == "full_model"]
    assert (full["n_subjects"] == 1).all()


def test_stability_counts_positive_seeds() -> None:
    stability = build_stability_summary(
        _subject_frame(), model_seeds=(7, 42, 2026)
    )
    subject = stability.loc[
        (stability["record_type"] == "subject")
        & (stability["method"] == "full_model")
        & (stability["metric"] == "macro_mae")
    ]
    assert (subject["positive_seeds_count"] >= 2).all()


def test_stability_has_subject_and_aggregate_records() -> None:
    stability = build_stability_summary(
        _subject_frame(), model_seeds=(7, 42, 2026)
    )
    assert set(stability["record_type"]) == {"subject", "aggregate"}


def test_target_summary_contains_all_seven_targets() -> None:
    summary = build_target_summary(
        _subject_frame(),
        model_seeds=(7, 42, 2026),
        bootstrap_samples=20,
        bootstrap_seed=42,
    )
    assert set(summary["target_name"]) == set(CANONICAL_TARGETS)


def test_target_summary_contains_per_target_gains() -> None:
    summary = build_target_summary(
        _subject_frame(),
        model_seeds=(7, 42, 2026),
        bootstrap_samples=20,
        bootstrap_seed=42,
    )
    assert {
        "mae_gain", "rmse_gain", "r2_gain", "pearson_gain",
        "spearman_gain", "abs_bias_gain",
    }.issubset(summary.columns)


def test_mae_pair_difference_has_error_reduction_direction() -> None:
    paired = _paired_comparisons(
        _subject_frame(),
        grouping=("model_seed",),
        bootstrap_samples=20,
        bootstrap_seed=42,
    )
    full = paired.loc[
        (paired["method"] == "full_model")
        & (paired["reference_method"] == "zero_shot")
        & (paired["metric"] == "macro_mae_gain")
    ]
    assert (full["mean_difference"] > 0).all()


def test_r2_and_spearman_pair_directions_reward_increase() -> None:
    paired = _paired_comparisons(
        _subject_frame(),
        grouping=(),
        bootstrap_samples=20,
        bootstrap_seed=42,
    )
    selected = paired.loc[
        (paired["method"] == "full_model")
        & paired["metric"].isin(("macro_r2_gain", "macro_spearman_gain"))
    ]
    assert (selected["mean_difference"] > 0).all()


def test_split_consistency_accepts_identical_hashes() -> None:
    experiment = PMRegressionPersonalizationMultiseedExperiment(MAIN_CONFIG)
    audit = experiment._split_consistency(_split_frames())
    assert audit["all_split_hashes_consistent"].all()


def test_split_consistency_rejects_changed_calibration_hash() -> None:
    experiment = PMRegressionPersonalizationMultiseedExperiment(MAIN_CONFIG)
    frames = _split_frames()
    changed = frames[2026]["splits"].copy()
    changed.loc[
        (changed["subject_id"] == "s0")
        & (changed["method"] == "full_model"),
        "calibration_sample_hash",
    ] = "changed"
    frames[2026]["splits"] = changed
    with pytest.raises(RuntimeError, match="hashes differ"):
        experiment._split_consistency(frames)


def test_split_consistency_rejects_overlap() -> None:
    experiment = PMRegressionPersonalizationMultiseedExperiment(MAIN_CONFIG)
    frames = _split_frames()
    changed = frames[7]["splits"].copy()
    changed.loc[
        changed["method"] == "full_model",
        "calibration_evaluation_overlap",
    ] = 1
    frames[7]["splits"] = changed
    with pytest.raises(RuntimeError, match="overlap"):
        experiment._split_consistency(frames)


def test_preprocessor_hash_is_part_of_consistency_contract() -> None:
    experiment = PMRegressionPersonalizationMultiseedExperiment(MAIN_CONFIG)
    frames = _split_frames()
    frames[42]["splits"].loc[
        frames[42]["splits"]["method"] == "full_model",
        "preprocessor_hash",
    ] = "other"
    with pytest.raises(RuntimeError, match="hashes differ"):
        experiment._split_consistency(frames)


def test_old_single_seed_config_still_initializes() -> None:
    config = (
        ROOT
        / "experiments/calibration/pm_regression_personalization_20pct.yaml"
    )
    experiment = PMRegressionPersonalizationExperiment(config)
    assert experiment.document["experiment"]["model_seed"] == 42


def test_invalid_model_seed_set_is_rejected(tmp_path) -> None:
    document = yaml.safe_load(MAIN_CONFIG.read_text(encoding="utf-8"))
    document["experiment"]["model_seeds"] = [1, 2, 3]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="model_seeds"):
        load_pm_multiseed_spec(path)


def test_invalid_split_seed_is_rejected(tmp_path) -> None:
    document = yaml.safe_load(MAIN_CONFIG.read_text(encoding="utf-8"))
    document["experiment"]["split_seed"] = 7
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="split_seed"):
        load_pm_multiseed_spec(path)


def test_config_resolution_does_not_mutate_template(tmp_path) -> None:
    template = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    original = deepcopy(template)
    resolve_seed_base_config(
        template, model_seed=7, split_seed=42, output_dir=tmp_path
    )
    assert template == original
