from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bench.experiments.user_calibration import _state_digest
from bench.experiments.user_calibration_multiseed import (
    UserCalibrationMultiseedExperiment,
    _canonical_hash,
    _ordered_hash,
    build_multiseed_aggregates,
    build_stability_summary,
    build_threshold_summary,
    load_multiseed_calibration_spec,
    resolve_seed_base_config,
    resolve_seed_calibration_config,
)
from cogstate.model_zoo.DL.adapter import TorchClassificationAdapter
from cogstate.model_zoo.factory import build_model


ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = (
    ROOT
    / "experiments"
    / "calibration"
    / "label_q5_finetuning_multiseed_20pct.yaml"
)
BASE_CONFIG = (
    ROOT
    / "experiments"
    / "calibration"
    / "label_q5_finetuning_multiseed_20pct_base.yaml"
)


def _subject_metrics() -> pd.DataFrame:
    rows = []
    for subject_index, subject in enumerate(("s1", "s2")):
        source = "gpn_data" if subject == "s1" else "Old_EEG"
        for seed_index, seed in enumerate((7, 42, 2026)):
            zero = 0.25 + subject_index * 0.02 + seed_index * 0.001
            for method, gain in (
                ("zero_shot", 0.0),
                ("head_only", 0.01 if subject == "s1" else -0.002),
                ("full_model", 0.02 if seed != 42 else 0.01),
            ):
                row = {
                    "subject_id": subject,
                    "source": source,
                    "source_group": source,
                    "outer_fold": f"fold_0{subject_index + 1}",
                    "split_seed": 42,
                    "model_seed": seed,
                    "budget": 0.2,
                    "method": method,
                    "status": "completed",
                }
                for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                    row[f"{metric}_after"] = zero + gain
                    row[f"{metric}_gain"] = gain
                rows.append(row)
    return pd.DataFrame(rows)


def _adapter(seed: int) -> tuple[TorchClassificationAdapter, np.ndarray]:
    rng = np.random.default_rng(91)
    X = rng.normal(size=(60, 5)).astype(np.float32)
    y = np.arange(60, dtype=np.int64) % 5
    adapter = build_model(
        model_name="torch_mlp",
        task_type="classification",
        input_shape=(5,),
        num_outputs=5,
        params={
            "hidden_dims": [10],
            "dropout": 0.0,
            "batch_size": 15,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "validation_size": 0.2,
            "device": "cpu",
            "random_state": seed,
        },
    )
    assert isinstance(adapter, TorchClassificationAdapter)
    adapter.fit(X, y)
    return adapter, X


def test_multiseed_config_separates_split_and_model_seeds() -> None:
    document = load_multiseed_calibration_spec(FULL_CONFIG)
    assert document["experiment"]["split_seed"] == 42
    assert document["experiment"]["model_seeds"] == [7, 42, 2026]
    assert document["calibration"]["budgets_fraction"] == [0.2]


def test_resolved_base_keeps_splits_and_changes_only_model_randomness() -> None:
    import yaml

    template = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    seed_7 = resolve_seed_base_config(
        template,
        model_seed=7,
        split_seed=42,
        output_dir="seed7",
    )
    seed_2026 = resolve_seed_base_config(
        template,
        model_seed=2026,
        split_seed=42,
        output_dir="seed2026",
    )
    assert seed_7["validation"]["random_state"] == 42
    assert seed_7["evaluation"]["random_state"] == 42
    assert seed_7["task_config"]["random_state"] == 42
    assert seed_7["models"]["torch_mlp"]["params"]["random_state"] == 7
    assert seed_2026["models"]["torch_mlp"]["params"]["random_state"] == 2026
    comparable_7 = {**seed_7, "output_dir": "same"}
    comparable_2026 = {**seed_2026, "output_dir": "same"}
    comparable_7["models"]["torch_mlp"]["params"]["random_state"] = 0
    comparable_2026["models"]["torch_mlp"]["params"]["random_state"] = 0
    assert comparable_7 == comparable_2026


def test_single_seed_document_remains_legacy_compatible() -> None:
    document = load_multiseed_calibration_spec(FULL_CONFIG)
    resolved = resolve_seed_calibration_config(
        document,
        model_seed=7,
        split_seed=42,
        base_config_path="base.yaml",
        output_dir="output",
    )
    assert resolved["experiment"]["type"] == "user_calibration"
    assert resolved["experiment"]["model_seed"] == 7
    assert resolved["experiment"]["split_seed"] == 42
    assert resolved["calibration"]["defaults"]["random_state"] == 7


def test_aggregation_averages_seeds_before_treating_subjects_as_independent() -> None:
    frame = _subject_metrics()
    per_seed, multiseed, source, subject_means = build_multiseed_aggregates(
        frame,
        model_seeds=(7, 42, 2026),
        bootstrap_samples=100,
        bootstrap_seed=42,
    )
    assert len(per_seed) == 3 * 3 * 3
    assert subject_means.subject_id.nunique() == 2
    accuracy = multiseed.loc[
        (multiseed.method == "full_model")
        & (multiseed.metric == "accuracy")
    ].iloc[0]
    assert accuracy.n_subjects == 2
    assert accuracy.mean_gain == np.mean([0.02, 0.01, 0.02])
    assert set(source.source_group) == {"gpn_data", "Old_EEG"}


def test_incomplete_subject_is_not_averaged_over_remaining_seeds() -> None:
    frame = _subject_metrics()
    frame = frame.loc[
        ~(
            (frame.subject_id == "s2")
            & (frame.model_seed == 2026)
            & (frame.method == "full_model")
        )
    ]
    _, multiseed, _, means = build_multiseed_aggregates(
        frame,
        model_seeds=(7, 42, 2026),
        bootstrap_samples=50,
    )
    full = means.loc[means.method == "full_model"]
    assert full.subject_id.tolist() == ["s1"]
    assert multiseed.loc[
        (multiseed.method == "full_model")
        & (multiseed.metric == "accuracy"),
        "n_subjects",
    ].item() == 1


def test_stability_counts_positive_seeds_per_subject() -> None:
    stability = build_stability_summary(
        _subject_metrics(), model_seeds=(7, 42, 2026)
    )
    subject = stability.loc[
        (stability.record_type == "subject")
        & (stability.subject_id == "s1")
        & (stability.method == "full_model")
        & (stability.metric == "macro_f1")
    ].iloc[0]
    assert subject.positive_seeds_count == 3
    assert bool(subject.improved_in_at_least_2_of_3)
    aggregate = stability.loc[
        (stability.record_type == "aggregate")
        & (stability.method == "head_only")
        & (stability.metric == "accuracy")
    ].iloc[0]
    assert aggregate.subjects_improved_at_least_2_of_3 == 1
    assert aggregate.fraction_improved_at_least_2_of_3 == 0.5


def test_threshold_requires_repetition_across_seeds() -> None:
    frame = _subject_metrics()
    mask = (
        (frame.subject_id == "s1")
        & (frame.method == "full_model")
        & (frame.model_seed.isin([7, 42]))
    )
    frame.loc[mask, "accuracy_after"] = 0.8
    summary = build_threshold_summary(frame, model_seeds=(7, 42, 2026))
    full = summary.loc[summary.method == "full_model"].iloc[0]
    assert full.subjects_accuracy_ge_075_any_seed == 1
    assert full.subjects_accuracy_ge_075_at_least_2_seeds == 1
    assert full.subjects_accuracy_ge_075_all_3_seeds == 0


def test_same_model_seed_reproduces_checkpoint_and_predictions() -> None:
    first, X = _adapter(7)
    second, _ = _adapter(7)
    assert _state_digest(first) == _state_digest(second)
    np.testing.assert_allclose(
        first.predict_proba(X), second.predict_proba(X), atol=1e-7
    )


def test_different_model_seeds_produce_different_checkpoints() -> None:
    first, _ = _adapter(7)
    second, _ = _adapter(2026)
    assert _state_digest(first) != _state_digest(second)


def test_split_hash_is_order_sensitive_and_config_hash_includes_model_seeds() -> None:
    assert _ordered_hash(["a", "b"]) == _ordered_hash(["a", "b"])
    assert _ordered_hash(["a", "b"]) != _ordered_hash(["b", "a"])
    document = load_multiseed_calibration_spec(FULL_CONFIG)
    altered = {
        **document,
        "experiment": {
            **document["experiment"],
            "model_seeds": [7, 42],
        },
    }
    assert _canonical_hash(document) != _canonical_hash(altered)


def test_seed42_reuse_is_rejected_without_historical_dataset_fingerprint() -> None:
    experiment = UserCalibrationMultiseedExperiment(FULL_CONFIG)
    compatibility = experiment._seed42_compatibility()
    assert compatibility["checks"]["historical_dataset_fingerprint_available"] is False
    assert compatibility["eligible"] is False
    assert "fingerprint" in compatibility["reason"]
