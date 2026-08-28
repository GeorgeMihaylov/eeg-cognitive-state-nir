from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.core.abstract_task import TaskSplit
from bench.experiments.auxiliary_corn_lambda_selection import (
    AUXILIARY_WEIGHTS,
    AuxiliaryCornLambdaSelectionSetupExperiment,
    CategoricalBaselineReference,
    LambdaValidationResult,
    NoEligibleAuxiliaryWeightError,
    build_inner_validation_prediction_frame,
    experiment_contains_training_loop,
    load_auxiliary_corn_lambda_setup_spec,
    load_categorical_baseline_references,
    materialize_categorical_baseline_validation,
    select_auxiliary_weight,
    validation_metrics_from_frame,
)
from bench.experiments.ordinal_transformer import build_ordinal_transformer_experiment
from cogstate.model_zoo import build_model


SPEC = Path("experiments/auxiliary_corn_lambda_selection_setup.yaml")


def _candidate(weight: float, ba: float, severe: float, mae: float):
    return LambdaValidationResult(
        auxiliary_weight=weight,
        balanced_accuracy=ba,
        severe_error_rate=severe,
        ordinal_mae=mae,
        macro_f1=ba,
    )


def test_selection_applies_ba_guard_then_ordinal_order() -> None:
    decision = select_auxiliary_weight(
        {"balanced_accuracy": 0.40},
        [
            _candidate(0.25, 0.395, 0.24, 0.95),
            _candidate(0.5, 0.391, 0.22, 0.94),
            _candidate(1.0, 0.389, 0.18, 0.90),
        ],
    )
    assert decision.selected.auxiliary_weight == 0.5
    assert [item.auxiliary_weight for item in decision.rejected] == [1.0]
    assert decision.to_dict()["outer_test_used"] is False


def test_selection_uses_mae_and_lower_lambda_as_tie_breakers() -> None:
    decision = select_auxiliary_weight(
        {"balanced_accuracy": 0.4},
        [
            _candidate(0.25, 0.4, 0.2, 0.91),
            _candidate(0.5, 0.4, 0.2, 0.90),
            _candidate(1.0, 0.4, 0.2, 0.90),
        ],
    )
    assert decision.selected.auxiliary_weight == 0.5


def test_selection_aborts_when_no_lambda_is_eligible() -> None:
    with pytest.raises(NoEligibleAuxiliaryWeightError, match="No auxiliary-CORN"):
        select_auxiliary_weight(
            {"balanced_accuracy": 0.4},
            [
                _candidate(0.25, 0.38, 0.2, 0.9),
                _candidate(0.5, 0.37, 0.2, 0.9),
                _candidate(1.0, 0.36, 0.2, 0.9),
            ],
        )


def test_spec_builder_and_baseline_index_are_complete() -> None:
    document = load_auxiliary_corn_lambda_setup_spec(SPEC)
    assert tuple(document["auxiliary_weights"]) == AUXILIARY_WEIGHTS
    references = load_categorical_baseline_references(
        document["categorical_baseline_index"]
    )
    assert len(references) == 6
    assert {(item.feature_group, item.seed) for item in references} == {
        (group, seed)
        for group in ("eeg_pow", "eeg_only")
        for seed in (7, 42, 123)
    }
    experiment = build_ordinal_transformer_experiment(SPEC)
    assert isinstance(experiment, AuxiliaryCornLambdaSelectionSetupExperiment)


def test_plan_counts_baselines_and_future_candidate_fits() -> None:
    plan = AuxiliaryCornLambdaSelectionSetupExperiment(SPEC).plan()
    assert plan.baseline_fold_materializations == 30
    assert plan.future_candidate_fold_fits == 90
    assert plan.auxiliary_weights == AUXILIARY_WEIGHTS
    assert plan.folds == (1, 2, 3, 4, 5)
    rendered = AuxiliaryCornLambdaSelectionSetupExperiment.render_plan(plan)
    assert "performs no model fitting" in rendered
    assert "Future candidate fold fits: 90" in rendered


def _synthetic_split() -> TaskSplit:
    n_train = 10
    n_test = 3
    metadata_train = {
        "sequence_id": np.asarray([f"seq-{index}" for index in range(n_train)]),
        "record_group_id": np.asarray([f"group-{index // 2}" for index in range(n_train)]),
        "source": np.asarray(["synthetic"] * n_train),
        "target_sample_id": np.arange(n_train),
        "target_time": np.arange(n_train, dtype=float) * 10,
    }
    return TaskSplit(
        X_train=np.zeros((n_train, 4, 6), dtype=np.float32),
        y_train=np.arange(n_train, dtype=np.int64) % 5,
        X_test=np.zeros((n_test, 4, 6), dtype=np.float32),
        y_test=np.arange(n_test, dtype=np.int64),
        subject_train=np.asarray([f"subject-{index // 2}" for index in range(n_train)]),
        subject_test=np.asarray(["test-a", "test-b", "test-c"]),
        feature_names=[f"feature-{index}" for index in range(6)],
        metadata={"fold": 1, "observation_unit": "sequence"},
        sample_id_train=np.arange(n_train),
        sample_id_test=np.arange(n_test) + 100,
        record_id_train=np.asarray([f"record-{index // 2}" for index in range(n_train)]),
        record_id_test=np.asarray(["test-r0", "test-r1", "test-r2"]),
        row_metadata_train=metadata_train,
        row_metadata_test={},
    )


def test_validation_frame_contains_only_inner_partition() -> None:
    split = _synthetic_split()
    indices = np.asarray([2, 3, 8, 9])
    probabilities = np.asarray(
        [
            [0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.6, 0.1],
        ]
    )
    detailed = {
        "indices": indices,
        "y_true": split.y_train[indices],
        "y_pred": probabilities.argmax(axis=1),
        "class_probabilities": probabilities,
        "head_type": "categorical",
    }
    frame = build_inner_validation_prediction_frame(
        split,
        detailed,
        feature_group="eeg_pow",
        seed=42,
        outer_fold=1,
    )
    assert len(frame) == 4
    assert set(frame["split"]) == {"inner_validation"}
    assert set(frame["sequence_id"]) == {"seq-2", "seq-3", "seq-8", "seq-9"}
    assert "test-a" not in set(frame["subject_id"])
    metrics = validation_metrics_from_frame(frame)
    assert set(("balanced_accuracy", "ordinal_mae", "severe_error_rate")) <= set(metrics)


def test_adapter_rebuilds_and_persists_validation_indices(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4, 6)).astype(np.float32)
    y = np.tile(np.arange(5, dtype=np.int64), 6)
    groups = np.asarray([f"record-{index // 3}" for index in range(30)])
    subjects = np.asarray([f"subject-{index // 5}" for index in range(30)])
    model = build_model(
        "torch_transformer",
        "classification",
        input_shape=(4, 6),
        num_outputs=5,
        params={
            "head_type": "categorical",
            "sequence_length": 4,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "batch_size": 8,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
            "standardize": True,
        },
    )
    model.set_validation_groups(
        groups,
        subject_ids=subjects,
        record_ids=groups,
        outer_test_record_ids=np.asarray(["outer-test"]),
        strategy="group_record",
        group_column="record_group_id",
        validation_size=0.2,
        random_state=42,
    )
    model.fit(X, y)
    original = model.validation_partition_detailed(X, y)
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)

    loaded = build_model(
        "torch_transformer",
        "classification",
        input_shape=(4, 6),
        num_outputs=5,
        params={
            "head_type": "categorical",
            "sequence_length": 4,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "batch_size": 8,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
            "standardize": True,
        },
    )
    loaded.load(checkpoint)
    restored = loaded.validation_partition_detailed(X, y)
    assert np.array_equal(original["indices"], restored["indices"])
    assert np.array_equal(original["y_pred"], restored["y_pred"])
    assert np.allclose(
        original["class_probabilities"], restored["class_probabilities"]
    )


def test_materializer_reconstructs_checkpoint_validation_without_fit(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 4, 6)).astype(np.float32)
    y = np.tile(np.arange(5, dtype=np.int64), 6)
    groups = np.asarray([f"record-{index // 3}" for index in range(30)])
    subjects = np.asarray([f"subject-{index // 5}" for index in range(30)])
    row_metadata = {
        "sequence_id": np.asarray([f"sequence-{index}" for index in range(30)]),
        "record_group_id": groups,
        "source": np.asarray(["synthetic"] * 30),
        "target_sample_id": np.arange(30),
        "target_time": np.arange(30, dtype=float) * 10,
    }
    split = TaskSplit(
        X_train=X,
        y_train=y,
        X_test=np.zeros((5, 4, 6), dtype=np.float32),
        y_test=np.arange(5, dtype=np.int64),
        subject_train=subjects,
        subject_test=np.asarray([f"outer-{index}" for index in range(5)]),
        feature_names=[f"feature-{index}" for index in range(6)],
        metadata={"fold": 1, "observation_unit": "sequence"},
        sample_id_train=np.arange(30),
        sample_id_test=np.arange(5) + 100,
        record_id_train=groups,
        record_id_test=np.asarray([f"outer-record-{index}" for index in range(5)]),
        row_metadata_train=row_metadata,
        row_metadata_test={},
    )
    params = {
        "head_type": "categorical",
        "sequence_length": 4,
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "batch_size": 8,
        "max_epochs": 1,
        "validation_size": 0.2,
        "early_stopping_patience": 1,
        "device": "cpu",
        "random_state": 42,
        "standardize": True,
    }
    model = build_model(
        "torch_transformer", "classification", (4, 6), 5, params
    )
    model.set_validation_groups(
        groups,
        subject_ids=subjects,
        record_ids=groups,
        outer_test_record_ids=split.record_id_test,
        strategy="group_record",
        group_column="record_group_id",
        validation_size=0.2,
        random_state=42,
    )
    model.fit(X, y)

    run_directory = tmp_path / "baseline-run"
    run_directory.mkdir()
    config = {
        "datasets": {
            "synthetic": {
                "feature_group": "eeg_pow",
                "feature_set": "eeg_pow",
            }
        },
        "models": {
            "torch_transformer": {
                "type": "torch_transformer",
                "params": params,
            }
        },
        "validation": {
            "strategy": "group_record",
            "group_column": "record_group_id",
            "validation_size": 0.2,
            "random_state": 42,
        },
        "evaluation": {
            "protocol": "group_kfold_subject",
            "n_splits": 5,
            "folds": [1, 2, 3, 4, 5],
            "random_state": 42,
        },
        "task_config": {"random_state": 42},
    }
    (run_directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_directory / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    for fold in range(1, 6):
        fold_directory = (
            run_directory
            / "synthetic"
            / "task"
            / "torch_transformer"
            / "group_kfold_subject"
            / f"fold_{fold:02d}"
        )
        fold_directory.mkdir(parents=True)
        model.save(fold_directory / "model.pt")
        (fold_directory / "validation_split.json").write_text(
            json.dumps(model.validation_split_), encoding="utf-8"
        )
        (fold_directory / "normalization_stats.json").write_text(
            json.dumps({
                "feature_names": split.feature_names,
                "mean": model.feature_mean_.tolist(),
                "scale": model.feature_scale_.tolist(),
            }),
            encoding="utf-8",
        )

    result = materialize_categorical_baseline_validation(
        CategoricalBaselineReference(
            feature_group="eeg_pow", seed=42, run_directory=run_directory
        ),
        output_root=tmp_path / "materialized",
        split_builder=lambda config: {
            f"fold_{fold:02d}": split for fold in range(1, 6)
        },
    )
    assert len(result["folds"]) == 5
    assert all(row["outer_test_used"] is False for row in result["folds"])
    assert all(row["strict_checkpoint_load"] is True for row in result["folds"])
    for fold in range(1, 6):
        artifact = (
            tmp_path
            / "materialized"
            / "baselines"
            / "eeg_pow"
            / "seed_42"
            / f"fold_{fold:02d}"
            / "validation_predictions.parquet"
        )
        assert artifact.is_file()
        assert set(pd.read_parquet(artifact)["split"]) == {"inner_validation"}


def test_setup_execute_with_fake_materializer(tmp_path: Path) -> None:
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    document["experiment"]["output_dir"] = str(tmp_path / "outputs")
    document["experiment"]["report_path"] = str(tmp_path / "report.md")
    document["experiment"]["summary_path"] = str(tmp_path / "summary.json")
    baseline_index = tmp_path / "baseline-index.json"
    baseline_index.write_text(
        json.dumps(
            {
                "run_index": [
                    {
                        "method": "categorical",
                        "feature_group": group,
                        "seed": seed,
                        "run_directory": str(tmp_path / group / str(seed)),
                    }
                    for group in ("eeg_pow", "eeg_only")
                    for seed in (7, 42, 123)
                ]
            }
        ),
        encoding="utf-8",
    )
    document["categorical_baseline_index"] = str(baseline_index)
    spec = tmp_path / "setup.yaml"
    spec.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def fake_materializer(reference: CategoricalBaselineReference, *, output_root: Path):
        return {
            "feature_group": reference.feature_group,
            "seed": reference.seed,
            "folds": [
                {
                    "outer_fold": fold,
                    "validation_identity_sha256": f"{reference.feature_group}-{fold}",
                }
                for fold in range(1, 6)
            ],
            "outer_test_used": False,
        }

    experiment = AuxiliaryCornLambdaSelectionSetupExperiment(
        spec, baseline_materializer=fake_materializer
    )
    manifest = experiment.execute(resume=False)
    assert manifest["model_training_performed"] is False
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["ready_for_nested_candidate_training"] is True
    assert all(
        item["exact"]
        for item in summary["cross_seed_validation_alignment"].values()
    )


def test_setup_layer_contains_no_training_loop() -> None:
    assert experiment_contains_training_loop() is False
    source = inspect.getsource(AuxiliaryCornLambdaSelectionSetupExperiment)
    assert ".fit(" not in source
    assert "runner.run(" not in source
