from __future__ import annotations

from pathlib import Path

from bench.automl.objective import AutoMLTrialResult, NestedBenchmarkObjective
from bench.automl.search_space import SearchSpaceSpec
from model_zoo.factory import build_model


def _space() -> SearchSpaceSpec:
    return SearchSpaceSpec.from_dict({
        "model.params.d_model": {"type": "categorical", "choices": [16]},
        "model.params.nhead": {"type": "categorical", "choices": [4]},
        "model.params.num_layers": {"type": "categorical", "choices": [1]},
        "model.params.dim_feedforward": {
            "type": "categorical",
            "choices": [32],
        },
        "model.params.dropout": {"type": "categorical", "choices": [0.1]},
        "training.learning_rate": {
            "type": "categorical",
            "choices": [0.001],
        },
        "training.weight_decay": {
            "type": "categorical",
            "choices": [0.0001],
        },
        "training.batch_size": {"type": "categorical", "choices": [16]},
    })


def _params() -> dict:
    return {
        parameter.path: parameter.choices[0]
        for parameter in _space().parameters
    }


def _base_config() -> dict:
    return {
        "output_dir": "unused",
        "datasets": {
            "emotiv_cognitive": {
                "data_path": "unused.parquet",
                "feature_set": "pow_plus_eeg",
                "target_col": "label_q5",
            }
        },
        "tasks": ["cognitive_load_5class"],
        "sequence": {"length": 8},
        "models": {
            "torch_transformer": {
                "type": "torch_transformer",
                "task_type": "classification",
                "params": {
                    "sequence_length": 8,
                    "pooling": "last",
                    "positional_encoding": "learned",
                },
            }
        },
        "validation": {
            "strategy": "group_record",
            "group_column": "record_group_id",
        },
        "evaluation": {
            "protocol": "group_kfold_subject",
            "group_column": "subject_id",
            "n_splits": 5,
        },
    }


class _CompletedReference:
    run_directory = Path("unused")

    def to_dict(self) -> dict:
        return {
            "status": "completed",
            "benchmark_run_directory": "canonical/run",
        }


class _FakeRunner:
    created_configs: list[dict] = []

    def __init__(self, config: dict) -> None:
        self.config = config
        self.created_configs.append(config)
        model_name = next(iter(config["models"]))
        self.results = {
            "emotiv_cognitive": {
                "models": {
                    "cognitive_load_5class": {
                        model_name: {
                            "group_kfold_subject": {
                                "folds": {"fold_01": {}, "fold_02": {}},
                                "aggregated": {
                                    "balanced_accuracy_mean": 0.35,
                                    "balanced_accuracy_std": 0.02,
                                    "macro_f1_mean": 0.33,
                                    "macro_f1_std": 0.03,
                                },
                            }
                        }
                    }
                }
            }
        }

    def run(self) -> None:
        return None

    def completed_run(self) -> _CompletedReference:
        return _CompletedReference()


def test_objective_calls_canonical_runner_without_outer_test_labels(tmp_path) -> None:
    _FakeRunner.created_configs.clear()
    objective = NestedBenchmarkObjective(
        study_name="test",
        base_config=_base_config(),
        search_space=_space(),
        outer_fold=1,
        outer_train_subjects=["S1", "S2", "S3"],
        outer_test_subjects=["S4"],
        inner_splits=2,
        random_state=42,
        benchmark_runs_root=tmp_path,
        max_epochs=2,
        runner_factory=_FakeRunner,
        completed_run_finder=lambda *args, **kwargs: None,
    )
    result = objective.execute(0, _params())
    assert result.state == "COMPLETE"
    assert result.objective_value == 0.35
    assert result.secondary_metrics["macro_f1"] == 0.33
    assert result.benchmark_run_reference["status"] == "completed"
    assert len(_FakeRunner.created_configs) == 1
    resolved = _FakeRunner.created_configs[0]
    included = resolved["datasets"]["emotiv_cognitive"]["include_subject_ids"]
    assert included == ["S1", "S2", "S3"]
    assert "S4" not in included
    assert resolved["evaluation"]["role"] == "inner_search"
    assert "outer_test_labels" not in resolved


def test_resolved_trial_still_uses_existing_model_factory(tmp_path) -> None:
    objective = NestedBenchmarkObjective(
        study_name="factory",
        base_config=_base_config(),
        search_space=_space(),
        outer_fold=1,
        outer_train_subjects=["S1", "S2"],
        outer_test_subjects=["S3"],
        inner_splits=2,
        random_state=42,
        benchmark_runs_root=tmp_path,
    )
    config, _ = objective.resolve(_params())
    model_config = config["models"]["torch_transformer"]
    model = build_model(
        model_name=model_config["type"],
        task_type=model_config["task_type"],
        input_shape=(8, 448),
        num_outputs=5,
        params=model_config["params"],
    )
    assert model.model.__class__.__name__ == "TorchFeatureTransformerClassifier"


def test_trial_result_round_trip() -> None:
    result = AutoMLTrialResult(
        study_name="study",
        trial_number=3,
        trial_parameters={"x": 1},
        resolved_config={"models": {}},
        resolved_config_hash="abc",
        outer_fold=1,
        inner_split="fold_01,fold_02",
        objective_value=0.4,
        secondary_metrics={"macro_f1": 0.38},
    )
    assert AutoMLTrialResult.from_dict(result.to_dict()) == result


def test_failed_trial_preserves_reason(tmp_path) -> None:
    class _FailingRunner:
        def __init__(self, config: dict) -> None:
            self.config = config

        def run(self) -> None:
            raise RuntimeError("synthetic training failure")

    objective = NestedBenchmarkObjective(
        study_name="failure",
        base_config=_base_config(),
        search_space=_space(),
        outer_fold=1,
        outer_train_subjects=["S1", "S2"],
        outer_test_subjects=["S3"],
        inner_splits=2,
        random_state=42,
        benchmark_runs_root=tmp_path,
        runner_factory=_FailingRunner,
        completed_run_finder=lambda *args, **kwargs: None,
    )
    result = objective.execute(4, _params())
    assert result.state == "FAIL"
    assert "synthetic training failure" in result.failure_reason
