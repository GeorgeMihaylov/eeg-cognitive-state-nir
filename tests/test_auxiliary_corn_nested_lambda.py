from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bench.core.abstract_task import TaskSplit
from bench.experiments.auxiliary_corn_nested_lambda import (
    AuxiliaryCornNestedLambdaExperiment,
    NestedFoldPlan,
    NestedLambdaPlan,
    load_auxiliary_corn_nested_spec,
)
from bench.experiments.ordinal_transformer import build_ordinal_transformer_experiment


SPEC = Path("experiments/auxiliary_corn_nested_lambda.yaml")


def test_nested_spec_and_dispatch() -> None:
    document = load_auxiliary_corn_nested_spec(SPEC)
    assert document["protocol"]["candidate_fold_fits"] == 90
    assert document["protocol"]["outer_test_selected_only"] is True
    experiment = build_ordinal_transformer_experiment(SPEC)
    assert isinstance(experiment, AuxiliaryCornNestedLambdaExperiment)


def test_nested_plan_has_90_candidate_fits() -> None:
    plan = AuxiliaryCornNestedLambdaExperiment(SPEC).plan()
    assert len(plan.folds) == 30
    assert plan.candidate_fold_fits == 90
    assert plan.selected_outer_evaluations == 30
    rendered = AuxiliaryCornNestedLambdaExperiment.render_plan(plan)
    assert "Candidate fold fits: 90" in rendered
    assert "Rejected candidates never receive outer-test predictions" in rendered


def _split() -> TaskSplit:
    n_train, n_test = 12, 5
    y_train = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 3])
    y_test = np.asarray([0, 1, 2, 3, 4])
    return TaskSplit(
        X_train=np.zeros((n_train, 4, 6), dtype=np.float32),
        y_train=y_train,
        X_test=np.zeros((n_test, 4, 6), dtype=np.float32),
        y_test=y_test,
        subject_train=np.asarray([f"subject-{i // 2}" for i in range(n_train)]),
        subject_test=np.asarray([f"test-subject-{i}" for i in range(n_test)]),
        feature_names=[f"feature-{i}" for i in range(6)],
        metadata={"fold": 1, "fold_name": "fold_01", "protocol": "group_kfold_subject"},
        sample_id_train=np.arange(n_train),
        sample_id_test=np.arange(n_test) + 100,
        record_id_train=np.asarray([f"record-{i // 2}" for i in range(n_train)]),
        record_id_test=np.asarray([f"test-record-{i}" for i in range(n_test)]),
        row_metadata_train={
            "sequence_id": np.asarray([f"sequence-{i:02d}" for i in range(n_train)]),
            "record_group_id": np.asarray([f"group-{i // 2}" for i in range(n_train)]),
            "source": np.asarray(["synthetic"] * n_train),
            "target_sample_id": np.arange(n_train),
            "target_time": np.arange(n_train, dtype=float) * 10.0,
        },
        row_metadata_test={
            "sequence_id": np.asarray([f"test-sequence-{i:02d}" for i in range(n_test)]),
            "record_group_id": np.asarray([f"test-group-{i}" for i in range(n_test)]),
            "source": np.asarray(["synthetic"] * n_test),
            "target_sample_id": np.arange(n_test) + 100,
            "target_time": np.arange(n_test, dtype=float) * 10.0 + 1000,
        },
    )


class _FakeModel:
    def __init__(self, params):
        self.weight = float(params["auxiliary_weight"])
        self.random_state = int(params["random_state"])
        self.training_log_ = []
        self.validation_split_ = {}
        self.feature_mean_ = np.zeros((4, 6), dtype=np.float32)
        self.feature_scale_ = np.ones((4, 6), dtype=np.float32)
        self.inner_validation_indices_ = np.asarray([1, 3, 5, 7, 9], dtype=np.int64)

    def set_validation_groups(self, *args, **kwargs):
        return self

    @staticmethod
    def _probabilities(y, quality):
        probs = np.full((len(y), 5), (1.0 - quality) / 4.0, dtype=np.float64)
        probs[np.arange(len(y)), y] = quality
        return probs

    def fit(self, X, y):
        self.training_log_ = [{
            "epoch": 1,
            "validation_categorical_loss": 1.0 - self.weight / 10.0,
            "validation_ordinal_loss": 0.5,
            "validation_total_loss": 1.0,
        }]
        self.validation_split_ = {
            "inner_validation_size": len(self.inner_validation_indices_),
            "group_overlap": [],
        }
        return self

    def _detailed(self, y):
        y = np.asarray(y, dtype=int)
        quality = {0.25: 0.52, 0.5: 0.58, 1.0: 0.72}[self.weight]
        predicted_targets = y.copy()
        if self.weight == 0.25:
            predicted_targets = (predicted_targets + 2) % 5
        elif self.weight == 0.5:
            predicted_targets[::2] = (predicted_targets[::2] + 1) % 5
        probabilities = self._probabilities(predicted_targets, quality)
        thresholds = np.asarray([
            [0.8, 0.6, 0.4, 0.2] for _ in range(len(y))
        ], dtype=np.float64)
        aux_probs = np.asarray([
            [0.2, 0.2, 0.2, 0.2, 0.2] for _ in range(len(y))
        ], dtype=np.float64)
        return {
            "head_type": "categorical_corn",
            "y_pred": probabilities.argmax(axis=1),
            "class_probabilities": probabilities,
            "categorical_expected_rank": (probabilities * np.arange(5)).sum(axis=1),
            "aux_threshold_probabilities": thresholds,
            "aux_class_probabilities": aux_probs,
            "auxiliary_raw_outputs": np.zeros((len(y), 4)),
            "aux_expected_rank": thresholds.sum(axis=1),
            "aux_ordinal_prediction": (thresholds >= 0.5).sum(axis=1),
            "aux_ordinal_argmax": aux_probs.argmax(axis=1),
            "auxiliary_weight": self.weight,
        }

    def validation_partition_detailed(self, X, y):
        indices = self.inner_validation_indices_
        return {
            "indices": indices,
            "y_true": np.asarray(y)[indices],
            **self._detailed(np.asarray(y)[indices]),
        }

    def predict_detailed(self, X):
        y = np.arange(len(X), dtype=int) % 5
        return self._detailed(y)

    def save(self, path):
        Path(path).write_text(json.dumps({"weight": self.weight}), encoding="utf-8")

    def load(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert float(payload["weight"]) == self.weight
        return self

    def get_training_summary(self):
        return {
            "epochs_trained": 1,
            "best_epoch": 1,
            "best_validation_loss": 1.0 - self.weight / 10.0,
            "early_stopping_monitor": "validation_categorical_loss",
        }


def _write_test_spec(tmp_path: Path) -> Path:
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    document["experiment"]["output_dir"] = str(tmp_path / "results")
    document["experiment"]["report_path"] = str(tmp_path / "report.md")
    document["experiment"]["summary_path"] = str(tmp_path / "summary.json")
    document["baseline_validation_root"] = str(tmp_path / "baseline-root")
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_execute_never_writes_outer_predictions_for_rejected_candidates(tmp_path: Path) -> None:
    spec = _write_test_spec(tmp_path)
    split = _split()
    baseline_run = tmp_path / "baseline-run"
    baseline_run.mkdir()
    baseline_config = {
        "output_dir": str(baseline_run),
        "datasets": {"emotiv_cognitive": {"feature_group": "eeg_pow"}},
        "tasks": ["cognitive_load_5class"],
        "models": {"torch_transformer": {"type": "torch_transformer", "params": {
            "head_type": "categorical",
            "random_state": 42,
        }}},
        "sequence": {"length": 4},
        "validation": {"strategy": "group_record", "group_column": "record_group_id", "validation_size": 0.2, "random_state": 42},
        "evaluation": {"protocol": "group_kfold_subject", "folds": [1], "random_state": 42},
        "task_config": {"random_state": 42},
    }
    (baseline_run / "config.yaml").write_text(
        yaml.safe_dump(baseline_config, sort_keys=False), encoding="utf-8"
    )
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    indices = np.asarray([1, 3, 5, 7, 9])
    base = pd.DataFrame({
        "outer_train_index": indices,
        "outer_fold": 1,
        "fold": 1,
        "split": "inner_validation",
        "feature_group": "eeg_pow",
        "seed": 42,
        "head_type": "categorical",
        "sample_id": split.sample_id_train[indices],
        "subject_id": split.subject_train[indices],
        "record_id": split.record_id_train[indices],
        "y_true": split.y_train[indices],
        "y_pred": split.y_train[indices],
        "sequence_id": split.row_metadata_train["sequence_id"][indices],
        "record_group_id": split.row_metadata_train["record_group_id"][indices],
        "source": split.row_metadata_train["source"][indices],
        "target_sample_id": split.row_metadata_train["target_sample_id"][indices],
        "target_time": split.row_metadata_train["target_time"][indices],
    })
    for i in range(5):
        base[f"proba_{i}"] = (base["y_true"] == i).astype(float)
        base[f"class_probability_{i}"] = base[f"proba_{i}"]
    base["categorical_expected_rank"] = base["y_true"].astype(float)
    baseline_predictions = baseline_dir / "validation_predictions.parquet"
    baseline_metrics = baseline_dir / "validation_metrics.json"
    base.to_parquet(baseline_predictions, index=False)
    baseline_metrics.write_text(json.dumps({"balanced_accuracy": 0.2}), encoding="utf-8")

    output = tmp_path / "results"
    fold_plan = NestedFoldPlan(
        feature_group="eeg_pow",
        seed=42,
        outer_fold=1,
        baseline_run_directory=baseline_run,
        baseline_validation_metrics=baseline_metrics,
        baseline_validation_predictions=baseline_predictions,
        candidate_root=output / "candidates" / "eeg_pow" / "seed_42" / "fold_01",
        selected_root=output / "selected" / "eeg_pow" / "seed_42" / "fold_01",
    )
    plan = NestedLambdaPlan(
        folds=(fold_plan,),
        candidate_fold_fits=3,
        selected_outer_evaluations=1,
        output_root=output,
    )
    experiment = AuxiliaryCornNestedLambdaExperiment(
        spec,
        split_builder=lambda config: {"fold_01": split},
        model_builder=lambda *args, **kwargs: _FakeModel(kwargs["params"]),
    )
    manifest = experiment.execute(plan, resume=False)
    assert manifest["status"] == "completed"
    selected = json.loads((fold_plan.selected_root / "selection_decision.json").read_text())
    assert selected["selected"]["auxiliary_weight"] == 1.0
    assert (fold_plan.selected_root / "outer_test_predictions.parquet").is_file()
    for weight in (0.25, 0.5, 1.0):
        candidate = fold_plan.candidate_root / f"lambda_{str(weight).replace('.', 'p').rstrip('0').rstrip('p')}"
        assert (candidate / "validation_predictions.parquet").is_file()
        assert not (candidate / "outer_test_predictions.parquet").exists()
