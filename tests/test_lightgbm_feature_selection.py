from __future__ import annotations

import sys
import types
import json
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bench.experiments.lightgbm_feature_selection as lightgbm_experiment

from bench.experiments.lightgbm_feature_selection import (
    _aggregate,
    _load_resumable_summary,
    LightGBMRunSpec,
    build_run_matrix,
    finalize_existing_results,
    load_config,
    protocol_plan,
    stable_hash,
)
from cogstate.model_zoo import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "feature_selection" / "lightgbm_feature_selection_v1.yaml"


def test_lightgbm_run_matrix_is_140_with_four_cell_smoke() -> None:
    config = load_config(CONFIG)
    specs = build_run_matrix(config)
    assert len(specs) == 7 * 2 * 2 * 5 == 140
    smoke = [spec for spec in specs if spec.metric == "attention" and spec.fold == 1]
    assert len(smoke) == 4
    assert {spec.task_type for spec in smoke} == {"classification", "regression"}
    assert {spec.feature_regime for spec in smoke} == {"all_features", "selected_top50"}


def test_lightgbm_optional_dependency_error_is_actionable() -> None:
    if __import__("importlib").util.find_spec("lightgbm") is not None:
        pytest.skip("LightGBM is installed")
    with pytest.raises(ModuleNotFoundError, match="pip install lightgbm"):
        build_model("lightgbm", "classification", None, None, {"random_state": 42})


def test_lightgbm_factory_classifier_and_regressor_contracts(monkeypatch) -> None:
    module = types.ModuleType("lightgbm")

    class Classifier:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            self.n_classes_ = len(np.unique(y))
            return self

        def predict_proba(self, X):
            return np.full((len(X), self.n_classes_), 1 / self.n_classes_)

        def predict(self, X):
            return self.predict_proba(X).argmax(axis=1)

    class Regressor:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mean_)

    module.LGBMClassifier = Classifier
    module.LGBMRegressor = Regressor
    monkeypatch.setitem(sys.modules, "lightgbm", module)
    classifier = build_model("lightgbm", "classification", None, None, {"random_state": 42})
    regressor = build_model("lightgbm", "regression", None, None, {"random_state": 42})
    assert isinstance(classifier, Classifier)
    assert isinstance(regressor, Regressor)
    assert classifier.params["random_state"] == regressor.params["random_state"] == 42
    X = np.zeros((6, 3), dtype=np.float32)
    classifier.fit(X, np.asarray([0, 1, 2, 0, 1, 2]))
    probabilities = classifier.predict_proba(X)
    assert probabilities.shape == (6, 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert classifier.predict(X).shape == (6,)
    regressor.fit(X, np.arange(6, dtype=float))
    assert regressor.predict(X).shape == (6,)


def test_real_lightgbm_plan_locks_folds_features_targets_and_dependency_state() -> None:
    plan = protocol_plan(CONFIG)
    assert plan["run_count"] == 140
    assert plan["feature_count"] == 448
    assert len(plan["fixed_outer_folds"]) == 5
    assert plan["selector"]["top_k"] == 50
    assert plan["selector_fit_scope"].startswith("outer_train_only")
    available = __import__("importlib").util.find_spec("lightgbm") is not None
    assert plan["lightgbm"]["available"] is available
    assert (plan["lightgbm"]["installed_version"] is not None) is available


def test_feature_stability_outputs_are_grouped_and_fold_pairwise(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for fold, names in ((1, ["EEG.a", "POW.b"]), (2, ["EEG.a", "POW.c"])):
        run = runs / f"fold{fold}"
        run.mkdir(parents=True)
        (run / "selector_manifest.json").write_text(
            json.dumps({
                "metric": "attention",
                "task_type": "regression",
                "fold": fold,
                "selected_names": names,
            }),
            encoding="utf-8",
        )
    summaries = []
    for regime, mae in (("all_features", 0.2), ("selected_top50", 0.1)):
        metrics = {
            "mae": mae,
            "rmse": mae + 0.02,
            "r2": 0.1,
            "pearson": 0.2,
            "spearman": 0.19,
        }
        summaries.append({
            "metric": "attention", "task_type": "regression",
            "feature_regime": regime, "fold": 1,
            "training_seconds": 0.1, "inference_seconds": 0.01,
            "model_feature_count": 448 if regime == "all_features" else 50,
            "window_metrics": metrics,
            "participant_macro_metrics": metrics,
        })
    _aggregate(tmp_path, summaries)
    stability = __import__("pandas").read_csv(tmp_path / "feature_stability_summary.csv")
    jaccard = __import__("pandas").read_csv(tmp_path / "feature_jaccard_similarity.csv")
    assert stability.loc[stability["feature_name"] == "EEG.a", "selection_count"].item() == 2
    assert len(jaccard) == 1
    assert jaccard.iloc[0]["jaccard"] == pytest.approx(1 / 3)


def test_specification_hash_is_deterministic_and_protocol_bound() -> None:
    spec = LightGBMRunSpec("attention", "classification", "all_features", 1)
    protocol_hash = "a" * 64
    assert spec.specification_hash(protocol_hash) == spec.specification_hash(protocol_hash)
    assert spec.specification_hash(protocol_hash) != spec.specification_hash("b" * 64)


def test_parameter_changes_flow_through_protocol_into_specification_hash() -> None:
    plan = protocol_plan(CONFIG)
    spec = LightGBMRunSpec("attention", "classification", "all_features", 1)
    current_hash = plan["protocol_hash"]
    without_hash = copy.deepcopy(plan)
    without_hash.pop("protocol_hash")

    changed_subsample = copy.deepcopy(without_hash)
    changed_subsample["lightgbm"]["params"]["subsample_freq"] = 2
    changed_subsample_hash = stable_hash(changed_subsample)
    assert changed_subsample_hash != current_hash
    assert spec.specification_hash(changed_subsample_hash) != spec.specification_hash(current_hash)

    changed_top_k = copy.deepcopy(without_hash)
    changed_top_k["selector"]["top_k"] = 25
    changed_top_k_hash = stable_hash(changed_top_k)
    assert changed_top_k_hash != current_hash
    assert spec.specification_hash(changed_top_k_hash) != spec.specification_hash(current_hash)


@pytest.mark.parametrize(
    "changed",
    [
        LightGBMRunSpec("focus", "classification", "all_features", 1),
        LightGBMRunSpec("attention", "regression", "all_features", 1),
        LightGBMRunSpec("attention", "classification", "selected_top50", 1),
        LightGBMRunSpec("attention", "classification", "all_features", 2),
    ],
)
def test_run_dimension_changes_specification_hash(changed: LightGBMRunSpec) -> None:
    protocol_hash = "c" * 64
    original = LightGBMRunSpec("attention", "classification", "all_features", 1)
    assert changed.specification_hash(protocol_hash) != original.specification_hash(protocol_hash)


def test_resume_rejects_stale_completed_summary(tmp_path: Path) -> None:
    path = tmp_path / "run_summary.json"
    path.write_text(
        json.dumps({"status": "complete", "specification_hash": "stale"}),
        encoding="utf-8",
    )
    assert _load_resumable_summary(
        path, resume=True, specification_hash="current"
    ) is None


def test_resume_accepts_only_current_completed_summary(tmp_path: Path) -> None:
    path = tmp_path / "run_summary.json"
    expected = {
        "status": "complete",
        "specification_hash": "current",
        "sentinel": 42,
    }
    path.write_text(json.dumps(expected), encoding="utf-8")
    assert _load_resumable_summary(
        path, resume=True, specification_hash="current"
    ) == expected
    assert _load_resumable_summary(
        path, resume=False, specification_hash="current"
    ) is None


def _synthetic_completed_results(output: Path) -> str:
    protocol_hash = "f9c3898cd2ce20055082e3d8e746c830fcadf71a72dff0a55c760880f3b736bf"
    protocol = {
        "experiment_id": "lightgbm_feature_selection_v1",
        "protocol_hash": protocol_hash,
        "expected_run_count": 140,
        "analysis_role": "confirmatory_protocol_not_executed",
    }
    output.mkdir(parents=True)
    (output / "protocol_manifest.json").write_text(
        json.dumps(protocol, sort_keys=True), encoding="utf-8"
    )
    specs = build_run_matrix(load_config(CONFIG))
    matrix_rows = []
    selected_names = [f"EEG.f{i}" for i in range(20)] + [
        f"POW.f{i}" for i in range(30)
    ]
    for spec in specs:
        specification_hash = spec.specification_hash(protocol_hash)
        matrix_rows.append({
            **spec.__dict__,
            "run_id": spec.run_id,
            "specification_hash": specification_hash,
        })
        run_dir = output / "runs" / spec.run_id
        run_dir.mkdir(parents=True)
        selected = spec.feature_regime == "selected_top50"
        if spec.task_type == "classification":
            value = 0.49 if selected else 0.50
            window_metrics = {
                "macro_f1": value,
                "balanced_accuracy": value + 0.01,
                "accuracy": value + 0.02,
            }
            participant_metrics = {key: number - 0.02 for key, number in window_metrics.items()}
            q3_thresholds = [0.33, 0.66]
            y_true = [0, 2]
        else:
            value = 0.11 if selected else 0.10
            window_metrics = {
                "mae": value,
                "rmse": value + 0.02,
                "r2": 0.19 if selected else 0.20,
                "pearson": 0.29 if selected else 0.30,
                "spearman": 0.28 if selected else 0.29,
            }
            participant_metrics = dict(window_metrics)
            q3_thresholds = None
            y_true = [0.2, 0.8]
        summary = {
            "status": "complete",
            "run_id": spec.run_id,
            "specification_hash": specification_hash,
            **spec.__dict__,
            "window_metrics": window_metrics,
            "participant_macro_metrics": participant_metrics,
            "q3_thresholds": q3_thresholds,
            "train_windows": 8,
            "test_windows": 2,
            "model_feature_count": 50 if selected else 448,
            "training_seconds": 1.0 if selected else 4.0,
            "inference_seconds": 0.1 if selected else 0.2,
            "lightgbm_version": "4.7.0",
            "outer_participant_overlap": [],
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        sample_prefix = f"{spec.metric}-{spec.task_type}-fold{spec.fold}"
        pd.DataFrame({
            "sample_id": [f"{sample_prefix}-a", f"{sample_prefix}-b"],
            "y_true": y_true,
        }).to_parquet(run_dir / "predictions.parquet", index=False)
        if selected:
            (run_dir / "selector_manifest.json").write_text(
                json.dumps({
                    "metric": spec.metric,
                    "task_type": spec.task_type,
                    "fold": spec.fold,
                    "selected_names": selected_names,
                    "selected_count": 50,
                    "feature_group_counts": {"EEG": 20, "POW": 30},
                }),
                encoding="utf-8",
            )
    pd.DataFrame(matrix_rows).to_csv(output / "run_matrix.csv", index=False)
    return protocol_hash


def test_finalization_uses_existing_artifacts_only_and_preserves_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "results"
    protocol_hash = _synthetic_completed_results(output)
    protocol_path = output / "protocol_manifest.json"
    before = hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    def forbidden(*args, **kwargs):
        raise AssertionError("fit/build must not be invoked during finalization")

    monkeypatch.setattr(lightgbm_experiment, "build_model", forbidden)
    monkeypatch.setattr(lightgbm_experiment.FeatureSelector, "fit", forbidden)
    manifest = finalize_existing_results(CONFIG, output_dir=output)

    after = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert before == after
    assert manifest["protocol_hash"] == protocol_hash
    assert manifest["execution_status"] == "complete"
    assert manifest["completed_run_count"] == 140
    assert manifest["failed_run_count"] == 0
    assert manifest["stale_or_mismatched_run_count"] == 0
    assert manifest["model_or_selector_fit_invoked"] is False
    comparison = pd.read_csv(output / "all_vs_selected_comparison.csv")
    regression = comparison.loc[comparison["task_type"] == "regression"]
    delta_columns = [
        f"delta_{name}_selected_minus_all"
        for name in lightgbm_experiment.REGRESSION_METRICS
    ]
    assert regression[delta_columns].notna().to_numpy().all()
    pm_macro = pd.read_csv(output / "pm_macro_comparison.csv")
    pm_regression = pm_macro.loc[pm_macro["task_type"] == "regression"]
    assert pm_regression[delta_columns].notna().to_numpy().all()
    cohort_audit = pd.read_csv(output / "evaluation_cohort_audit.csv")
    assert len(cohort_audit) == 70
    assert cohort_audit[[
        "sample_ids_match", "targets_match", "train_test_counts_match",
        "q3_thresholds_match",
    ]].to_numpy(dtype=bool).all()
