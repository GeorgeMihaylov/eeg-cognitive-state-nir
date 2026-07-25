import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.bench_runner import BenchmarkRunner
from model_zoo.DL.feature_preprocessing import FeaturePreprocessor
from model_zoo.factory import build_model


FEATURE_NAMES = ("EEG.AF3__mean", "POW.T8.Alpha__min", "POW.T8.BetaL__min")


def test_standard_reproduces_existing_mean_std_behavior():
    X = np.asarray([[1.0, 2.0], [3.0, 8.0], [5.0, 14.0]])
    fitted = FeaturePreprocessor(
        {"strategy": "standard"}, feature_names=["a", "b"]
    ).fit(X)
    expected = (X - X.mean(axis=0)) / X.std(axis=0)
    np.testing.assert_allclose(fitted.transform(X), expected, rtol=1e-6)


def test_robust_and_clipping_are_fitted_only_on_train():
    train = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    validation = np.asarray([[1e9, -1e9]])
    robust = FeaturePreprocessor(
        {"strategy": "robust"}, feature_names=["a", "b"]
    ).fit(train)
    clipped = FeaturePreprocessor(
        {
            "strategy": "standard_clip",
            "clip_percentiles": [25, 75],
        },
        feature_names=["a", "b"],
    ).fit(train)
    center_before = robust.center_.copy()
    bounds_before = (clipped.clip_lower_.copy(), clipped.clip_upper_.copy())
    robust.transform(validation)
    transformed = clipped.transform(validation)
    np.testing.assert_array_equal(robust.center_, center_before)
    np.testing.assert_array_equal(clipped.clip_lower_, bounds_before[0])
    np.testing.assert_array_equal(clipped.clip_upper_, bounds_before[1])
    assert np.max(np.abs(transformed)) < 10


def test_pow_log_changes_only_pow_columns_and_supports_signed_values():
    X = np.asarray([
        [4.0, -9.0, 99.0],
        [8.0, 3.0, 999.0],
        [12.0, 15.0, 9999.0],
    ])
    fitted = FeaturePreprocessor(
        {"strategy": "pow_log_standard"},
        feature_names=FEATURE_NAMES,
    ).fit(X)
    assert fitted.pow_log_rule_ == "signed_log1p"
    transformed_before_scaling = fitted._apply_pow_log(X)
    np.testing.assert_array_equal(transformed_before_scaling[:, 0], X[:, 0])
    np.testing.assert_allclose(
        transformed_before_scaling[:, 1],
        np.sign(X[:, 1]) * np.log1p(np.abs(X[:, 1])),
    )
    np.testing.assert_allclose(
        transformed_before_scaling[:, 2],
        np.log1p(X[:, 2]),
    )


@pytest.mark.parametrize("strategy", [
    "standard",
    "robust",
    "standard_clip",
    "robust_clip",
    "pow_log_standard",
    "pow_log_robust",
])
def test_strategies_preserve_shapes_and_finite_values(strategy):
    rng = np.random.default_rng(42)
    train = rng.normal(size=(12, 4, 3))
    transformed = FeaturePreprocessor(
        {"strategy": strategy}, feature_names=FEATURE_NAMES
    ).fit(train).transform(train[:3])
    assert transformed.shape == (3, 4, 3)
    assert np.isfinite(transformed).all()


def test_feature_order_and_hash_round_trip():
    X = np.arange(12, dtype=float).reshape(4, 3)
    fitted = FeaturePreprocessor(
        {"strategy": "robust"}, feature_names=FEATURE_NAMES
    ).fit(X)
    state = fitted.to_state()
    restored = FeaturePreprocessor.from_state(state)
    assert state["feature_names"] == list(FEATURE_NAMES)
    assert restored.feature_hash == fitted.feature_hash
    np.testing.assert_array_equal(restored.transform(X), fitted.transform(X))


def test_near_constant_feature_does_not_create_nonfinite_values():
    X = np.asarray([[1.0, 2.0], [1.0 + 1e-12, 3.0], [1.0, 4.0]])
    transformed = FeaturePreprocessor(
        {"strategy": "standard", "scale_floor": 1e-8},
        feature_names=["constant", "varying"],
    ).fit(X).transform(X)
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_raise_clear_error(bad_value):
    X = np.asarray([[1.0, 2.0], [3.0, bad_value]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        FeaturePreprocessor(
            {"strategy": "standard"}, feature_names=["a", "b"]
        ).fit(X)


def _adapter(
    *,
    task_type,
    num_outputs,
    feature_scaling="robust",
    random_state=42,
):
    return build_model(
        model_name="torch_mlp",
        task_type=task_type,
        input_shape=(3,),
        num_outputs=num_outputs,
        params={
            "hidden_dims": [8],
            "dropout": 0.0,
            "batch_size": 8,
            "max_epochs": 1,
            "validation_size": 0.25,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": random_state,
            "feature_scaling": {"strategy": feature_scaling},
            "feature_names": FEATURE_NAMES,
        },
    )


def test_same_seed_gives_same_split_and_preprocessing_state():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3)).astype(np.float32)
    y = rng.normal(size=(40, 2)).astype(np.float32)
    first = _adapter(task_type="regression", num_outputs=2).fit(X, y)
    second = _adapter(task_type="regression", num_outputs=2).fit(X, y)
    np.testing.assert_array_equal(
        first.inner_validation_indices_, second.inner_validation_indices_
    )
    assert (
        first.get_feature_preprocessing_state()
        == second.get_feature_preprocessing_state()
    )


def test_classification_scalar_and_multioutput_regression_still_fit():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 3)).astype(np.float32)
    classification = _adapter(
        task_type="classification", num_outputs=2
    ).fit(X, np.tile([0, 1], 20))
    assert classification.predict(X[:4]).shape == (4,)

    scalar = _adapter(task_type="regression", num_outputs=1).fit(
        X, rng.normal(size=40).astype(np.float32)
    )
    assert scalar.predict(X[:4]).shape == (4,)

    multioutput = _adapter(task_type="regression", num_outputs=2).fit(
        X, rng.normal(size=(40, 2)).astype(np.float32)
    )
    assert multioutput.predict(X[:4]).shape == (4, 2)


def test_save_load_preserves_preprocessing_state_and_predictions(tmp_path):
    rng = np.random.default_rng(21)
    X = rng.normal(size=(40, 3)).astype(np.float32)
    X[:, 2] *= 100
    y = rng.normal(size=(40, 2)).astype(np.float32)
    fitted = _adapter(
        task_type="regression",
        num_outputs=2,
        feature_scaling="robust_clip",
    ).fit(X, y)
    expected = fitted.predict(X[:8])
    path = tmp_path / "model.pt"
    fitted.save(path)
    loaded = _adapter(
        task_type="regression",
        num_outputs=2,
        feature_scaling="standard",
    ).load(path)
    np.testing.assert_allclose(loaded.predict(X[:8]), expected, rtol=1e-7)
    assert loaded.get_feature_preprocessing_state()["strategy"] == "robust_clip"


def test_groupkfold_runner_applies_model_level_feature_scaling(tmp_path):
    rng = np.random.default_rng(123)
    n_subjects = 25
    rows = n_subjects * 2
    frame = pd.DataFrame({
        "sample_id": np.arange(rows),
        "subject_id": np.repeat(
            [f"S{index:02d}" for index in range(n_subjects)], 2
        ),
        "record_id": np.repeat(
            [f"R{index:02d}" for index in range(n_subjects)], 2
        ),
        "source": "synthetic",
        "EEG.AF3__mean": rng.normal(size=rows),
        "POW.T8.Alpha__min": rng.lognormal(size=rows),
        "POW.T8.BetaL__min": rng.lognormal(size=rows),
    })
    for target in (
        "attention",
        "engagement",
        "excitement",
        "stress",
        "relaxation",
        "interest",
        "focus",
    ):
        frame[f"target_{target}"] = rng.uniform(size=rows)
    data_path = tmp_path / "pm.parquet"
    frame.to_parquet(data_path, index=False)
    config = {
        "output_dir": str(tmp_path / "results"),
        "datasets": {
            "emotiv_pm_regression": {
                "data_path": str(data_path),
                "feature_set": "pow_plus_eeg",
                "target_cols": [
                    column for column in frame if column.startswith("target_")
                ],
                "n_outputs": 7,
                "task_type": "regression",
                "discretize": False,
            }
        },
        "tasks": ["performance_metrics_regression"],
        "validation": {
            "strategy": "group_holdout",
            "group_column": "subject_id",
            "fraction": 0.2,
            "random_state": 42,
        },
        "models": {
            "robust_clip": {
                "type": "torch_mlp",
                "task_type": "regression",
                "feature_scaling": {
                    "strategy": "robust_clip",
                    "clip_percentiles": [1, 99],
                },
                "params": {
                    "hidden_dims": [8],
                    "dropout": 0.0,
                    "batch_size": 8,
                    "max_epochs": 1,
                    "early_stopping_patience": 1,
                    "device": "cpu",
                    "random_state": 42,
                },
            }
        },
        "task_config": {"n_outputs": 7, "random_state": 42},
        "evaluation": {
            "protocol": "group_kfold_subject",
            "group_column": "subject_id",
            "n_splits": 5,
            "folds": [1],
            "random_state": 42,
        },
        "run_within_subject": False,
        "run_loso": False,
    }
    runner = BenchmarkRunner(config)
    runner.run()
    fold = runner.results["emotiv_pm_regression"]["models"][
        "performance_metrics_regression"
    ]["robust_clip"]["group_kfold_subject"]["folds"]["fold_01"]
    state = json.loads(
        Path(fold["artifacts"]["feature_scaling"]).read_text(
            encoding="utf-8"
        )
    )
    assert state["strategy"] == "robust_clip"
    assert state["train_only"]
    assert Path(fold["artifacts"]["feature_transform"]).is_file()
