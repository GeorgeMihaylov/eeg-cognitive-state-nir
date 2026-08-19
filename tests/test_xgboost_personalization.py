import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from model_zoo.ML.xgboost_personalization import (
    XGBoostMarginHeadAdapter,
    xgboost_state_sha256,
)


def _dataset():
    X, y = make_classification(
        n_samples=600,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    return (
        X[:400],
        y[:400],
        X[400:500],
        y[400:500],
        X[500:],
        y[500:],
    )


def _model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def test_multiclass_margin_shape():
    X_train, y_train, _, _, X_eval, _ = _dataset()

    model = _model(X_train, y_train)
    adapter = XGBoostMarginHeadAdapter(model)

    margins = adapter.predict_margin(X_eval)

    assert margins.shape == (len(X_eval), 3)


def test_identity_head_reproduces_xgboost_probabilities():
    X_train, y_train, _, _, X_eval, _ = _dataset()

    model = _model(X_train, y_train)
    adapter = XGBoostMarginHeadAdapter(model)

    error = adapter.identity_probability_error(X_eval)

    assert error < 1e-5

    np.testing.assert_allclose(
        adapter.zero_shot_predict_proba(X_eval),
        adapter.predict_proba(X_eval),
        rtol=1e-5,
        atol=1e-6,
    )


def test_participant_heads_are_independent():
    X_train, y_train, _, _, _, _ = _dataset()

    model = _model(X_train, y_train)

    first = XGBoostMarginHeadAdapter(model)
    second = first.clone_for_participant()

    assert first.head is not second.head
    assert first.head_hash == second.head_hash
    assert first.global_model is second.global_model


def test_fitting_head_does_not_modify_xgboost():
    (
        X_train,
        y_train,
        X_calibration,
        y_calibration,
        X_eval,
        y_eval,
    ) = _dataset()

    model = _model(X_train, y_train)

    adapter = XGBoostMarginHeadAdapter(model)

    booster_hash_before = xgboost_state_sha256(model)
    head_hash_before = adapter.head_hash

    log = adapter.fit_head(
        X_calibration[:80],
        y_calibration[:80],
        X_calibration[80:],
        y_calibration[80:],
        learning_rate=1e-2,
        max_epochs=10,
        patience=3,
    )

    booster_hash_after = xgboost_state_sha256(model)
    head_hash_after = adapter.head_hash

    assert log
    assert booster_hash_before == booster_hash_after
    assert head_hash_before != head_hash_after

    probabilities = adapter.predict_proba(X_eval)

    assert probabilities.shape == (len(X_eval), 3)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X_eval)),
        atol=1e-6,
    )

    predictions = adapter.predict(X_eval)

    assert predictions.shape == y_eval.shape
    assert set(np.unique(predictions)).issubset(
        set(model.classes_)
    )