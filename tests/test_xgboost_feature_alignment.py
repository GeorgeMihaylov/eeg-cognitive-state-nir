import numpy as np
import pytest
from xgboost import XGBClassifier

from cogstate.adaptation import (
    FeatureAligner,
    FeatureAlignmentConfig,
)
from model_zoo.ML.xgboost_personalization import (
    xgboost_state_sha256,
)


def _make_reference_data():
    x = np.linspace(-4.0, 4.0, 600)

    X = np.column_stack(
        [
            x,
            x ** 2,
            np.sin(x),
            np.cos(0.5 * x),
        ]
    ).astype(np.float64)

    y = np.digitize(
        x,
        bins=[-1.0, 1.0],
    ).astype(np.int64)

    return X, y


@pytest.mark.parametrize(
    "method",
    [
        "standard_location_scale",
        "robust_location_scale",
    ],
)
def test_feature_alignment_restores_xgboost_coordinate_system(method):
    X_train, y_train = _make_reference_data()

    model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    # Evaluation points are different from train points.
    x_eval = np.linspace(-3.9, 3.9, 257)

    X_eval_reference = np.column_stack(
        [
            x_eval,
            x_eval ** 2,
            np.sin(x_eval),
            np.cos(0.5 * x_eval),
        ]
    ).astype(np.float64)

    # Synthetic participant-specific affine feature shift.
    participant_scale = np.asarray(
        [2.5, 0.4, 3.0, 1.7],
        dtype=np.float64,
    )
    participant_shift = np.asarray(
        [25.0, -40.0, 15.0, 30.0],
        dtype=np.float64,
    )

    # Use a transformed copy of the reference distribution as
    # calibration data. The evaluation rows remain unseen.
    X_calibration = (
        X_train * participant_scale
        + participant_shift
    )

    X_eval_participant = (
        X_eval_reference * participant_scale
        + participant_shift
    )

    booster_hash_before = xgboost_state_sha256(model)

    aligner = FeatureAligner(
        FeatureAlignmentConfig(
            method=method,
        )
    )

    aligner.fit_reference(X_train)
    aligner.fit_calibration(X_calibration)

    X_eval_aligned = aligner.transform(
        X_eval_participant
    )

    booster_hash_after_alignment = (
        xgboost_state_sha256(model)
    )

    # Affine participant shift must be inverted.
    np.testing.assert_allclose(
        X_eval_aligned,
        X_eval_reference,
        rtol=1e-9,
        atol=1e-9,
    )

    reference_proba = model.predict_proba(
        X_eval_reference
    )
    aligned_proba = model.predict_proba(
        X_eval_aligned
    )

    np.testing.assert_allclose(
        aligned_proba,
        reference_proba,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_array_equal(
        model.predict(X_eval_aligned),
        model.predict(X_eval_reference),
    )

    # Feature adaptation must never mutate the global model.
    assert (
        booster_hash_before
        == booster_hash_after_alignment
    )

    manifest = aligner.to_manifest()

    assert manifest["reference_fitted"] is True
    assert manifest["calibration_fitted"] is True
    assert manifest["reference_n_samples"] == len(
        X_train
    )
    assert manifest["calibration_n_samples"] == len(
        X_calibration
    )
    assert manifest["n_features"] == X_train.shape[1]


def test_shifted_features_change_xgboost_input_without_alignment():
    X_train, y_train = _make_reference_data()

    model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    shift = np.asarray(
        [50.0, 100.0, -25.0, 40.0]
    )

    reference_pred = model.predict(X_train)
    shifted_pred = model.predict(
        X_train + shift
    )

    # Demonstrates why independently shifted participant
    # coordinates are a problem for fixed tree thresholds.
    assert not np.array_equal(
        reference_pred,
        shifted_pred,
    )
