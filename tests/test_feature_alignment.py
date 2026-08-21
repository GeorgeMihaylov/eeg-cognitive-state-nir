import numpy as np
import pytest

from cogstate.adaptation.feature_alignment import (
    FeatureAligner,
    FeatureAlignmentConfig,
    apply_alignment_shrinkage,
)


@pytest.mark.parametrize(
    "method",
    [
        "standard_location_scale",
        "robust_location_scale",
    ],
)
def test_affine_participant_shift_maps_back_to_reference(method):
    reference = np.asarray(
        [
            [0.0, 10.0, -4.0],
            [1.0, 12.0, -1.0],
            [2.0, 15.0, 2.0],
            [4.0, 18.0, 5.0],
            [7.0, 24.0, 10.0],
        ],
        dtype=float,
    )

    scale = np.asarray([2.0, 4.0, 0.5])
    shift = np.asarray([20.0, -100.0, 50.0])

    calibration = reference * scale + shift

    aligner = FeatureAligner(
        FeatureAlignmentConfig(method=method)
    )
    aligner.fit_reference(reference)
    aligner.fit_calibration(calibration)

    aligned = aligner.transform(calibration)

    np.testing.assert_allclose(
        aligned,
        reference,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "method",
    [
        "standard_location_scale",
        "robust_location_scale",
    ],
)
def test_unseen_evaluation_rows_use_calibration_statistics_only(method):
    reference = np.asarray(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 8.0],
            [8.0, 15.0],
        ],
        dtype=float,
    )

    scale = np.asarray([3.0, 2.0])
    shift = np.asarray([50.0, -20.0])

    calibration = reference * scale + shift

    reference_eval = np.asarray(
        [
            [10.0, 20.0],
            [-2.0, -4.0],
        ],
        dtype=float,
    )

    participant_eval = (
        reference_eval * scale + shift
    )

    aligner = FeatureAligner(
        FeatureAlignmentConfig(method=method)
    )

    aligner.fit_reference(reference)
    aligner.fit_calibration(calibration)

    before_manifest = aligner.to_manifest()
    aligned_eval = aligner.transform(participant_eval)
    after_manifest = aligner.to_manifest()

    np.testing.assert_allclose(
        aligned_eval,
        reference_eval,
        rtol=1e-10,
        atol=1e-10,
    )

    # Evaluation data must not update any fitted state.
    assert before_manifest == after_manifest


def test_transform_requires_reference_and_calibration_fit():
    aligner = FeatureAligner()
    X = np.ones((3, 2), dtype=float)

    with pytest.raises(
        RuntimeError,
        match="fit_reference",
    ):
        aligner.transform(X)

    aligner.fit_reference(X)

    with pytest.raises(
        RuntimeError,
        match="fit_calibration",
    ):
        aligner.transform(X)


def test_fit_calibration_requires_reference_fit():
    aligner = FeatureAligner()

    with pytest.raises(
        RuntimeError,
        match="fit_reference",
    ):
        aligner.fit_calibration(
            np.ones((3, 2), dtype=float)
        )


def test_feature_width_mismatch_is_rejected():
    aligner = FeatureAligner()

    aligner.fit_reference(
        np.ones((5, 3), dtype=float)
    )

    with pytest.raises(
        ValueError,
        match="expected 3",
    ):
        aligner.fit_calibration(
            np.ones((5, 4), dtype=float)
        )


def test_degenerate_calibration_feature_maps_to_reference_center():
    reference = np.asarray(
        [
            [0.0, 10.0],
            [2.0, 20.0],
            [4.0, 30.0],
            [6.0, 40.0],
        ],
        dtype=float,
    )

    calibration = np.asarray(
        [
            [100.0, 1.0],
            [100.0, 2.0],
            [100.0, 3.0],
            [100.0, 4.0],
        ],
        dtype=float,
    )

    evaluation = np.asarray(
        [
            [90.0, 2.5],
            [110.0, 3.5],
        ],
        dtype=float,
    )

    aligner = FeatureAligner(
        FeatureAlignmentConfig(
            method="standard_location_scale"
        )
    )

    aligner.fit_reference(reference)
    aligner.fit_calibration(calibration)

    aligned = aligner.transform(evaluation)

    expected_reference_center = np.mean(
        reference[:, 0]
    )

    np.testing.assert_allclose(
        aligned[:, 0],
        expected_reference_center,
    )
    assert np.isfinite(aligned).all()

    manifest = aligner.to_manifest()
    assert (
        manifest[
            "calibration_degenerate_features"
        ]
        == 1
    )


def test_new_reference_invalidates_old_calibration():
    aligner = FeatureAligner()

    reference_a = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 4.0],
        ]
    )
    calibration = reference_a + 10.0

    aligner.fit_reference(reference_a)
    aligner.fit_calibration(calibration)

    assert aligner.is_calibration_fitted

    reference_b = reference_a + 100.0
    aligner.fit_reference(reference_b)

    assert aligner.is_reference_fitted
    assert not aligner.is_calibration_fitted

    with pytest.raises(
        RuntimeError,
        match="fit_calibration",
    ):
        aligner.transform(calibration)


def test_manifest_is_deterministic():
    reference = np.asarray(
        [
            [0.0, 2.0],
            [1.0, 4.0],
            [3.0, 8.0],
        ]
    )
    calibration = reference * 2.0 + 7.0

    first = FeatureAligner()
    first.fit_reference(reference)
    first.fit_calibration(calibration)

    second = FeatureAligner()
    second.fit_reference(reference.copy())
    second.fit_calibration(calibration.copy())

    assert first.to_manifest() == second.to_manifest()


@pytest.mark.parametrize(
    "bad_method",
    [
        "",
        "standard",
        "coral",
        "unknown",
    ],
)
def test_unknown_method_is_rejected(bad_method):
    with pytest.raises(ValueError):
        FeatureAlignmentConfig(
            method=bad_method
        )


def test_alignment_shrinkage_endpoints_and_validation():
    original = np.asarray([[0.0, 2.0], [4.0, 6.0]])
    aligned = np.asarray([[10.0, 12.0], [14.0, 16.0]])

    np.testing.assert_allclose(
        apply_alignment_shrinkage(original, aligned, 0.0),
        original,
    )
    np.testing.assert_allclose(
        apply_alignment_shrinkage(original, aligned, 1.0),
        aligned,
    )
    np.testing.assert_allclose(
        apply_alignment_shrinkage(original, aligned, 0.25),
        original + 0.25 * (aligned - original),
    )

    with pytest.raises(ValueError, match="alpha"):
        apply_alignment_shrinkage(original, aligned, 1.01)

