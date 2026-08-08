import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from cogstate.preprocessing.artifact_removal import (
    ArtifactICA,
    FasterConfig,
    IcaConfig,
    run_faster,
)
from cogstate.preprocessing.filtering import FilterConfig, StreamingFilter
from cogstate.preprocessing.pipeline import build_default_pipeline
from cogstate.streaming.buffer import Window


def test_run_faster_detects_artifacts_and_preserves_original_indices():
    rng = np.random.default_rng(42)

    epochs = rng.normal(size=(30, 256, 14))
    epochs[:, :, 3] *= 30.0
    epochs[11] *= 20.0

    cleaned, report = run_faster(epochs, FasterConfig())

    assert cleaned.shape == (29, 256, 14)
    assert np.isfinite(cleaned).all()

    assert report.bad_channels == [3]
    assert report.bad_epochs == [11]
    assert report.bad_components == []

    # These indices refer to the original 30-epoch array, not the
    # compressed array after bad epoch 11 has been removed.
    assert report.bad_channel_epoch_pairs == [(15, 2), (27, 12)]

    assert all(type(value) is int for value in report.bad_channels)
    assert all(type(value) is int for value in report.bad_epochs)


@pytest.mark.parametrize(
    "epochs, message",
    [
        (
            np.zeros((256, 14)),
            "run_faster expects",
        ),
        (
            np.zeros((0, 256, 14)),
            "zero epochs",
        ),
    ],
)
def test_run_faster_rejects_invalid_shapes(epochs, message):
    with pytest.raises(ValueError, match=message):
        run_faster(epochs)


def test_run_faster_rejects_nonfinite_input():
    epochs = np.zeros((5, 256, 14))
    epochs[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        run_faster(epochs)


def test_ica_reduces_components_to_signal_rank():
    rng = np.random.default_rng(42)

    sources = rng.laplace(size=(6000, 5))
    mixing = rng.normal(size=(5, 14))
    signal = sources @ mixing

    assert np.linalg.matrix_rank(signal) == 5

    with pytest.warns(RuntimeWarning, match="reduced from 14"):
        ica = ArtifactICA().fit(
            signal,
            sample_rate=256.0,
        )

    transformed = ica.transform(signal)

    assert ica.input_rank == 5
    assert ica.n_components == 5
    assert ica.converged is True
    assert transformed.shape == signal.shape
    assert np.isfinite(transformed).all()


def test_ica_is_deterministic():
    rng = np.random.default_rng(42)

    sources = rng.laplace(size=(5000, 8))
    mixing = rng.normal(size=(8, 14))
    signal = sources @ mixing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        first = ArtifactICA().fit(
            signal,
            sample_rate=256.0,
        )
        second = ArtifactICA().fit(
            signal,
            sample_rate=256.0,
        )

    first_output = first.transform(signal)
    second_output = second.transform(signal)

    assert first.artifact_components == second.artifact_components
    np.testing.assert_allclose(first_output, second_output)


def test_ica_reports_non_convergence():
    rng = np.random.default_rng(42)

    sources = rng.laplace(size=(4000, 14))
    mixing = rng.normal(size=(14, 14))
    signal = sources @ mixing

    with pytest.warns(ConvergenceWarning):
        ica = ArtifactICA(
            IcaConfig(
                max_iter=1,
                random_state=42,
            )
        ).fit(
            signal,
            sample_rate=256.0,
        )

    assert ica.n_iter == 1
    assert ica.converged is False


def test_ica_rejects_rank_one_signal():
    signal = np.ones((1000, 14))

    with pytest.raises(ValueError, match="rank is too low"):
        ArtifactICA().fit(
            signal,
            sample_rate=256.0,
        )


def test_ica_rejects_channel_mismatch_at_transform():
    rng = np.random.default_rng(42)
    signal = rng.laplace(size=(3000, 14))

    ica = ArtifactICA().fit(
        signal,
        sample_rate=256.0,
    )

    with pytest.raises(ValueError, match="fitted on 14 channels"):
        ica.transform(
            np.zeros((100, 13))
        )


def test_streaming_filter_chunking_matches_single_call():
    rng = np.random.default_rng(42)

    sample_rate = 256.0
    n_samples = 4096
    n_channels = 14

    t = np.arange(n_samples) / sample_rate

    signal = np.column_stack(
        [
            np.sin(2 * np.pi * (6 + channel) * t)
            + 0.2 * np.sin(2 * np.pi * 50 * t)
            for channel in range(n_channels)
        ]
    )
    signal += rng.normal(0.0, 0.05, signal.shape)

    config = FilterConfig(sample_rate=sample_rate)

    whole_filter = StreamingFilter(config, n_channels)
    whole = whole_filter.process(signal)

    chunk_filter = StreamingFilter(config, n_channels)

    parts = []
    start = 0
    for stop in [173, 511, 1024, 1900, 2800, n_samples]:
        parts.append(
            chunk_filter.process(
                signal[start:stop]
            )
        )
        start = stop

    chunked = np.concatenate(parts, axis=0)

    np.testing.assert_allclose(
        whole,
        chunked,
        rtol=1e-10,
        atol=1e-10,
    )


def test_streaming_filter_reset_is_deterministic():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(1024, 14))

    streaming_filter = StreamingFilter(
        FilterConfig(sample_rate=256.0),
        n_channels=14,
    )

    first = streaming_filter.process(signal)
    streaming_filter.reset()
    second = streaming_filter.process(signal)

    np.testing.assert_allclose(
        first,
        second,
        rtol=1e-12,
        atol=1e-12,
    )


def test_streaming_filter_rejects_channel_mismatch():
    streaming_filter = StreamingFilter(
        FilterConfig(sample_rate=256.0),
        n_channels=14,
    )

    with pytest.raises(ValueError, match="configured for 14 channels"):
        streaming_filter.process(
            np.zeros((256, 13))
        )


def test_filter_config_rejects_frequency_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        FilterConfig(sample_rate=64.0)


def test_preprocessing_pipeline_preserves_shape_without_ica():
    rng = np.random.default_rng(42)

    signal = rng.normal(size=(512, 14))

    window = Window(
        start_time=0.0,
        end_time=2.0,
        data={"eeg": signal},
        timestamps={
            "eeg": np.arange(512) / 256.0,
        },
    )

    pipeline = build_default_pipeline(
        sample_rate=256.0,
        n_channels=14,
    )

    output = pipeline(window)

    assert output.shape == signal.shape
    assert np.isfinite(output).all()


def test_preprocessing_pipeline_preserves_shape_with_ica():
    rng = np.random.default_rng(42)

    calibration = rng.laplace(
        size=(5000, 14)
    )

    ica = ArtifactICA().fit(
        calibration,
        sample_rate=256.0,
    )

    signal = rng.laplace(
        size=(512, 14)
    )

    window = Window(
        start_time=0.0,
        end_time=2.0,
        data={"eeg": signal},
        timestamps={
            "eeg": np.arange(512) / 256.0,
        },
    )

    pipeline = build_default_pipeline(
        sample_rate=256.0,
        n_channels=14,
        ica=ica,
    )

    output = pipeline(window)

    assert ica.converged is True
    assert output.shape == signal.shape
    assert np.isfinite(output).all()


def test_streaming_filter_suppresses_large_initial_dc_offset():
    signal = np.full(
        (512, 14),
        4370.0,
        dtype=float,
    )

    streaming_filter = StreamingFilter(
        FilterConfig(sample_rate=256.0),
        n_channels=14,
    )

    output = streaming_filter.process(signal)

    assert np.isfinite(output).all()

    # The initial filter state must be scaled to the first observed
    # sample. Otherwise the large Emotiv DC offset creates an artificial
    # transient of several thousand units at stream startup.
    assert float(np.max(np.abs(output))) < 1e-3
