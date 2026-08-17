import numpy as np
import pytest

from cogstate.preprocessing import (
    FilterConfig,
    OfflinePreprocessingConfig,
    OfflinePreprocessingPipeline,
    StreamingFilter,
    apply_causal,
)


def test_filter_config_rejects_frequencies_outside_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        FilterConfig(sample_rate=100.0, bandpass_high_hz=50.0)


def test_streaming_filter_has_no_zero_state_startup_spike() -> None:
    config = FilterConfig(sample_rate=128.0, notch_freq_hz=None)
    signal = np.full((256, 3), 7.0)

    filtered = StreamingFilter(config, 3).process(signal)

    assert np.max(np.abs(filtered)) < 1e-9


def test_chunked_streaming_matches_offline_causal_mode() -> None:
    rng = np.random.default_rng(5)
    signal = rng.normal(size=(1000, 4))
    config = FilterConfig(sample_rate=256.0)
    streaming = StreamingFilter(config, 4)

    chunked = np.vstack(
        [streaming.process(signal[start : start + 73]) for start in range(0, len(signal), 73)]
    )

    np.testing.assert_allclose(chunked, apply_causal(signal, config), atol=1e-12)


def test_offline_pipeline_exposes_streaming_equivalent_filter_mode() -> None:
    rng = np.random.default_rng(6)
    signal = rng.normal(size=(1024, 4))
    config = OfflinePreprocessingConfig(
        sample_rate=256.0,
        filter_mode="causal",
        detrend_order=None,
        reference_method="none",
        detect_and_interpolate_bad_channels=False,
    )

    result = OfflinePreprocessingPipeline(config).transform(signal)

    np.testing.assert_allclose(
        result.values, apply_causal(signal, config.filter_config), atol=1e-12
    )
    assert result.report.filter_mode == "causal"
