from __future__ import annotations

import numpy as np
import pytest

from cogstate.preprocessing import FilterConfig, StreamingFilter, apply_causal, apply_offline


def test_filter_config_rejects_frequencies_outside_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        FilterConfig(sample_rate=100.0, bandpass_high_hz=50.0)


def test_streaming_filter_has_no_zero_state_startup_spike() -> None:
    config = FilterConfig(sample_rate=128.0, notch_enabled=False)
    filtered = StreamingFilter(config, 3).process(np.full((256, 3), 7.0))
    assert np.max(np.abs(filtered)) < 1e-9


def test_chunked_streaming_matches_whole_record_causal() -> None:
    rng = np.random.default_rng(5)
    signal = rng.normal(size=(1000, 4))
    config = FilterConfig(sample_rate=256.0)
    streaming = StreamingFilter(config, 4)
    chunked = np.vstack(
        [streaming.process(signal[start : start + 73]) for start in range(0, len(signal), 73)]
    )
    np.testing.assert_allclose(chunked, apply_causal(signal, config), atol=1e-12)


def test_streaming_reset_restarts_from_first_sample_state() -> None:
    signal = np.random.default_rng(6).normal(size=(300, 2))
    streaming = StreamingFilter(FilterConfig(sample_rate=256.0), 2)
    first = streaming.process(signal)
    streaming.reset()
    second = streaming.process(signal)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "config",
    [
        FilterConfig(sample_rate=256.0, bandpass_enabled=False),
        FilterConfig(sample_rate=256.0, notch_enabled=False),
        FilterConfig(sample_rate=256.0, bandpass_enabled=False, notch_enabled=False),
    ],
)
def test_optional_filter_stages_work_in_causal_and_offline_modes(config: FilterConfig) -> None:
    signal = np.random.default_rng(7).normal(size=(512, 3))
    assert np.isfinite(apply_causal(signal, config)).all()
    assert np.isfinite(apply_offline(signal, config)).all()


@pytest.mark.parametrize(
    "values",
    [np.zeros(10), np.empty((0, 2)), np.full((10, 2), np.nan), np.full((10, 2), np.inf)],
)
def test_filter_entry_points_reject_invalid_input(values: np.ndarray) -> None:
    config = FilterConfig(sample_rate=256.0)
    with pytest.raises(ValueError):
        apply_causal(values, config)
    with pytest.raises(ValueError):
        apply_offline(values, config)
