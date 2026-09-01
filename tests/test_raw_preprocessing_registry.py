from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bench.datasets.raw_preprocessing import (
    PreprocessingSpec,
    apply_preprocessing_spec,
    apply_raw_preprocessing,
    get_preprocessing_component,
    get_preprocessing_registry,
)


def _spec(*, bandpass: bool = False, notch: bool = False, car: bool = False):
    return PreprocessingSpec.from_dict(
        {
            "target_sampling_rate": 256,
            "padding_seconds": 2,
            "window_seconds": 10,
            "output_dtype": "float32",
            "bandpass": {
                "enabled": bandpass,
                "low_hz": 1.0,
                "high_hz": 45.0,
                "order": 4,
            },
            "notch": {
                "enabled": notch,
                "frequency_hz": 50.0,
                "q": 30.0,
            },
            "car": {"enabled": car},
        }
    )


def _amplitude(signal: np.ndarray, frequency_hz: float, sfreq: float = 256.0):
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sfreq)
    spectrum = np.abs(np.fft.rfft(signal))
    return float(spectrum[np.argmin(np.abs(frequencies - frequency_hz))])


def test_registry_contains_stable_components_and_order():
    registry = get_preprocessing_registry()
    assert set(registry) == {"identity", "bandpass", "notch", "car"}
    assert [
        component.name
        for component in sorted(
            registry.values(), key=lambda component: component.execution_order
        )
    ] == ["identity", "bandpass", "notch", "car"]
    assert all(component.fit_scope == "stateless" for component in registry.values())
    assert all(component.cacheable for component in registry.values())


def test_unknown_component_and_parameter_raise_clear_errors():
    with pytest.raises(ValueError, match="Unknown preprocessing component"):
        get_preprocessing_component("ica")
    with pytest.raises(ValueError, match="Unknown preprocessing parameters"):
        PreprocessingSpec.from_dict({"ica": {"enabled": True}})
    with pytest.raises(ValueError, match="Unknown preprocessing.bandpass parameters"):
        PreprocessingSpec.from_dict({"bandpass": {"enabled": True, "bad": 1}})


def test_serialization_and_hash_are_stable_under_dict_key_order():
    first = PreprocessingSpec.from_dict(
        {
            "bandpass": {
                "enabled": True,
                "low_hz": 1.0,
                "high_hz": 45.0,
                "order": 4,
            },
            "notch": {"enabled": True, "frequency_hz": 50.0, "q": 30.0},
            "car": {"enabled": True},
        }
    )
    second = PreprocessingSpec.from_dict(
        {
            "car": {"enabled": True},
            "notch": {"q": 30.0, "frequency_hz": 50.0, "enabled": True},
            "bandpass": {
                "order": 4,
                "high_hz": 45.0,
                "low_hz": 1.0,
                "enabled": True,
            },
        }
    )
    assert first.stable_serialization() == second.stable_serialization()
    assert first.stable_hash(channels=["A", "B"]) == second.stable_hash(
        channels=["A", "B"]
    )
    assert [step.execution_order for step in first.steps] == [0, 10, 20, 30]


def test_invalid_values_are_rejected_without_side_effects():
    with pytest.raises(ValueError, match="bandpass order 4"):
        PreprocessingSpec.from_dict({"bandpass": {"enabled": True, "order": 2}})
    with pytest.raises(ValueError, match="below Nyquist"):
        PreprocessingSpec.from_dict(
            {"notch": {"enabled": True, "frequency_hz": 200.0}}
        )
    with pytest.raises(ValueError, match="only supports float32"):
        PreprocessingSpec.from_dict({"output_dtype": "float64"})


def test_identity_preserves_signal_shape_dtype_and_values():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(4, 2560)).astype(np.float32)
    output = apply_preprocessing_spec(signal, sampling_rate=256.0, spec=_spec())
    assert output.shape == signal.shape
    assert output.dtype == np.float32
    assert np.array_equal(output, signal)
    assert np.isfinite(output).all()


def test_bandpass_suppresses_out_of_band_frequencies():
    sfreq = 256.0
    time = np.arange(2560) / sfreq
    signal = (
        np.sin(2 * np.pi * 0.5 * time)
        + np.sin(2 * np.pi * 10.0 * time)
        + np.sin(2 * np.pi * 60.0 * time)
    ).astype(np.float32)
    output = apply_preprocessing_spec(
        signal[None, :], sampling_rate=sfreq, spec=_spec(bandpass=True)
    )[0]
    assert _amplitude(output, 0.5) < 0.25 * _amplitude(signal, 0.5)
    assert _amplitude(output, 60.0) < 0.25 * _amplitude(signal, 60.0)
    assert _amplitude(output, 10.0) > 0.70 * _amplitude(signal, 10.0)


def test_notch_attenuates_50_hz_and_preserves_allowed_frequency():
    sfreq = 256.0
    time = np.arange(2560) / sfreq
    signal = (
        np.sin(2 * np.pi * 10.0 * time)
        + np.sin(2 * np.pi * 50.0 * time)
    ).astype(np.float32)
    output = apply_preprocessing_spec(
        signal[None, :], sampling_rate=sfreq, spec=_spec(notch=True)
    )[0]
    assert _amplitude(output, 50.0) < 0.20 * _amplitude(signal, 50.0)
    assert _amplitude(output, 10.0) > 0.80 * _amplitude(signal, 10.0)


def test_car_centers_channels_and_full_pipeline_matches_legacy_execution():
    rng = np.random.default_rng(7)
    signal = rng.normal(size=(14, 2560)).astype(np.float32)
    car_output = apply_preprocessing_spec(
        signal, sampling_rate=256.0, spec=_spec(car=True)
    )
    assert np.max(np.abs(car_output.mean(axis=0))) < 1e-5

    full_spec = _spec(bandpass=True, notch=True, car=True)
    registered = apply_preprocessing_spec(
        signal, sampling_rate=256.0, spec=full_spec
    )
    legacy = apply_raw_preprocessing(
        signal,
        sampling_rate=256.0,
        config=full_spec.to_legacy_raw_preprocessing(),
    )
    assert registered.shape == signal.shape
    assert registered.dtype == np.float32
    assert np.isfinite(registered).all()
    assert np.array_equal(registered, legacy)


def test_global_cache_rejects_outer_train_only_step():
    stateless = _spec(bandpass=True)
    stateless.assert_global_cacheable()
    steps = tuple(
        replace(step, fit_scope="outer_train_only")
        if step.name == "bandpass"
        else step
        for step in stateless.steps
    )
    stateful = replace(stateless, steps=steps)
    with pytest.raises(ValueError, match="Global raw cache requires"):
        stateful.assert_global_cacheable()

