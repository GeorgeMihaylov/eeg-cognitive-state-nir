from __future__ import annotations

import numpy as np
import pytest

from cogstate.features import (
    CANONICAL_REGIONS,
    EMOTIV_14_CHANNELS,
    FEATURE_SCHEMA_VERSION,
    FeaturePipeline,
    FeaturePipelineConfig,
    RegionalFeatureConfig,
    RegionalFeaturePipeline,
)


STANDARD_64 = (
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10", "Oz",
)


def _pipeline(**kwargs: object) -> RegionalFeaturePipeline:
    return RegionalFeaturePipeline(RegionalFeatureConfig(sample_rate=256.0, **kwargs))


def _montages() -> tuple[tuple[str, ...], ...]:
    return (
        EMOTIV_14_CHANNELS[:8],
        EMOTIV_14_CHANNELS,
        STANDARD_64[:32],
        STANDARD_64,
    )


def test_fixed_schema_across_8_14_32_64_channel_montages() -> None:
    pipeline = _pipeline()
    rng = np.random.default_rng(42)
    names = pipeline.feature_names()
    schema_hash = pipeline.schema_hash()
    montage_hashes: list[str] = []
    for channels in _montages():
        vector = pipeline.transform_window(
            rng.normal(size=(256, len(channels))), channel_names=channels
        )
        assert vector.shape == (728,)
        assert len(names) == 728
        assert pipeline.feature_names() == names
        assert pipeline.schema_hash() == schema_hash
        assert np.isfinite(vector).all()
        montage_hashes.append(pipeline.montage_hash(channels))
    assert len(set(montage_hashes)) == 4


def test_channel_permutation_invariance_and_determinism() -> None:
    pipeline = _pipeline()
    rng = np.random.default_rng(7)
    channels = EMOTIV_14_CHANNELS
    window = rng.normal(size=(384, len(channels)))
    permutation = rng.permutation(len(channels))
    expected = pipeline.transform_window(window, channel_names=channels)
    actual = pipeline.transform_window(
        window[:, permutation],
        channel_names=tuple(channels[index] for index in permutation),
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        pipeline.transform_window(window, channel_names=channels), expected
    )
    assert pipeline.montage_manifest(channels) == pipeline.montage_manifest(channels)
    assert pipeline.montage_hash(channels) == pipeline.montage_hash(channels)


def test_missing_region_is_finite_and_distinct_from_measured_zero() -> None:
    pipeline = _pipeline(include_spectral=False, include_entropy=False)
    vector = pipeline.transform_window(
        np.zeros((128, 1), dtype=float), channel_names=("AF3",)
    )
    values = dict(zip(pipeline.feature_names(), vector))
    assert np.isfinite(vector).all()
    assert values["statistical__mean__frontal_left__median"] == 0.0
    assert values["statistical__mean__frontal_right__median"] == 0.0
    assert values["coverage__frontal_left__present"] == 1.0
    assert values["coverage__frontal_left__channel_count"] == 1.0
    assert values["coverage__frontal_right__present"] == 0.0
    assert values["coverage__frontal_right__channel_count"] == 0.0


def test_emotiv_14_mapping_is_explicit_and_physiologically_ordered() -> None:
    manifest = _pipeline().montage_manifest(EMOTIV_14_CHANNELS)
    mapping = {
        row["channel_name"]: row["region"] for row in manifest["channels"]
    }
    assert mapping == {
        "AF3": "frontal_left", "F7": "frontal_left", "F3": "frontal_left",
        "FC5": "central_left", "T7": "temporal_left", "P7": "parietal_left",
        "O1": "occipital_left", "O2": "occipital_right",
        "P8": "parietal_right", "T8": "temporal_right",
        "FC6": "central_right", "F4": "frontal_right",
        "F8": "frontal_right", "AF4": "frontal_right",
    }
    assert sum(manifest["region_channel_counts"].values()) == 14
    assert set(manifest["region_channel_counts"]) == set(CANONICAL_REGIONS)


def test_midline_and_custom_mapping_are_explicit() -> None:
    standard = _pipeline().montage_manifest(("Fz", "Cz", "Pz", "Oz"))
    assert [row["region"] for row in standard["channels"]] == [
        "frontal_midline", "central_midline", "parietal_midline", "occipital_midline"
    ]
    custom = _pipeline(
        custom_channel_mapping={"DEVICE_A": "central_left", "DEVICE_B": "central_right"}
    )
    manifest = custom.montage_manifest(("device_a", "device_b"))
    assert [row["region"] for row in manifest["channels"]] == [
        "central_left", "central_right"
    ]
    assert {row["mapping_source"] for row in manifest["channels"]} == {"custom"}
    assert custom.schema_hash() == _pipeline().schema_hash()
    assert "DEVICE_A" not in str(custom.feature_specification())


def test_schema_hash_tracks_semantics_but_not_montage_mapping() -> None:
    baseline = _pipeline()
    custom_left = _pipeline(custom_channel_mapping={"X": "frontal_left"})
    custom_right = _pipeline(custom_channel_mapping={"X": "frontal_right"})
    assert baseline.schema_hash() == custom_left.schema_hash()
    assert custom_left.schema_hash() == custom_right.schema_hash()
    assert custom_left.montage_hash(("X",)) != custom_right.montage_hash(("X",))
    assert _pipeline(missing_fill_value=-1.0).schema_hash() != baseline.schema_hash()
    assert _pipeline(include_entropy=False).schema_hash() != baseline.schema_hash()


@pytest.mark.parametrize(
    "channels,custom,message",
    [
        (("AF3", "af3"), None, "unique"),
        (("UNKNOWN",), None, "explicit custom mapping"),
        (("UNKNOWN",), {"UNKNOWN": "not_a_region"}, "unknown region"),
        ((), None, "non-empty"),
    ],
)
def test_invalid_duplicate_and_unknown_channels_fail_clearly(
    channels: tuple[str, ...], custom: dict[str, str] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _pipeline(custom_channel_mapping=custom).montage_manifest(channels)


def test_batch_equals_stack_of_single_windows() -> None:
    pipeline = _pipeline(include_entropy=False)
    windows = np.random.default_rng(13).normal(size=(3, 256, 14))
    batch = pipeline.transform_batch(
        windows, channel_names=EMOTIV_14_CHANNELS, chunk_size=2
    )
    expected = np.stack(
        [
            pipeline.transform_window(window, channel_names=EMOTIV_14_CHANNELS)
            for window in windows
        ]
    )
    np.testing.assert_array_equal(batch, expected)


def test_duplicate_channels_within_region_preserve_median_and_zero_iqr() -> None:
    pipeline = _pipeline(include_spectral=False, include_entropy=False)
    time = np.arange(256, dtype=float) / 256.0
    signal = np.sin(2.0 * np.pi * 10.0 * time)
    one = pipeline.transform_window(signal[:, None], channel_names=("F3",))
    three = pipeline.transform_window(
        np.column_stack((signal, signal, signal)), channel_names=("F1", "F3", "F5")
    )
    one_values = dict(zip(pipeline.feature_names(), one))
    three_values = dict(zip(pipeline.feature_names(), three))
    for feature_name in pipeline.feature_names():
        if feature_name.startswith("statistical__") and "__frontal_left__" in feature_name:
            assert three_values[feature_name] == pytest.approx(one_values[feature_name])
        if feature_name.startswith("statistical__") and feature_name.endswith("__iqr"):
            assert three_values[feature_name] == pytest.approx(0.0)


def test_v1_pipeline_schema_and_hash_remain_unchanged() -> None:
    channels = tuple(f"C{index}" for index in range(14))
    pipeline = FeaturePipeline(
        FeaturePipelineConfig(sample_rate=256.0, channel_names=channels)
    )
    assert FEATURE_SCHEMA_VERSION == "cogstate-features-v1"
    assert len(pipeline.feature_names()) == 371
    assert pipeline.feature_hash() == (
        "a06eb9e844c229366e604768c3e9a47a16790731e5be2b85622376f3bac2b493"
    )
