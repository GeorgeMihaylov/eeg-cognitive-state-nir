import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.datasets.raw_eeg_window_dataset import (
    CANONICAL_EEG_CHANNELS,
    RawEEGWindowArrayView,
    RawEEGWindowDataset,
    RawEEGWindowError,
    _valid_cache_shard,
    extract_raw_eeg_window,
    load_raw_eeg_record,
)
from bench.datasets.logical_recordings import (
    build_deduplication_selection,
    infer_record_group_id,
)
from bench.datasets.raw_preprocessing import (
    apply_raw_preprocessing,
    raw_preprocessing_hash,
)


def _write_raw_csv(
    tmp_path: Path,
    *,
    sampling_rate: float = 128.0,
    missing_rows: int = 0,
) -> pd.Series:
    n_samples = int(10 * sampling_rate)
    timestamps = 1_700_000_000.0 + np.arange(n_samples) / sampling_rate
    if missing_rows:
        keep = np.ones(n_samples, dtype=bool)
        keep[100 : 100 + missing_rows] = False
        timestamps = timestamps[keep]
    frame = pd.DataFrame({"Timestamp": timestamps})
    for channel_index, channel in enumerate(reversed(CANONICAL_EEG_CHANNELS)):
        frame[channel] = channel_index + np.sin(
            np.arange(len(timestamps)) / sampling_rate
        )
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    return pd.Series(
        {
            "main_path": str(path),
            "main_rel_path": str(path),
            "header_row": 0,
            "separator": ",",
            "time_columns": json.dumps(["Timestamp"]),
            "eeg_columns": json.dumps(list(CANONICAL_EEG_CHANNELS)),
        }
    )


def test_raw_loader_uses_timestamp_grid_channel_order_and_resampling(tmp_path):
    row = _write_raw_csv(tmp_path, sampling_rate=128.0)
    raw = load_raw_eeg_record(row)
    window, diagnostics = extract_raw_eeg_window(
        raw,
        -5.0,
        5.0,
        target_sfreq=256.0,
        max_missing_fraction=0.02,
    )

    assert raw.channels == CANONICAL_EEG_CHANNELS
    assert window.shape == (14, 2560)
    assert window.dtype == np.float32
    assert np.isfinite(window).all()
    assert diagnostics["resampled"] is True
    assert diagnostics["sfreq_regularized"] == 128.0
    # CSV columns were reversed; the returned order is canonical.
    assert window[0].mean() > window[-1].mean()


def test_raw_loader_rejects_window_above_missing_fraction(tmp_path):
    row = _write_raw_csv(tmp_path, sampling_rate=128.0, missing_rows=40)
    raw = load_raw_eeg_record(row)

    with pytest.raises(RawEEGWindowError, match="Missing fraction") as error:
        extract_raw_eeg_window(
            raw,
            -5.0,
            5.0,
            target_sfreq=128.0,
            max_missing_fraction=0.02,
        )

    assert error.value.reason == "missing_fraction_exceeded"


def test_raw_loader_collapses_duplicate_timestamps(tmp_path):
    row = _write_raw_csv(tmp_path, sampling_rate=128.0)
    path = Path(row["main_path"])
    frame = pd.read_csv(path)
    frame = pd.concat([frame.iloc[[0]], frame], ignore_index=True)
    frame.to_csv(path, index=False)
    raw = load_raw_eeg_record(row)

    window, diagnostics = extract_raw_eeg_window(
        raw, -5.0, 5.0, target_sfreq=128.0, max_missing_fraction=0.02
    )

    assert window.shape == (14, 1280)
    assert diagnostics["missing_fraction"] == 0.0
    assert np.isfinite(window).all()


def test_lazy_raw_view_shape_indexing_and_channel_statistics(tmp_path):
    array = np.arange(6 * 3 * 8, dtype=np.float32).reshape(6, 3, 8)
    cache_path = tmp_path / "record.npy"
    np.save(cache_path, array)
    manifest = pd.DataFrame(
        {
            "sample_id": np.arange(6),
            "status": "ok",
            "cache_file": str(cache_path),
            "cache_offset": np.arange(6),
            "n_channels": 3,
            "n_samples_expected": 8,
        }
    )
    view = RawEEGWindowArrayView(manifest)

    assert view.shape == (6, 1, 3, 8)
    assert view[2].shape == (1, 3, 8)
    subset = view[np.asarray([1, 4])]
    assert subset.shape == (2, 1, 3, 8)
    mean, scale = view.compute_channel_statistics()
    expected = array.transpose(1, 0, 2).reshape(3, -1)
    np.testing.assert_allclose(mean, expected.mean(axis=1), rtol=1e-6)
    np.testing.assert_allclose(scale, expected.std(axis=1), rtol=1e-6)
    normalized = view.with_channel_normalization(mean, scale)
    assert np.isfinite(normalized[0]).all()


def test_corrupted_cache_shard_is_not_reused(tmp_path):
    array_path = tmp_path / "record.npy"
    metadata_path = tmp_path / "record.json"
    np.save(array_path, np.zeros((2, 3, 8), dtype=np.float32))
    metadata_path.write_text(
        json.dumps({"config_hash": "expected", "accepted_windows": 2}),
        encoding="utf-8",
    )
    assert _valid_cache_shard(
        array_path, metadata_path, "expected", (3, 8)
    ) is not None

    payload = array_path.read_bytes()
    array_path.write_bytes(payload[: len(payload) // 2])

    assert _valid_cache_shard(
        array_path, metadata_path, "expected", (3, 8)
    ) is None


def _spectral_amplitude(signal: np.ndarray, frequency: float, sfreq: float) -> float:
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sfreq)
    spectrum = np.abs(np.fft.rfft(signal))
    return float(spectrum[np.argmin(np.abs(frequencies - frequency))])


def test_bandpass_suppresses_frequency_outside_passband():
    sfreq = 256.0
    time = np.arange(int(8 * sfreq)) / sfreq
    signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 80 * time)
    output = apply_raw_preprocessing(
        signal[None, :].astype(np.float32),
        sampling_rate=sfreq,
        config={
            "resample_hz": sfreq,
            "bandpass": {"enabled": True, "low_hz": 1, "high_hz": 45},
        },
    )

    assert _spectral_amplitude(output[0], 10, sfreq) > 20 * _spectral_amplitude(
        output[0], 80, sfreq
    )


def test_notch_suppresses_50_hz():
    sfreq = 256.0
    time = np.arange(int(8 * sfreq)) / sfreq
    signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 50 * time)
    output = apply_raw_preprocessing(
        signal[None, :].astype(np.float32),
        sampling_rate=sfreq,
        config={
            "resample_hz": sfreq,
            "notch": {
                "enabled": True,
                "frequency_hz": 50,
                "quality_factor": 30,
            },
        },
    )

    assert _spectral_amplitude(output[0], 10, sfreq) > 10 * _spectral_amplitude(
        output[0], 50, sfreq
    )


def test_common_average_reference_preserves_shape_order_and_float32():
    signals = np.vstack([
        np.full(128, 1.0),
        np.full(128, 3.0),
        np.linspace(-2.0, 2.0, 128),
    ]).astype(np.float32)
    output = apply_raw_preprocessing(
        signals,
        sampling_rate=128.0,
        config={
            "resample_hz": 128.0,
            "rereference": {"mode": "common_average"},
        },
    )

    assert output.shape == signals.shape
    assert output.dtype == np.float32
    np.testing.assert_allclose(output.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(output[1] - output[0], 2.0, atol=1e-6)


def test_cache_preprocessing_hash_changes_with_filters_and_channel_order():
    raw_hash = raw_preprocessing_hash(None, channels=["A", "B"])
    filtered_hash = raw_preprocessing_hash(
        {"bandpass": {"enabled": True, "low_hz": 1, "high_hz": 45}},
        channels=["A", "B"],
    )
    reordered_hash = raw_preprocessing_hash(None, channels=["B", "A"])

    assert len({raw_hash, filtered_hash, reordered_hash}) == 3


def _logical_manifest(tmp_path: Path) -> pd.DataFrame:
    array_path = tmp_path / "logical.npy"
    np.save(array_path, np.ones((4, 2, 8), dtype=np.float32))
    return pd.DataFrame({
        "sample_id": [0, 1, 2, 3],
        "record_id": [
            "gpn_data__S01__day1__recordA",
            "Old_EEG__S01__day1__recordA",
            "gpn_data__S02__day1__recordB",
            "gpn_data__S02__day1__recordB",
        ],
        "record_group_id": [
            "S01__day1__recordA",
            "S01__day1__recordA",
            "S02__day1__recordB",
            "S02__day1__recordB",
        ],
        "source": ["gpn_data", "Old_EEG", "gpn_data", "gpn_data"],
        "subject_id": ["S01", "S01", "S02", "S02"],
        "status": ["ok"] * 4,
        "label_q5": [0, 0, 1, 1],
        "cache_file": str(array_path),
        "cache_offset": [0, 1, 2, 3],
        "n_channels": 2,
        "n_samples_expected": 8,
        "sfreq_target": 128.0,
        "sfreq_original": 128.0,
        "outer_fold": [1, 1, 2, 2],
        "missing_fraction": [0.0] * 4,
        "raw_n_rows": [1000, 1000, 900, 900],
    })


def test_logical_duplicate_identity_and_deduplication_are_deterministic(tmp_path):
    manifest = _logical_manifest(tmp_path)
    assert infer_record_group_id(manifest.loc[0, "record_id"]) == infer_record_group_id(
        manifest.loc[1, "record_id"]
    )
    first = build_deduplication_selection(manifest)
    second = build_deduplication_selection(
        manifest.sample(frac=1.0, random_state=17)
    )
    first_selected = first.loc[first["selected"], "record_id"].tolist()
    second_selected = second.loc[second["selected"], "record_id"].tolist()
    assert first_selected == second_selected
    assert "gpn_data__S01__day1__recordA" in first_selected


def test_deduplicated_dataset_counts_each_logical_record_once(tmp_path):
    manifest = _logical_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    dataset = RawEEGWindowDataset({
        "data_path": manifest_path,
        "target_col": "label_q5",
        "dataset_mode": "raw_deduplicated_logical_records",
        "raw_preprocessing": {"resample_hz": 128.0},
        "channel_names": ["A", "B"],
    })

    data = dataset.load()
    retained = data.data.manifest.drop_duplicates("record_id")
    assert retained["record_group_id"].is_unique
    assert len(data.labels) == 3
    assert data.metadata["removed_source_records"] == 1
