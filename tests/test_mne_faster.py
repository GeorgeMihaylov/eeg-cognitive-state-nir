import json

import numpy as np

from cogstate.preprocessing import (
    MNEFasterBundle,
    MNEFasterCalibrator,
    MNEFasterConfig,
)


CHANNELS = ("F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")


def _calibration_epochs() -> np.ndarray:
    rng = np.random.default_rng(12)
    samples = 256
    time = np.arange(samples) / 128.0
    epochs = []
    mixing = rng.normal(size=(len(CHANNELS), len(CHANNELS)))
    for _ in range(20):
        sources = np.column_stack(
            [
                np.sin(2 * np.pi * frequency * time)
                + 0.05 * rng.normal(size=samples)
                for frequency in (6, 8, 10, 12, 15, 18, 22, 28)
            ]
        )
        epochs.append(sources @ mixing.T)
    return np.asarray(epochs)


def _config() -> MNEFasterConfig:
    return MNEFasterConfig(
        sample_rate=128.0,
        channel_names=CHANNELS,
        input_scale_to_volts=1.0,
        ica_n_components=6,
        ica_max_iter=1000,
        power_gradient_range_hz=(1.0, 45.0),
        preprocessing_contract={
            "bandpass_low_hz": 1.0,
            "bandpass_high_hz": 45.0,
            "notch_hz": 50.0,
            "filter_mode": "causal",
        },
    )


def test_mne_faster_calibration_preserves_label_alignment() -> None:
    epochs = _calibration_epochs()

    cleaned, bundle, report = MNEFasterCalibrator(_config()).fit_transform(epochs)

    assert cleaned.shape[0] == report.kept_epoch_mask.sum()
    assert cleaned.shape[1:] == epochs.shape[1:]
    assert bundle.channel_names == CHANNELS
    assert tuple(bundle.ica.exclude) == report.bad_components
    assert np.isfinite(cleaned).all()


def test_mne_faster_bundle_round_trip_and_stream_transform(tmp_path) -> None:
    epochs = _calibration_epochs()
    _, bundle, _ = MNEFasterCalibrator(_config()).fit_transform(epochs)
    bundle.save(tmp_path)

    restored = MNEFasterBundle.load(tmp_path)
    restored.validate(
        sample_rate=128.0,
        channel_names=CHANNELS,
        preprocessing_contract=_config().preprocessing_contract,
    )
    cleaned = restored.transform(epochs[0])

    assert cleaned.shape == epochs[0].shape
    assert np.isfinite(cleaned).all()
    manifest = json.loads(
        (tmp_path / "mne-faster-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bad_components"] == restored.ica.exclude
