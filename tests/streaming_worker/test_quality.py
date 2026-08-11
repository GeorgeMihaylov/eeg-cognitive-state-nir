import numpy as np

from apps.streaming_worker.config import QualityConfig
from apps.streaming_worker.quality import EEGQualityGate
from cogstate.streaming.buffer import Window


def make_window(signal, timestamps, end=1.0):
    return Window(
        start_time=0.0,
        end_time=end,
        data={"eeg": np.asarray(signal)},
        timestamps={"eeg": np.asarray(timestamps)},
    )


def test_quality_accepts_complete_regular_window():
    sample_rate = 128
    timestamps = np.arange(sample_rate) / sample_rate
    report = EEGQualityGate(
        sample_rate=sample_rate,
        n_channels=4,
        config=QualityConfig(),
    ).evaluate(make_window(np.ones((sample_rate, 4)), timestamps))

    assert report.valid
    assert report.status == "good"
    assert report.sample_count == sample_rate


def test_quality_rejects_channel_and_sample_loss():
    sample_rate = 128
    timestamps = np.arange(100) / sample_rate
    report = EEGQualityGate(
        sample_rate=sample_rate,
        n_channels=4,
        config=QualityConfig(max_missing_ratio=0.05),
    ).evaluate(make_window(np.ones((100, 3)), timestamps))

    assert not report.valid
    assert "channel_count_mismatch" in report.reasons
    assert "sample_count_mismatch" in report.reasons
