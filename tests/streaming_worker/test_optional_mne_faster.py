import numpy as np
import pytest

from apps.streaming_worker.config import WorkerConfig
from apps.streaming_worker.runtime import _WindowArtifactPreprocessor
from cogstate.streaming.buffer import Window


def _window() -> Window:
    signal = np.arange(32, dtype=float).reshape(8, 4)
    return Window(
        start_time=0.0,
        end_time=1.0,
        data={"eeg": signal},
        timestamps={"eeg": np.arange(8, dtype=float)},
    )


def test_disabled_mne_faster_returns_an_independent_copy() -> None:
    window = _window()

    result = _WindowArtifactPreprocessor(None)(window)

    np.testing.assert_array_equal(result, window.data["eeg"])
    assert result is not window.data["eeg"]


def test_enabled_mne_faster_uses_calibration_bundle() -> None:
    class FakeBundle:
        def transform(self, signal):
            return np.asarray(signal) * 2.0

    result = _WindowArtifactPreprocessor(FakeBundle())(_window())

    np.testing.assert_array_equal(result, _window().data["eeg"] * 2.0)


def test_enabled_mne_faster_requires_bundle_directory() -> None:
    with pytest.raises(ValueError, match="mne_faster_bundle_dir"):
        WorkerConfig.from_dict(
            {
                "source": {"type": "replay", "path": "unused.npy"},
                "preprocessing": {"mne_faster_enabled": True},
            }
        )
