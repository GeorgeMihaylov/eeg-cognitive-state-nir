import numpy as np

from apps.streaming_worker.sources.replay import ReplayEEGSource


def test_replay_emits_ordered_samples_and_completes():
    signal = np.arange(20, dtype=float).reshape(10, 2)
    source = ReplayEEGSource(signal, sample_rate=5, realtime=False)
    received = []

    source.start(received.append)

    assert source.wait(timeout=2)
    assert len(received) == 10
    assert [sample.timestamp for sample in received] == list(np.arange(10) / 5)
    np.testing.assert_array_equal(received[-1].values, signal[-1])
