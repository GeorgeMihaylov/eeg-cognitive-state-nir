import numpy as np
from bench.analysis.temporal_state_space import ssa_energy, fnn_curve, robust_trajectory


def test_ssa_matches_direct_svd_and_low_rank_sine():
    x = np.sin(np.arange(240) * .13)
    energy, metrics = ssa_energy(x)
    scaled, _ = robust_trajectory(x)
    s = np.linalg.svd(np.lib.stride_tricks.sliding_window_view(scaled, 60).T, compute_uv=False)
    np.testing.assert_allclose(energy, s*s / (s*s).sum(), atol=1e-13)
    assert metrics['rank90'] <= 3
    _, noise = ssa_energy(np.random.default_rng(42).normal(size=240))
    assert noise['effective_rank'] > 20
    _, affine = ssa_energy(x * 3 + 10)
    assert abs(affine['effective_rank'] - metrics['effective_rank']) < 1e-9


def test_length_and_constant_guards():
    assert ssa_energy(np.arange(99.))[1]['status'] == 'insufficient_sequence_length'
    assert ssa_energy(np.ones(120))[1]['status'] == 'zero_iqr'
    assert all(r['status'] == 'insufficient_sequence_length' for r in fnn_curve(np.arange(500.)))
    assert all(r['status'] == 'zero_iqr' for r in fnn_curve(np.ones(1000)))


def test_fnn_periodic_vs_random_and_determinism():
    x = np.sin(np.arange(1200) * .137)
    observed = fnn_curve(x)
    random = fnn_curve(np.random.default_rng(42).normal(size=1200))
    assert observed[1]['fnn_fraction'] < .01
    assert random[1]['fnn_fraction'] > .1
    assert observed == fnn_curve(x)
    assert all(r['theiler_windows'] == 12 for r in observed)
