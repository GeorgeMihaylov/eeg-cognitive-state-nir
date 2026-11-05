import itertools
import math

import numpy as np
import pandas as pd
import pytest

from bench.analysis.temporal_feature_diagnostics import contiguous_segments, permute_finite_columns
from scripts.analysis.audit_pm_temporal_permutation_entropy import permutation_entropy_columns


@pytest.mark.parametrize("m", [3, 4, 5])
def test_ordinal_codes_are_injective_and_entropy_limits(m):
    permutations = np.array(list(itertools.permutations(range(m))))
    codes = (permutations * m ** np.arange(m)).sum(axis=1)
    assert len(np.unique(codes)) == math.factorial(m)
    monotonic = np.arange(5000.)[:, None]
    random = np.random.default_rng(42).normal(size=(5000, 1))
    periodic = np.tile([0., 1., 2., 1.], 1250)[:, None]
    pe = lambda x: permutation_entropy_columns(x, m)[0][0]
    assert pe(monotonic) == 0
    assert pe(random) > .97
    assert pe(periodic) < pe(random) - .1
    assert pe(np.ones((5000, 1))) == 0


def test_pe_length_rule_and_missing_mask():
    assert np.isnan(permutation_entropy_columns(np.arange(100.)[:, None], 5)[0][0])
    x = np.arange(400., dtype=float).reshape(200, 2)
    x[10:20, 0] = np.nan
    a = permute_finite_columns(x, np.random.default_rng(42))
    b = permute_finite_columns(x, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.isnan(x), np.isnan(a))
    np.testing.assert_array_equal(np.sort(x, axis=0), np.sort(a, axis=0))


def test_segmentation_gaps_and_boundaries():
    frame = pd.DataFrame(dict(t_start=[30, 0, 10, 40], record_id="r", subject_id="s", outer_fold=1))
    assert [s.tolist() for s in contiguous_segments(frame)] == [[1, 2], [0, 3]]
    for key, value in [("record_id", "other"), ("subject_id", "other"), ("outer_fold", 2)]:
        bad = frame.copy()
        bad.loc[0, key] = value
        with pytest.raises(ValueError, match=key):
            contiguous_segments(bad)
