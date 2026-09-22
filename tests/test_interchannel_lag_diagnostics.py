from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.analysis import interchannel_lag_diagnostics as diagnostic
from bench.analysis.pm_frozen_raw_preflight import load_verified_raw_window
from cogstate.features.connectivity import channel_pairs, compute_correlation_matrix


def test_canonical_pair_order():
    pairs = channel_pairs(14)
    assert len(pairs) == len(set(pairs)) == 91
    assert pairs == tuple(sorted(pairs))
    assert pairs[0] == (0, 1) and pairs[-1] == (12, 13)


def test_identical_signals_and_zero_lag():
    x = np.random.default_rng(6).normal(size=512)
    result, _ = diagnostic.diagnose_window(np.stack([x, x]))
    assert result.peak_lag_samples.item() == 0
    assert result.peak_corr_abs.item() == pytest.approx(1)


@pytest.mark.parametrize("delay", [-7, 7])
def test_known_delay_sign_and_reverse_antisymmetry(delay):
    rng = np.random.default_rng(15)
    x = rng.normal(size=512)
    # Linear delay with independent boundary samples, never circular wrapping.
    y = np.concatenate([rng.normal(size=abs(delay)), x[:-abs(delay)]])
    window = np.stack([x, y]) if delay > 0 else np.stack([y, x])
    forward, _ = diagnostic.diagnose_window(window)
    reverse, _ = diagnostic.diagnose_window(window[::-1])
    assert forward.peak_lag_samples.item() == delay
    assert reverse.peak_lag_samples.item() == -delay
    assert forward.peak_corr_abs.item() == pytest.approx(1)


def test_overlap_only_no_circular_wrap_and_pearson_recentering():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(2, 50))
    x[-5:] = np.arange(5) * 100
    y[:5] = x[-5:]
    lags, values = diagnostic.overlap_correlations(np.stack([x, y]), 5)
    for lag, value in zip(lags, values[:, 0]):
        a, b = (x[:-lag], y[lag:]) if lag > 0 else ((x[-lag:], y[:lag]) if lag < 0 else (x, y))
        assert value == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)
    circular = np.corrcoef(x, np.concatenate([y[5:], y[:5]]))[0, 1]
    assert abs(values[lags == 5, 0].item() - circular) > .1


@pytest.mark.parametrize("constant", [np.ones(256), np.zeros(256), 1+np.arange(256)*1e-16])
def test_constant_is_invalid_not_zero(constant):
    result, qc = diagnostic.diagnose_window(np.stack([constant, np.arange(256)]))
    assert not result.valid.item()
    assert result[list(diagnostic.METRICS)].isna().all().all()
    assert qc["constant_channels"] == [0]


@pytest.mark.parametrize("ms,rate,expected", [(100, 256, 26), (50, 256, 13), (250, 256, 64), (100, 500, 50), (0, 256, 0)])
def test_ms_to_samples(ms, rate, expected):
    assert diagnostic.lag_samples(ms, rate) == expected


def test_sampling_balanced_uniform_deterministic():
    frame = pd.DataFrame([
        {"sample_id": i + 100*r + 1000*s, "subject_id": str(s), "record_group_id": str(r),
         "record_id": f"{s}-{r}", "t_start": float(i*10)}
        for s in range(2) for r in range(3) for i in range(30)
    ])
    selected = diagnostic.select_windows(frame, 10)
    pd.testing.assert_frame_equal(selected, diagnostic.select_windows(frame.sample(frac=1, random_state=9), 10))
    assert selected.groupby("subject_id").size().eq(10).all()
    assert sorted(selected.loc[selected.subject_id.eq("0")].groupby("record_id").size()) == [3, 3, 4]
    assert selected.t_start.max() == 290
    assert diagnostic.select_windows(frame, None).sample_id.nunique() == len(frame)
    with pytest.raises(ValueError):
        diagnostic.select_windows(frame, 0)


def test_aggregation_exact_median_iqr_fractions_and_invalid_denominator():
    frame = pd.DataFrame({
        "pair_index": 0, "channel_a": "A", "channel_b": "B", "valid": [True]*5+[False],
        "r_zero": [0, .1, .2, .3, .4, np.nan], "peak_corr_signed": [.5]*5+[np.nan],
        "peak_corr_abs": [.5]*5+[np.nan], "peak_lag_samples": [-4, -1, 0, 2, 4, np.nan],
        "peak_lag_ms": [-16, -4, 0, 8, 16, np.nan], "delta_abs_corr": [.5, .4, .3, .2, .1, np.nan],
    })
    row = diagnostic.summarize_pairs(frame, []).iloc[0]
    assert row.n_valid_windows == 5 and row.invalid_fraction == pytest.approx(1/6)
    assert row.median_peak_lag_ms == 0 and row.iqr_peak_lag_ms == 12
    assert row.median_delta_abs_corr == .3
    assert row.fraction_positive_lag == .4
    assert row.fraction_negative_lag == .2
    assert row.fraction_near_zero_lag == .4
    assert row.sign_consistency == .4
    frame.valid = False
    assert np.isnan(diagnostic.summarize_pairs(frame, []).iloc[0].fraction_near_zero_lag)


def test_existing_zero_lag_matches_valid_channels():
    window = np.random.default_rng(44).normal(size=(14, 512))
    result, qc = diagnostic.diagnose_window(window)
    matrix = compute_correlation_matrix(window.T)
    expected = [matrix[a, b] for a, b in channel_pairs(14)]
    np.testing.assert_allclose(result.r_zero, expected, atol=1e-12)
    assert qc["zero_lag_compared_pairs"] == 91


def test_channel_order_mismatch_uses_preflight(tmp_path):
    from bench.datasets.channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
    row = pd.Series({"sample_id": 1, "subject_id": "s", "record_id": "r", "record_group_id": "g",
                     "outer_fold": 1, "t_start": 0., "t_end": 10., "status": "ok", "n_channels": 14,
                     "n_samples_expected": 2560, "sfreq_target": 256., "preprocessing_hash": "h", "cache_file": "a.npy"})
    (tmp_path / "a.json").write_text(json.dumps({"channels": list(reversed(PROJECT_EMOTIV_CHANNEL_ORDER))}))
    with pytest.raises(ValueError, match="channel order"):
        load_verified_raw_window(row, row, identity={"raw_preprocessing_hash": "h"}, cache_path_root=tmp_path)


def test_cli_modes_conflict_and_no_overwrite(tmp_path):
    with pytest.raises(SystemExit) as error:
        diagnostic.main(["--feature-cache-dir", "unused", "--output-dir", str(tmp_path),
                         "--windows-per-subject", "10", "--all-windows"])
    assert error.value.code == 2
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostics(root=tmp_path, feature_cache_dir=tmp_path,
                                   raw_manifest_path=tmp_path, cache_path_root=tmp_path, output_dir=tmp_path)


def test_label_free_runner_outputs(monkeypatch, tmp_path):
    # Drive the real loop/aggregators, while keeping contract IO synthetic.
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/protocol.json").read_text())
    _, pipeline = diagnostic.load_feature_profile(root / "experiments/features/preliminary_model_zoo_features_v1.json")
    index = pd.DataFrame([{"sample_id": i, "subject_id": "s", "record_id": "r", "record_group_id": "g",
                           "outer_fold": 1, "t_start": i*10., "t_end": (i+1)*10.} for i in range(3)])
    monkeypatch.setattr(diagnostic, "load_feature_cache", lambda _: (np.zeros((3,371)), index, pipeline.feature_names(), {}))
    reads = []
    def read_metadata(path, columns):
        reads.append(str(path))
        assert not any(c.startswith(("target_", "label_")) for c in columns)
        return index.copy()
    monkeypatch.setattr(diagnostic.pd, "read_parquet", read_metadata)
    monkeypatch.setattr(diagnostic, "validate_frozen_raw_inputs", lambda *a, **k: (protocol["feature_cache_identity"], protocol["fixed_fold_hash"]))
    signal = np.random.default_rng(8).normal(size=(14,2560)).astype(np.float32)
    monkeypatch.setattr(diagnostic, "load_verified_raw_window", lambda *a, **k: signal)
    output = tmp_path / "result"
    summary = diagnostic.run_diagnostics(root=root, feature_cache_dir=tmp_path, raw_manifest_path=tmp_path / "raw.parquet",
                                        cache_path_root=tmp_path, output_dir=output, windows_per_subject=2)
    assert len(reads) == 1 and not summary["targets_loaded"]
    assert summary["n_windows"] == 2 and summary["n_pairs"] == 91
    assert len(list(output.iterdir())) == 6
    assert not (output / "window_pair_metrics.csv").exists()
    assert len(pd.read_csv(output / "global_pair_summary.csv")) == 91
    assert json.loads((output / "run_summary.json").read_text())["selected_sample_ids"] == [0, 2]
