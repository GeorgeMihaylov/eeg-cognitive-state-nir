"""Read-only, label-free interchannel lag diagnostics on frozen raw EEG windows.

R_xy(k) = corr(x[t], y[t+k]) over real overlap only. Positive k means x
leads y; negative k means y leads x. No circular wrapping or signal shifts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bench.analysis.pm_frozen_raw_preflight import (
    load_verified_raw_window, validate_frozen_raw_inputs,
)
from bench.datasets.channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
from bench.experiments.pm_eeg_lag_confirmatory import stable_hash
from bench.features.cogstate_feature_cache import load_feature_cache, load_feature_profile
from cogstate.features.connectivity import channel_pairs, compute_correlation_matrix


SIGN_CONVENTION = "R_xy(k)=corr(x[t], y[t+k]); k>0: x leads y; k<0: y leads x; overlap only"
RELATIVE_CONSTANT_TOL = 1e-12
TIE_ATOL = 1e-12
METRICS = ("r_zero", "peak_corr_signed", "peak_corr_abs", "peak_lag_samples", "peak_lag_ms", "delta_abs_corr")


def lag_samples(max_lag_ms: float, sampling_rate: float) -> int:
    """Nearest sample, with half samples rounded upward (requested vs effective saved)."""
    if not np.isfinite([max_lag_ms, sampling_rate]).all() or max_lag_ms < 0 or sampling_rate <= 0:
        raise ValueError("Lag must be nonnegative and sampling rate positive and finite")
    return int(np.floor(max_lag_ms * sampling_rate / 1000.0 + 0.5))


def _normalized_overlap(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = signal - signal.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    scale = np.maximum(np.linalg.norm(signal, axis=1), np.finfo(float).tiny)
    valid = norm > RELATIVE_CONSTANT_TOL * scale
    normalized = np.divide(centered, norm[:, None], out=np.zeros_like(centered), where=valid[:, None])
    return normalized, valid


def overlap_correlations(window: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return [lag,pair] Pearson values for R_xy(k)=corr(x[t],y[t+k]).

    For k>0 use x[:T-k], y[k:]; for k<0 use x[-k:], y[:T+k].
    Recenter and normalize EACH overlap. Constant overlaps are NaN, never zero.
    Lag order prefers zero, then smallest absolute lag, then negative in ties.
    """
    signal = np.asarray(window, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] < 2 or not np.isfinite(signal).all():
        raise ValueError("Expected finite [channels,time] signal")
    if not isinstance(max_lag, (int, np.integer)) or not 0 <= max_lag <= signal.shape[1] - 3:
        raise ValueError("Lag must leave at least three overlapping samples")
    pairs = np.asarray(channel_pairs(signal.shape[0]))
    lags = np.array(sorted(range(-max_lag, max_lag + 1), key=lambda k: (abs(k), k)))
    values = []
    for k in lags:
        first = signal[:, :signal.shape[1] - k] if k >= 0 else signal[:, -k:]
        second = signal[:, k:] if k >= 0 else signal[:, :signal.shape[1] + k]
        a, valid_a = _normalized_overlap(first)
        b, valid_b = _normalized_overlap(second)
        corr = np.clip((a @ b.T)[pairs[:, 0], pairs[:, 1]], -1.0, 1.0)
        corr[~(valid_a[pairs[:, 0]] & valid_b[pairs[:, 1]])] = np.nan
        values.append(corr)
    return lags, np.stack(values)


def diagnose_window(window: np.ndarray, *, sampling_rate: float = 256.0,
                    max_lag_ms: float = 100.0) -> tuple[pd.DataFrame, dict]:
    lags, curves = overlap_correlations(window, lag_samples(max_lag_ms, sampling_rate))
    pairs = np.asarray(channel_pairs(window.shape[0]))
    # Confirm the common Pearson definition on every finite, nonconstant pair.
    legacy = compute_correlation_matrix(np.asarray(window, dtype=np.float64).T)[pairs[:, 0], pairs[:, 1]]
    zero_valid = np.isfinite(curves[0])
    error = float(np.max(np.abs(curves[0, zero_valid] - legacy[zero_valid]))) if zero_valid.any() else 0.0
    if error > 1e-10:
        raise ValueError(f"Existing zero-lag correlation mismatch: {error}")
    # Conservative QC: the pair must be mathematically defined at every tested lag.
    valid = np.isfinite(curves).all(axis=0)
    absolute = np.where(np.isfinite(curves), np.abs(curves), -np.inf)
    maximum = absolute.max(axis=0)
    tie = np.isfinite(curves) & (absolute >= maximum[None, :] - TIE_ATOL)
    peak_indices = tie.argmax(axis=0)
    columns = np.arange(len(pairs))
    signed = curves[peak_indices, columns]
    peaks = lags[peak_indices].astype(float)
    frame = pd.DataFrame({
        "pair_index": columns, "valid": valid, "r_zero": curves[0],
        "peak_corr_signed": signed, "peak_corr_abs": np.abs(signed),
        "peak_lag_samples": peaks, "peak_lag_ms": peaks * 1000.0 / sampling_rate,
        "delta_abs_corr": np.maximum(0.0, np.abs(signed) - np.abs(curves[0])),
    })
    frame.loc[~valid, list(METRICS)] = np.nan
    _, channel_valid = _normalized_overlap(np.asarray(window, dtype=np.float64))
    return frame, {
        "invalid_pairs": int((~valid).sum()), "constant_channels": np.flatnonzero(~channel_valid).tolist(),
        "invalid_overlap_pairs": int((~zero_valid).sum()), "zero_lag_max_abs_error": error,
        "zero_lag_compared_pairs": int(zero_valid.sum()),
    }


def select_windows(index: pd.DataFrame, windows_per_subject: int | None = 10) -> pd.DataFrame:
    """Round-robin logical records, then evenly spaced times within each record."""
    if windows_per_subject is not None and windows_per_subject <= 0:
        raise ValueError("windows_per_subject must be positive")
    if index.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id")
    if index.groupby(["subject_id", "record_group_id"]).record_id.nunique().gt(1).any():
        raise ValueError("Ambiguous logical record: multiple physical records in frozen cohort")
    selected = []
    for _, subject in index.groupby("subject_id", sort=True):
        records = [g.sort_values(["t_start", "sample_id"], kind="stable")
                   for _, g in subject.groupby("record_group_id", sort=True)]
        budget = len(subject) if windows_per_subject is None else min(windows_per_subject, len(subject))
        counts = [0] * len(records)
        while sum(counts) < budget:
            for i, record in enumerate(records):
                if counts[i] < len(record) and sum(counts) < budget:
                    counts[i] += 1
        for record, count in zip(records, counts):
            if count:
                positions = np.rint(np.linspace(0, len(record) - 1, count)).astype(int) if count > 1 else [len(record) // 2]
                selected.append(record.iloc[positions])
    return pd.concat(selected).sort_values(["subject_id", "record_group_id", "t_start", "sample_id"], kind="stable").reset_index(drop=True)


def summarize_pairs(windows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Exact window-level medians/IQR within each group, not medians of medians.

    Positive/negative fractions use >1 / <-1 samples, disjoint from near-zero.
    All fractions use valid windows; no valid observations -> NaN, not zero.
    Global/fold summaries are window-pooled descriptive distributions; subject
    and record tables retain the hierarchy and unequal-window counts explicitly.
    """
    rows = []
    keys = group_columns + ["pair_index", "channel_a", "channel_b"]
    for key, group in windows.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        valid = group.loc[group.valid]
        row.update(n_windows=len(group), n_valid_windows=len(valid), n_invalid_windows=len(group)-len(valid),
                   invalid_fraction=float(1.0 - len(valid)/len(group)))
        for metric in METRICS:
            row[f"median_{metric}"] = float(valid[metric].median()) if len(valid) else np.nan
        row["median_abs_r_zero"] = float(valid.r_zero.abs().median()) if len(valid) else np.nan
        lag = valid.peak_lag_samples
        row["iqr_peak_lag_ms"] = float(valid.peak_lag_ms.quantile(.75) - valid.peak_lag_ms.quantile(.25)) if len(valid) else np.nan
        row["fraction_positive_lag"] = float((lag > 1).mean()) if len(valid) else np.nan
        row["fraction_negative_lag"] = float((lag < -1).mean()) if len(valid) else np.nan
        row["fraction_near_zero_lag"] = float((lag.abs() <= 1).mean()) if len(valid) else np.nan
        row["sign_consistency"] = max(row["fraction_positive_lag"], row["fraction_negative_lag"])
        rows.append(row)
    return pd.DataFrame(rows)


def run_diagnostics(*, root: Path, feature_cache_dir: Path, raw_manifest_path: Path,
                    cache_path_root: Path, output_dir: Path, windows_per_subject: int | None = 10,
                    max_lag_ms: float = 100.0, save_window_metrics: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    started = time.perf_counter()
    # Read ONLY identities/folds from the saved contract, never targets or labels.
    protocol = json.loads((root / "reports/diagnostics/pm_low_high_q3_extremes_confirmatory_v1/protocol.json").read_text(encoding="utf-8"))
    matrix, index, names, manifest = load_feature_cache(feature_cache_dir)
    _, pipeline = load_feature_profile(root / "experiments/features/preliminary_model_zoo_features_v1.json")
    if names != pipeline.feature_names() or pipeline.feature_hash() != protocol["feature_cache_identity"]["feature_hash"]:
        raise ValueError("Feature profile/names contract mismatch")
    raw_columns = ["sample_id", "subject_id", "record_id", "record_group_id", "outer_fold", "t_start", "t_end",
                   "status", "cache_file", "cache_offset", "n_channels", "n_samples_expected", "sfreq_target", "preprocessing_hash"]
    raw = pd.read_parquet(raw_manifest_path, columns=raw_columns)
    identity, fold_hash = validate_frozen_raw_inputs(
        matrix, index, names, manifest, raw, expected_identity=protocol["feature_cache_identity"],
        expected_fold_hash=protocol["fixed_fold_hash"], folds=protocol["fold_ids"],
    )
    selected = select_windows(index, windows_per_subject)
    sampling_rate = float(pipeline.config.sample_rate)
    max_samples = lag_samples(max_lag_ms, sampling_rate)
    if max_samples > 2560 - 3:
        raise ValueError("Requested lag leaves insufficient overlap")
    channels = [c.removeprefix("EEG.") for c in PROJECT_EMOTIV_CHANNEL_ORDER]
    pairs = channel_pairs(len(channels))
    if len(pairs) != 91:
        raise ValueError("Canonical 14-channel contract requires 91 pairs")
    raw_lookup = raw.set_index("sample_id", drop=False)
    frames, timings, qc_rows = [], [], []
    analysis_started = time.perf_counter()
    with threadpool_limits(limits=1):
        for _, feature in selected.iterrows():
            tick = time.perf_counter()
            if feature.sample_id not in raw_lookup.index:
                raise ValueError(f"Raw sample {feature.sample_id} missing")
            window = load_verified_raw_window(feature, raw_lookup.loc[feature.sample_id],
                                              identity=identity, cache_path_root=cache_path_root)
            frame, qc = diagnose_window(window, sampling_rate=sampling_rate, max_lag_ms=max_lag_ms)
            for column in ("sample_id", "subject_id", "record_id", "record_group_id", "outer_fold"):
                frame[column] = feature[column]
            frame["channel_a"] = [channels[a] for a, _ in pairs]
            frame["channel_b"] = [channels[b] for _, b in pairs]
            frames.append(frame)
            qc_rows.append(dict(qc, sample_id=int(feature.sample_id)))
            timings.append(time.perf_counter()-tick)
    analysis_seconds = time.perf_counter() - analysis_started
    windows = pd.concat(frames, ignore_index=True)
    tables = {
        "global_pair_summary.csv": summarize_pairs(windows, []),
        "subject_pair_summary.csv": summarize_pairs(windows, ["subject_id"]),
        "record_pair_summary.csv": summarize_pairs(windows, ["subject_id", "record_group_id", "record_id"]),
        "fold_pair_summary.csv": summarize_pairs(windows, ["outer_fold"]),
    }
    valid = windows.loc[windows.valid].copy()
    valid["abs_r_zero"] = valid.r_zero.abs()
    distributions = {name: {str(q): float(values) for q, values in valid[name].quantile([0, .25, .5, .75, 1]).items()}
                     for name in ("abs_r_zero", "peak_corr_abs", "delta_abs_corr", "peak_lag_ms")}
    summary = {
        "status": "complete", "result_status": "diagnostic", "schema_version": "interchannel-lag-diagnostics-v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "input_identity": identity, "fixed_fold_hash": fold_hash, "channel_order": channels,
        "pair_order": [[channels[a], channels[b]] for a, b in pairs], "n_pairs": len(pairs),
        "sampling_rate": sampling_rate, "raw_shape": [14, 2560], "raw_dtype": "float32",
        "max_lag_ms_requested": max_lag_ms, "max_lag_samples": max_samples,
        "max_lag_ms_effective": max_samples * 1000/sampling_rate, "sign_convention": SIGN_CONVENTION,
        "rounding": "nearest sample; half upward", "near_zero": "abs(lag_samples)<=1",
        "sign_fractions": "positive: lag>1; negative: lag<-1; near-zero: abs(lag)<=1; denominator=valid windows",
        "sign_consistency": "max(fraction_positive_lag,fraction_negative_lag)",
        "tie_policy": "within absolute correlation 1e-12: smallest absolute lag, then negative lag",
        "constant_policy": "centered L2 norm <= 1e-12 * raw L2 norm; any undefined tested overlap invalidates pair",
        "zero_lag_comparison": "same Pearson on valid channels; existing connectivity substitutes 0 for constants, diagnostic uses NaN",
        "aggregation": "exact window-level medians/IQR at record, subject, fold and global levels; global/fold are window-pooled, not participant-macro",
        "sampling": {"mode": "all_windows" if windows_per_subject is None else "balanced_pilot",
                     "windows_per_subject": windows_per_subject, "method": "round-robin record quotas; uniform time positions per record; singleton midpoint"},
        "selected_sample_ids": [int(s) for s in selected.sample_id],
        "selected_sample_hash": stable_hash([int(s) for s in selected.sample_id]),
        "n_windows": len(selected), "n_subjects": int(selected.subject_id.nunique()),
        "n_records": int(selected.record_id.nunique()), "n_record_groups": int(selected.record_group_id.nunique()),
        "n_folds": int(selected.outer_fold.nunique()), "n_window_pairs": len(windows),
        "invalid_window_pairs": int((~windows.valid).sum()),
        "windows_with_invalid_pairs": sum(q["invalid_pairs"] > 0 for q in qc_rows),
        "constant_channel_windows": sum(len(q["constant_channels"]) for q in qc_rows),
        "constant_channel_counts": {c: sum(i in q["constant_channels"] for q in qc_rows) for i, c in enumerate(channels)},
        "zero_lag_max_abs_error": max(q["zero_lag_max_abs_error"] for q in qc_rows),
        "zero_lag_compared_window_pairs": sum(q["zero_lag_compared_pairs"] for q in qc_rows),
        "channel_order_validation": "all selected raw shard sidecars verified by preflight helper",
        "distributions": distributions,
        "fraction_near_zero": float((valid.peak_lag_samples.abs() <= 1).mean()) if len(valid) else None,
        "fraction_search_boundary": float((valid.peak_lag_samples.abs() == max_samples).mean()) if len(valid) else None,
        "analysis_seconds": analysis_seconds, "first_three_windows_seconds": sum(timings[:3]),
        "median_window_seconds": float(np.median(timings)),
        "estimated_full_analysis_seconds": analysis_seconds / len(selected) * len(index),
        "elapsed_before_write_seconds": time.perf_counter()-started,
        "training_executed": False, "targets_loaded": False, "signals_modified": False,
    }
    # All-invalid distributions are explicitly null in strict JSON.
    summary["distributions"] = {k: {q: v if np.isfinite(v) else None for q, v in d.items()} for k, d in distributions.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    selected[["sample_id", "subject_id", "record_id", "record_group_id", "outer_fold", "t_start", "t_end"]].to_csv(output_dir / "selected_windows.csv", index=False)
    if save_window_metrics:
        windows.to_csv(output_dir / "window_pair_metrics.csv", index=False)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, default=Path("data/interim/raw_eeg_window_index_w10_pm_union_composite_v1.parquet"))
    parser.add_argument("--cache-path-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--windows-per-subject", type=int)
    sampling.add_argument("--all-windows", action="store_true")
    parser.add_argument("--max-lag-ms", type=float, default=100.0)
    parser.add_argument("--save-window-metrics", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    def resolve(p):
        return p if p.is_absolute() else root / p
    try:
        summary = run_diagnostics(
            root=root, feature_cache_dir=resolve(args.feature_cache_dir),
            raw_manifest_path=resolve(args.raw_manifest), cache_path_root=resolve(args.cache_path_root or root),
            output_dir=resolve(args.output_dir), max_lag_ms=args.max_lag_ms,
            windows_per_subject=None if args.all_windows else (10 if args.windows_per_subject is None else args.windows_per_subject),
            save_window_metrics=args.save_window_metrics,
        )
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"STATUS: FAIL\n{exc}")
        return 1
    print(json.dumps({k: summary[k] for k in ("status", "n_windows", "n_subjects", "n_records", "n_folds", "n_pairs", "invalid_window_pairs", "analysis_seconds")}, indent=2))
    return 0
