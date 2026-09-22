"""Read-only label-free quality audit of the frozen 14-channel raw representation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from threadpoolctl import threadpool_limits

from bench.analysis.interchannel_lag_diagnostics import select_windows
from bench.analysis.pm_frozen_raw_preflight import validate_frozen_raw_inputs, load_verified_raw_window
from bench.datasets.channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
from bench.datasets.raw_eeg_window_dataset import RawEEGWindowArrayView
from bench.datasets.raw_preprocessing import raw_window_artifact_metrics
from bench.experiments.pm_eeg_lag_confirmatory import stable_hash
from bench.features.cogstate_feature_cache import load_feature_cache, sample_id_universe_hash
from cogstate.features.spectral import SpectralConfig, compute_power_spectrum, _band_powers_from_psd

CHANNELS = tuple(c.removeprefix("EEG.") for c in PROJECT_EMOTIV_CHANNEL_ORDER)
SPEC = "experiments/pm_diagnostics/channel_quality_v1.json"
META = ["sample_id", "subject_id", "record_id", "record_group_id", "outer_fold"]


def robust_peer_scores(values: np.ndarray) -> np.ndarray:
    """Leave-one-out unscaled MAD score; no artificial epsilon for zero MAD."""
    values = np.asarray(values, dtype=float)
    if values.shape != (14,):
        raise ValueError("Expected 14 channel values")
    result = np.full(14, np.nan)
    for i in range(14):
        peers = np.delete(values, i)
        peers = peers[np.isfinite(peers)]
        if len(peers) < 7 or not np.isfinite(values[i]):
            continue
        center = np.median(peers)
        mad = np.median(np.abs(peers - center))
        if mad > 0:
            result[i] = (values[i] - center) / mad
    return result


def window_quality(window: np.ndarray, config: dict, artifact: dict) -> pd.DataFrame:
    x = np.asarray(window, dtype=np.float64)
    if x.shape != tuple(config["shape"]):
        raise ValueError(f"Raw shape mismatch: {x.shape}")
    valid = np.isfinite(x).all(axis=1)
    metrics = {"nonfinite_fraction": 1 - np.isfinite(x).mean(axis=1)}
    # Invalid channels never become plausible QC values through imputation.
    names = ["mean", "std", "mad", "rms", "peak_to_peak", "absolute_max", "exact_repeat_fraction",
             "flat_fraction", "diff_median_abs", "diff_rms", "roughness_ratio", "total_spectral_power",
             "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "high_frequency_ratio"]
    metrics.update({name: np.full(14, np.nan) for name in names})
    if valid.any():
        signal = x[valid]
        delta = np.diff(signal, axis=1)
        qc = raw_window_artifact_metrics(signal)
        values = {"mean": signal.mean(axis=1), "std": signal.std(axis=1),
                  "mad": np.median(np.abs(signal - np.median(signal, axis=1)[:, None]), axis=1),
                  "rms": np.sqrt(np.mean(signal ** 2, axis=1)),
                  "peak_to_peak": np.asarray(qc["channel_peak_to_peak"]),
                  "absolute_max": np.max(np.abs(signal), axis=1),
                  "exact_repeat_fraction": np.mean(delta == 0, axis=1),
                  "flat_fraction": np.asarray(qc["channel_flat_fraction"]),
                  "diff_median_abs": np.median(np.abs(delta), axis=1),
                  "diff_rms": np.sqrt(np.mean(delta ** 2, axis=1))}
        values["roughness_ratio"] = np.divide(values["diff_rms"], values["rms"],
            out=np.full(len(signal), np.nan), where=values["rms"] > 0)
        spectral = SpectralConfig(sample_rate=config["sampling_rate"])
        spectrum = compute_power_spectrum(signal.T, spectral)
        bands = _band_powers_from_psd(spectrum.frequencies, spectrum.psd, spectral)
        total = np.sum(list(bands.values()), axis=0)
        values["total_spectral_power"] = trapezoid(spectrum.psd, spectrum.frequencies, axis=0)
        for name in ("delta", "theta", "alpha", "beta", "gamma"):
            key = "high_frequency_ratio" if name == "gamma" else f"rel_{name}"
            values[key] = np.divide(bands[name], total, out=np.full(len(signal), np.nan), where=total > 0)
        for name, value in values.items():
            metrics[name][valid] = value
    for field, source in (("amplitude_violation", "absolute_max"), ("flat_violation", "flat_fraction")):
        limit = artifact.get("max_abs_amplitude" if field == "amplitude_violation" else "max_flat_fraction")
        metrics[field] = np.where(valid, (metrics[source] > limit).astype(float), np.nan) if limit is not None else np.full(14, np.nan)
    for name in config["relative_metrics"]:
        score = robust_peer_scores(metrics[name])
        metrics[f"z_{name}"] = score
        metrics[f"z_{name}_undefined"] = (~np.isfinite(score)).astype(float)
        for suffix, mask in (("high", score > config["descriptive_z_cutoff"]),
                             ("low", score < -config["descriptive_z_cutoff"]),
                             ("any", np.abs(score) > config["descriptive_z_cutoff"])):
            metrics[f"anomaly_{name}_{suffix}"] = np.where(np.isfinite(score), mask.astype(float), np.nan)
    return pd.DataFrame({"channel": CHANNELS, **metrics})


def summarize(windows: pd.DataFrame):
    """Keep record-balanced participant summaries separate from pooled windows."""
    metrics = [c for c in windows if c not in [*META, "channel"]]
    rates = [c for c in metrics if c.startswith("anomaly_") or c.endswith(("violation", "undefined")) or c == "nonfinite_fraction"]
    reducers = {c: "mean" if c in rates else "median" for c in metrics}
    keys = ["subject_id", "record_id", "outer_fold", "channel"]
    record = windows.groupby(keys, sort=True).agg(reducers)
    record["n_windows"] = windows.groupby(keys).size()
    record = record.reset_index()
    subject = record.groupby(["subject_id", "outer_fold", "channel"], sort=True)[metrics].mean()
    subject["n_records"] = record.groupby(["subject_id", "outer_fold", "channel"]).size()
    subject["n_windows"] = record.groupby(["subject_id", "outer_fold", "channel"]).n_windows.sum()
    subject = subject.reset_index()
    anomalies = [c for c in rates if c.startswith("anomaly_")]
    for c in anomalies:
        reference = subject.groupby("subject_id")[c].transform("median")
        subject[f"{c}_above_subject_channel_median"] = np.where(subject[c].notna() & reference.notna(), (subject[c] > reference).astype(float), np.nan)
    def global_row(channel, group):
        w = windows.loc[windows.channel.eq(channel)]
        row = {"channel": channel, "n_windows": len(w), "n_subjects": len(group)}
        for c in metrics:
            row[f"window_median_{c}"] = w[c].median()
            row[f"window_iqr_{c}"] = w[c].quantile(.75) - w[c].quantile(.25)
            row[f"participant_median_{c}"] = group[c].median()
            row[f"participant_iqr_{c}"] = group[c].quantile(.75) - group[c].quantile(.25)
            row[f"valid_windows_{c}"] = int(w[c].notna().sum())
            row[f"valid_subjects_{c}"] = int(group[c].notna().sum())
            if c in rates:
                row[f"window_rate_{c}"] = w[c].mean()
            if c in anomalies:
                row[f"subjects_elevated_{c}"] = group[f"{c}_above_subject_channel_median"].mean()
        return row
    global_frame = pd.DataFrame([global_row(c, subject.loc[subject.channel.eq(c)]) for c in CHANNELS])
    fold = subject.groupby(["outer_fold", "channel"])[metrics].agg(["median", lambda s: s.quantile(.75)-s.quantile(.25)])
    fold.columns = [f"participant_{'iqr' if op == '<lambda_0>' else op}_{c}" for c, op in fold.columns]
    fold = fold.reset_index()
    fold["n_subjects"] = subject.groupby(["outer_fold", "channel"]).size().to_numpy()
    return record, subject, global_frame, fold


def align_frozen(index, raw):
    if raw.sample_id.duplicated().any() or index.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs")
    aligned = raw.set_index("sample_id", drop=False).loc[index.sample_id].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(index[[*META, "t_start", "t_end"]].reset_index(drop=True),
                                      aligned[[*META, "t_start", "t_end"]], check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("Frozen sample/subject/fold metadata mismatch") from exc
    return aligned


def run(root: Path, feature_cache: Path, output: Path, runtime: Path, mode: str):
    started = time.perf_counter()
    config = json.loads((root / SPEC).read_text())
    config_hash = stable_hash(config)
    if output.exists() or runtime.exists():
        raise FileExistsError("Use a new output/runtime directory; existing audit is never overwritten")
    if mode == "full":
        smoke = json.loads((output.parent / "channel_quality_smoke_v1/run_summary.json").read_text())
        if smoke["status"] != "PASS" or smoke["config_hash"] != config_hash:
            raise ValueError("Matching smoke PASS required")
    protocol = json.loads((root / config["identity_reference"]).read_text())
    matrix, index, names, manifest = load_feature_cache(feature_cache)
    columns = [*META, "t_start", "t_end", "status", "cache_file", "cache_offset", "n_channels", "n_samples_expected", "sfreq_target", "preprocessing_hash"]
    raw = pd.read_parquet(root / config["raw_manifest"], columns=columns)
    identity, fold_hash = validate_frozen_raw_inputs(matrix, index, names, manifest, raw,
        expected_identity=protocol["feature_cache_identity"], expected_fold_hash=protocol["fixed_fold_hash"], folds=protocol["fold_ids"])
    counts = {"windows": len(index), "subjects": index.subject_id.nunique(), "records": index.record_id.nunique()}
    if counts != config["expected_counts"]:
        raise ValueError("Frozen counts mismatch")
    aligned = align_frozen(index, raw)
    artifacts, input_stats = {}, []
    for cache_file, shard in aligned.groupby("cache_file", sort=True):
        row = shard.iloc[0]
        feature = index.loc[index.sample_id.eq(row.sample_id)].iloc[0]
        load_verified_raw_window(feature, row, identity=identity, cache_path_root=root)
        path = Path(cache_file) if Path(cache_file).is_absolute() else root / cache_file
        sidecar = json.loads(path.with_suffix(".json").read_text())
        entries = {r["sample_id"]: r for r in sidecar["window_results"]}
        for item in shard.itertuples():
            if entries[item.sample_id]["cache_offset"] != item.cache_offset or entries[item.sample_id]["status"] != "ok":
                raise ValueError("Shard sample provenance mismatch")
        artifacts[cache_file] = sidecar["raw_preprocessing"]["artifact_rejection"]
        input_stats.extend((p, p.stat().st_size, p.stat().st_mtime_ns) for p in (path, path.with_suffix(".json")))
    selected = select_windows(index, config["smoke_windows_per_subject"]) if mode == "smoke" else index
    selected_raw = align_frozen(selected, raw)
    view = RawEEGWindowArrayView(selected_raw, cache_path_root=root)
    output.mkdir(parents=True); runtime.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(config, indent=2)+"\n")
    frames = []
    for i, row in selected_raw.iterrows():
        frame = window_quality(view[i][0], config, artifacts[row.cache_file])
        for col in META:
            frame[col] = row[col]
        frames.append(frame)
        if (i+1) % 2000 == 0:
            print(f"QC {i+1}/{len(selected_raw)}", flush=True)
    windows = pd.concat(frames, ignore_index=True)
    if windows.groupby("sample_id").size().ne(14).any() or windows.sample_id.drop_duplicates().tolist() != selected.sample_id.tolist():
        raise ValueError("Window/channel universe mismatch")
    windows.to_parquet(runtime / "window_channel_quality.parquet", index=False)
    record, subject, global_frame, fold = summarize(windows)
    for name, frame in (("channel_record_quality", record), ("channel_subject_quality", subject),
                        ("channel_global_quality_summary", global_frame), ("channel_fold_quality", fold)):
        frame.to_csv(output / f"{name}.csv", index=False)
    redundancy_path = root / config["redundancy_reference"]
    redundancy = pd.read_csv(redundancy_path, decimal=",")
    if set(redundancy.channel) != set(CHANNELS) or redundancy.channel.duplicated().any():
        raise ValueError("Redundancy channel contract mismatch")
    if not pd.api.types.is_numeric_dtype(redundancy.participant_median_mean_abs_corr):
        raise ValueError("Redundancy decimal parsing failed")
    global_frame.merge(redundancy, on="channel", validate="one_to_one", suffixes=("", "_redundancy")).to_csv(output / "channel_quality_vs_redundancy.csv", index=False)
    if any(p.stat().st_size != size or p.stat().st_mtime_ns != mtime for p, size, mtime in input_stats):
        raise ValueError("Raw inputs changed during audit")
    summary = dict(status="PASS", result_status="diagnostic", mode=mode, config=config, config_hash=config_hash,
        frozen_cache_identity=identity, sample_universe_hash=sample_id_universe_hash(index.sample_id),
        selected_sample_hash=sample_id_universe_hash(selected.sample_id), fold_hash=fold_hash,
        channel_order=CHANNELS, sampling_rate=config["sampling_rate"], frozen_counts=counts,
        n_windows=len(selected), n_window_channels=len(windows), n_subject_rows=len(subject), n_record_rows=len(record),
        raw_preflight="PASS via existing validate_frozen_raw_inputs and verified shard loader; no target table read",
        artifact_contracts=sorted({json.dumps(v, sort_keys=True) for v in artifacts.values()}),
        training_executed=False, targets_loaded=False, signals_modified=False, input_stats_unchanged=True,
        redundancy_scope="existing 533-window balanced pilot, 54 participants; quality full cohort or smoke as labelled; descriptive join only",
        redundancy_sha256=hashlib.sha256(redundancy_path.read_bytes()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        python_version=sys.version, numpy_version=np.__version__, pandas_version=pd.__version__,
        code_hashes={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in (
            "bench/analysis/channel_quality.py", "bench/analysis/pm_frozen_raw_preflight.py",
            "bench/datasets/raw_preprocessing.py", "cogstate/features/spectral.py")},
        elapsed_seconds=time.perf_counter()-started)
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"{mode}: PASS; {len(windows)} window-channel rows; {summary['elapsed_seconds']:.2f}s", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        run(Path(__file__).resolve().parents[2], args.feature_cache, args.output, args.runtime, args.mode)
    return 0
