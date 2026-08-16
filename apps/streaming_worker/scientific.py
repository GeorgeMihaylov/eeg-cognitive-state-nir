"""Leakage-safe fold-1 training and replay for the scientific streaming demo."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from cogstate.features.streaming import (
    build_lightweight_pipeline,
    build_streaming_full_pipeline,
)
from model_zoo.ML.multitask import PMMultiTaskClassifier
from cogstate.protocol import EEG_CHANNELS, PM_METRICS, SAMPLE_RATE, WINDOW_SECONDS
from cogstate.streaming.buffer import Window

from .config import WorkerConfig
from .contracts import (
    feature_schema_hash,
    preprocessing_contract,
    preprocessing_hash,
    stable_hash,
)
from .runtime import StreamingRuntime


TARGET_COLUMNS = tuple(f"target_{metric}" for metric in PM_METRICS)
CLASS_TO_ID = {"low": 0, "medium": 1, "high": 2}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not str(payload.get("experiment_id", "")).startswith("streaming_scientific"):
        raise ValueError("Expected a streaming_scientific experiment_id")
    WorkerConfig.from_dict(payload)
    if payload["model"].get("allow_bootstrap") is not False:
        raise ValueError("Scientific streaming requires model.allow_bootstrap=false")
    if payload["features"].get("profile") not in {"full", "lightweight"}:
        raise ValueError("Scientific streaming requires features.profile full or lightweight")
    pre = payload["preprocessing"]
    if pre.get("bandpass_enabled") or pre.get("notch_enabled") or pre.get("faster"):
        raise ValueError(
            "Scientific streaming v1 is locked to the canonical raw/bypass preprocessing"
        )
    return payload


def _load_fold(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["scientific"]["fixed_fold_manifest"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold_number = int(config["scientific"]["outer_fold"])
    fold = payload["folds"][str(fold_number)]
    if fold.get("subject_overlap"):
        raise ValueError("Fixed fold manifest contains participant overlap")
    if set(fold["train_subject_ids"]) & set(fold["test_subject_ids"]):
        raise ValueError("Train/test participant overlap is not zero")
    return fold


def _load_joined_rows(config: Mapping[str, Any]) -> pd.DataFrame:
    paths = config["scientific"]
    raw = pd.read_parquet(paths["raw_window_index"])
    feature_table = pd.read_parquet(paths["target_table"])
    logical = pd.read_parquet(paths["logical_record_map"])
    keys = ["source", "subject_id", "record_id", "t_start", "t_end"]
    if raw.duplicated(keys).any() or feature_table.duplicated(keys).any():
        raise ValueError("Raw/target join keys are not unique")
    selected = set(logical["selected_record_id"].dropna().astype(str))
    raw = raw[
        raw["record_id"].astype(str).isin(selected)
        & raw["status"].eq("ok")
        & raw["cache_file"].fillna("").ne("")
    ].copy()
    missing_shards = sorted(
        path for path in raw["cache_file"].astype(str).unique() if not Path(path).is_file()
    )
    if missing_shards:
        raise FileNotFoundError(f"Raw cache shards are missing: {missing_shards[:3]}")
    joined = raw.merge(
        feature_table[keys + list(TARGET_COLUMNS)],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Some deduplicated raw windows have no exact target row")
    joined = joined.drop(columns="_merge")
    return joined[joined[list(TARGET_COLUMNS)].notna().all(axis=1)].copy()


def _contract(config: Mapping[str, Any]) -> tuple[list[str], dict[str, Any], str, str]:
    worker = WorkerConfig.from_dict(dict(config))
    pipeline = _build_feature_pipeline(
        config["features"]["profile"], worker.signal.sample_rate
    )
    names = pipeline.feature_names(len(worker.signal.channels))
    pre = preprocessing_contract(
        sample_rate=worker.signal.sample_rate,
        bandpass_enabled=worker.preprocessing.bandpass_enabled,
        bandpass_low_hz=worker.preprocessing.bandpass_low_hz,
        bandpass_high_hz=worker.preprocessing.bandpass_high_hz,
        notch_enabled=worker.preprocessing.notch_enabled,
        notch_hz=worker.preprocessing.notch_hz,
        faster=worker.preprocessing.faster,
    )
    return names, pre, feature_schema_hash(names), preprocessing_hash(pre)


def _build_feature_pipeline(profile: str, sample_rate: float) -> Any:
    if profile == "full":
        return build_streaming_full_pipeline(sample_rate)
    if profile == "lightweight":
        return build_lightweight_pipeline(sample_rate)
    raise ValueError(f"Unknown feature profile {profile!r}")


def _feature_inventory(profile: str, names: list[str]) -> dict[str, Any]:
    spectral_prefixes = (
        "power_", "relpower_", "theta_beta_ratio_", "alpha_theta_ratio_",
        "engagement_index_", "spectral_edge_freq_",
    )
    spectral = [name for name in names if name.startswith(spectral_prefixes)]
    statistical = [name for name in names if name not in spectral]
    return {
        "profile": profile,
        "feature_count": len(names),
        "spectral_feature_count": len(spectral),
        "statistical_feature_count": len(statistical),
        "entropy_included": profile == "full",
        "connectivity_included": profile == "full",
        "feature_names": names,
        "order_deterministic": True,
    }


def _thresholds(train: pd.DataFrame) -> dict[str, list[float]]:
    return {
        metric: [float(value) for value in train[f"target_{metric}"].quantile([1 / 3, 2 / 3])]
        for metric in PM_METRICS
    }


def _labels(rows: pd.DataFrame, thresholds: Mapping[str, Iterable[float]]) -> np.ndarray:
    columns = []
    for metric in PM_METRICS:
        boundaries = np.asarray(list(thresholds[metric]), dtype=float)
        if boundaries.shape != (2,) or not np.all(np.diff(boundaries) > 0):
            raise ValueError(f"Q3 thresholds are not strictly increasing for {metric}")
        columns.append(
            np.searchsorted(boundaries, rows[f"target_{metric}"].to_numpy(float), side="right")
        )
    return np.column_stack(columns).astype(np.int8)


def _evenly_spaced(group: pd.DataFrame, count: int) -> pd.DataFrame:
    ordered = group.sort_values(["t_start", "sample_id"], kind="mergesort")
    if len(ordered) <= count:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered.iloc[np.unique(positions)]


def select_training_rows(
    train: pd.DataFrame,
    labels: np.ndarray,
    *,
    max_windows_per_record: int,
) -> pd.DataFrame:
    labelled = train.copy()
    for index, metric in enumerate(PM_METRICS):
        labelled[f"__q3_{metric}"] = labels[:, index]
    selected = pd.concat(
        [
            _evenly_spaced(group, max_windows_per_record)
            for _, group in labelled.groupby("record_id", sort=True)
        ],
        ignore_index=False,
    )
    # Deterministically supplement a rare class if even spacing missed it.
    selected_ids = set(selected["sample_id"].astype(str))
    for metric in PM_METRICS:
        for class_id in range(3):
            if (selected[f"__q3_{metric}"] == class_id).any():
                continue
            candidates = labelled[
                (labelled[f"__q3_{metric}"] == class_id)
                & ~labelled["sample_id"].astype(str).isin(selected_ids)
            ].sort_values("sample_id")
            if candidates.empty:
                raise ValueError(f"Outer-train has no class {class_id} for {metric}")
            supplement = candidates.iloc[[0]]
            selected = pd.concat([selected, supplement], ignore_index=False)
            selected_ids.add(str(supplement.iloc[0]["sample_id"]))
    return selected.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _extract_one(payload: tuple[str, int, float, str]) -> np.ndarray:
    cache_file, cache_offset, sample_rate, profile = payload
    raw = np.load(cache_file, mmap_mode="r", allow_pickle=False)[cache_offset]
    signal = np.asarray(raw, dtype=np.float32).T
    timestamps = np.arange(len(signal), dtype=float) / sample_rate
    window = Window(0.0, len(signal) / sample_rate, {"eeg": signal}, {"eeg": timestamps})
    return _build_feature_pipeline(profile, sample_rate)(signal, window).astype(np.float32)


def extract_features(
    rows: pd.DataFrame, *, sample_rate: float, workers: int, profile: str = "full"
) -> np.ndarray:
    payloads = [
        (str(row.cache_file), int(row.cache_offset), float(sample_rate), profile)
        for row in rows.itertuples(index=False)
    ]
    if workers == 1:
        values = [_extract_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(_extract_one, payloads, chunksize=1))
    matrix = np.stack(values)
    if matrix.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got {matrix.shape}")
    return matrix


def _reference_protocol(config: Mapping[str, Any]) -> dict[str, Any] | None:
    reference = config["scientific"].get("reference_experiment_dir")
    if not reference:
        return None
    path = Path(reference) / "protocol_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _select_training_rows_for_config(
    config: Mapping[str, Any], train: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    reference = config["scientific"].get("reference_experiment_dir")
    if not reference:
        return select_training_rows(
            train,
            labels,
            max_windows_per_record=int(config["scientific"]["max_windows_per_record"]),
        )
    selection_path = Path(reference) / "training_selection.parquet"
    reference_selection = pd.read_parquet(selection_path)
    ids = reference_selection["sample_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Reference training selection contains duplicate sample_id")
    indexed = train.assign(__sample_id=train["sample_id"].astype(str)).set_index("__sample_id")
    missing = [sample_id for sample_id in ids if sample_id not in indexed.index]
    if missing:
        raise ValueError(f"Reference training sample IDs are unavailable: {missing[:3]}")
    selected = indexed.loc[ids].reset_index(drop=True)
    expected_columns = ["subject_id", "record_id", "source", "t_start", "t_end"]
    for column in expected_columns:
        actual = selected[column].astype(str).tolist()
        expected = reference_selection[column].astype(str).tolist()
        if actual != expected:
            raise ValueError(f"Reference training cohort mismatch in {column}")
    return selected


def plan_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    fold = _load_fold(config)
    rows = _load_joined_rows(config)
    fold_number = int(config["scientific"]["outer_fold"])
    train = rows[rows["subject_id"].isin(fold["train_subject_ids"])]
    test = rows[rows["subject_id"].isin(fold["test_subject_ids"])]
    if set(train["subject_id"]) & set(test["subject_id"]):
        raise ValueError("Participant leakage detected")
    if not train["outer_fold"].ne(fold_number).all() or not test["outer_fold"].eq(fold_number).all():
        raise ValueError("Raw index outer_fold disagrees with the fixed fold manifest")
    names, pre, feature_hash, pre_hash = _contract(config)
    thresholds = _thresholds(train)
    reference_protocol = _reference_protocol(config)
    if reference_protocol is not None:
        if reference_protocol["outer_fold"] != fold_number:
            raise ValueError("Reference experiment uses a different outer fold")
        if reference_protocol["preprocessing_hash"] != pre_hash:
            raise ValueError("Reference signal preprocessing hash differs")
        if stable_hash(reference_protocol["q3_thresholds"]) != stable_hash(thresholds):
            raise ValueError("Reference Q3 thresholds differ from current outer-train thresholds")
        thresholds = reference_protocol["q3_thresholds"]
    labels = _labels(train, thresholds)
    selected = _select_training_rows_for_config(config, train, labels)
    selection_hash = stable_hash(selected["sample_id"].astype(str).tolist())
    if reference_protocol is not None:
        reference_manifest = json.loads(
            (Path(config["scientific"]["reference_bundle_dir"]) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected_hash = reference_manifest["training_fold"]["training_sample_ids_hash"]
        if selection_hash != expected_hash:
            raise ValueError("Training cohort hash differs from the full-profile experiment")
    return {
        "experiment_id": config["experiment_id"],
        "result_status": config["result_status"],
        "outer_fold": fold_number,
        "split_hash": fold["split_hash"],
        "train_participant_ids": fold["train_subject_ids"],
        "test_participant_ids": fold["test_subject_ids"],
        "participant_overlap": [],
        "deduplicated_complete_target_windows": int(len(rows)),
        "outer_train_windows": int(len(train)),
        "outer_test_windows": int(len(test)),
        "training_feature_windows": int(len(selected)),
        "training_record_count": int(selected["record_id"].nunique()),
        "training_sample_ids_hash": selection_hash,
        "reference_experiment_dir": config["scientific"].get("reference_experiment_dir"),
        "reference_protocol_hash": (
            reference_protocol["protocol_hash"] if reference_protocol is not None else None
        ),
        "feature_profile": config["features"]["profile"],
        "feature_count": len(names),
        "feature_schema_hash": feature_hash,
        "feature_inventory": _feature_inventory(config["features"]["profile"], names),
        "preprocessing_contract": pre,
        "preprocessing_hash": pre_hash,
        "signal_preprocessing_hash": pre_hash,
        "q3_thresholds": thresholds,
        "q3_thresholds_hash": stable_hash(thresholds),
        "target_metrics": list(PM_METRICS),
        "target_contract": "outer-train tertiles; searchsorted(side='right')",
        "protocol_hash": stable_hash(
            {
                "config": config,
                "split_hash": fold["split_hash"],
                "feature_schema_hash": feature_hash,
                "preprocessing_hash": pre_hash,
                "q3_thresholds": thresholds,
                "training_sample_ids": selected["sample_id"].astype(str).tolist(),
            }
        ),
    }


def _git_provenance() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)
    return {
        "head": head,
        "working_tree_dirty": bool(status.strip()),
        "working_tree_state_hash": stable_hash(status.splitlines()),
    }


def train_bundle(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    plan = plan_experiment(config_path)
    fold = _load_fold(config)
    rows = _load_joined_rows(config)
    train = rows[rows["subject_id"].isin(fold["train_subject_ids"])].copy()
    labels_all = _labels(train, plan["q3_thresholds"])
    selected = _select_training_rows_for_config(config, train, labels_all)
    labels = _labels(selected, plan["q3_thresholds"])
    started = time.perf_counter()
    features = extract_features(
        selected,
        sample_rate=float(config["signal"]["sample_rate"]),
        workers=int(config["scientific"]["feature_workers"]),
        profile=config["features"]["profile"],
    )
    if features.shape != (len(selected), plan["feature_count"]):
        raise ValueError(f"Unexpected extracted feature shape {features.shape}")
    imputer = SimpleImputer(strategy="median").fit(features)
    model_features = imputer.transform(features)
    if not np.isfinite(model_features).all():
        raise ValueError("Non-finite training features remain after outer-train imputation")
    params = dict(config["scientific"]["xgboost_params"])
    estimator = PMMultiTaskClassifier("xgboost", params=params).fit(model_features, labels)
    elapsed = time.perf_counter() - started

    artifact_dir = Path(config["model"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, artifact_dir / "model.joblib")
    joblib.dump(imputer, artifact_dir / "imputer.joblib")
    manifest = {
        "version": config["scientific"].get(
            "model_version", "pm-xgboost-streaming-v1-fold01-seed42"
        ),
        "model_type": "xgboost_multitask_q3",
        "diagnostic_only": False,
        "model_file": "model.joblib",
        "imputer_file": "imputer.joblib",
        "scaler_file": None,
        "selector_file": None,
        "n_features": plan["feature_count"],
        "feature_profile": plan["feature_profile"],
        "feature_schema_hash": plan["feature_schema_hash"],
        "preprocessing_hash": plan["preprocessing_hash"],
        "signal_preprocessing_hash": plan["signal_preprocessing_hash"],
        "sample_rate": float(config["signal"]["sample_rate"]),
        "channels": list(config["signal"]["channels"]),
        "window_seconds": float(config["windowing"]["window_seconds"]),
        "training_fold": {
            "outer_fold": plan["outer_fold"],
            "split_hash": plan["split_hash"],
            "train_participant_ids": plan["train_participant_ids"],
            "test_participant_ids": plan["test_participant_ids"],
            "training_window_count": len(selected),
            "training_record_count": int(selected["record_id"].nunique()),
            "training_sample_ids_hash": plan["training_sample_ids_hash"],
        },
        "training_participant_ids_hash": stable_hash(plan["train_participant_ids"]),
        "target_metrics": list(PM_METRICS),
        "target_contract": plan["target_contract"],
        "q3_thresholds": plan["q3_thresholds"],
        "q3_thresholds_hash": plan["q3_thresholds_hash"],
        "model_parameters": params,
        "provenance": _git_provenance(),
        "training_seconds": elapsed,
        "protocol_hash": plan["protocol_hash"],
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    output_dir = Path(config["scientific"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "protocol_manifest.json", plan)
    selected[["sample_id", "subject_id", "record_id", "source", "t_start", "t_end"]].to_parquet(
        output_dir / "training_selection.parquet", index=False
    )
    np.save(output_dir / "training_features.npy", features, allow_pickle=False)
    return {**plan, "training_seconds": elapsed, "model_manifest": manifest}


def _consecutive_runs(group: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = group.sort_values(["t_start", "sample_id"], kind="mergesort")
    breaks = ordered["t_start"].diff().fillna(10.0).sub(10.0).abs().gt(1e-6)
    return [part for _, part in ordered.groupby(breaks.cumsum(), sort=False)]


def materialize_replay(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    plan = plan_experiment(config_path)
    reference = config["scientific"].get("reference_experiment_dir")
    if reference:
        reference_manifest = json.loads(
            (Path(reference) / "replay_manifest.json").read_text(encoding="utf-8")
        )
        replay_path = Path(config["source"]["path"])
        actual_hash = _sha256_file(replay_path)
        if actual_hash != reference_manifest["replay_sha256"]:
            raise ValueError("Replay SHA-256 differs from the full-profile experiment")
        if reference_manifest["outer_fold"] != plan["outer_fold"]:
            raise ValueError("Reference replay uses a different outer fold")
        expected_participant = config["scientific"].get("expected_replay_participant_id")
        expected_record = config["scientific"].get("expected_replay_record_id")
        if expected_participant and reference_manifest["participant_id"] != expected_participant:
            raise ValueError("Reference replay participant differs from the locked participant")
        if expected_record and reference_manifest["record_id"] != expected_record:
            raise ValueError("Reference replay record differs from the locked recording")
        output_dir = Path(config["scientific"]["output_dir"])
        _write_json(output_dir / "replay_manifest.json", reference_manifest)
        return reference_manifest
    fold = _load_fold(config)
    rows = _load_joined_rows(config)
    test = rows[rows["subject_id"].isin(fold["test_subject_ids"])]
    canonical_windows = int(
        np.ceil(float(config["scientific"]["replay_duration_seconds"]) / WINDOW_SECONDS)
    )
    candidates: list[pd.DataFrame] = []
    for _, group in test.groupby("record_id", sort=True):
        candidates.extend(run for run in _consecutive_runs(group) if len(run) >= canonical_windows)
    if not candidates:
        raise ValueError("No outer-test record has a long enough exact 10-second run")
    candidates.sort(key=lambda frame: (-len(frame), str(frame.iloc[0]["record_id"])))
    selected = candidates[0].iloc[:canonical_windows].copy()
    arrays = []
    for row in selected.itertuples(index=False):
        raw = np.load(row.cache_file, mmap_mode="r", allow_pickle=False)[int(row.cache_offset)]
        if raw.shape != (len(EEG_CHANNELS), SAMPLE_RATE * WINDOW_SECONDS):
            raise ValueError(f"Unexpected raw cache window shape {raw.shape}")
        arrays.append(np.asarray(raw, dtype=np.float32).T)
    replay = np.concatenate(arrays, axis=0)
    replay_path = Path(config["source"]["path"])
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(replay_path, replay, allow_pickle=False)
    source_path = Path(str(selected.iloc[0]["raw_file_path"]))
    labels = _labels(selected, plan["q3_thresholds"])
    target_windows = []
    for offset, (row, target) in enumerate(zip(selected.itertuples(index=False), labels)):
        target_windows.append(
            {
                "replay_start_seconds": float(offset * WINDOW_SECONDS),
                "replay_end_seconds": float((offset + 1) * WINDOW_SECONDS),
                "sample_id": str(row.sample_id),
                "labels": {metric: int(target[index]) for index, metric in enumerate(PM_METRICS)},
            }
        )
    manifest = {
        "participant_id": str(selected.iloc[0]["subject_id"]),
        "record_id": str(selected.iloc[0]["record_id"]),
        "record_group_id": str(selected.iloc[0]["record_group_id"]),
        "source": str(selected.iloc[0]["source"]),
        "outer_fold": plan["outer_fold"],
        "outer_test_only": True,
        "participant_in_outer_train": False,
        "sampling_rate_hz": SAMPLE_RATE,
        "channels": list(EEG_CHANNELS),
        "duration_seconds": len(replay) / SAMPLE_RATE,
        "sample_count": int(len(replay)),
        "replay_sha256": _sha256_file(replay_path),
        "source_file": source_path.as_posix(),
        "source_file_sha256": _sha256_file(source_path),
        "canonical_target_windows": target_windows,
    }
    output_dir = Path(config["scientific"]["output_dir"])
    _write_json(output_dir / "replay_manifest.json", manifest)
    return manifest


def _latency_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def run_replay(config_path: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["scientific"]["output_dir"])
    replay_manifest_path = output_dir / "replay_manifest.json"
    if not replay_manifest_path.exists():
        materialize_replay(config_path)
    predictions_path = Path(config["output"]["jsonl_path"])
    if predictions_path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to append to existing {predictions_path}")
        predictions_path.unlink()
    started = time.perf_counter()
    runtime = StreamingRuntime(WorkerConfig.from_dict(config))
    runtime.run()
    wall_seconds = time.perf_counter() - started
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    accepted = [row for row in rows if row["raw_prediction"] is not None]
    required = int(config["scientific"]["minimum_prediction_windows"])
    if len(accepted) < required:
        raise RuntimeError(f"Only {len(accepted)} predictions were produced; require {required}")
    for row in accepted:
        if row["diagnostic_model"]:
            raise RuntimeError("Scientific replay used a diagnostic model")
        targets = row["raw_prediction"]["target_probabilities"]
        if set(targets) != set(PM_METRICS) or len(targets) != len(PM_METRICS):
            raise RuntimeError("Prediction does not contain the seven canonical PM targets")
        for probabilities in targets.values():
            values = np.asarray(list(probabilities.values()), dtype=float)
            if not np.isfinite(values).all() or not np.isclose(values.sum(), 1.0, atol=1e-6):
                raise RuntimeError("Prediction probabilities are invalid")

    latency = pd.DataFrame(
        [
            {"window_start": row["window_start"], "window_end": row["window_end"], **row["stage_latencies_ms"]}
            for row in accepted
        ]
    )
    latency.to_csv(output_dir / "latency_per_window.csv", index=False, lineterminator="\n")
    latency_summary = {
        stage: _latency_stats(latency[stage].to_numpy(float))
        for stage in ("preprocessing", "feature_extraction", "inference", "total_processing")
    }
    replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    replay_duration = float(replay_manifest["duration_seconds"])
    latency_summary.update(
        {
            "windows_processed": runtime.processed_windows,
            "windows_rejected": runtime.rejected_windows,
            "samples_rejected": runtime.rejected_samples,
            "replay_duration_seconds": replay_duration,
            "wall_clock_seconds": wall_seconds,
            "real_time_factor": replay_duration / wall_seconds,
            "scope_note": "processing latency only; not full live sensor-to-user latency",
        }
    )
    p95_ms = latency_summary["total_processing"]["p95_ms"]
    latency_summary["latency_headroom_ms"] = 1000.0 - p95_ms
    latency_summary["step_utilization"] = p95_ms / 1000.0
    latency_summary["near_real_time_step_1s"] = (
        "supported" if p95_ms < 1000.0 else "not_supported"
    )
    _write_json(output_dir / "latency_summary.json", latency_summary)
    rejection_reasons: dict[str, int] = {}
    for row in rows:
        if row["raw_prediction"] is not None:
            continue
        for reason in row["quality"].get("reasons", []):
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    quality = {
        "processed_windows": runtime.processed_windows,
        "rejected_windows": runtime.rejected_windows,
        "rejected_samples": runtime.rejected_samples,
        "rejection_reasons": rejection_reasons,
    }
    _write_json(output_dir / "quality_summary.json", quality)

    by_start = {round(float(row["window_start"]), 6): row for row in accepted}
    metric_rows: dict[str, dict[str, float | int]] = {}
    for metric in PM_METRICS:
        y_true: list[int] = []
        y_pred: list[int] = []
        for target_window in replay_manifest["canonical_target_windows"]:
            prediction = by_start.get(round(float(target_window["replay_start_seconds"]), 6))
            if prediction is None:
                continue
            y_true.append(int(target_window["labels"][metric]))
            label = prediction["raw_prediction"]["target_labels"][metric]
            y_pred.append(CLASS_TO_ID[label])
        metric_rows[metric] = {
            "n_aligned_windows": len(y_true),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
    metrics = {
        "alignment": "exact replay-relative 10-second window starts only; no nearest-neighbour merge",
        "per_pm": metric_rows,
        "pm_macro": {
            key: float(np.mean([values[key] for values in metric_rows.values()]))
            for key in ("macro_f1", "balanced_accuracy", "accuracy")
        },
    }
    _write_json(output_dir / "classification_metrics.json", metrics)

    suppressed = 0
    raw_switches = 0
    post_switches = 0
    previous_raw: dict[str, str] = {}
    previous_post: dict[str, str] = {}
    for row in accepted:
        raw_labels = row["raw_prediction"]["target_labels"]
        post_labels = row["postprocessed_prediction"]["target_labels"]
        suppressed += sum(label == "unknown" for label in post_labels.values())
        raw_switches += sum(previous_raw.get(metric, label) != label for metric, label in raw_labels.items())
        post_switches += sum(previous_post.get(metric, label) != label for metric, label in post_labels.items())
        previous_raw, previous_post = raw_labels, post_labels
    postprocessing = {
        "suppressed_low_confidence_outputs": suppressed,
        "raw_state_switches": raw_switches,
        "postprocessed_state_switches": post_switches,
        "smoothed_switches": max(0, raw_switches - post_switches),
        "confirmation_windows": config["postprocessing"]["confirmation_windows"],
        "confirmation_delay_seconds_max": (
            config["postprocessing"]["confirmation_windows"] - 1
        ) * config["windowing"]["step_seconds"],
    }
    manifest_path = Path(config["model"]["artifact_dir"]) / "manifest.json"
    shutil.copyfile(manifest_path, output_dir / "model_manifest_snapshot.json")
    shutil.copyfile(config_path, output_dir / "streaming_config_snapshot.yaml")
    summary = {
        "experiment_id": config["experiment_id"],
        "result_status": config["result_status"],
        "diagnostic_model": False,
        "model_version": accepted[0]["model_version"],
        "model_manifest_sha256": _sha256_file(manifest_path),
        "latency": latency_summary,
        "quality": quality,
        "classification": metrics,
        "postprocessing": postprocessing,
        "near_real_time_step_1s": latency_summary["near_real_time_step_1s"],
    }
    comparison = _write_comparison_artifact(config, summary)
    if comparison is not None:
        summary["comparison_full_vs_lightweight"] = comparison
    _write_json(output_dir / "run_summary.json", summary)
    _write_report(
        output_dir / f"{config['experiment_id']}.md", config, summary, replay_manifest
    )
    return summary


def _write_comparison_artifact(
    config: Mapping[str, Any], summary: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    reference = config["scientific"].get("reference_experiment_dir")
    if not reference:
        return None
    full_dir = Path(reference)
    full_summary = json.loads((full_dir / "run_summary.json").read_text(encoding="utf-8"))
    full_protocol = json.loads(
        (full_dir / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    light_protocol = json.loads(
        (Path(config["scientific"]["output_dir"]) / "protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    full_predictions = [
        json.loads(line)
        for line in (full_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    light_predictions_path = Path(config["output"]["jsonl_path"])
    light_predictions = [
        json.loads(line)
        for line in light_predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    full_starts = {
        round(float(row["window_start"]), 6)
        for row in full_predictions
        if row["raw_prediction"] is not None
    }
    light_starts = {
        round(float(row["window_start"]), 6)
        for row in light_predictions
        if row["raw_prediction"] is not None
    }
    if full_starts != light_starts:
        raise ValueError("Full/lightweight evaluation window starts differ")
    metrics = [
        ("feature_count", full_protocol["feature_count"], light_protocol["feature_count"]),
        (
            "feature_extraction_mean_ms",
            full_summary["latency"]["feature_extraction"]["mean_ms"],
            summary["latency"]["feature_extraction"]["mean_ms"],
        ),
        (
            "feature_extraction_p95_ms",
            full_summary["latency"]["feature_extraction"]["p95_ms"],
            summary["latency"]["feature_extraction"]["p95_ms"],
        ),
        (
            "model_p95_ms",
            full_summary["latency"]["inference"]["p95_ms"],
            summary["latency"]["inference"]["p95_ms"],
        ),
        (
            "total_processing_p95_ms",
            full_summary["latency"]["total_processing"]["p95_ms"],
            summary["latency"]["total_processing"]["p95_ms"],
        ),
        (
            "wall_clock_seconds",
            full_summary["latency"]["wall_clock_seconds"],
            summary["latency"]["wall_clock_seconds"],
        ),
        (
            "real_time_factor",
            full_summary["latency"]["real_time_factor"],
            summary["latency"]["real_time_factor"],
        ),
        (
            "pm_macro_macro_f1",
            full_summary["classification"]["pm_macro"]["macro_f1"],
            summary["classification"]["pm_macro"]["macro_f1"],
        ),
        (
            "pm_macro_balanced_accuracy",
            full_summary["classification"]["pm_macro"]["balanced_accuracy"],
            summary["classification"]["pm_macro"]["balanced_accuracy"],
        ),
    ]
    rows = [
        {
            "metric": metric,
            "full": float(full),
            "lightweight": float(light),
            "change_lightweight_minus_full": float(light) - float(full),
        }
        for metric, full, light in metrics
    ]
    pd.DataFrame(rows).to_csv(
        Path(config["scientific"]["output_dir"]) / "comparison_full_vs_lightweight.csv",
        index=False,
        lineterminator="\n",
    )
    return rows


def _write_report(path: Path, config: Mapping[str, Any], summary: Mapping[str, Any], replay: Mapping[str, Any]) -> None:
    latency = summary["latency"]
    metrics = summary["classification"]
    lines = [
        f"# {config['experiment_id']}",
        "",
        f"Result status: `{config['result_status']}`. Model: `{summary['model_version']}`; diagnostic model: `false`.",
        "",
        f"Replay: participant `{replay['participant_id']}`, record `{replay['record_id']}`, "
        f"{replay['duration_seconds']:.1f} s, outer fold {replay['outer_fold']} test only.",
        "",
        f"Processed/rejected windows: {latency['windows_processed']}/{latency['windows_rejected']}. "
        f"Wall time: {latency['wall_clock_seconds']:.3f} s; real-time factor: {latency['real_time_factor']:.3f}x.",
        f"Step-1s decision: `{latency['near_real_time_step_1s']}`; P95 headroom: "
        f"{latency['latency_headroom_ms']:.3f} ms; utilization: {latency['step_utilization']:.4f}.",
        "",
        "## Latency, ms",
        "",
        "| Stage | Mean | Median | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in ("preprocessing", "feature_extraction", "inference", "total_processing"):
        values = latency[stage]
        lines.append(
            f"| {stage} | {values['mean_ms']:.3f} | {values['median_ms']:.3f} | "
            f"{values['p95_ms']:.3f} | {values['p99_ms']:.3f} | {values['max_ms']:.3f} |"
        )
    lines += ["", "## Exact-alignment raw-model metrics", "", "| PM | N | Macro F1 | Balanced accuracy | Accuracy |", "|---|---:|---:|---:|---:|"]
    for metric, values in metrics["per_pm"].items():
        lines.append(
            f"| {metric} | {values['n_aligned_windows']} | {values['macro_f1']:.4f} | "
            f"{values['balanced_accuracy']:.4f} | {values['accuracy']:.4f} |"
        )
    macro = metrics["pm_macro"]
    lines.append(f"| PM-macro | — | {macro['macro_f1']:.4f} | {macro['balanced_accuracy']:.4f} | {macro['accuracy']:.4f} |")
    lines += ["", "Processing latency is not a full live sensor-to-user latency measurement.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def check_api(config_path: str | Path, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        raise RuntimeError("FastAPI test dependencies are not installed") from exc
    from .api.app import create_app

    config = load_config(config_path)
    output_dir = Path(config["scientific"]["output_dir"]) / "api"
    output_dir.mkdir(parents=True, exist_ok=True)
    api_config = json.loads(json.dumps(config))
    api_config["output"]["jsonl_path"] = None
    worker = WorkerConfig.from_dict(api_config)
    with TestClient(create_app(config=worker)) as client:
        deadline = time.monotonic() + timeout_seconds
        status_payload = None
        while time.monotonic() < deadline:
            status_response = client.get("/v1/status")
            status_payload = status_response.json()
            if status_payload.get("last_error"):
                raise RuntimeError(status_payload["last_error"])
            if status_payload.get("processed_windows", 0) >= 1:
                break
            time.sleep(0.1)
        if not status_payload or status_payload.get("processed_windows", 0) < 1:
            raise TimeoutError("API worker did not produce a prediction in time")
        responses = {
            "health": client.get("/health").json(),
            "status": client.get("/v1/status").json(),
            "latest": client.get("/v1/predictions/latest").json(),
        }
    for name, payload in responses.items():
        _write_json(output_dir / f"{name}.json", payload)
    return responses


__all__ = [
    "check_api",
    "extract_features",
    "load_config",
    "materialize_replay",
    "plan_experiment",
    "run_replay",
    "select_training_rows",
    "train_bundle",
]
