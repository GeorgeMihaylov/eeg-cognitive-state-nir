"""Leakage-safe COG-BCI N-Back preprocessing ablation orchestration."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import signal

from bench.analysis.cog_bci_nback_diagnostics import (
    aggregate_spectral_records,
    audit_source_units,
    evaluate_subject_disjoint,
    spectral_features,
)
from bench.datasets.cog_bci_dataset import COGBCIDataset
from bench.datasets.cog_bci_window_cache import (
    BUILDER_VERSION,
    COGBCIWindowBuilder,
    RawWindowSpec,
    audit_window_index,
)
from bench.experiments.cog_bci_nback_baseline import (
    BaselineRunOptions,
    COGBCINBackBaselineRunner,
)
from bench.preprocessing.cog_bci_preprocessing import (
    COGBCIWholeRecordPreprocessing,
    VARIANT_ORDER,
    build_preprocessing_variants,
)


RESULT_STATUS = "diagnostic"
EXPECTED_RECORDS = 261
EXPECTED_WINDOWS = 16927
EXPECTED_SHAPE = (14, 2560)
SIMPLICITY_ORDER = (
    "A_raw",
    "C_notch",
    "D_bandpass",
    "G_bandpass_notch",
    "B_record_demean",
    "E_demean_notch",
    "F_demean_bandpass",
    "H_demean_bandpass_notch",
)
TASK_PATHS = {
    "zero_back": "zeroBACK",
    "one_back": "oneBACK",
    "two_back": "twoBACK",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.drive:
        raise ValueError(f"{label} must be repository-relative")
    return path


def _shard_stem(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class AblationPaths:
    repository_root: Path
    source_root: Path
    source_cache: Path
    protocol_dir: Path
    output_dir: Path
    tracked_report: Path


def _resolve_paths(
    config: Mapping[str, Any], repository_root: Path
) -> AblationPaths:
    return AblationPaths(
        repository_root=repository_root,
        source_root=repository_root
        / _relative(config["source_dataset_root"], label="source_dataset_root"),
        source_cache=repository_root
        / _relative(config["input_cache"], label="input_cache"),
        protocol_dir=repository_root
        / _relative(config["task_protocol"], label="task_protocol"),
        output_dir=repository_root
        / _relative(config["output_dir"], label="output_dir"),
        tracked_report=repository_root
        / _relative(config["tracked_report"], label="tracked_report"),
    )


def _input_paths(paths: AblationPaths) -> dict[str, Path]:
    return {
        "source_cache_manifest": paths.source_cache / "dataset_manifest.json",
        "window_index": paths.source_cache / "window_index.parquet",
        "task_definition": paths.protocol_dir / "task_definition.json",
        "target_index": paths.protocol_dir / "target_index.parquet",
        "outer_assignments": paths.protocol_dir / "outer_assignments.parquet",
        "outer_folds": paths.protocol_dir / "outer_folds.json",
        "inner_assignments": paths.protocol_dir / "inner_assignments.parquet",
        "inner_folds": paths.protocol_dir / "inner_folds.json",
    }


def _input_hashes(paths: AblationPaths) -> dict[str, str]:
    result = {}
    for name, path in _input_paths(paths).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing ablation input: {path}")
        result[name] = _sha256(path)
    return result


def _source_provenance(
    paths: AblationPaths, config: Mapping[str, Any]
) -> dict[str, Any]:
    audit_config = config["unit_audit"]
    unit_audit = audit_source_units(
        paths.source_root,
        subjects=audit_config["subjects"],
        sessions=audit_config["sessions"],
        task_variants=audit_config["task_variants"],
    )
    header_evidence = []
    from scipy import io as scipy_io

    for row in unit_audit["files"]:
        session_source = str(row["session_id"]).replace("ses-0", "ses-S")
        task = TASK_PATHS[str(row["task_variant"])]
        candidates = sorted(
            (paths.source_root / str(row["subject_id"])).rglob(
                f"{session_source}/eeg/{task}.set"
            )
        )
        if len(candidates) != 1:
            raise FileNotFoundError("Could not resolve audited EEGLAB header")
        header = scipy_io.loadmat(
            candidates[0], squeeze_me=True, struct_as_record=False
        )
        header_evidence.append(
            {
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "task_variant": row["task_variant"],
                "EEG_srate": float(header["srate"]),
                "EEG_ref": str(header.get("ref", "")),
                "EEG_comments": str(header.get("comments", "")),
                "EEG_history": str(header.get("history", "")),
                "EEG_xmin": float(header["xmin"]),
                "EEG_xmax": float(header["xmax"]),
                "EEG_pnts": int(header["pnts"]),
                "EEG_etc_fields": list(
                    getattr(header.get("etc"), "_fieldnames", []) or []
                ),
                "EEG_chaninfo_fields": list(
                    getattr(header.get("chaninfo"), "_fieldnames", []) or []
                ),
                "EEG_urchanloc_count": int(
                    len(np.atleast_1d(header.get("urchanlocs", [])))
                ),
            }
        )
    pdf_path = paths.repository_root / "data/raw/cog_bci/metadata/COG-BCI_info.pdf"
    trigger_path = (
        paths.repository_root / "data/raw/cog_bci/metadata/triggerlist.txt"
    )
    notebook_path = (
        paths.repository_root / "data/raw/cog_bci/metadata/notebook.mat"
    )
    return {
        "status": "partially_confirmed",
        "source_unit_declared": False,
        "source_unit_evidence": unit_audit["evidence"],
        "mne_output_unit": unit_audit["mne_output_unit"],
        "mne_calibration_factor": unit_audit["mne_applied_factor"],
        "hardware_filter_status": "unknown",
        "software_filter_status": "unknown",
        "notch_filter_status": "unknown",
        "reference_status": "partially_confirmed_EEG.ref_common_without_numeric_reference",
        "provenance_status": "partially_confirmed",
        "scale_factor": 1.0,
        "sampling_rate_hz": 500.0,
        "documentation_findings": {
            "acquisition": (
                "64 active Ag-AgCl electrodes with ActiChamp amplifier; "
                "Cz unavailable for participants 1-9; ECG recorded separately"
            ),
            "units_gain_filters_reference": "not_declared",
            "notebook_scope": "task order, interruptions, dates and notes",
            "triggerlist_scope": "event-code semantics only",
        },
        "documentation_hashes": {
            "COG-BCI_info.pdf": _sha256(pdf_path),
            "triggerlist.txt": _sha256(trigger_path),
            "notebook.mat": _sha256(notebook_path),
        },
        "audited_file_count": unit_audit["audited_file_count"],
        "header_evidence": header_evidence,
        "unit_audit": unit_audit,
    }


def _load_protocol_frames(
    paths: AblationPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = pd.read_parquet(paths.protocol_dir / "target_index.parquet")
    target = target.loc[
        target["included_for_supervised"].astype(bool)
        & target["status"].eq("accepted")
    ].copy()
    outer = pd.read_parquet(paths.protocol_dir / "outer_assignments.parquet")[
        ["sample_id", "fold"]
    ].rename(columns={"fold": "outer_fold"})
    target = target.merge(outer, on="sample_id", validate="one_to_one")
    inner = pd.read_parquet(paths.protocol_dir / "inner_assignments.parquet")
    if (
        len(target) != EXPECTED_WINDOWS
        or target["record_id"].nunique() != EXPECTED_RECORDS
        or target["subject_id"].nunique() != 29
        or sorted(target["target"].unique().tolist()) != [0, 1, 2]
    ):
        raise RuntimeError("Canonical N-Back target contract changed")
    return target, outer, inner


def _build_sample_mapping(
    raw_target: pd.DataFrame,
    variant_index: pd.DataFrame,
) -> pd.DataFrame:
    variant = variant_index.loc[variant_index["status"].eq("accepted")].copy()
    keys = [
        "record_id",
        "window_index",
        "start_sample",
        "stop_sample",
        "start_time_seconds",
        "stop_time_seconds",
    ]
    raw = raw_target[
        ["sample_id", *keys, "subject_id", "session_id", "target", "class_name"]
    ].rename(columns={"sample_id": "raw_sample_id"})
    mapping = raw.merge(
        variant[["sample_id", *keys]].rename(
            columns={"sample_id": "variant_sample_id"}
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if mapping["variant_sample_id"].isna().any():
        raise RuntimeError("Variant cache changed canonical window boundaries")
    if (
        mapping["raw_sample_id"].duplicated().any()
        or mapping["variant_sample_id"].duplicated().any()
        or mapping["raw_sample_id"].eq(mapping["variant_sample_id"]).any()
    ):
        raise RuntimeError("Variant sample IDs are not isolated by preprocessing")
    return mapping


def _materialize_caches(
    paths: AblationPaths,
    config: Mapping[str, Any],
    variants: Sequence[COGBCIWholeRecordPreprocessing],
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, Path]]:
    dataset = COGBCIDataset(
        {
            "data_path": paths.source_root,
            "index_cache_path": paths.repository_root
            / _relative(config["index_cache"], label="index_cache"),
            "use_index_cache": True,
            "require_canonical_complete": True,
        }
    )
    raw_index = pd.read_parquet(paths.source_cache / "window_index.parquet")
    raw_rows = raw_index.loc[
        raw_index["sample_id"].isin(target["sample_id"])
    ].copy()
    if len(raw_rows) != EXPECTED_WINDOWS:
        raise RuntimeError("Source raw cache no longer covers canonical target")
    spec = RawWindowSpec(
        window_duration_seconds=5.12,
        window_stride_seconds=5.12,
        drop_incomplete_window=True,
        minimum_valid_fraction=1.0,
        segmentation_mode="record_full",
        preprocessing="none",
        target_sampling_rate_hz=500.0,
        allow_filtering_when_source_status_unknown=True,
    )
    records = dataset.query(task_families=["n_back"])
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(f"Expected 261 N-Back records, got {len(records)}")
    inventory = []
    cache_paths: dict[str, Path] = {}
    mapping_paths: dict[str, Path] = {}
    for variant in variants:
        variant_dir = paths.output_dir / variant.variant_id
        cache_paths[variant.variant_id] = (
            paths.source_cache if variant.is_identity else variant_dir
        )
        mapping_path = variant_dir / "sample_id_mapping.parquet"
        mapping_paths[variant.variant_id] = mapping_path
        if variant.is_identity:
            variant_dir.mkdir(parents=True, exist_ok=True)
            mapping = target[
                [
                    "sample_id",
                    "record_id",
                    "window_index",
                    "start_sample",
                    "stop_sample",
                    "start_time_seconds",
                    "stop_time_seconds",
                    "subject_id",
                    "session_id",
                    "target",
                    "class_name",
                ]
            ].rename(columns={"sample_id": "raw_sample_id"})
            mapping["variant_sample_id"] = mapping["raw_sample_id"]
            mapping.to_parquet(mapping_path, index=False)
            _write_json(
                variant_dir / "cache_reference.json",
                {
                    "result_status": RESULT_STATUS,
                    "immutable_reference": True,
                    "cache_path": str(
                        paths.source_cache.relative_to(paths.repository_root)
                    ).replace("\\", "/"),
                    "config_hash": json.loads(
                        (paths.source_cache / "dataset_manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )["config_hash"],
                },
            )
            cache_hash = json.loads(
                (paths.source_cache / "dataset_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["config_hash"]
            size_bytes = 0
            materialized = False
        else:
            builder = COGBCIWindowBuilder(
                dataset,
                output_dir=variant_dir,
                channel_policy_name="emotiv_common",
                spec=spec,
                whole_record_preprocessor=variant,
            )
            summary = builder.run(records, resume=True)
            if summary["record_count"] != EXPECTED_RECORDS:
                raise RuntimeError("Variant cache record count mismatch")
            variant_index = pd.read_parquet(variant_dir / "window_index.parquet")
            mapping = _build_sample_mapping(target, variant_index)
            mapping.to_parquet(mapping_path, index=False)
            audit = audit_window_index(variant_index)
            if not audit["leakage_safe"]:
                raise RuntimeError("Variant window identity audit failed")
            _write_json(variant_dir / "leakage_audit.json", audit)
            cache_hash = builder.config_hash
            size_bytes = sum(
                path.stat().st_size
                for path in variant_dir.rglob("*")
                if path.is_file()
            )
            materialized = True
        manifest = json.loads(
            (cache_paths[variant.variant_id] / "dataset_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        inventory.append(
            {
                "variant_id": variant.variant_id,
                "preprocessing_name": variant.name,
                "cache_path": str(
                    cache_paths[variant.variant_id].relative_to(
                        paths.repository_root
                    )
                ).replace("\\", "/"),
                "materialized": materialized,
                "records": EXPECTED_RECORDS,
                "accepted_windows": EXPECTED_WINDOWS,
                "shape": json.dumps(list(EXPECTED_SHAPE)),
                "sampling_rate_hz": manifest["sampling_rate_hz"],
                "channel_order": json.dumps(manifest["channel_order"]),
                "config_hash": cache_hash,
                "preprocessing_hash": variant.stable_hash(
                    channels=manifest["channel_order"],
                    loader_schema_version=BUILDER_VERSION,
                ),
                "mapping_hash": _sha256(mapping_path),
                "size_bytes": size_bytes,
                "complete": True,
            }
        )
    return pd.DataFrame(inventory), cache_paths, mapping_paths


def _band_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies <= high)
    if int(mask.sum()) < 2:
        raise ValueError(f"Insufficient spectral bins for {low}-{high} Hz")
    return np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)


def spectral_qc_features(
    windows: np.ndarray, *, sampling_rate: float
) -> pd.DataFrame:
    """Window-level spectral QC preserving DC for the 0-1 Hz audit."""

    array = np.asarray(windows, dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("Spectral QC requires finite [window, channel, time]")
    dc = array.mean(axis=-1)
    standard_deviation = array.std(axis=-1)
    frequencies, psd = signal.welch(
        array,
        fs=sampling_rate,
        nperseg=min(512, array.shape[-1]),
        noverlap=min(256, array.shape[-1] // 2),
        detrend=False,
        scaling="density",
        axis=-1,
    )
    powers = {
        "power_0_1": _band_power(frequencies, psd, 0.0, 1.0),
        "power_1_45": _band_power(frequencies, psd, 1.0, 45.0),
        "power_49_51": _band_power(frequencies, psd, 49.0, 51.0),
        "theta_power": _band_power(frequencies, psd, 4.0, 8.0),
        "alpha_power": _band_power(frequencies, psd, 8.0, 13.0),
        "beta_power": _band_power(frequencies, psd, 13.0, 30.0),
    }
    epsilon = np.finfo(np.float64).tiny
    result = {
        "dc_magnitude": np.median(np.abs(dc), axis=1),
        "within_record_channel_std": np.median(standard_deviation, axis=1),
        "dc_std_ratio": np.median(
            np.abs(dc) / np.maximum(standard_deviation, epsilon), axis=1
        ),
    }
    for name, values in powers.items():
        result[name] = np.median(values, axis=1)
    result["line_to_1_45_ratio"] = np.median(
        powers["power_49_51"] / np.maximum(powers["power_1_45"], epsilon),
        axis=1,
    )
    result["theta_alpha"] = np.median(
        powers["theta_power"] / np.maximum(powers["alpha_power"], epsilon),
        axis=1,
    )
    result["theta_beta"] = np.median(
        powers["theta_power"] / np.maximum(powers["beta_power"], epsilon),
        axis=1,
    )
    frame = pd.DataFrame(result)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Spectral QC produced NaN or Inf")
    return frame


def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    total = float(np.sum((array - array.mean()) ** 2))
    if total <= 0:
        return 0.0
    grouped_mean = values.groupby(groups).mean()
    counts = groups.value_counts()
    between = sum(
        float(counts[group]) * (float(mean) - float(array.mean())) ** 2
        for group, mean in grouped_mean.items()
    )
    return float(between / total)


def _variant_spectral_tables(
    cache_dir: Path,
    mapping_path: Path,
    target: pd.DataFrame,
    *,
    raw_variant: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cache_index = pd.read_parquet(cache_dir / "window_index.parquet")
    cache_index = cache_index.loc[cache_index["status"].eq("accepted")]
    mapping = pd.read_parquet(mapping_path)
    selected = mapping[
        [
            "raw_sample_id",
            "record_id",
            "window_index",
            *([] if raw_variant else ["variant_sample_id"]),
        ]
    ].copy()
    selected["cache_sample_id"] = (
        selected["raw_sample_id"]
        if raw_variant
        else selected.pop("variant_sample_id")
    )
    selected = selected.merge(
        cache_index[
            ["sample_id", "record_id", "window_index", "cache_offset"]
        ].rename(columns={"sample_id": "cache_sample_id"}),
        on=["cache_sample_id", "record_id", "window_index"],
        validate="one_to_one",
    )
    metadata = target[
        [
            "sample_id",
            "subject_id",
            "session_id",
            "record_id",
            "window_index",
            "target",
            "class_name",
            "outer_fold",
        ]
    ].rename(columns={"sample_id": "raw_sample_id"})
    selected = selected.merge(
        metadata,
        on=["raw_sample_id", "record_id", "window_index"],
        validate="one_to_one",
    )
    channel_order = json.loads(
        (cache_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )["channel_order"]
    feature_names: list[str] | None = None
    frames = []
    qc_frames = []
    for record_id, group in selected.groupby("record_id", sort=True):
        ordered = group.sort_values("window_index", kind="stable")
        array = np.load(
            cache_dir / "shards" / f"{_shard_stem(str(record_id))}.npy",
            mmap_mode="r",
        )
        windows = np.asarray(
            array[ordered["cache_offset"].to_numpy(dtype=int)],
            dtype=np.float32,
        )
        model_features, current_names = spectral_features(
            windows,
            sampling_rate=500.0,
            channel_names=channel_order,
        )
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise RuntimeError("Spectral feature order changed")
        identity = ordered[
            [
                "raw_sample_id",
                "subject_id",
                "session_id",
                "record_id",
                "window_index",
                "target",
                "class_name",
                "outer_fold",
            ]
        ].reset_index(drop=True)
        frames.append(
            pd.concat(
                [
                    identity,
                    pd.DataFrame(model_features, columns=current_names),
                ],
                axis=1,
            )
        )
        qc_frames.append(
            pd.concat(
                [identity, spectral_qc_features(windows, sampling_rate=500.0)],
                axis=1,
            )
        )
    if feature_names is None:
        raise RuntimeError("No spectral features were extracted")
    window_features = pd.concat(frames, ignore_index=True)
    window_qc = pd.concat(qc_frames, ignore_index=True)
    record_features = aggregate_spectral_records(
        window_features, feature_names
    )
    return record_features, window_qc, feature_names


def _spectral_and_lightweight(
    paths: AblationPaths,
    config: Mapping[str, Any],
    variants: Sequence[COGBCIWholeRecordPreprocessing],
    target: pd.DataFrame,
    inner: pd.DataFrame,
    cache_paths: Mapping[str, Path],
    mapping_paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qc_rows = []
    fold_frames = []
    aggregate_rows = []
    for variant in variants:
        record_features, window_qc, feature_names = _variant_spectral_tables(
            cache_paths[variant.variant_id],
            mapping_paths[variant.variant_id],
            target,
            raw_variant=variant.is_identity,
        )
        record_features.to_parquet(
            paths.output_dir
            / variant.variant_id
            / "spectral_record_features.parquet",
            index=False,
        )
        qc_columns = [
            column
            for column in window_qc
            if column
            not in {
                "raw_sample_id",
                "subject_id",
                "session_id",
                "record_id",
                "window_index",
                "target",
                "class_name",
                "outer_fold",
            }
        ]
        record_qc = (
            window_qc.groupby(
                [
                    "record_id",
                    "subject_id",
                    "session_id",
                    "target",
                    "class_name",
                ],
                as_index=False,
            )[qc_columns]
            .median()
        )
        row: dict[str, Any] = {
            "variant_id": variant.variant_id,
            "preprocessing_name": variant.name,
            "windows": len(window_qc),
            "records": len(record_qc),
            "nonfinite_values": int(
                (~np.isfinite(window_qc[qc_columns].to_numpy())).sum()
            ),
            "model_feature_count": len(feature_names),
        }
        for column in qc_columns:
            row[f"median_{column}"] = float(window_qc[column].median())
        for group_name, group_column in (
            ("class", "target"),
            ("subject", "subject_id"),
            ("session", "session_id"),
        ):
            values = [
                _eta_squared(record_qc[column], record_qc[group_column])
                for column in qc_columns
            ]
            row[f"{group_name}_eta_squared_mean"] = float(np.mean(values))
            row[f"{group_name}_eta_squared_median"] = float(np.median(values))
        qc_rows.append(row)
        _, folds, evaluation = evaluate_subject_disjoint(
            record_features, inner, seed=int(config["seed"])
        )
        folds.insert(0, "variant_id", variant.variant_id)
        folds.insert(1, "preprocessing_name", variant.name)
        fold_frames.append(folds)
        for model, group in folds.groupby("model", sort=True):
            aggregate_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "preprocessing_name": variant.name,
                    "model": model,
                    **{
                        f"{column}_{statistic}": float(
                            getattr(group[column], statistic)(ddof=0)
                            if statistic == "std"
                            else getattr(group[column], statistic)()
                        )
                        for column in (
                            "validation_balanced_accuracy",
                            "validation_macro_f1",
                            "validation_ordinal_mae",
                            "validation_qwk",
                            "test_balanced_accuracy",
                            "test_macro_f1",
                            "test_ordinal_mae",
                            "test_qwk",
                        )
                        for statistic in ("mean", "std")
                    },
                    "scaler_fit_partition": "inner_train",
                    "outer_test_used_for_fit": False,
                    "scaler_audit_hash": _stable_hash(
                        evaluation["scaler_audit"]
                    ),
                }
            )
    qc = pd.DataFrame(qc_rows)
    folds = pd.concat(fold_frames, ignore_index=True)
    aggregates = pd.DataFrame(aggregate_rows)
    qc.to_csv(paths.output_dir / "spectral_qc_by_variant.csv", index=False)
    raw = qc.loc[qc["variant_id"].eq("A_raw")].iloc[0]
    effect_rows = []
    for _, row in qc.iterrows():
        effect = {
            "variant_id": row["variant_id"],
            "preprocessing_name": row["preprocessing_name"],
        }
        for column in [
            item
            for item in qc.columns
            if item.startswith("median_") or "_eta_squared_" in item
        ]:
            effect[f"delta_{column}"] = float(row[column] - raw[column])
            if column.startswith("median_") and float(raw[column]) != 0:
                effect[f"ratio_{column}"] = float(row[column] / raw[column])
        effect_rows.append(effect)
    pd.DataFrame(effect_rows).to_csv(
        paths.output_dir / "spectral_effects.csv", index=False
    )
    folds.to_csv(paths.output_dir / "lightweight_fold_metrics.csv", index=False)
    aggregates.to_csv(
        paths.output_dir / "lightweight_aggregate_metrics.csv", index=False
    )
    return qc, folds, aggregates


def select_preprocessing(
    fold_metrics: pd.DataFrame,
    spectral_qc: pd.DataFrame,
    *,
    tie_tolerance: float,
    max_ordinal_mae_increase: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select by inner validation only with a fixed simplicity tie-break."""

    raw_fold = (
        fold_metrics.loc[fold_metrics["variant_id"].eq("A_raw")]
        .groupby("fold", as_index=True)
        .agg(
            raw_macro_f1=("validation_macro_f1", "mean"),
            raw_ordinal_mae=("validation_ordinal_mae", "mean"),
        )
    )
    raw_line = float(
        spectral_qc.loc[
            spectral_qc["variant_id"].eq("A_raw"),
            "median_line_to_1_45_ratio",
        ].iloc[0]
    )
    rows = []
    for rank, variant_id in enumerate(SIMPLICITY_ORDER):
        group = fold_metrics.loc[fold_metrics["variant_id"].eq(variant_id)]
        by_fold = group.groupby("fold").agg(
            macro_f1=("validation_macro_f1", "mean"),
            ordinal_mae=("validation_ordinal_mae", "mean"),
        )
        joined = by_fold.join(raw_fold)
        line = float(
            spectral_qc.loc[
                spectral_qc["variant_id"].eq(variant_id),
                "median_line_to_1_45_ratio",
            ].iloc[0]
        )
        rows.append(
            {
                "variant_id": variant_id,
                "preprocessing_name": group["preprocessing_name"].iloc[0],
                "selection_metric": "mean_inner_validation_record_macro_f1",
                "selection_value": float(group["validation_macro_f1"].mean()),
                "selection_std": float(
                    by_fold["macro_f1"].std(ddof=0)
                ),
                "folds_improved_vs_raw": int(
                    joined["macro_f1"].gt(joined["raw_macro_f1"]).sum()
                ),
                "mean_inner_ordinal_mae": float(
                    group["validation_ordinal_mae"].mean()
                ),
                "ordinal_mae_delta_vs_raw": float(
                    group["validation_ordinal_mae"].mean()
                    - fold_metrics.loc[
                        fold_metrics["variant_id"].eq("A_raw"),
                        "validation_ordinal_mae",
                    ].mean()
                ),
                "line_ratio": line,
                "line_ratio_vs_raw": line / raw_line,
                "finite": bool(
                    spectral_qc.loc[
                        spectral_qc["variant_id"].eq(variant_id),
                        "nonfinite_values",
                    ].iloc[0]
                    == 0
                ),
                "line_not_worse_than_raw": bool(line <= raw_line * (1 + 1e-9)),
                "ordinal_constraint_passed": bool(
                    group["validation_ordinal_mae"].mean()
                    - fold_metrics.loc[
                        fold_metrics["variant_id"].eq("A_raw"),
                        "validation_ordinal_mae",
                    ].mean()
                    <= max_ordinal_mae_increase
                ),
                "simplicity_rank": rank,
                "values_by_fold": json.dumps(
                    {
                        str(int(fold)): float(value)
                        for fold, value in by_fold["macro_f1"].items()
                    },
                    sort_keys=True,
                ),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.loc[
        table["finite"]
        & table["line_not_worse_than_raw"]
        & table["ordinal_constraint_passed"]
    ].copy()
    if eligible.empty:
        raise RuntimeError("No preprocessing candidate passed safety constraints")
    best = float(eligible["selection_value"].max())
    tied = eligible.loc[
        eligible["selection_value"].ge(best - tie_tolerance)
    ].sort_values("simplicity_rank", kind="stable")
    selected = tied.iloc[0]
    document = {
        "selected_preprocessing": selected["preprocessing_name"],
        "selected_variant_id": selected["variant_id"],
        "selection_metric": selected["selection_metric"],
        "selection_value": float(selected["selection_value"]),
        "selection_values_by_fold": json.loads(selected["values_by_fold"]),
        "selection_model_aggregation": "mean_of_two_preregistered_models",
        "outer_test_used_for_selection": False,
        "tie_tolerance": tie_tolerance,
        "tie_break": list(SIMPLICITY_ORDER),
        "selection_timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_table_hash": _stable_hash(
            table.drop(columns=["values_by_fold"]).to_dict(orient="records")
        ),
    }
    document["selection_hash"] = _stable_hash(
        {key: value for key, value in document.items() if key != "selection_timestamp"}
    )
    return table, document


def _deep_check(
    paths: AblationPaths,
    config: Mapping[str, Any],
    selected: Mapping[str, Any],
    variant: COGBCIWholeRecordPreprocessing,
    cache_dir: Path,
    mapping_path: Path,
    cache_inventory: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    deep = config["deep_check"]
    protocol_summary = json.loads(
        (paths.protocol_dir / "protocol_summary.json").read_text(encoding="utf-8")
    )
    cache_row = cache_inventory.loc[
        cache_inventory["variant_id"].eq(variant.variant_id)
    ].iloc[0]
    preregistration = {
        "result_status": RESULT_STATUS,
        "selected_preprocessing": selected["selected_preprocessing"],
        "selected_variant_id": selected["selected_variant_id"],
        "selection_hash": selected["selection_hash"],
        "selection_reason": (
            "highest mean inner-validation record macro F1 across five folds "
            "and both preregistered lightweight models, subject to safety constraints"
        ),
        "fold_id": 1,
        "model": "torch_eegnet",
        "seed": 42,
        "max_epochs": int(deep["max_epochs"]),
        "patience": int(deep["patience"]),
        "input_shape": [1, 14, 2560],
        "task_protocol_hash": protocol_summary["protocol_hash"],
        "outer_split_hash": protocol_summary["outer_split_hash"],
        "inner_split_hash": protocol_summary["inner_split_hash"],
        "preprocessing_hash": cache_row["preprocessing_hash"],
        "primary_metric": "outer_test_record_macro_f1",
        "decision_rule": deepcopy(config["decision_rule"]),
    }
    prereg_path = paths.output_dir / "deep_check_preregistration.json"
    if prereg_path.exists():
        existing = json.loads(prereg_path.read_text(encoding="utf-8"))
        if (
            existing.get("selected_variant_id")
            == preregistration["selected_variant_id"]
            and existing.get("selected_preprocessing")
            == preregistration["selected_preprocessing"]
            and existing.get("preprocessing_hash")
            == preregistration["preprocessing_hash"]
            and existing.get("task_protocol_hash")
            == preregistration["task_protocol_hash"]
            and existing.get("outer_split_hash")
            == preregistration["outer_split_hash"]
            and existing.get("inner_split_hash")
            == preregistration["inner_split_hash"]
        ):
            preregistration["selection_hash"] = existing["selection_hash"]
            if isinstance(selected, dict):
                selected["selection_hash"] = existing["selection_hash"]
        if existing != preregistration:
            raise RuntimeError("Existing deep preregistration is immutable")
    else:
        _write_json(prereg_path, preregistration)
    prereg_hash = _sha256(prereg_path)

    probability_columns = [
        f"mean_probability_class_{class_id}" for class_id in range(3)
    ]
    raw_output = paths.repository_root / _relative(
        deep["raw_reference_dir"], label="deep_check.raw_reference_dir"
    )
    raw_summary = json.loads(
        (raw_output / "run_summary.json").read_text(encoding="utf-8")
    )
    raw_fold = next(
        item for item in raw_summary["folds"] if int(item["fold_id"]) == 1
    )
    raw_predictions = pd.read_parquet(
        raw_output / "record_predictions.parquet"
    )
    raw_predictions = raw_predictions.loc[raw_predictions["fold_id"].eq(1)]
    raw_probabilities = raw_predictions[probability_columns].to_numpy(dtype=float)
    if variant.is_identity:
        fold = raw_fold
        probabilities = raw_probabilities
        deep_execution = "reused_existing_raw_fold1_without_retraining"
    else:
        raw_config = json.loads(
            (
                paths.repository_root
                / _relative(deep["base_config"], label="deep_check.base_config")
            ).read_text(encoding="utf-8")
        )
        output_dir = paths.output_dir / "deep_check_eegnet_fold1"
        resolved = deepcopy(raw_config)
        resolved["window_cache"] = str(
            cache_dir.relative_to(paths.repository_root)
        ).replace("\\", "/")
        resolved["sample_id_mapping_path"] = str(
            mapping_path.relative_to(paths.repository_root)
        ).replace("\\", "/")
        resolved["epochs"] = int(deep["max_epochs"])
        resolved["early_stopping"]["patience"] = int(deep["patience"])
        resolved["output_dir"] = str(
            output_dir.relative_to(paths.repository_root)
        ).replace("\\", "/")
        resolved["hashes"]["window_cache_config_hash"] = str(
            cache_row["config_hash"]
        )
        summary_path = output_dir / "run_summary.json"
        if summary_path.is_file():
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            run_summary = COGBCINBackBaselineRunner(
                resolved,
                repository_root=paths.repository_root,
                options=BaselineRunOptions(fold=1),
            ).run()
        fold = run_summary["folds"][0]
        record_predictions = pd.read_parquet(
            output_dir / "record_predictions.parquet"
        )
        probabilities = record_predictions[probability_columns].to_numpy(
            dtype=float
        )
        deep_execution = "trained_selected_nonraw_variant_once"
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)),
        axis=1,
    )
    raw_entropy = -np.sum(
        raw_probabilities
        * np.log(np.maximum(raw_probabilities, np.finfo(float).tiny)),
        axis=1,
    )
    metrics = {
        "preregistration_hash": prereg_hash,
        "selected_variant_id": variant.variant_id,
        "selected_preprocessing": variant.name,
        "execution": deep_execution,
        "fold_id": 1,
        "seed": 42,
        "epochs_trained": fold["epochs_trained"],
        "best_epoch": fold["best_epoch"],
        "best_validation_loss": fold["best_validation_loss"],
        "inner_validation_record_macro_f1": fold[
            "best_validation_record_macro_f1"
        ],
        "outer_test_record_metrics": fold["record_metrics"],
        "raw_outer_test_record_metrics": raw_fold["record_metrics"],
        "record_macro_f1_delta": float(
            fold["record_metrics"]["macro_f1"]
            - raw_fold["record_metrics"]["macro_f1"]
        ),
        "prediction_entropy_mean": float(entropy.mean()),
        "raw_prediction_entropy_mean": float(raw_entropy.mean()),
        "maximum_probability_mean": float(probabilities.max(axis=1).mean()),
        "raw_maximum_probability_mean": float(
            raw_probabilities.max(axis=1).mean()
        ),
        "confusion_matrix": fold["record_metrics"]["confusion_matrix"],
        "raw_confusion_matrix": raw_fold["record_metrics"]["confusion_matrix"],
        "training_time_seconds": fold["training_time_seconds"],
    }
    partial_output = paths.output_dir / "deep_check_eegnet_fold1"
    if (
        variant.is_identity
        and partial_output.is_dir()
        and not (partial_output / "run_summary.json").is_file()
    ):
        metrics["ignored_aborted_redundant_attempt"] = {
            "path": str(
                partial_output.relative_to(paths.repository_root)
            ).replace("\\", "/"),
            "reason": (
                "raw won inner selection; redundant raw retraining was stopped "
                "and no model result was used"
            ),
            "files": sorted(
                str(path.relative_to(partial_output)).replace("\\", "/")
                for path in partial_output.rglob("*")
                if path.is_file()
            ),
        }
    _write_json(paths.output_dir / "deep_check_metrics.json", metrics)
    return metrics, prereg_hash


def _decision(
    selection_table: pd.DataFrame,
    selected: Mapping[str, Any],
    deep_metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    row = selection_table.loc[
        selection_table["variant_id"].eq(selected["selected_variant_id"])
    ].iloc[0]
    raw = selection_table.loc[
        selection_table["variant_id"].eq("A_raw")
    ].iloc[0]
    rule = config["decision_rule"]
    conditions = {
        "lightweight_inner_better_than_raw": bool(
            row["selection_value"] > raw["selection_value"]
        ),
        "improved_at_least_three_inner_folds": bool(
            row["folds_improved_vs_raw"] >= 3
        ),
        "line_contamination_reduced_or_suppressed": bool(
            row["line_ratio_vs_raw"]
            <= float(rule["maximum_line_ratio_vs_raw"])
        ),
        "deep_record_macro_f1_not_worse_by_more_than_0_01": bool(
            deep_metrics["record_macro_f1_delta"]
            >= -float(rule["maximum_deep_macro_f1_degradation"])
        ),
        "cache_split_provenance_valid": bool(
            row["finite"]
            and row["line_not_worse_than_raw"]
            and provenance["provenance_status"]
            in {"confirmed", "partially_confirmed"}
        ),
    }
    all_pass = all(conditions.values())
    if all_pass and deep_metrics["record_macro_f1_delta"] >= float(
        rule["strong_proceed_deep_macro_f1_improvement"]
    ):
        status = "strong_proceed"
    elif all_pass:
        status = "proceed"
    elif not conditions["cache_split_provenance_valid"]:
        status = "inconclusive"
    else:
        status = "do_not_proceed"
    return {
        "result_status": RESULT_STATUS,
        "recommendation": status,
        "conditions": conditions,
        "thresholds": deepcopy(rule),
        "interpretation": (
            "This threshold governs the next computational stage and is not "
            "a claim of statistical significance."
        ),
    }


def _report(
    summary: Mapping[str, Any],
    qc: pd.DataFrame,
    aggregate: pd.DataFrame,
    selection_table: pd.DataFrame,
) -> str:
    unit = summary["source_provenance"]
    selected = summary["selection"]
    deep = summary["deep_check"]
    lines = [
        "# COG-BCI N-Back: аудит единиц и preprocessing-ablation",
        "",
        f"- Branch / HEAD: `{summary['branch']}` / `{summary['head']}`",
        f"- Result status: `{RESULT_STATUS}`",
        "- Исходные `.set/.fdt`, raw cache, task protocol и split manifests не изменены.",
        f"- Source physical unit: `{unit['provenance_status']}`; явная единица не объявлена.",
        f"- MNE convention: `{unit['mne_output_unit']}`, factor `{unit['mne_calibration_factor']}`.",
        "- Hardware/software/notch filter history: `unknown`.",
        "- `EEG.ref=common` подтверждён, численный reference не указан.",
        "",
        "## Контракт фильтрации",
        "",
        "Все операции применялись к полной непрерывной физической записи до нарезки:",
        "`demean → Butterworth IIR 1–45 Hz → IIR notch 50 Hz`; только включённые",
        "операции выполнялись. Фильтры zero-phase forward/backward, explicit odd",
        "padding; точные коэффициенты и padlen сохранены в runtime specs.",
        "Notch после low-pass 45 Hz в G/H является потенциально избыточным и включён",
        "только как эмпирическая проверка.",
        "",
        "## Spectral QC",
        "",
        qc.to_markdown(index=False),
        "",
        "## Lightweight baselines",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Inner-only selection",
        "",
        selection_table[
            [
                "variant_id",
                "preprocessing_name",
                "selection_value",
                "selection_std",
                "folds_improved_vs_raw",
                "mean_inner_ordinal_mae",
                "line_ratio_vs_raw",
            ]
        ].to_markdown(index=False),
        "",
        f"Выбран `{selected['selected_variant_id']}` / "
        f"`{selected['selected_preprocessing']}` исключительно по inner-validation.",
        f"Selection hash: `{selected['selection_hash']}`.",
        "",
        "## Preregistered EEGNet fold 1",
        "",
        f"- preregistration hash: `{summary['preregistration_hash']}`",
        f"- epochs: {deep['epochs_trained']}; best epoch: {deep['best_epoch']}",
        f"- inner record macro F1: {deep['inner_validation_record_macro_f1']:.6f}",
        f"- outer record balanced accuracy: "
        f"{deep['outer_test_record_metrics']['balanced_accuracy']:.6f}",
        f"- outer record macro F1: "
        f"{deep['outer_test_record_metrics']['macro_f1']:.6f}",
        f"- raw outer record macro F1: "
        f"{deep['raw_outer_test_record_metrics']['macro_f1']:.6f}",
        f"- delta: {deep['record_macro_f1_delta']:+.6f}",
        "",
        "## Решение",
        "",
        f"`{summary['decision']['recommendation']}`.",
        "",
        "Ограничения: физическая единица и acquisition filter history не разрешены;",
        "lightweight outer-test результаты диагностические; deep check ограничен",
        "одним заранее выбранным fold и seed 42. Полный deep GroupKFold и multiseed",
        "не запускались. Поскольку raw выиграл inner-selection, новый raw model result",
        "не использовался; сохранённый fold 1 был переиспользован без переобучения.",
    ]
    return "\n".join(lines) + "\n"


def run_cog_bci_preprocessing_ablation(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    paths = _resolve_paths(config, repository_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes_before = _input_hashes(paths)
    target, _, inner = _load_protocol_frames(paths)
    variants = build_preprocessing_variants(config["candidate_variants"])
    specs = {
        "result_status": RESULT_STATUS,
        "variants": [variant.to_dict() for variant in variants],
    }
    _write_json(paths.output_dir / "preprocessing_specs.json", specs)
    provenance = _source_provenance(paths, config)
    inventory, cache_paths, mapping_paths = _materialize_caches(
        paths, config, variants, target
    )
    inventory.to_csv(paths.output_dir / "cache_inventory.csv", index=False)
    spectral_path = paths.output_dir / "spectral_qc_by_variant.csv"
    fold_path = paths.output_dir / "lightweight_fold_metrics.csv"
    aggregate_path = paths.output_dir / "lightweight_aggregate_metrics.csv"
    if all(path.is_file() for path in (spectral_path, fold_path, aggregate_path)):
        qc = pd.read_csv(spectral_path)
        fold_metrics = pd.read_csv(fold_path)
        aggregate = pd.read_csv(aggregate_path)
    else:
        qc, fold_metrics, aggregate = _spectral_and_lightweight(
            paths,
            config,
            variants,
            target,
            inner,
            cache_paths,
            mapping_paths,
        )
    selection_table, selected = select_preprocessing(
        fold_metrics,
        qc,
        tie_tolerance=float(config["selection_rule"]["tie_tolerance"]),
        max_ordinal_mae_increase=float(
            config["selection_rule"]["max_ordinal_mae_increase"]
        ),
    )
    selection_table.to_csv(
        paths.output_dir / "inner_selection_table.csv", index=False
    )
    _write_json(paths.output_dir / "selected_preprocessing.json", selected)
    selected_variant = next(
        item
        for item in variants
        if item.variant_id == selected["selected_variant_id"]
    )
    deep_metrics, prereg_hash = _deep_check(
        paths,
        config,
        selected,
        selected_variant,
        cache_paths[selected_variant.variant_id],
        mapping_paths[selected_variant.variant_id],
        inventory,
    )
    _write_json(paths.output_dir / "selected_preprocessing.json", selected)
    decision = _decision(
        selection_table, selected, deep_metrics, provenance, config
    )
    _write_json(paths.output_dir / "decision.json", decision)
    input_hashes_after = _input_hashes(paths)
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("Immutable source cache/protocol inputs changed")
    import subprocess

    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "result_status": RESULT_STATUS,
        "branch": branch,
        "head": head,
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "source_provenance": provenance,
        "dataset": {
            "records": EXPECTED_RECORDS,
            "subjects": 29,
            "sessions": 3,
            "accepted_windows": EXPECTED_WINDOWS,
            "shape": list(EXPECTED_SHAPE),
            "sampling_rate_hz": 500.0,
        },
        "cache_inventory": inventory.to_dict(orient="records"),
        "selection": selected,
        "preregistration_hash": prereg_hash,
        "deep_check": deep_metrics,
        "decision": decision,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(paths.output_dir / "ablation_summary.json", summary)
    report = _report(summary, qc, aggregate, selection_table)
    (paths.output_dir / "ablation_report.md").write_text(
        report, encoding="utf-8"
    )
    paths.tracked_report.parent.mkdir(parents=True, exist_ok=True)
    paths.tracked_report.write_text(report, encoding="utf-8")
    error_rows = []
    if "ignored_aborted_redundant_attempt" in deep_metrics:
        error_rows.append(
            {
                "stage": "deep_check",
                "variant_id": selected_variant.variant_id,
                "error_type": "RedundantRawRunStopped",
                "message": deep_metrics[
                    "ignored_aborted_redundant_attempt"
                ]["reason"],
            }
        )
    pd.DataFrame(
        error_rows,
        columns=["stage", "variant_id", "error_type", "message"],
    ).to_csv(paths.output_dir / "errors.csv", index=False)
    return summary
