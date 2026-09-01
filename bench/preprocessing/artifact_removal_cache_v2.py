"""Persistent target-independent, record-local artifact-removal cache."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.preprocessing.fold_artifact_transform import stable_hash
from cogstate.preprocessing.artifact_removal import ArtifactICA, IcaConfig
from cogstate.preprocessing.full_faster import (
    FasterConfig,
    apply_faster_online as apply_full_metric_faster_online,
    detect_bad_channels,
    run_faster,
)


ARTIFACT_VARIANTS_V2 = (
    "raw",
    "ica_record_local",
    "faster_online",
    "faster_full_record_local",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preprocessing_implementation_hashes() -> dict[str, str]:
    return {
        "cache_layer": _file_sha256(Path(__file__)),
        "full_faster": _file_sha256(Path(run_faster.__code__.co_filename)),
    }


class ArtifactCacheError(RuntimeError):
    """Raised when a cache shard cannot be built without a raw fallback."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def _record_key(record_group_id: str) -> str:
    digest = hashlib.sha256(str(record_group_id).encode("utf-8")).hexdigest()[:16]
    return f"record-{digest}"


def source_contract_hash(
    manifest: pd.DataFrame,
    *,
    raw_preprocessing: Mapping[str, Any],
    input_shape: Sequence[int],
) -> str:
    """Hash the immutable raw signal universe and its preprocessing provenance."""
    required = {
        "sample_id",
        "subject_id",
        "record_id",
        "record_group_id",
        "source",
        "preprocessing_hash",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Signal-universe manifest is missing columns: {missing}")
    ordered = manifest.sort_values("sample_id", kind="mergesort")
    payload = {
        "sample_ids": ordered["sample_id"].astype(str).tolist(),
        "record_assignments": ordered[
            ["sample_id", "subject_id", "record_id", "record_group_id", "source"]
        ].astype(str).to_dict("records"),
        "raw_preprocessing": dict(raw_preprocessing),
        "raw_preprocessing_hashes": sorted(
            ordered["preprocessing_hash"].dropna().astype(str).unique().tolist()
        ),
        "input_shape": [int(value) for value in input_shape],
    }
    return stable_hash(payload)


def preprocessing_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only scientific preprocessing parameters used by cache builders."""
    return {
        "implementation_sha256": preprocessing_implementation_hashes(),
        "variants": list(config["variants"]),
        "sample_rate_hz": float(config["sample_rate_hz"]),
        "z_threshold": float(config["z_threshold"]),
        "interpolation_method": str(config["interpolation_method"]),
        "average_reference": bool(config["average_reference"]),
        "ica": {
            "n_components": config["ica"].get("n_components"),
            "max_iter": int(config["ica"]["max_iter"]),
            "random_state": int(config["ica"]["random_state"]),
        },
        "full_faster": {
            "max_iter": int(config["full_faster"]["max_iter"]),
            "run_ica": bool(config["full_faster"]["run_ica"]),
            "ica_n_components": config["full_faster"].get("ica_n_components"),
            "ica_max_iter": int(config["full_faster"]["ica_max_iter"]),
            "ica_random_state": int(config["full_faster"]["ica_random_state"]),
        },
    }


def preprocessing_cache_hash(
    config: Mapping[str, Any], source_hash: str
) -> str:
    return stable_hash(
        {
            "schema_version": 2,
            "source_preprocessing_contract_hash": str(source_hash),
            "preprocessing": preprocessing_config_payload(config),
        }
    )


def variant_config_hash(
    variant: str,
    config: Mapping[str, Any],
    source_hash: str,
) -> str:
    if variant not in ARTIFACT_VARIANTS_V2:
        raise ValueError(f"Unknown v2 artifact variant: {variant}")
    return stable_hash(
        {
            "schema_version": 2,
            "variant": variant,
            "source_preprocessing_contract_hash": str(source_hash),
            "preprocessing": preprocessing_config_payload(config),
        }
    )


def _base_faster_config(
    config: Mapping[str, Any], *, run_ica: bool
) -> FasterConfig:
    return FasterConfig(
        z_threshold=float(config["z_threshold"]),
        max_iter=int(config["full_faster"]["max_iter"]),
        interpolate_bad_channels=True,
        interpolate_bad_channel_epoch=True,
        interpolation_method=str(config["interpolation_method"]),
        run_ica=bool(run_ica),
        ica_n_components=config["full_faster"].get("ica_n_components"),
        ica_max_iter=int(config["full_faster"]["ica_max_iter"]),
        ica_random_state=int(config["full_faster"]["ica_random_state"]),
        average_reference=bool(config["average_reference"]),
    )


def _record_tensor(base_view: Any, positions: np.ndarray) -> np.ndarray:
    values = np.stack([np.asarray(base_view[int(index)]) for index in positions])
    expected = (len(positions), 1, 14, 2560)
    if values.shape != expected:
        raise ValueError(f"Record tensor shape changed: {values.shape} != {expected}")
    if not np.isfinite(values).all():
        raise ValueError("Record tensor contains NaN or infinite values")
    return np.ascontiguousarray(values, dtype=np.float32)


def _report_lists_by_metric(values: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    return {
        str(name): [int(value) for value in indices]
        for name, indices in values.items()
    }


def _process_record(
    tensor: np.ndarray,
    variant: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transform one logical record without mixing it with any other record."""
    epochs = np.asarray(tensor[:, 0].transpose(0, 2, 1), dtype=np.float32)
    n_epochs = len(epochs)
    all_indices = np.arange(n_epochs, dtype=np.int64)
    started = time.perf_counter()
    diagnostics: dict[str, Any] = {
        "input_window_count": int(n_epochs),
        "epoch_zscore_structurally_limited": bool(n_epochs < 11),
        "average_reference": False,
    }
    if variant == "raw":
        cleaned = tensor.copy()
        retained = all_indices
        diagnostics.update({"implementation": "canonical_raw_reference"})
    elif variant == "faster_online":
        faster = _base_faster_config(config, run_ica=False)
        cleaned_epochs: list[np.ndarray] = []
        bad_channels_by_epoch: dict[str, list[int]] = {}
        for index, epoch in enumerate(epochs):
            bad = detect_bad_channels(epoch, faster)
            if bad:
                bad_channels_by_epoch[str(index)] = [int(value) for value in bad]
            cleaned_epochs.append(apply_full_metric_faster_online(epoch, faster))
        cleaned = np.stack(cleaned_epochs).transpose(0, 2, 1)[:, None]
        retained = all_indices
        diagnostics.update(
            {
                "implementation": "apply_faster_per_window",
                "ica_fitted": False,
                "bad_channels_by_epoch": bad_channels_by_epoch,
                "windows_with_bad_channels": int(len(bad_channels_by_epoch)),
            }
        )
    elif variant == "ica_record_local":
        faster = _base_faster_config(config, run_ica=False)
        ica_config = IcaConfig(
            n_components=config["ica"].get("n_components"),
            max_iter=int(config["ica"]["max_iter"]),
            random_state=int(config["ica"]["random_state"]),
            faster_config=faster,
            component_metric_profile="full_faster",
        )
        flattened = epochs.reshape(-1, epochs.shape[2])
        ica = ArtifactICA(ica_config).fit(
            flattened, sample_rate=float(config["sample_rate_hz"])
        )
        transformed = [ica.transform(epoch) for epoch in epochs]
        cleaned = np.stack(transformed).transpose(0, 2, 1)[:, None]
        retained = all_indices
        fitted = ica._ica
        if fitted is None:
            raise RuntimeError("Record-local ArtifactICA did not retain fitted state")
        n_iter = int(getattr(fitted, "n_iter_", 0))
        diagnostics.update(
            {
                "implementation": "artifact_ica_record_local",
                "component_metric_profile": "full_faster",
                "ica_fitted": True,
                "ica_converged": bool(ica.converged),
                "ica_n_iter": n_iter,
                "ica_input_rank": ica.input_rank,
                "ica_n_components": ica.n_components,
                "bad_components": [int(value) for value in ica.artifact_components],
                "bad_component_count": int(ica.n_artifact_components),
            }
        )
    elif variant == "faster_full_record_local":
        faster = _base_faster_config(config, run_ica=True)
        cleaned_epochs, report = run_faster(
            epochs,
            faster,
            sample_rate=float(config["sample_rate_hz"]),
        )
        retained = np.asarray(report.kept_epoch_indices, dtype=np.int64)
        cleaned = cleaned_epochs.transpose(0, 2, 1)[:, None]
        diagnostics.update(
            {
                "implementation": "run_faster_full_record_local",
                "bad_channels": [int(value) for value in report.bad_channels],
                "bad_epochs": [int(value) for value in report.bad_epochs],
                "bad_components": [int(value) for value in report.bad_components],
                "bad_channel_epoch_pairs": [
                    [int(epoch), int(channel)]
                    for epoch, channel in report.bad_channel_epoch_pairs_original
                ],
                "channel_bads_by_metric": _report_lists_by_metric(
                    report.channel_bads_by_metric
                ),
                "epoch_bads_by_metric": _report_lists_by_metric(
                    report.epoch_bads_by_metric
                ),
                "component_bads_by_metric": _report_lists_by_metric(
                    report.component_bads_by_metric
                ),
                "ica_fitted": bool(report.ica_fitted),
                "ica_converged": report.ica_converged,
                "ica_input_rank": getattr(report, "ica_input_rank", None),
                "ica_n_components": getattr(report, "ica_n_components", None),
                "interpolation_method": report.interpolation_method,
            }
        )
    else:
        raise ValueError(f"Unknown artifact variant: {variant}")
    cleaned = np.ascontiguousarray(cleaned, dtype=np.float32)
    if cleaned.shape[1:] != (1, 14, 2560) or not np.isfinite(cleaned).all():
        raise ValueError(f"Invalid cleaned record tensor: {cleaned.shape}")
    rejected = np.setdiff1d(all_indices, retained, assume_unique=True)
    diagnostics.update(
        {
            "retained_window_count": int(len(retained)),
            "rejected_window_count": int(len(rejected)),
            "coverage": float(len(retained) / n_epochs),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    return cleaned, retained, diagnostics


def _valid_record_cache(
    metadata_path: Path,
    *,
    expected_hash: str,
    expected_sample_ids: Sequence[str],
) -> dict[str, Any] | None:
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if metadata.get("status") != "complete":
        return None
    if metadata.get("preprocessing_hash") != expected_hash:
        return None
    if metadata.get("input_sample_ids") != list(expected_sample_ids):
        return None
    shard_value = metadata.get("shard")
    if shard_value is not None:
        shard = metadata_path.parent / str(shard_value)
        if not shard.is_file():
            return None
        try:
            values = np.load(shard, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            return None
        if tuple(values.shape) != tuple(metadata["output_shape"]):
            return None
    return metadata


def _rows_from_record_metadata(
    record: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    variant: str,
    variant_hash: str,
) -> list[dict[str, Any]]:
    retained_indices = [int(value) for value in metadata["retained_epoch_indices"]]
    retained_offsets = {index: offset for offset, index in enumerate(retained_indices)}
    rows: list[dict[str, Any]] = []
    for epoch_index, (_, source_row) in enumerate(record.iterrows()):
        retained = epoch_index in retained_offsets
        if variant == "raw":
            shard = str(source_row["cache_file"])
            offset = int(source_row["cache_offset"])
            storage_root = "repo"
        else:
            shard = str(metadata["shard"])
            offset = retained_offsets.get(epoch_index)
            storage_root = "variant"
        rows.append(
            {
                "sample_id": source_row["sample_id"],
                "subject_id": str(source_row["subject_id"]),
                "record_id": str(source_row["record_id"]),
                "record_group_id": str(source_row["record_group_id"]),
                "source": str(source_row["source"]),
                "outer_fold": int(source_row["outer_fold"]),
                "variant": variant,
                "retained": bool(retained),
                "shard": shard,
                "offset": offset,
                "storage_root": storage_root,
                "preprocessing_hash": variant_hash,
                "t_start": float(source_row["t_start"]),
                "t_end": float(source_row["t_end"]),
            }
        )
    return rows


def build_variant_cache(
    base_view: Any,
    *,
    variant: str,
    variant_root: Path,
    config: Mapping[str, Any],
    source_hash: str,
    resume: bool,
) -> dict[str, Any]:
    """Build one deterministic cache shard per logical record."""
    manifest = base_view.manifest.reset_index(drop=True).copy()
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Signal universe contains duplicate sample_id")
    group_contract = manifest.groupby("record_group_id", sort=True).agg(
        subjects=("subject_id", "nunique"),
        records=("record_id", "nunique"),
        sources=("source", "nunique"),
    )
    invalid = group_contract.loc[
        (group_contract["subjects"] != 1)
        | (group_contract["records"] != 1)
        | (group_contract["sources"] != 1)
    ]
    if not invalid.empty:
        raise ValueError(
            "record_group_id must map to one subject/source/selected record: "
            f"{invalid.index.astype(str).tolist()}"
        )
    variant_hash = variant_config_hash(variant, config, source_hash)
    variant_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    record_reports: list[dict[str, Any]] = []
    resumed_records = 0
    started = time.perf_counter()
    for record_group_id, record in manifest.groupby(
        "record_group_id", sort=True, observed=True
    ):
        record = record.sort_values(["t_start", "sample_id"], kind="mergesort")
        positions = record.index.to_numpy(dtype=np.int64)
        sample_ids = record["sample_id"].astype(str).tolist()
        key = _record_key(str(record_group_id))
        metadata_path = variant_root / f"{key}.json"
        existing = (
            _valid_record_cache(
                metadata_path,
                expected_hash=variant_hash,
                expected_sample_ids=sample_ids,
            )
            if resume
            else None
        )
        if existing is not None:
            metadata = existing
            resumed_records += 1
        else:
            try:
                if variant == "raw":
                    retained = np.arange(len(record), dtype=np.int64)
                    diagnostics = {
                        "implementation": "canonical_raw_reference",
                        "input_window_count": int(len(record)),
                        "retained_window_count": int(len(record)),
                        "rejected_window_count": 0,
                        "coverage": 1.0,
                        "elapsed_seconds": 0.0,
                        "epoch_zscore_structurally_limited": False,
                        "average_reference": False,
                    }
                    shard_name = None
                    output_shape = [int(len(record)), 1, 14, 2560]
                else:
                    tensor = _record_tensor(base_view, positions)
                    cleaned, retained, diagnostics = _process_record(
                        tensor, variant, config
                    )
                    shard_name = f"{key}.npy"
                    _write_npy(variant_root / shard_name, cleaned)
                    output_shape = [int(value) for value in cleaned.shape]
                rejected = np.setdiff1d(
                    np.arange(len(record), dtype=np.int64), retained, assume_unique=True
                )
                metadata = {
                    "schema_version": 2,
                    "status": "complete",
                    "variant": variant,
                    "record_group_id": str(record_group_id),
                    "subject_id": str(record.iloc[0]["subject_id"]),
                    "record_id": str(record.iloc[0]["record_id"]),
                    "source": str(record.iloc[0]["source"]),
                    "preprocessing_hash": variant_hash,
                    "source_preprocessing_contract_hash": source_hash,
                    "input_sample_ids": sample_ids,
                    "retained_epoch_indices": retained.astype(int).tolist(),
                    "rejected_epoch_indices": rejected.astype(int).tolist(),
                    "retained_sample_ids": record.iloc[retained]["sample_id"].astype(str).tolist(),
                    "rejected_sample_ids": record.iloc[rejected]["sample_id"].astype(str).tolist(),
                    "shard": shard_name,
                    "output_shape": output_shape,
                    "diagnostics": diagnostics,
                }
                _write_json(metadata_path, metadata)
            except Exception as error:
                failure = {
                    "schema_version": 2,
                    "status": "error",
                    "variant": variant,
                    "record_group_id": str(record_group_id),
                    "preprocessing_hash": variant_hash,
                    "source_preprocessing_contract_hash": source_hash,
                    "input_sample_ids": sample_ids,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                _write_json(metadata_path, failure)
                _write_json(
                    variant_root / "cache_manifest.json",
                    {
                        "schema_version": 2,
                        "status": "incomplete",
                        "variant": variant,
                        "preprocessing_hash": variant_hash,
                        "failed_record_group_id": str(record_group_id),
                        "error": failure,
                    },
                )
                raise ArtifactCacheError(
                    f"{variant} failed for record_group_id={record_group_id}: {error}"
                ) from error
        rows.extend(
            _rows_from_record_metadata(
                record, metadata, variant=variant, variant_hash=variant_hash
            )
        )
        record_reports.append(
            {
                "record_group_id": str(record_group_id),
                "subject_id": str(record.iloc[0]["subject_id"]),
                "record_id": str(record.iloc[0]["record_id"]),
                "source": str(record.iloc[0]["source"]),
                **dict(metadata["diagnostics"]),
            }
        )
    cache_index = pd.DataFrame(rows).sort_values("sample_id", kind="mergesort")
    if len(cache_index) != len(manifest):
        raise RuntimeError("Variant cache manifest lost signal-universe rows")
    _write_parquet(variant_root / "window_manifest.parquet", cache_index)
    reports = pd.DataFrame(record_reports)
    _write_parquet(variant_root / "record_diagnostics.parquet", reports)
    retained_count = int(cache_index["retained"].sum())
    rejected = cache_index.loc[~cache_index["retained"]].copy()
    _write_parquet(variant_root / "rejected_windows.parquet", rejected)
    def total_list(name: str) -> int:
        return int(sum(len(row.get(name, [])) for row in record_reports))

    def aggregate_metric_lists(name: str) -> dict[str, int]:
        names = sorted(
            {
                metric
                for row in record_reports
                for metric in row.get(name, {})
            }
        )
        return {
            metric: int(
                sum(len(row.get(name, {}).get(metric, [])) for row in record_reports)
            )
            for metric in names
        }

    fitted_reports = [row for row in record_reports if row.get("ica_fitted")]
    summary = {
        "schema_version": 2,
        "status": "complete",
        "variant": variant,
        "preprocessing_hash": variant_hash,
        "source_preprocessing_contract_hash": source_hash,
        "input_window_count": int(len(cache_index)),
        "retained_window_count": retained_count,
        "rejected_window_count": int(len(cache_index) - retained_count),
        "coverage": float(retained_count / len(cache_index)),
        "record_group_count": int(cache_index["record_group_id"].nunique()),
        "resumed_record_count": int(resumed_records),
        "elapsed_seconds": float(time.perf_counter() - started),
        "manifest_hash": stable_hash(
            cache_index[
                ["sample_id", "record_group_id", "retained", "shard", "offset"]
            ].astype(str).to_dict("records")
        ),
        "record_diagnostics": reports.to_dict("records"),
        "aggregate_diagnostics": {
            "bad_channels": total_list("bad_channels"),
            "bad_epochs": total_list("bad_epochs"),
            "bad_components": total_list("bad_components"),
            "bad_channel_epoch_pairs": total_list("bad_channel_epoch_pairs"),
            "ica_fitted_records": int(len(fitted_reports)),
            "ica_converged_records": int(
                sum(row.get("ica_converged") is True for row in fitted_reports)
            ),
            "ica_nonconverged_records": int(
                sum(row.get("ica_converged") is False for row in fitted_reports)
            ),
            "channel_bads_by_metric": aggregate_metric_lists(
                "channel_bads_by_metric"
            ),
            "epoch_bads_by_metric": aggregate_metric_lists("epoch_bads_by_metric"),
            "component_bads_by_metric": aggregate_metric_lists(
                "component_bads_by_metric"
            ),
        },
    }
    _write_json(variant_root / "cache_manifest.json", summary)
    return summary


def build_preprocessing_cache(
    base_view: Any,
    *,
    cache_root: Path,
    config: Mapping[str, Any],
    source_hash: str,
    resume: bool = True,
) -> dict[str, Any]:
    summaries = {}
    for variant in config["variants"]:
        summaries[str(variant)] = build_variant_cache(
            base_view,
            variant=str(variant),
            variant_root=cache_root / str(variant),
            config=config,
            source_hash=source_hash,
            resume=resume,
        )
    summary = {
        "schema_version": 2,
        "status": "complete",
        "cache_hash": preprocessing_cache_hash(config, source_hash),
        "source_preprocessing_contract_hash": source_hash,
        "variants": summaries,
        "target_independent": True,
        "target_columns_used_during_preprocessing": [],
    }
    _write_json(cache_root / "preprocessing_cache_manifest.json", summary)
    return summary


class CachedArtifactWindowView:
    """NumPy-shaped lazy view over v2 artifact-removal cache shards."""

    is_lazy_raw_eeg = True

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        repo_root: Path,
        variant_root: Path,
        channel_mean: np.ndarray | None = None,
        channel_scale: np.ndarray | None = None,
    ) -> None:
        frame = manifest.loc[manifest["retained"].astype(bool)].reset_index(drop=True)
        if frame.empty:
            raise ValueError("CachedArtifactWindowView cannot be empty")
        self.manifest = frame
        self.repo_root = Path(repo_root)
        self.variant_root = Path(variant_root)
        self.shape = (len(frame), 1, 14, 2560)
        self.ndim = 4
        self.dtype = np.dtype(np.float32)
        self.channel_mean = (
            None if channel_mean is None else np.asarray(channel_mean, dtype=np.float32)
        )
        self.channel_scale = (
            None if channel_scale is None else np.asarray(channel_scale, dtype=np.float32)
        )
        if (self.channel_mean is None) != (self.channel_scale is None):
            raise ValueError("channel_mean and channel_scale must be set together")
        self._mapped: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def _path(self, row: pd.Series) -> Path:
        shard = Path(str(row["shard"]))
        if shard.is_absolute():
            return shard
        return (self.repo_root if row["storage_root"] == "repo" else self.variant_root) / shard

    def _read_scalar(self, index: int) -> np.ndarray:
        row = self.manifest.iloc[int(index)]
        path = self._path(row)
        key = str(path)
        if key not in self._mapped:
            self._mapped[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        window = np.asarray(self._mapped[key][int(row["offset"])], dtype=np.float32)
        if window.shape == (14, 2560):
            window = window[None]
        if window.shape != (1, 14, 2560) or not np.isfinite(window).all():
            raise ValueError(f"Invalid cached artifact window: {window.shape}")
        if self.channel_mean is not None and self.channel_scale is not None:
            window = (
                window - self.channel_mean[None, :, None]
            ) / self.channel_scale[None, :, None]
        return np.ascontiguousarray(window, dtype=np.float32)

    def __getitem__(self, index: Any) -> Any:
        if np.isscalar(index):
            scalar = int(index)
            if scalar < 0:
                scalar += len(self)
            if scalar < 0 or scalar >= len(self):
                raise IndexError(scalar)
            return self._read_scalar(scalar)
        positions = np.arange(len(self))[index]
        return CachedArtifactWindowView(
            self.manifest.iloc[np.asarray(positions, dtype=np.int64)],
            repo_root=self.repo_root,
            variant_root=self.variant_root,
            channel_mean=self.channel_mean,
            channel_scale=self.channel_scale,
        )

    def with_channel_normalization(
        self, mean: np.ndarray, scale: np.ndarray
    ) -> "CachedArtifactWindowView":
        return CachedArtifactWindowView(
            self.manifest,
            repo_root=self.repo_root,
            variant_root=self.variant_root,
            channel_mean=mean,
            channel_scale=scale,
        )

    def compute_channel_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros(14, dtype=np.float64)
        squares = np.zeros(14, dtype=np.float64)
        count = 0
        for index in range(len(self)):
            window = self._read_scalar(index)[0].astype(np.float64, copy=False)
            total += window.sum(axis=1)
            squares += np.square(window).sum(axis=1)
            count += window.shape[1]
        mean = total / count
        scale = np.sqrt(np.maximum(squares / count - np.square(mean), 0.0))
        scale[scale < 1e-8] = 1.0
        return mean.astype(np.float32), scale.astype(np.float32)


def load_cached_view(
    cache_root: Path,
    variant: str,
    *,
    repo_root: Path,
) -> CachedArtifactWindowView:
    variant_root = cache_root / variant
    manifest_path = variant_root / "window_manifest.parquet"
    summary_path = variant_root / "cache_manifest.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Preprocessing cache is incomplete for {variant}: {variant_root}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ArtifactCacheError(f"Preprocessing cache is not complete: {variant_root}")
    return CachedArtifactWindowView(
        pd.read_parquet(manifest_path),
        repo_root=repo_root,
        variant_root=variant_root,
    )
