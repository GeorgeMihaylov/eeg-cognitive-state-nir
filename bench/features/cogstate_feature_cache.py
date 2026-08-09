"""Target-free, resumable materialization of canonical ``cogstate.features``.

The cache stores one float32 row per canonical raw EEG ``sample_id``.  Target
columns are intentionally absent so the same representation can be joined to
any approved target cohort by a dataset loader.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.datasets.raw_eeg_window_dataset import RawEEGWindowArrayView
from cogstate.features import FEATURE_SCHEMA_VERSION, FeaturePipeline, FeaturePipelineConfig


FEATURE_CACHE_SCHEMA_VERSION = "cogstate-feature-cache-v1"
FEATURE_MATRIX_NAME = "features.npy"
FEATURE_INDEX_NAME = "feature_index.parquet"
FEATURE_NAMES_NAME = "feature_names.json"
FEATURE_MANIFEST_NAME = "feature_materialization_manifest.json"
FEATURE_SUMMARY_JSON_NAME = "feature_materialization_summary.json"
FEATURE_SUMMARY_CSV_NAME = "feature_materialization_summary.csv"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample_id_universe_hash(sample_ids: Iterable[Any]) -> str:
    """Hash the ordered sample universe without depending on NumPy dtype."""
    digest = hashlib.sha256()
    for value in sample_ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_feature_profile(path: str | Path) -> tuple[dict[str, Any], FeaturePipeline]:
    profile_path = Path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "Feature profile schema mismatch: "
            f"{payload.get('schema_version')!r} != {FEATURE_SCHEMA_VERSION!r}"
        )
    if payload.get("selection") != "none":
        raise ValueError("Preliminary materialization requires selection='none'")
    pipeline = FeaturePipeline(FeaturePipelineConfig.from_mapping(payload))
    return payload, pipeline


def build_canonical_feature_index(
    manifest_path: str | Path,
    logical_recording_map_path: str | Path,
) -> pd.DataFrame:
    """Return the canonical deduplicated, QC-accepted raw-window universe."""
    manifest = pd.read_parquet(manifest_path)
    logical = pd.read_parquet(logical_recording_map_path)
    required_manifest = {
        "sample_id", "record_id", "record_group_id", "subject_id", "status",
        "cache_file", "cache_offset", "n_channels", "n_samples_expected",
        "preprocessing_hash",
    }
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"Raw manifest is missing columns: {missing}")
    required_logical = {"record_group_id", "selected_record_id"}
    missing_logical = sorted(required_logical - set(logical.columns))
    if missing_logical:
        raise ValueError(f"Logical-record map is missing columns: {missing_logical}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Raw manifest contains duplicate sample_id")
    if logical["record_group_id"].astype(str).duplicated().any():
        raise ValueError("Logical-record map contains duplicate record_group_id")
    selected = dict(
        zip(
            logical["record_group_id"].astype(str),
            logical["selected_record_id"].astype(str),
        )
    )
    selected_mask = manifest["record_id"].astype(str).eq(
        manifest["record_group_id"].astype(str).map(selected)
    )
    index = manifest.loc[selected_mask & manifest["status"].eq("ok")].copy()
    index = index.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if index.empty:
        raise ValueError("Canonical deduplicated status=ok universe is empty")
    if index["sample_id"].duplicated().any():
        raise ValueError("Canonical feature index contains duplicate sample_id")
    shape_pairs = index[["n_channels", "n_samples_expected"]].drop_duplicates()
    if len(shape_pairs) != 1:
        raise ValueError(
            "Canonical raw universe has multiple window shapes: "
            f"{shape_pairs.to_dict('records')}"
        )
    preprocessing_hashes = sorted(
        index["preprocessing_hash"].dropna().astype(str).unique()
    )
    if len(preprocessing_hashes) != 1:
        raise ValueError(
            "Canonical raw universe has ambiguous preprocessing identity: "
            f"{preprocessing_hashes}"
        )
    return index


def feature_cache_identity(
    index: pd.DataFrame,
    pipeline: FeaturePipeline,
    *,
    dtype: str = "float32",
) -> dict[str, Any]:
    channel_names = list(pipeline.config.channel_names or ())
    if not channel_names:
        raise ValueError("Feature profile must define channel_names")
    raw_shapes = index[["n_channels", "n_samples_expected"]].drop_duplicates()
    if len(raw_shapes) != 1:
        raise ValueError("Feature cache requires one raw window shape")
    shape = raw_shapes.iloc[0]
    if int(shape["n_channels"]) != len(channel_names):
        raise ValueError(
            "Feature profile channel count does not match raw data: "
            f"{len(channel_names)} != {int(shape['n_channels'])}"
        )
    preprocessing_hashes = sorted(
        index["preprocessing_hash"].dropna().astype(str).unique()
    )
    identity = {
        "cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "sample_id_universe_hash": sample_id_universe_hash(index["sample_id"]),
        "raw_preprocessing_hash": preprocessing_hashes[0],
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_hash": pipeline.feature_hash(len(channel_names), channel_names),
        "channel_order": channel_names,
        "dtype": str(np.dtype(dtype)),
        "rows": int(len(index)),
        "n_features": int(len(pipeline.feature_names(len(channel_names), channel_names))),
        "raw_window_shape": [
            int(shape["n_samples_expected"]),
            int(shape["n_channels"]),
        ],
    }
    identity["cache_identity_hash"] = _semantic_hash(identity)
    return identity


def _window_batch(view: RawEEGWindowArrayView, start: int, stop: int) -> np.ndarray:
    # Raw cache layout is [1, channels, time]; cogstate.features is [time, channels].
    return np.ascontiguousarray(
        np.stack([view[index][0].T for index in range(start, stop)]),
        dtype=np.float32,
    )


def _transform_payload(payload: tuple[Mapping[str, Any], np.ndarray]) -> np.ndarray:
    profile, windows = payload
    pipeline = FeaturePipeline(FeaturePipelineConfig.from_mapping(profile))
    return np.ascontiguousarray(pipeline.transform_batch(windows), dtype=np.float32)


def benchmark_worker_counts(
    profile: Mapping[str, Any],
    windows: np.ndarray,
    worker_counts: Sequence[int] = (1, 2, 4),
) -> list[dict[str, Any]]:
    """Benchmark deterministic process counts on one already-loaded real batch."""
    if windows.ndim != 3 or len(windows) == 0:
        raise ValueError("windows must be a non-empty [batch, samples, channels] array")
    rows: list[dict[str, Any]] = []
    reference: np.ndarray | None = None
    for configured_workers in worker_counts:
        workers = int(configured_workers)
        if workers <= 0:
            raise ValueError("worker counts must be positive")
        chunks = [
            chunk for chunk in np.array_split(windows, min(workers, len(windows)))
            if len(chunk)
        ]
        started = time.perf_counter()
        if workers == 1:
            transformed = _transform_payload((profile, chunks[0]))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                transformed = np.concatenate(
                    list(executor.map(_transform_payload, ((profile, chunk) for chunk in chunks))),
                    axis=0,
                )
        elapsed = time.perf_counter() - started
        if reference is None:
            reference = transformed
        deterministic = bool(np.array_equal(reference, transformed))
        rows.append(
            {
                "workers": workers,
                "windows": int(len(windows)),
                "elapsed_seconds": float(elapsed),
                "seconds_per_window": float(elapsed / len(windows)),
                "deterministic_equal_to_single_worker": deterministic,
            }
        )
    return rows


def materialize_cogstate_features(
    *,
    manifest_path: str | Path,
    logical_recording_map_path: str | Path,
    cache_path_root: str | Path,
    feature_profile_path: str | Path,
    output_dir: str | Path,
    chunk_size: int = 32,
    workers: int = 1,
    resume: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Materialize a deterministic target-free feature cache."""
    if chunk_size <= 0 or workers <= 0:
        raise ValueError("chunk_size and workers must be positive")
    profile, pipeline = load_feature_profile(feature_profile_path)
    index = build_canonical_feature_index(manifest_path, logical_recording_map_path)
    if max_rows is not None:
        if int(max_rows) <= 0:
            raise ValueError("max_rows must be positive or None")
        index = index.head(int(max_rows)).copy()
    identity = feature_cache_identity(index, pipeline)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / FEATURE_MANIFEST_NAME
    matrix_file = output / FEATURE_MATRIX_NAME
    index_file = output / FEATURE_INDEX_NAME
    names_file = output / FEATURE_NAMES_NAME
    names = pipeline.feature_names(
        len(identity["channel_order"]), identity["channel_order"]
    )

    completed_rows = 0
    elapsed_before = 0.0
    if manifest_file.exists():
        if not resume:
            raise FileExistsError(
                f"Feature cache already exists; use --resume: {output}"
            )
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        existing_identity = existing.get("identity")
        if existing_identity != identity:
            raise ValueError(
                "Incompatible feature cache identity; refusing to append: "
                f"stored={existing_identity}, requested={identity}"
            )
        completed_rows = int(existing.get("completed_rows", 0))
        elapsed_before = float(existing.get("elapsed_seconds", 0.0))
        if not matrix_file.is_file() or not index_file.is_file() or not names_file.is_file():
            raise FileNotFoundError("Resumable feature cache is missing required files")
    else:
        index.to_parquet(index_file, index=False)
        _atomic_json(names_file, {"feature_names": names})
        matrix = np.lib.format.open_memmap(
            matrix_file,
            mode="w+",
            dtype=np.float32,
            shape=(len(index), len(names)),
        )
        matrix.flush()
        del matrix
        _atomic_json(
            manifest_file,
            {
                "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                "status": "in_progress",
                "identity": identity,
                "completed_rows": 0,
                "elapsed_seconds": 0.0,
                "artifacts": {
                    "features": FEATURE_MATRIX_NAME,
                    "index": FEATURE_INDEX_NAME,
                    "feature_names": FEATURE_NAMES_NAME,
                },
            },
        )

    if completed_rows < 0 or completed_rows > len(index):
        raise ValueError(f"Invalid completed_rows in cache manifest: {completed_rows}")
    stored_index = pd.read_parquet(index_file)
    if not stored_index["sample_id"].reset_index(drop=True).equals(
        index["sample_id"].reset_index(drop=True)
    ):
        raise ValueError("Stored feature index does not match requested sample order")
    matrix = np.lib.format.open_memmap(matrix_file, mode="r+")
    if matrix.shape != (len(index), len(names)) or matrix.dtype != np.float32:
        raise ValueError(
            f"Stored feature matrix has incompatible shape/dtype: {matrix.shape}/{matrix.dtype}"
        )
    view = RawEEGWindowArrayView(index, cache_path_root=Path(cache_path_root))
    started = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(completed_rows, len(index), chunk_size):
            stop = min(start + chunk_size, len(index))
            windows = _window_batch(view, start, stop)
            if tuple(windows.shape[1:]) != tuple(identity["raw_window_shape"]):
                raise RuntimeError(
                    f"Unexpected raw batch shape {windows.shape}; expected "
                    f"[batch,{identity['raw_window_shape'][0]},{identity['raw_window_shape'][1]}]"
                )
            if executor is None:
                values = _transform_payload((profile, windows))
            else:
                chunks = [chunk for chunk in np.array_split(windows, min(workers, len(windows))) if len(chunk)]
                values = np.concatenate(
                    list(executor.map(_transform_payload, ((profile, chunk) for chunk in chunks))),
                    axis=0,
                )
            if values.shape != (stop - start, len(names)) or not np.isfinite(values).all():
                raise RuntimeError(
                    f"Feature chunk is invalid: shape={values.shape}, finite={np.isfinite(values).all()}"
                )
            matrix[start:stop] = values
            matrix.flush()
            completed_rows = stop
            _atomic_json(
                manifest_file,
                {
                    "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                    "status": "in_progress" if stop < len(index) else "complete",
                    "identity": identity,
                    "completed_rows": completed_rows,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                    "artifacts": {
                        "features": FEATURE_MATRIX_NAME,
                        "index": FEATURE_INDEX_NAME,
                        "feature_names": FEATURE_NAMES_NAME,
                    },
                },
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        matrix.flush()
        del matrix

    loaded_matrix, loaded_index, loaded_names, final_manifest = load_feature_cache(output)
    finite = bool(np.isfinite(loaded_matrix).all())
    if not finite:
        raise RuntimeError("Completed feature cache contains NaN or Inf")
    elapsed_total = float(final_manifest["elapsed_seconds"])
    summary = {
        "status": "complete",
        "rows": int(len(loaded_index)),
        "n_features": int(loaded_matrix.shape[1]),
        "dtype": str(loaded_matrix.dtype),
        "finite_fraction": 1.0,
        "sample_id_unique": bool(loaded_index["sample_id"].is_unique),
        "feature_schema": identity["feature_schema"],
        "feature_hash": identity["feature_hash"],
        "sample_id_universe_hash": identity["sample_id_universe_hash"],
        "raw_preprocessing_hash": identity["raw_preprocessing_hash"],
        "cache_identity_hash": identity["cache_identity_hash"],
        "elapsed_seconds": elapsed_total,
        "workers": int(workers),
        "chunk_size": int(chunk_size),
        "feature_matrix_size_bytes": int(matrix_file.stat().st_size),
        "feature_names_count": int(len(loaded_names)),
        "target_columns_present": False,
        "label_q5_dependency": False,
    }
    _atomic_json(output / FEATURE_SUMMARY_JSON_NAME, summary)
    pd.DataFrame([summary]).to_csv(output / FEATURE_SUMMARY_CSV_NAME, index=False)
    return summary


def load_feature_cache(
    output_dir: str | Path,
) -> tuple[np.ndarray, pd.DataFrame, list[str], dict[str, Any]]:
    """Load and validate a completed feature cache without copying the matrix."""
    output = Path(output_dir)
    manifest = json.loads((output / FEATURE_MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported feature cache schema")
    identity = manifest.get("identity", {})
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Feature cache is not complete: status={manifest.get('status')!r}"
        )
    index = pd.read_parquet(output / FEATURE_INDEX_NAME)
    names_payload = json.loads((output / FEATURE_NAMES_NAME).read_text(encoding="utf-8"))
    names = [str(value) for value in names_payload["feature_names"]]
    matrix = np.load(output / FEATURE_MATRIX_NAME, mmap_mode="r", allow_pickle=False)
    expected = (int(identity["rows"]), int(identity["n_features"]))
    if matrix.shape != expected or str(matrix.dtype) != identity["dtype"]:
        raise ValueError(
            f"Feature matrix shape/dtype mismatch: {matrix.shape}/{matrix.dtype}, expected {expected}/{identity['dtype']}"
        )
    if len(index) != len(matrix) or len(names) != matrix.shape[1]:
        raise ValueError("Feature cache index/names are not aligned with the matrix")
    if index["sample_id"].duplicated().any():
        raise ValueError("Feature cache index contains duplicate sample_id")
    if sample_id_universe_hash(index["sample_id"]) != identity["sample_id_universe_hash"]:
        raise ValueError("Feature cache sample_id universe hash mismatch")
    return matrix, index, names, manifest
