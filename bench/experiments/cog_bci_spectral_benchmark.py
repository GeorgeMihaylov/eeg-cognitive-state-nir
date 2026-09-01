"""Record-level 14- versus 62-channel COG-BCI spectral benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from bench.datasets.channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
from bench.datasets.cog_bci_dataset import COGBCIDataset
from bench.experiments.cog_bci_nback_baseline import classification_metrics
from bench.features.cog_bci_spectral_features import (
    COG_BCI_SPECTRAL_SCHEMA_VERSION,
    NUISANCE_FEATURE_TYPES,
    SpectralFeatureSpec,
    aggregate_record_features,
    extract_spectral_feature_bundle,
    feature_columns_for,
)


RESULT_STATUS = "diagnostic"
EXPECTED_WINDOWS = 16_927
EXPECTED_RECORDS = 261
EXPECTED_SUBJECTS = 29
EXPECTED_SESSIONS = 3
EXPECTED_RECORDS_PER_CLASS = 87
METADATA_COLUMNS = [
    "sample_id",
    "subject_id",
    "session_id",
    "record_id",
    "window_index",
    "start_sample",
    "stop_sample",
    "target",
    "class_name",
    "outer_fold",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str | Path, *, label: str) -> Path:
    text = str(value).replace("\\", "/")
    if Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise ValueError(f"{label} must be repository-relative, got {value!r}")
    path = Path(text)
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{label} must not escape the repository")
    return path


def _relative_string(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_branch(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _shard_stem(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class BenchmarkPaths:
    root: Path
    raw_cache: Path
    source_root: Path
    index_cache: Path
    protocol_dir: Path
    channel_audit_dir: Path
    output_dir: Path
    features_dir: Path
    tracked_report: Path


def resolve_paths(config: Mapping[str, Any], repository_root: Path) -> BenchmarkPaths:
    root = repository_root.resolve()

    def resolved(key: str) -> Path:
        return root / _relative_path(config[key], label=key)

    output = resolved("output_dir")
    return BenchmarkPaths(
        root=root,
        raw_cache=resolved("input_cache"),
        source_root=resolved("source_dataset_root"),
        index_cache=resolved("index_cache"),
        protocol_dir=resolved("task_protocol"),
        channel_audit_dir=resolved("channel_audit_dir"),
        output_dir=output,
        features_dir=output / "features",
        tracked_report=resolved("tracked_report"),
    )


def input_paths(paths: BenchmarkPaths) -> dict[str, Path]:
    return {
        "window_index": paths.raw_cache / "window_index.parquet",
        "task_definition": paths.protocol_dir / "task_definition.json",
        "target_index": paths.protocol_dir / "target_index.parquet",
        "outer_assignments": paths.protocol_dir / "outer_assignments.parquet",
        "outer_folds": paths.protocol_dir / "outer_folds.json",
        "inner_assignments": paths.protocol_dir / "inner_assignments.parquet",
        "inner_folds": paths.protocol_dir / "inner_folds.json",
        "emotiv_channel_mapping": paths.channel_audit_dir
        / "cog_bci_emotiv_mapping.csv",
        "common_channel_contract": paths.channel_audit_dir
        / "cog_bci_common_channels.json",
        "project_emotiv_contract": paths.channel_audit_dir
        / "project_emotiv_channel_contract.json",
    }


def input_hashes(paths: BenchmarkPaths) -> dict[str, str]:
    files = input_paths(paths)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing spectral benchmark inputs: {missing}")
    return {name: _sha256(path) for name, path in files.items()}


def _load_protocol(
    paths: BenchmarkPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = pd.read_parquet(paths.protocol_dir / "target_index.parquet")
    target = target.loc[
        target["included_for_supervised"].astype(bool)
        & target["status"].eq("accepted")
    ].copy()
    outer = pd.read_parquet(paths.protocol_dir / "outer_assignments.parquet")
    fold_map = outer[["sample_id", "fold"]].rename(columns={"fold": "outer_fold"})
    target = target.merge(fold_map, on="sample_id", validate="one_to_one")
    inner = pd.read_parquet(paths.protocol_dir / "inner_assignments.parquet")
    class_records = target.groupby("target")["record_id"].nunique().to_dict()
    observed = {
        "windows": len(target),
        "records": target["record_id"].nunique(),
        "subjects": target["subject_id"].nunique(),
        "sessions": target["session_id"].nunique(),
        "classes": sorted(target["target"].unique().tolist()),
        "records_per_class": {int(key): int(value) for key, value in class_records.items()},
    }
    expected = {
        "windows": EXPECTED_WINDOWS,
        "records": EXPECTED_RECORDS,
        "subjects": EXPECTED_SUBJECTS,
        "sessions": EXPECTED_SESSIONS,
        "classes": [0, 1, 2],
        "records_per_class": {0: 87, 1: 87, 2: 87},
    }
    if observed != expected:
        raise RuntimeError(
            f"Canonical COG-BCI N-Back protocol changed: "
            f"observed={observed}, expected={expected}"
        )
    return target, outer, inner


def _validate_window_identity(
    target: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    label: str,
) -> None:
    keys = [
        "sample_id",
        "record_id",
        "window_index",
        "start_sample",
        "stop_sample",
    ]
    expected = target[keys].sort_values("sample_id").reset_index(drop=True)
    actual = observed[keys].sort_values("sample_id").reset_index(drop=True)
    if not expected.equals(actual):
        raise RuntimeError(f"{label} does not preserve canonical window identity")


def _feature_frame_for_policy(
    paths: BenchmarkPaths,
    target: pd.DataFrame,
    dataset: COGBCIDataset,
    *,
    channel_policy: str,
    spec: SpectralFeatureSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if channel_policy == "emotiv_common":
        cache_index = pd.read_parquet(paths.raw_cache / "window_index.parquet")
        cache_index = cache_index.loc[cache_index["status"].eq("accepted")]
        selected = target.merge(
            cache_index[
                [
                    "sample_id",
                    "record_id",
                    "window_index",
                    "start_sample",
                    "stop_sample",
                    "cache_offset",
                ]
            ],
            on=[
                "sample_id",
                "record_id",
                "window_index",
                "start_sample",
                "stop_sample",
            ],
            validate="one_to_one",
        )
        source_manifest = json.loads(
            (paths.raw_cache / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        channel_order = tuple(source_manifest["channel_order"])
        if channel_order != PROJECT_EMOTIV_CHANNEL_ORDER:
            raise RuntimeError("14-channel raw cache order changed")
    elif channel_policy == "cog_bci_common":
        selected = target.copy()
        policy = dataset.get_channel_policy("cog_bci_common")
        channel_order = tuple(policy.required_names)
        if (
            len(channel_order) != 62
            or "Cz" in channel_order
            or "ECG1" in channel_order
        ):
            raise RuntimeError("62-channel common policy violates its contract")
    else:
        raise ValueError(f"Unsupported channel policy {channel_policy!r}")

    frames: list[pd.DataFrame] = []
    channel_columns: tuple[str, ...] | None = None
    global_columns: tuple[str, ...] | None = None
    channel_spectral: tuple[str, ...] | None = None
    channel_nuisance: tuple[str, ...] | None = None
    global_spectral: tuple[str, ...] | None = None
    global_nuisance: tuple[str, ...] | None = None
    for record_id, group in selected.groupby("record_id", sort=True):
        ordered = group.sort_values("window_index", kind="stable").reset_index(drop=True)
        if channel_policy == "emotiv_common":
            shard = np.load(
                paths.raw_cache / "shards" / f"{_shard_stem(str(record_id))}.npy",
                mmap_mode="r",
            )
            windows = np.asarray(
                shard[ordered["cache_offset"].to_numpy(dtype=int)],
                dtype=np.float32,
            )
        else:
            selection = dataset.select_raw_channels(
                str(record_id),
                "cog_bci_common",
                preload=False,
            )
            if tuple(selection.selected_names) != channel_order:
                raise RuntimeError(f"Channel order changed in record {record_id}")
            start = int(ordered["start_sample"].min())
            stop = int(ordered["stop_sample"].max())
            if start < 0 or stop > int(selection.raw.n_times):
                raise RuntimeError(f"Canonical bounds exceed record {record_id}")
            continuous = selection.raw.get_data(start=start, stop=stop)
            windows = np.stack(
                [
                    continuous[
                        :,
                        int(row.start_sample) - start : int(row.stop_sample) - start,
                    ]
                    for row in ordered.itertuples(index=False)
                ]
            ).astype(np.float32)
            selection.raw.close()
        if windows.shape != (len(ordered), len(channel_order), 2560):
            raise RuntimeError(
                f"Unexpected {channel_policy} window shape for {record_id}: "
                f"{windows.shape}"
            )
        bundle = extract_spectral_feature_bundle(
            windows,
            channel_names=channel_order,
            spec=spec,
        )
        current = (
            bundle.channel_wise_columns,
            bundle.global_summary_columns,
            bundle.channel_wise_spectral_columns,
            bundle.channel_wise_nuisance_columns,
            bundle.global_summary_spectral_columns,
            bundle.global_summary_nuisance_columns,
        )
        previous = (
            channel_columns,
            global_columns,
            channel_spectral,
            channel_nuisance,
            global_spectral,
            global_nuisance,
        )
        if channel_columns is None:
            (
                channel_columns,
                global_columns,
                channel_spectral,
                channel_nuisance,
                global_spectral,
                global_nuisance,
            ) = current
        elif current != previous:
            raise RuntimeError("Spectral feature order changed between records")
        metadata = ordered[METADATA_COLUMNS].reset_index(drop=True)
        features = pd.DataFrame(
            np.column_stack([bundle.channel_wise, bundle.global_summary]),
            columns=[*bundle.channel_wise_columns, *bundle.global_summary_columns],
        )
        frames.append(pd.concat([metadata, features], axis=1))

    if channel_columns is None or global_columns is None:
        raise RuntimeError(f"No features extracted for {channel_policy}")
    window_frame = pd.concat(frames, ignore_index=True)
    _validate_window_identity(target, window_frame, label=channel_policy)
    feature_columns = [*channel_columns, *global_columns]
    if not np.isfinite(window_frame[feature_columns].to_numpy()).all():
        raise RuntimeError(f"{channel_policy} feature cache contains NaN or Inf")
    record_frame = aggregate_record_features(
        window_frame,
        feature_columns=feature_columns,
    )
    if (
        len(record_frame) != EXPECTED_RECORDS
        or record_frame["record_id"].nunique() != EXPECTED_RECORDS
    ):
        raise RuntimeError(f"{channel_policy} record aggregation changed row count")
    schema = {
        "channel_policy": channel_policy,
        "channel_count": len(channel_order),
        "channel_order": list(channel_order),
        "window_feature_count": len(feature_columns),
        "representations": {
            "channel_wise": {
                "spectral_only": list(channel_spectral or ()),
                "nuisance": list(channel_nuisance or ()),
            },
            "global_summary": {
                "spectral_only": list(global_spectral or ()),
                "nuisance": list(global_nuisance or ()),
            },
        },
    }
    return window_frame, record_frame, schema


def build_feature_caches(
    paths: BenchmarkPaths,
    config: Mapping[str, Any],
    target: pd.DataFrame,
    *,
    hashes: Mapping[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """Build compact Parquet features without materializing 62-channel raw windows."""

    paths.features_dir.mkdir(parents=True, exist_ok=True)
    feature_config = config["spectral_features"]
    spec = SpectralFeatureSpec(
        sampling_rate_hz=float(feature_config["sampling_rate_hz"]),
        nperseg=int(feature_config["welch_nperseg"]),
        noverlap=int(feature_config["welch_noverlap"]),
        detrend=str(feature_config["detrend"]),
        scaling=str(feature_config["scaling"]),
    )
    dataset = COGBCIDataset(
        {
            "data_path": str(paths.source_root),
            "index_cache_path": str(paths.index_cache),
            "use_index_cache": True,
            "rebuild_index": False,
            "require_canonical_complete": True,
        }
    )
    schemas: dict[str, Any] = {}
    record_tables: dict[str, pd.DataFrame] = {}
    qc: dict[str, Any] = {}
    for policy, suffix in (("emotiv_common", "14"), ("cog_bci_common", "62")):
        window_frame, record_frame, schema = _feature_frame_for_policy(
            paths,
            target,
            dataset,
            channel_policy=policy,
            spec=spec,
        )
        window_path = paths.features_dir / f"window_features_{suffix}.parquet"
        record_path = paths.features_dir / f"record_features_{suffix}.parquet"
        window_frame.to_parquet(window_path, index=False, compression="zstd")
        record_frame.to_parquet(record_path, index=False, compression="zstd")
        schemas[policy] = schema
        record_tables[policy] = record_frame
        model_columns = [
            column for column in window_frame.columns if column.startswith(("cw__", "gs__"))
        ]
        qc[policy] = {
            "windows": len(window_frame),
            "records": len(record_frame),
            "subjects": window_frame["subject_id"].nunique(),
            "sessions": window_frame["session_id"].nunique(),
            "records_per_class": {
                str(key): int(value)
                for key, value in record_frame.groupby("target")["record_id"]
                .nunique()
                .items()
            },
            "nonfinite_feature_values": int(
                (~np.isfinite(window_frame[model_columns].to_numpy())).sum()
            ),
            "unique_sample_ids": window_frame["sample_id"].nunique(),
            "unique_record_ids": record_frame["record_id"].nunique(),
            "window_bounds_hash": _stable_hash(
                window_frame[
                    ["sample_id", "record_id", "start_sample", "stop_sample"]
                ]
                .sort_values("sample_id")
                .to_dict("records")
            ),
        }
    if qc["emotiv_common"]["window_bounds_hash"] != qc["cog_bci_common"][
        "window_bounds_hash"
    ]:
        raise RuntimeError("14- and 62-channel feature caches use different windows")

    implementation_path = Path(__file__).resolve().parents[1] / "features" / (
        "cog_bci_spectral_features.py"
    )
    feature_schema = {
        "schema_version": COG_BCI_SPECTRAL_SCHEMA_VERSION,
        "result_status": RESULT_STATUS,
        "specification": spec.to_dict(),
        "specification_hash": spec.stable_hash(),
        "policies": schemas,
        "record_aggregation": ["mean", "median", "std", "iqr"],
        "feature_sets": {
            "spectral_only": "excludes DC and 49-51 Hz nuisance features",
            "spectral_plus_nuisance": "adds explicitly named DC and line-noise features",
        },
    }
    provenance = {
        "schema_version": COG_BCI_SPECTRAL_SCHEMA_VERSION,
        "result_status": RESULT_STATUS,
        "input_hashes": dict(hashes),
        "source_window_hash": hashes["window_index"],
        "task_protocol_hash": json.loads(
            (paths.protocol_dir / "protocol_summary.json").read_text(encoding="utf-8")
        )["protocol_hash"],
        "implementation_hash": _sha256(implementation_path),
        "feature_specification_hash": spec.stable_hash(),
        "source_preprocessing": "raw",
        "raw_arrays_saved": False,
        "paths_are_repository_relative": True,
    }
    _write_json(paths.features_dir / "feature_schema.json", feature_schema)
    _write_json(paths.features_dir / "feature_qc.json", qc)
    _write_json(paths.features_dir / "feature_provenance.json", provenance)
    pd.DataFrame(columns=["stage", "record_id", "error_type", "message"]).to_csv(
        paths.features_dir / "errors.csv", index=False
    )
    return record_tables, feature_schema, qc


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    result = classification_metrics(y_true, prediction, probabilities)
    return {
        "accuracy": float(result["accuracy"]),
        "balanced_accuracy": float(result["balanced_accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_f1": float(result.get("weighted_f1", result["f1_weighted"])),
        "ordinal_mae": float(result["ordinal_mae"]),
        "within_one_class_accuracy": float(result["within_one_class_accuracy"]),
        "quadratic_weighted_kappa": float(result["quadratic_weighted_kappa"]),
        "severe_0_to_2_errors": int(
            np.sum((y_true == 0) & (prediction == 2))
            + np.sum((y_true == 2) & (prediction == 0))
        ),
        "confusion_matrix": result["confusion_matrix"],
    }


def _build_model(
    model: str,
    params: Mapping[str, Any],
    *,
    seed: int,
) -> Any:
    if model == "multinomial_logistic_regression":
        return LogisticRegression(
            C=float(params["C"]),
            solver="lbfgs",
            max_iter=int(params.get("max_iter", 3000)),
            random_state=int(seed),
        )
    if model == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=float(params["l2_regularization"]),
            early_stopping=False,
            random_state=int(seed),
        )
    raise ValueError(f"Unsupported model {model!r}")


def _model_grid(config: Mapping[str, Any], model: str) -> list[dict[str, Any]]:
    if model == "multinomial_logistic_regression":
        values = config["models"]["logistic_regression"]["C"]
        if len(values) > 4:
            raise ValueError("Logistic C grid must contain no more than four values")
        return [{"C": float(value), "max_iter": 3000} for value in values]
    values = [dict(item) for item in config["models"]["hist_gradient_boosting"]["grid"]]
    if len(values) > 4:
        raise ValueError("HGB grid must contain no more than four configurations")
    return values


def _split_records(
    records: pd.DataFrame,
    inner: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = inner.loc[inner["outer_fold"].eq(fold)].drop_duplicates(
        ["record_id", "partition"]
    )
    train_ids = set(
        assignments.loc[
            assignments["partition"].eq("inner_train"), "record_id"
        ].astype(str)
    )
    validation_ids = set(
        assignments.loc[
            assignments["partition"].eq("inner_validation"), "record_id"
        ].astype(str)
    )
    test_ids = set(
        records.loc[records["outer_fold"].eq(fold), "record_id"].astype(str)
    )
    if train_ids & validation_ids or (train_ids | validation_ids) & test_ids:
        raise RuntimeError(f"Fold {fold} has record leakage")
    frames = tuple(
        records.loc[records["record_id"].astype(str).isin(ids)].copy()
        for ids in (train_ids, validation_ids, test_ids)
    )
    subject_sets = [set(frame["subject_id"].astype(str)) for frame in frames]
    if (
        subject_sets[0] & subject_sets[1]
        or subject_sets[0] & subject_sets[2]
        or subject_sets[1] & subject_sets[2]
    ):
        raise RuntimeError(f"Fold {fold} has subject leakage")
    if any(frame.empty for frame in frames):
        raise RuntimeError(f"Fold {fold} contains an empty partition")
    return frames  # type: ignore[return-value]


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    params = candidate["params"]
    model = candidate["model"]
    if model == "multinomial_logistic_regression":
        complexity = float(params["C"])
    else:
        complexity = (
            int(params["max_leaf_nodes"]),
            int(params["max_iter"]),
            -float(params["l2_regularization"]),
        )
    return (
        -float(candidate["mean_inner_macro_f1"]),
        0 if candidate["feature_set"] == "spectral_only" else 1,
        int(candidate["n_features"]),
        complexity,
    )


def run_nested_benchmark(
    record_tables: Mapping[str, pd.DataFrame],
    inner: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested inner-only selection followed by untouched outer-fold prediction."""

    feature_sets = list(config["feature_sets"])
    representations = list(config["representations"])
    models = list(config["models"]["enabled"])
    hgb_seeds = [int(value) for value in config["models"]["hist_gradient_boosting"]["seeds"]]
    selection_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for channel_policy, records in record_tables.items():
        for representation in representations:
            for fold in range(1, 6):
                train, validation, test = _split_records(records, inner, fold)
                transformed: dict[str, dict[str, Any]] = {}
                for feature_set in feature_sets:
                    columns = feature_columns_for(
                        records,
                        representation=representation,
                        feature_set=feature_set,
                    )
                    scaler = StandardScaler().fit(train[columns])
                    transformed[feature_set] = {
                        "columns": columns,
                        "scaler": scaler,
                        "train": scaler.transform(train[columns]),
                        "validation": scaler.transform(validation[columns]),
                        "test": scaler.transform(test[columns]),
                    }
                for model_name in models:
                    candidates: list[dict[str, Any]] = []
                    grid = _model_grid(config, model_name)
                    selection_seed = int(
                        config["models"]
                        .get("hist_gradient_boosting", {})
                        .get("selection_seed", config["seed"])
                    )
                    candidate_seeds = [
                        int(config["seed"])
                        if model_name == "multinomial_logistic_regression"
                        else selection_seed
                    ]
                    evaluation_seeds = (
                        [int(config["seed"])]
                        if model_name == "multinomial_logistic_regression"
                        else hgb_seeds
                    )
                    for feature_set in feature_sets:
                        data = transformed[feature_set]
                        for grid_index, params in enumerate(grid):
                            seed_scores = []
                            candidate_row_indices = []
                            for seed in candidate_seeds:
                                model = _build_model(model_name, params, seed=seed)
                                model.fit(
                                    data["train"],
                                    train["target"].to_numpy(dtype=int),
                                )
                                probabilities = model.predict_proba(data["validation"])
                                metric = _metrics(
                                    validation["target"].to_numpy(dtype=int),
                                    probabilities,
                                )
                                candidate_row_indices.append(len(selection_rows))
                                selection_rows.append(
                                    {
                                        "channel_policy": channel_policy,
                                        "representation": representation,
                                        "fold": fold,
                                        "model": model_name,
                                        "seed": seed,
                                        "feature_set": feature_set,
                                        "n_features": len(data["columns"]),
                                        "grid_index": grid_index,
                                        "params": json.dumps(
                                            params, sort_keys=True, separators=(",", ":")
                                        ),
                                        "inner_macro_f1": metric["macro_f1"],
                                        "inner_balanced_accuracy": metric[
                                            "balanced_accuracy"
                                        ],
                                        "inner_ordinal_mae": metric["ordinal_mae"],
                                        "inner_qwk": metric[
                                            "quadratic_weighted_kappa"
                                        ],
                                        "scaler_fit_partition": "inner_train",
                                        "scaler_fit_records": len(train),
                                        "scaler_fit_subjects": train[
                                            "subject_id"
                                        ].nunique(),
                                        "outer_test_used_for_selection": False,
                                        "selected": False,
                                    }
                                )
                                seed_scores.append(metric["macro_f1"])
                            candidates.append(
                                {
                                    "model": model_name,
                                    "feature_set": feature_set,
                                    "n_features": len(data["columns"]),
                                    "grid_index": grid_index,
                                    "params": params,
                                    "mean_inner_macro_f1": float(
                                        np.mean(seed_scores)
                                    ),
                                    "row_indices": candidate_row_indices,
                                }
                            )
                    selected = sorted(candidates, key=_candidate_sort_key)[0]
                    for index in selected["row_indices"]:
                        selection_rows[index]["selected"] = True
                        selection_rows[index]["candidate_mean_inner_macro_f1"] = (
                            selected["mean_inner_macro_f1"]
                        )
                    selected_data = transformed[selected["feature_set"]]
                    for seed in evaluation_seeds:
                        model = _build_model(
                            model_name, selected["params"], seed=seed
                        )
                        model.fit(
                            selected_data["train"],
                            train["target"].to_numpy(dtype=int),
                        )
                        validation_probability = model.predict_proba(
                            selected_data["validation"]
                        )
                        test_probability = model.predict_proba(selected_data["test"])
                        validation_metric = _metrics(
                            validation["target"].to_numpy(dtype=int),
                            validation_probability,
                        )
                        test_metric = _metrics(
                            test["target"].to_numpy(dtype=int), test_probability
                        )
                        row = {
                            "channel_policy": channel_policy,
                            "representation": representation,
                            "fold": fold,
                            "model": model_name,
                            "seed": seed,
                            "feature_set": selected["feature_set"],
                            "n_features": selected["n_features"],
                            "params": json.dumps(
                                selected["params"],
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "train_records": len(train),
                            "validation_records": len(validation),
                            "test_records": len(test),
                            "train_subjects": train["subject_id"].nunique(),
                            "validation_subjects": validation["subject_id"].nunique(),
                            "test_subjects": test["subject_id"].nunique(),
                        }
                        row.update(
                            {
                                f"validation_{key}": value
                                for key, value in validation_metric.items()
                                if key != "confusion_matrix"
                            }
                        )
                        row.update(
                            {
                                f"test_{key}": value
                                for key, value in test_metric.items()
                                if key != "confusion_matrix"
                            }
                        )
                        fold_rows.append(row)
                        predictions = test[
                            [
                                "record_id",
                                "subject_id",
                                "session_id",
                                "target",
                                "class_name",
                            ]
                        ].reset_index(drop=True)
                        predictions = predictions.rename(columns={"target": "y_true"})
                        predictions["y_pred"] = np.argmax(test_probability, axis=1)
                        for class_id in range(3):
                            predictions[f"proba_{class_id}"] = test_probability[
                                :, class_id
                            ]
                        predictions["channel_policy"] = channel_policy
                        predictions["representation"] = representation
                        predictions["fold"] = fold
                        predictions["model"] = model_name
                        predictions["seed"] = seed
                        predictions["feature_set"] = selected["feature_set"]
                        prediction_frames.append(predictions)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    key_columns = [
        "channel_policy",
        "representation",
        "model",
        "seed",
        "record_id",
    ]
    if prediction_frame.duplicated(key_columns).any():
        raise RuntimeError("Duplicate OOF record predictions")
    model_seed_groups = sum(
        1 if model == "multinomial_logistic_regression" else len(hgb_seeds)
        for model in models
    )
    expected_groups = len(record_tables) * len(representations) * model_seed_groups
    if prediction_frame.groupby(key_columns[:-1]).ngroups != expected_groups:
        raise RuntimeError("OOF prediction grid is incomplete")
    return (
        prediction_frame,
        pd.DataFrame(fold_rows),
        pd.DataFrame(selection_rows),
    )


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    probabilities = frame[["proba_0", "proba_1", "proba_2"]].to_numpy(dtype=float)
    return _metrics(frame["y_true"].to_numpy(dtype=int), probabilities)


def aggregate_metrics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    aggregate_rows = []
    subject_rows = []
    matrices: dict[str, Any] = {}
    group_columns = ["channel_policy", "representation", "model", "seed"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        identity = dict(zip(group_columns, keys, strict=True))
        metrics = _prediction_metrics(group)
        matching_folds = fold_metrics
        for column, value in identity.items():
            matching_folds = matching_folds.loc[matching_folds[column].eq(value)]
        aggregate_rows.append(
            {
                **identity,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                **{
                    f"fold_{metric}_{statistic}": float(
                        getattr(matching_folds[f"test_{metric}"], statistic)(ddof=0)
                        if statistic == "std"
                        else getattr(
                            matching_folds[f"test_{metric}"], statistic
                        )()
                    )
                    for metric in (
                        "balanced_accuracy",
                        "macro_f1",
                        "ordinal_mae",
                    )
                    for statistic in ("mean", "std")
                },
            }
        )
        matrix_key = "|".join(str(value) for value in keys)
        matrices[matrix_key] = metrics["confusion_matrix"]
        for subject_id, subject in group.groupby("subject_id", sort=True):
            subject_metrics = _prediction_metrics(subject)
            subject_rows.append(
                {
                    **identity,
                    "subject_id": subject_id,
                    "records": len(subject),
                    **{
                        key: value
                        for key, value in subject_metrics.items()
                        if key != "confusion_matrix"
                    },
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    hgb = aggregate.loc[aggregate["model"].eq("hist_gradient_boosting")]
    for keys, group in hgb.groupby(["channel_policy", "representation"], sort=True):
        row: dict[str, Any] = {
            "channel_policy": keys[0],
            "representation": keys[1],
            "model": "hist_gradient_boosting",
            "seed": "mean_across_seeds",
        }
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "ordinal_mae",
            "within_one_class_accuracy",
            "quadratic_weighted_kappa",
        ):
            row[metric] = float(group[metric].mean())
            row[f"{metric}_between_seed_std"] = float(group[metric].std(ddof=0))
        aggregate_rows.append(row)
    return pd.DataFrame(aggregate_rows), pd.DataFrame(subject_rows), matrices


def _paired_policy_frames(
    predictions: pd.DataFrame,
    *,
    model: str,
    representation: str,
    seed: int,
) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["model"].eq(model)
        & predictions["representation"].eq(representation)
        & predictions["seed"].eq(seed)
    ]
    frames = {}
    for policy in ("emotiv_common", "cog_bci_common"):
        frame = subset.loc[subset["channel_policy"].eq(policy)].copy()
        frames[policy] = frame.rename(
            columns={
                "y_pred": f"y_pred_{policy}",
                "proba_0": f"proba_0_{policy}",
                "proba_1": f"proba_1_{policy}",
                "proba_2": f"proba_2_{policy}",
                "feature_set": f"feature_set_{policy}",
            }
        )
    keep = [
        "record_id",
        "subject_id",
        "session_id",
        "fold",
        "y_true",
        "class_name",
    ]
    paired = frames["emotiv_common"][
        [
            *keep,
            "y_pred_emotiv_common",
            "proba_0_emotiv_common",
            "proba_1_emotiv_common",
            "proba_2_emotiv_common",
            "feature_set_emotiv_common",
        ]
    ].merge(
        frames["cog_bci_common"][
            [
                "record_id",
                "y_true",
                "y_pred_cog_bci_common",
                "proba_0_cog_bci_common",
                "proba_1_cog_bci_common",
                "proba_2_cog_bci_common",
                "feature_set_cog_bci_common",
            ]
        ],
        on=["record_id", "y_true"],
        validate="one_to_one",
    )
    if len(paired) != EXPECTED_RECORDS:
        raise RuntimeError("Paired channel-policy comparison lost records")
    return paired


def _paired_metrics(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    probabilities = frame[
        [f"proba_{class_id}_{policy}" for class_id in range(3)]
    ].to_numpy(dtype=float)
    return _metrics(frame["y_true"].to_numpy(dtype=int), probabilities)


def subject_bootstrap_indices(
    subjects: Sequence[str],
    *,
    rng: np.random.Generator,
) -> list[str]:
    """Sample subjects with replacement; callers retain every subject record."""

    unique = np.asarray(sorted(set(str(value) for value in subjects)), dtype=object)
    if len(unique) == 0:
        raise ValueError("Cannot bootstrap an empty subject list")
    return [str(value) for value in rng.choice(unique, size=len(unique), replace=True)]


def subject_bootstrap_differences(
    paired: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    if repeats <= 0:
        raise ValueError("Bootstrap repeats must be positive")
    records_by_subject = {
        str(subject): group.index.to_numpy(dtype=int)
        for subject, group in paired.groupby("subject_id", sort=True)
    }
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(repeats):
        sampled = subject_bootstrap_indices(records_by_subject, rng=rng)
        indices = np.concatenate([records_by_subject[subject] for subject in sampled])
        sample = paired.loc[indices]
        y_true = sample["y_true"].to_numpy(dtype=int)
        predictions = {
            policy: sample[f"y_pred_{policy}"].to_numpy(dtype=int)
            for policy in ("emotiv_common", "cog_bci_common")
        }
        rows.append(
            {
                "replicate": replicate,
                "sampled_subjects": len(sampled),
                "sampled_records": len(sample),
                "balanced_accuracy_delta_62_minus_14": float(
                    balanced_accuracy_score(y_true, predictions["cog_bci_common"])
                    - balanced_accuracy_score(y_true, predictions["emotiv_common"])
                ),
                "macro_f1_delta_62_minus_14": float(
                    f1_score(
                        y_true,
                        predictions["cog_bci_common"],
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
                    - f1_score(
                        y_true,
                        predictions["emotiv_common"],
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "ordinal_mae_delta_62_minus_14": float(
                    np.mean(np.abs(y_true - predictions["cog_bci_common"]))
                    - np.mean(np.abs(y_true - predictions["emotiv_common"]))
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_channel_policies(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    comparison_rows = []
    bootstrap_frames = []
    bootstrap_summaries: dict[str, Any] = {}
    specifications = []
    for model in config["models"]["enabled"]:
        seeds = (
            [int(config["seed"])]
            if model == "multinomial_logistic_regression"
            else [
                int(value)
                for value in config["models"]["hist_gradient_boosting"]["seeds"]
            ]
        )
        for representation in config["representations"]:
            for seed in seeds:
                specifications.append((model, representation, seed))
    for model, representation, seed in specifications:
        paired = _paired_policy_frames(
            predictions,
            model=model,
            representation=representation,
            seed=seed,
        )
        policy_metrics = {
            policy: _paired_metrics(paired, policy)
            for policy in ("emotiv_common", "cog_bci_common")
        }
        fold_deltas = []
        for fold, fold_frame in paired.groupby("fold", sort=True):
            fold_metrics = {
                policy: _paired_metrics(fold_frame, policy)
                for policy in ("emotiv_common", "cog_bci_common")
            }
            delta = (
                fold_metrics["cog_bci_common"]["balanced_accuracy"]
                - fold_metrics["emotiv_common"]["balanced_accuracy"]
            )
            fold_deltas.append(delta)
            comparison_rows.append(
                {
                    "model": model,
                    "representation": representation,
                    "seed": seed,
                    "scope": "fold",
                    "fold": int(fold),
                    "balanced_accuracy_14": fold_metrics["emotiv_common"][
                        "balanced_accuracy"
                    ],
                    "balanced_accuracy_62": fold_metrics["cog_bci_common"][
                        "balanced_accuracy"
                    ],
                    "balanced_accuracy_delta_62_minus_14": delta,
                    "macro_f1_delta_62_minus_14": fold_metrics[
                        "cog_bci_common"
                    ]["macro_f1"]
                    - fold_metrics["emotiv_common"]["macro_f1"],
                    "ordinal_mae_delta_62_minus_14": fold_metrics[
                        "cog_bci_common"
                    ]["ordinal_mae"]
                    - fold_metrics["emotiv_common"]["ordinal_mae"],
                }
            )
        subject_deltas = []
        for _, subject in paired.groupby("subject_id", sort=True):
            subject_deltas.append(
                _paired_metrics(subject, "cog_bci_common")["balanced_accuracy"]
                - _paired_metrics(subject, "emotiv_common")["balanced_accuracy"]
            )
        pooled_delta = (
            policy_metrics["cog_bci_common"]["balanced_accuracy"]
            - policy_metrics["emotiv_common"]["balanced_accuracy"]
        )
        comparison_rows.append(
            {
                "model": model,
                "representation": representation,
                "seed": seed,
                "scope": "pooled",
                "fold": 0,
                "balanced_accuracy_14": policy_metrics["emotiv_common"][
                    "balanced_accuracy"
                ],
                "balanced_accuracy_62": policy_metrics["cog_bci_common"][
                    "balanced_accuracy"
                ],
                "balanced_accuracy_delta_62_minus_14": pooled_delta,
                "macro_f1_delta_62_minus_14": policy_metrics["cog_bci_common"][
                    "macro_f1"
                ]
                - policy_metrics["emotiv_common"]["macro_f1"],
                "ordinal_mae_delta_62_minus_14": policy_metrics[
                    "cog_bci_common"
                ]["ordinal_mae"]
                - policy_metrics["emotiv_common"]["ordinal_mae"],
                "folds_with_62_advantage": int(np.sum(np.asarray(fold_deltas) > 0)),
                "subjects_with_62_advantage": int(
                    np.sum(np.asarray(subject_deltas) > 0)
                ),
                "subjects_with_14_advantage": int(
                    np.sum(np.asarray(subject_deltas) < 0)
                ),
                "subjects_tied": int(np.sum(np.asarray(subject_deltas) == 0)),
            }
        )
        bootstrap = subject_bootstrap_differences(
            paired,
            repeats=int(config["bootstrap"]["repeats"]),
            seed=int(config["bootstrap"]["seed"]),
        )
        bootstrap.insert(0, "seed", seed)
        bootstrap.insert(0, "representation", representation)
        bootstrap.insert(0, "model", model)
        bootstrap_frames.append(bootstrap)
        key = f"{model}|{representation}|seed{seed}"
        bootstrap_summaries[key] = {
            column: {
                "mean": float(bootstrap[column].mean()),
                "lower_95": float(bootstrap[column].quantile(0.025)),
                "upper_95": float(bootstrap[column].quantile(0.975)),
                "positive_fraction": float((bootstrap[column] > 0).mean()),
                "negative_fraction": float((bootstrap[column] < 0).mean()),
            }
            for column in (
                "balanced_accuracy_delta_62_minus_14",
                "macro_f1_delta_62_minus_14",
                "ordinal_mae_delta_62_minus_14",
            )
        }
    return (
        pd.DataFrame(comparison_rows),
        pd.concat(bootstrap_frames, ignore_index=True),
        bootstrap_summaries,
    )


def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    total = float(np.sum((array - np.mean(array)) ** 2))
    if total <= 0:
        return 0.0
    means = values.groupby(groups).mean()
    counts = groups.value_counts()
    between = sum(
        float(counts[group]) * (float(mean) - float(np.mean(array))) ** 2
        for group, mean in means.items()
    )
    return float(between / total)


def effect_audit(
    record_tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    for channel_policy, records in record_tables.items():
        for representation in config["representations"]:
            for feature_set in config["feature_sets"]:
                columns = feature_columns_for(
                    records,
                    representation=representation,
                    feature_set=feature_set,
                )
                for effect, group_column in (
                    ("class", "target"),
                    ("subject", "subject_id"),
                    ("session", "session_id"),
                ):
                    values = [
                        _eta_squared(records[column], records[group_column])
                        for column in columns
                    ]
                    rows.append(
                        {
                            "channel_policy": channel_policy,
                            "representation": representation,
                            "feature_set": feature_set,
                            "effect": effect,
                            "features": len(columns),
                            "eta_squared_mean": float(np.mean(values)),
                            "eta_squared_median": float(np.median(values)),
                            "eta_squared_min": float(np.min(values)),
                            "eta_squared_max": float(np.max(values)),
                        }
                    )
    return pd.DataFrame(rows)


def _decision(
    comparison: pd.DataFrame,
    bootstrap_summary: Mapping[str, Any],
    fold_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rule = config["decision_rule"]
    model = str(rule["primary_model"])
    representation = str(rule["primary_representation"])
    seed = int(config["seed"])
    primary = comparison.loc[
        comparison["model"].eq(model)
        & comparison["representation"].eq(representation)
        & comparison["seed"].eq(seed)
        & comparison["scope"].eq("pooled")
    ].iloc[0]
    bootstrap_key = f"{model}|{representation}|seed{seed}"
    bootstrap = bootstrap_summary[bootstrap_key][
        "balanced_accuracy_delta_62_minus_14"
    ]
    selected_62 = fold_metrics.loc[
        fold_metrics["model"].eq(model)
        & fold_metrics["representation"].eq(representation)
        & fold_metrics["seed"].eq(seed)
        & fold_metrics["channel_policy"].eq("cog_bci_common")
    ]
    spectral_only_folds = int(selected_62["feature_set"].eq("spectral_only").sum())
    conditions = {
        "pooled_balanced_accuracy_gain_at_least_threshold": float(
            primary["balanced_accuracy_delta_62_minus_14"]
        )
        >= float(rule["minimum_balanced_accuracy_gain"]),
        "advantage_in_at_least_three_folds": int(
            primary["folds_with_62_advantage"]
        )
        >= int(rule["minimum_advantage_folds"]),
        "bootstrap_sign_not_obviously_unstable": float(
            bootstrap["positive_fraction"]
        )
        >= float(rule["minimum_positive_bootstrap_fraction"]),
        "advantage_not_only_nuisance": spectral_only_folds
        >= int(rule["minimum_spectral_only_selected_folds"]),
    }
    proceed = all(conditions.values())
    return {
        "result_status": RESULT_STATUS,
        "recommendation": (
            "build_62_channel_raw_cache" if proceed else "retain_14_channel_cache"
        ),
        "primary_comparison": {
            "model": model,
            "representation": representation,
            "seed": seed,
            "balanced_accuracy_14": float(primary["balanced_accuracy_14"]),
            "balanced_accuracy_62": float(primary["balanced_accuracy_62"]),
            "balanced_accuracy_delta_62_minus_14": float(
                primary["balanced_accuracy_delta_62_minus_14"]
            ),
            "folds_with_62_advantage": int(primary["folds_with_62_advantage"]),
            "spectral_only_selected_folds_62": spectral_only_folds,
            "bootstrap": bootstrap,
        },
        "conditions": conditions,
        "thresholds": dict(rule),
        "interpretation": (
            "The subject bootstrap is a stability diagnostic, not a claim of "
            "statistical significance."
        ),
    }


def _feature_inventory(paths: BenchmarkPaths) -> dict[str, Any]:
    rows = []
    for path in sorted(paths.features_dir.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "path": _relative_string(path, paths.root),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {
        "result_status": RESULT_STATUS,
        "files": rows,
        "total_size_bytes": int(sum(row["size_bytes"] for row in rows)),
    }


def _report(
    *,
    summary: Mapping[str, Any],
    aggregate: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    effects: pd.DataFrame,
    decision: Mapping[str, Any],
) -> str:
    lines = [
        "# COG-BCI N-Back: 14- и 62-канальный спектральный benchmark",
        "",
        f"- Ветка: `{summary['branch']}`.",
        f"- HEAD: `{summary['head']}`.",
        f"- Статус: `{RESULT_STATUS}`.",
        "- Исходные EEG, raw cache, task protocol и split manifests не изменены.",
        "",
        "## Feature contract",
        "",
        "Raw-сигнал; Welch 512/256, constant detrend. Для каждого канала "
        "вычислены log/relative band powers, theta/alpha, theta/beta и log "
        "variance. DC и 49–51 Hz доступны только в явно именованном "
        "`spectral_plus_nuisance`.",
        "",
        "Окна агрегированы внутри каждой записи через mean, median, std и IQR. "
        "Основной объект оценки — 261 запись; split — исходный пятифолдовый "
        "GroupKFold по subject_id с готовым subject-disjoint inner split.",
        "",
        "Размерности record-level feature sets: 14-channel channel-wise "
        "`728/896`, 62-channel channel-wise `3224/3968`, global summary для "
        "обеих политик `260/320` (`spectral_only/spectral_plus_nuisance`).",
        "",
        "SHA-256 всех входных manifests и channel contracts сохранены в "
        "`benchmark_summary.json`; hashes до и после запуска совпадают.",
        "",
        "## Pooled record-level metrics",
        "",
        "| Policy | Representation | Model | Seed | BA | Macro F1 | Ordinal MAE | QWK |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.loc[aggregate["seed"].astype(str).ne("mean_across_seeds")].itertuples():
        lines.append(
            f"| {row.channel_policy} | {row.representation} | {row.model} | "
            f"{row.seed} | {row.balanced_accuracy:.4f} | {row.macro_f1:.4f} | "
            f"{row.ordinal_mae:.4f} | {row.quadratic_weighted_kappa:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fold-level primary Logistic Regression",
            "",
        "| Policy | Representation | Fold | Feature set | Features | BA | Macro F1 |",
        "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    primary_folds = fold_metrics.loc[
        fold_metrics["model"].eq("multinomial_logistic_regression")
    ]
    for row in primary_folds.itertuples():
        lines.append(
            f"| {row.channel_policy} | {row.representation} | {row.fold} | "
            f"{row.feature_set} | {row.n_features} | "
            f"{row.test_balanced_accuracy:.4f} | {row.test_macro_f1:.4f} |"
        )
    lines.extend(["", "Inner-selected Logistic `C` по folds:"])
    for (policy, representation), group in primary_folds.groupby(
        ["channel_policy", "representation"], sort=True
    ):
        ordered = group.sort_values("fold")
        values = [json.loads(value)["C"] for value in ordered["params"]]
        lines.append(
            f"- `{policy}/{representation}`: "
            f"`{json.dumps(values, separators=(',', ':'))}`."
        )
    lines.extend(
        [
            "",
            "HGB использовал две фиксированные конфигурации: simple "
            "`lr=0.05/leaves=7/l2=0.001` и extended "
            "`lr=0.08/leaves=15/l2=0.0001`; точный fold-level выбор сохранён "
            "в `hyperparameter_selection.csv`.",
        ]
    )
    lines.extend(
        [
            "",
            "## 62 минус 14 каналов",
            "",
            "| Model | Representation | Seed | Δ BA | Δ Macro F1 | Δ Ordinal MAE | "
            "Folds 62>14 | Subjects 62>14 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pooled = comparison.loc[comparison["scope"].eq("pooled")]
    for row in pooled.itertuples():
        lines.append(
            f"| {row.model} | {row.representation} | {row.seed} | "
            f"{row.balanced_accuracy_delta_62_minus_14:+.4f} | "
            f"{row.macro_f1_delta_62_minus_14:+.4f} | "
            f"{row.ordinal_mae_delta_62_minus_14:+.4f} | "
            f"{int(row.folds_with_62_advantage)} | "
            f"{int(row.subjects_with_62_advantage)} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive effects",
            "",
            "| Policy | Representation | Feature set | Effect | Mean eta² | Median eta² |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in effects.itertuples():
        lines.append(
            f"| {row.channel_policy} | {row.representation} | {row.feature_set} | "
            f"{row.effect} | {row.eta_squared_mean:.4f} | "
            f"{row.eta_squared_median:.4f} |"
        )
    primary = decision["primary_comparison"]
    bootstrap = primary["bootstrap"]
    lines.extend(
        [
            "",
            "## Subject bootstrap и решение",
            "",
            f"Primary comparison: `{primary['model']}` / "
            f"`{primary['representation']}`, seed {primary['seed']}. "
            f"Δ balanced accuracy = "
            f"`{primary['balanced_accuracy_delta_62_minus_14']:+.4f}`; "
            f"95% subject-bootstrap interval "
            f"`[{bootstrap['lower_95']:+.4f}, {bootstrap['upper_95']:+.4f}]`, "
            f"positive fraction `{bootstrap['positive_fraction']:.3f}`.",
            "",
            f"Решение: `{decision['recommendation']}`.",
            "",
            "Bootstrap используется как диагностика устойчивости знака эффекта и "
            "не интерпретируется как доказательство статистической значимости.",
            "",
            "## Сравнение с CNN и ограничения",
            "",
            "Сохранённый 14-канальный CNN baseline имеет record balanced accuracy "
            "около 0.356 для EEGNet и ShallowConvNet. Настоящий benchmark "
            "сравнивает только лёгкие record-level модели и не запускает глубокое "
            "обучение.",
            "",
            "Ограничения: 261 записи от 29 участников; признаки агрегируют полные "
            "record_full записи; acquisition units/filter history остаются "
            "частично неразрешёнными. Outer-test не использовался для выбора "
            "feature set, гиперпараметров или scaler.",
            "",
            "Рекомендуемый следующий этап определяется `decision.json`; полный "
            "62-канальный raw cache не строился.",
            "",
        ]
    )
    return "\n".join(lines)


def run_cog_bci_spectral_benchmark(
    config: Mapping[str, Any],
    *,
    repository_root: Path | str = ".",
    verbose: bool = False,
) -> dict[str, Any]:
    """Run feature materialization, nested models and paired comparison."""

    started = time.perf_counter()
    root = Path(repository_root).resolve()
    paths = resolve_paths(config, root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.features_dir.mkdir(parents=True, exist_ok=True)
    hashes_before = input_hashes(paths)
    target, _, inner = _load_protocol(paths)
    resolved_config = json.loads(json.dumps(config))
    resolved_config["config_hash"] = _stable_hash(config)
    resolved_config["result_status"] = RESULT_STATUS
    _write_json(paths.output_dir / "resolved_config.json", resolved_config)
    if verbose:
        print("Building deterministic 14/62-channel spectral feature caches...")
    record_tables, feature_schema, feature_qc = build_feature_caches(
        paths,
        config,
        target,
        hashes=hashes_before,
    )
    if verbose:
        print("Running nested inner-only model selection and outer evaluation...")
    predictions, fold_metrics, selection = run_nested_benchmark(
        record_tables,
        inner,
        config,
    )
    aggregate, subject_metrics, matrices = aggregate_metrics(
        predictions, fold_metrics
    )
    comparison, bootstrap, bootstrap_summary = compare_channel_policies(
        predictions, config
    )
    effects = effect_audit(record_tables, config)
    decision = _decision(
        comparison,
        bootstrap_summary,
        fold_metrics,
        config,
    )
    predictions.to_parquet(
        paths.output_dir / "record_predictions.parquet", index=False
    )
    fold_metrics.to_csv(paths.output_dir / "feature_fold_metrics.csv", index=False)
    aggregate.to_csv(
        paths.output_dir / "feature_aggregate_metrics.csv", index=False
    )
    subject_metrics.to_csv(paths.output_dir / "subject_metrics.csv", index=False)
    selection.to_csv(
        paths.output_dir / "hyperparameter_selection.csv", index=False
    )
    comparison.to_csv(
        paths.output_dir / "channel_policy_comparison.csv", index=False
    )
    bootstrap.to_csv(paths.output_dir / "subject_bootstrap.csv", index=False)
    effects.to_csv(paths.output_dir / "effect_audit.csv", index=False)
    _write_json(paths.output_dir / "confusion_matrices.json", matrices)
    _write_json(paths.output_dir / "bootstrap_summary.json", bootstrap_summary)
    _write_json(paths.output_dir / "decision.json", decision)
    pd.DataFrame(columns=["stage", "error_type", "message"]).to_csv(
        paths.output_dir / "errors.csv", index=False
    )
    inventory = _feature_inventory(paths)
    _write_json(paths.output_dir / "feature_inventory.json", inventory)
    hashes_after = input_hashes(paths)
    if hashes_after != hashes_before:
        raise RuntimeError("A canonical input changed during the spectral benchmark")
    summary = {
        "result_status": RESULT_STATUS,
        "branch": _git_branch(root),
        "head": _git_commit(root),
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "inputs_unchanged": True,
        "dataset": {
            "windows": EXPECTED_WINDOWS,
            "records": EXPECTED_RECORDS,
            "subjects": EXPECTED_SUBJECTS,
            "sessions": EXPECTED_SESSIONS,
            "records_per_class": EXPECTED_RECORDS_PER_CLASS,
        },
        "feature_schema": feature_schema,
        "feature_qc": feature_qc,
        "feature_inventory": inventory,
        "decision": decision,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(paths.output_dir / "benchmark_summary.json", summary)
    report = _report(
        summary=summary,
        aggregate=aggregate,
        fold_metrics=fold_metrics,
        comparison=comparison,
        effects=effects,
        decision=decision,
    )
    (paths.output_dir / "benchmark_report.md").write_text(
        report, encoding="utf-8"
    )
    paths.tracked_report.parent.mkdir(parents=True, exist_ok=True)
    paths.tracked_report.write_text(report, encoding="utf-8")
    if verbose:
        print(
            f"Completed spectral benchmark: {decision['recommendation']} "
            f"({summary['elapsed_seconds']:.1f}s)"
        )
    return summary
