"""Zero-shot XGBoost transfer from full Emotiv-14 to nested montages.

The experiment materializes one target-free regional cache, trains exactly one
XGBoost classifier per PM and outer fold on ``full_14``, and evaluates that
unchanged booster on every registered montage profile.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from bench.experiments.artifact_removal_ablation_v2 import (
    SignalUniverse,
    load_signal_universe,
)
from bench.tasks.target_registry import PM_METRICS, get_target_spec
from bench.tasks.target_transforms import (
    FoldLocalQuantileTargetTransform,
    build_target_transform_manifest,
)
from cogstate.features import (
    CANONICAL_REGIONS,
    EMOTIV_14_CHANNELS,
    REGIONAL_FEATURE_SCHEMA_VERSION,
    REGIONAL_FEATURE_SCHEMA_VERSION_V2,
    RegionalFeatureConfig,
    RegionalFeaturePipeline,
)
from cogstate.model_zoo import build_model
from cogstate.model_zoo.ML.xgboost_personalization import xgboost_state_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "xgboost-regional-montage-transfer-v1"
EXPERIMENT_ID = "xgboost_regional_montage_transfer_v1"
SCHEMA_VERSION_V2 = "xgboost-regional-montage-transfer-v2"
EXPERIMENT_ID_V2 = "seven_pm_xgboost_regional_montage_transfer_v2"
EXPERIMENT_CONTRACTS = {
    SCHEMA_VERSION: {
        "experiment_id": EXPERIMENT_ID,
        "output_dir": "benchmark_results/xgboost_regional_montage_transfer_v1",
        "regional_features": {
            "schema_version": REGIONAL_FEATURE_SCHEMA_VERSION,
            "sample_rate_hz": 256,
            "feature_width": 728,
            "dtype": "float32",
            "connectivity_included": False,
        },
    },
    SCHEMA_VERSION_V2: {
        "experiment_id": EXPERIMENT_ID_V2,
        "output_dir": "benchmark_results/xgboost_regional_montage_transfer_v2",
        "regional_features": {
            "schema_version": REGIONAL_FEATURE_SCHEMA_VERSION_V2,
            "sample_rate_hz": 256,
            "feature_width": 364,
            "dtype": "float32",
            "connectivity_included": False,
            "aggregations": ["median"],
            "include_region_present": True,
            "include_channel_count": False,
        },
    },
}
# Historical v1 hashes are provenance inputs of the completed experiment.  They
# remain pinned for v1 so adding the opt-in v2 representation cannot invalidate
# or overwrite its existing protocol/cache identity.
V1_IMPLEMENTATION_SHA256 = {
    "experiment": "f529a5fe26ca42e6152e7c25c01b7e0e2831c9984952f63d535685e481f106b9",
    "regional": "6ed3dcf9cc8556cafb097496100dd1a0ad35dcbe5011df9c14968fd1211a1e84",
    "montage": "4b490c7543bfd263145a36f415fcb714033c9547fdf84421c72e2b972b1a1404",
}
PROFILE_ORDER = (
    "full_14",
    "reduced_12",
    "regional_10",
    "coverage_8",
    "coverage_6",
)
MONTAGE_PROFILES: dict[str, tuple[str, ...]] = {
    "full_14": EMOTIV_14_CHANNELS,
    "reduced_12": (
        "AF3", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
        "FC6", "F4", "AF4",
    ),
    "regional_10": (
        "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4",
    ),
    "coverage_8": ("F3", "T7", "P7", "O1", "O2", "P8", "T8", "F4"),
    "coverage_6": ("F3", "T7", "P7", "P8", "T8", "F4"),
}
PROFILE_INTERPRETATION = {
    "full_14": "reference Emotiv-14 montage",
    "reduced_12": "remove F7/F8 while preserving all ten lateral regions",
    "regional_10": "one channel per Emotiv-accessible lateral region",
    "coverage_8": "central_left and central_right absent",
    "coverage_6": "central and occipital lateral regions absent",
}
CACHE_SCHEMA_VERSION = "regional-montage-feature-cache-v1"
FEATURE_MATRIX_NAME = "regional_features.npy"
FEATURE_INDEX_NAME = "feature_index.parquet"
FEATURE_NAMES_NAME = "feature_names.json"
CACHE_MANIFEST_NAME = "feature_cache_manifest.json"


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_path(value: str | Path) -> str:
    path = _repo_path(value)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    schema_version = config.get("schema_version")
    contract = EXPERIMENT_CONTRACTS.get(schema_version)
    if contract is None:
        raise ValueError(
            f"schema_version must be one of {tuple(EXPERIMENT_CONTRACTS)}"
        )
    if config.get("experiment_id") != contract["experiment_id"]:
        raise ValueError(
            f"experiment_id must equal {contract['experiment_id']}"
        )
    if config.get("output_dir") != contract["output_dir"]:
        raise ValueError(f"output_dir must equal {contract['output_dir']}")
    if tuple(config.get("targets", ())) != PM_METRICS:
        raise ValueError(f"targets must contain all seven PM in order: {PM_METRICS}")
    profiles = config.get("profiles", {})
    if tuple(profiles) != PROFILE_ORDER:
        raise ValueError(f"profile order must equal {PROFILE_ORDER}")
    normalized_profiles = {
        name: tuple(str(channel) for channel in channels)
        for name, channels in profiles.items()
    }
    if normalized_profiles != MONTAGE_PROFILES:
        raise ValueError("montage profiles differ from the preregistered registry")
    if config.get("regional_features") != contract["regional_features"]:
        raise ValueError(
            "regional_features must equal the locked representation profile"
        )
    if config.get("model") != {
        "name": "xgboost",
        "params": {"n_estimators": 200, "n_jobs": 4, "random_state": 42},
    }:
        raise ValueError("XGBoost configuration differs from the fixed benchmark")
    evaluation = config.get("evaluation", {})
    if tuple(map(int, evaluation.get("folds", ()))) != (1, 2, 3, 4, 5):
        raise ValueError("evaluation folds must be [1,2,3,4,5]")
    if evaluation.get("group_column") != "subject_id":
        raise ValueError("outer grouping must use subject_id")
    if evaluation.get("record_group_column") != "record_group_id":
        raise ValueError("record-group identity must use record_group_id")
    if evaluation.get("q3_fit_scope") != "outer_train_only":
        raise ValueError("Q3 fit scope must be outer_train_only")
    if tuple(map(int, config.get("smoke", {}).get("folds", ()))) != (1,):
        raise ValueError("smoke must contain only outer fold 1")
    if tuple(config["smoke"].get("targets", ())) != PM_METRICS:
        raise ValueError("smoke must contain all seven PM")
    validate_nested_profiles(normalized_profiles)
    return config


def validate_nested_profiles(
    profiles: Mapping[str, Sequence[str]] = MONTAGE_PROFILES,
) -> None:
    if tuple(profiles) != PROFILE_ORDER:
        raise ValueError(f"profile order must equal {PROFILE_ORDER}")
    previous: set[str] | None = None
    for name in PROFILE_ORDER:
        channels = tuple(str(channel) for channel in profiles[name])
        if len(channels) != len(set(channels)):
            raise ValueError(f"profile {name} contains duplicate channels")
        current = set(channels)
        if previous is not None and not current < previous:
            raise ValueError(f"profile {name} must be a strict subset of its predecessor")
        previous = current


def _regional_feature_config(
    regional: Mapping[str, Any],
) -> RegionalFeatureConfig:
    return RegionalFeatureConfig(
        sample_rate=float(regional["sample_rate_hz"]),
        schema_version=str(regional["schema_version"]),
        aggregations=tuple(regional.get("aggregations", ("median", "iqr"))),
        include_region_present=bool(regional.get("include_region_present", True)),
        include_channel_count=bool(regional.get("include_channel_count", True)),
    )


def _pipeline(config: Mapping[str, Any]) -> RegionalFeaturePipeline:
    return RegionalFeaturePipeline(
        _regional_feature_config(config["regional_features"])
    )


def profile_registry_manifest(
    pipeline: RegionalFeaturePipeline,
    profiles: Mapping[str, Sequence[str]] = MONTAGE_PROFILES,
) -> dict[str, Any]:
    validate_nested_profiles(profiles)
    schema_hash = pipeline.schema_hash()
    rows: list[dict[str, Any]] = []
    for profile_name in PROFILE_ORDER:
        channels = tuple(profiles[profile_name])
        montage = pipeline.montage_manifest(channels)
        counts = montage["region_channel_counts"]
        present = [region for region in CANONICAL_REGIONS if int(counts[region]) > 0]
        absent = [region for region in CANONICAL_REGIONS if int(counts[region]) == 0]
        rows.append(
            {
                "profile": profile_name,
                "interpretation": PROFILE_INTERPRETATION[profile_name],
                "channels": list(channels),
                "channel_count": len(channels),
                "regions_present": present,
                "regions_absent": absent,
                "region_channel_counts": counts,
                "constant_missing_region_feature_count": sum(
                    any(
                        marker in feature_name
                        for marker in (
                            f"__{region}__",
                            f"coverage__{region}__",
                        )
                    )
                    for region in absent
                    for feature_name in pipeline.feature_names()
                ),
                "schema_hash": schema_hash,
                "montage_hash": pipeline.montage_hash(channels),
                "montage_manifest": montage,
            }
        )
    payload = {
        "schema_version": (
            SCHEMA_VERSION_V2
            if pipeline.config.schema_version == REGIONAL_FEATURE_SCHEMA_VERSION_V2
            else SCHEMA_VERSION
        ),
        "profile_order": list(PROFILE_ORDER),
        "profiles_are_strictly_nested": True,
        "regional_feature_schema_hash": schema_hash,
        "feature_width": len(pipeline.feature_names()),
        "profiles": rows,
    }
    payload["profile_registry_hash"] = stable_hash(payload)
    return payload


@dataclass(frozen=True)
class MontageTrainingSpec:
    pm: str
    fold: int
    seed: int = 42

    @property
    def run_id(self) -> str:
        return f"fold_{self.fold:02d}__{self.pm}__xgboost_full14"


def build_run_matrix(config: Mapping[str, Any]) -> list[MontageTrainingSpec]:
    specs = [
        MontageTrainingSpec(pm=str(pm), fold=int(fold), seed=42)
        for fold in config["evaluation"]["folds"]
        for pm in config["targets"]
    ]
    if len(specs) != 35:
        raise RuntimeError(f"Expected 35 XGBoost trainings, got {len(specs)}")
    return specs


def _sample_hash(values: Iterable[Any]) -> str:
    return stable_hash([str(value) for value in values])


def _fold_audit(
    universe: SignalUniverse, config: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = universe.manifest
    subject_folds = manifest.groupby("subject_id", sort=True)["outer_fold"].nunique()
    if not subject_folds.eq(1).all():
        raise RuntimeError("A participant appears in multiple outer folds")
    reference = json.loads(
        _repo_path(config["dataset"]["reference_fold_manifest"]).read_text(
            encoding="utf-8"
        )
    )["folds"]
    expected = {
        str(subject): int(fold)
        for fold, payload in reference.items()
        for subject in payload["test_subject_ids"]
    }
    observed = (
        manifest[["subject_id", "outer_fold"]]
        .drop_duplicates()
        .assign(subject_id=lambda frame: frame["subject_id"].astype(str))
    )
    mismatches = {
        row.subject_id: {"observed": int(row.outer_fold), "expected": expected.get(row.subject_id)}
        for row in observed.itertuples(index=False)
        if expected.get(row.subject_id) != int(row.outer_fold)
    }
    folds: dict[str, Any] = {}
    for fold in config["evaluation"]["folds"]:
        test = manifest["outer_fold"].astype(int).eq(int(fold))
        train_subjects = set(manifest.loc[~test, "subject_id"].astype(str))
        test_subjects = set(manifest.loc[test, "subject_id"].astype(str))
        train_groups = set(manifest.loc[~test, "record_group_id"].astype(str))
        test_groups = set(manifest.loc[test, "record_group_id"].astype(str))
        subject_overlap = sorted(train_subjects & test_subjects)
        group_overlap = sorted(train_groups & test_groups)
        if subject_overlap or group_overlap:
            raise RuntimeError(f"Outer leakage in fold {fold}")
        folds[str(fold)] = {
            "train_rows": int((~test).sum()),
            "test_rows": int(test.sum()),
            "train_subjects": len(train_subjects),
            "test_subjects": len(test_subjects),
            "test_subject_ids": sorted(test_subjects),
            "subject_overlap": subject_overlap,
            "record_group_overlap": group_overlap,
            "train_sample_hash": _sample_hash(manifest.loc[~test, "sample_id"]),
            "test_sample_hash": _sample_hash(manifest.loc[test, "sample_id"]),
        }
    return {
        "reference_manifest": _relative_path(config["dataset"]["reference_fold_manifest"]),
        "reference_assignments_match": not mismatches,
        "mismatches": mismatches,
        "folds": folds,
    }


def _target_fold_audit(
    universe: SignalUniverse, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    transforms: dict[str, dict[str, Any]] = {}
    folds = universe.manifest["outer_fold"].to_numpy(dtype=int)
    sample_ids = universe.manifest["sample_id"].to_numpy()
    for fold in config["evaluation"]["folds"]:
        for pm in config["targets"]:
            values = universe.targets[f"target_{pm}"].to_numpy(dtype=np.float32)
            valid = np.isfinite(values)
            train = valid & (folds != int(fold))
            test = valid & (folds == int(fold))
            transform = FoldLocalQuantileTargetTransform(3, duplicates="drop").fit(
                values[train]
            )
            target_spec = get_target_spec(f"pm_{pm}_q3_fold_local")
            manifest = build_target_transform_manifest(
                target_spec,
                transform,
                outer_fold=int(fold),
                outer_train_sample_ids=sample_ids[train],
                outer_train_targets=values[train],
            )
            if int(manifest["actual_class_count"]) != 3:
                raise RuntimeError(f"Fold {fold} PM {pm} does not retain three Q3 classes")
            train_labels = transform.transform(values[train]).astype(int)
            test_labels = transform.transform(values[test]).astype(int)
            key = f"fold_{int(fold):02d}__{pm}"
            transforms[key] = manifest
            rows.append(
                {
                    "fold": int(fold),
                    "pm": pm,
                    "train_rows": int(train.sum()),
                    "test_rows": int(test.sum()),
                    "train_sample_hash": _sample_hash(sample_ids[train]),
                    "test_sample_hash": _sample_hash(sample_ids[test]),
                    "train_class_0": int((train_labels == 0).sum()),
                    "train_class_1": int((train_labels == 1).sum()),
                    "train_class_2": int((train_labels == 2).sum()),
                    "test_class_0": int((test_labels == 0).sum()),
                    "test_class_1": int((test_labels == 1).sum()),
                    "test_class_2": int((test_labels == 2).sum()),
                    "q3_transform_hash": manifest["transform_hash"],
                }
            )
    return pd.DataFrame(rows), transforms


def protocol_plan(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    universe = load_signal_universe(config)
    pipeline = _pipeline(config)
    registry = profile_registry_manifest(pipeline, config["profiles"])
    specs = build_run_matrix(config)
    fold_audit = _fold_audit(universe, config)
    target_audit, transforms = _target_fold_audit(universe, config)
    if not fold_audit["reference_assignments_match"]:
        raise RuntimeError("Fixed outer folds differ from the reference manifest")
    expected_width = int(config["regional_features"]["feature_width"])
    if len(pipeline.feature_names()) != expected_width:
        raise RuntimeError("Regional feature width changed")
    schema_hashes = {row["schema_hash"] for row in registry["profiles"]}
    montage_hashes = {row["montage_hash"] for row in registry["profiles"]}
    if len(schema_hashes) != 1 or len(montage_hashes) != len(PROFILE_ORDER):
        raise RuntimeError("Schema/montage hash contract failed")

    run_matrix_hash = stable_hash([asdict(spec) for spec in specs])
    schema_version = str(config["schema_version"])
    implementation_sha256 = (
        dict(V1_IMPLEMENTATION_SHA256)
        if schema_version == SCHEMA_VERSION
        else {
            "experiment": file_sha256(__file__),
            "regional": file_sha256(REPO_ROOT / "cogstate/features/regional.py"),
            "montage": file_sha256(REPO_ROOT / "cogstate/features/montage.py"),
        }
    )
    semantic = {
        "schema_version": schema_version,
        "experiment_id": config["experiment_id"],
        "implementation_sha256": implementation_sha256,
        "dataset": config["dataset"],
        "source_preprocessing_contract_hash": universe.source_contract_hash,
        "sample_identity_hash": _sample_hash(universe.manifest["sample_id"]),
        "subject_fold_identity_hash": stable_hash(
            universe.manifest[["sample_id", "subject_id", "record_group_id", "outer_fold"]]
            .astype(str)
            .to_dict("records")
        ),
        "profile_registry_hash": registry["profile_registry_hash"],
        "regional_schema_hash": pipeline.schema_hash(),
        "feature_names_hash": stable_hash(pipeline.feature_names()),
        "model": config["model"],
        "targets": list(config["targets"]),
        "evaluation": config["evaluation"],
        "run_matrix_hash": run_matrix_hash,
        "fold_split_hashes": {
            fold: payload["test_sample_hash"]
            for fold, payload in fold_audit["folds"].items()
        },
        "q3_transform_hashes": {
            key: payload["transform_hash"] for key, payload in transforms.items()
        },
    }
    protocol_hash = stable_hash(semantic)
    plan_hash = stable_hash(
        {
            "protocol_hash": protocol_hash,
            "training_units": [asdict(spec) for spec in specs],
            "evaluation_profiles": list(PROFILE_ORDER),
        }
    )
    rows = len(universe.manifest)
    estimated_bytes = (
        rows
        * len(PROFILE_ORDER)
        * expected_width
        * np.dtype(np.float32).itemsize
    )
    return {
        "schema_version": schema_version,
        "experiment_id": config["experiment_id"],
        "result_status": config["result_status"],
        "analysis_status": "confirmatory_preregistered_not_complete",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "protocol_hash": protocol_hash,
        "plan_hash": plan_hash,
        "run_matrix_hash": run_matrix_hash,
        "protocol_semantic": semantic,
        "profile_registry": registry,
        "fold_audit": fold_audit,
        "target_fold_audit": target_audit.to_dict("records"),
        "q3_transforms": transforms,
        "sample_identity_audit": {
            "shared_index_for_all_profiles": True,
            "sample_ids_unique": bool(universe.manifest["sample_id"].is_unique),
            "rows": rows,
            "subjects": int(universe.manifest["subject_id"].nunique()),
            "record_group_ids": int(universe.manifest["record_group_id"].nunique()),
            "sample_id_hash": _sample_hash(universe.manifest["sample_id"]),
            "target_and_fold_identity_shared": True,
        },
        "expected_xgboost_trainings": 35,
        "expected_prediction_evaluations": 175,
        "expected_smoke_trainings": 7,
        "expected_smoke_evaluations": 35,
        "feature_width": expected_width,
        "estimated_cache_size_bytes": int(estimated_bytes),
        "estimated_cache_size_mib": float(estimated_bytes / 2**20),
    }


def write_plan(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    plan = protocol_plan(config_path)
    output = _repo_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "protocol_manifest.json", plan)
    _atomic_json(output / "montage_profile_manifest.json", plan["profile_registry"])
    _atomic_json(output / "q3_transform_manifests.json", plan["q3_transforms"])
    _write_csv(output / "coverage_audit.csv", pd.DataFrame(plan["profile_registry"]["profiles"]).drop(columns=["montage_manifest"]))
    _write_csv(output / "target_fold_audit.csv", pd.DataFrame(plan["target_fold_audit"]))
    matrix = pd.DataFrame(
        [
            {
                **asdict(spec),
                "run_id": spec.run_id,
                "training_profile": "full_14",
                "evaluation_profiles": ";".join(PROFILE_ORDER),
                "evaluation_count": len(PROFILE_ORDER),
                "specification_hash": run_specification_hash(
                    spec,
                    protocol_hash=plan["protocol_hash"],
                    cache_identity_hash="resolved_after_cache_materialization",
                ),
            }
            for spec in build_run_matrix(config)
        ]
    )
    _write_csv(output / "run_matrix.csv", matrix)
    dry = {
        key: plan[key]
        for key in (
            "experiment_id", "protocol_hash", "plan_hash", "feature_width",
            "expected_xgboost_trainings", "expected_prediction_evaluations",
            "expected_smoke_trainings", "expected_smoke_evaluations",
            "estimated_cache_size_bytes", "estimated_cache_size_mib",
            "sample_identity_audit",
        )
    }
    dry["profiles"] = [
        {
            key: row[key]
            for key in (
                "profile", "channel_count", "regions_present", "regions_absent",
                "constant_missing_region_feature_count", "schema_hash", "montage_hash",
            )
        }
        for row in plan["profile_registry"]["profiles"]
    ]
    _atomic_json(output / "dry_run_summary.json", dry)
    return plan


def feature_cache_identity(
    universe: SignalUniverse,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    pipeline = _pipeline(config)
    names = pipeline.feature_names()
    raw_hashes = sorted(
        universe.manifest["preprocessing_hash"].dropna().astype(str).unique()
    )
    identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "regional_schema_hash": pipeline.schema_hash(),
        "profile_registry_hash": plan["profile_registry"]["profile_registry_hash"],
        "profile_order": list(PROFILE_ORDER),
        "montage_hashes": {
            row["profile"]: row["montage_hash"]
            for row in plan["profile_registry"]["profiles"]
        },
        "source_preprocessing_contract_hash": universe.source_contract_hash,
        "raw_preprocessing_hashes": raw_hashes,
        "sample_identity_hash": _sample_hash(universe.manifest["sample_id"]),
        "feature_names_hash": stable_hash(names),
        "rows": int(len(universe.manifest)),
        "profile_count": len(PROFILE_ORDER),
        "feature_width": len(names),
        "dtype": "float32",
        "matrix_shape": [len(universe.manifest), len(PROFILE_ORDER), len(names)],
        "target_columns_present": False,
    }
    identity["cache_identity_hash"] = stable_hash(identity)
    return identity


def _transform_profile_batch(
    payload: tuple[
        np.ndarray,
        Mapping[str, Any],
        tuple[tuple[str, tuple[str, ...]], ...],
    ]
) -> np.ndarray:
    windows, regional_config, profile_items = payload
    pipeline = RegionalFeaturePipeline(
        _regional_feature_config(regional_config)
    )
    profiles = dict(profile_items)
    rows = []
    for window in windows:
        transformed = pipeline.transform_profiles(
            window,
            channel_names=EMOTIV_14_CHANNELS,
            profiles=profiles,
        )
        rows.append(np.stack([transformed[name] for name in PROFILE_ORDER]))
    return np.ascontiguousarray(np.stack(rows), dtype=np.float32)


def _feature_cache_dir(config: Mapping[str, Any]) -> Path:
    return _repo_path(config["output_dir"]) / "feature_cache"


def build_feature_cache(
    config_path: str | Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    plan = write_plan(config_path)
    universe = load_signal_universe(config)
    identity = feature_cache_identity(universe, config, plan)
    cache_dir = _feature_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / CACHE_MANIFEST_NAME
    matrix_path = cache_dir / FEATURE_MATRIX_NAME
    index_path = cache_dir / FEATURE_INDEX_NAME
    names_path = cache_dir / FEATURE_NAMES_NAME
    completed_rows = 0
    elapsed_before = 0.0
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(f"Feature cache exists; use resume: {cache_dir}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError("Existing regional cache identity is incompatible")
        completed_rows = int(existing.get("completed_rows", 0))
        elapsed_before = float(existing.get("elapsed_seconds", 0.0))
    else:
        index_columns = [
            "sample_id", "source", "subject_id", "record_id", "record_group_id",
            "t_start", "t_end", "outer_fold", "preprocessing_hash",
        ]
        universe.manifest[index_columns].to_parquet(index_path, index=False)
        _atomic_json(names_path, {"feature_names": _pipeline(config).feature_names()})
        matrix = np.lib.format.open_memmap(
            matrix_path,
            mode="w+",
            dtype=np.float32,
            shape=tuple(identity["matrix_shape"]),
        )
        matrix.flush()
        del matrix
        _atomic_json(
            manifest_path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "in_progress",
                "identity": identity,
                "completed_rows": 0,
                "elapsed_seconds": 0.0,
                "finite_value_audit": "pending",
            },
        )
    for required in (matrix_path, index_path, names_path):
        if not required.is_file():
            raise FileNotFoundError(f"Regional feature cache is missing {required}")
    matrix = np.lib.format.open_memmap(matrix_path, mode="r+")
    if matrix.shape != tuple(identity["matrix_shape"]) or matrix.dtype != np.float32:
        raise ValueError("Regional feature matrix shape/dtype is incompatible")
    if not 0 <= completed_rows <= len(universe.manifest):
        raise ValueError("Invalid regional feature cache completed_rows")
    chunk_size = int(config["cache"]["chunk_size"])
    workers = int(config["cache"]["workers"])
    if chunk_size <= 0 or workers <= 0:
        raise ValueError("cache chunk_size/workers must be positive")
    profile_items = tuple((name, MONTAGE_PROFILES[name]) for name in PROFILE_ORDER)
    started = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(completed_rows, len(universe.manifest), chunk_size):
            stop = min(start + chunk_size, len(universe.manifest))
            windows = np.ascontiguousarray(
                np.stack([universe.data.data[index][0].T for index in range(start, stop)]),
                dtype=np.float32,
            )
            chunks = [
                chunk for chunk in np.array_split(windows, min(workers, len(windows)))
                if len(chunk)
            ]
            payloads = [
                (chunk, dict(config["regional_features"]), profile_items)
                for chunk in chunks
            ]
            values = (
                _transform_profile_batch(payloads[0])
                if executor is None
                else np.concatenate(list(executor.map(_transform_profile_batch, payloads)))
            )
            expected_shape = (
                stop - start,
                len(PROFILE_ORDER),
                int(config["regional_features"]["feature_width"]),
            )
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise RuntimeError(
                    f"Regional feature chunk invalid: {values.shape}, finite="
                    f"{np.isfinite(values).all()}"
                )
            matrix[start:stop] = values
            matrix.flush()
            completed_rows = stop
            _atomic_json(
                manifest_path,
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "status": "in_progress" if stop < len(universe.manifest) else "complete",
                    "identity": identity,
                    "completed_rows": stop,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                    "finite_value_audit": "pending" if stop < len(universe.manifest) else True,
                },
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        matrix.flush()
        del matrix
    matrix, index, names, manifest = load_feature_cache(config)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Completed regional feature cache contains NaN or Inf")
    summary = {
        "status": "complete",
        "cache_identity_hash": identity["cache_identity_hash"],
        "rows": len(index),
        "profiles": len(PROFILE_ORDER),
        "feature_width": len(names),
        "matrix_size_bytes": matrix_path.stat().st_size,
        "finite_values": True,
        "sample_ids_unique": bool(index["sample_id"].is_unique),
        "elapsed_seconds": float(manifest["elapsed_seconds"]),
    }
    _atomic_json(cache_dir / "feature_cache_summary.json", summary)
    return summary


def load_feature_cache(
    config: Mapping[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, list[str], dict[str, Any]]:
    cache_dir = _feature_cache_dir(config)
    manifest = json.loads((cache_dir / CACHE_MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Regional feature cache is not complete")
    matrix = np.load(cache_dir / FEATURE_MATRIX_NAME, mmap_mode="r", allow_pickle=False)
    index = pd.read_parquet(cache_dir / FEATURE_INDEX_NAME)
    names = json.loads((cache_dir / FEATURE_NAMES_NAME).read_text(encoding="utf-8"))[
        "feature_names"
    ]
    identity = manifest["identity"]
    if tuple(matrix.shape) != tuple(identity["matrix_shape"]):
        raise ValueError("Regional cache matrix shape changed")
    if len(index) != len(matrix) or len(names) != matrix.shape[2]:
        raise ValueError("Regional cache index/names are misaligned")
    if index["sample_id"].duplicated().any():
        raise ValueError("Regional cache contains duplicate sample_id")
    if _sample_hash(index["sample_id"]) != identity["sample_identity_hash"]:
        raise ValueError("Regional cache sample identity hash mismatch")
    return matrix, index, [str(name) for name in names], manifest


def run_specification_hash(
    spec: MontageTrainingSpec,
    *,
    protocol_hash: str,
    cache_identity_hash: str,
) -> str:
    return stable_hash(
        {
            "run_spec": asdict(spec),
            "protocol_hash": protocol_hash,
            "cache_identity_hash": cache_identity_hash,
            "training_profile": "full_14",
            "evaluation_profiles": list(PROFILE_ORDER),
        }
    )


def resumable_summary(
    summary_path: Path,
    *,
    specification_hash: str,
) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        return None
    if summary.get("specification_hash") != specification_hash:
        return None
    required = [Path(path) for path in summary.get("required_artifacts", [])]
    return summary if required and all(path.is_file() for path in required) else None


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    labels = [0, 1, 2]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
    }


def _participant_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for subject, group in predictions.groupby("subject_id", sort=True):
        rows.append(
            {
                "subject_id": str(subject),
                "n_samples": int(len(group)),
                **_classification_metrics(
                    group["y_true"].to_numpy(dtype=int),
                    group["y_pred"].to_numpy(dtype=int),
                ),
            }
        )
    frame = pd.DataFrame(rows)
    macro = {
        metric: float(frame[metric].mean())
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    }
    return frame, macro


def fit_full_and_evaluate_profiles(
    model: Any,
    X_train_full: np.ndarray,
    y_train: np.ndarray,
    X_test_by_profile: Mapping[str, np.ndarray],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str]:
    """Fit once on full_14 and use the unchanged model for all profiles."""
    model.fit(X_train_full, y_train)
    booster_hash = xgboost_state_sha256(model)
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for profile in PROFILE_ORDER:
        before = xgboost_state_sha256(model)
        y_pred = np.asarray(model.predict(X_test_by_profile[profile]), dtype=int)
        proba = np.asarray(model.predict_proba(X_test_by_profile[profile]), dtype=float)
        after = xgboost_state_sha256(model)
        if before != booster_hash or after != booster_hash:
            raise RuntimeError("XGBoost booster changed during montage evaluation")
        if proba.shape != (len(y_pred), 3) or not np.isfinite(proba).all():
            raise RuntimeError(f"Invalid probabilities for profile {profile}")
        if not np.allclose(proba.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Probabilities do not sum to one for {profile}")
        predictions[profile] = (y_pred, proba)
    return predictions, booster_hash


def audit_prediction_identity(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    reference = frames["full_14"]
    columns = ["sample_id", "subject_id", "record_group_id", "outer_fold", "y_true"]
    mismatches: dict[str, list[str]] = {}
    for profile in PROFILE_ORDER:
        current = frames[profile]
        changed = [
            column
            for column in columns
            if not current[column].reset_index(drop=True).equals(
                reference[column].reset_index(drop=True)
            )
        ]
        if changed:
            mismatches[profile] = changed
    if mismatches:
        raise RuntimeError(f"Montage prediction identity mismatch: {mismatches}")
    return {
        "exact_identity": True,
        "columns": columns,
        "profiles": list(PROFILE_ORDER),
        "rows_per_profile": len(reference),
        "sample_id_hash": _sample_hash(reference["sample_id"]),
    }


def _run_dir(config: Mapping[str, Any], profile: str, spec: MontageTrainingSpec) -> Path:
    return _repo_path(config["output_dir"]) / profile / "runs" / spec.run_id


def execute_training_unit(
    config_path: str | Path,
    spec: MontageTrainingSpec,
    *,
    smoke: bool,
    resume: bool,
    plan: Mapping[str, Any],
    universe: SignalUniverse,
    matrix: np.ndarray,
    index: pd.DataFrame,
    cache_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    config = load_config(config_path)
    execution_profile = "smoke" if smoke else "full"
    specification_hash = run_specification_hash(
        spec,
        protocol_hash=str(plan["protocol_hash"]),
        cache_identity_hash=str(cache_manifest["identity"]["cache_identity_hash"]),
    )
    run_dir = _run_dir(config, execution_profile, spec)
    summary_path = run_dir / "run_summary.json"
    if resume:
        existing = resumable_summary(
            summary_path, specification_hash=specification_hash
        )
        if existing is not None:
            return existing
    if not index["sample_id"].reset_index(drop=True).equals(
        universe.manifest["sample_id"].reset_index(drop=True)
    ):
        raise RuntimeError("Feature cache and canonical universe sample IDs differ")
    values = universe.targets[f"target_{spec.pm}"].to_numpy(dtype=np.float32)
    folds = index["outer_fold"].to_numpy(dtype=int)
    valid = np.isfinite(values)
    train = valid & (folds != spec.fold)
    test = valid & (folds == spec.fold)
    train_subjects = set(index.loc[train, "subject_id"].astype(str))
    test_subjects = set(index.loc[test, "subject_id"].astype(str))
    train_groups = set(index.loc[train, "record_group_id"].astype(str))
    test_groups = set(index.loc[test, "record_group_id"].astype(str))
    if train_subjects & test_subjects or train_groups & test_groups:
        raise RuntimeError("Outer subject/record-group leakage")
    transform = FoldLocalQuantileTargetTransform(3, duplicates="drop").fit(values[train])
    y_train = transform.transform(values[train]).astype(int)
    y_test = transform.transform(values[test]).astype(int)
    transform_manifest = build_target_transform_manifest(
        get_target_spec(f"pm_{spec.pm}_q3_fold_local"),
        transform,
        outer_fold=spec.fold,
        outer_train_sample_ids=index.loc[train, "sample_id"].to_numpy(),
        outer_train_targets=values[train],
    )
    expected_transform = plan["q3_transforms"][f"fold_{spec.fold:02d}__{spec.pm}"]
    if transform_manifest["transform_hash"] != expected_transform["transform_hash"]:
        raise RuntimeError("Runtime Q3 transform differs from dry plan")

    model = build_model(
        "xgboost",
        "classification",
        (matrix.shape[2],),
        3,
        config["model"]["params"],
    )
    X_train = np.asarray(matrix[train, PROFILE_ORDER.index("full_14")])
    X_test = {
        profile: np.asarray(matrix[test, profile_index])
        for profile_index, profile in enumerate(PROFILE_ORDER)
    }
    started = time.perf_counter()
    evaluated, booster_hash = fit_full_and_evaluate_profiles(
        model, X_train, y_train, X_test
    )
    elapsed = time.perf_counter() - started
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.ubj"
    model.save_model(model_path)
    _atomic_json(run_dir / "q3_transform.json", transform_manifest)

    frames: dict[str, pd.DataFrame] = {}
    profile_metrics: dict[str, Any] = {}
    participant_frames: list[pd.DataFrame] = []
    required_artifacts = [model_path, run_dir / "q3_transform.json"]
    test_metadata = index.loc[test].reset_index(drop=True)
    for profile in PROFILE_ORDER:
        y_pred, probabilities = evaluated[profile]
        frame = test_metadata[
            ["sample_id", "source", "subject_id", "record_id", "record_group_id", "outer_fold"]
        ].copy()
        frame.insert(1, "pm", spec.pm)
        frame.insert(2, "profile", profile)
        frame["y_true"] = y_test
        frame["y_pred"] = y_pred
        for class_id in range(3):
            frame[f"proba_{class_id}"] = probabilities[:, class_id]
        frames[profile] = frame
        participant, participant_macro = _participant_metrics(frame)
        participant.insert(0, "pm", spec.pm)
        participant.insert(1, "profile", profile)
        participant.insert(2, "outer_fold", spec.fold)
        participant_frames.append(participant)
        evaluation_dir = run_dir / "evaluations" / profile
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = evaluation_dir / "predictions.parquet"
        metrics_path = evaluation_dir / "metrics.json"
        frame.to_parquet(predictions_path, index=False)
        window_metrics = _classification_metrics(y_test, y_pred)
        profile_metrics[profile] = {
            "window": window_metrics,
            "participant_macro": participant_macro,
            "sample_id_hash": _sample_hash(frame["sample_id"]),
            "booster_hash": booster_hash,
        }
        _atomic_json(metrics_path, profile_metrics[profile])
        required_artifacts.extend([predictions_path, metrics_path])
    identity_audit = audit_prediction_identity(frames)
    participant_path = run_dir / "participant_metrics.csv"
    _write_csv(participant_path, pd.concat(participant_frames, ignore_index=True))
    required_artifacts.append(participant_path)
    reference = profile_metrics["full_14"]["participant_macro"]
    deltas = {
        profile: {
            metric: float(profile_metrics[profile]["participant_macro"][metric] - reference[metric])
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        }
        for profile in PROFILE_ORDER
    }
    summary = {
        "status": "complete",
        "result_status": "smoke" if smoke else config["result_status"],
        "run_id": spec.run_id,
        "specification_hash": specification_hash,
        "protocol_hash": plan["protocol_hash"],
        "plan_hash": plan["plan_hash"],
        **asdict(spec),
        "training_profile": "full_14",
        "evaluation_profiles": list(PROFILE_ORDER),
        "model_fit_count": 1,
        "evaluation_count": 5,
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        "subject_overlap": [],
        "record_group_overlap": [],
        "q3_transform_hash": transform_manifest["transform_hash"],
        "booster_hash": booster_hash,
        "training_and_evaluation_seconds": float(elapsed),
        "metrics": profile_metrics,
        "deltas_vs_full_14": deltas,
        "sample_identity_audit": identity_audit,
        "required_artifacts": [str(path) for path in required_artifacts],
    }
    _atomic_json(summary_path, summary)
    return summary


def aggregate_smoke(
    config: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    output = _repo_path(config["output_dir"])
    prediction_frames = []
    participant_frames = []
    for summary in summaries:
        run_dir = _run_dir(
            config,
            "smoke",
            MontageTrainingSpec(pm=str(summary["pm"]), fold=int(summary["fold"])),
        )
        participant_frames.append(pd.read_csv(run_dir / "participant_metrics.csv"))
        for profile in PROFILE_ORDER:
            prediction_frames.append(
                pd.read_parquet(run_dir / "evaluations" / profile / "predictions.parquet")
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    participants = pd.concat(participant_frames, ignore_index=True)
    predictions.to_parquet(output / "smoke_predictions.parquet", index=False)
    _write_csv(output / "smoke_participant_metrics.csv", participants)
    per_pm = (
        participants.groupby(["pm", "profile"], as_index=False, sort=True)[
            ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
        ]
        .mean()
    )
    _write_csv(output / "smoke_per_pm_metrics.csv", per_pm)
    subject_pm_macro = (
        participants.groupby(["subject_id", "profile"], as_index=False, sort=True)[
            ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
        ]
        .mean()
    )
    overall = (
        subject_pm_macro.groupby("profile", as_index=False, sort=True)[
            ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
        ]
        .mean()
    )
    reference = overall.loc[overall["profile"].eq("full_14")].iloc[0]
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        overall[f"delta_{metric}_vs_full_14"] = overall[metric] - float(reference[metric])
    overall["profile"] = pd.Categorical(overall["profile"], PROFILE_ORDER, ordered=True)
    overall = overall.sort_values("profile").reset_index(drop=True)
    overall["profile"] = overall["profile"].astype(str)
    _write_csv(output / "smoke_participant_macro_by_profile.csv", overall)
    all_identity = all(
        bool(summary["sample_identity_audit"]["exact_identity"])
        for summary in summaries
    )
    booster_reused = all(
        len({
            payload["booster_hash"]
            for payload in summary["metrics"].values()
        }) == 1
        for summary in summaries
    )
    result = {
        "status": "smoke_complete",
        "result_status": "smoke",
        "completed_xgboost_trainings": len(summaries),
        "completed_prediction_evaluations": len(summaries) * len(PROFILE_ORDER),
        "all_seven_pm": {str(summary["pm"]) for summary in summaries} == set(PM_METRICS),
        "outer_folds": sorted({int(summary["fold"]) for summary in summaries}),
        "one_fit_per_pm_fold": all(int(summary["model_fit_count"]) == 1 for summary in summaries),
        "same_booster_all_profiles": booster_reused,
        "exact_profile_sample_identity": all_identity,
        "participant_macro_by_profile": overall.to_dict("records"),
        "per_pm_metrics_path": _relative_path(output / "smoke_per_pm_metrics.csv"),
        "predictions_path": _relative_path(output / "smoke_predictions.parquet"),
    }
    _atomic_json(output / "smoke_summary.json", result)
    return result


def run_smoke(
    config_path: str | Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    plan = write_plan(config_path)
    cache_path = _feature_cache_dir(config) / CACHE_MANIFEST_NAME
    if not cache_path.is_file() or json.loads(cache_path.read_text(encoding="utf-8")).get("status") != "complete":
        build_feature_cache(config_path, resume=resume)
    matrix, index, _, cache_manifest = load_feature_cache(config)
    universe = load_signal_universe(config)
    specs = [
        spec
        for spec in build_run_matrix(config)
        if spec.fold in set(map(int, config["smoke"]["folds"]))
        and spec.pm in config["smoke"]["targets"]
    ]
    if len(specs) != 7:
        raise RuntimeError(f"Smoke must train seven XGBoost models, got {len(specs)}")
    summaries = [
        execute_training_unit(
            config_path,
            spec,
            smoke=True,
            resume=resume,
            plan=plan,
            universe=universe,
            matrix=matrix,
            index=index,
            cache_manifest=cache_manifest,
        )
        for spec in specs
    ]
    return aggregate_smoke(config, summaries)



def aggregate_full(
    config: Mapping[str, Any],
    runs: Sequence[tuple[Mapping[str, Any], str]],
) -> dict[str, Any]:
    """Aggregate the complete 5-fold execution.

    `execution_profile` is either ``smoke`` for validated fold-1 units
    reused from the smoke run or ``full`` for normal full-run units.
    """
    output = _repo_path(config["output_dir"])

    prediction_frames = []
    participant_frames = []

    for summary, execution_profile in runs:
        spec = MontageTrainingSpec(
            pm=str(summary["pm"]),
            fold=int(summary["fold"]),
        )
        run_dir = _run_dir(config, execution_profile, spec)

        participant_frames.append(
            pd.read_csv(run_dir / "participant_metrics.csv")
        )

        for profile in PROFILE_ORDER:
            prediction_frames.append(
                pd.read_parquet(
                    run_dir
                    / "evaluations"
                    / profile
                    / "predictions.parquet"
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    participants = pd.concat(participant_frames, ignore_index=True)

    predictions.to_parquet(
        output / "full_predictions.parquet",
        index=False,
    )
    _write_csv(
        output / "full_participant_metrics.csv",
        participants,
    )

    per_pm = (
        participants.groupby(
            ["pm", "profile"],
            as_index=False,
            sort=True,
        )[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ]
        .mean()
    )
    _write_csv(output / "full_per_pm_metrics.csv", per_pm)

    per_fold = (
        participants.groupby(
            ["outer_fold", "profile"],
            as_index=False,
            sort=True,
        )[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ]
        .mean()
    )
    _write_csv(output / "full_per_fold_metrics.csv", per_fold)

    # First average PMs within participant, then give every participant
    # equal weight in the final aggregate.
    subject_pm_macro = (
        participants.groupby(
            ["subject_id", "profile"],
            as_index=False,
            sort=True,
        )[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ]
        .mean()
    )

    overall = (
        subject_pm_macro.groupby(
            "profile",
            as_index=False,
            sort=True,
        )[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ]
        .mean()
    )

    reference = overall.loc[
        overall["profile"].eq("full_14")
    ].iloc[0]

    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ):
        overall[f"delta_{metric}_vs_full_14"] = (
            overall[metric] - float(reference[metric])
        )

    overall["profile"] = pd.Categorical(
        overall["profile"],
        PROFILE_ORDER,
        ordered=True,
    )
    overall = overall.sort_values("profile").reset_index(drop=True)
    overall["profile"] = overall["profile"].astype(str)

    _write_csv(
        output / "full_participant_macro_by_profile.csv",
        overall,
    )

    all_identity = all(
        bool(summary["sample_identity_audit"]["exact_identity"])
        for summary, _ in runs
    )

    booster_reused = all(
        len(
            {
                payload["booster_hash"]
                for payload in summary["metrics"].values()
            }
        )
        == 1
        for summary, _ in runs
    )

    folds = sorted(
        {int(summary["fold"]) for summary, _ in runs}
    )
    pms = {
        str(summary["pm"])
        for summary, _ in runs
    }

    smoke_units = sum(
        execution_profile == "smoke"
        for _, execution_profile in runs
    )
    full_units = sum(
        execution_profile == "full"
        for _, execution_profile in runs
    )

    result = {
        "status": "complete",
        "result_status": config["result_status"],
        "completed_xgboost_trainings": len(runs),
        "completed_prediction_evaluations": (
            len(runs) * len(PROFILE_ORDER)
        ),
        "all_seven_pm": pms == set(PM_METRICS),
        "outer_folds": folds,
        "one_fit_per_pm_fold": all(
            int(summary["model_fit_count"]) == 1
            for summary, _ in runs
        ),
        "same_booster_all_profiles": booster_reused,
        "exact_profile_sample_identity": all_identity,
        "reused_smoke_units": int(smoke_units),
        "full_execution_units": int(full_units),
        "participant_macro_by_profile": overall.to_dict("records"),
        "per_pm_metrics_path": _relative_path(
            output / "full_per_pm_metrics.csv"
        ),
        "per_fold_metrics_path": _relative_path(
            output / "full_per_fold_metrics.csv"
        ),
        "predictions_path": _relative_path(
            output / "full_predictions.parquet"
        ),
    }

    if len(runs) != 35:
        raise RuntimeError(
            f"Full execution must contain 35 PM×fold units, got {len(runs)}"
        )
    if folds != [1, 2, 3, 4, 5]:
        raise RuntimeError(
            f"Full execution must contain folds 1-5, got {folds}"
        )
    if pms != set(PM_METRICS):
        raise RuntimeError(
            "Full execution does not contain all seven PM"
        )

    _atomic_json(output / "full_summary.json", result)
    return result


def run_full(
    config_path: str | Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    plan = write_plan(config_path)

    cache_path = _feature_cache_dir(config) / CACHE_MANIFEST_NAME
    if (
        not cache_path.is_file()
        or json.loads(
            cache_path.read_text(encoding="utf-8")
        ).get("status")
        != "complete"
    ):
        build_feature_cache(config_path, resume=resume)

    matrix, index, _, cache_manifest = load_feature_cache(config)
    universe = load_signal_universe(config)

    specs = list(build_run_matrix(config))
    if len(specs) != 35:
        raise RuntimeError(
            f"Full run must contain 35 XGBoost units, got {len(specs)}"
        )

    smoke_folds = set(
        map(int, config["smoke"]["folds"])
    )
    smoke_targets = set(config["smoke"]["targets"])

    runs: list[tuple[Mapping[str, Any], str]] = []

    for spec in specs:
        # Fold-1 smoke is scientifically the same PM×fold specification.
        # Reuse it only if its specification/protocol/plan/cache identity
        # still matches the current execution exactly.
        if (
            resume
            and spec.fold in smoke_folds
            and spec.pm in smoke_targets
        ):
            specification_hash = run_specification_hash(
                spec,
                protocol_hash=str(plan["protocol_hash"]),
                cache_identity_hash=str(
                    cache_manifest["identity"]["cache_identity_hash"]
                ),
            )

            smoke_summary_path = (
                _run_dir(config, "smoke", spec)
                / "run_summary.json"
            )

            existing = resumable_summary(
                smoke_summary_path,
                specification_hash=specification_hash,
            )

            if (
                existing is not None
                and str(existing.get("protocol_hash"))
                == str(plan["protocol_hash"])
                and str(existing.get("plan_hash"))
                == str(plan["plan_hash"])
            ):
                runs.append((existing, "smoke"))
                continue

        summary = execute_training_unit(
            config_path,
            spec,
            smoke=False,
            resume=resume,
            plan=plan,
            universe=universe,
            matrix=matrix,
            index=index,
            cache_manifest=cache_manifest,
        )
        runs.append((summary, "full"))

    return aggregate_full(config, runs)

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/cross_montage/xgboost_regional_montage_transfer_v1.json",
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--plan-only", action="store_true")
    actions.add_argument("--build-cache", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.plan_only:
        result = write_plan(args.config)
        print(json.dumps({
            "protocol_hash": result["protocol_hash"],
            "plan_hash": result["plan_hash"],
            "expected_xgboost_trainings": result["expected_xgboost_trainings"],
            "expected_prediction_evaluations": result["expected_prediction_evaluations"],
            "expected_smoke_trainings": result["expected_smoke_trainings"],
            "expected_smoke_evaluations": result["expected_smoke_evaluations"],
            "estimated_cache_size_mib": result["estimated_cache_size_mib"],
            "profiles": result["profile_registry"]["profiles"],
        }, indent=2, ensure_ascii=False, default=str))
    elif args.build_cache:
        print(json.dumps(build_feature_cache(args.config, resume=args.resume), indent=2))
    elif args.smoke:
        print(json.dumps(run_smoke(args.config, resume=args.resume), indent=2, default=str))
    else:
        print(json.dumps(run_full(args.config, resume=args.resume), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
