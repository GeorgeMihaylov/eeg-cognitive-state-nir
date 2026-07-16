"""Factorial raw-EEG preprocessing ablation with resumable neutral trials."""

from __future__ import annotations

import csv
import io
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from bench.bench_runner import (
    BenchmarkRunner,
    CompletedBenchmarkRun,
    benchmark_config_hash,
)
from bench.datasets.raw_eeg_window_dataset import (
    CANONICAL_EEG_CHANNELS,
    RAW_LOADER_VERSION,
    _cache_config_hash,
    _valid_cache_shard,
    build_raw_eeg_cache,
    build_raw_window_index,
)
from bench.datasets.raw_preprocessing import (
    DEFAULT_FILTER_PADDING_SECONDS,
    PreprocessingSpec,
    normalize_raw_preprocessing,
    preprocessing_variant_name,
    raw_preprocessing_hash,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FACTOR_PATHS = (
    "preprocessing.bandpass.enabled",
    "preprocessing.notch.enabled",
    "preprocessing.car.enabled",
)
TRIAL_IDS = {
    (False, False, False): "A",
    (True, False, False): "B",
    (False, True, False): "C",
    (False, False, True): "D",
    (True, True, False): "E",
    (True, False, True): "F",
    (False, True, True): "G",
    (True, True, True): "H",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_experiment_spec(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one declarative experiment matrix."""
    spec_path = _repo_path(path)
    with open(spec_path, encoding="utf-8") as input_file:
        document = yaml.safe_load(input_file) or {}
    required = {
        "experiment",
        "cache",
        "dataset",
        "preprocessing",
        "model",
        "evaluation",
        "search_space",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Experiment specification is missing keys: {missing}")
    experiment = document["experiment"]
    if experiment.get("objective_metric") != "balanced_accuracy":
        raise ValueError("The preprocessing ablation objective must be balanced_accuracy")
    if experiment.get("objective_direction") != "maximize":
        raise ValueError("The preprocessing ablation objective direction must be maximize")
    unknown_factors = sorted(set(document["search_space"]) - set(FACTOR_PATHS))
    if unknown_factors:
        raise ValueError(f"Unsupported preprocessing search-space keys: {unknown_factors}")
    missing_factors = sorted(set(FACTOR_PATHS) - set(document["search_space"]))
    if missing_factors:
        raise ValueError(f"Missing factorial preprocessing keys: {missing_factors}")
    return document


def _set_nested(document: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    cursor = document
    for key in keys[:-1]:
        nested = cursor.setdefault(key, {})
        if not isinstance(nested, dict):
            raise ValueError(f"Cannot assign {dotted_path!r}: {key!r} is not a mapping")
        cursor = nested
    cursor[keys[-1]] = value


SUPPORTED_TRIAL_PARAMETERS = frozenset({
    *FACTOR_PATHS,
    "training.random_state",
    "training.max_epochs",
    "dataset.max_windows",
    "evaluation.folds",
})


def resolve_trial_config(
    base_config: Mapping[str, Any],
    trial_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve neutral dotted trial parameters into a benchmark config.

    This is the single parameter entry point for matrix execution, the CLI and
    future optimizers. Trial IDs are deliberately absent from the API.
    """
    unknown = sorted(set(trial_parameters) - SUPPORTED_TRIAL_PARAMETERS)
    if unknown:
        raise ValueError(
            f"Unsupported trial parameters: {unknown}; available="
            f"{sorted(SUPPORTED_TRIAL_PARAMETERS)}"
        )
    resolved = deepcopy(dict(base_config))
    preprocessing = resolved.get("raw_preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("base_config.raw_preprocessing must be a mapping")

    for path in FACTOR_PATHS[:2]:
        if path not in trial_parameters:
            continue
        value = trial_parameters[path]
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be boolean")
        component = path.split(".")[1]
        preprocessing.setdefault(component, {})["enabled"] = value

    car_path = FACTOR_PATHS[2]
    if car_path in trial_parameters:
        value = trial_parameters[car_path]
        if not isinstance(value, bool):
            raise ValueError(f"{car_path} must be boolean")
        preprocessing.setdefault("rereference", {})["mode"] = (
            "common_average" if value else "none"
        )

    if "training.random_state" in trial_parameters:
        seed = int(trial_parameters["training.random_state"])
        for model in resolved.get("models", {}).values():
            model.setdefault("params", {})["random_state"] = seed
        resolved.setdefault("validation", {})["random_state"] = seed
        resolved.setdefault("evaluation", {})["random_state"] = seed
        resolved.setdefault("task_config", {})["random_state"] = seed

    if "training.max_epochs" in trial_parameters:
        max_epochs = int(trial_parameters["training.max_epochs"])
        if max_epochs < 1:
            raise ValueError("training.max_epochs must be positive")
        for model in resolved.get("models", {}).values():
            model.setdefault("params", {})["max_epochs"] = max_epochs

    if "dataset.max_windows" in trial_parameters:
        max_windows = int(trial_parameters["dataset.max_windows"])
        if max_windows < 1:
            raise ValueError("dataset.max_windows must be positive")
        for dataset in resolved.get("datasets", {}).values():
            dataset["max_windows"] = max_windows

    if "evaluation.folds" in trial_parameters:
        folds = [int(value) for value in trial_parameters["evaluation.folds"]]
        if not folds or any(value < 1 for value in folds):
            raise ValueError("evaluation.folds must contain positive fold IDs")
        if len(folds) != len(set(folds)):
            raise ValueError("evaluation.folds must not contain duplicates")
        resolved.setdefault("evaluation", {})["folds"] = folds

    try:
        return json.loads(json.dumps(resolved, default=str, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Resolved benchmark config is not serializable: {exc}") from exc


@dataclass(frozen=True)
class ExperimentTrial:
    trial_id: str
    parameters: tuple[tuple[str, bool], ...]
    preprocessing: PreprocessingSpec
    preprocessing_hash: str
    legacy_preprocessing_hash: str
    cache_key_hash: str

    def parameter_dict(self) -> dict[str, bool]:
        return dict(self.parameters)


def expand_factorial_trials(document: Mapping[str, Any]) -> list[ExperimentTrial]:
    """Expand the declared Cartesian product into deterministic trials A--H."""
    search_space = document["search_space"]
    values: list[list[bool]] = []
    for factor_path in FACTOR_PATHS:
        entry = search_space[factor_path]
        if not isinstance(entry, Mapping) or "values" not in entry:
            raise ValueError(f"search_space.{factor_path} must contain values")
        factor_values = [bool(value) for value in entry["values"]]
        if set(factor_values) != {False, True} or len(factor_values) != 2:
            raise ValueError(f"{factor_path} must define exactly [false, true]")
        values.append(factor_values)

    channels = tuple(document["cache"].get("channel_order", CANONICAL_EEG_CHANNELS))
    loader_version = str(
        document["cache"].get("loader_schema_version", RAW_LOADER_VERSION)
    )
    source_identity = _source_configuration_identity(document["cache"])
    trials: list[ExperimentTrial] = []
    for combination in product(*values):
        parameter_values = tuple(bool(value) for value in combination)
        trial_id = TRIAL_IDS[parameter_values]
        preprocessing_document = deepcopy(document["preprocessing"])
        for path, value in zip(FACTOR_PATHS, parameter_values):
            local_path = path.removeprefix("preprocessing.")
            _set_nested(preprocessing_document, local_path, bool(value))
        preprocessing = PreprocessingSpec.from_dict(preprocessing_document)
        preprocessing.assert_global_cacheable()
        legacy = preprocessing.to_legacy_raw_preprocessing()
        semantic_hash = preprocessing.stable_hash(
            channels=channels,
            loader_schema_version=loader_version,
        )
        cache_key_hash = preprocessing.stable_hash(
            channels=channels,
            loader_schema_version=loader_version,
            source_identity=source_identity,
        )
        trials.append(
            ExperimentTrial(
                trial_id=trial_id,
                parameters=tuple(zip(FACTOR_PATHS, parameter_values)),
                preprocessing=preprocessing,
                preprocessing_hash=semantic_hash,
                legacy_preprocessing_hash=raw_preprocessing_hash(
                    legacy,
                    channels=channels,
                    default_resample_hz=preprocessing.target_sampling_rate,
                ),
                cache_key_hash=cache_key_hash,
            )
        )
    trials.sort(key=lambda trial: trial.trial_id)
    if [trial.trial_id for trial in trials] != list("ABCDEFGH"):
        raise RuntimeError("Factorial expansion did not produce trials A--H")
    if len({trial.preprocessing_hash for trial in trials}) != 8:
        raise RuntimeError("All factorial combinations must have unique stable hashes")
    return trials


def _source_configuration_identity(cache_config: Mapping[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for key in ("processed_path", "catalog_path", "audit_schema_path"):
        configured = cache_config.get(key)
        if configured is None:
            continue
        path = _repo_path(configured)
        item: dict[str, Any] = {"path": _relative_or_absolute(path)}
        if path.exists():
            stat = path.stat()
            item.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        identity[key] = item
    return identity


@dataclass(frozen=True)
class CacheResolution:
    exists: bool
    complete: bool
    reusable: bool
    index_path: Path
    cache_path: Path
    size_bytes: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "complete": self.complete,
            "reusable": self.reusable,
            "index_path": _relative_or_absolute(self.index_path),
            "cache_path": _relative_or_absolute(self.cache_path),
            "size_bytes": int(self.size_bytes),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrialPlan:
    trial: ExperimentTrial
    seed: int
    cache: CacheResolution
    existing_benchmark_result: str | None
    action: str
    estimated_new_cache_size: int
    reference_path: Path
    benchmark_output_dir: Path
    legacy_output_dir: Path
    resolved_config: Mapping[str, Any]
    config_hash: str
    completed_run: CompletedBenchmarkRun | None
    run_mode: str
    fold_limit: int | None = None
    max_windows: int | None = None
    max_epochs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial.trial_id,
            "parameters": self.trial.parameter_dict(),
            "preprocessing_hash": self.trial.preprocessing_hash,
            "legacy_preprocessing_hash": self.trial.legacy_preprocessing_hash,
            "cache_key_hash": self.trial.cache_key_hash,
            "cache_path": _relative_or_absolute(self.cache.cache_path),
            "cache_index_path": _relative_or_absolute(self.cache.index_path),
            "cache_exists": self.cache.exists,
            "cache_complete": self.cache.complete,
            "cache_reusable": self.cache.reusable,
            "cache_reason": self.cache.reason,
            "estimated_new_cache_size": int(self.estimated_new_cache_size),
            "existing_benchmark_result": self.existing_benchmark_result,
            "action": self.action,
            "model_name": "torch_shallow_convnet",
            "run_mode": self.run_mode,
            "seed": int(self.seed),
            "config_hash": self.config_hash,
            "fold_protocol": "group_kfold_subject",
            "objective_metric": "balanced_accuracy",
            "objective_direction": "maximize",
            "reference_path": _relative_or_absolute(self.reference_path),
            "benchmark_output_dir": _relative_or_absolute(
                self.benchmark_output_dir
            ),
            "legacy_output_dir": _relative_or_absolute(self.legacy_output_dir),
            "completed_run_directory": (
                None
                if self.completed_run is None
                else str(self.completed_run.run_directory)
            ),
            "fold_limit": self.fold_limit,
            "max_windows": self.max_windows,
            "max_epochs": self.max_epochs,
            "resolved_config": deepcopy(dict(self.resolved_config)),
        }


def _candidate_index_paths(
    document: Mapping[str, Any], extra: Iterable[Path] = ()
) -> list[Path]:
    configured = [
        _repo_path(path)
        for path in document["cache"].get("candidate_index_paths", [])
    ]
    index_dir = _repo_path(document["cache"]["index_dir"])
    generated = sorted(index_dir.glob("*.parquet")) if index_dir.exists() else []
    unique: dict[str, Path] = {}
    for path in [*configured, *generated, *extra]:
        unique[str(path.resolve())] = path
    return list(unique.values())


def _validate_semantic_cache(
    index_path: Path,
    trial: ExperimentTrial,
    document: Mapping[str, Any],
) -> CacheResolution | None:
    if not index_path.exists():
        return None
    frame = pd.read_parquet(index_path)
    required = {
        "record_id",
        "sample_id",
        "t_start",
        "t_end",
        "status",
        "cache_file",
        "sfreq_target",
        "preprocessing_hash",
    }
    if not required.issubset(frame.columns):
        return None
    hashes = sorted(frame["preprocessing_hash"].dropna().astype(str).unique())
    if hashes != [trial.legacy_preprocessing_hash]:
        return None
    accepted = frame.loc[frame["status"].astype(str).eq("ok")]
    if accepted.empty:
        return CacheResolution(
            True, False, False, index_path, index_path.parent, 0,
            "manifest has no accepted windows",
        )
    cache_files = sorted({
        _repo_path(value) for value in accepted["cache_file"].astype(str)
    }, key=str)
    cache_roots = {path.parent.resolve() for path in cache_files}
    if len(cache_roots) != 1:
        return CacheResolution(
            True, False, False, index_path, index_path.parent, 0,
            "accepted rows reference multiple cache roots",
        )
    cache_root = Path(next(iter(cache_roots)))
    legacy = trial.preprocessing.to_legacy_raw_preprocessing()
    channels = tuple(document["cache"].get("channel_order", CANONICAL_EEG_CHANNELS))
    max_missing_fraction = float(
        document["cache"].get("max_missing_fraction", 0.02)
    )
    target_sfreq = float(trial.preprocessing.target_sampling_rate)
    metadata_paths = sorted(cache_root.glob("*.json"))
    if not metadata_paths:
        return CacheResolution(
            True, False, False, index_path, cache_root, 0,
            "cache root has no shard metadata",
        )
    records_in_index = set(frame["record_id"].astype(str))
    records_in_cache: set[str] = set()
    cached_sample_ids: set[int] = set()
    try:
        for metadata_path in metadata_paths:
            with open(metadata_path, encoding="utf-8") as input_file:
                metadata = json.load(input_file)
            record_id = str(metadata["record_id"])
            records_in_cache.add(record_id)
            if metadata.get("loader_version") != RAW_LOADER_VERSION:
                raise ValueError("loader schema version mismatch")
            if metadata.get("preprocessing_hash") != trial.legacy_preprocessing_hash:
                raise ValueError("shard preprocessing hash mismatch")
            if normalize_raw_preprocessing(metadata.get("raw_preprocessing")) != legacy:
                raise ValueError("shard preprocessing metadata mismatch")
            record_rows = frame.loc[frame["record_id"].astype(str).eq(record_id)]
            if record_rows.empty:
                raise ValueError(f"shard record {record_id!r} is absent from index")
            window_results = metadata.get("window_results", [])
            shard_sample_ids = {
                int(item["sample_id"]) for item in window_results
            }
            cached_sample_ids.update(
                int(item["sample_id"])
                for item in window_results
                if item.get("status") == "ok"
            )
            hash_rows = record_rows.loc[
                record_rows["sample_id"].astype(int).isin(shard_sample_ids)
            ]
            if len(hash_rows) != len(shard_sample_ids):
                raise ValueError(
                    f"shard {record_id!r} references sample IDs absent from index"
                )
            raw_path = Path(metadata["raw_file_path"])
            if not raw_path.exists():
                raise ValueError(f"raw source is missing: {raw_path}")
            record_payload = {
                "record_id": record_id,
                "windows": hash_rows[["sample_id", "t_start", "t_end"]].to_dict(
                    "records"
                ),
            }
            expected_hash = _cache_config_hash(
                record_payload,
                raw_path,
                channels,
                target_sfreq,
                max_missing_fraction,
                legacy,
            )
            duration = float(
                (record_rows["t_end"] - record_rows["t_start"]).iloc[0]
            )
            target_count = int(round(duration * target_sfreq))
            array_path = metadata_path.with_suffix(".npy")
            if _valid_cache_shard(
                array_path,
                metadata_path,
                expected_hash,
                (len(channels), target_count),
            ) is None:
                raise ValueError(f"invalid or incomplete shard {array_path.name}")
        if records_in_cache != records_in_index:
            missing = sorted(records_in_index - records_in_cache)
            raise ValueError(f"cache is missing record shards: {missing[:5]}")
        accepted_sample_ids = set(accepted["sample_id"].astype(int))
        if cached_sample_ids != accepted_sample_ids:
            missing = sorted(accepted_sample_ids - cached_sample_ids)
            extra = sorted(cached_sample_ids - accepted_sample_ids)
            raise ValueError(
                "accepted sample IDs do not match cache metadata: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        size = sum(path.stat().st_size for path in cache_root.glob("*") if path.is_file())
        return CacheResolution(
            True, False, False, index_path, cache_root, size, str(exc)
        )
    size = sum(path.stat().st_size for path in cache_root.glob("*") if path.is_file())
    return CacheResolution(
        True,
        True,
        True,
        index_path,
        cache_root,
        size,
        "semantic preprocessing, source identity, shard hash, shape and dtype match",
    )


def _planned_cache_resolution(
    trial: ExperimentTrial, document: Mapping[str, Any]
) -> CacheResolution:
    cache_config = document["cache"]
    cache_root = _repo_path(cache_config["cache_dir"])
    variant = preprocessing_variant_name(
        trial.preprocessing.to_legacy_raw_preprocessing()
    )
    planned_root = cache_root / f"{variant}-{trial.legacy_preprocessing_hash[:16]}"
    index_dir = _repo_path(cache_config["index_dir"])
    planned_index = index_dir / (
        f"raw_eeg_window_index_w10_{trial.trial_id.lower()}_"
        f"{trial.legacy_preprocessing_hash[:16]}.parquet"
    )
    return CacheResolution(
        False,
        False,
        False,
        planned_index,
        planned_root,
        0,
        "no semantically compatible complete cache was found",
    )


def resolve_cache(
    trial: ExperimentTrial,
    document: Mapping[str, Any],
    *,
    extra_candidates: Iterable[Path] = (),
) -> CacheResolution:
    for path in _candidate_index_paths(document, extra_candidates):
        resolution = _validate_semantic_cache(path, trial, document)
        if resolution is not None and resolution.reusable:
            return resolution
    return _planned_cache_resolution(trial, document)


class PreprocessingAblation:
    """Plan, build, execute, and resume the declared preprocessing trials."""

    def __init__(self, spec_path: str | Path):
        self.spec_path = _repo_path(spec_path)
        self.document = load_experiment_spec(self.spec_path)
        self.trials = expand_factorial_trials(self.document)

    def select_trials(self, trial_ids: Sequence[str] | None) -> list[ExperimentTrial]:
        if not trial_ids:
            return list(self.trials)
        requested = {str(value).strip().upper() for value in trial_ids}
        unknown = sorted(requested - set("ABCDEFGH"))
        if unknown:
            raise ValueError(f"Unknown trial IDs: {unknown}; available: A--H")
        return [trial for trial in self.trials if trial.trial_id in requested]

    def _profile_name(
        self,
        fold_limit: int | None,
        max_windows: int | None,
        max_epochs: int | None,
    ) -> str:
        return "smoke" if any(
            value is not None for value in (fold_limit, max_windows, max_epochs)
        ) else "full"

    def _reference_path(
        self,
        trial: ExperimentTrial,
        seed: int,
        fold_limit: int | None,
        max_windows: int | None,
        max_epochs: int | None,
    ) -> Path:
        root = _repo_path(self.document["experiment"]["output_dir"])
        profile = self._profile_name(fold_limit, max_windows, max_epochs)
        return (
            root / "references" / profile
            / f"trial_{trial.trial_id}" / f"seed_{seed}"
        )

    def _legacy_output_path(
        self,
        trial: ExperimentTrial,
        seed: int,
        run_mode: str,
    ) -> Path:
        root = _repo_path(self.document["experiment"]["output_dir"])
        return root / run_mode / f"trial_{trial.trial_id}" / f"seed_{seed}"

    def _base_benchmark_config(
        self,
        cache: CacheResolution,
    ) -> dict[str, Any]:
        dataset = self.document["dataset"]
        model = self.document["model"]
        return {
            "output_dir": str(
                _repo_path(self.document["experiment"]["output_dir"])
                / "runs"
            ),
            "raw_preprocessing": PreprocessingSpec.from_dict(
                self.document["preprocessing"]
            ).to_legacy_raw_preprocessing(),
            "datasets": {
                dataset["name"]: {
                    "data_path": str(cache.index_path),
                    "target_col": dataset["target"],
                    "dataset_mode": dataset["mode"],
                    "logical_recording_map_path": str(
                        _repo_path(dataset["logical_recording_map_path"])
                    ),
                }
            },
            "tasks": [dataset["task"]],
            "validation": deepcopy(self.document["validation"]),
            "models": {
                model["name"]: {
                    "type": model["name"],
                    "task_type": "classification",
                    "params": deepcopy(model["params"]),
                }
            },
            "evaluation": deepcopy(self.document["evaluation"]),
            "task_config": {
                "random_state": int(model["params"].get("random_state", 42))
            },
            "run_within_subject": False,
            "run_loso": False,
        }

    @staticmethod
    def _trial_parameters(
        trial: ExperimentTrial,
        *,
        seed: int,
        fold_limit: int | None,
        max_windows: int | None,
        max_epochs: int | None,
    ) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            **trial.parameter_dict(),
            "training.random_state": int(seed),
        }
        if fold_limit is not None:
            if fold_limit < 1:
                raise ValueError("fold_limit must be positive")
            parameters["evaluation.folds"] = list(range(1, fold_limit + 1))
        if max_windows is not None:
            parameters["dataset.max_windows"] = int(max_windows)
        if max_epochs is not None:
            parameters["training.max_epochs"] = int(max_epochs)
        return parameters

    def plan(
        self,
        *,
        trial_ids: Sequence[str] | None = None,
        seed: int = 42,
        fold_limit: int | None = None,
        max_windows: int | None = None,
        max_epochs: int | None = None,
    ) -> list[TrialPlan]:
        estimated_size = int(
            self.document["cache"].get("estimated_cache_size_bytes", 0)
        )
        plans = []
        for trial in self.select_trials(trial_ids):
            cache = resolve_cache(trial, self.document)
            run_mode = self._profile_name(fold_limit, max_windows, max_epochs)
            parameters = self._trial_parameters(
                trial,
                seed=seed,
                fold_limit=fold_limit,
                max_windows=max_windows,
                max_epochs=max_epochs,
            )
            resolved_config = resolve_trial_config(
                self._base_benchmark_config(cache), parameters
            )
            config_hash = benchmark_config_hash(resolved_config)
            benchmark_output_dir = (
                _repo_path(self.document["experiment"]["output_dir"])
                / "runs" / config_hash[:20]
            )
            resolved_config["output_dir"] = str(benchmark_output_dir)
            reference_path = self._reference_path(
                trial, seed, fold_limit, max_windows, max_epochs
            )
            legacy_output_dir = self._legacy_output_path(
                trial, seed, run_mode
            )
            completed_run = BenchmarkRunner.find_completed_run(
                resolved_config,
                search_directories=[benchmark_output_dir, legacy_output_dir],
            )
            existing = (
                None
                if completed_run is None
                else str(completed_run.result_file)
            )
            if completed_run is not None:
                action = "skip_completed"
            elif cache.reusable:
                action = "reuse_cache_and_run"
            else:
                action = "build_cache_and_run"
            plans.append(
                TrialPlan(
                    trial=trial,
                    seed=seed,
                    cache=cache,
                    existing_benchmark_result=existing,
                    action=action,
                    estimated_new_cache_size=0 if cache.reusable else estimated_size,
                    reference_path=reference_path,
                    benchmark_output_dir=benchmark_output_dir,
                    legacy_output_dir=legacy_output_dir,
                    resolved_config=resolved_config,
                    config_hash=config_hash,
                    completed_run=completed_run,
                    run_mode=run_mode,
                    fold_limit=fold_limit,
                    max_windows=max_windows,
                    max_epochs=max_epochs,
                )
            )
        return plans

    def assert_disk_capacity(self, plans: Sequence[TrialPlan]) -> dict[str, int]:
        required = sum(
            plan.estimated_new_cache_size
            for plan in plans
            if not plan.cache.reusable
        )
        reserve = int(
            self.document["cache"].get("minimum_free_reserve_bytes", 20 * 2**30)
        )
        cache_root = _repo_path(self.document["cache"]["cache_dir"])
        usage = shutil.disk_usage(cache_root.anchor or cache_root)
        if usage.free < required + reserve:
            raise RuntimeError(
                "Insufficient free space for missing preprocessing caches: "
                f"free={usage.free}, estimated_new={required}, reserve={reserve}"
            )
        return {
            "free_bytes": int(usage.free),
            "estimated_new_cache_bytes": int(required),
            "minimum_reserve_bytes": int(reserve),
        }

    def build_cache(self, plan: TrialPlan) -> CacheResolution:
        if plan.cache.reusable:
            return plan.cache
        preprocessing = plan.trial.preprocessing
        preprocessing.assert_global_cacheable()
        if (
            preprocessing.effective_padding_seconds
            and preprocessing.padding_seconds != DEFAULT_FILTER_PADDING_SECONDS
        ):
            raise ValueError(
                "The established raw loader requires two-second filter padding"
            )
        cache_config = self.document["cache"]
        evaluation = self.document["evaluation"]
        index, _ = build_raw_window_index(
            _repo_path(cache_config["processed_path"]),
            _repo_path(cache_config["catalog_path"]),
            audit_schema_path=_repo_path(cache_config["audit_schema_path"]),
            target_sfreq=preprocessing.target_sampling_rate,
            n_splits=int(evaluation["n_splits"]),
        )
        index, cache_stats = build_raw_eeg_cache(
            index,
            _repo_path(cache_config["cache_dir"]),
            channels=tuple(cache_config.get("channel_order", CANONICAL_EEG_CHANNELS)),
            target_sfreq=preprocessing.target_sampling_rate,
            max_missing_fraction=float(
                cache_config.get("max_missing_fraction", 0.02)
            ),
            repo_root=REPO_ROOT,
            raw_preprocessing=preprocessing.to_legacy_raw_preprocessing(),
        )
        plan.cache.index_path.parent.mkdir(parents=True, exist_ok=True)
        index.to_parquet(plan.cache.index_path, index=False)
        stats_path = plan.cache.index_path.with_suffix(".stats.json")
        stats = {
            "trial_id": plan.trial.trial_id,
            "preprocessing": preprocessing.to_dict(),
            "preprocessing_hash": plan.trial.preprocessing_hash,
            "legacy_preprocessing_hash": plan.trial.legacy_preprocessing_hash,
            "cache": cache_stats,
            "accepted_windows": int(index["status"].astype(str).eq("ok").sum()),
            "rejected_windows": int(index["status"].astype(str).ne("ok").sum()),
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        resolution = resolve_cache(
            plan.trial, self.document, extra_candidates=[plan.cache.index_path]
        )
        if not resolution.reusable:
            raise RuntimeError(
                f"Built cache did not pass semantic validation: {resolution.reason}"
            )
        return resolution

    def _write_trial_metadata(
        self,
        plan: TrialPlan,
        cache: CacheResolution,
        completed_run: CompletedBenchmarkRun | None = None,
    ) -> dict[str, Any] | None:
        """Write only the allowed experiment metadata and optional run pointer."""
        plan.reference_path.mkdir(parents=True, exist_ok=True)
        resolved = {
            "trial_id": plan.trial.trial_id,
            "trial_parameters": self._trial_parameters(
                plan.trial,
                seed=plan.seed,
                fold_limit=plan.fold_limit,
                max_windows=plan.max_windows,
                max_epochs=plan.max_epochs,
            ),
            "preprocessing_hash": plan.trial.preprocessing_hash,
            "legacy_preprocessing_hash": plan.trial.legacy_preprocessing_hash,
            "cache": cache.to_dict(),
            "run_mode": plan.run_mode,
            "config_hash": plan.config_hash,
            "benchmark_config": deepcopy(dict(plan.resolved_config)),
        }
        with open(
            plan.reference_path / "resolved_trial.yaml",
            "w",
            encoding="utf-8",
        ) as output:
            yaml.safe_dump(resolved, output, sort_keys=False)
        if completed_run is None:
            return None
        reference = {
            "trial_id": plan.trial.trial_id,
            "preprocessing_hash": plan.trial.preprocessing_hash,
            "seed": int(plan.seed),
            "config_hash": plan.config_hash,
            **completed_run.to_dict(),
        }
        (plan.reference_path / "trial_reference.json").write_text(
            json.dumps(reference, indent=2), encoding="utf-8"
        )
        return reference

    def run_trial(
        self,
        plan: TrialPlan,
        *,
        build_missing_cache: bool,
        resume: bool,
    ) -> dict[str, Any]:
        if not plan.cache.reusable and not build_missing_cache:
            raise FileNotFoundError(
                f"Trial {plan.trial.trial_id} has no reusable cache; "
                "pass --build-missing-caches"
            )
        cache = self.build_cache(plan) if not plan.cache.reusable else plan.cache
        completed = BenchmarkRunner.find_completed_run(
            plan.resolved_config,
            search_directories=[
                plan.benchmark_output_dir,
                plan.legacy_output_dir,
            ],
        )
        if resume and completed is not None:
            reference = self._write_trial_metadata(plan, cache, completed)
            return {**plan.to_dict(), "trial_reference": reference}

        self._write_trial_metadata(plan, cache)
        runner = BenchmarkRunner(deepcopy(dict(plan.resolved_config)))
        runner.run()
        completed = runner.completed_run()
        reference = self._write_trial_metadata(plan, cache, completed)
        return {**plan.to_dict(), "trial_reference": reference}

    def execute(
        self,
        plans: Sequence[TrialPlan],
        *,
        build_missing_caches: bool,
        run: bool,
        resume: bool,
    ) -> list[dict[str, Any]]:
        if build_missing_caches:
            self.assert_disk_capacity(plans)
        results: list[dict[str, Any]] = []
        for plan in plans:
            if build_missing_caches and not plan.cache.reusable and not run:
                cache = self.build_cache(plan)
                results.append({
                    **plan.to_dict(),
                    "status": "cache_ready",
                    "cache": cache.to_dict(),
                })
            elif run:
                results.append(
                    self.run_trial(
                        plan,
                        build_missing_cache=build_missing_caches,
                        resume=resume,
                    )
                )
        return results


def render_plan_csv(plans: Sequence[TrialPlan]) -> str:
    rows = [plan.to_dict() for plan in plans]
    if not rows:
        return ""
    output = io.StringIO()
    fieldnames = list(rows[0])
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = {
            key: _canonical_json(value) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        writer.writerow(serialized)
    return output.getvalue()


def render_plan_markdown(plans: Sequence[TrialPlan]) -> str:
    lines = [
        "# Preprocessing ablation plan",
        "",
        "| Trial | Band-pass | Notch | CAR | Hash | Cache | Existing result | Action | Estimated new cache |",
        "|---|---:|---:|---:|---|---|---|---|---:|",
    ]
    for plan in plans:
        parameters = plan.trial.parameter_dict()
        lines.append(
            "| {trial} | {bp} | {notch} | {car} | `{hash}` | {cache} | {result} | {action} | {size} |".format(
                trial=plan.trial.trial_id,
                bp=str(parameters[FACTOR_PATHS[0]]).lower(),
                notch=str(parameters[FACTOR_PATHS[1]]).lower(),
                car=str(parameters[FACTOR_PATHS[2]]).lower(),
                hash=plan.trial.preprocessing_hash[:16],
                cache=(
                    f"reuse `{_relative_or_absolute(plan.cache.cache_path)}`"
                    if plan.cache.reusable
                    else "missing"
                ),
                result=plan.existing_benchmark_result or "none",
                action=plan.action,
                size=plan.estimated_new_cache_size,
            )
        )
    missing = sum(not plan.cache.reusable for plan in plans)
    estimate = sum(plan.estimated_new_cache_size for plan in plans)
    lines.extend([
        "",
        f"Existing reusable caches: **{len(plans) - missing}**.",
        f"Missing caches: **{missing}**.",
        f"Estimated new cache bytes: **{estimate}**.",
        f"Planned benchmark runs: **{sum(plan.action != 'skip_completed' for plan in plans)}**.",
        "",
    ])
    return "\n".join(lines)
