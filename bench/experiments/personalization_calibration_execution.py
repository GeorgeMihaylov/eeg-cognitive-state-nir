"""Execution bridge for the leakage-safe personalization-calibration protocol.

Base models are trained exclusively by :class:`BenchmarkRunner`. Participant
adaptation delegates to the existing Torch adapter ``clone``/``fine_tune``
surface; this module only owns deterministic orchestration and artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from bench.bench_runner import BenchmarkRunner, benchmark_config_hash
from bench.core.artifact_paths import portable_artifact_directory
from bench.tasks.tasks_registry import get_task
from bench.validation.cross_val import CrossValidator
from bench.validation.metrics import MetricsCalculator
from model_zoo import build_model
from model_zoo.DL.adapter import TorchClassificationAdapter

from .personalization_calibration import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    PersonalizationCalibrationPlanner,
    PlanFilters,
    _participant_partition,
    stable_hash,
    validate_temporal_partition,
)


EXECUTION_SCHEMA_VERSION = "personalization-calibration-execution-v1"
RESULT_FILES = (
    "participant_results.csv",
    "aggregate_results.csv",
    "budget_curve.csv",
    "eligibility.csv",
    "execution_manifest.json",
)


def base_run_directory(output_dir: str | Path, unit_id: str) -> Path:
    """Return the legacy base path or a compact portable equivalent."""

    return portable_artifact_directory(
        output_dir,
        ("base_runs", str(unit_id)),
        compact_namespace="_b",
    )


def participant_run_directory(output_dir: str | Path, execution_id: str) -> Path:
    """Return the legacy participant path or a compact portable equivalent."""

    return portable_artifact_directory(
        output_dir,
        ("participant_runs", str(execution_id)),
        compact_namespace="_p",
    )


def participant_execution_identity(
    *,
    protocol_hash: str,
    plan_hash: str,
    base_checkpoint_identity_hash: str,
    condition: Mapping[str, Any],
    participant: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable identity shared by full and scoped execution."""

    return {
        "protocol_hash": str(protocol_hash),
        "plan_hash": str(plan_hash),
        "base_checkpoint_identity_hash": str(base_checkpoint_identity_hash),
        "condition_id": condition["condition_id"],
        "subject_id": str(participant["subject_id"]),
        "budget_fraction": float(condition["budget_fraction"]),
        "mode": condition["mode"],
        "calibration_sample_hash": participant["calibration_sample_hash"],
        "evaluation_sample_hash": participant["evaluation_sample_hash"],
        "q3_transform_hash": participant["q3_transform_hash"],
    }


def execution_scope_directory(
    output_dir: str | Path,
    execution_model: str | None,
) -> Path:
    """Keep unfiltered artifacts stable and isolate operational model scopes."""

    if execution_model is None:
        return Path(output_dir)
    return portable_artifact_directory(
        output_dir,
        ("execution_scopes", f"model_{execution_model}"),
        compact_namespace="_x",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_state_hash(adapter: TorchClassificationAdapter) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def adapter_normalization_hash(adapter: TorchClassificationAdapter) -> str:
    """Hash the fitted outer-train transform without reading evaluation data."""
    state = adapter.get_feature_preprocessing_state()
    if state is not None:
        return stable_hash(state)
    payload: dict[str, Any] = {"feature_mean": None, "feature_scale": None}
    if adapter.feature_mean_ is not None:
        payload["feature_mean"] = np.asarray(adapter.feature_mean_).tolist()
    if adapter.feature_scale_ is not None:
        payload["feature_scale"] = np.asarray(adapter.feature_scale_).tolist()
    return stable_hash(payload)


def temporal_adaptation_split(
    calibration_metadata: pd.DataFrame,
    *,
    validation_fraction: float,
    minimum_train_windows: int,
    minimum_validation_windows: int,
    evaluation_metadata: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split calibration chronologically; evaluation is audit-only and untouched."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    ordered = calibration_metadata.sort_values(
        ["absolute_t_start", "source", "record_id", "t_start", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    n_validation = max(
        int(minimum_validation_windows),
        int(np.ceil(len(ordered) * validation_fraction)),
    )
    n_train = len(ordered) - n_validation
    if n_train < minimum_train_windows or n_validation < minimum_validation_windows:
        raise ValueError(
            "insufficient_data: calibration prefix cannot provide the required "
            f"temporal train/validation split ({len(ordered)} windows)"
        )
    train = ordered.iloc[:n_train]
    validation = ordered.iloc[n_train:]
    if float(train["absolute_t_start"].max()) >= float(
        validation["absolute_t_start"].min()
    ):
        raise RuntimeError("adaptation-train must be earlier than validation")
    if evaluation_metadata is not None and not evaluation_metadata.empty:
        if float(validation["absolute_t_start"].max()) >= float(
            evaluation_metadata["absolute_t_start"].min()
        ):
            raise RuntimeError("validation must be earlier than evaluation")
    train_ids = train["sample_id"].astype(str).to_numpy()
    validation_ids = validation["sample_id"].astype(str).to_numpy()
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("adaptation train/validation sample overlap")
    return train_ids, validation_ids, {
        "strategy": "calibration_only_chronological_suffix",
        "random_split": False,
        "n_train": int(len(train_ids)),
        "n_validation": int(len(validation_ids)),
        "train_sample_hash": stable_hash(sorted(train_ids.tolist())),
        "validation_sample_hash": stable_hash(sorted(validation_ids.tolist())),
        "train_max_absolute_t_start": float(train["absolute_t_start"].max()),
        "validation_min_absolute_t_start": float(
            validation["absolute_t_start"].min()
        ),
        "validation_max_absolute_t_start": float(
            validation["absolute_t_start"].max()
        ),
        "evaluation_min_absolute_t_start": (
            None
            if evaluation_metadata is None or evaluation_metadata.empty
            else float(evaluation_metadata["absolute_t_start"].min())
        ),
    }


def base_unit_key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pm": str(row["pm"]),
        "task_type": str(row["task_type"]),
        "target_id": str(row["target_id"]),
        "model": str(row["model"]),
        "input_family": str(row["input_family"]),
        "outer_fold": int(row["outer_fold"]),
        "seed": int(row["seed"]),
    }


def base_unit_id(row: Mapping[str, Any]) -> str:
    return stable_hash(base_unit_key(row))[:20]


def validate_base_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol_hash: str,
    unit: Mapping[str, Any],
) -> None:
    """Reject name-only checkpoint reuse when the scientific contract differs."""
    if manifest.get("protocol_hash") != protocol_hash:
        raise ValueError("Base checkpoint protocol hash mismatch")
    if manifest.get("base_unit") != base_unit_key(unit):
        raise ValueError("Base checkpoint unit contract mismatch")
    required = {
        "plan_hash", "benchmark_config_hash", "preprocessing_hashes",
        "sample_universe_hash", "input_shape",
        "normalization_hash", "task_type", "num_outputs", "seed",
        "model_config_hash", "target_transform_hash", "checkpoint_sha256",
        "checkpoint_identity_hash",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Base checkpoint manifest is incomplete: {missing}")
    identity_payload = {
        key: value for key, value in manifest.items()
        if key != "checkpoint_identity_hash"
    }
    if stable_hash(identity_payload) != manifest["checkpoint_identity_hash"]:
        raise ValueError("Base checkpoint identity hash mismatch")


def validate_participant_resume_result(
    result: Mapping[str, Any], expected_identity: Mapping[str, Any]
) -> None:
    if result.get("execution_identity") != dict(expected_identity):
        raise ValueError("Participant resume identity mismatch")
    if result.get("status") not in {"completed", "insufficient_data"}:
        raise ValueError("Participant resume result is not terminal")
    checkpoint = result.get("adapted_checkpoint")
    if checkpoint:
        path = Path(str(checkpoint))
        if not path.is_file():
            raise ValueError("Participant resume checkpoint is missing")
        if _file_sha256(path) != result.get("adapted_checkpoint_sha256"):
            raise ValueError("Participant checkpoint hash mismatch")


def _metrics_for(task_type: str) -> tuple[str, ...]:
    return (
        CLASSIFICATION_METRICS
        if task_type == "classification"
        else REGRESSION_METRICS
    )


def build_eligibility_table(
    run_matrix: pd.DataFrame,
    participant_plan: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in run_matrix.itertuples(index=False):
        participants = participant_plan.loc[
            participant_plan["pm"].eq(condition.pm)
            & participant_plan["outer_fold"].eq(condition.outer_fold)
            & participant_plan["budget_fraction"].eq(condition.budget_fraction)
        ]
        eligible = int(participants["status"].eq("planned").sum())
        total = int(len(participants))
        rows.append({
            "pm": condition.pm,
            "task_type": condition.task_type,
            "model": condition.model,
            "outer_fold": int(condition.outer_fold),
            "mode": condition.mode,
            "budget_fraction": float(condition.budget_fraction),
            "n_total_participants": total,
            "n_eligible": eligible if condition.status != "unsupported" else 0,
            "n_insufficient_data": int(total - eligible),
            "eligibility_fraction": (0.0 if not total else eligible / total),
            "condition_status": condition.status,
        })
    return pd.DataFrame(rows)


def aggregate_execution_results(
    participant_results: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = participant_results.loc[
        participant_results["status"].eq("completed")
    ].copy()
    aggregate_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    group_columns = ["pm", "task_type", "model", "mode", "budget_fraction"]
    for keys, group in completed.groupby(group_columns, sort=True):
        key_payload = dict(zip(group_columns, keys))
        for metric in _metrics_for(str(key_payload["task_type"])):
            for value_kind, column in (
                ("zero_shot", f"zero_shot_{metric}"),
                ("adapted", f"adapted_{metric}"),
                ("delta", f"delta_{metric}"),
            ):
                values = pd.to_numeric(group.get(column), errors="coerce").dropna()
                if values.empty:
                    continue
                aggregate_rows.append({
                    **key_payload,
                    "metric": metric,
                    "value_kind": value_kind,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "median": float(values.median()),
                    "n_participants": int(group.loc[values.index, "subject_id"].nunique()),
                })

    positive = completed.loc[
        completed["mode"].isin(["head_only", "full_model"])
        & completed["budget_fraction"].gt(0)
    ].copy()
    common_keys = ["outer_fold", "subject_id"]
    for keys, modes in positive.groupby(
        ["pm", "task_type", "model", "mode"], sort=True
    ):
        available_budgets = sorted(modes["budget_fraction"].unique())
        participant_sets = [
            set(map(tuple, modes.loc[
                modes["budget_fraction"].eq(budget), common_keys
            ].astype(str).to_numpy()))
            for budget in available_budgets
        ]
        common = set.intersection(*participant_sets) if participant_sets else set()
        for budget in available_budgets:
            budget_group = modes.loc[modes["budget_fraction"].eq(budget)].copy()
            paired_mask = budget_group[common_keys].astype(str).apply(tuple, axis=1).isin(common)
            eligible_match = eligibility.loc[
                eligibility["pm"].eq(keys[0])
                & eligibility["task_type"].eq(keys[1])
                & eligibility["model"].eq(keys[2])
                & eligibility["mode"].eq(keys[3])
                & eligibility["budget_fraction"].eq(budget)
            ]
            n_total = int(eligible_match["n_total_participants"].sum())
            n_eligible = int(eligible_match["n_eligible"].sum())
            for metric in _metrics_for(str(keys[1])):
                column = f"delta_{metric}"
                available_values = pd.to_numeric(
                    budget_group[column], errors="coerce"
                ).dropna()
                paired_values = pd.to_numeric(
                    budget_group.loc[paired_mask, column], errors="coerce"
                ).dropna()
                budget_rows.append({
                    "pm": keys[0], "task_type": keys[1], "model": keys[2],
                    "mode": keys[3], "budget_fraction": float(budget),
                    "metric": metric,
                    "mean_delta_available_participant": (
                        np.nan if available_values.empty else float(available_values.mean())
                    ),
                    "n_eligible": n_eligible,
                    "n_total_participants": n_total,
                    "eligibility_fraction": 0.0 if not n_total else n_eligible / n_total,
                    "mean_delta_paired_common_cohort": (
                        np.nan if paired_values.empty else float(paired_values.mean())
                    ),
                    "n_paired_common_cohort": int(len(common)),
                })
    return pd.DataFrame(aggregate_rows), pd.DataFrame(budget_rows)


def _scope_cost_table(matrix: pd.DataFrame) -> pd.DataFrame:
    scopes: list[tuple[str, pd.Series]] = [
        ("full_protocol", pd.Series(True, index=matrix.index)),
        ("shallowconvnet_only", matrix["model"].eq("torch_shallow_convnet")),
        (
            "head_only_plus_shared_zero_shot",
            matrix["mode"].isin(["zero_shot", "head_only"]),
        ),
    ]
    for fold in range(1, 6):
        scopes.append((f"fold_{fold:02d}", matrix["outer_fold"].eq(fold)))
    scopes.append((
        "fold_01_smoke_focus_q3_shallow_head",
        matrix["outer_fold"].eq(1)
        & matrix["pm"].eq("focus")
        & matrix["task_type"].eq("classification")
        & matrix["model"].eq("torch_shallow_convnet")
        & matrix["mode"].isin(["zero_shot", "head_only"]),
    ))
    rows = []
    for name, mask in scopes:
        frame = matrix.loc[mask & matrix["status"].eq("planned")]
        bases = len({base_unit_id(row) for row in frame.to_dict("records")})
        zero = int(frame.loc[frame["mode"].eq("zero_shot"), "participant_execution_count"].sum())
        head = int(frame.loc[frame["mode"].eq("head_only"), "participant_execution_count"].sum())
        full = int(frame.loc[frame["mode"].eq("full_model"), "participant_execution_count"].sum())
        rows.append({
            "scope": name, "base_trainings": bases,
            "zero_shot_inferences": zero, "head_only_adaptations": head,
            "full_model_adaptations": full,
            "training_jobs": bases + head + full,
            "estimated_checkpoint_files": bases + head + full,
        })
    return pd.DataFrame(rows)


class ExecutionBackend(Protocol):
    def ensure_base(self, base: Mapping[str, Any], *, resume: bool) -> Any: ...

    def execute_participant(
        self,
        base_handle: Any,
        condition: Mapping[str, Any],
        participant: Mapping[str, Any],
        *,
        resume: bool,
    ) -> dict[str, Any]: ...


@dataclass
class BaseRunHandle:
    unit: dict[str, Any]
    adapter: TorchClassificationAdapter
    split: Any
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_identity_hash: str
    normalization_hash: str
    target_transform_hash: str
    resumed: bool


class BenchmarkPersonalizationBackend:
    """Production backend composed from BenchmarkRunner and the shared adapter."""

    def __init__(
        self,
        planner: PersonalizationCalibrationPlanner,
        *,
        plan_hash: str,
        max_epochs: int | None = None,
        device: str | None = None,
    ) -> None:
        self.planner = planner
        self.config = planner.config
        self.plan_hash = str(plan_hash)
        self.max_epochs = max_epochs
        self.device = device
        self._target_frames: dict[str, pd.DataFrame] = {}
        self._zero_shot_cache: dict[
            tuple[str, str, str], tuple[np.ndarray, np.ndarray | None, dict[str, Any]]
        ] = {}

    def _model_params(self, model: str, task_type: str) -> dict[str, Any]:
        model_config = next(
            item for item in self.config["models"] if item["model_id"] == model
        )
        params = deepcopy(model_config.get("params", {}))
        params.update(deepcopy(model_config.get(f"{task_type}_params", {})))
        execution = self.config.get("execution", {})
        params.update(deepcopy(execution.get("base_training", {})))
        configured_model_params = execution.get("model_params", {}).get(
            model, {}
        )
        params.update(deepcopy(configured_model_params.get("common", {})))
        params.update(deepcopy(configured_model_params.get(task_type, {})))
        params.setdefault("batch_size", 128)
        params.setdefault("max_epochs", 15)
        params.setdefault("learning_rate", 0.001)
        params.setdefault("weight_decay", 0.0001)
        params.setdefault("validation_size", 0.15)
        params.setdefault("early_stopping_patience", 4)
        params.setdefault("device", "auto")
        params.setdefault("random_state", int(self.config["experiment"]["random_state"]))
        params.setdefault("num_workers", 0)
        params.setdefault("standardize", True)
        if self.max_epochs is not None:
            params["max_epochs"] = int(self.max_epochs)
        if self.device is not None:
            params["device"] = str(self.device)
        return params

    def _base_config(self, base: Mapping[str, Any]) -> dict[str, Any]:
        pm = str(base["pm"])
        family = str(base["input_family"])
        data = self.config["data"]
        if family == "raw":
            dataset_name = "emotiv_raw_eeg"
            dataset = {
                "data_path": str(self.planner.data_root / data["raw_manifest"]),
                "cache_path_root": str(self.planner.data_root),
                "target_data_path": str(
                    self.planner.data_root / data["processed_targets"]
                ),
                "target_id": base["target_id"],
                "dataset_mode": data["dataset_mode"],
                "logical_recording_map_path": str(
                    self.planner.data_root / data["logical_recording_map"]
                ),
                "raw_preprocessing": deepcopy(data["raw_preprocessing"]),
            }
        else:
            dataset_name = "emotiv_cognitive"
            dataset = {
                "data_path": str(self.planner.data_root / data["processed_targets"]),
                "target_id": base["target_id"],
                "feature_set": "pow_plus_eeg",
                "logical_recording_map_path": str(
                    self.planner.data_root / data["logical_recording_map"]
                ),
                "cohort_manifest_path": str(
                    self.planner.output_dir / "cohorts" / f"{pm}.parquet"
                ),
            }
        unit_id = base_unit_id(base)
        model_params = self._model_params(str(base["model"]), str(base["task_type"]))
        model = {
            "type": base["model"],
            "task_type": base["task_type"],
            "params": model_params,
        }
        if family == "features":
            model["feature_scaling"] = deepcopy(
                self.config.get("execution", {}).get(
                    "feature_scaling",
                    {
                        "strategy": "standard_clip",
                        "clip_percentiles": [0.5, 99.5],
                    },
                )
            )
        return {
            "output_dir": str(base_run_directory(self.planner.output_dir, unit_id)),
            "raw_preprocessing": deepcopy(data["raw_preprocessing"]),
            "datasets": {dataset_name: dataset},
            "tasks": [base["target_id"]],
            "validation": {
                "strategy": "group_record",
                "group_column": self.config["protocol"]["inner_validation_group_column"],
                "validation_size": 0.15,
                "random_state": int(base["seed"]),
            },
            "models": {"personalization_base": model},
            "evaluation": {
                "protocol": "group_kfold_subject",
                "n_splits": int(self.config["protocol"]["n_outer_folds"]),
                "group_column": self.config["protocol"]["outer_group_column"],
                "precomputed_fold_column": self.config["protocol"]["fixed_outer_fold_column"],
                "folds": [int(base["outer_fold"])],
                "random_state": int(base["seed"]),
            },
            "task_config": {
                "random_state": int(base["seed"]),
                "target_id": base["target_id"],
            },
            "run_within_subject": False,
            "run_loso": False,
        }

    @staticmethod
    def _result_fold(
        completed: Any,
        dataset_name: str,
        task_name: str,
        fold: int,
    ) -> dict[str, Any]:
        results = json.loads(completed.result_file.read_text(encoding="utf-8"))
        return results[dataset_name]["models"][task_name][
            "personalization_base"
        ]["group_kfold_subject"]["folds"][f"fold_{fold:02d}"]

    def ensure_base(self, base: Mapping[str, Any], *, resume: bool) -> BaseRunHandle:
        config = self._base_config(base)
        runner = BenchmarkRunner(deepcopy(config))
        completed = BenchmarkRunner.find_completed_run(
            config, search_directories=[config["output_dir"]]
        )
        resumed = completed is not None
        if completed is None:
            runner.run()
            completed = runner.completed_run()
        dataset_name = next(iter(config["datasets"]))
        data = runner.load_dataset(dataset_name)
        task_name = str(base["target_id"])
        task = get_task(task_name, data, config["task_config"])
        folds = CrossValidator(task).run_group_kfold(
            group_column=config["evaluation"]["group_column"],
            n_splits=config["evaluation"]["n_splits"],
            random_state=config["evaluation"]["random_state"],
            precomputed_fold_column=config["evaluation"]["precomputed_fold_column"],
        )
        split = folds[f"fold_{int(base['outer_fold']):02d}"]
        fold_result = self._result_fold(
            completed, dataset_name, task_name, int(base["outer_fold"])
        )
        checkpoint = Path(fold_result["artifacts"]["model"])
        model_config = config["models"]["personalization_base"]
        params = deepcopy(model_config["params"])
        if split.metadata.get("observation_unit") == "raw_eeg_window":
            rates = np.asarray(split.row_metadata_train["sfreq_target"], dtype=float)
            params.setdefault("sampling_rate", float(np.median(rates)))
            params.setdefault("channel_names", list(split.feature_names or []))
        adapter = build_model(
            model_name=str(base["model"]),
            task_type=str(base["task_type"]),
            input_shape=tuple(split.X_train.shape[1:]),
            num_outputs=3 if base["task_type"] == "classification" else 1,
            params=params,
        )
        if not isinstance(adapter, TorchClassificationAdapter):
            raise TypeError("Personalization requires TorchClassificationAdapter")
        adapter.load(checkpoint)
        target_hash = str(split.metadata.get("target_transform_hash", ""))
        if base["task_type"] == "classification" and target_hash != str(
            base.get("q3_transform_hash", "")
        ):
            raise RuntimeError("Q3 transform hash differs between plan and base fold")
        preprocessing_hashes = sorted(set(map(
            str, split.row_metadata_train.get("preprocessing_hash", [])
        )))
        identity = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "protocol_hash": self.planner.protocol_hash,
            "plan_hash": self.plan_hash,
            "base_unit": base_unit_key(base),
            "benchmark_config_hash": benchmark_config_hash(config),
            "preprocessing_hashes": preprocessing_hashes,
            "sample_universe_hash": stable_hash(sorted(map(str, data.sample_ids))),
            "input_shape": list(split.X_train.shape[1:]),
            "normalization_hash": adapter_normalization_hash(adapter),
            "task_type": base["task_type"],
            "num_outputs": 3 if base["task_type"] == "classification" else 1,
            "seed": int(base["seed"]),
            "model_config_hash": stable_hash(model_config),
            "target_transform_hash": target_hash,
            "checkpoint_sha256": _file_sha256(checkpoint),
        }
        identity["checkpoint_identity_hash"] = stable_hash(identity)
        manifest_path = Path(config["output_dir"]) / "base_checkpoint_manifest.json"
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous != identity:
                raise RuntimeError("Incompatible base checkpoint identity manifest")
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        return BaseRunHandle(
            unit=base_unit_key(base), adapter=adapter, split=split,
            checkpoint_path=checkpoint,
            checkpoint_sha256=identity["checkpoint_sha256"],
            checkpoint_identity_hash=identity["checkpoint_identity_hash"],
            normalization_hash=identity["normalization_hash"],
            target_transform_hash=target_hash, resumed=resumed,
        )

    @staticmethod
    def _subset_by_ids(split: Any, ids: np.ndarray) -> tuple[Any, np.ndarray]:
        sample_ids = np.asarray(split.sample_id_test).astype(str)
        position = {value: index for index, value in enumerate(sample_ids)}
        missing = [value for value in ids if str(value) not in position]
        if missing:
            raise RuntimeError(f"Participant samples are absent from outer test: {missing[:5]}")
        indices = np.asarray([position[str(value)] for value in ids], dtype=np.int64)
        return split.X_test[indices], np.asarray(split.y_test)[indices]

    def execute_participant(
        self,
        base_handle: BaseRunHandle,
        condition: Mapping[str, Any],
        participant: Mapping[str, Any],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        identity = participant_execution_identity(
            protocol_hash=self.planner.protocol_hash,
            plan_hash=self.plan_hash,
            base_checkpoint_identity_hash=base_handle.checkpoint_identity_hash,
            condition=condition,
            participant=participant,
        )
        execution_id = stable_hash(identity)[:24]
        run_dir = participant_run_directory(self.planner.output_dir, execution_id)
        result_path = run_dir / "result.json"
        if result_path.is_file() and resume:
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            validate_participant_resume_result(saved, identity)
            saved["participant_resumed"] = True
            return saved
        if result_path.exists() and not resume:
            raise FileExistsError(f"Participant result exists: {result_path}")

        pm = str(condition["pm"])
        frame = self._target_frames.setdefault(pm, self.planner._load_target_frame(pm))
        subject = frame.loc[
            frame["outer_fold"].eq(int(condition["outer_fold"]))
            & frame["subject_id"].astype(str).eq(str(participant["subject_id"]))
        ]
        partition, _ = _participant_partition(
            subject,
            budget=float(condition["budget_fraction"]),
            reference_budget=float(
                self.config["protocol"]["fixed_evaluation_reference_budget_fraction"]
            ),
            protocol=self.config["protocol"],
        )
        validate_temporal_partition(partition)
        evaluation_ids = partition.evaluation_metadata["sample_id"].astype(str).to_numpy()
        X_evaluation, y_evaluation = self._subset_by_ids(
            base_handle.split, evaluation_ids
        )
        adapter = base_handle.adapter
        base_state_before = _adapter_state_hash(adapter)
        normalization_before = adapter_normalization_hash(adapter)
        zero_key = (
            base_handle.checkpoint_identity_hash,
            str(participant["subject_id"]),
            str(participant["evaluation_sample_hash"]),
        )
        cached_zero = self._zero_shot_cache.get(zero_key)
        if cached_zero is None:
            zero_pred = adapter.predict(X_evaluation)
            zero_proba = (
                adapter.predict_proba(X_evaluation)
                if condition["task_type"] == "classification" else None
            )
            zero_metrics = MetricsCalculator.calculate_all_metrics(
                y_evaluation, zero_pred, zero_proba,
                task_type=str(condition["task_type"]),
                labels=(
                    np.arange(3)
                    if condition["task_type"] == "classification" else None
                ),
            )
            self._zero_shot_cache[zero_key] = (
                np.asarray(zero_pred).copy(),
                None if zero_proba is None else np.asarray(zero_proba).copy(),
                dict(zero_metrics),
            )
        else:
            zero_pred, zero_proba, zero_metrics = cached_zero
        adapted = adapter
        validation_audit: dict[str, Any] = {}
        training_time = 0.0
        adapted_checkpoint: Path | None = None
        if condition["mode"] != "zero_shot":
            try:
                train_ids, validation_ids, validation_audit = temporal_adaptation_split(
                    partition.calibration_metadata,
                    validation_fraction=float(
                        self.config["calibration"]["adaptation_validation_fraction"]
                    ),
                    minimum_train_windows=int(
                        self.config["protocol"].get("minimum_adaptation_train_samples", 4)
                    ),
                    minimum_validation_windows=int(
                        self.config["protocol"].get("minimum_adaptation_validation_samples", 1)
                    ),
                    evaluation_metadata=partition.evaluation_metadata,
                )
            except ValueError as exc:
                insufficient = {
                    **identity, "execution_identity": identity,
                    "execution_id": execution_id, "status": "insufficient_data",
                    "reason": str(exc), "participant_resumed": False,
                }
                run_dir.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(insufficient, indent=2) + "\n", encoding="utf-8"
                )
                return insufficient
            X_train, y_train = self._subset_by_ids(base_handle.split, train_ids)
            X_validation, y_validation = self._subset_by_ids(
                base_handle.split, validation_ids
            )
            adapted = adapter.clone()
            if _adapter_state_hash(adapted) != base_state_before:
                raise RuntimeError("Cloned participant model differs from base")
            started = time.perf_counter()
            adapted.fine_tune(
                X_train, y_train, mode=str(condition["mode"]),
                X_validation=X_validation, y_validation=y_validation,
                max_epochs=(
                    int(self.max_epochs)
                    if self.max_epochs is not None
                    else int(self.config["calibration"]["max_epochs"])
                ),
                learning_rate=float(self.config["calibration"][
                    "head_only_learning_rate"
                    if condition["mode"] == "head_only"
                    else "full_model_learning_rate"
                ]),
                weight_decay=float(self.config["calibration"]["weight_decay"]),
                early_stopping_patience=int(
                    self.config["calibration"]["early_stopping_patience"]
                ),
                random_state=int(condition["seed"]),
            )
            training_time = time.perf_counter() - started
            if adapter_normalization_hash(adapted) != normalization_before:
                raise RuntimeError("Fine-tuning changed frozen outer-train normalization")
            run_dir.mkdir(parents=True, exist_ok=True)
            if bool(self.config.get("execution", {}).get(
                "save_adapted_checkpoints", True
            )):
                adapted_checkpoint = run_dir / "model.pt"
                adapted.save(adapted_checkpoint)
                pd.DataFrame(adapted.training_log_).to_csv(
                    run_dir / "training_log.csv", index=False
                )
        adapted_pred = adapted.predict(X_evaluation)
        adapted_proba = (
            adapted.predict_proba(X_evaluation)
            if condition["task_type"] == "classification" else None
        )
        adapted_metrics = MetricsCalculator.calculate_all_metrics(
            y_evaluation, adapted_pred, adapted_proba,
            task_type=str(condition["task_type"]),
            labels=(np.arange(3) if condition["task_type"] == "classification" else None),
        )
        if _adapter_state_hash(adapter) != base_state_before:
            raise RuntimeError("Participant adaptation mutated the shared base model")
        if adapter_normalization_hash(adapter) != base_handle.normalization_hash:
            raise RuntimeError("Shared base normalization changed")
        if condition["task_type"] == "classification" and str(
            participant["q3_transform_hash"]
        ) != base_handle.target_transform_hash:
            raise RuntimeError("Q3 hash invariant failed during participant execution")
        prediction_frame = pd.DataFrame({
            "sample_id": evaluation_ids,
            "y_true": np.asarray(y_evaluation),
            "zero_shot_y_pred": np.asarray(zero_pred),
            "adapted_y_pred": np.asarray(adapted_pred),
        })
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_parquet(run_dir / "predictions.parquet", index=False)
        result: dict[str, Any] = {
            **identity,
            "execution_identity": identity,
            "execution_id": execution_id,
            "status": "completed",
            "reason": "",
            "participant_resumed": False,
            "base_resumed": bool(base_handle.resumed),
            "base_checkpoint": str(base_handle.checkpoint_path),
            "base_checkpoint_sha256": base_handle.checkpoint_sha256,
            "adapted_checkpoint": (
                None if adapted_checkpoint is None else str(adapted_checkpoint)
            ),
            "adapted_checkpoint_sha256": (
                None if adapted_checkpoint is None else _file_sha256(adapted_checkpoint)
            ),
            "normalization_hash": normalization_before,
            "normalization_refit": False,
            "zero_shot_semantics": "zero_shot_shared_eval",
            "calibration_windows": int(len(partition.calibration_metadata)),
            "evaluation_windows": int(len(evaluation_ids)),
            "training_time_seconds": float(training_time),
            "epochs_trained": int(getattr(adapted, "n_epochs_trained_", 0)) if adapted is not adapter else 0,
            "best_validation_loss": getattr(adapted, "best_validation_loss_", None) if adapted is not adapter else None,
            **validation_audit,
        }
        for metric in _metrics_for(str(condition["task_type"])):
            before = float(zero_metrics.get(metric, np.nan))
            after = float(adapted_metrics.get(metric, np.nan))
            result[f"zero_shot_{metric}"] = before
            result[f"adapted_{metric}"] = after
            result[f"delta_{metric}"] = after - before
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        return result


class PersonalizationCalibrationExecutor:
    """Resolve costs, reuse base units and execute participant conditions."""

    def __init__(self, planner: PersonalizationCalibrationPlanner) -> None:
        self.planner = planner

    def _tables(self, filters: PlanFilters) -> dict[str, Any]:
        return self.planner.materialize_tables(filters=filters)

    def _select_execution_matrix(
        self,
        matrix: pd.DataFrame,
        execution_model: str | None,
    ) -> pd.DataFrame:
        if execution_model is None:
            return matrix
        configured = {
            str(item["model_id"]) for item in self.planner.config["models"]
        }
        if execution_model not in configured:
            raise ValueError(
                f"Unknown execution model {execution_model!r}; "
                f"expected one of {sorted(configured)}"
            )
        selected = matrix.loc[matrix["model"].eq(execution_model)].copy()
        if selected.empty:
            raise ValueError(
                f"Execution model {execution_model!r} has no rows in the "
                "scientific plan selected by PlanFilters"
            )
        return selected

    def _plan_hash(self, filters: PlanFilters) -> str:
        return stable_hash({
            "protocol_hash": self.planner.protocol_hash,
            "filters": {
                "outer_fold": filters.outer_fold,
                "pm": filters.pm,
                "task_type": filters.task_type,
                "calibration_mode": filters.calibration_mode,
                **({"model": filters.model} if filters.model is not None else {}),
                **(
                    {"budget_fraction": filters.budget_fraction}
                    if filters.budget_fraction is not None else {}
                ),
            },
        })

    def _write_cohort_manifests(self, cohorts: Mapping[str, pd.DataFrame]) -> None:
        root = self.planner.output_dir / "cohorts"
        root.mkdir(parents=True, exist_ok=True)
        columns = [
            "sample_id", "source", "subject_id", "record_id",
            "record_group_id", "t_start", "t_end", "absolute_t_start",
            "outer_fold",
        ]
        for pm, cohort in cohorts.items():
            selected = cohort.loc[:, columns].sort_values(
                "sample_id", kind="mergesort"
            )
            path = root / f"{pm}.parquet"
            if path.is_file():
                existing = pd.read_parquet(path)
                if stable_hash(existing.to_dict("records")) != stable_hash(
                    selected.to_dict("records")
                ):
                    raise RuntimeError(f"Existing cohort manifest differs: {path}")
            else:
                selected.to_parquet(path, index=False)

    def dry_execution(
        self,
        *,
        filters: PlanFilters | None = None,
        execution_model: str | None = None,
        write_artifacts: bool = True,
    ) -> dict[str, Any]:
        filters = filters or PlanFilters()
        tables = self._tables(filters)
        full_matrix = tables["run_matrix"]
        matrix = self._select_execution_matrix(full_matrix, execution_model)
        planned = matrix.loc[matrix["status"].eq("planned")]
        bases = {
            base_unit_id(row): base_unit_key(row)
            for row in planned.to_dict("records")
        }
        reusable = 0
        base_rows = []
        for unit_id, unit in sorted(bases.items()):
            manifest = (
                base_run_directory(self.planner.output_dir, unit_id)
                / "base_checkpoint_manifest.json"
            )
            can_reuse = False
            reason = "no exact execution-bridge manifest"
            if manifest.is_file():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                try:
                    validate_base_checkpoint_manifest(
                        payload,
                        protocol_hash=self.planner.protocol_hash,
                        unit=unit,
                    )
                    can_reuse = True
                    reason = "exact manifest match"
                except ValueError as exc:
                    reason = str(exc)
            reusable += int(can_reuse)
            base_rows.append({
                "base_unit_id": unit_id, **unit,
                "reusable": can_reuse, "reason": reason,
            })
        eligibility = build_eligibility_table(matrix, tables["participants"])
        cost = _scope_cost_table(matrix)
        zero = int(planned.loc[planned["mode"].eq("zero_shot"), "participant_execution_count"].sum())
        head = int(planned.loc[planned["mode"].eq("head_only"), "participant_execution_count"].sum())
        full = int(planned.loc[planned["mode"].eq("full_model"), "participant_execution_count"].sum())
        insufficient = int(matrix.loc[
            matrix["status"].ne("unsupported"), "participants_insufficient"
        ].sum())
        report = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "result_status": "dry_execution",
            "training_executed": False,
            "protocol_hash": self.planner.protocol_hash,
            "base_training_units": len(bases),
            "base_checkpoints_reusable": reusable,
            "base_checkpoints_to_train": len(bases) - reusable,
            "zero_shot_shared_eval_inferences": zero,
            "head_only_adaptation_trainings": head,
            "full_model_adaptation_trainings": full,
            "adaptation_trainings": head + full,
            "participant_executions": zero + head + full,
            "unsupported_conditions": int(matrix["status"].eq("unsupported").sum()),
            "insufficient_data_participant_conditions": insufficient,
            "formal_criteria": {
                "classification_accuracy_threshold": float(
                    self.planner.config["analysis"]["formal_accuracy_threshold"]
                ),
                "aggregation": self.planner.config["analysis"]["aggregation"],
                "threshold_role": self.planner.config["analysis"]["threshold_role"],
            },
            "estimated_checkpoint_files": len(bases) + head + full,
            "runtime_estimate": None,
            "runtime_estimate_reason": (
                "No exact compatible completed base/adaptation runtime artifacts; "
                "no benchmark was launched for timing."
            ),
        }
        if execution_model is not None:
            report.update({
                "execution_filter": {"model": execution_model},
                "plan_hash": self._plan_hash(filters),
                "full_plan_conditions": int(len(full_matrix)),
                "selected_execution_conditions": int(len(matrix)),
            })
        if write_artifacts:
            artifact_dir = execution_scope_directory(
                self.planner.output_dir, execution_model
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(base_rows).to_csv(
                artifact_dir / "base_execution_units.csv", index=False
            )
            eligibility.to_csv(artifact_dir / "eligibility.csv", index=False)
            cost.to_csv(artifact_dir / "scope_cost_table.csv", index=False)
            (artifact_dir / "dry_execution.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        return report

    def run(
        self,
        *,
        filters: PlanFilters | None = None,
        execution_model: str | None = None,
        resume: bool = False,
        subject_limit: int | None = None,
        max_epochs: int | None = None,
        device: str | None = None,
        backend: ExecutionBackend | None = None,
    ) -> dict[str, Any]:
        filters = filters or PlanFilters()
        if subject_limit is not None and subject_limit <= 0:
            raise ValueError("subject_limit must be positive")
        tables = self._tables(filters)
        full_matrix = tables["run_matrix"]
        matrix = self._select_execution_matrix(full_matrix, execution_model)
        participants = tables["participants"]
        self._write_cohort_manifests(tables["cohorts"])
        plan_hash = self._plan_hash(filters)
        if backend is None:
            backend = BenchmarkPersonalizationBackend(
                self.planner, plan_hash=plan_hash,
                max_epochs=max_epochs, device=device,
            )
        rows: list[dict[str, Any]] = []
        planned = matrix.loc[matrix["status"].eq("planned")]
        for _, base_group in planned.groupby(
            ["pm", "task_type", "target_id", "model", "input_family", "outer_fold", "seed"],
            sort=True,
        ):
            representative = base_group.iloc[0].to_dict()
            representative["q3_transform_hash"] = str(
                base_group["q3_transform_hash"].iloc[0]
            )
            handle = backend.ensure_base(representative, resume=resume)
            for condition in base_group.to_dict("records"):
                condition_participants = participants.loc[
                    participants["pm"].eq(condition["pm"])
                    & participants["outer_fold"].eq(condition["outer_fold"])
                    & participants["budget_fraction"].eq(condition["budget_fraction"])
                ].sort_values("subject_id", kind="mergesort")
                if subject_limit is not None:
                    condition_participants = condition_participants.head(
                        int(subject_limit)
                    )
                ineligible = condition_participants.loc[
                    ~condition_participants["status"].eq("planned")
                ]
                for participant in ineligible.to_dict("records"):
                    rows.append({
                        "pm": condition["pm"],
                        "task_type": condition["task_type"],
                        "model": condition["model"],
                        "outer_fold": int(condition["outer_fold"]),
                        "subject_id": participant["subject_id"],
                        "mode": condition["mode"],
                        "budget_fraction": float(condition["budget_fraction"]),
                        "calibration_windows": int(participant["budget_windows"]),
                        "evaluation_windows": int(participant["evaluation_windows"]),
                        "status": "insufficient_data",
                        "reason": participant["reason"],
                        "calibration_sample_hash": participant[
                            "calibration_sample_hash"
                        ],
                        "evaluation_sample_hash": participant[
                            "evaluation_sample_hash"
                        ],
                        "q3_transform_hash": participant["q3_transform_hash"],
                    })
                eligible = condition_participants.loc[
                    condition_participants["status"].eq("planned")
                ]
                for participant in eligible.to_dict("records"):
                    result = backend.execute_participant(
                        handle, condition, participant, resume=resume
                    )
                    rows.append({
                        "pm": condition["pm"],
                        "task_type": condition["task_type"],
                        "model": condition["model"],
                        "outer_fold": int(condition["outer_fold"]),
                        "subject_id": participant["subject_id"],
                        "mode": condition["mode"],
                        "budget_fraction": float(condition["budget_fraction"]),
                        **result,
                    })
        participant_results = pd.DataFrame(rows)
        eligibility = build_eligibility_table(matrix, participants)
        aggregate, budget_curve = aggregate_execution_results(
            participant_results, eligibility
        )
        artifact_dir = execution_scope_directory(
            self.planner.output_dir, execution_model
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        participant_results.to_csv(
            artifact_dir / "participant_results.csv", index=False
        )
        aggregate.to_csv(
            artifact_dir / "aggregate_results.csv", index=False
        )
        budget_curve.to_csv(
            artifact_dir / "budget_curve.csv", index=False
        )
        eligibility.to_csv(artifact_dir / "eligibility.csv", index=False)
        manifest = {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "result_status": "completed",
            "training_executed": True,
            "protocol_hash": self.planner.protocol_hash,
            "plan_hash": plan_hash,
            "completed_participant_executions": int(
                participant_results["status"].eq("completed").sum()
            ) if not participant_results.empty else 0,
            "insufficient_data": int(
                participant_results["status"].eq("insufficient_data").sum()
            ) if not participant_results.empty else 0,
            "formal_criteria": {
                "classification_accuracy_threshold": float(
                    self.planner.config["analysis"]["formal_accuracy_threshold"]
                ),
                "aggregation": self.planner.config["analysis"]["aggregation"],
                "threshold_role": self.planner.config["analysis"]["threshold_role"],
            },
            "result_files": list(RESULT_FILES),
        }
        if execution_model is not None:
            manifest.update({
                "execution_filter": {"model": execution_model},
                "full_plan_conditions": int(len(full_matrix)),
                "selected_execution_conditions": int(len(matrix)),
                "full_plan_execution": False,
            })
        (artifact_dir / "execution_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest


__all__ = [
    "EXECUTION_SCHEMA_VERSION",
    "RESULT_FILES",
    "BaseRunHandle",
    "BenchmarkPersonalizationBackend",
    "ExecutionBackend",
    "PersonalizationCalibrationExecutor",
    "adapter_normalization_hash",
    "aggregate_execution_results",
    "base_unit_id",
    "base_unit_key",
    "base_run_directory",
    "execution_scope_directory",
    "participant_execution_identity",
    "build_eligibility_table",
    "participant_run_directory",
    "temporal_adaptation_split",
    "validate_base_checkpoint_manifest",
    "validate_participant_resume_result",
]
