"""Leakage-safe nested training and selection for categorical+auxiliary CORN."""

from __future__ import annotations

import gc
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import yaml
import torch

from bench.bench_runner import benchmark_config_hash
from bench.core.abstract_task import TaskSplit
from bench.experiments.auxiliary_corn_lambda_selection import (
    AUXILIARY_WEIGHTS,
    FEATURE_GROUPS,
    INNER_ALIGNMENT_COLUMNS,
    SEEDS,
    LambdaValidationResult,
    NoEligibleAuxiliaryWeightError,
    _configure_group_validation,
    build_inner_validation_prediction_frame,
    load_categorical_baseline_references,
    select_auxiliary_weight,
    validation_metrics_from_frame,
)
from bench.experiments.ordinal_transformer import (
    _relative_path,
    _repo_path,
    _write_json,
    stable_frame_sha256,
)
from bench.experiments.ordinal_transformer_full import (
    FULL_FOLDS,
    OrdinalTransformerFullExperiment,
)
from bench.validation.metrics import MetricsCalculator
from model_zoo import build_model


OUTER_ALIGNMENT_COLUMNS = (
    "sequence_id",
    "fold",
    "subject_id",
    "record_id",
    "source",
    "target_sample_id",
    "target_time",
    "y_true",
    "split",
)


def _lambda_token(value: float) -> str:
    return str(float(value)).replace(".", "p").rstrip("0").rstrip("p")


def load_auxiliary_corn_nested_spec(path: str | Path) -> dict[str, Any]:
    resolved = _repo_path(path)
    document = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    required = {
        "experiment",
        "dataset",
        "task",
        "feature_groups",
        "feature_definitions",
        "auxiliary_weights",
        "seeds",
        "categorical_baseline_index",
        "baseline_validation_root",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "selection",
        "protocol",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Nested auxiliary-CORN spec is missing sections: {missing}")
    if document["experiment"].get("type") != "auxiliary_corn_nested_lambda":
        raise ValueError("experiment.type must be 'auxiliary_corn_nested_lambda'")
    if tuple(document["feature_groups"]) != FEATURE_GROUPS:
        raise ValueError("feature_groups must be eeg_pow, eeg_only")
    if tuple(float(value) for value in document["auxiliary_weights"]) != AUXILIARY_WEIGHTS:
        raise ValueError("auxiliary_weights must be [0.25, 0.5, 1.0]")
    if tuple(int(value) for value in document["seeds"]) != SEEDS:
        raise ValueError("seeds must be [7, 42, 123]")
    if tuple(int(value) for value in document["evaluation"]["folds"]) != FULL_FOLDS:
        raise ValueError("evaluation.folds must be [1,2,3,4,5]")
    if int(document["validation"].get("random_state", -1)) != 42:
        raise ValueError("validation.random_state must remain 42")
    if int(document["evaluation"].get("random_state", -1)) != 42:
        raise ValueError("evaluation.random_state must remain 42")
    if float(document["selection"]["balanced_accuracy_tolerance"]) != 0.0100:
        raise ValueError("balanced_accuracy_tolerance must be exactly 0.0100")
    if document["selection"].get("no_eligible_action") != "abort_fold":
        raise ValueError("selection.no_eligible_action must be 'abort_fold'")
    if not bool(document["protocol"].get("outer_test_selected_only", False)):
        raise ValueError("protocol.outer_test_selected_only must be true")
    if int(document["protocol"].get("candidate_fold_fits", -1)) != 90:
        raise ValueError("protocol.candidate_fold_fits must be 90")
    params = document["model"]["params"]
    if int(params["max_epochs"]) != 15:
        raise ValueError("Nested candidates must use the canonical 15-epoch limit")
    return document


@dataclass(frozen=True)
class NestedFoldPlan:
    feature_group: str
    seed: int
    outer_fold: int
    baseline_run_directory: Path
    baseline_validation_metrics: Path
    baseline_validation_predictions: Path
    candidate_root: Path
    selected_root: Path
    auxiliary_weights: tuple[float, ...] = AUXILIARY_WEIGHTS

    @property
    def selection_id(self) -> str:
        return f"{self.feature_group}_seed{self.seed}_fold{self.outer_fold:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "feature_group": self.feature_group,
            "seed": self.seed,
            "outer_fold": self.outer_fold,
            "baseline_run_directory": _relative_path(self.baseline_run_directory),
            "baseline_validation_metrics": _relative_path(
                self.baseline_validation_metrics
            ),
            "baseline_validation_predictions": _relative_path(
                self.baseline_validation_predictions
            ),
            "candidate_root": _relative_path(self.candidate_root),
            "selected_root": _relative_path(self.selected_root),
            "auxiliary_weights": list(self.auxiliary_weights),
        }


@dataclass(frozen=True)
class NestedLambdaPlan:
    folds: tuple[NestedFoldPlan, ...]
    candidate_fold_fits: int
    selected_outer_evaluations: int
    output_root: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": [item.to_dict() for item in self.folds],
            "selection_units": len(self.folds),
            "candidate_fold_fits": self.candidate_fold_fits,
            "selected_outer_evaluations": self.selected_outer_evaluations,
            "output_root": _relative_path(self.output_root),
            "outer_test_selected_only": True,
        }


def _resolved_candidate_config(
    baseline_config: Mapping[str, Any],
    *,
    output_dir: Path,
    feature_group: str,
    seed: int,
    outer_fold: int,
    auxiliary_weight: float,
    experiment_name: str,
) -> dict[str, Any]:
    config = deepcopy(dict(baseline_config))
    config["output_dir"] = str(output_dir)
    model_config = config["models"]["torch_transformer"]
    params = model_config["params"]
    params["head_type"] = "categorical_corn"
    params["auxiliary_weight"] = float(auxiliary_weight)
    params["random_state"] = int(seed)
    config["evaluation"]["folds"] = [int(outer_fold)]
    config["validation"]["random_state"] = 42
    config.setdefault("task_config", {})["random_state"] = 42
    config["experiment"] = {
        "name": experiment_name,
        "type": "auxiliary_corn_nested_lambda_candidate",
        "feature_group": feature_group,
        "seed": int(seed),
        "outer_fold": int(outer_fold),
        "auxiliary_weight": float(auxiliary_weight),
        "outer_test_used": False,
    }
    return config


def _joint_validation_frame(
    split: TaskSplit,
    detailed: Mapping[str, Any],
    *,
    feature_group: str,
    seed: int,
    outer_fold: int,
    auxiliary_weight: float,
) -> pd.DataFrame:
    frame = build_inner_validation_prediction_frame(
        split,
        detailed,
        feature_group=feature_group,
        seed=seed,
        outer_fold=outer_fold,
    )
    frame["auxiliary_weight"] = float(auxiliary_weight)
    source_indices = np.asarray(detailed["indices"], dtype=np.int64)
    positions = pd.Index(source_indices).get_indexer(
        frame["outer_train_index"].to_numpy(dtype=np.int64)
    )
    if np.any(positions < 0):
        raise ValueError("Unable to align auxiliary validation outputs by index")
    for key, prefix, width in (
        ("aux_threshold_probabilities", "aux_threshold_probability", 4),
        ("aux_class_probabilities", "aux_class_probability", 5),
        ("auxiliary_raw_outputs", "aux_threshold_logit", 4),
    ):
        values = np.asarray(detailed[key], dtype=np.float64)
        if values.shape != (len(frame), width):
            raise ValueError(f"{key} has invalid shape {values.shape}")
        values = values[positions]
        for index in range(width):
            frame[f"{prefix}_{index}"] = values[:, index]
    for key in (
        "aux_expected_rank",
        "aux_ordinal_prediction",
        "aux_ordinal_argmax",
    ):
        values = np.asarray(detailed[key])
        if values.shape != (len(frame),):
            raise ValueError(f"{key} must be one-dimensional")
        frame[key] = values[positions]
    return frame


def _outer_prediction_frame(
    split: TaskSplit,
    detailed: Mapping[str, Any],
    *,
    feature_group: str,
    seed: int,
    outer_fold: int,
    auxiliary_weight: float,
) -> pd.DataFrame:
    n_rows = len(split.y_test)
    probabilities = np.asarray(detailed["class_probabilities"], dtype=np.float64)
    aux_probabilities = np.asarray(
        detailed["aux_class_probabilities"], dtype=np.float64
    )
    thresholds = np.asarray(
        detailed["aux_threshold_probabilities"], dtype=np.float64
    )
    if probabilities.shape != (n_rows, 5):
        raise ValueError("Outer primary probabilities must have shape [N,5]")
    if aux_probabilities.shape != (n_rows, 5):
        raise ValueError("Outer auxiliary probabilities must have shape [N,5]")
    if thresholds.shape != (n_rows, 4):
        raise ValueError("Outer threshold probabilities must have shape [N,4]")
    data: dict[str, Any] = {
        "fold": int(outer_fold),
        "outer_fold": int(outer_fold),
        "split": "outer_test",
        "feature_group": feature_group,
        "seed": int(seed),
        "head_type": "categorical_corn",
        "auxiliary_weight": float(auxiliary_weight),
        "sample_id": np.asarray(split.sample_id_test),
        "subject_id": np.asarray(split.subject_test).astype(str),
        "record_id": np.asarray(split.record_id_test).astype(str),
        "y_true": np.asarray(split.y_test, dtype=np.int64),
        "y_pred": np.asarray(detailed["y_pred"], dtype=np.int64),
        "categorical_expected_rank": np.asarray(
            detailed["categorical_expected_rank"], dtype=np.float64
        ),
        "aux_expected_rank": np.asarray(
            detailed["aux_expected_rank"], dtype=np.float64
        ),
        "aux_ordinal_prediction": np.asarray(
            detailed["aux_ordinal_prediction"], dtype=np.int64
        ),
        "aux_ordinal_argmax": np.asarray(
            detailed["aux_ordinal_argmax"], dtype=np.int64
        ),
    }
    for name, values in split.row_metadata_test.items():
        if name not in data:
            data[name] = np.asarray(values)
    for index in range(5):
        data[f"proba_{index}"] = probabilities[:, index]
        data[f"class_probability_{index}"] = probabilities[:, index]
        data[f"aux_class_probability_{index}"] = aux_probabilities[:, index]
    for index in range(4):
        data[f"aux_threshold_probability_{index}"] = thresholds[:, index]
    frame = pd.DataFrame(data)
    identity = "sequence_id" if "sequence_id" in frame.columns else "sample_id"
    if frame[identity].duplicated().any():
        raise ValueError("Outer-test prediction identities are duplicated")
    return frame.sort_values(identity, kind="mergesort").reset_index(drop=True)


def _metrics_from_joint_frame(frame: pd.DataFrame) -> dict[str, Any]:
    probabilities = frame[[f"proba_{index}" for index in range(5)]].to_numpy(
        dtype=np.float64
    )
    metrics = MetricsCalculator.calculate_all_metrics(
        frame["y_true"].to_numpy(dtype=np.int64),
        frame["y_pred"].to_numpy(dtype=np.int64),
        probabilities,
        expected_rank=frame["categorical_expected_rank"].to_numpy(dtype=np.float64),
    )
    aux_probabilities = frame[
        [f"aux_class_probability_{index}" for index in range(5)]
    ].to_numpy(dtype=np.float64)
    aux_metrics = MetricsCalculator.calculate_all_metrics(
        frame["y_true"].to_numpy(dtype=np.int64),
        frame["aux_ordinal_prediction"].to_numpy(dtype=np.int64),
        aux_probabilities,
        expected_rank=frame["aux_expected_rank"].to_numpy(dtype=np.float64),
    )
    metrics.update({
        f"aux_{name}": value
        for name, value in aux_metrics.items()
        if name != "confusion_matrix"
    })
    metrics["aux_confusion_matrix"] = aux_metrics["confusion_matrix"]
    metrics["categorical_aux_prediction_agreement"] = float(
        np.mean(frame["y_pred"].to_numpy() == frame["aux_ordinal_prediction"].to_numpy())
    )
    return metrics


def _save_candidate_artifacts(
    target: Path,
    *,
    model: Any,
    config: Mapping[str, Any],
    split: TaskSplit,
    validation_frame: pd.DataFrame,
    validation_metrics: Mapping[str, Any],
    validation_identity_sha256: str,
    training_seconds: float,
    config_hash: str,
) -> dict[str, str]:
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": target / "model.pt",
        "training_log": target / "training_log.csv",
        "validation_predictions": target / "validation_predictions.parquet",
        "validation_metrics": target / "validation_metrics.json",
        "validation_split": target / "validation_split.json",
        "normalization_stats": target / "normalization_stats.json",
        "resolved_config": target / "config.yaml",
        "candidate_manifest": target / "candidate_manifest.json",
    }
    model.save(paths["model"])
    pd.DataFrame(model.training_log_).to_csv(paths["training_log"], index=False)
    validation_frame.to_parquet(paths["validation_predictions"], index=False)
    _write_json(paths["validation_metrics"], dict(validation_metrics))
    _write_json(paths["validation_split"], dict(model.validation_split_ or {}))
    _write_json(paths["normalization_stats"], {
        "scope": "inner_train_only",
        "feature_names": list(split.feature_names or []),
        "mean": np.asarray(model.feature_mean_).tolist(),
        "scale": np.asarray(model.feature_scale_).tolist(),
    })
    paths["resolved_config"].write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        "status": "completed",
        "outer_test_used": False,
        "validation_identity_sha256": validation_identity_sha256,
        "validation_rows": int(len(validation_frame)),
        "training_seconds": float(training_seconds),
        "config_hash": str(config_hash),
        "training": model.get_training_summary(),
        "metrics": dict(validation_metrics),
        "artifacts": {name: _relative_path(path) for name, path in paths.items()},
    }
    _write_json(paths["candidate_manifest"], manifest)
    return {name: str(path) for name, path in paths.items()}


class AuxiliaryCornNestedLambdaExperiment:
    """Train three inner-validation candidates, then test only the selected one."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        split_builder: Callable[[Mapping[str, Any]], Mapping[str, TaskSplit]] = (
            OrdinalTransformerFullExperiment._rebuild_splits
        ),
        model_builder: Callable[..., Any] = build_model,
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.document = load_auxiliary_corn_nested_spec(self.spec_path)
        self.output_root = _repo_path(
            output_dir or self.document["experiment"]["output_dir"]
        )
        self.split_builder = split_builder
        self.model_builder = model_builder

    def plan(self) -> NestedLambdaPlan:
        references = load_categorical_baseline_references(
            self.document["categorical_baseline_index"]
        )
        reference_map = {
            (item.feature_group, item.seed): item for item in references
        }
        baseline_root = _repo_path(self.document["baseline_validation_root"])
        rows: list[NestedFoldPlan] = []
        for group in FEATURE_GROUPS:
            for seed in SEEDS:
                reference = reference_map[(group, seed)]
                for fold in FULL_FOLDS:
                    base = baseline_root / "baselines" / group / f"seed_{seed}" / f"fold_{fold:02d}"
                    rows.append(NestedFoldPlan(
                        feature_group=group,
                        seed=seed,
                        outer_fold=fold,
                        baseline_run_directory=reference.run_directory,
                        baseline_validation_metrics=base / "validation_metrics.json",
                        baseline_validation_predictions=base / "validation_predictions.parquet",
                        candidate_root=(
                            self.output_root / "candidates" / group / f"seed_{seed}" / f"fold_{fold:02d}"
                        ),
                        selected_root=(
                            self.output_root / "selected" / group / f"seed_{seed}" / f"fold_{fold:02d}"
                        ),
                    ))
        return NestedLambdaPlan(
            folds=tuple(rows),
            candidate_fold_fits=len(rows) * len(AUXILIARY_WEIGHTS),
            selected_outer_evaluations=len(rows),
            output_root=self.output_root,
        )

    @staticmethod
    def render_plan(plan: NestedLambdaPlan) -> str:
        lines = [
            "# Auxiliary-CORN nested lambda experiment plan",
            "",
            "Each selection unit trains three candidates on the same outer-train and inner-validation partition. Outer-test prediction is executed only after the validation-only selection decision.",
            "",
            "| Group | Seeds | Folds | Lambdas | Candidate fits | Selected outer tests |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
        for group in FEATURE_GROUPS:
            lines.append(
                f"| {group} | {list(SEEDS)} | {list(FULL_FOLDS)} | {list(AUXILIARY_WEIGHTS)} | 45 | 15 |"
            )
        lines.extend([
            "",
            f"Selection units: {len(plan.folds)}.",
            f"Candidate fold fits: {plan.candidate_fold_fits}.",
            f"Selected outer-test evaluations: {plan.selected_outer_evaluations}.",
            "Rejected candidates never receive outer-test predictions.",
        ])
        return "\n".join(lines)

    def _load_baseline_config(self, plan: NestedFoldPlan) -> dict[str, Any]:
        path = plan.baseline_run_directory / "config.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Baseline config not found: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        params = config["models"]["torch_transformer"]["params"]
        if str(params.get("head_type", "categorical")) != "categorical":
            raise ValueError("Paired baseline must use head_type=categorical")
        if int(params.get("random_state", -1)) != plan.seed:
            raise ValueError("Paired baseline seed mismatch")
        dataset_config = next(iter(config.get("datasets", {}).values()), {})
        observed_group = str(dataset_config.get("feature_group", ""))
        if not observed_group:
            observed_group = {
                "eeg_only": "eeg_only",
                "eeg_pow": "eeg_pow",
                "pow_plus_eeg": "eeg_pow",
            }.get(str(dataset_config.get("feature_set", "")), "")
        if observed_group != plan.feature_group:
            raise ValueError(
                "Paired baseline feature group mismatch: "
                f"{observed_group!r} != {plan.feature_group!r}"
            )
        return config

    def _train_or_resume_candidate(
        self,
        plan: NestedFoldPlan,
        split: TaskSplit,
        baseline_config: Mapping[str, Any],
        auxiliary_weight: float,
        *,
        resume: bool,
    ) -> dict[str, Any]:
        target = plan.candidate_root / f"lambda_{_lambda_token(auxiliary_weight)}"
        manifest_path = target / "candidate_manifest.json"
        config = _resolved_candidate_config(
            baseline_config,
            output_dir=target,
            feature_group=plan.feature_group,
            seed=plan.seed,
            outer_fold=plan.outer_fold,
            auxiliary_weight=auxiliary_weight,
            experiment_name=self.document["experiment"]["name"],
        )
        config_hash = benchmark_config_hash(config)
        if resume and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            required = (
                target / "model.pt",
                target / "validation_predictions.parquet",
                target / "validation_metrics.json",
            )
            if (
                manifest.get("status") == "completed"
                and not manifest.get("outer_test_used", True)
                and manifest.get("config_hash") == config_hash
                and all(path.is_file() for path in required)
            ):
                return {**manifest, "action": "resumed"}
        params = config["models"]["torch_transformer"]["params"]
        model = self.model_builder(
            "torch_transformer",
            "classification",
            input_shape=tuple(split.X_train.shape[1:]),
            num_outputs=5,
            params=params,
        )
        _configure_group_validation(model, split, config["validation"])
        started = time.time()
        model.fit(split.X_train, split.y_train)
        training_seconds = time.time() - started
        detailed = model.validation_partition_detailed(split.X_train, split.y_train)
        frame = _joint_validation_frame(
            split,
            detailed,
            feature_group=plan.feature_group,
            seed=plan.seed,
            outer_fold=plan.outer_fold,
            auxiliary_weight=auxiliary_weight,
        )
        metrics = validation_metrics_from_frame(frame)
        identity_hash = stable_frame_sha256(frame, INNER_ALIGNMENT_COLUMNS)
        baseline_predictions = pd.read_parquet(plan.baseline_validation_predictions)
        baseline_hash = stable_frame_sha256(
            baseline_predictions, INNER_ALIGNMENT_COLUMNS
        )
        if identity_hash != baseline_hash:
            raise RuntimeError(
                f"Candidate validation identities differ from paired baseline: {plan.selection_id} lambda={auxiliary_weight}"
            )
        artifacts = _save_candidate_artifacts(
            target,
            model=model,
            config=config,
            split=split,
            validation_frame=frame,
            validation_metrics=metrics,
            validation_identity_sha256=identity_hash,
            training_seconds=training_seconds,
            config_hash=config_hash,
        )
        del detailed, frame
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({
            "feature_group": plan.feature_group,
            "seed": plan.seed,
            "outer_fold": plan.outer_fold,
            "auxiliary_weight": float(auxiliary_weight),
            "artifacts": {name: _relative_path(path) for name, path in artifacts.items()},
        })
        _write_json(manifest_path, manifest)
        return {**manifest, "action": "trained"}

    def _evaluate_selected_outer(
        self,
        plan: NestedFoldPlan,
        split: TaskSplit,
        baseline_config: Mapping[str, Any],
        decision: Mapping[str, Any],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        selected_weight = float(decision["selected"]["auxiliary_weight"])
        target = plan.selected_root
        manifest_path = target / "selected_outer_manifest.json"
        if resume and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") == "completed"
                and float(manifest.get("selected_auxiliary_weight", float("nan")))
                == selected_weight
                and (target / "outer_test_predictions.parquet").is_file()
                and (target / "outer_test_metrics.json").is_file()
            ):
                return {**manifest, "action": "resumed"}
        candidate_dir = plan.candidate_root / f"lambda_{_lambda_token(selected_weight)}"
        checkpoint = candidate_dir / "model.pt"
        config = _resolved_candidate_config(
            baseline_config,
            output_dir=candidate_dir,
            feature_group=plan.feature_group,
            seed=plan.seed,
            outer_fold=plan.outer_fold,
            auxiliary_weight=selected_weight,
            experiment_name=self.document["experiment"]["name"],
        )
        model = self.model_builder(
            "torch_transformer",
            "classification",
            input_shape=tuple(split.X_train.shape[1:]),
            num_outputs=5,
            params=config["models"]["torch_transformer"]["params"],
        )
        model.load(checkpoint)
        detailed = model.predict_detailed(split.X_test)
        frame = _outer_prediction_frame(
            split,
            detailed,
            feature_group=plan.feature_group,
            seed=plan.seed,
            outer_fold=plan.outer_fold,
            auxiliary_weight=selected_weight,
        )
        metrics = _metrics_from_joint_frame(frame)
        target.mkdir(parents=True, exist_ok=True)
        predictions_path = target / "outer_test_predictions.parquet"
        metrics_path = target / "outer_test_metrics.json"
        decision_path = target / "selection_decision.json"
        frame.to_parquet(predictions_path, index=False)
        _write_json(metrics_path, metrics)
        _write_json(decision_path, decision)
        manifest = {
            "status": "completed",
            "selection_id": plan.selection_id,
            "feature_group": plan.feature_group,
            "seed": plan.seed,
            "outer_fold": plan.outer_fold,
            "selected_auxiliary_weight": selected_weight,
            "selected_checkpoint": _relative_path(checkpoint),
            "outer_test_used": True,
            "outer_test_rows": int(len(frame)),
            "outer_test_identity_sha256": stable_frame_sha256(
                frame, OUTER_ALIGNMENT_COLUMNS
            ),
            "metrics": metrics,
            "artifacts": {
                "predictions": _relative_path(predictions_path),
                "metrics": _relative_path(metrics_path),
                "selection_decision": _relative_path(decision_path),
            },
        }
        _write_json(manifest_path, manifest)
        del detailed, frame
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {**manifest, "action": "evaluated"}

    def execute(
        self,
        plan: NestedLambdaPlan | None = None,
        *,
        resume: bool = True,
    ) -> dict[str, Any]:
        resolved = self.plan() if plan is None else plan
        outcomes: list[dict[str, Any]] = []
        split_cache: dict[str, Mapping[str, TaskSplit]] = {}
        for fold_plan in resolved.folds:
            baseline_config = self._load_baseline_config(fold_plan)
            cache_key = fold_plan.feature_group
            if cache_key not in split_cache:
                split_cache[cache_key] = self.split_builder(baseline_config)
            split = split_cache[cache_key][f"fold_{fold_plan.outer_fold:02d}"]
            baseline_metrics = json.loads(
                fold_plan.baseline_validation_metrics.read_text(encoding="utf-8")
            )
            candidate_manifests: list[dict[str, Any]] = []
            candidate_results: list[LambdaValidationResult] = []
            for weight in AUXILIARY_WEIGHTS:
                manifest = self._train_or_resume_candidate(
                    fold_plan,
                    split,
                    baseline_config,
                    weight,
                    resume=resume,
                )
                candidate_manifests.append(manifest)
                metrics = manifest["metrics"]
                candidate_results.append(LambdaValidationResult(
                    auxiliary_weight=float(weight),
                    balanced_accuracy=float(metrics["balanced_accuracy"]),
                    severe_error_rate=float(metrics["severe_error_rate"]),
                    ordinal_mae=float(metrics["ordinal_mae"]),
                    macro_f1=float(metrics["macro_f1"]),
                    artifact=manifest["artifacts"]["validation_predictions"],
                ))
            try:
                decision = select_auxiliary_weight(
                    baseline_metrics,
                    candidate_results,
                    ba_tolerance=float(
                        self.document["selection"]["balanced_accuracy_tolerance"]
                    ),
                ).to_dict()
            except NoEligibleAuxiliaryWeightError as error:
                fold_plan.selected_root.mkdir(parents=True, exist_ok=True)
                failure = {
                    "status": "aborted_no_eligible_lambda",
                    "selection_id": fold_plan.selection_id,
                    "feature_group": fold_plan.feature_group,
                    "seed": fold_plan.seed,
                    "outer_fold": fold_plan.outer_fold,
                    "outer_test_used": False,
                    "reason": str(error),
                    "baseline_metrics": baseline_metrics,
                    "candidates": [item.to_dict() for item in candidate_results],
                    "candidate_manifests": candidate_manifests,
                }
                _write_json(
                    fold_plan.selected_root / "selection_decision.json", failure
                )
                outcomes.append(failure)
                continue
            selected = self._evaluate_selected_outer(
                fold_plan,
                split,
                baseline_config,
                decision,
                resume=resume,
            )
            outcomes.append({
                "status": "completed",
                "selection_id": fold_plan.selection_id,
                "feature_group": fold_plan.feature_group,
                "seed": fold_plan.seed,
                "outer_fold": fold_plan.outer_fold,
                "baseline_metrics": baseline_metrics,
                "candidate_manifests": candidate_manifests,
                "selection": decision,
                "selected_outer": selected,
            })

        completed = [item for item in outcomes if item["status"] == "completed"]
        aborted = [item for item in outcomes if item["status"] != "completed"]
        aggregated: dict[str, Any] = {}
        for group in FEATURE_GROUPS:
            for seed in SEEDS:
                rows = [
                    item for item in completed
                    if item["feature_group"] == group and int(item["seed"]) == seed
                ]
                if not rows:
                    continue
                frames = [
                    pd.read_parquet(item["selected_outer"]["artifacts"]["predictions"])
                    for item in rows
                ]
                combined = pd.concat(frames, ignore_index=True)
                identity = "sequence_id" if "sequence_id" in combined.columns else "sample_id"
                if combined[identity].duplicated().any():
                    raise RuntimeError("Selected outer predictions contain duplicate identities")
                output = self.output_root / "selected" / group / f"seed_{seed}"
                output.mkdir(parents=True, exist_ok=True)
                combined_path = output / "outer_test_predictions.parquet"
                metrics_path = output / "outer_test_metrics.json"
                combined.sort_values(["fold", identity], kind="mergesort").to_parquet(
                    combined_path, index=False
                )
                metrics = _metrics_from_joint_frame(combined)
                _write_json(metrics_path, metrics)
                selected_weights = {
                    str(item["outer_fold"]): float(
                        item["selection"]["selected"]["auxiliary_weight"]
                    )
                    for item in rows
                }
                aggregated[f"{group}_seed_{seed}"] = {
                    "folds_completed": len(rows),
                    "selected_weights": selected_weights,
                    "metrics": metrics,
                    "predictions": _relative_path(combined_path),
                    "outer_test_used_after_selection_only": True,
                }

        candidate_actions = [
            candidate.get("action", "unknown")
            for item in outcomes
            for candidate in item.get("candidate_manifests", [])
        ]
        selected_actions = [
            item.get("selected_outer", {}).get("action", "unknown")
            for item in completed
        ]
        summary = {
            "schema_version": 1,
            "status": "completed" if not aborted else "incomplete",
            "experiment": self.document["experiment"]["name"],
            "plan": resolved.to_dict(),
            "outcomes": outcomes,
            "aggregated": aggregated,
            "candidate_fold_fits_expected": resolved.candidate_fold_fits,
            "candidate_fold_fits_trained_this_run": candidate_actions.count("trained"),
            "candidate_fold_fits_resumed": candidate_actions.count("resumed"),
            "candidate_fold_fits_completed": (
                candidate_actions.count("trained") + candidate_actions.count("resumed")
            ),
            "selected_outer_evaluations_this_run": selected_actions.count("evaluated"),
            "selected_outer_evaluations_resumed": selected_actions.count("resumed"),
            "selection_units_completed": len(completed),
            "selection_units_aborted": len(aborted),
            "outer_test_selected_only": True,
            "ready_for_subject_level_analysis": bool(not aborted and len(completed) == 30),
        }
        summary_path = _repo_path(self.document["experiment"]["summary_path"])
        report_path = _repo_path(self.document["experiment"]["report_path"])
        _write_json(summary_path, summary)
        counts: dict[str, int] = {}
        for item in completed:
            key = str(item["selection"]["selected"]["auxiliary_weight"])
            counts[key] = counts.get(key, 0) + 1
        report_lines = [
            "# Auxiliary-CORN nested lambda experiment",
            "",
            f"- Selection units completed: {len(completed)}/30.",
            f"- Selection units aborted: {len(aborted)}.",
            f"- Candidate fold fits expected: {resolved.candidate_fold_fits}.",
            f"- Candidate fold fits completed: {summary['candidate_fold_fits_completed']}.",
            f"- Selected lambda counts: {counts}.",
            "- Outer-test predictions were produced only after each validation-only selection decision.",
            f"- Ready for subject-level analysis: {summary['ready_for_subject_level_analysis']}.",
        ]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": summary["status"],
            "config": _relative_path(self.spec_path),
            "summary": _relative_path(summary_path),
            "report": _relative_path(report_path),
            "output_directory": _relative_path(self.output_root),
            "candidate_fold_fits_expected": resolved.candidate_fold_fits,
            "candidate_fold_fits_completed": summary["candidate_fold_fits_completed"],
            "outer_test_selected_only": True,
            "ready_for_subject_level_analysis": summary[
                "ready_for_subject_level_analysis"
            ],
        }
        _write_json(
            self.output_root / "auxiliary_corn_nested_lambda_manifest.json",
            manifest,
        )
        return manifest


__all__ = [
    "AuxiliaryCornNestedLambdaExperiment",
    "NestedFoldPlan",
    "NestedLambdaPlan",
    "load_auxiliary_corn_nested_spec",
]
