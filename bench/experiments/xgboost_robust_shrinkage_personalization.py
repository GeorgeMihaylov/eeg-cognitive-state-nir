"""Nested, leakage-safe robust feature alignment for XGBoost personalization.

The experiment selects one shrinkage coefficient per real outer fold.  The
selection uses only the four remaining fixed subject folds as inner
pseudo-test folds and aggregates all seven PM targets.  The locked outer test
is evaluated only after the decision artifact has been written.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.analysis.xgboost_feature_alignment_diagnostics import (
    _prepare_read_only_base,
    _sample_hash,
    sha256_file,
)
from bench.experiments.personalization_calibration import (
    PersonalizationCalibrationPlanner,
    _participant_partition,
    stable_hash,
    validate_temporal_partition,
)
from bench.experiments.personalization_calibration_execution import (
    XGBOOST_CHECKPOINT_NAME,
    base_run_directory,
    base_unit_id,
)
from bench.tasks.target_registry import get_target_spec
from bench.tasks.target_transforms import (
    build_fold_local_target_transform,
    build_target_transform_manifest,
    stable_target_transform_hash,
    validate_target_transform_manifest,
)
from bench.validation.metrics import MetricsCalculator
from cogstate.adaptation.feature_alignment import (
    FeatureAligner,
    FeatureAlignmentConfig,
    apply_alignment_shrinkage,
)
from model_zoo import build_model
from model_zoo.ML.xgboost_personalization import xgboost_state_sha256


SCHEMA_VERSION = "xgboost-robust-shrinkage-personalization-v1"
PM_NAMES = (
    "attention", "engagement", "excitement", "stress",
    "relaxation", "interest", "focus",
)
OUTER_FOLDS = (1, 2, 3, 4, 5)
ALPHA_CANDIDATES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False, default=str,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_config(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the preregistered v1 scientific contract without relaxing it."""
    config = json.loads(json.dumps(dict(document)))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if tuple(config.get("scope", {}).get("pms", ())) != PM_NAMES:
        raise ValueError("scope.pms must contain all seven canonical PMs in order")
    if tuple(config["scope"].get("outer_folds", ())) != OUTER_FOLDS:
        raise ValueError("scope.outer_folds must remain [1, 2, 3, 4, 5]")
    if config["scope"].get("task_type") != "classification":
        raise ValueError("Only classification is supported")
    if int(config["scope"].get("target_classes", 0)) != 3:
        raise ValueError("target_classes must remain 3")
    if float(config["scope"].get("calibration_budget_fraction", -1)) != 0.2:
        raise ValueError("The confirmatory calibration budget must remain 0.20")
    alignment = config.get("alignment", {})
    if alignment.get("method") != "robust_location_scale":
        raise ValueError("alignment.method must be robust_location_scale")
    if tuple(map(float, alignment.get("alpha_candidates", ()))) != ALPHA_CANDIDATES:
        raise ValueError(f"alpha_candidates must remain {list(ALPHA_CANDIDATES)}")
    if tuple(config.get("selection", {}).get("aggregation_order", ())) != (
        "mean_pm_within_participant",
        "mean_participants_within_inner_fold",
        "mean_inner_folds",
    ):
        raise ValueError("selection.aggregation_order changed")
    if tuple(config["selection"].get("tie_break", ())) != (
        "higher_macro_f1", "higher_balanced_accuracy", "smaller_alpha",
    ):
        raise ValueError("selection.tie_break changed")
    params = config.get("model", {}).get("params", {})
    if params != {"n_estimators": 200, "n_jobs": 4, "random_state": 42}:
        raise ValueError("XGBoost parameters differ from the locked source model")
    protocol = config.get("protocol", {})
    required = {
        "outer_group_column": "subject_id",
        "fixed_outer_fold_column": "outer_fold",
        "q3_inner_fit_scope": "inner_train_only",
        "q3_outer_fit_scope": "outer_train_only",
        "calibration_split": "chronological_global_prefix",
        "outer_test_locked_until_selection": True,
        "outer_test_labels_used_for_selection": False,
        "alignment_fit_uses_labels": False,
        "evaluation_used_for_fit": False,
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError(f"protocol.{key} must remain {expected!r}")
    if not str(config.get("experiment", {}).get("experiment_id", "")).strip():
        raise ValueError("experiment.experiment_id is required")
    if config["experiment"].get("result_status") != "confirmatory":
        raise ValueError("result_status must be confirmatory")
    if not str(config.get("source", {}).get("protocol_hash", "")):
        raise ValueError("source.protocol_hash is required")
    if not str(config["source"].get("plan_hash", "")):
        raise ValueError("source.plan_hash is required")
    return config


def protocol_hash(config: Mapping[str, Any]) -> str:
    """Hash scientific configuration while excluding only artifact placement."""
    payload = deepcopy(dict(config))
    payload["experiment"] = dict(payload["experiment"])
    payload["experiment"].pop("output_dir", None)
    return stable_hash(payload)


def build_full_plan(config: Mapping[str, Any]) -> tuple[pd.DataFrame, str, str]:
    """Build the immutable 140-inner/35-outer plan."""
    phash = protocol_hash(config)
    rows: list[dict[str, Any]] = []
    for outer_fold in OUTER_FOLDS:
        for pseudo_fold in OUTER_FOLDS:
            if pseudo_fold == outer_fold:
                continue
            for pm in PM_NAMES:
                identity = {
                    "phase": "inner_model", "outer_fold": outer_fold,
                    "inner_pseudo_test_fold": pseudo_fold, "pm": pm,
                }
                rows.append({**identity, "unit_id": stable_hash(identity)[:20]})
        for pm in PM_NAMES:
            identity = {
                "phase": "outer_evaluation", "outer_fold": outer_fold,
                "inner_pseudo_test_fold": None, "pm": pm,
            }
            rows.append({**identity, "unit_id": stable_hash(identity)[:20]})
    frame = pd.DataFrame(rows).sort_values(
        ["outer_fold", "phase", "inner_pseudo_test_fold", "pm"],
        kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    plan_hash = stable_hash({
        "protocol_hash": phash,
        "rows": frame.fillna("").to_dict("records"),
    })
    return frame, phash, plan_hash


def aggregate_candidate_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the preregistered PM -> participant -> inner-fold aggregation."""
    required = {
        "inner_pseudo_test_fold", "subject_id", "pm", "alpha", *METRIC_NAMES,
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Candidate scores are missing columns: {missing}")
    participant = scores.groupby(
        ["inner_pseudo_test_fold", "subject_id", "alpha"], sort=True,
        as_index=False,
    ).agg(
        **{metric: (metric, "mean") for metric in METRIC_NAMES},
        pm_count=("pm", "nunique"),
    )
    inner = participant.groupby(
        ["inner_pseudo_test_fold", "alpha"], sort=True, as_index=False,
    ).agg(
        **{metric: (metric, "mean") for metric in METRIC_NAMES},
        participant_count=("subject_id", "nunique"),
        participant_pm_count_min=("pm_count", "min"),
        participant_pm_count_max=("pm_count", "max"),
    )
    summary = inner.groupby("alpha", sort=True, as_index=False).agg(
        **{metric: (metric, "mean") for metric in METRIC_NAMES},
        inner_fold_count=("inner_pseudo_test_fold", "nunique"),
        participants_per_fold_min=("participant_count", "min"),
        participants_per_fold_max=("participant_count", "max"),
    )
    return inner, summary


def select_alpha(summary: pd.DataFrame, *, tolerance: float) -> dict[str, Any]:
    """Select by macro F1, balanced accuracy, then the smaller alpha."""
    if summary.empty:
        raise ValueError("Candidate summary is empty")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    best_macro = float(summary["macro_f1"].max())
    eligible = summary.loc[summary["macro_f1"] >= best_macro - tolerance]
    best_balanced = float(eligible["balanced_accuracy"].max())
    finalists = eligible.loc[
        eligible["balanced_accuracy"] >= best_balanced - tolerance
    ].sort_values("alpha", kind="mergesort")
    selected = finalists.iloc[0]
    return {
        "selected_alpha": float(selected["alpha"]),
        "selected_macro_f1": float(selected["macro_f1"]),
        "selected_balanced_accuracy": float(selected["balanced_accuracy"]),
        "macro_f1_optimum": best_macro,
        "tie_tolerance": float(tolerance),
        "tie_break_applied": int(len(finalists)) > 1,
    }


def evaluate_alignment_candidates(
    estimator: Any,
    *,
    X_reference: np.ndarray,
    X_calibration: np.ndarray,
    X_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    alphas: Sequence[float] = ALPHA_CANDIDATES,
    scale_epsilon: float = 1e-12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit target-free statistics once and evaluate one frozen estimator."""
    aligner = FeatureAligner(FeatureAlignmentConfig(
        method="robust_location_scale", scale_epsilon=float(scale_epsilon),
    )).fit_reference(X_reference).fit_calibration(X_calibration)
    before = xgboost_state_sha256(estimator)
    aligned = aligner.transform(X_evaluation)
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        transformed = apply_alignment_shrinkage(X_evaluation, aligned, float(alpha))
        probabilities = np.asarray(estimator.predict_proba(transformed), dtype=float)
        predictions = np.asarray(estimator.classes_)[np.argmax(probabilities, axis=1)]
        metrics = MetricsCalculator.calculate_all_metrics(
            y_evaluation, predictions, probabilities,
            task_type="classification", labels=np.arange(3),
        )
        rows.append({"alpha": float(alpha), **{
            metric: float(metrics[metric]) for metric in METRIC_NAMES
        }})
    after = xgboost_state_sha256(estimator)
    if before != after:
        raise RuntimeError("Frozen XGBoost booster changed during alignment evaluation")
    return rows, {**aligner.to_manifest(), "booster_hash_before": before,
                  "booster_hash_after": after}


@dataclass(frozen=True)
class TrainBundle:
    X: np.ndarray
    sample_ids: np.ndarray
    subjects: np.ndarray
    fixed_folds: np.ndarray
    feature_names: tuple[str, ...]


class RobustShrinkagePersonalizationExperiment:
    """Plan, validate, and execute the nested confirmatory protocol."""

    def __init__(self, config_path: str | Path, *, output_dir: str | Path | None = None):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config_path = _resolve(self.repo_root, config_path)
        self.config = validate_config(_read_json(self.config_path, "experiment config"))
        configured = self.config["experiment"]["output_dir"]
        self.output_dir = _resolve(self.repo_root, configured) if output_dir is None else Path(output_dir).resolve()
        self.plan_frame, self.protocol_hash, self.plan_hash = build_full_plan(self.config)
        self.source_planner: PersonalizationCalibrationPlanner | None = None
        self.source_matrix: pd.DataFrame | None = None
        self.source_participants: pd.DataFrame | None = None
        self.source_plan_manifest: dict[str, Any] | None = None

    def _load_source(self) -> None:
        if self.source_planner is not None:
            return
        source = self.config["source"]
        plan_dir = _resolve(self.repo_root, source["plan_dir"])
        run_dir = _resolve(self.repo_root, source["run_dir"])
        source_config = _resolve(self.repo_root, source["personalization_config"])
        manifest = _read_json(plan_dir / "protocol_manifest.json", "source plan manifest")
        execution = _read_json(run_dir / "execution_manifest.json", "source execution manifest")
        if manifest.get("protocol_hash") != source["protocol_hash"]:
            raise RuntimeError("Configured source protocol hash does not match source plan")
        if manifest.get("plan_hash") != source["plan_hash"]:
            raise RuntimeError("Configured source plan hash does not match source plan")
        if execution.get("protocol_hash") != source["protocol_hash"] or execution.get("plan_hash") != source["plan_hash"]:
            raise RuntimeError("Source execution provenance differs from source plan")
        planner = PersonalizationCalibrationPlanner(
            source_config, data_root=self.repo_root, output_dir=run_dir,
        )
        if planner.protocol_hash != source["protocol_hash"]:
            raise RuntimeError("Current source config no longer reproduces source protocol")
        matrix = pd.read_csv(plan_dir / "run_matrix.csv")
        participants = pd.read_csv(plan_dir / "participant_calibration_plan.csv")
        self.source_planner = planner
        self.source_matrix = matrix
        self.source_participants = participants
        self.source_plan_manifest = manifest

    def _base_row(self, pm: str, outer_fold: int) -> dict[str, Any]:
        assert self.source_matrix is not None
        selected = self.source_matrix.loc[
            self.source_matrix["pm"].eq(pm)
            & self.source_matrix["task_type"].eq("classification")
            & self.source_matrix["model"].eq("xgboost")
            & self.source_matrix["outer_fold"].eq(outer_fold)
            & self.source_matrix["mode"].eq("zero_shot")
            & self.source_matrix["budget_fraction"].eq(0.0)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one source base for {pm}/fold {outer_fold}")
        row = selected.iloc[0].to_dict()
        row["outer_fold"] = int(row["outer_fold"])
        row["seed"] = int(row["seed"])
        row["budget_fraction"] = float(row["budget_fraction"])
        return row

    def plan(self, *, write_artifacts: bool, resume: bool = False) -> dict[str, Any]:
        self._load_source()
        inner_count = int(self.plan_frame["phase"].eq("inner_model").sum())
        outer_count = int(self.plan_frame["phase"].eq("outer_evaluation").sum())
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.config["experiment"]["experiment_id"],
            "result_status": self.config["experiment"]["result_status"],
            "training_executed": False,
            "protocol_hash": self.protocol_hash,
            "plan_hash": self.plan_hash,
            "source_protocol_hash": self.config["source"]["protocol_hash"],
            "source_plan_hash": self.config["source"]["plan_hash"],
            "inner_model_units": inner_count,
            "outer_evaluation_units": outer_count,
            "total_units": int(len(self.plan_frame)),
            "outer_folds": list(OUTER_FOLDS),
            "pms": list(PM_NAMES),
            "alpha_candidates": list(ALPHA_CANDIDATES),
            "outer_execution_filter_affects_hash": False,
        }
        if write_artifacts:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "protocol_manifest.json"
            if path.is_file():
                old = _read_json(path, "existing protocol manifest")
                if old.get("protocol_hash") != self.protocol_hash or old.get("plan_hash") != self.plan_hash:
                    raise RuntimeError("Existing output has incompatible protocol/plan hash")
                if not resume:
                    raise FileExistsError("Plan exists; pass --resume for idempotent rewrite")
            self.plan_frame.to_csv(self.output_dir / "run_matrix.csv", index=False)
            _write_json(self.output_dir / "resolved_config.json", self.config)
            _write_json(path, manifest)
        return manifest

    def _source_base_precheck(self, pm: str, outer_fold: int) -> dict[str, Any]:
        assert self.source_planner is not None
        row = self._base_row(pm, outer_fold)
        directory = base_run_directory(self.source_planner.output_dir, base_unit_id(row))
        checkpoint = directory / XGBOOST_CHECKPOINT_NAME
        manifest = directory / "base_checkpoint_manifest.json"
        result_files = sorted(directory.glob("benchmark_results_*.json"))
        if not checkpoint.is_file() or not manifest.is_file() or not result_files:
            raise FileNotFoundError(f"Exact completed source base is incomplete: {directory}")
        return {
            "pm": pm, "outer_fold": outer_fold, "base_directory": str(directory),
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "manifest_sha256": sha256_file(manifest), "completed_result_count": len(result_files),
        }

    def dry_run(self, *, outer_fold: int | None = None) -> dict[str, Any]:
        self._load_source()
        folds = OUTER_FOLDS if outer_fold is None else (int(outer_fold),)
        if any(fold not in OUTER_FOLDS for fold in folds):
            raise ValueError("outer_fold must be in 1..5")
        bases = [self._source_base_precheck(pm, fold) for fold in folds for pm in PM_NAMES]
        inner = self.plan_frame.loc[
            self.plan_frame["phase"].eq("inner_model")
            & self.plan_frame["outer_fold"].isin(folds)
        ]
        return {
            "protocol_hash": self.protocol_hash, "plan_hash": self.plan_hash,
            "execution_outer_folds": list(folds),
            "inner_models_required_or_resumable": int(len(inner)),
            "outer_bases_read_only_reusable": int(len(bases)),
            "outer_base_checks": bases,
            "training_executed": False,
        }

    @staticmethod
    def _train_bundle(handle: Any) -> TrainBundle:
        split = handle.split
        X = np.asarray(split.X_train)
        sample_ids = np.asarray(split.sample_id_train).astype(str)
        subjects = np.asarray(split.subject_train).astype(str)
        fixed_folds = np.asarray(split.row_metadata_train.get("outer_fold"), dtype=int)
        names = tuple(map(str, split.feature_names or ()))
        if X.ndim != 2 or len(X) != len(sample_ids) or len(X) != len(subjects) or len(X) != len(fixed_folds):
            raise RuntimeError("Malformed train-only feature bundle")
        if len(names) != X.shape[1] or len(set(names)) != len(names):
            raise RuntimeError("Feature names are missing or duplicated")
        return TrainBundle(X, sample_ids, subjects, fixed_folds, names)

    @staticmethod
    def _subset(bundle: TrainBundle, ids: Sequence[Any]) -> np.ndarray:
        lookup = {value: index for index, value in enumerate(bundle.sample_ids)}
        normalized = np.asarray(ids).astype(str)
        missing = [value for value in normalized if value not in lookup]
        if missing:
            raise RuntimeError(f"Samples are absent from authorized train bundle: {missing[:5]}")
        return bundle.X[np.asarray([lookup[value] for value in normalized], dtype=int)]

    @staticmethod
    def _inner_transform(pm: str, real_outer: int, pseudo_fold: int, frame: pd.DataFrame, train_mask: np.ndarray) -> tuple[Any, dict[str, Any]]:
        spec = get_target_spec(f"pm_{pm}_q3_fold_local")
        transform = build_fold_local_target_transform(spec).fit(frame.loc[train_mask, "target_value"].to_numpy())
        manifest = build_target_transform_manifest(
            spec, transform, outer_fold=pseudo_fold,
            outer_train_sample_ids=frame.loc[train_mask, "sample_id"].to_numpy(),
            outer_train_targets=frame.loc[train_mask, "target_value"].to_numpy(),
        )
        manifest.update({
            "fit_scope": "inner_train_only", "nested_real_outer_fold": real_outer,
            "inner_pseudo_test_fold": pseudo_fold,
        })
        manifest["transform_hash"] = stable_target_transform_hash(manifest)
        validate_target_transform_manifest(manifest)
        if manifest["actual_class_count"] != 3:
            raise RuntimeError("Inner-train Q3 did not preserve three classes")
        return transform, manifest

    def _inner_model(
        self, *, outer_fold: int, pseudo_fold: int, pm: str,
        bundle: TrainBundle, frame: pd.DataFrame, transform: Any,
        target_manifest: Mapping[str, Any], resume: bool,
    ) -> tuple[Any, dict[str, Any]]:
        inner_train = bundle.fixed_folds != pseudo_fold
        pseudo = bundle.fixed_folds == pseudo_fold
        if set(bundle.subjects[inner_train]) & set(bundle.subjects[pseudo]):
            raise RuntimeError("Inner train/pseudo-test subject overlap")
        if outer_fold in set(bundle.fixed_folds.tolist()):
            raise RuntimeError("Real outer-test fold entered inner train bundle")
        target_lookup = frame.set_index(frame["sample_id"].astype(str))["target_value"]
        try:
            y_continuous = target_lookup.loc[bundle.sample_ids].to_numpy(dtype=float)
        except KeyError as exc:
            raise RuntimeError("Target frame does not match feature train bundle") from exc
        y_train = transform.transform(y_continuous[inner_train]).astype(int)
        params = dict(self.config["model"]["params"])
        spec = {
            "protocol_hash": self.protocol_hash, "plan_hash": self.plan_hash,
            "outer_fold": outer_fold, "inner_pseudo_test_fold": pseudo_fold,
            "pm": pm, "model": "xgboost", "params": params,
            "train_sample_hash": _sample_hash(bundle.sample_ids[inner_train]),
            "pseudo_test_sample_hash": _sample_hash(bundle.sample_ids[pseudo]),
            "train_subject_hash": stable_hash(sorted(set(bundle.subjects[inner_train]))),
            "pseudo_test_subject_hash": stable_hash(sorted(set(bundle.subjects[pseudo]))),
            "target_transform_hash": target_manifest["transform_hash"],
            "feature_names_hash": stable_hash(list(bundle.feature_names)),
        }
        specification_hash = stable_hash(spec)
        subject_identity = {
            "train_subject_ids": sorted(set(bundle.subjects[inner_train])),
            "pseudo_test_subject_ids": sorted(set(bundle.subjects[pseudo])),
        }
        directory = self.output_dir / f"fold_{outer_fold:02d}" / "inner_models" / f"pseudo_{pseudo_fold:02d}" / pm
        checkpoint = directory / XGBOOST_CHECKPOINT_NAME
        manifest_path = directory / "manifest.json"
        resumed = False
        if resume and checkpoint.is_file() and manifest_path.is_file():
            saved = _read_json(manifest_path, "inner model manifest")
            if saved.get("status") != "complete" or saved.get("specification_hash") != specification_hash:
                raise RuntimeError(f"Stale inner model cache: {directory}")
            model = build_model("xgboost", "classification", None, 3, params)
            model.load_model(checkpoint)
            resumed = True
            training_time = float(saved["training_time_seconds"])
        else:
            if checkpoint.exists() or manifest_path.exists():
                raise FileExistsError(f"Incomplete or non-resumable inner unit: {directory}")
            model = build_model("xgboost", "classification", None, 3, params)
            started = time.perf_counter()
            model.fit(bundle.X[inner_train], y_train)
            training_time = time.perf_counter() - started
            directory.mkdir(parents=True, exist_ok=True)
            model.save_model(checkpoint)
            _write_json(manifest_path, {
                **spec, **subject_identity, "specification_hash": specification_hash,
                "status": "complete", "training_time_seconds": training_time,
                "checkpoint_sha256": sha256_file(checkpoint),
                "booster_hash": xgboost_state_sha256(model),
                "train_samples": int(inner_train.sum()),
                "pseudo_test_samples": int(pseudo.sum()),
                "train_subjects": int(len(set(bundle.subjects[inner_train]))),
                "pseudo_test_subjects": int(len(set(bundle.subjects[pseudo]))),
            })
        return model, {
            **spec, **subject_identity, "specification_hash": specification_hash,
            "resumed": resumed,
            "training_time_seconds": training_time,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "booster_hash": xgboost_state_sha256(model),
            "train_samples": int(inner_train.sum()), "pseudo_test_samples": int(pseudo.sum()),
            "train_subject_overlap": int(len(set(bundle.subjects[inner_train]) & set(bundle.subjects[pseudo]))),
            "real_outer_fold_present": bool(outer_fold in set(bundle.fixed_folds.tolist())),
        }

    def _inner_scores_for_pm(
        self, *, outer_fold: int, pm: str, handle: Any, frame: pd.DataFrame,
        resume: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        assert self.source_participants is not None
        bundle = self._train_bundle(handle)
        if set(bundle.sample_ids) != set(frame.loc[frame["outer_fold"] != outer_fold, "sample_id"].astype(str)):
            raise RuntimeError("Train-only feature and continuous-target cohorts differ")
        outer_test_ids = set(np.asarray(handle.split.sample_id_test).astype(str))
        outer_test_subjects = set(np.asarray(handle.split.subject_test).astype(str))
        if set(bundle.sample_ids) & outer_test_ids:
            raise RuntimeError("Real outer-test sample IDs entered inner selection bundle")
        if set(bundle.subjects) & outer_test_subjects:
            raise RuntimeError("Real outer-test subjects entered inner selection bundle")
        scores: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for pseudo_fold in OUTER_FOLDS:
            if pseudo_fold == outer_fold:
                continue
            train_mask_frame = ~frame["outer_fold"].isin([outer_fold, pseudo_fold])
            transform, target_manifest = self._inner_transform(
                pm, outer_fold, pseudo_fold, frame, train_mask_frame.to_numpy(),
            )
            model, model_audit = self._inner_model(
                outer_fold=outer_fold, pseudo_fold=pseudo_fold, pm=pm,
                bundle=bundle, frame=frame, transform=transform,
                target_manifest=target_manifest, resume=resume,
            )
            reference_X = bundle.X[bundle.fixed_folds != pseudo_fold]
            eligible = self.source_participants.loc[
                self.source_participants["pm"].eq(pm)
                & self.source_participants["outer_fold"].eq(pseudo_fold)
                & np.isclose(self.source_participants["budget_fraction"], 0.2)
                & self.source_participants["status"].eq("planned")
            ].sort_values("subject_id", kind="mergesort")
            booster_before = xgboost_state_sha256(model)
            for row in eligible.to_dict("records"):
                subject = frame.loc[
                    frame["outer_fold"].eq(pseudo_fold)
                    & frame["subject_id"].astype(str).eq(str(row["subject_id"]))
                ]
                partition, _ = _participant_partition(
                    subject, budget=0.2, reference_budget=0.2,
                    protocol=self.source_planner.config["protocol"],
                )
                temporal = validate_temporal_partition(partition)
                cal_ids = partition.calibration_metadata["sample_id"].astype(str).to_numpy()
                eval_ids = partition.evaluation_metadata["sample_id"].astype(str).to_numpy()
                if _sample_hash(cal_ids) != str(row["calibration_sample_hash"]) or _sample_hash(eval_ids) != str(row["evaluation_sample_hash"]):
                    raise RuntimeError("Nested participant partition differs from fixed source plan")
                if set(cal_ids) & set(eval_ids):
                    raise RuntimeError("Nested calibration/evaluation overlap")
                X_cal = self._subset(bundle, cal_ids)
                X_eval = self._subset(bundle, eval_ids)
                target_lookup = frame.set_index(frame["sample_id"].astype(str))["target_value"]
                y_eval = transform.transform(target_lookup.loc[eval_ids].to_numpy(dtype=float)).astype(int)
                candidate_rows, alignment_audit = evaluate_alignment_candidates(
                    model, X_reference=reference_X, X_calibration=X_cal,
                    X_evaluation=X_eval, y_evaluation=y_eval,
                    alphas=ALPHA_CANDIDATES,
                    scale_epsilon=float(self.config["alignment"]["scale_epsilon"]),
                )
                for candidate in candidate_rows:
                    scores.append({
                        "outer_fold": outer_fold, "inner_pseudo_test_fold": pseudo_fold,
                        "pm": pm, "subject_id": str(row["subject_id"]),
                        "calibration_windows": len(cal_ids), "evaluation_windows": len(eval_ids),
                        "calibration_sample_hash": _sample_hash(cal_ids),
                        "evaluation_sample_hash": _sample_hash(eval_ids),
                        **candidate,
                    })
                audits.append({
                    **model_audit, "subject_id": str(row["subject_id"]),
                    "target_transform_hash": target_manifest["transform_hash"],
                    "target_fit_sample_hash": target_manifest["outer_train_sample_hash"],
                    "calibration_sample_hash": _sample_hash(cal_ids),
                    "evaluation_sample_hash": _sample_hash(eval_ids),
                    "calibration_evaluation_overlap": 0,
                    "calibration_before_evaluation": temporal["calibration_before_evaluation"],
                    "reference_stats_hash": alignment_audit["reference_stats_hash"],
                    "calibration_stats_hash": alignment_audit["calibration_stats_hash"],
                    "booster_hash_after_alignment": alignment_audit["booster_hash_after"],
                    "feature_alignment_mutated_model": False,
                    "evaluation_used_for_fit": False,
                    "outer_test_sample_overlap": 0,
                    "outer_test_subject_overlap": 0,
                })
            if xgboost_state_sha256(model) != booster_before:
                raise RuntimeError("Inner model changed while evaluating alpha candidates")
        return scores, audits

    def _outer_evaluate(
        self, *, outer_fold: int, selected_alpha: float,
        handles: Mapping[str, Any], target_frames: Mapping[str, pd.DataFrame],
        base_audits: Mapping[str, Mapping[str, Any]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        assert self.source_participants is not None
        rows: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        pm_audits: dict[str, Any] = {}
        for pm in PM_NAMES:
            handle = handles[pm]
            split = handle.split
            estimator = handle.adapter.global_model
            booster_before = xgboost_state_sha256(estimator)
            reference = FeatureAligner(FeatureAlignmentConfig(
                method="robust_location_scale",
                scale_epsilon=float(self.config["alignment"]["scale_epsilon"]),
            )).fit_reference(np.asarray(split.X_train))
            eligible = self.source_participants.loc[
                self.source_participants["pm"].eq(pm)
                & self.source_participants["outer_fold"].eq(outer_fold)
                & np.isclose(self.source_participants["budget_fraction"], 0.2)
                & self.source_participants["status"].eq("planned")
            ].sort_values("subject_id", kind="mergesort")
            frame = target_frames[pm]
            test_lookup = {value: index for index, value in enumerate(np.asarray(split.sample_id_test).astype(str))}
            participant_audits: list[dict[str, Any]] = []
            for source_row in eligible.to_dict("records"):
                subject_id = str(source_row["subject_id"])
                subject = frame.loc[frame["outer_fold"].eq(outer_fold) & frame["subject_id"].astype(str).eq(subject_id)]
                partition, _ = _participant_partition(
                    subject, budget=0.2, reference_budget=0.2,
                    protocol=self.source_planner.config["protocol"],
                )
                temporal = validate_temporal_partition(partition)
                cal_ids = partition.calibration_metadata["sample_id"].astype(str).to_numpy()
                eval_ids = partition.evaluation_metadata["sample_id"].astype(str).to_numpy()
                if _sample_hash(cal_ids) != str(source_row["calibration_sample_hash"]) or _sample_hash(eval_ids) != str(source_row["evaluation_sample_hash"]):
                    raise RuntimeError("Outer participant partition differs from source plan")
                if str(source_row["q3_transform_hash"]) != handle.target_transform_hash:
                    raise RuntimeError("Outer participant Q3 hash differs from source base")
                indices_cal = np.asarray([test_lookup[value] for value in cal_ids], dtype=int)
                indices_eval = np.asarray([test_lookup[value] for value in eval_ids], dtype=int)
                X_cal = np.asarray(split.X_test)[indices_cal]
                X_eval = np.asarray(split.X_test)[indices_eval]
                y_eval = np.asarray(split.y_test, dtype=int)[indices_eval]
                aligner = deepcopy(reference).fit_calibration(X_cal)
                aligned = aligner.transform(X_eval)
                adapted_X = apply_alignment_shrinkage(X_eval, aligned, selected_alpha)
                zero_proba = np.asarray(estimator.predict_proba(X_eval), dtype=float)
                adapted_proba = np.asarray(estimator.predict_proba(adapted_X), dtype=float)
                classes = np.asarray(estimator.classes_)
                zero_pred = classes[np.argmax(zero_proba, axis=1)]
                adapted_pred = classes[np.argmax(adapted_proba, axis=1)]
                zero_metrics = MetricsCalculator.calculate_all_metrics(y_eval, zero_pred, zero_proba, task_type="classification", labels=np.arange(3))
                adapted_metrics = MetricsCalculator.calculate_all_metrics(y_eval, adapted_pred, adapted_proba, task_type="classification", labels=np.arange(3))
                metric_row: dict[str, Any] = {
                    "outer_fold": outer_fold, "pm": pm, "subject_id": subject_id,
                    "selected_alpha": selected_alpha, "calibration_windows": len(cal_ids),
                    "evaluation_windows": len(eval_ids), "calibration_sample_hash": _sample_hash(cal_ids),
                    "evaluation_sample_hash": _sample_hash(eval_ids),
                    "target_transform_hash": handle.target_transform_hash,
                }
                for metric in METRIC_NAMES:
                    metric_row[f"zero_shot_{metric}"] = float(zero_metrics[metric])
                    metric_row[f"adapted_{metric}"] = float(adapted_metrics[metric])
                    metric_row[f"delta_{metric}"] = float(adapted_metrics[metric] - zero_metrics[metric])
                rows.append(metric_row)
                for position, sample_id in enumerate(eval_ids):
                    prediction = {
                        "sample_id": sample_id, "outer_fold": outer_fold, "pm": pm,
                        "subject_id": subject_id, "y_true": int(y_eval[position]),
                        "zero_shot_y_pred": int(zero_pred[position]),
                        "adapted_y_pred": int(adapted_pred[position]),
                        "selected_alpha": selected_alpha,
                    }
                    for class_id in range(3):
                        prediction[f"zero_shot_proba_{class_id}"] = float(zero_proba[position, class_id])
                        prediction[f"adapted_proba_{class_id}"] = float(adapted_proba[position, class_id])
                    predictions.append(prediction)
                manifest = aligner.to_manifest()
                participant_audits.append({
                    "subject_id": subject_id, "calibration_evaluation_overlap": int(len(set(cal_ids) & set(eval_ids))),
                    "calibration_before_evaluation": temporal["calibration_before_evaluation"],
                    "reference_stats_hash": manifest["reference_stats_hash"],
                    "calibration_stats_hash": manifest["calibration_stats_hash"],
                })
            booster_after = xgboost_state_sha256(estimator)
            if booster_after != booster_before:
                raise RuntimeError("Source outer XGBoost booster changed")
            checkpoint = Path(handle.checkpoint_path)
            pm_audits[pm] = {
                **base_audits[pm], "checkpoint_sha256_after": sha256_file(checkpoint),
                "booster_hash_before": booster_before, "booster_hash_after": booster_after,
                "source_base_unchanged": sha256_file(checkpoint) == base_audits[pm]["base_checkpoint_sha256"],
                "outer_train_subject_overlap": 0,
                "participant_audits": participant_audits,
            }
        result = pd.DataFrame(rows)
        prediction_frame = pd.DataFrame(predictions)
        if prediction_frame["sample_id"].duplicated().any():
            # The same sample is legitimately present once per PM; uniqueness is
            # therefore PM + sample_id, never sample_id alone across targets.
            if prediction_frame.duplicated(["pm", "sample_id"]).any():
                raise RuntimeError("Duplicate PM/sample_id outer predictions")
        audit = {
            "outer_fold": outer_fold, "selected_alpha": selected_alpha,
            "outer_test_opened_after_selection_decision": True,
            "outer_subject_overlap": 0,
            "calibration_evaluation_overlap_max": max(
                item["calibration_evaluation_overlap"]
                for pm in pm_audits.values() for item in pm["participant_audits"]
            ),
            "all_calibration_before_evaluation": all(
                item["calibration_before_evaluation"]
                for pm in pm_audits.values() for item in pm["participant_audits"]
            ),
            "source_bases": pm_audits,
        }
        return result, prediction_frame, audit

    @staticmethod
    def _outer_summary(frame: pd.DataFrame) -> dict[str, Any]:
        per_pm: dict[str, Any] = {}
        for pm, group in frame.groupby("pm", sort=False):
            per_pm[str(pm)] = {
                metric: float(group[metric].mean())
                for metric in frame.columns if metric.startswith(("zero_shot_", "adapted_", "delta_"))
            }
            per_pm[str(pm)]["participant_positive_fraction_macro_f1"] = float(
                (group["delta_macro_f1"] > 0.0).mean()
            )
            per_pm[str(pm)]["participant_positive_fraction_balanced_accuracy"] = float(
                (group["delta_balanced_accuracy"] > 0.0).mean()
            )
        participant = frame.groupby(["outer_fold", "subject_id"], as_index=False).agg(
            **{column: (column, "mean") for column in frame.columns if column.startswith(("zero_shot_", "adapted_", "delta_"))},
            pm_count=("pm", "nunique"),
        )
        overall = {column: float(participant[column].mean()) for column in participant.columns if column.startswith(("zero_shot_", "adapted_", "delta_"))}
        positive = {
            "macro_f1": float((participant["delta_macro_f1"] > 0.0).mean()),
            "balanced_accuracy": float(
                (participant["delta_balanced_accuracy"] > 0.0).mean()
            ),
            "accuracy": float((participant["delta_accuracy"] > 0.0).mean()),
        }
        return {"overall_participant_macro": overall, "per_pm_participant_macro": per_pm,
                "participant_positive_fraction": positive,
                "participants": int(participant["subject_id"].nunique()),
                "participant_pm_count_min": int(participant["pm_count"].min()),
                "participant_pm_count_max": int(participant["pm_count"].max())}

    def run(self, *, outer_fold: int, resume: bool) -> dict[str, Any]:
        if outer_fold not in OUTER_FOLDS:
            raise ValueError("Smoke execution outer_fold must be in 1..5")
        self._load_source()
        self.plan(write_artifacts=True, resume=resume)
        fold_dir = self.output_dir / f"fold_{outer_fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        handles: dict[str, Any] = {}
        base_audits: dict[str, Any] = {}
        target_frames: dict[str, pd.DataFrame] = {}
        all_scores: list[dict[str, Any]] = []
        all_inner_audits: list[dict[str, Any]] = []
        source_plan_hash = self.config["source"]["plan_hash"]
        for pm in PM_NAMES:
            base = self._base_row(pm, outer_fold)
            _, handle, audit = _prepare_read_only_base(
                self.source_planner, plan_hash=source_plan_hash, base=base,
            )
            if not handle.resumed:
                raise RuntimeError("Final outer base was not resumed read-only")
            handles[pm] = handle
            base_audits[pm] = audit
            frame = self.source_planner._load_target_frame(pm)
            target_frames[pm] = frame
            scores, audits = self._inner_scores_for_pm(
                outer_fold=outer_fold, pm=pm, handle=handle, frame=frame,
                resume=resume,
            )
            all_scores.extend(scores)
            all_inner_audits.extend(audits)
            pd.DataFrame(scores).to_csv(fold_dir / f"inner_candidate_scores_{pm}.csv", index=False)
        score_frame = pd.DataFrame(all_scores)
        score_frame.to_csv(fold_dir / "inner_candidate_scores.csv", index=False)
        inner_summary, candidate_summary = aggregate_candidate_scores(score_frame)
        inner_summary.to_csv(fold_dir / "inner_fold_candidate_summary.csv", index=False)
        candidate_summary.to_csv(fold_dir / "candidate_summary.csv", index=False)
        decision = {
            "schema_version": SCHEMA_VERSION,
            "protocol_hash": self.protocol_hash, "plan_hash": self.plan_hash,
            "outer_fold": outer_fold,
            "selection_data_scope": "real_outer_train_only",
            "outer_test_opened": False,
            "aggregation_order": self.config["selection"]["aggregation_order"],
            "tie_break": self.config["selection"]["tie_break"],
            **select_alpha(candidate_summary, tolerance=float(self.config["selection"]["numeric_tolerance"])),
        }
        _write_json(fold_dir / "selection_decision.json", decision)
        _write_json(fold_dir / "inner_leakage_audit.json", {
            "outer_fold": outer_fold,
            "participant_audit_rows": len(all_inner_audits),
            "inner_model_units": len(set(item["specification_hash"] for item in all_inner_audits)),
            "max_inner_train_pseudo_subject_overlap": max(item["train_subject_overlap"] for item in all_inner_audits),
            "real_outer_fold_present_in_any_inner_bundle": any(item["real_outer_fold_present"] for item in all_inner_audits),
            "max_calibration_evaluation_overlap": max(item["calibration_evaluation_overlap"] for item in all_inner_audits),
            "max_outer_test_sample_overlap": max(item["outer_test_sample_overlap"] for item in all_inner_audits),
            "max_outer_test_subject_overlap": max(item["outer_test_subject_overlap"] for item in all_inner_audits),
            "all_calibration_before_evaluation": all(item["calibration_before_evaluation"] for item in all_inner_audits),
            "unique_inner_specifications": len(set(item["specification_hash"] for item in all_inner_audits)),
            "expected_inner_specifications": 28,
            "participant_audits": all_inner_audits,
        })
        outer, predictions, outer_audit = self._outer_evaluate(
            outer_fold=outer_fold, selected_alpha=decision["selected_alpha"],
            handles=handles, target_frames=target_frames, base_audits=base_audits,
        )
        outer.to_csv(fold_dir / "outer_participant_results.csv", index=False)
        predictions.to_parquet(fold_dir / "outer_predictions.parquet", index=False)
        _write_json(fold_dir / "outer_leakage_audit.json", outer_audit)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.config["experiment"]["experiment_id"],
            "result_status": "smoke" if outer_fold == 1 else "confirmatory_partial",
            "protocol_hash": self.protocol_hash, "plan_hash": self.plan_hash,
            "executed_outer_folds": [outer_fold], "full_five_fold_complete": False,
            "selected_alpha": decision["selected_alpha"],
            "candidate_summary": candidate_summary.to_dict("records"),
            "inner_models_expected": 28,
            "inner_models_resumed": int(pd.DataFrame(all_inner_audits).drop_duplicates("specification_hash")["resumed"].sum()),
            "inner_training_time_seconds": float(pd.DataFrame(all_inner_audits).drop_duplicates("specification_hash")["training_time_seconds"].sum()),
            "outer_bases_reused": 7,
            **self._outer_summary(outer),
            "leakage_audit": {
                "inner_train_pseudo_subject_overlap_max": 0,
                "real_outer_fold_in_inner": False,
                "outer_subject_overlap": outer_audit["outer_subject_overlap"],
                "calibration_evaluation_overlap_max": outer_audit["calibration_evaluation_overlap_max"],
                "outer_test_used_for_selection": False,
                "source_bases_unchanged": all(value["source_base_unchanged"] for value in outer_audit["source_bases"].values()),
            },
        }
        _write_json(fold_dir / "fold_summary.json", summary)
        _write_json(self.output_dir / "execution_summary.json", summary)
        return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    experiment = RobustShrinkagePersonalizationExperiment(
        args.config, output_dir=args.output_dir,
    )
    if args.plan_only:
        result = experiment.plan(write_artifacts=True, resume=args.resume)
    elif args.dry_run:
        result = experiment.dry_run(outer_fold=args.outer_fold)
    else:
        if args.outer_fold is None:
            raise SystemExit("--run requires explicit --outer-fold; full execution is intentionally not implicit")
        result = experiment.run(outer_fold=args.outer_fold, resume=args.resume)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALPHA_CANDIDATES", "PM_NAMES", "RobustShrinkagePersonalizationExperiment",
    "aggregate_candidate_scores", "build_full_plan", "evaluate_alignment_candidates",
    "protocol_hash", "select_alpha", "validate_config",
]
