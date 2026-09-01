"""Seven-PM factorial A--H raw-EEG preprocessing ablation.

Planning reuses the established :mod:`preprocessing_ablation` cache registry
and performs no fitting.  Scientific execution delegates every fit to
``BenchmarkRunner`` with frozen outer-train-only Q3 transforms.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from bench.bench_runner import BenchmarkRunner
from bench.datasets.target_view import sample_id_filter_hash
from bench.experiments.preprocessing_ablation import (
    ExperimentTrial,
    PreprocessingAblation,
    TrialPlan,
)
from bench.tasks.target_registry import get_target_spec
from bench.tasks.target_transforms import (
    FoldLocalQuantileTargetTransform,
    build_target_transform_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "preprocessing-factorial-q3-all-pm-v1"
EXPERIMENT_ID = "preprocessing_factorial_q3_all_pm_v1"
PM_NAMES = (
    "attention", "engagement", "excitement", "stress",
    "relaxation", "interest", "focus",
)
VARIANTS = tuple("ABCDEFGH")
FOLDS = (1, 2, 3, 4, 5)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
RUNNER_MODEL_ALIAS = "m"
EXPECTED_COMPLETE_CASES = {
    "attention": 29569,
    "engagement": 30958,
    "excitement": 30958,
    "stress": 30958,
    "relaxation": 30958,
    "interest": 30958,
    "focus": 30958,
}
FACTOR_CONTRASTS = {
    "car": (("D", "A"), ("F", "B"), ("G", "C"), ("H", "E")),
    "bandpass": (("B", "A"), ("E", "C"), ("F", "D"), ("H", "G")),
    "notch": (("C", "A"), ("E", "B"), ("G", "D"), ("H", "F")),
}


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


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def load_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(repo_path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Expected experiment_id={EXPERIMENT_ID!r}")
    if tuple(document.get("targets", ())) != PM_NAMES:
        raise ValueError("All seven PM targets must remain in canonical order")
    if tuple(document.get("variants", ())) != VARIANTS:
        raise ValueError("The factorial matrix must contain exactly A--H")
    if tuple(document.get("folds", ())) != FOLDS:
        raise ValueError("Fixed outer folds must be [1,2,3,4,5]")
    if int(document.get("seed", -1)) != 42:
        raise ValueError("The experiment is fixed to seed 42")
    if document.get("task_type") != "classification":
        raise ValueError("This experiment is classification-only")
    if document.get("model", {}).get("name") != "torch_shallow_convnet":
        raise ValueError("The experiment requires torch_shallow_convnet")
    if "label_q5" in json.dumps(document).lower():
        raise ValueError("Legacy label_q5 is forbidden")
    if "regression" in json.dumps(document).lower():
        raise ValueError("Regression is forbidden")
    if document["validation"] != {
        "strategy": "group_record",
        "group_column": "record_group_id",
        "validation_size": 0.15,
        "random_state": 42,
    }:
        raise ValueError("Inner validation must remain record_group_id-disjoint")
    return document


def target_id(pm: str) -> str:
    value = str(pm)
    if value not in PM_NAMES:
        raise ValueError(f"Unknown PM target: {value}")
    return f"pm_{value}_q3_fold_local"


def target_column(pm: str) -> str:
    return f"target_{pm}"


@dataclass(frozen=True)
class FactorialRunSpec:
    outer_fold: int
    pm: str
    variant: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"f{self.outer_fold:02d}_{self.pm}_{self.variant}"


def build_run_matrix(config: Mapping[str, Any]) -> list[FactorialRunSpec]:
    specs = [
        FactorialRunSpec(int(fold), str(pm), str(variant), int(config["seed"]))
        for fold in config["folds"]
        for pm in config["targets"]
        for variant in config["variants"]
    ]
    if len(specs) != 280 or len({spec.run_id for spec in specs}) != 280:
        raise RuntimeError("Expected exactly 280 unique factorial run IDs")
    return specs


def smoke_run_matrix(config: Mapping[str, Any]) -> list[FactorialRunSpec]:
    smoke = config["smoke"]
    specs = [
        spec for spec in build_run_matrix(config)
        if spec.outer_fold == int(smoke["outer_fold"])
        and spec.pm == str(smoke["target"])
        and spec.variant in tuple(smoke["variants"])
    ]
    if len(specs) != 2 or {spec.variant for spec in specs} != {"A", "H"}:
        raise RuntimeError("Smoke matrix must be fold 1 / one PM / variants A and H")
    return specs


def _preprocessing_plan(config: Mapping[str, Any]) -> tuple[PreprocessingAblation, dict[str, TrialPlan]]:
    experiment = PreprocessingAblation(config["preprocessing_matrix_spec"])
    plans = {plan.trial.trial_id: plan for plan in experiment.plan(seed=int(config["seed"]))}
    if tuple(sorted(plans)) != VARIANTS:
        raise RuntimeError(f"Legacy preprocessing matrix did not resolve A--H: {sorted(plans)}")
    return experiment, plans


def _selected_record_ids(config: Mapping[str, Any]) -> set[str]:
    logical = pd.read_parquet(repo_path(config["data"]["logical_recording_map"]))
    if logical["record_group_id"].astype(str).duplicated().any():
        raise RuntimeError("Logical recording map has duplicate record_group_id")
    return set(logical["selected_record_id"].astype(str))


def _variant_index(plan: TrialPlan, selected_records: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(plan.cache.index_path)
    required = {
        "sample_id", "subject_id", "record_id", "record_group_id", "outer_fold",
        "status", "preprocessing_hash", "cache_file",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Variant {plan.trial.trial_id} manifest lacks {missing}")
    selected = frame.loc[
        frame["status"].astype(str).eq("ok")
        & frame["record_id"].astype(str).isin(selected_records)
    ].copy()
    selected["sample_id"] = selected["sample_id"].astype(np.int64)
    selected = selected.sort_values("sample_id").reset_index(drop=True)
    if selected["sample_id"].duplicated().any():
        raise RuntimeError(f"Variant {plan.trial.trial_id} has duplicate sample IDs")
    return selected


def audit_caches(
    config: Mapping[str, Any], plans: Mapping[str, TrialPlan]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    selected_records = _selected_record_ids(config)
    indices: dict[str, pd.DataFrame] = {}
    rows = []
    for variant in VARIANTS:
        plan = plans[variant]
        if not plan.cache.reusable or not plan.cache.complete:
            raise RuntimeError(
                f"Variant {variant} cache is not reusable: {plan.cache.reason}"
            )
        index = _variant_index(plan, selected_records)
        indices[variant] = index
        parameters = plan.trial.parameter_dict()
        rows.append({
            "variant": variant,
            "bandpass": bool(parameters["preprocessing.bandpass.enabled"]),
            "notch": bool(parameters["preprocessing.notch.enabled"]),
            "car": bool(parameters["preprocessing.car.enabled"]),
            "preprocessing_hash": plan.trial.preprocessing_hash,
            "legacy_preprocessing_hash": plan.trial.legacy_preprocessing_hash,
            "cache_index_path": str(plan.cache.index_path),
            "cache_path": str(plan.cache.cache_path),
            "cache_key_hash": plan.trial.cache_key_hash,
            "cache_reusable": bool(plan.cache.reusable),
            "accepted_windows": len(index),
            "subjects": index["subject_id"].astype(str).nunique(),
            "records": index["record_id"].astype(str).nunique(),
            "sample_id_hash": sample_id_filter_hash(index["sample_id"]),
            "outer_fold_hash": stable_hash(sorted(zip(
                index["sample_id"].astype(str), index["outer_fold"].astype(int)
            ))),
            "new_cache_bytes": 0,
        })
    variants = pd.DataFrame(rows)
    baseline = indices["A"]
    baseline_identity = baseline[[
        "sample_id", "subject_id", "record_id", "record_group_id", "outer_fold"
    ]].astype({
        "sample_id": str, "subject_id": str, "record_id": str,
        "record_group_id": str, "outer_fold": int,
    })
    matched = {}
    for variant, index in indices.items():
        identity = index[[
            "sample_id", "subject_id", "record_id", "record_group_id", "outer_fold"
        ]].astype({
            "sample_id": str, "subject_id": str, "record_id": str,
            "record_group_id": str, "outer_fold": int,
        })
        matched[variant] = bool(identity.equals(baseline_identity))
    if len(baseline) != 30958 or not all(matched.values()):
        raise RuntimeError(
            f"A--H do not share the canonical 30,958-window cohort: "
            f"rows={len(baseline)}, matched={matched}"
        )
    audit = {
        "canonical_windows": len(baseline),
        "subjects": baseline["subject_id"].astype(str).nunique(),
        "records": baseline["record_id"].astype(str).nunique(),
        "all_variants_cache_reusable": True,
        "all_variants_exact_sample_identity": True,
        "all_variants_exact_outer_fold_identity": True,
        "matched_eligible_cohort_required": False,
        "matched_cohort_status": "exact_A_to_H_identity",
        "estimated_new_cache_bytes": 0,
        "variant_matches_A": matched,
    }
    return variants, indices, audit


def _load_targets(config: Mapping[str, Any]) -> pd.DataFrame:
    columns = ["subject_id", "record_id", *(target_column(pm) for pm in PM_NAMES)]
    frame = pd.read_parquet(repo_path(config["data"]["target_table"]), columns=columns)
    if "sample_id" not in frame.columns:
        frame = frame.copy()
        frame.insert(0, "sample_id", frame.index.to_numpy(dtype=np.int64))
    frame["sample_id"] = frame["sample_id"].astype(np.int64)
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("Target table has duplicate sample_id")
    return frame


def _target_fold_plan(
    config: Mapping[str, Any], canonical: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, int]]:
    targets = _load_targets(config)
    joined = canonical[[
        "sample_id", "subject_id", "record_id", "record_group_id", "outer_fold"
    ]].merge(
        targets,
        on="sample_id",
        how="left",
        suffixes=("", "_target"),
        sort=False,
        validate="one_to_one",
    )
    for column in ("subject_id", "record_id"):
        if not joined[column].astype(str).eq(joined[f"{column}_target"].astype(str)).all():
            raise RuntimeError(f"Target join changed {column}")
    complete_cases = {
        pm: int(pd.to_numeric(joined[target_column(pm)], errors="coerce").notna().sum())
        for pm in PM_NAMES
    }
    if complete_cases != EXPECTED_COMPLETE_CASES:
        raise RuntimeError(
            f"Complete-case counts changed: {complete_cases} != {EXPECTED_COMPLETE_CASES}"
        )
    rows: list[dict[str, Any]] = []
    transforms: dict[str, dict[str, Any]] = {}
    for fold in FOLDS:
        for pm in PM_NAMES:
            values = pd.to_numeric(joined[target_column(pm)], errors="coerce").to_numpy(np.float32)
            available = np.isfinite(values)
            train = available & joined["outer_fold"].ne(fold).to_numpy()
            test = available & joined["outer_fold"].eq(fold).to_numpy()
            transform = FoldLocalQuantileTargetTransform(3, duplicates="drop").fit(values[train])
            if transform.actual_class_count != 3:
                raise RuntimeError(f"Fold {fold}/{pm} did not retain three Q3 classes")
            target_spec = get_target_spec(target_id(pm))
            manifest = build_target_transform_manifest(
                target_spec,
                transform,
                outer_fold=fold,
                outer_train_sample_ids=joined.loc[train, "sample_id"].to_numpy(),
                outer_train_targets=values[train],
            )
            if manifest["fit_scope"] != "outer_train_only":
                raise RuntimeError("Q3 transform fit scope is not outer_train_only")
            key = f"f{fold:02d}__{pm}"
            transforms[key] = manifest
            y_all = np.full(len(joined), -1, dtype=np.int8)
            y_all[available] = transform.transform(values[available]).astype(np.int8)
            train_ids = joined.loc[train, "sample_id"]
            test_ids = joined.loc[test, "sample_id"]
            train_target_hash = stable_hash(sorted(zip(
                train_ids.astype(str), y_all[train].astype(int)
            )))
            test_target_hash = stable_hash(sorted(zip(
                test_ids.astype(str), y_all[test].astype(int)
            )))
            rows.append({
                "outer_fold": fold,
                "pm": pm,
                "complete_cases": int(available.sum()),
                "train_samples": int(train.sum()),
                "test_samples": int(test.sum()),
                "train_subjects": joined.loc[train, "subject_id"].astype(str).nunique(),
                "test_subjects": joined.loc[test, "subject_id"].astype(str).nunique(),
                "train_sample_id_hash": sample_id_filter_hash(train_ids),
                "test_sample_id_hash": sample_id_filter_hash(test_ids),
                "train_target_hash": train_target_hash,
                "test_target_hash": test_target_hash,
                "q3_transform_hash": manifest["transform_hash"],
                "q3_fit_scope": manifest["fit_scope"],
                "all_variants_target_mask_identical": True,
                **{
                    f"train_class_{label}": int((y_all[train] == label).sum())
                    for label in range(3)
                },
                **{
                    f"test_class_{label}": int((y_all[test] == label).sum())
                    for label in range(3)
                },
            })
    return pd.DataFrame(rows), transforms, complete_cases


def _fold_manifest(canonical: pd.DataFrame) -> dict[str, Any]:
    folds = []
    all_test_subjects: set[str] = set()
    for fold in FOLDS:
        train = canonical["outer_fold"].ne(fold)
        test = canonical["outer_fold"].eq(fold)
        train_subjects = sorted(canonical.loc[train, "subject_id"].astype(str).unique())
        test_subjects = sorted(canonical.loc[test, "subject_id"].astype(str).unique())
        overlap = sorted(set(train_subjects) & set(test_subjects))
        if overlap:
            raise RuntimeError(f"Outer subject leakage in fold {fold}: {overlap}")
        if all_test_subjects & set(test_subjects):
            raise RuntimeError("A subject appears in multiple outer test folds")
        all_test_subjects.update(test_subjects)
        folds.append({
            "outer_fold": fold,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "train_samples": int(train.sum()),
            "test_samples": int(test.sum()),
            "train_sample_id_hash": sample_id_filter_hash(canonical.loc[train, "sample_id"]),
            "test_sample_id_hash": sample_id_filter_hash(canonical.loc[test, "sample_id"]),
            "subject_overlap": 0,
        })
    if all_test_subjects != set(canonical["subject_id"].astype(str)):
        raise RuntimeError("Fixed outer folds do not cover every subject exactly once")
    return {
        "protocol": "fixed 5-fold GroupKFold by subject_id",
        "precomputed_fold_column": "outer_fold",
        "folds": folds,
    }


def _target_row(plan: Mapping[str, Any], spec: FactorialRunSpec) -> Mapping[str, Any]:
    for row in plan["target_fold_audit"]:
        if int(row["outer_fold"]) == spec.outer_fold and row["pm"] == spec.pm:
            return row
    raise KeyError(f"Missing target/fold audit for {spec.run_id}")


def run_specification_hash(spec: FactorialRunSpec, protocol_hash: str) -> str:
    return stable_hash({"run_spec": asdict(spec), "protocol_hash": protocol_hash})


def build_protocol(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, plans = _preprocessing_plan(config)
    variants, indices, cache_audit = audit_caches(config, plans)
    canonical = indices["A"]
    target_audit, transforms, complete_cases = _target_fold_plan(
        config, canonical
    )
    folds = _fold_manifest(canonical)
    protocol_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "result_status": config["result_status"],
        "task_type": "classification",
        "targets": list(PM_NAMES),
        "target_ids": [target_id(pm) for pm in PM_NAMES],
        "q3_policy": "fold_local_quantile_q3_outer_train_only",
        "variants": list(VARIANTS),
        "folds": list(FOLDS),
        "seed": int(config["seed"]),
        "model": deepcopy(config["model"]),
        "validation": deepcopy(config["validation"]),
        "evaluation": deepcopy(config["evaluation"]),
        "canonical_windows": 30958,
        "complete_case_counts": complete_cases,
        "canonical_sample_id_hash": variants.loc[
            variants["variant"].eq("A"), "sample_id_hash"
        ].iloc[0],
        "canonical_outer_fold_hash": variants.loc[
            variants["variant"].eq("A"), "outer_fold_hash"
        ].iloc[0],
        "cache_key_hashes": dict(zip(variants["variant"], variants["cache_key_hash"])),
        "q3_transform_hashes": {
            key: value["transform_hash"] for key, value in transforms.items()
        },
        "matched_cohort_status": cache_audit["matched_cohort_status"],
        "planned_runs": 280,
        "unsupported_runs": 0,
        "primary_metrics": [
            "participant_macro_macro_f1",
            "participant_macro_balanced_accuracy_fixed_labels",
        ],
        "participant_balanced_accuracy_definition": (
            "mean recall over fixed labels [0,1,2], zero_division=0"
        ),
        "factor_contrasts": {
            factor: [f"{left}-{right}" for left, right in pairs]
            for factor, pairs in FACTOR_CONTRASTS.items()
        },
    }
    protocol_hash = stable_hash(protocol_payload)
    specs = build_run_matrix(config)
    run_rows = []
    for spec in specs:
        target = _target_row({"target_fold_audit": target_audit.to_dict("records")}, spec)
        run_rows.append({
            **asdict(spec),
            "run_id": spec.run_id,
            "target_id": target_id(spec.pm),
            "preprocessing_hash": plans[spec.variant].trial.preprocessing_hash,
            "legacy_preprocessing_hash": plans[spec.variant].trial.legacy_preprocessing_hash,
            "cache_index_path": str(plans[spec.variant].cache.index_path),
            "train_samples": int(target["train_samples"]),
            "test_samples": int(target["test_samples"]),
            "test_sample_id_hash": target["test_sample_id_hash"],
            "test_target_hash": target["test_target_hash"],
            "q3_transform_hash": target["q3_transform_hash"],
            "protocol_hash": protocol_hash,
            "specification_hash": run_specification_hash(spec, protocol_hash),
            "supported": True,
        })
    run_matrix = pd.DataFrame(run_rows)
    plan_hash = stable_hash(run_rows)
    leakage = {
        "status": "clean",
        "q3_threshold_leakage": "none",
        "q3_fit_scope_outer_train_only": bool(
            target_audit["q3_fit_scope"].eq("outer_train_only").all()
        ),
        "outer_subject_leakage": "none",
        "inner_validation_group": "record_group_id",
        "all_variants_identical_test_participants": True,
        "all_variants_identical_test_sample_ids": True,
        "all_variants_identical_test_labels": True,
        "all_seven_pm_present": tuple(complete_cases) == PM_NAMES,
        "focus_only_shortcut": False,
        "regression_runs": 0,
        "matched_cohort_status": cache_audit["matched_cohort_status"],
    }
    if not all(
        value is True
        for key, value in leakage.items()
        if key.endswith("_only") or key.startswith("all_") or key == "q3_fit_scope_outer_train_only"
    ):
        raise RuntimeError(f"Leakage gates failed: {leakage}")
    protocol = {
        **protocol_payload,
        "protocol_hash": protocol_hash,
        "plan_hash": plan_hash,
        "run_matrix_hash": stable_hash(run_rows),
        "cache_audit": cache_audit,
        "leakage_audit": leakage,
        "execution_ready": True,
        "training_status": "training_not_started",
    }
    output = repo_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "protocol_manifest.json", protocol)
    _atomic_json(output / "q3_target_transforms.json", transforms)
    _atomic_json(output / "fold_manifest.json", folds)
    _atomic_json(output / "leakage_audit.json", leakage)
    _write_csv(output / "run_matrix.csv", run_matrix)
    _write_csv(output / "preprocessing_variants.csv", variants)
    _write_csv(output / "target_fold_audit.csv", target_audit)
    _write_csv(output / "dataset_summary.csv", pd.DataFrame([{
        "canonical_signal_windows": len(canonical),
        "subjects": canonical["subject_id"].astype(str).nunique(),
        "records": canonical["record_id"].astype(str).nunique(),
        **{f"{pm}_complete_cases": count for pm, count in complete_cases.items()},
        "matched_A_H": True,
        "new_cache_bytes": 0,
    }]))
    _write_csv(output / "compatibility_matrix.csv", pd.DataFrame([{
        "model": "torch_shallow_convnet",
        "input_shape": "[B,1,14,2560]",
        "task_type": "classification",
        "targets": 7,
        "variants": 8,
        "folds": 5,
        "planned_runs": 280,
        "supported": True,
        "reason": "all A-H caches are semantically reusable and share exact sample/fold identity",
    }]))
    protocol["q3_transforms"] = transforms
    protocol["target_fold_audit"] = target_audit.to_dict("records")
    protocol["run_matrix"] = run_rows
    protocol["variant_plans"] = plans
    return protocol


def plan_experiment(config_path: str | Path) -> dict[str, Any]:
    plan = build_protocol(config_path)
    return {
        "experiment_id": plan["experiment_id"],
        "protocol_hash": plan["protocol_hash"],
        "plan_hash": plan["plan_hash"],
        "planned_runs": plan["planned_runs"],
        "unsupported_runs": plan["unsupported_runs"],
        "canonical_windows": plan["canonical_windows"],
        "complete_case_counts": plan["complete_case_counts"],
        "matched_cohort_status": plan["matched_cohort_status"],
        "estimated_new_cache_bytes": plan["cache_audit"]["estimated_new_cache_bytes"],
        "leakage_status": plan["leakage_audit"]["status"],
        "execution_ready": True,
        "training_status": "training_not_started",
        "models_trained": 0,
    }


def _benchmark_config(
    config: Mapping[str, Any], spec: FactorialRunSpec, plan: Mapping[str, Any], *, smoke: bool
) -> dict[str, Any]:
    trial_plan: TrialPlan = plan["variant_plans"][spec.variant]
    trial: ExperimentTrial = trial_plan.trial
    params = deepcopy(config["model"]["params"])
    if smoke:
        params["max_epochs"] = int(config["smoke"]["max_epochs"])
        params["early_stopping_patience"] = 1
    profile = "smoke" if smoke else "full"
    return {
        "output_dir": str(
            repo_path(config["runner_output_dir"]) / profile / spec.run_id
        ),
        "result_status": "smoke" if smoke else config["result_status"],
        "raw_preprocessing": trial.preprocessing.to_legacy_raw_preprocessing(),
        "datasets": {
            "emotiv_raw_eeg": {
                "data_path": str(trial_plan.cache.index_path),
                "cache_path_root": str(ROOT),
                "target_data_path": str(repo_path(config["data"]["target_table"])),
                "target_id": target_id(spec.pm),
                "dataset_mode": "raw_deduplicated_logical_records",
                "logical_recording_map_path": str(
                    repo_path(config["data"]["logical_recording_map"])
                ),
                "raw_preprocessing": trial.preprocessing.to_legacy_raw_preprocessing(),
            }
        },
        "tasks": [target_id(spec.pm)],
        "task_config": {
            "target_id": target_id(spec.pm),
            "random_state": spec.seed,
            "target_transform_manifests": {
                str(fold): deepcopy(plan["q3_transforms"][f"f{fold:02d}__{spec.pm}"])
                for fold in FOLDS
            },
        },
        "models": {
            RUNNER_MODEL_ALIAS: {
                "type": "torch_shallow_convnet",
                "task_type": "classification",
                "params": params,
            }
        },
        "evaluation": {
            "protocol": "group_kfold_subject",
            "n_splits": 5,
            "group_column": "subject_id",
            "precomputed_fold_column": "outer_fold",
            "folds": [spec.outer_fold],
            "random_state": spec.seed,
        },
        "validation": deepcopy(config["validation"]),
        "run_within_subject": False,
        "run_loso": False,
    }


def _split_identity(split: Any) -> dict[str, Any]:
    train_ids = np.asarray(split.sample_id_train).astype(str)
    test_ids = np.asarray(split.sample_id_test).astype(str)
    train_targets = np.asarray(split.y_train, dtype=int)
    test_targets = np.asarray(split.y_test, dtype=int)
    return {
        "train_sample_id_hash": sample_id_filter_hash(train_ids),
        "test_sample_id_hash": sample_id_filter_hash(test_ids),
        "train_target_hash": stable_hash(sorted(zip(train_ids, train_targets))),
        "test_target_hash": stable_hash(sorted(zip(test_ids, test_targets))),
        "train_subject_hash": stable_hash(
            sorted(set(np.asarray(split.subject_train).astype(str)))
        ),
        "test_subject_hash": stable_hash(
            sorted(set(np.asarray(split.subject_test).astype(str)))
        ),
    }


def resumable_summary(path: Path, specification_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    if payload.get("specification_hash") != specification_hash:
        return None
    artifacts = [Path(value) for value in payload.get("required_artifacts", [])]
    return payload if artifacts and all(item.is_file() for item in artifacts) else None


def execute_run(
    config_path: str | Path,
    spec: FactorialRunSpec,
    *,
    smoke: bool,
    resume: bool,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    config = load_config(config_path)
    profile = "smoke" if smoke else "full"
    run_dir = repo_path(config["output_dir"]) / profile / "runs" / spec.run_id
    summary_path = run_dir / "run_summary.json"
    specification_hash = run_specification_hash(spec, str(plan["protocol_hash"]))
    if resume:
        previous = resumable_summary(summary_path, specification_hash)
        if previous is not None:
            return previous
    runner_config = _benchmark_config(config, spec, plan, smoke=smoke)
    runner = BenchmarkRunner(runner_config)
    started = time.perf_counter()
    runner.run()
    elapsed = time.perf_counter() - started
    split = runner.last_evaluated_split
    if split is None:
        raise RuntimeError("BenchmarkRunner did not retain the evaluated split")
    expected = _target_row(plan, spec)
    actual = _split_identity(split)
    for key in (
        "train_sample_id_hash", "test_sample_id_hash", "train_target_hash", "test_target_hash"
    ):
        if actual[key] != str(expected[key]):
            raise RuntimeError(f"Runtime identity mismatch for {spec.run_id}/{key}")
    if split.metadata.get("target_transform_hash") != expected["q3_transform_hash"]:
        raise RuntimeError("Runtime did not use the frozen outer-train Q3 transform")
    fold_name = f"fold_{spec.outer_fold:02d}"
    result = runner.results["emotiv_raw_eeg"]["models"][target_id(spec.pm)][
        RUNNER_MODEL_ALIAS
    ]["group_kfold_subject"]["folds"][fold_name]
    artifacts = {
        key: value for key, value in result.get("artifacts", {}).items()
        if isinstance(value, str) and Path(value).is_file()
    }
    if "predictions" not in artifacts:
        raise RuntimeError("BenchmarkRunner did not save predictions")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete",
        "result_status": "smoke" if smoke else config["result_status"],
        "specification_hash": specification_hash,
        "protocol_hash": plan["protocol_hash"],
        "plan_hash": plan["plan_hash"],
        **asdict(spec),
        "run_id": spec.run_id,
        "q3_transform_hash": expected["q3_transform_hash"],
        "preprocessing_hash": plan["variant_plans"][spec.variant].trial.preprocessing_hash,
        "split_identity": actual,
        "training_time_seconds": elapsed,
        "metrics": result.get("metrics", {}),
        "artifacts": artifacts,
        "required_artifacts": list(artifacts.values()),
    }
    _atomic_json(summary_path, summary)
    return summary


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = [0, 1, 2]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )),
    }


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (len(values) - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _safe_wilcoxon(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite) or np.allclose(finite, 0.0):
        return 1.0
    return float(wilcoxon(finite, zero_method="wilcox", alternative="two-sided").pvalue)


def _factor_effects(participants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = participants.pivot(
        index=["outer_fold", "subject_id", "pm"], columns="variant", values=list(METRICS)
    )
    rows = []
    per_pm = []
    for factor, contrasts in FACTOR_CONTRASTS.items():
        for metric in ("macro_f1", "balanced_accuracy"):
            deltas = pd.concat(
                [wide[(metric, left)] - wide[(metric, right)] for left, right in contrasts],
                axis=1,
            ).mean(axis=1)
            named = deltas.rename("delta").reset_index()
            participant = named.groupby("subject_id", as_index=False)["delta"].mean()
            rows.append({
                "factor": factor,
                "metric": metric,
                "participants": len(participant),
                "mean_delta": float(participant["delta"].mean()),
                "median_delta": float(participant["delta"].median()),
                "positive_fraction": float((participant["delta"] > 0).mean()),
                "wilcoxon_p": _safe_wilcoxon(participant["delta"].to_numpy()),
            })
            for pm, group in named.groupby("pm", sort=True):
                per_pm.append({
                    "factor": factor,
                    "metric": metric,
                    "pm": pm,
                    "participant_pm_rows": len(group),
                    "mean_delta": float(group["delta"].mean()),
                    "median_delta": float(group["delta"].median()),
                })
    effects = pd.DataFrame(rows)
    effects["holm_family"] = "three_factors_x_two_primary_metrics"
    effects["holm_p"] = _holm_adjust(effects["wilcoxon_p"].tolist())
    return effects, pd.DataFrame(per_pm)


def aggregate_execution(
    config: Mapping[str, Any], profile: str, summaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = 2 if profile == "smoke" else 280
    if len(summaries) != expected:
        raise RuntimeError(f"Expected {expected} completed {profile} runs")
    prediction_frames = []
    run_rows = []
    for summary in summaries:
        frame = pd.read_parquet(summary["artifacts"]["predictions"])
        frame["outer_fold"] = int(summary["outer_fold"])
        frame["pm"] = str(summary["pm"])
        frame["variant"] = str(summary["variant"])
        prediction_frames.append(frame)
        run_rows.append({
            key: summary.get(key)
            for key in (
                "run_id", "outer_fold", "pm", "variant", "status", "result_status",
                "protocol_hash", "specification_hash", "training_time_seconds",
            )
        })
    predictions = pd.concat(prediction_frames, ignore_index=True)
    participant_rows = []
    for keys, group in predictions.groupby(
        ["outer_fold", "subject_id", "pm", "variant"], sort=True
    ):
        fold, subject, pm, variant = keys
        participant_rows.append({
            "outer_fold": int(fold),
            "subject_id": str(subject),
            "pm": str(pm),
            "variant": str(variant),
            "n_samples": len(group),
            **_classification_metrics(
                group["y_true"].to_numpy(int), group["y_pred"].to_numpy(int)
            ),
        })
    participants = pd.DataFrame(participant_rows)
    output = repo_path(config["output_dir"]) / profile
    _write_csv(output / "run_results.csv", pd.DataFrame(run_rows))
    _write_csv(output / "participant_metrics.csv", participants)
    if profile == "smoke":
        result = {
            "status": "smoke_complete",
            "completed_runs": 2,
            "variants": sorted(participants["variant"].unique()),
            "target": str(participants["pm"].iloc[0]),
            "outer_fold": int(participants["outer_fold"].iloc[0]),
            "training_not_full_matrix": True,
        }
        _atomic_json(output / "aggregate_summary.json", result)
        return result
    overall = participants.groupby("variant", as_index=False)[list(METRICS)].mean()
    per_pm = participants.groupby(["variant", "pm"], as_index=False)[list(METRICS)].mean()
    baseline = overall.loc[overall["variant"].eq("A")].iloc[0]
    overall = overall.rename(columns={
        "macro_f1": "participant_macro_macro_f1",
        "balanced_accuracy": "participant_macro_balanced_accuracy",
        "accuracy": "participant_macro_accuracy",
        "weighted_f1": "participant_macro_weighted_f1",
    })
    overall["delta_macro_f1_vs_A"] = (
        overall["participant_macro_macro_f1"] - float(baseline["macro_f1"])
    )
    overall["delta_balanced_accuracy_vs_A"] = (
        overall["participant_macro_balanced_accuracy"]
        - float(baseline["balanced_accuracy"])
    )
    flags = {
        row["variant"]: row for row in config["preprocessing_variants"]
    }
    for position, name in enumerate(("bandpass", "notch", "car"), 1):
        overall.insert(position, name, overall["variant"].map(
            lambda variant: bool(flags[str(variant)][name])
        ))
    paired = []
    reference = participants.loc[participants["variant"].eq("A")]
    for variant in VARIANTS[1:]:
        candidate = participants.loc[participants["variant"].eq(variant)]
        merged = reference.merge(
            candidate,
            on=["outer_fold", "subject_id", "pm"],
            suffixes=("_A", "_candidate"),
            validate="one_to_one",
        )
        if len(merged) != len(reference) or len(merged) != len(candidate):
            raise RuntimeError(f"Participant×PM pairing differs for variant {variant}")
        merged["variant"] = variant
        for metric in METRICS:
            merged[f"delta_{metric}"] = (
                merged[f"{metric}_candidate"] - merged[f"{metric}_A"]
            )
        paired.append(merged)
    paired_frame = pd.concat(paired, ignore_index=True)
    effects, per_pm_effects = _factor_effects(participants)
    pooled_rows = []
    confusion_rows = []
    for keys, group in predictions.groupby(["variant", "pm"], sort=True):
        variant, pm = keys
        pooled_rows.append({
            "variant": variant, "pm": pm, "n_samples": len(group),
            **_classification_metrics(group["y_true"].to_numpy(int), group["y_pred"].to_numpy(int)),
        })
        matrix = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1, 2])
        confusion_rows.append({"variant": variant, "pm": pm, "confusion_matrix": json.dumps(matrix.tolist())})
    _write_csv(output / "aggregate_by_variant.csv", overall)
    _write_csv(output / "aggregate_by_variant_pm.csv", per_pm)
    _write_csv(output / "paired_preprocessing_effects.csv", paired_frame)
    _write_csv(output / "factor_effects.csv", effects)
    _write_csv(output / "factor_effects_by_pm.csv", per_pm_effects)
    _write_csv(output / "pooled_window_metrics.csv", pd.DataFrame(pooled_rows))
    _write_csv(output / "confusion_matrices.csv", pd.DataFrame(confusion_rows))
    result = {
        "status": "full_complete",
        "completed_runs": 280,
        "primary_table": overall.to_dict("records"),
        "factor_effects": effects.to_dict("records"),
    }
    _atomic_json(output / "aggregate_summary.json", result)
    return result


def run_experiment(
    config_path: str | Path, *, smoke: bool, resume: bool
) -> dict[str, Any]:
    config = load_config(config_path)
    plan = build_protocol(config_path)
    specs = smoke_run_matrix(config) if smoke else build_run_matrix(config)
    summaries = [
        execute_run(config_path, spec, smoke=smoke, resume=resume, plan=plan)
        for spec in specs
    ]
    return aggregate_execution(config, "smoke" if smoke else "full", summaries)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/preprocessing/preprocessing_factorial_q3_all_pm_v1.json",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    result = (
        plan_experiment(args.config)
        if args.plan_only
        else run_experiment(args.config, smoke=args.smoke, resume=args.resume)
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_COMPLETE_CASES", "EXPERIMENT_ID", "FACTOR_CONTRASTS", "FOLDS",
    "FactorialRunSpec", "PM_NAMES", "SCHEMA_VERSION", "VARIANTS",
    "aggregate_execution", "audit_caches", "build_protocol", "build_run_matrix",
    "execute_run", "load_config", "main", "plan_experiment", "resumable_summary",
    "run_experiment", "run_specification_hash", "smoke_run_matrix", "stable_hash",
    "target_id",
]
