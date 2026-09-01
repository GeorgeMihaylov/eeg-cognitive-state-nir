"""Three-seed orchestration and subject-level stability analysis for calibration."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from bench.bench_runner import benchmark_config_hash
from bench.experiments.user_calibration import (
    REPO_ROOT,
    UserCalibrationExperiment,
    _bootstrap_mean_interval,
    _implementation_hash as calibration_implementation_hash,
)


MULTISEED_SCHEMA_VERSION = "user-calibration-multiseed-v1"
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
METHODS = ("zero_shot", "head_only", "full_model")
SPLIT_HASH_COLUMNS = (
    "outer_train_subject_hash",
    "inner_train_subject_hash",
    "inner_validation_subject_hash",
    "calibration_sample_hash",
    "adaptation_train_sample_hash",
    "adaptation_validation_sample_hash",
    "evaluation_sample_hash",
    "preprocessor_hash",
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _canonical_hash(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _ordered_hash(values: Sequence[Any]) -> str:
    return _canonical_hash([str(value) for value in values])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        REPO_ROOT / "bench" / "experiments" / "user_calibration.py",
        REPO_ROOT / "model_zoo" / "DL" / "adapter.py",
        REPO_ROOT / "model_zoo" / "DL" / "mlp.py",
    ):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_multiseed_calibration_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _repo_path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Multiseed calibration config not found: {spec_path}")
    document = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    required = {"experiment", "base_template", "calibration"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Multiseed calibration config is missing: {missing}")
    experiment = document["experiment"]
    if experiment.get("type") != "user_calibration_multiseed":
        raise ValueError(
            "experiment.type must be 'user_calibration_multiseed'"
        )
    split_seed = int(experiment.get("split_seed", 42))
    model_seeds = tuple(int(value) for value in experiment.get("model_seeds", []))
    if not model_seeds or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("experiment.model_seeds must contain unique seeds")
    if not bool(experiment.get("require_cuda", True)):
        raise ValueError("Multiseed personalization requires CUDA")
    fractions = [float(value) for value in document["calibration"]["budgets_fraction"]]
    if fractions != [0.2]:
        raise ValueError("Multiseed personalization must use only budget 0.20")
    methods = tuple(str(value) for value in document["calibration"]["methods"])
    if methods != METHODS:
        raise ValueError(f"calibration.methods must be {list(METHODS)}")
    defaults = document["calibration"].get("defaults", {})
    if defaults.get("fraction_allocation") != "global_prefix":
        raise ValueError("fraction_allocation must be 'global_prefix'")
    document["experiment"]["split_seed"] = split_seed
    document["experiment"]["model_seeds"] = list(model_seeds)
    return document


def resolve_seed_base_config(
    template: Mapping[str, Any],
    *,
    model_seed: int,
    split_seed: int,
    output_dir: str | Path,
    maximum_epochs: Optional[int] = None,
) -> dict[str, Any]:
    """Resolve one base run while keeping every split seed fixed."""
    config = deepcopy(dict(template))
    config["output_dir"] = str(output_dir)
    config.setdefault("validation", {})["random_state"] = int(split_seed)
    config.setdefault("evaluation", {})["random_state"] = int(split_seed)
    config.setdefault("task_config", {})["random_state"] = int(split_seed)
    for model_config in config["models"].values():
        params = model_config.setdefault("params", {})
        params["random_state"] = int(model_seed)
        if maximum_epochs is not None:
            params["max_epochs"] = int(maximum_epochs)
            params["early_stopping_patience"] = min(
                int(params.get("early_stopping_patience", maximum_epochs)),
                int(maximum_epochs),
            )
    return config


def resolve_seed_calibration_config(
    document: Mapping[str, Any],
    *,
    model_seed: int,
    split_seed: int,
    base_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a legacy-compatible single-seed document for the canonical runner."""
    calibration = deepcopy(dict(document["calibration"]))
    calibration.setdefault("defaults", {})["random_state"] = int(model_seed)
    return {
        "experiment": {
            "type": "user_calibration",
            "name": (
                f"{document['experiment']['name']}_seed_{int(model_seed)}"
            ),
            "output_dir": str(output_dir),
            "seed": int(model_seed),
            "split_seed": int(split_seed),
            "model_seed": int(model_seed),
            "resume": True,
            "require_cuda": True,
            "bootstrap_samples": int(
                document["experiment"].get("bootstrap_samples", 1000)
            ),
        },
        "base_run": {
            "config_path": str(base_config_path),
            "train_if_missing": True,
            "dataset": str(document["base_template"].get(
                "dataset", "emotiv_cognitive"
            )),
            "task": str(document["base_template"].get(
                "task", "cognitive_load_5class"
            )),
            "model": str(document["base_template"].get("model", "torch_mlp")),
        },
        "calibration": calibration,
    }


def _metric_summary(
    frame: pd.DataFrame,
    *,
    grouping: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(grouping), sort=True, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(grouping, keys))
        for metric in METRICS:
            values = pd.to_numeric(
                group[f"{metric}_after"], errors="coerce"
            ).dropna()
            gains = pd.to_numeric(
                group[f"{metric}_gain"], errors="coerce"
            ).dropna()
            low, high = _bootstrap_mean_interval(
                gains,
                samples=bootstrap_samples,
                random_state=bootstrap_seed,
            )
            rows.append({
                **identity,
                "metric": metric,
                "n_subjects": int(group["subject_id"].nunique()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "mean_gain": float(gains.mean()),
                "median_gain": float(gains.median()),
                "positive_fraction": float((gains > 0).mean()),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_resamples": int(bootstrap_samples),
            })
    return pd.DataFrame(rows)


def build_multiseed_aggregates(
    subject_metrics: pd.DataFrame,
    *,
    model_seeds: Sequence[int],
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate per seed, then across complete subjects without seed pseudoreplication."""
    expected = {int(value) for value in model_seeds}
    completed = subject_metrics.loc[
        subject_metrics["status"].astype(str) == "completed"
    ].copy()
    per_seed = _metric_summary(
        completed,
        grouping=("model_seed", "method"),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    seed_counts = completed.groupby(
        ["subject_id", "method"], sort=True
    )["model_seed"].agg(lambda values: len(set(map(int, values))))
    complete_keys = set(seed_counts.loc[seed_counts == len(expected)].index)
    complete = completed.loc[
        [
            (str(row.subject_id), str(row.method)) in complete_keys
            and int(row.model_seed) in expected
            for row in completed.itertuples()
        ]
    ].copy()
    mean_columns = [
        f"{metric}_{suffix}"
        for metric in METRICS
        for suffix in ("after", "gain")
    ]
    subject_seed_means = (
        complete.groupby(
            ["subject_id", "source", "source_group", "outer_fold", "method"],
            sort=True,
            as_index=False,
        )[mean_columns]
        .mean()
    )
    multiseed = _metric_summary(
        subject_seed_means,
        grouping=("method",),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    ).rename(columns={
        "mean": "mean_over_subject_seed_means",
        "median": "median_over_subject_seed_means",
        "std": "std_over_subject_seed_means",
        "positive_fraction": "positive_subject_fraction",
    })
    source_summary = _metric_summary(
        subject_seed_means,
        grouping=("source_group", "method"),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return per_seed, multiseed, source_summary, subject_seed_means


def build_stability_summary(
    subject_metrics: pd.DataFrame,
    *,
    model_seeds: Sequence[int],
) -> pd.DataFrame:
    expected = [int(value) for value in model_seeds]
    completed = subject_metrics.loc[
        subject_metrics["status"].astype(str) == "completed"
    ].copy()
    subject_rows: list[dict[str, Any]] = []
    for (subject, source, source_group, method), group in completed.groupby(
        ["subject_id", "source", "source_group", "method"], sort=True
    ):
        if set(group["model_seed"].astype(int)) != set(expected):
            continue
        for metric in METRICS:
            gains = {
                int(row.model_seed): float(getattr(row, f"{metric}_gain"))
                for row in group.itertuples()
            }
            values = np.asarray([gains[seed] for seed in expected], dtype=float)
            positive = int(np.sum(values > 0))
            subject_rows.append({
                "record_type": "subject",
                "subject_id": str(subject),
                "source": str(source),
                "source_group": str(source_group),
                "method": str(method),
                "metric": metric,
                **{
                    f"gain_seed_{seed}": gains[seed]
                    for seed in expected
                },
                "mean_gain": float(values.mean()),
                "std_gain": float(values.std(ddof=1)),
                "minimum_gain": float(values.min()),
                "maximum_gain": float(values.max()),
                "positive_seeds_count": positive,
                "improved_in_at_least_2_of_3": positive >= 2,
                "improved_in_all_3": positive == len(expected),
            })
    subjects = pd.DataFrame(subject_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for (method, metric), group in subjects.groupby(
        ["method", "metric"], sort=True
    ):
        aggregate_rows.append({
            "record_type": "aggregate",
            "subject_id": None,
            "source": "overall",
            "source_group": "overall",
            "method": method,
            "metric": metric,
            "n_subjects": int(len(group)),
            "subjects_improved_at_least_2_of_3": int(
                group["improved_in_at_least_2_of_3"].sum()
            ),
            "fraction_improved_at_least_2_of_3": float(
                group["improved_in_at_least_2_of_3"].mean()
            ),
            "subjects_improved_all_3": int(group["improved_in_all_3"].sum()),
            "fraction_improved_all_3": float(
                group["improved_in_all_3"].mean()
            ),
            "mean_subject_gain": float(group["mean_gain"].mean()),
            "median_subject_gain": float(group["mean_gain"].median()),
            "mean_seed_std": float(group["std_gain"].mean()),
        })
    return pd.concat(
        [subjects, pd.DataFrame(aggregate_rows)],
        ignore_index=True,
        sort=False,
    )


def _paired_comparisons(
    frame: pd.DataFrame,
    *,
    grouping: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    comparisons = (
        ("head_only", "zero_shot"),
        ("full_model", "zero_shot"),
        ("full_model", "head_only"),
    )
    rows: list[dict[str, Any]] = []
    groupers = list(grouping)
    grouped = [((), frame)] if not groupers else frame.groupby(
        groupers, sort=True, dropna=False
    )
    for key, scoped in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(groupers, keys))
        for left, right in comparisons:
            for metric in METRICS:
                pivot = scoped.pivot_table(
                    index="subject_id",
                    columns="method",
                    values=f"{metric}_after",
                    aggfunc="first",
                )
                if left not in pivot or right not in pivot:
                    continue
                differences = (pivot[left] - pivot[right]).dropna()
                low, high = _bootstrap_mean_interval(
                    differences,
                    samples=bootstrap_samples,
                    random_state=bootstrap_seed,
                )
                rows.append({
                    **identity,
                    "left_method": left,
                    "right_method": right,
                    "metric": metric,
                    "n_subjects": int(len(differences)),
                    "mean_difference": float(differences.mean()),
                    "median_difference": float(differences.median()),
                    "positive_fraction": float((differences > 0).mean()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_resamples": int(bootstrap_samples),
                })
    return pd.DataFrame(rows)


def build_threshold_summary(
    subject_metrics: pd.DataFrame,
    *,
    model_seeds: Sequence[int],
) -> pd.DataFrame:
    expected = {int(value) for value in model_seeds}
    completed = subject_metrics.loc[
        subject_metrics["status"].astype(str) == "completed"
    ]
    rows: list[dict[str, Any]] = []
    for method, group in completed.groupby("method", sort=True):
        counts: list[int] = []
        means: list[float] = []
        for _, subject in group.groupby("subject_id", sort=True):
            if set(subject["model_seed"].astype(int)) != expected:
                continue
            accuracy = subject["accuracy_after"].to_numpy(dtype=float)
            counts.append(int(np.sum(accuracy >= 0.75)))
            means.append(float(accuracy.mean()))
        array = np.asarray(counts, dtype=int)
        mean_array = np.asarray(means, dtype=float)
        rows.append({
            "method": method,
            "n_complete_subjects": int(len(array)),
            "subjects_accuracy_ge_075_any_seed": int(np.sum(array >= 1)),
            "subjects_accuracy_ge_075_at_least_2_seeds": int(
                np.sum(array >= 2)
            ),
            "subjects_accuracy_ge_075_all_3_seeds": int(
                np.sum(array == len(expected))
            ),
            "subjects_mean_accuracy_ge_075": int(
                np.sum(mean_array >= 0.75)
            ),
        })
    return pd.DataFrame(rows)


class UserCalibrationMultiseedExperiment:
    """Compose canonical single-seed runs and aggregate complete subjects."""

    def __init__(self, config_path: str | Path):
        self.config_path = _repo_path(config_path)
        self.document = load_multiseed_calibration_spec(self.config_path)
        template_path = _repo_path(self.document["base_template"]["config_path"])
        self.base_template_path = template_path
        self.base_template = yaml.safe_load(
            template_path.read_text(encoding="utf-8")
        )
        self.experiment = self.document["experiment"]
        self.split_seed = int(self.experiment["split_seed"])
        self.model_seeds = tuple(
            int(value) for value in self.experiment["model_seeds"]
        )

    def _seed42_compatibility(self) -> dict[str, Any]:
        configured = self.experiment.get("seed_42_source_run")
        checks: dict[str, Any] = {}
        if configured is None:
            return {
                "eligible": False,
                "checks": {"source_run_configured": False},
                "reason": "No seed-42 source run was configured",
            }
        source = _repo_path(configured)
        manifest_path = source / "run_manifest.json"
        if not manifest_path.is_file():
            return {
                "eligible": False,
                "checks": {"source_manifest_exists": False},
                "reason": "Seed-42 source manifest is missing",
            }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base_run = _repo_path(manifest["base_run_directory"])
        source_base = yaml.safe_load(
            (base_run / "config.yaml").read_text(encoding="utf-8")
        )
        metrics = pd.read_csv(source / "calibration_subject_metrics.csv")
        selected = metrics.loc[
            np.isclose(metrics["budget"].astype(float), 0.2)
            & metrics["method"].isin(METHODS)
        ]
        data_path = _repo_path(
            self.base_template["datasets"][
                self.document["base_template"].get(
                    "dataset", "emotiv_cognitive"
                )
            ]["data_path"]
        )
        checks.update({
            "source_status_completed": manifest.get("status") == "completed",
            "calibration_implementation_match": (
                manifest.get("implementation_hash")
                == calibration_implementation_hash()
            ),
            "base_config_exact_match": source_base == self.base_template,
            "base_config_hash_match": (
                manifest.get("base_config_hash")
                == benchmark_config_hash(self.base_template)
            ),
            "seed_42_rows_present": len(selected) == 54 * len(METHODS),
            "model_seed_is_42": set(selected["seed"].astype(int)) == {42},
            "dataset_file_exists": data_path.is_file(),
            "feature_count_is_448": (
                int(self.base_template["datasets"][
                    self.document["base_template"].get(
                        "dataset", "emotiv_cognitive"
                    )
                ]["expected_feature_count"]) == 448
            ),
            "target_contract_matches": (
                self.base_template["datasets"][
                    self.document["base_template"].get(
                        "dataset", "emotiv_cognitive"
                    )
                ]["target_col"] == "label_q5"
                and int(self.base_template["datasets"][
                    self.document["base_template"].get(
                        "dataset", "emotiv_cognitive"
                    )
                ]["n_classes"]) == 5
            ),
            "historical_dataset_fingerprint_available": False,
        })
        current_fingerprint = _file_sha256(data_path)
        eligible = all(checks.values())
        reason = (
            "Compatible"
            if eligible
            else (
                "Historical source manifest has no dataset SHA-256; exact "
                "dataset fingerprint compatibility cannot be proven"
            )
        )
        return {
            "eligible": eligible,
            "checks": checks,
            "reason": reason,
            "source_run": str(source),
            "source_manifest_hash": _file_sha256(manifest_path),
            "current_dataset_sha256": current_fingerprint,
        }

    @staticmethod
    def _seed_run_frames(
        run_manifest: Mapping[str, Any],
        *,
        model_seed: int,
        split_seed: int,
    ) -> dict[str, Any]:
        run = Path(run_manifest["run_directory"])
        subjects = pd.read_csv(run / "calibration_subject_metrics.csv")
        subjects["model_seed"] = int(model_seed)
        subjects["split_seed"] = int(split_seed)
        subjects["source_group"] = subjects["source"].replace(
            {"gpn_data": "gpn_data", "Old_EEG": "Old_EEG", "both": "both"}
        )
        subjects["source_group"] = subjects["source_group"].fillna("unknown")
        predictions = pd.read_parquet(run / "predictions.parquet")
        predictions["model_seed"] = int(model_seed)
        predictions["split_seed"] = int(split_seed)
        checkpoints = pd.read_csv(run / "checkpoint_audit.csv")
        checkpoints["model_seed"] = int(model_seed)
        checkpoints["split_seed"] = int(split_seed)
        splits = pd.read_csv(run / "calibration_split_audit.csv")
        splits["model_seed"] = int(model_seed)
        splits["split_seed"] = int(split_seed)
        global_folds = pd.read_csv(run / "global_fold_summary.csv")
        global_folds["model_seed"] = int(model_seed)
        global_folds["split_seed"] = int(split_seed)
        global_folds["torch_version"] = torch.__version__
        global_folds["cuda_version"] = torch.version.cuda
        global_folds["global_training_time_seconds"] = global_folds[
            "training_time_seconds"
        ]
        failures = pd.read_csv(run / "failures.csv")
        failures["model_seed"] = int(model_seed)
        failures["split_seed"] = int(split_seed)
        return {
            "run": run,
            "subjects": subjects,
            "predictions": predictions,
            "checkpoints": checkpoints,
            "splits": splits,
            "global_folds": global_folds,
            "failures": failures,
        }

    @staticmethod
    def _validation_split_file(base_run: Path, fold: str) -> Path:
        matches = list(base_run.rglob(f"{fold}/validation_split.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one validation_split.json for {fold}, got {len(matches)}"
            )
        return matches[0]

    def _split_consistency(
        self,
        seed_frames: Mapping[int, Mapping[str, Any]],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for model_seed, frames in sorted(seed_frames.items()):
            subjects = frames["subjects"]
            split_audit = frames["splits"]
            checkpoint_audit = frames["checkpoints"]
            run = Path(frames["run"])
            base_run = _repo_path(
                json.loads((run / "run_manifest.json").read_text(
                    encoding="utf-8"
                ))["base_run_directory"]
            )
            for (fold, subject_id), subject_group in subjects.groupby(
                ["outer_fold", "subject_id"], sort=True
            ):
                method = (
                    "full_model"
                    if "full_model" in set(subject_group["method"])
                    else str(subject_group.iloc[0]["method"])
                )
                row = subject_group.loc[
                    subject_group["method"].astype(str) == method
                ].iloc[0]
                artifact = (
                    run
                    / str(fold)
                    / str(subject_id)
                    / f"budget_{float(row['budget_fraction']):.4f}fraction"
                    / method
                )
                split_path = artifact / "calibration_split.json"
                preprocessing_path = artifact / "preprocessing_audit.json"
                if not split_path.is_file() or not preprocessing_path.is_file():
                    raise RuntimeError(
                        f"Missing split/preprocessing audit for {artifact}"
                    )
                split = json.loads(split_path.read_text(encoding="utf-8"))
                preprocessing = json.loads(
                    preprocessing_path.read_text(encoding="utf-8")
                )
                validation = json.loads(
                    self._validation_split_file(
                        base_run, str(fold)
                    ).read_text(encoding="utf-8")
                )
                outer_train = sorted(
                    set(validation["inner_train_subject_ids"])
                    | set(validation["inner_validation_subject_ids"])
                )
                audit = split_audit.loc[
                    (split_audit["outer_fold"].astype(str) == str(fold))
                    & (
                        split_audit["subject_id"].astype(str)
                        == str(subject_id)
                    )
                    & (split_audit["method"].astype(str) == method)
                ].iloc[0]
                checkpoint = checkpoint_audit.loc[
                    (checkpoint_audit["outer_fold"].astype(str) == str(fold))
                    & (
                        checkpoint_audit["subject_id"].astype(str)
                        == str(subject_id)
                    )
                    & (checkpoint_audit["method"].astype(str) == method)
                ].iloc[0]
                prediction_path = artifact / "predictions.parquet"
                prediction_hash = (
                    None
                    if not prediction_path.is_file()
                    else _file_sha256(prediction_path)
                )
                rows.append({
                    "outer_fold": str(fold),
                    "subject_id": str(subject_id),
                    "split_seed": self.split_seed,
                    "model_seed": int(model_seed),
                    "outer_train_subject_hash": _ordered_hash(outer_train),
                    "inner_train_subject_hash": _ordered_hash(
                        validation["inner_train_subject_ids"]
                    ),
                    "inner_validation_subject_hash": _ordered_hash(
                        validation["inner_validation_subject_ids"]
                    ),
                    "calibration_sample_hash": _ordered_hash(
                        split["calibration_sample_ids"]
                    ),
                    "adaptation_train_sample_hash": _ordered_hash(
                        split["adaptation_train_sample_ids"]
                    ),
                    "adaptation_validation_sample_hash": _ordered_hash(
                        split["adaptation_validation_sample_ids"]
                    ),
                    "evaluation_sample_hash": _ordered_hash(
                        split["evaluation_sample_ids"]
                    ),
                    "preprocessor_hash": preprocessing["state_hash"],
                    "global_checkpoint_hash": checkpoint[
                        "global_checkpoint_hash"
                    ],
                    "fine_tune_final_hash": checkpoint[
                        "fine_tune_final_hash"
                    ],
                    "predictions_hash": prediction_hash,
                    "global_target_overlap": int(
                        audit["global_target_overlap"]
                    ),
                    "calibration_evaluation_overlap": int(
                        audit["calibration_evaluation_overlap"]
                    ),
                    "adaptation_validation_overlap": int(
                        audit["adaptation_validation_overlap"]
                    ),
                    "evaluation_overlap": int(audit["evaluation_overlap"]),
                    "duplicate_sample_ids": int(audit["duplicate_sample_ids"]),
                })
        frame = pd.DataFrame(rows)
        grouped = frame.groupby(
            ["outer_fold", "subject_id"], sort=True
        )
        for column in SPLIT_HASH_COLUMNS:
            frame[f"{column}_consistent"] = grouped[column].transform(
                "nunique"
            ).eq(1)
        frame["all_split_hashes_consistent"] = frame[
            [f"{column}_consistent" for column in SPLIT_HASH_COLUMNS]
        ].all(axis=1)
        if not frame["all_split_hashes_consistent"].all():
            raise RuntimeError("Split consistency audit failed between model seeds")
        overlap_columns = [
            "global_target_overlap",
            "calibration_evaluation_overlap",
            "adaptation_validation_overlap",
            "evaluation_overlap",
            "duplicate_sample_ids",
        ]
        if int(frame[overlap_columns].to_numpy(dtype=int).sum()) != 0:
            raise RuntimeError("Non-zero overlap found in multiseed audit")
        for fold, group in frame.groupby("outer_fold", sort=True):
            hashes = group.groupby("model_seed")[
                "global_checkpoint_hash"
            ].first()
            if hashes.nunique() != len(self.model_seeds):
                raise RuntimeError(
                    f"Global checkpoint hashes do not differ by seed in {fold}"
                )
        return frame

    def execute(
        self,
        *,
        fold_limit: Optional[int] = None,
        subject_limit: Optional[int] = None,
        max_epochs: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for multiseed personalization; CPU fallback "
                "is disabled"
            )
        implementation_hash = _implementation_hash()
        resolved = {
            "schema_version": MULTISEED_SCHEMA_VERSION,
            "document": self.document,
            "fold_limit": fold_limit,
            "subject_limit": subject_limit,
            "max_epochs": max_epochs,
            "implementation_hash": implementation_hash,
        }
        config_hash = _canonical_hash(resolved)
        root = _repo_path(
            output_dir or self.experiment["output_dir"]
        )
        resume_enabled = bool(resume or self.experiment.get("resume", False))
        run_dir: Optional[Path] = None
        if resume_enabled and root.is_dir():
            for candidate in sorted(
                (path for path in root.iterdir() if path.is_dir()),
                reverse=True,
            ):
                progress_path = candidate / "progress.json"
                if not progress_path.is_file():
                    continue
                progress = json.loads(
                    progress_path.read_text(encoding="utf-8")
                )
                if progress.get("config_hash") != config_hash:
                    continue
                if progress.get("implementation_hash") != implementation_hash:
                    raise RuntimeError(
                        "Multiseed resume implementation hash mismatch"
                    )
                run_dir = candidate
                manifest_path = run_dir / "run_manifest.json"
                if (
                    progress.get("status") == "completed"
                    and manifest_path.is_file()
                ):
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["resumed"] = True
                    manifest["resume_skipped_model_seeds"] = len(
                        progress.get("completed_model_seeds", [])
                    )
                    return manifest
                break
        if run_dir is None:
            run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=False)
            _write_json(
                run_dir / "progress.json",
                {
                    "schema_version": MULTISEED_SCHEMA_VERSION,
                    "status": "running",
                    "config_hash": config_hash,
                    "implementation_hash": implementation_hash,
                    "completed_model_seeds": [],
                },
            )
            with open(
                run_dir / "resolved_multiseed.yaml", "w", encoding="utf-8"
            ) as output:
                yaml.safe_dump(resolved, output, sort_keys=False)

        compatibility = self._seed42_compatibility()
        completed_seeds: dict[int, dict[str, Any]] = {}
        for manifest_path in run_dir.glob(
            "seed_*/personalization/*/run_manifest.json"
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                continue
            seed = int(manifest_path.parts[-4].split("_", 1)[1])
            completed_seeds[seed] = manifest

        started = time.perf_counter()
        provenance_rows: list[dict[str, Any]] = []
        new_global_trainings = 0
        for model_seed in self.model_seeds:
            was_completed = model_seed in completed_seeds
            seed_root = run_dir / f"seed_{model_seed}"
            base_config_path = seed_root / "resolved_base.yaml"
            calibration_config_path = seed_root / "resolved_calibration.yaml"
            base_output = seed_root / "global_base"
            calibration_output = seed_root / "personalization"
            seed_root.mkdir(parents=True, exist_ok=True)
            global_maximum = self.experiment.get("global_max_epochs")
            base_config = resolve_seed_base_config(
                self.base_template,
                model_seed=model_seed,
                split_seed=self.split_seed,
                output_dir=base_output,
                maximum_epochs=(
                    None if global_maximum is None else int(global_maximum)
                ),
            )
            calibration_config = resolve_seed_calibration_config(
                self.document,
                model_seed=model_seed,
                split_seed=self.split_seed,
                base_config_path=base_config_path,
                output_dir=calibration_output,
            )
            with open(base_config_path, "w", encoding="utf-8") as output:
                yaml.safe_dump(base_config, output, sort_keys=False)
            with open(calibration_config_path, "w", encoding="utf-8") as output:
                yaml.safe_dump(calibration_config, output, sort_keys=False)
            if model_seed not in completed_seeds:
                experiment = UserCalibrationExperiment(calibration_config_path)
                manifest = experiment.execute(
                    fold_limit=fold_limit,
                    subject_limit=subject_limit,
                    max_epochs=max_epochs,
                    random_state=model_seed,
                    write_reports=False,
                    resume=resume_enabled,
                )
                completed_seeds[model_seed] = manifest
                new_global_trainings += len(manifest["folds"])
                del experiment
                gc.collect()
                torch.cuda.empty_cache()
            provenance_rows.append({
                "model_seed": model_seed,
                "split_seed": self.split_seed,
                "seed_42_source_run": compatibility.get("source_run"),
                "seed_42_manifest_hash": compatibility.get(
                    "source_manifest_hash"
                ),
                "seed_42_reused": False,
                "seed_42_compatibility_eligible": compatibility["eligible"],
                "seed_42_compatibility_checks": json.dumps(
                    compatibility["checks"], sort_keys=True
                ),
                "seed_42_compatibility_reason": compatibility["reason"],
                "run_directory": completed_seeds[model_seed]["run_directory"],
                "base_run_directory": completed_seeds[model_seed][
                    "base_run_directory"
                ],
                "outcome": "resumed" if was_completed else "completed",
            })
            _write_json(
                run_dir / "progress.json",
                {
                    "schema_version": MULTISEED_SCHEMA_VERSION,
                    "status": "running",
                    "config_hash": config_hash,
                    "implementation_hash": implementation_hash,
                    "completed_model_seeds": sorted(completed_seeds),
                },
            )

        seed_frames = {
            seed: self._seed_run_frames(
                completed_seeds[seed],
                model_seed=seed,
                split_seed=self.split_seed,
            )
            for seed in self.model_seeds
        }
        subjects = pd.concat(
            [frame["subjects"] for frame in seed_frames.values()],
            ignore_index=True,
        )
        predictions = pd.concat(
            [frame["predictions"] for frame in seed_frames.values()],
            ignore_index=True,
        )
        checkpoints = pd.concat(
            [frame["checkpoints"] for frame in seed_frames.values()],
            ignore_index=True,
        )
        valid_checkpoints = checkpoints.loc[
            checkpoints["fine_tune_initial_hash"].notna()
        ]
        if not valid_checkpoints["initial_matches_global"].astype(bool).all():
            raise RuntimeError("Fine-tuning did not start from its global checkpoint")
        if not valid_checkpoints[
            "initial_predictions_match_global"
        ].astype(bool).all():
            raise RuntimeError("Initial clone predictions differ from global model")
        head_checkpoints = valid_checkpoints.loc[
            valid_checkpoints["method"].astype(str) == "head_only"
        ]
        if not head_checkpoints[
            "frozen_parameters_unchanged"
        ].astype(bool).all():
            raise RuntimeError("Head-only changed frozen backbone parameters")
        zero_checkpoints = valid_checkpoints.loc[
            valid_checkpoints["method"].astype(str) == "zero_shot"
        ]
        if not zero_checkpoints["fine_tune_initial_hash"].eq(
            zero_checkpoints["fine_tune_final_hash"]
        ).all():
            raise RuntimeError("Zero-shot changed checkpoint parameters")
        global_folds = pd.concat(
            [frame["global_folds"] for frame in seed_frames.values()],
            ignore_index=True,
        )
        failures = pd.concat(
            [frame["failures"] for frame in seed_frames.values()],
            ignore_index=True,
        )
        split_consistency = self._split_consistency(seed_frames)
        subject_fold_counts = subjects.groupby("subject_id").agg(
            model_seeds=("model_seed", "nunique"),
            outer_folds=("outer_fold", "nunique"),
        )
        if (
            not subject_fold_counts["model_seeds"].eq(
                len(self.model_seeds)
            ).all()
            or not subject_fold_counts["outer_folds"].eq(1).all()
        ):
            raise RuntimeError(
                "Outer-fold subject assignments differ between model seeds"
            )

        probability_columns = [f"proba_{index}" for index in range(5)]
        probability = predictions[probability_columns].to_numpy(dtype=float)
        if (
            not np.isfinite(probability).all()
            or np.min(probability) < -1e-8
            or np.max(np.abs(probability.sum(axis=1) - 1.0)) > 1e-6
        ):
            raise RuntimeError("Invalid probabilities in multiseed predictions")
        prediction_key = [
            "model_seed", "outer_fold", "subject_id", "method", "sample_id"
        ]
        predictions["method"] = predictions["calibration_method"]
        if predictions.duplicated(prediction_key).any():
            raise RuntimeError("Duplicate multiseed condition/sample prediction")

        bootstrap_samples = int(
            self.experiment.get("bootstrap_samples", 1000)
        )
        per_seed, multiseed, source_summary, subject_seed_means = (
            build_multiseed_aggregates(
                subjects,
                model_seeds=self.model_seeds,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=self.split_seed,
            )
        )
        stability = build_stability_summary(
            subjects, model_seeds=self.model_seeds
        )
        paired_by_seed = _paired_comparisons(
            subjects.loc[subjects["status"] == "completed"],
            grouping=("model_seed",),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=self.split_seed,
        )
        paired_multiseed = _paired_comparisons(
            subject_seed_means,
            grouping=(),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=self.split_seed,
        )
        threshold = build_threshold_summary(
            subjects, model_seeds=self.model_seeds
        )
        completed_seed_counts = subjects.loc[
            subjects["status"] == "completed"
        ].groupby(["subject_id", "method"])["model_seed"].nunique()
        all_keys = set(
            subjects[["subject_id", "method"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        complete_keys = set(
            completed_seed_counts.loc[
                completed_seed_counts == len(self.model_seeds)
            ].index
        )
        incomplete_keys = all_keys - complete_keys
        incomplete = subjects.loc[
            [
                (str(row.subject_id), str(row.method)) in incomplete_keys
                for row in subjects.itertuples()
            ]
        ]

        paths = {
            "seed_provenance": run_dir / "seed_provenance.csv",
            "global_fold_summary": run_dir / "global_fold_summary.csv",
            "multiseed_subject_metrics": run_dir / "multiseed_subject_metrics.csv",
            "per_seed_aggregate": run_dir / "per_seed_aggregate.csv",
            "multiseed_aggregate": run_dir / "multiseed_aggregate.csv",
            "multiseed_source_summary": run_dir / "multiseed_source_summary.csv",
            "stability_summary": run_dir / "stability_summary.csv",
            "paired_comparisons_by_seed": (
                run_dir / "paired_comparisons_by_seed.csv"
            ),
            "paired_comparisons_multiseed": (
                run_dir / "paired_comparisons_multiseed.csv"
            ),
            "threshold_75_multiseed": run_dir / "threshold_75_multiseed.csv",
            "split_consistency_audit": run_dir / "split_consistency_audit.csv",
            "checkpoint_audit": run_dir / "checkpoint_audit.csv",
            "predictions": run_dir / "predictions.parquet",
            "failures": run_dir / "failures.csv",
            "incomplete_subjects": run_dir / "incomplete_subjects.csv",
        }
        pd.DataFrame(provenance_rows).to_csv(
            paths["seed_provenance"], index=False
        )
        global_folds.to_csv(paths["global_fold_summary"], index=False)
        subjects.to_csv(paths["multiseed_subject_metrics"], index=False)
        per_seed.to_csv(paths["per_seed_aggregate"], index=False)
        multiseed.to_csv(paths["multiseed_aggregate"], index=False)
        source_summary.to_csv(paths["multiseed_source_summary"], index=False)
        stability.to_csv(paths["stability_summary"], index=False)
        paired_by_seed.to_csv(paths["paired_comparisons_by_seed"], index=False)
        paired_multiseed.to_csv(
            paths["paired_comparisons_multiseed"], index=False
        )
        threshold.to_csv(paths["threshold_75_multiseed"], index=False)
        split_consistency.to_csv(
            paths["split_consistency_audit"], index=False
        )
        checkpoints.to_csv(paths["checkpoint_audit"], index=False)
        predictions.to_parquet(paths["predictions"], index=False)
        failures.to_csv(paths["failures"], index=False)
        incomplete.to_csv(paths["incomplete_subjects"], index=False)

        status_counts = subjects["status"].value_counts().to_dict()
        manifest = {
            "schema_version": MULTISEED_SCHEMA_VERSION,
            "status": "completed",
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "split_seed": self.split_seed,
            "model_seeds": list(self.model_seeds),
            "run_directory": str(run_dir),
            "seed_42_reused": False,
            "seed_42_compatibility": compatibility,
            "new_global_trainings": int(new_global_trainings),
            "global_fold_rows": int(len(global_folds)),
            "subjects": int(subjects["subject_id"].nunique()),
            "complete_case_subjects": int(
                subject_seed_means["subject_id"].nunique()
            ),
            "condition_rows": int(len(subjects)),
            "status_counts": {
                str(key): int(value) for key, value in status_counts.items()
            },
            "prediction_rows": int(len(predictions)),
            "failed_conditions": int(len(failures)),
            "elapsed_seconds": float(time.perf_counter() - started),
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "maximum_probability_sum_error": float(
                np.max(np.abs(probability.sum(axis=1) - 1.0))
            ),
            "artifacts": {
                key: str(value) for key, value in paths.items()
            },
            "seed_runs": {
                str(seed): completed_seeds[seed]["run_directory"]
                for seed in self.model_seeds
            },
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        _write_json(
            run_dir / "progress.json",
            {
                "schema_version": MULTISEED_SCHEMA_VERSION,
                "status": "completed",
                "config_hash": config_hash,
                "implementation_hash": implementation_hash,
                "completed_model_seeds": list(self.model_seeds),
            },
        )
        return manifest


__all__ = [
    "METRICS",
    "METHODS",
    "UserCalibrationMultiseedExperiment",
    "build_multiseed_aggregates",
    "build_stability_summary",
    "build_threshold_summary",
    "load_multiseed_calibration_spec",
    "resolve_seed_base_config",
    "resolve_seed_calibration_config",
]
