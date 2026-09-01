"""Canonical seven-PM LightGBM and fold-local feature-selection benchmark."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from bench.experiments.pm_all_targets_feature_baseline import (
    prepare_protocol as prepare_base_protocol,
    stable_hash,
)
from bench.tasks.target_registry import PM_METRICS
from cogstate.features.selection import FeatureSelector, SelectionConfig
from cogstate.model_zoo import build_model


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("classification", "regression")
FEATURE_REGIMES = ("all_features", "selected_top50")
CLASSIFICATION_METRICS = ("macro_f1", "balanced_accuracy", "accuracy")
REGRESSION_METRICS = ("mae", "rmse", "r2", "pearson", "spearman")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    document = yaml.safe_load(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("LightGBM feature-selection config must be a mapping")
    required = {"experiment_id", "base_protocol_config", "targets", "tasks", "feature_regimes", "selector", "lightgbm", "evaluation", "smoke", "output_dir"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"LightGBM config is missing sections: {missing}")
    if tuple(document["targets"]) != PM_METRICS:
        raise ValueError(f"targets must equal {PM_METRICS}")
    if tuple(document["tasks"]) != TASKS:
        raise ValueError(f"tasks must equal {TASKS}")
    if tuple(document["feature_regimes"]) != FEATURE_REGIMES:
        raise ValueError(f"feature_regimes must equal {FEATURE_REGIMES}")
    if document["selector"].get("method") != "tree_importance":
        raise ValueError("Primary selector is locked to tree_importance")
    if int(document["selector"].get("top_k", 0)) != 50:
        raise ValueError("Primary selector top_k is locked to 50")
    if tuple(map(int, document["evaluation"]["folds"])) != (1, 2, 3, 4, 5):
        raise ValueError("Evaluation folds must be [1, 2, 3, 4, 5]")
    if int(document["lightgbm"]["params"].get("random_state", -1)) != 42:
        raise ValueError("LightGBM random_state must be 42")
    return document


@dataclass(frozen=True)
class LightGBMRunSpec:
    metric: str
    task_type: str
    feature_regime: str
    fold: int
    seed: int = 42

    @property
    def run_id(self) -> str:
        return (
            f"{self.metric}__{self.task_type}__{self.feature_regime}__"
            f"fold{self.fold:02d}__seed{self.seed}"
        )

    def specification_hash(self, protocol_hash: str) -> str:
        """Bind one run cell to the complete resolved experiment protocol."""
        normalized = str(protocol_hash).strip()
        if not normalized:
            raise ValueError("protocol_hash must be non-empty")
        return stable_hash({
            "run_spec": asdict(self),
            "protocol_hash": normalized,
        })


def build_run_matrix(config: Mapping[str, Any]) -> list[LightGBMRunSpec]:
    return [
        LightGBMRunSpec(metric, task, regime, int(fold), 42)
        for metric in config["targets"]
        for task in config["tasks"]
        for regime in config["feature_regimes"]
        for fold in config["evaluation"]["folds"]
    ]


def _lightgbm_version() -> str | None:
    if importlib.util.find_spec("lightgbm") is None:
        return None
    return importlib.metadata.version("lightgbm")


def protocol_plan(config_path: str | Path) -> dict[str, Any]:
    path = _repo_path(config_path)
    config = load_config(path)
    base = prepare_base_protocol(_repo_path(config["base_protocol_config"]))
    specs = build_run_matrix(config)
    names = base.feature_names["eeg_pow"]
    if len(names) != 448:
        raise ValueError(f"Canonical EEG+POW feature count changed: {len(names)}")
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "result_status": config.get("result_status", "baseline"),
        "analysis_role": "confirmatory_protocol_not_executed",
        "config_path": path.relative_to(REPO_ROOT).as_posix(),
        "base_protocol_hash": base.preregistration["protocol_hash"],
        "fixed_outer_folds": base.folds,
        "targets": list(PM_METRICS),
        "tasks": list(TASKS),
        "feature_regimes": list(FEATURE_REGIMES),
        "feature_count": len(names),
        "feature_list_hash": base.preregistration["feature_sets"]["eeg_pow"]["feature_list_sha256"],
        "target_contract": {
            "classification": "outer_train_q3_low_medium_high",
            "regression": "continuous_identity",
        },
        "selector": config["selector"],
        "selector_fit_scope": "outer_train_only_including_correlation_filter",
        "lightgbm": {
            **config["lightgbm"],
            "installed_version": _lightgbm_version(),
            "available": _lightgbm_version() is not None,
        },
        "run_count": len(specs),
        "expected_run_count": 140,
        "run_matrix_hash": stable_hash([asdict(spec) for spec in specs]),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    manifest["protocol_hash"] = stable_hash(manifest)
    return manifest


def write_plan(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output = _repo_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = protocol_plan(config_path)
    _write_json(output / "protocol_manifest.json", manifest)
    specs = build_run_matrix(config)
    protocol_hash = str(manifest["protocol_hash"])
    rows = [
        {
            **asdict(spec),
            "run_id": spec.run_id,
            "specification_hash": spec.specification_hash(protocol_hash),
        }
        for spec in specs
    ]
    pd.DataFrame(rows).to_csv(output / "run_matrix.csv", index=False, lineterminator="\n")
    smoke = config["smoke"]
    smoke_rows = [
        row for row, spec in zip(rows, specs)
        if spec.metric in set(smoke["targets"])
        and spec.fold in set(map(int, smoke["folds"]))
    ]
    pd.DataFrame(smoke_rows).to_csv(
        output / "smoke_run_matrix.csv", index=False, lineterminator="\n"
    )
    return manifest


def _load_resumable_summary(
    summary_path: Path,
    *,
    resume: bool,
    specification_hash: str,
) -> dict[str, Any] | None:
    """Return only a completed summary bound to the current protocol and run."""
    if not resume or not summary_path.exists():
        return None
    existing = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        existing.get("status") == "complete"
        and existing.get("specification_hash") == specification_hash
    ):
        return existing
    return None


def _indices(base: Any, metric: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    mask = base.target_masks[f"pm_{metric}_regression"]
    fold_values = base.frame["subject_id"].astype(str).map(base.fold_by_subject).to_numpy()
    train = np.flatnonzero(mask & (fold_values != fold))
    test = np.flatnonzero(mask & (fold_values == fold))
    overlap = sorted(
        set(base.frame.iloc[train]["subject_id"].astype(str))
        & set(base.frame.iloc[test]["subject_id"].astype(str))
    )
    if overlap:
        raise RuntimeError(f"Outer participant leakage: {overlap}")
    return train, test


def _q3(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    thresholds = np.quantile(values.astype(float), [1 / 3, 2 / 3]).tolist()
    if len(set(map(float, thresholds))) != 2:
        raise ValueError("Outer-train Q3 thresholds are not unique")
    return np.searchsorted(thresholds, values, side="right").astype(np.int64), [float(x) for x in thresholds]


def _correlation(values: np.ndarray, prediction: np.ndarray, kind: str) -> float:
    if len(values) < 2 or np.ptp(values) == 0 or np.ptp(prediction) == 0:
        return float("nan")
    function = pearsonr if kind == "pearson" else spearmanr
    return float(function(values, prediction).statistic)


def _metric_values(task: str, truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if task == "classification":
        return {
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "accuracy": float(accuracy_score(truth, prediction)),
        }
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else float("nan"),
        "pearson": _correlation(truth, prediction, "pearson"),
        "spearman": _correlation(truth, prediction, "spearman"),
    }


def _participant_macro(
    task: str, truth: np.ndarray, prediction: np.ndarray, subjects: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    for subject in sorted(np.unique(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        rows.append({"subject_id": subject, "n_windows": int(mask.sum()), **_metric_values(task, truth[mask], prediction[mask])})
    frame = pd.DataFrame(rows)
    metrics = {
        column: float(frame[column].mean(skipna=True))
        for column in frame.columns if column not in {"subject_id", "n_windows"}
    }
    return metrics, frame


def _selector_manifest(
    selector: FeatureSelector,
    names: Sequence[str],
    *,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    task_type: str,
    selector_config: Mapping[str, Any],
    metric: str,
    fold: int,
) -> dict[str, Any]:
    result = selector.result
    if result is None:
        raise RuntimeError("Selector result is absent after fit")
    selected = list(result.selected_names)
    groups = {
        "EEG": sum(name.startswith("EEG.") for name in selected),
        "POW": sum(name.startswith("POW.") for name in selected),
    }
    payload = {
        "task_type": task_type,
        "metric": metric,
        "fold": int(fold),
        "selector_config": dict(selector_config),
        "fit_scope": "outer_train_only",
        "correlation_fit_scope": "outer_train_only",
        "transform_refits": False,
        "original_feature_count": len(names),
        "post_redundancy_feature_count": len(names) - len(result.dropped_redundant),
        "selected_count": len(selected),
        "selected_names": selected,
        "selected_indices": list(map(int, result.selected_indices)),
        "importances": np.asarray(result.importances, dtype=float).tolist(),
        "dropped_redundant": list(result.dropped_redundant),
        "feature_group_counts": groups,
        "outer_train_ids_hash": stable_hash(train_ids.astype(str).tolist()),
        "outer_test_ids_hash": stable_hash(test_ids.astype(str).tolist()),
        "sample_id_overlap_count": int(len(np.intersect1d(train_ids, test_ids))),
    }
    payload["selector_fit_hash"] = stable_hash(payload)
    return payload


def execute_run(
    config_path: str | Path,
    spec: LightGBMRunSpec,
    *,
    base: Any,
    resume: bool,
    protocol_hash: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    run_dir = _repo_path(config["output_dir"]) / "runs" / spec.run_id
    summary_path = run_dir / "run_summary.json"
    specification_hash = spec.specification_hash(protocol_hash)
    existing = _load_resumable_summary(
        summary_path,
        resume=resume,
        specification_hash=specification_hash,
    )
    if existing is not None:
        return existing
    train_idx, test_idx = _indices(base, spec.metric, spec.fold)
    names = list(base.feature_names["eeg_pow"])
    X = base.features["eeg_pow"]
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_train_raw)
    X_train = np.asarray(imputer.transform(X_train_raw), dtype=np.float32)
    X_test = np.asarray(imputer.transform(X_test_raw), dtype=np.float32)
    continuous = np.asarray(base.target_values[f"pm_{spec.metric}_regression"], dtype=np.float64)
    thresholds: list[float] | None = None
    if spec.task_type == "classification":
        _, thresholds = _q3(continuous[train_idx])
        y_train = np.searchsorted(thresholds, continuous[train_idx], side="right").astype(np.int64)
        y_test = np.searchsorted(thresholds, continuous[test_idx], side="right").astype(np.int64)
    else:
        y_train, y_test = continuous[train_idx], continuous[test_idx]
    selector_manifest: dict[str, Any] | None = None
    if spec.feature_regime == "selected_top50":
        selection = dict(config["selector"])
        selection.pop("provenance", None)
        selector = FeatureSelector(SelectionConfig(task_type=spec.task_type, **selection))
        selector.fit(X_train, y_train, names)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        selector_manifest = _selector_manifest(
            selector, names,
            train_ids=base.frame.iloc[train_idx]["sample_id"].to_numpy(),
            test_ids=base.frame.iloc[test_idx]["sample_id"].to_numpy(),
            task_type=spec.task_type,
            selector_config=selection,
            metric=spec.metric,
            fold=spec.fold,
        )
    params = dict(config["lightgbm"]["params"])
    params["random_state"] = spec.seed
    model = build_model("lightgbm", spec.task_type, None, None, params)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = np.asarray(model.predict(X_test))
    inference_seconds = time.perf_counter() - started
    probabilities = model.predict_proba(X_test) if spec.task_type == "classification" else None
    window_metrics = _metric_values(spec.task_type, y_test, prediction)
    subjects = base.frame.iloc[test_idx]["subject_id"].astype(str).to_numpy()
    participant_metrics, participant_frame = _participant_macro(
        spec.task_type, y_test, prediction, subjects
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if selector_manifest is not None:
        _write_json(run_dir / "selector_manifest.json", selector_manifest)
        pd.DataFrame({
            "selected_index": selector_manifest["selected_indices"],
            "selected_name": selector_manifest["selected_names"],
        }).to_csv(run_dir / "selected_features.csv", index=False, lineterminator="\n")
    predictions = pd.DataFrame({
        "sample_id": base.frame.iloc[test_idx]["sample_id"].to_numpy(),
        "subject_id": subjects,
        "record_id": base.frame.iloc[test_idx]["record_id"].astype(str).to_numpy(),
        "fold": spec.fold, "pm": spec.metric, "task_type": spec.task_type,
        "feature_regime": spec.feature_regime, "y_true": y_test, "y_pred": prediction,
    })
    if probabilities is not None:
        for class_id in range(probabilities.shape[1]):
            predictions[f"proba_{class_id}"] = probabilities[:, class_id]
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    participant_frame.to_csv(run_dir / "participant_metrics.csv", index=False, lineterminator="\n")
    summary = {
        "status": "complete", "result_status": config.get("result_status", "baseline"),
        "run_id": spec.run_id, "specification_hash": specification_hash,
        **asdict(spec), "window_metrics": window_metrics,
        "participant_macro_metrics": participant_metrics,
        "q3_thresholds": thresholds,
        "q3_fit_scope": "outer_train_only" if thresholds is not None else "not_applicable",
        "train_windows": int(len(train_idx)), "test_windows": int(len(test_idx)),
        "original_feature_count": len(names), "model_feature_count": int(X_train.shape[1]),
        "training_seconds": training_seconds, "inference_seconds": inference_seconds,
        "lightgbm_version": _lightgbm_version(),
        "lightgbm_params": params,
        "parameter_provenance": config["lightgbm"]["provenance"],
        "outer_participant_overlap": [],
    }
    _write_json(summary_path, summary)
    return summary


def _build_all_vs_selected_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    for keys, group in frame.groupby(
        ["metric", "task_type", "fold", "aggregation"], dropna=False
    ):
        indexed = group.set_index("feature_regime")
        if set(indexed.index) != set(FEATURE_REGIMES) or len(indexed) != 2:
            raise ValueError(
                "Each PM/task/fold/aggregation comparison must contain exactly "
                f"{FEATURE_REGIMES}; got {list(indexed.index)} for {keys}"
            )
        row: dict[str, Any] = dict(
            zip(("metric", "task_type", "fold", "aggregation"), keys)
        )
        metric_names = (
            CLASSIFICATION_METRICS if keys[1] == "classification" else REGRESSION_METRICS
        )
        for metric_name in metric_names:
            all_value = float(indexed.loc["all_features", metric_name])
            selected_value = float(indexed.loc["selected_top50", metric_name])
            row[f"all_features_{metric_name}"] = all_value
            row[f"selected_top50_{metric_name}"] = selected_value
            if keys[1] == "classification":
                row[f"delta_{metric_name}"] = selected_value - all_value
            else:
                row[f"delta_{metric_name}_selected_minus_all"] = (
                    selected_value - all_value
                )
        comparisons.append(row)
    return pd.DataFrame(comparisons)


def _summaries_frame(summaries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        common = {key: summary[key] for key in ("metric", "task_type", "feature_regime", "fold", "training_seconds", "inference_seconds", "model_feature_count")}
        rows.append({**common, "aggregation": "window", **summary["window_metrics"]})
        rows.append({**common, "aggregation": "participant_macro", **summary["participant_macro_metrics"]})
    return pd.DataFrame(rows)


def _aggregate(output: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    frame = _summaries_frame(summaries)
    frame.to_csv(output / "aggregate_metrics.csv", index=False, lineterminator="\n")
    _build_all_vs_selected_comparison(frame).to_csv(
        output / "all_vs_selected_comparison.csv", index=False, lineterminator="\n"
    )
    manifests = []
    for path in sorted((output / "runs").glob("*/selector_manifest.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    if manifests:
        stability_rows: list[dict[str, Any]] = []
        jaccard_rows: list[dict[str, Any]] = []
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for manifest in manifests:
            grouped.setdefault(
                (str(manifest["metric"]), str(manifest["task_type"])), []
            ).append(manifest)
        for (metric, task_type), group in sorted(grouped.items()):
            counts: dict[str, int] = {}
            for manifest in group:
                for name in manifest["selected_names"]:
                    counts[name] = counts.get(name, 0) + 1
            observed_folds = len({int(item["fold"]) for item in group})
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                stability_rows.append({
                    "metric": metric,
                    "task_type": task_type,
                    "feature_name": name,
                    "feature_group": (
                        "EEG" if name.startswith("EEG.")
                        else "POW" if name.startswith("POW.")
                        else "other"
                    ),
                    "selection_count": count,
                    "observed_folds": observed_folds,
                    "stable_at_least_4_of_5": bool(observed_folds == 5 and count >= 4),
                    "selected_all_5_folds": bool(observed_folds == 5 and count == 5),
                })
            ordered = sorted(group, key=lambda item: int(item["fold"]))
            for left_index, left in enumerate(ordered):
                left_names = set(left["selected_names"])
                for right in ordered[left_index + 1:]:
                    right_names = set(right["selected_names"])
                    union = left_names | right_names
                    jaccard_rows.append({
                        "metric": metric,
                        "task_type": task_type,
                        "fold_a": int(left["fold"]),
                        "fold_b": int(right["fold"]),
                        "intersection_count": len(left_names & right_names),
                        "union_count": len(union),
                        "jaccard": float(len(left_names & right_names) / len(union))
                        if union else 1.0,
                    })
        pd.DataFrame(stability_rows).to_csv(
            output / "feature_stability_summary.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(jaccard_rows, columns=[
            "metric", "task_type", "fold_a", "fold_b",
            "intersection_count", "union_count", "jaccard",
        ]).to_csv(
            output / "feature_jaccard_similarity.csv", index=False, lineterminator="\n"
        )
        composition_rows = [
            {
                "metric": manifest["metric"],
                "task_type": manifest["task_type"],
                "fold": int(manifest["fold"]),
                "eeg_features": int(
                    manifest.get("feature_group_counts", {}).get(
                        "EEG",
                        sum(name.startswith("EEG.") for name in manifest["selected_names"]),
                    )
                ),
                "pow_features": int(
                    manifest.get("feature_group_counts", {}).get(
                        "POW",
                        sum(name.startswith("POW.") for name in manifest["selected_names"]),
                    )
                ),
                "selected_count": int(
                    manifest.get("selected_count", len(manifest["selected_names"]))
                ),
            }
            for manifest in manifests
        ]
        composition = pd.DataFrame(composition_rows)
        composition.to_csv(
            output / "feature_group_composition.csv", index=False, lineterminator="\n"
        )
        stability = pd.DataFrame(stability_rows)
        jaccard = pd.DataFrame(jaccard_rows)
        overview_rows = []
        for (metric, task_type), composition_group in composition.groupby(
            ["metric", "task_type"], sort=True
        ):
            stable_group = stability.loc[
                stability["metric"].eq(metric)
                & stability["task_type"].eq(task_type)
            ]
            jaccard_group = jaccard.loc[
                jaccard["metric"].eq(metric)
                & jaccard["task_type"].eq(task_type)
            ]
            overview_rows.append({
                "metric": metric,
                "task_type": task_type,
                "stable_at_least_4_of_5_count": int(
                    stable_group["stable_at_least_4_of_5"].sum()
                ),
                "selected_all_5_folds_count": int(
                    stable_group["selected_all_5_folds"].sum()
                ),
                "mean_eeg_features": float(composition_group["eeg_features"].mean()),
                "mean_pow_features": float(composition_group["pow_features"].mean()),
                "mean_jaccard": float(jaccard_group["jaccard"].mean()),
                "min_jaccard": float(jaccard_group["jaccard"].min()),
                "max_jaccard": float(jaccard_group["jaccard"].max()),
            })
        pd.DataFrame(overview_rows).to_csv(
            output / "feature_stability_overview.csv", index=False, lineterminator="\n"
        )
    (output / "report.md").write_text(
        "# LightGBM + fold-local feature selection\n\n"
        "All selectors, including correlation removal, are fitted on outer-train only. "
        "Feature importance is predictive, not causal.\n\n"
        + frame.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )


def _metric_summary(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(group_columns, key_values))
        task_type = str(group["task_type"].iloc[0])
        metric_names = (
            CLASSIFICATION_METRICS
            if task_type == "classification"
            else REGRESSION_METRICS
        )
        row["fold_count"] = int(group["fold"].nunique())
        for metric_name in metric_names:
            values = pd.to_numeric(group[metric_name], errors="coerce")
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_sample_sd"] = float(values.std(ddof=1))
            row[f"{metric_name}_valid_count"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _cohort_signature(frame: pd.DataFrame) -> str:
    ordered = frame[["sample_id", "y_true"]].copy()
    ordered["sample_id"] = ordered["sample_id"].astype(str)
    ordered = ordered.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _audit_evaluation_cohorts(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    by_key = {
        (
            str(summary["metric"]),
            str(summary["task_type"]),
            int(summary["fold"]),
            str(summary["feature_regime"]),
        ): summary
        for summary in summaries
    }
    rows = []
    base_keys = sorted({key[:3] for key in by_key})
    for metric, task_type, fold in base_keys:
        all_summary = by_key[(metric, task_type, fold, "all_features")]
        selected_summary = by_key[(metric, task_type, fold, "selected_top50")]
        all_path = output / "runs" / str(all_summary["run_id"]) / "predictions.parquet"
        selected_path = (
            output / "runs" / str(selected_summary["run_id"]) / "predictions.parquet"
        )
        all_predictions = pd.read_parquet(all_path, columns=["sample_id", "y_true"])
        selected_predictions = pd.read_parquet(
            selected_path, columns=["sample_id", "y_true"]
        )
        all_ordered = all_predictions.sort_values("sample_id", kind="mergesort")
        selected_ordered = selected_predictions.sort_values(
            "sample_id", kind="mergesort"
        )
        sample_ids_match = np.array_equal(
            all_ordered["sample_id"].astype(str).to_numpy(),
            selected_ordered["sample_id"].astype(str).to_numpy(),
        )
        targets_match = (
            len(all_ordered) == len(selected_ordered)
            and np.allclose(
                all_ordered["y_true"].to_numpy(dtype=float),
                selected_ordered["y_true"].to_numpy(dtype=float),
                equal_nan=True,
            )
        )
        counts_match = (
            int(all_summary["train_windows"]) == int(selected_summary["train_windows"])
            and int(all_summary["test_windows"]) == int(selected_summary["test_windows"])
        )
        thresholds_match = stable_hash(all_summary.get("q3_thresholds")) == stable_hash(
            selected_summary.get("q3_thresholds")
        )
        rows.append({
            "metric": metric,
            "task_type": task_type,
            "fold": fold,
            "sample_count": int(len(all_ordered)),
            "sample_ids_match": bool(sample_ids_match),
            "targets_match": bool(targets_match),
            "train_test_counts_match": bool(counts_match),
            "q3_thresholds_match": bool(thresholds_match),
            "all_features_cohort_hash": _cohort_signature(all_predictions),
            "selected_top50_cohort_hash": _cohort_signature(selected_predictions),
        })
    audit = pd.DataFrame(rows)
    required = [
        "sample_ids_match", "targets_match", "train_test_counts_match",
        "q3_thresholds_match",
    ]
    if len(audit) != 70 or not audit[required].to_numpy(dtype=bool).all():
        raise RuntimeError("Feature regimes do not use identical evaluation cohorts")
    return audit


def _computational_summary(summaries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(summaries)
    rows = []
    for scope, group in [
        ("all_tasks", frame),
        *[(task, frame.loc[frame["task_type"].eq(task)]) for task in TASKS],
    ]:
        all_features = group.loc[group["feature_regime"].eq("all_features")]
        selected = group.loc[group["feature_regime"].eq("selected_top50")]
        all_training = float(all_features["training_seconds"].mean())
        selected_training = float(selected["training_seconds"].mean())
        all_inference = float(all_features["inference_seconds"].mean())
        selected_inference = float(selected["inference_seconds"].mean())
        rows.append({
            "scope": scope,
            "all_features_count": 448,
            "selected_top50_count": 50,
            "absolute_feature_reduction": 398,
            "relative_feature_reduction": 398 / 448,
            "all_features_mean_training_seconds": all_training,
            "selected_top50_mean_training_seconds": selected_training,
            "training_speedup_all_divided_by_selected": all_training / selected_training,
            "all_features_mean_inference_seconds": all_inference,
            "selected_top50_mean_inference_seconds": selected_inference,
            "inference_speedup_all_divided_by_selected": all_inference / selected_inference,
            "timing_scope": "downstream_lightgbm_only_excludes_feature_selector_fit",
        })
    return pd.DataFrame(rows)


def _pm_macro_comparison(pm_macro_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task_type, aggregation), group in pm_macro_summary.groupby(
        ["task_type", "aggregation"], sort=True
    ):
        indexed = group.set_index("feature_regime")
        if set(indexed.index) != set(FEATURE_REGIMES):
            raise RuntimeError("PM-macro comparison requires both feature regimes")
        row: dict[str, Any] = {
            "task_type": task_type,
            "aggregation": aggregation,
        }
        metric_names = (
            CLASSIFICATION_METRICS
            if task_type == "classification"
            else REGRESSION_METRICS
        )
        for metric_name in metric_names:
            column = f"{metric_name}_mean"
            all_value = float(indexed.loc["all_features", column])
            selected_value = float(indexed.loc["selected_top50", column])
            row[f"all_features_{metric_name}"] = all_value
            row[f"selected_top50_{metric_name}"] = selected_value
            delta_name = (
                f"delta_{metric_name}"
                if task_type == "classification"
                else f"delta_{metric_name}_selected_minus_all"
            )
            row[delta_name] = selected_value - all_value
        rows.append(row)
    return pd.DataFrame(rows)


def _load_finalization_summaries(
    output: Path,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    matrix = pd.read_csv(output / "run_matrix.csv")
    expected = int(protocol["expected_run_count"])
    summaries: list[dict[str, Any]] = []
    missing = 0
    failed = 0
    stale = 0
    for row in matrix.itertuples(index=False):
        path = output / "runs" / str(row.run_id) / "run_summary.json"
        if not path.exists():
            missing += 1
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            failed += 1
        if summary.get("specification_hash") != row.specification_hash:
            stale += 1
        summaries.append(summary)
    known_ids = set(matrix["run_id"].astype(str))
    unexpected = sum(
        path.parent.name not in known_ids
        for path in (output / "runs").glob("*/run_summary.json")
    )
    stale += unexpected
    completed = sum(summary.get("status") == "complete" for summary in summaries)
    audit = {
        "expected_run_count": expected,
        "matrix_run_count": int(len(matrix)),
        "completed_run_count": int(completed),
        "failed_run_count": int(failed),
        "missing_run_count": int(missing),
        "stale_or_mismatched_run_count": int(stale),
        "unique_specification_hash_count": int(
            len({summary.get("specification_hash") for summary in summaries})
        ),
    }
    if audit != {
        "expected_run_count": 140,
        "matrix_run_count": 140,
        "completed_run_count": 140,
        "failed_run_count": 0,
        "missing_run_count": 0,
        "stale_or_mismatched_run_count": 0,
        "unique_specification_hash_count": 140,
    }:
        raise RuntimeError(f"Cannot finalize incomplete or stale result set: {audit}")
    return summaries, audit


def finalize_existing_results(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Finalize completed artifacts without invoking any model or selector fit."""
    config = load_config(config_path)
    output = _repo_path(output_dir or config["output_dir"])
    protocol_path = output / "protocol_manifest.json"
    protocol_sha_before = _sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != config["experiment_id"]:
        raise RuntimeError("Protocol experiment_id does not match finalization config")
    summaries, run_audit = _load_finalization_summaries(output, protocol)
    if any(summary.get("outer_participant_overlap") for summary in summaries):
        raise RuntimeError("Outer participant overlap is non-zero in completed results")

    cohort_audit = _audit_evaluation_cohorts(output, summaries)
    cohort_audit.to_csv(output / "evaluation_cohort_audit.csv", index=False, lineterminator="\n")
    _aggregate(output, summaries)
    frame = _summaries_frame(summaries)
    aggregate_summary = _metric_summary(
        frame,
        group_columns=["metric", "task_type", "feature_regime", "aggregation"],
    )
    aggregate_summary.to_csv(
        output / "aggregate_summary.csv", index=False, lineterminator="\n"
    )

    pm_macro_by_fold = (
        frame.groupby(
            ["task_type", "feature_regime", "aggregation", "fold"],
            as_index=False,
            sort=True,
        )[[*CLASSIFICATION_METRICS, *REGRESSION_METRICS]]
        .mean()
    )
    pm_counts = frame.groupby(
        ["task_type", "feature_regime", "aggregation", "fold"], sort=True
    )["metric"].nunique()
    if not (pm_counts == 7).all():
        raise RuntimeError("PM-macro aggregation requires all seven PM in every fold")
    pm_macro_by_fold.to_csv(
        output / "pm_macro_by_fold.csv", index=False, lineterminator="\n"
    )
    pm_macro_summary = _metric_summary(
        pm_macro_by_fold,
        group_columns=["task_type", "feature_regime", "aggregation"],
    )
    pm_macro_summary.to_csv(
        output / "pm_macro_summary.csv", index=False, lineterminator="\n"
    )
    pm_macro_comparison = _pm_macro_comparison(pm_macro_summary)
    pm_macro_comparison.to_csv(
        output / "pm_macro_comparison.csv", index=False, lineterminator="\n"
    )

    computational = _computational_summary(summaries)
    computational.to_csv(
        output / "computational_summary.csv", index=False, lineterminator="\n"
    )
    participant_pm = aggregate_summary.loc[
        aggregate_summary["aggregation"].eq("participant_macro")
    ]
    participant_macro = pm_macro_summary.loc[
        pm_macro_summary["aggregation"].eq("participant_macro")
    ]
    classification_columns = [
        "metric", "feature_regime",
        "macro_f1_mean", "macro_f1_sample_sd",
        "balanced_accuracy_mean", "balanced_accuracy_sample_sd",
        "accuracy_mean", "accuracy_sample_sd",
    ]
    regression_columns = [
        "metric", "feature_regime",
        "mae_mean", "mae_sample_sd", "rmse_mean", "rmse_sample_sd",
        "r2_mean", "r2_sample_sd", "pearson_mean", "pearson_sample_sd",
        "spearman_mean", "spearman_sample_sd",
    ]
    participant_pm_classification = participant_pm.loc[
        participant_pm["task_type"].eq("classification"), classification_columns
    ]
    participant_pm_regression = participant_pm.loc[
        participant_pm["task_type"].eq("regression"), regression_columns
    ]
    pm_macro_classification = participant_macro.loc[
        participant_macro["task_type"].eq("classification"),
        [column for column in classification_columns if column != "metric"],
    ]
    pm_macro_regression = participant_macro.loc[
        participant_macro["task_type"].eq("regression"),
        [column for column in regression_columns if column != "metric"],
    ]
    stability_overview = pd.read_csv(output / "feature_stability_overview.csv")
    report = "\n".join([
        "# LightGBM feature-selection final report",
        "",
        f"Protocol hash: `{protocol['protocol_hash']}`. Execution: 140/140 complete.",
        "The immutable protocol manifest remains the preregistration snapshot; "
        "completion is recorded separately in execution_manifest.json.",
        "",
        "Outer folds are participant-disjoint. Q3 thresholds, imputation, correlation "
        "removal and feature selection use outer-train only. All 70 paired all/top-50 "
        "evaluation cohorts have identical sample IDs and targets.",
        "",
        f"LightGBM `{summaries[0]['lightgbm_version']}` uses the fixed parameters from "
        "the protocol without tuning.",
        "",
        "## Participant-macro PM-macro classification",
        "",
        pm_macro_classification.to_markdown(index=False),
        "",
        "## Participant-macro PM-macro regression",
        "",
        pm_macro_regression.to_markdown(index=False),
        "",
        "## PM-macro all-features versus top-50",
        "",
        pm_macro_comparison.to_markdown(index=False),
        "",
        "Classification deltas are selected minus all (positive is better). Regression "
        "deltas are also selected minus all: positive MAE/RMSE is worse, while positive "
        "R²/Pearson/Spearman is better.",
        "",
        "## Participant-macro classification by PM",
        "",
        participant_pm_classification.to_markdown(index=False),
        "",
        "## Participant-macro regression by PM",
        "",
        participant_pm_regression.to_markdown(index=False),
        "",
        "## Computational effect",
        "",
        computational.to_markdown(index=False),
        "",
        "`training_seconds` measures downstream LightGBM only; FeatureSelector fit time "
        "was not recorded separately and is not included.",
        "",
        "## Feature-selection stability",
        "",
        stability_overview.to_markdown(index=False),
        "",
        "Selection frequency, >=4/5 and 5/5 stability flags, EEG/POW composition and "
        "pairwise fold Jaccard are stored in the accompanying CSV tables.",
        "",
        "Participant-macro R² values are retained, but can be unstable for participants "
        "with very low within-participant PM variance and are not a primary comparison metric.",
        "Warnings about single-label participant subsets arise when a held-out participant "
        "contains only one Q3 class; no metrics were recomputed and no warning was suppressed.",
        "",
        "## Conclusion",
        "",
        "Automatic fold-local top-50 selection sharply reduces dimensionality and downstream "
        "model cost, but does not improve quality on average. This negative result is not a "
        "basis for changing the prespecified top_k.",
        "",
    ])
    report_path = output / "final_report.md"
    report_path.write_text(report, encoding="utf-8")

    protocol_sha_after = _sha256_file(protocol_path)
    if protocol_sha_after != protocol_sha_before:
        raise RuntimeError("Immutable protocol manifest changed during finalization")
    table_names = [
        "aggregate_metrics.csv", "all_vs_selected_comparison.csv",
        "aggregate_summary.csv", "pm_macro_by_fold.csv", "pm_macro_summary.csv",
        "pm_macro_comparison.csv",
        "computational_summary.csv", "evaluation_cohort_audit.csv",
        "feature_stability_summary.csv", "feature_stability_overview.csv",
        "feature_group_composition.csv", "feature_jaccard_similarity.csv",
        "final_report.md",
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol_hash": protocol["protocol_hash"],
        "protocol_manifest_sha256": protocol_sha_after,
        "protocol_manifest_immutable": True,
        "protocol_analysis_role_snapshot": protocol.get("analysis_role"),
        "execution_status": "complete",
        **run_audit,
        "lightgbm_version": sorted(
            {str(summary["lightgbm_version"]) for summary in summaries}
        ),
        "execution_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "targets": list(config["targets"]),
        "tasks": list(config["tasks"]),
        "feature_regimes": list(config["feature_regimes"]),
        "folds": list(map(int, config["evaluation"]["folds"])),
        "outer_participant_overlap_zero_all_runs": True,
        "evaluation_cohort_match_all_pairs": bool(
            cohort_audit[[
                "sample_ids_match", "targets_match", "train_test_counts_match",
                "q3_thresholds_match",
            ]].to_numpy(dtype=bool).all()
        ),
        "paired_cohort_count": int(len(cohort_audit)),
        "finalization_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "finalization_source": "existing_run_summaries_predictions_and_selector_manifests_only",
        "model_or_selector_fit_invoked": False,
        "table_sha256": {
            name: _sha256_file(output / name) for name in table_names
        },
    }
    _write_json(output / "execution_manifest.json", manifest)
    return manifest


def run_experiment(
    config_path: str | Path,
    *,
    smoke: bool,
    resume: bool = True,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    manifest = write_plan(config_path)
    if not manifest["lightgbm"]["available"]:
        raise ModuleNotFoundError(
            "LightGBM smoke cannot run because the optional dependency is absent. "
            "Install it with: conda run -n eeg_benchmark python -m pip install lightgbm"
        )
    base = prepare_base_protocol(_repo_path(config["base_protocol_config"]))
    specs = build_run_matrix(config)
    if smoke:
        smoke_config = config["smoke"]
        specs = [
            spec for spec in specs
            if spec.metric in set(smoke_config["targets"])
            and spec.fold in set(map(int, smoke_config["folds"]))
        ]
    protocol_hash = str(manifest["protocol_hash"])
    summaries = [
        execute_run(
            config_path,
            spec,
            base=base,
            resume=resume,
            protocol_hash=protocol_hash,
        )
        for spec in specs
    ]
    _aggregate(_repo_path(config["output_dir"]), summaries)
    return summaries
