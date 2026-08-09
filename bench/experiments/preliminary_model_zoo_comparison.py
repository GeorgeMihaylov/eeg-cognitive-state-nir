"""Planning and aggregation for the preliminary one-fold model-zoo comparison."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from bench.features.cogstate_feature_cache import load_feature_cache, load_feature_profile
from model_zoo.factory import SEQUENCE_MODEL_NAMES, TORCH_MODEL_NAMES, model_requires_sequences
from model_zoo.ML.sklearn_models import (
    CLASSIFICATION_MODEL_NAMES,
    REGRESSION_MODEL_NAMES,
    SKLEARN_MODEL_NAMES,
)


COMPARISON_SCHEMA_VERSION = "preliminary-model-zoo-comparison-v1"
PM_NAMES = (
    "attention", "engagement", "excitement", "stress", "relaxation",
    "interest", "focus",
)
RAW_MODEL_NAMES = frozenset({"torch_eegnet", "torch_shallow_convnet"})


def factory_model_names() -> tuple[str, ...]:
    """Enumerate the current factory rather than a documentation list."""
    return tuple(sorted(SKLEARN_MODEL_NAMES | TORCH_MODEL_NAMES))


def model_input_family(model_id: str) -> str:
    normalized = str(model_id).strip().lower()
    if normalized not in factory_model_names():
        raise ValueError(f"Unknown factory model: {model_id!r}")
    if normalized in RAW_MODEL_NAMES:
        return "raw"
    if model_requires_sequences(normalized):
        return "sequence"
    return "features"


def compatibility_matrix() -> pd.DataFrame:
    rows = []
    for model_id in factory_model_names():
        family = model_input_family(model_id)
        rows.append(
            {
                "model_id": model_id,
                "classification_supported": model_id in (
                    CLASSIFICATION_MODEL_NAMES | TORCH_MODEL_NAMES
                ),
                "regression_supported": model_id in (
                    REGRESSION_MODEL_NAMES | {"torch_mlp", "torch_shallow_convnet"}
                ),
                "input_family": family,
                "required_input_shape": {
                    "raw": "[batch,1,14,2560]",
                    "sequence": "[batch,sequence_length,371]",
                    "features": "[batch,371]",
                }[family],
                "requires_raw_eeg": family == "raw",
                "requires_sequence": family == "sequence",
                "requires_features": family in {"sequence", "features"},
                "cuda_capable": model_id in TORCH_MODEL_NAMES,
                "checkpoint_save_load": model_id in TORCH_MODEL_NAMES,
                "default_parameter_source": (
                    "factory builder defaults + preliminary shared Torch budget"
                    if model_id in TORCH_MODEL_NAMES
                    else "sklearn estimator defaults + deterministic seed"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)


def classification_target_ids() -> tuple[str, ...]:
    return tuple(f"pm_{name}_q3_fold_local" for name in PM_NAMES)


def regression_target_ids() -> tuple[str, ...]:
    return tuple(f"pm_{name}_regression" for name in PM_NAMES)


def build_run_status_matrix() -> pd.DataFrame:
    compatibility = compatibility_matrix().set_index("model_id")
    rows: list[dict[str, Any]] = []
    for model_id in factory_model_names():
        for task_type, targets, supported_column in (
            ("classification", classification_target_ids(), "classification_supported"),
            ("regression", regression_target_ids(), "regression_supported"),
        ):
            supported = bool(compatibility.at[model_id, supported_column])
            for target in targets:
                rows.append(
                    {
                        "model": model_id,
                        "target": target,
                        "task_type": task_type,
                        "input_family": compatibility.at[model_id, "input_family"],
                        "outer_fold": 1,
                        "seed": 42,
                        "status": "blocked" if supported else "unsupported",
                        "stage": "awaiting_execution" if supported else "factory_compatibility",
                        "error_type": "",
                        "error_message": "" if supported else f"{task_type} is not exposed by model_zoo.factory",
                    }
                )
    return pd.DataFrame(rows)


def _default_params(model_id: str, task_type: str) -> dict[str, Any]:
    """Deterministic preliminary params without outer-test tuning."""
    if model_id in TORCH_MODEL_NAMES:
        params: dict[str, Any] = {
            "batch_size": 128 if model_input_family(model_id) == "raw" else 256,
            "max_epochs": 5,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "validation_size": 0.15,
            "early_stopping_patience": 2,
            "device": "auto",
            "random_state": 42,
            "standardize": True,
            "num_workers": 0,
        }
        if model_id == "torch_transformer":
            params.update(
                d_model=128, nhead=4, num_layers=2,
                dim_feedforward=256, dropout=0.1, head_type="categorical",
            )
        return params
    params = {}
    if model_id in {"random_forest", "mlp", "svm", "xgboost", "logistic_regression"}:
        params["random_state"] = 42
    if model_id == "random_forest":
        params.update(n_estimators=200, n_jobs=-1)
    elif model_id == "logistic_regression":
        params["max_iter"] = 1000
    elif model_id == "mlp":
        params.update(max_iter=200, early_stopping=True)
    elif model_id == "svm" and task_type == "classification":
        params["probability"] = True
    elif model_id == "xgboost":
        params.update(n_estimators=200, n_jobs=4)
    return params


def benchmark_run_config(
    config: Mapping[str, Any],
    *,
    model_id: str,
    target_id: str,
    output_dir: Path,
    data_root: Path,
) -> dict[str, Any]:
    """Build one standard BenchmarkRunner config with runtime-only path resolution."""
    task_type = "classification" if target_id.endswith("_q3_fold_local") else "regression"
    family = model_input_family(model_id)
    data = config["data"]

    def resolve(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else data_root / path)

    if family == "raw":
        dataset_name = "emotiv_raw_eeg"
        dataset = {
            "data_path": resolve(data["raw_manifest"]),
            "cache_path_root": str(data_root),
            "target_data_path": resolve(data["processed_targets"]),
            "target_id": target_id,
            "dataset_mode": "raw_deduplicated_logical_records",
            "logical_recording_map_path": resolve(data["logical_recording_map"]),
            "raw_preprocessing": dict(config["raw_preprocessing"]),
        }
    else:
        dataset_name = "cogstate_features"
        dataset = {
            "data_path": str(Path(config["output_dir"]).resolve()),
            "target_data_path": resolve(data["processed_targets"]),
            "target_id": target_id,
            "sampling_rate": 256,
        }
    params = _default_params(model_id, task_type)
    params.update(dict(config.get("model_params", {}).get(model_id, {})))
    result: dict[str, Any] = {
        "output_dir": str(output_dir / "runs" / model_id / target_id),
        "result_status": "preliminary",
        "datasets": {dataset_name: dataset},
        "tasks": [target_id],
        "task_config": {"target_id": target_id, "random_state": 42},
        "models": {model_id: {"type": model_id, "task_type": task_type, "params": params}},
        "evaluation": {
            "protocol": "group_kfold_subject", "n_splits": 5,
            "group_column": "subject_id", "precomputed_fold_column": "outer_fold",
            "folds": [1], "random_state": 42,
        },
        "validation": {
            "strategy": "group_record", "group_column": "record_group_id",
            "validation_size": 0.15, "random_state": 42,
        },
        "run_within_subject": False,
        "run_loso": False,
    }
    if model_id in SEQUENCE_MODEL_NAMES:
        result["sequence"] = dict(config["sequence"])
    return result


def latency_percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Latency values must be non-empty and finite")
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def measure_prediction_latency(
    model: Any, sample: np.ndarray, *, warmup: int = 20, repetitions: int = 100
) -> dict[str, float]:
    batch = np.ascontiguousarray(sample[None], dtype=np.float32)
    for _ in range(warmup):
        model.predict(batch)
    timings = []
    for _ in range(repetitions):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        model.predict(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return latency_percentiles(timings)


def aggregate_model_summary(rows: pd.DataFrame, *, task_type: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    output: list[dict[str, Any]] = []
    for model, group in rows.groupby("model", sort=True):
        completed = group.loc[group["status"].eq("completed")]
        row: dict[str, Any] = {
            "model": model,
            "input_family": group["input_family"].iloc[0],
            "completed_targets": int(group["status"].eq("completed").sum()),
            "failed_targets": int(group["status"].eq("failed").sum()),
            "unsupported_targets": int(group["status"].isin(["unsupported", "blocked"]).sum()),
        }
        if task_type == "classification" and not completed.empty:
            macro = pd.to_numeric(completed["macro_f1"], errors="coerce")
            row.update(
                mean_macro_f1=float(macro.mean()), median_macro_f1=float(macro.median()),
                min_macro_f1=float(macro.min()), max_macro_f1=float(macro.max()),
                mean_balanced_accuracy=float(pd.to_numeric(completed["balanced_accuracy"], errors="coerce").mean()),
            )
        elif task_type == "regression" and not completed.empty:
            for metric in ("mae", "rmse", "r2", "pearson", "spearman"):
                row[f"mean_{metric}"] = float(pd.to_numeric(completed[metric], errors="coerce").mean())
        for metric in ("model_latency_p95_ms", "end_to_end_latency_p95_ms", "train_time_s"):
            if metric in completed:
                values = pd.to_numeric(completed[metric], errors="coerce")
                row[f"mean_{metric}"] = float(values.mean())
                if metric != "train_time_s":
                    row[f"max_{metric}"] = float(values.max())
        for metric in ("peak_ram_mb", "peak_vram_mb", "parameter_count", "checkpoint_size_mb"):
            if metric in completed:
                row[metric] = float(pd.to_numeric(completed[metric], errors="coerce").max())
        output.append(row)
    return pd.DataFrame(output)


def build_streaming_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    ranking = summary.reindex(columns=[
        "model", "input_family", "mean_macro_f1", "mean_balanced_accuracy",
        "mean_model_latency_p95_ms", "mean_end_to_end_latency_p95_ms",
        "peak_ram_mb", "peak_vram_mb", "checkpoint_size_mb",
    ]).rename(columns={
        "mean_model_latency_p95_ms": "model_p95_latency_ms",
        "mean_end_to_end_latency_p95_ms": "end_to_end_p95_latency_ms",
    })
    for output, source, ascending in (
        ("rank_f1", "mean_macro_f1", False),
        ("rank_model_latency", "model_p95_latency_ms", True),
        ("rank_end_to_end_latency", "end_to_end_p95_latency_ms", True),
        ("rank_ram", "peak_ram_mb", True),
        ("rank_vram", "peak_vram_mb", True),
        ("rank_model_size", "checkpoint_size_mb", True),
    ):
        ranking[output] = pd.to_numeric(ranking[source], errors="coerce").rank(
            method="min", ascending=ascending, na_option="bottom"
        )
    return ranking


def validate_shallow_reuse(
    source_dir: str | Path, *, raw_preprocessing_hash: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(source_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checks = {
        "result_status": manifest.get("result_status") == "preliminary",
        "outer_fold": manifest.get("evaluation", {}).get("folds") == [1],
        "seed": manifest.get("evaluation", {}).get("random_state") == 42,
        "preprocessing_hash": manifest.get("composite_audit", {}).get("preprocessing_hash") == raw_preprocessing_hash,
        "model": manifest.get("model", {}).get("type") == "torch_shallow_convnet",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Existing ShallowConvNet handoff is incompatible: {failed}")
    summary = pd.read_csv(root / "summary.csv")
    latency = pd.read_csv(root / "latency.csv")
    expected = set(classification_target_ids()) | set(regression_target_ids())
    if set(summary["target_id"]) != expected or not summary["status"].eq("completed").all():
        raise ValueError("Existing ShallowConvNet handoff lacks 14 completed targets")
    for column in ("subject_overlap", "inner_group_overlap"):
        if summary[column].fillna(0).astype(int).ne(0).any():
            raise ValueError(f"Existing ShallowConvNet handoff fails leakage check: {column}")
    return summary, latency


def import_reusable_shallow_results(
    *, output_dir: str | Path, source_dir: str | Path, raw_preprocessing_hash: str
) -> pd.DataFrame:
    """Import a compatible completed handoff without retraining it."""
    output = Path(output_dir)
    source = Path(source_dir)
    summary, latency = validate_shallow_reuse(
        source, raw_preprocessing_hash=raw_preprocessing_hash
    )
    latency_lookup = {
        (str(row.target_id), str(row.latency_mode)): row
        for row in latency.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for saved in summary.to_dict("records"):
        target = str(saved["target_id"])
        model_latency = latency_lookup.get((target, "model_only"))
        end_to_end = latency_lookup.get((target, "channel_normalization_plus_model"))
        checkpoint = source / str(saved.get("checkpoint", ""))
        rows.append({
            "model": "torch_shallow_convnet", "target": target,
            "outer_fold": 1, "seed": 42, "task_type": str(saved["task_type"]),
            "input_family": "raw", "accuracy": saved.get("accuracy"),
            "balanced_accuracy": saved.get("balanced_accuracy"),
            "macro_f1": saved.get("macro_f1"), "weighted_f1": saved.get("weighted_f1"),
            "mae": saved.get("mae"), "rmse": saved.get("rmse"),
            "r2": saved.get("r2"), "pearson": saved.get("pearson"),
            "spearman": saved.get("spearman"),
            "train_time_s": saved.get("training_time_seconds"),
            "model_latency_p50_ms": getattr(model_latency, "p50_ms", np.nan),
            "model_latency_p95_ms": getattr(model_latency, "p95_ms", np.nan),
            "model_latency_p99_ms": getattr(model_latency, "p99_ms", np.nan),
            "end_to_end_latency_p50_ms": getattr(end_to_end, "p50_ms", np.nan),
            "end_to_end_latency_p95_ms": getattr(end_to_end, "p95_ms", np.nan),
            "end_to_end_latency_p99_ms": getattr(end_to_end, "p99_ms", np.nan),
            "feature_extraction_p95_ms": np.nan, "peak_ram_mb": np.nan,
            "peak_vram_mb": float(saved.get("peak_training_gpu_memory_bytes", np.nan)) / 2**20,
            "parameter_count": saved.get("trainable_parameter_count"),
            "checkpoint_size_mb": float(checkpoint.stat().st_size) / 2**20 if checkpoint.is_file() else np.nan,
            "checkpoint_reload_verified": bool(
                np.isfinite(float(saved.get("checkpoint_reload_absolute_difference", np.nan)))
            ),
            "reused_existing_run": True, "status": "completed",
            "notes": "Reused compatible preliminary_streaming_handoff_shallow_fold1",
        })
    imported = pd.DataFrame(rows)
    status_path = output / "run_status.csv"
    status = pd.read_csv(status_path) if status_path.is_file() else build_run_status_matrix()
    for column in ("status", "stage", "error_type", "error_message"):
        status[column] = status[column].fillna("").astype(str)
    keys = set(zip(imported["model"], imported["target"]))
    mask = pd.Series(
        [(model, target) in keys for model, target in zip(status.model, status.target)],
        index=status.index,
    )
    status.loc[mask, "status"] = "completed"
    status.loc[mask, "stage"] = "reused_existing_run"
    status.loc[mask, "error_type"] = ""
    status.loc[mask, "error_message"] = ""
    status.to_csv(status_path, index=False)
    comparison_rows = pd.concat(
        [imported, status.loc[~mask].assign(reused_existing_run=False)],
        ignore_index=True, sort=False,
    )
    write_comparison_tables(output, comparison_rows)
    latency.to_csv(output / "latency.csv", index=False)
    pd.DataFrame([{
        "model": "torch_shallow_convnet", "target": row["target"],
        "training_wall_time_s": row["train_time_s"],
        "peak_ram_mb": row["peak_ram_mb"], "peak_vram_mb": row["peak_vram_mb"],
        "device": summary.loc[summary.target_id.eq(row["target"]), "device"].iloc[0],
    } for row in rows]).to_csv(output / "resource_usage.csv", index=False)
    cohort_source = source / "cohort_counts.csv"
    if cohort_source.is_file():
        pd.read_csv(cohort_source).to_csv(output / "cohort_counts.csv", index=False)
    return imported


def verify_feature_cache_for_comparison(cache_dir: str | Path, profile_path: str | Path) -> dict[str, Any]:
    matrix, index, names, manifest = load_feature_cache(cache_dir)
    _, pipeline = load_feature_profile(profile_path)
    identity = manifest["identity"]
    checks = {
        "rows": len(index) == 34354,
        "feature_dimension": matrix.shape[1] == len(names) == 371,
        "sample_id_unique": index["sample_id"].is_unique,
        "finite": bool(np.isfinite(matrix).all()),
        "feature_hash": identity["feature_hash"] == pipeline.feature_hash(14, pipeline.config.channel_names),
        "no_target_columns": not any(str(column).startswith("target_") or str(column) == "label_q5" for column in index.columns),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Feature cache comparison gate failed: {failed}")
    return {"checks": checks, "identity": identity}


def write_comparison_tables(output_dir: str | Path, rows: pd.DataFrame) -> None:
    output = Path(output_dir)
    q3 = rows.loc[rows.task_type.eq("classification")].copy()
    regression = rows.loc[rows.task_type.eq("regression")].copy()
    q3.to_csv(output / "model_comparison_q3.csv", index=False)
    regression.to_csv(output / "model_comparison_regression.csv", index=False)
    q3_summary = aggregate_model_summary(q3, task_type="classification")
    regression_summary = aggregate_model_summary(regression, task_type="regression")
    q3_summary.to_csv(output / "model_summary_q3.csv", index=False)
    regression_summary.to_csv(output / "model_summary_regression.csv", index=False)
    build_streaming_ranking(q3_summary).to_csv(output / "streaming_model_ranking.csv", index=False)


def write_plan(*, output_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    compatibility = compatibility_matrix()
    status = build_run_status_matrix()
    compatibility.to_csv(output / "model_compatibility.csv", index=False)
    status.to_csv(output / "run_status.csv", index=False)
    raw = ", ".join(compatibility.loc[compatibility.input_family.eq("raw"), "model_id"])
    sequence = ", ".join(compatibility.loc[compatibility.input_family.eq("sequence"), "model_id"])
    features = ", ".join(compatibility.loc[compatibility.input_family.eq("features"), "model_id"])
    (output / "README.md").write_text(
        "# PRELIMINARY model-zoo comparison — ONE OUTER FOLD ONLY\n\n"
        "Engineering handoff for outer fold 1 and seed 42; not a final five-fold result.\n\n"
        f"- Raw (`[14,2560] → adapter`): {raw}.\n"
        f"- Sequence (`raw → FeaturePipeline → [sequence,371]`): {sequence}.\n"
        f"- Features (`raw → FeaturePipeline → [371]`): {features}.\n\n"
        "Feature models require separate cached-vector/model-only and online "
        "FeaturePipeline-plus-model latency. No opaque weighted score is used.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "result_status": "preliminary",
        "scope": "one_outer_fold_only",
        "outer_fold": 1,
        "seed": 42,
        "factory_models": factory_model_names(),
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cuda": torch.version.cuda,
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "planned_runs": int(status.status.ne("unsupported").sum()),
        "unsupported_runs": int(status.status.eq("unsupported").sum()),
        "config": dict(config),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
