"""Fold-1 PM handoff using the canonical raw runner and ShallowConvNet."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from bench.bench_runner import BenchmarkRunner
from bench.datasets.raw_eeg_window_dataset import RawEEGWindowDataset
from bench.tasks.tasks_registry import get_task
from bench.validation.cross_val import CrossValidator
from model_zoo import build_model


HANDOFF_SCHEMA_VERSION = "preliminary-streaming-handoff-v1"
DATASET_NAME = "emotiv_raw_eeg"
MODEL_RUN_NAME = "shallow"
CHANNEL_ORDER = (
    "EEG.AF3", "EEG.F7", "EEG.F3", "EEG.FC5", "EEG.T7", "EEG.P7",
    "EEG.O1", "EEG.O2", "EEG.P8", "EEG.T8", "EEG.FC6", "EEG.F4",
    "EEG.F8", "EEG.AF4",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _target_ids(metrics: Iterable[str]) -> list[str]:
    names = [str(metric) for metric in metrics]
    return [
        *(f"pm_{metric}_regression" for metric in names),
        *(f"pm_{metric}_q3_fold_local" for metric in names),
    ]


def _target_parts(target_id: str) -> tuple[str, str]:
    if target_id.endswith("_regression"):
        return target_id.removeprefix("pm_").removesuffix("_regression"), "regression"
    if target_id.endswith("_q3_fold_local"):
        return (
            target_id.removeprefix("pm_").removesuffix("_q3_fold_local"),
            "classification",
        )
    raise ValueError(f"Unsupported preliminary target_id: {target_id}")


def _target_slug(target_id: str) -> str:
    metric, task_type = _target_parts(target_id)
    return f"{metric}_{'reg' if task_type == 'regression' else 'q3'}"


def _dataset_config(
    config: Mapping[str, Any], data_root: Path, target_id: str
) -> dict[str, Any]:
    data = config["data"]
    return {
        "data_path": str(_resolve(data_root, data["composite_manifest"])),
        "cache_path_root": str(data_root),
        "target_data_path": str(_resolve(data_root, data["processed_targets"])),
        "target_id": target_id,
        "dataset_mode": data["dataset_mode"],
        "logical_recording_map_path": str(
            _resolve(data_root, data["logical_recording_map"])
        ),
        "raw_preprocessing": dict(config["raw_preprocessing"]),
    }


def load_target_data(
    config: Mapping[str, Any], data_root: Path, target_id: str
):
    return RawEEGWindowDataset(_dataset_config(config, data_root, target_id)).load()


def fold_one_split(config: Mapping[str, Any], data: Any, target_id: str):
    task = get_task(
        target_id,
        data,
        {"target_id": target_id, "random_state": int(config["evaluation"]["random_state"])},
    )
    splits = CrossValidator(task).run_group_kfold(
        group_column=str(config["evaluation"]["group_column"]),
        n_splits=int(config["evaluation"]["n_splits"]),
        random_state=int(config["evaluation"]["random_state"]),
        precomputed_fold_column=str(config["evaluation"]["precomputed_fold_column"]),
    )
    return splits["fold_01"]


def audit_composite(
    config: Mapping[str, Any], data_root: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    data_config = config["data"]
    manifest_path = _resolve(data_root, data_config["composite_manifest"])
    logical_path = _resolve(data_root, data_config["logical_recording_map"])
    manifest = pd.read_parquet(manifest_path)
    logical = pd.read_parquet(logical_path)
    selected = dict(zip(
        logical["record_group_id"].astype(str),
        logical["selected_record_id"].astype(str),
    ))
    selected_mask = manifest["record_id"].astype(str).eq(
        manifest["record_group_id"].astype(str).map(selected)
    )
    deduplicated = manifest.loc[selected_mask].copy()
    if not manifest["sample_id"].is_unique or not deduplicated["sample_id"].is_unique:
        raise ValueError("Composite manifest contains duplicate sample_id")
    if not deduplicated.groupby("subject_id")["outer_fold"].nunique().eq(1).all():
        raise ValueError("Composite manifest changes fixed subject folds")
    hashes = sorted(deduplicated["preprocessing_hash"].dropna().astype(str).unique())
    if len(hashes) != 1:
        raise ValueError(f"Composite preprocessing identity is ambiguous: {hashes}")

    summary_path = _resolve(
        data_root,
        "data/interim/raw_eeg_window_index_w10_pm_union_composite_v1_stats.json",
    )
    plan_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cohort_rows: list[dict[str, Any]] = []
    for target_id in _target_ids(config["targets"]):
        metric, task_type = _target_parts(target_id)
        data = load_target_data(config, data_root, target_id)
        split = fold_one_split(config, data, target_id)
        row: dict[str, Any] = {
            "target_id": target_id,
            "source_pm": f"target_{metric}",
            "task_type": task_type,
            "candidate_pre_qc": int(
                plan_summary["target_candidate_deduplicated_rows"][f"target_{metric}"]
            ),
            "status_ok": int(data.n_samples),
            "subjects": int(data.n_subjects),
            "logical_records": int(len(np.unique(data.row_metadata["record_group_id"]))),
            "fold1_train": int(len(split.y_train)),
            "fold1_test": int(len(split.y_test)),
            "fold1_train_subjects": int(len(np.unique(split.subject_train))),
            "fold1_test_subjects": int(len(np.unique(split.subject_test))),
            "subject_overlap": int(
                len(set(split.subject_train.astype(str)) & set(split.subject_test.astype(str)))
            ),
        }
        if task_type == "classification":
            transform = split.metadata["target_transform"]
            row.update({
                "q3_boundaries": json.dumps(transform["boundaries"]),
                "q3_fit_sample_count": int(transform["fit_sample_count"]),
                "q3_train_class_counts": json.dumps(_counts(split.y_train)),
                "q3_test_class_counts": json.dumps(_counts(split.y_test)),
                "target_transform_hash": transform["transform_hash"],
            })
        cohort_rows.append(row)

    fold_subject_counts = {
        str(int(fold)): int(count)
        for fold, count in (
            deduplicated.drop_duplicates("subject_id")["outer_fold"]
            .value_counts().sort_index().items()
        )
    }
    audit = {
        "manifest_path": str(data_config["composite_manifest"]),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_rows": int(len(manifest)),
        "manifest_unique_sample_id": True,
        "deduplicated_candidate": int(len(deduplicated)),
        "deduplicated_status_ok": int(deduplicated["status"].eq("ok").sum()),
        "deduplicated_rejected": int(deduplicated["status"].ne("ok").sum()),
        "old_status_ok": int(
            (deduplicated["status"].eq("ok") & deduplicated["cache_file"].astype(str)
             .str.contains("raw_eeg_cache_w10_v3", regex=False)).sum()
        ),
        "delta_status_ok": int(
            (deduplicated["status"].eq("ok") & deduplicated["cache_file"].astype(str)
             .str.contains("pm_union_delta", regex=False)).sum()
        ),
        "subject_counts_by_fold": fold_subject_counts,
        "preprocessing_hash": hashes[0],
        "selected_record_mapping_hash": plan_summary["selected_record_mapping_hash"],
        "candidate_manifest_hash": plan_summary["candidate_manifest_hash"],
    }
    if fold_subject_counts != {"1": 11, "2": 11, "3": 11, "4": 10, "5": 11}:
        raise ValueError(f"Unexpected fixed fold subject counts: {fold_subject_counts}")
    return audit, pd.DataFrame(cohort_rows)


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts)}


def _benchmark_config(
    config: Mapping[str, Any], data_root: Path, output_dir: Path, target_id: str
) -> dict[str, Any]:
    _, task_type = _target_parts(target_id)
    return {
        "output_dir": str(output_dir / "runs" / _target_slug(target_id)),
        "result_status": "preliminary",
        "raw_preprocessing": dict(config["raw_preprocessing"]),
        "datasets": {DATASET_NAME: _dataset_config(config, data_root, target_id)},
        "tasks": [target_id],
        "task_config": {"target_id": target_id, "random_state": 42},
        "models": {
            MODEL_RUN_NAME: {
                "type": config["model"]["type"],
                "task_type": task_type,
                "params": dict(config["model"]["params"]),
            }
        },
        "validation": dict(config["validation"]),
        "evaluation": dict(config["evaluation"]),
        "run_within_subject": False,
        "run_loso": False,
    }


def _relative(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_single_window_latency(
    adapter: Any,
    window: np.ndarray,
    *,
    warmup: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    raw = np.ascontiguousarray(window[None], dtype=np.float32)
    normalized = adapter.transform_features_for_audit(raw)
    device = adapter.device_
    tensor = torch.from_numpy(normalized).to(device)
    adapter.model.eval()

    def model_only() -> None:
        with torch.no_grad():
            adapter.model(tensor)

    def normalization_and_model() -> None:
        transformed = adapter.transform_features_for_audit(raw)
        batch = torch.from_numpy(transformed).to(device)
        with torch.no_grad():
            outputs = adapter.model(batch)
            if adapter.task_type == "classification":
                adapter.objective_handler.decode(outputs)

    rows = []
    for name, operation in (
        ("model_only", model_only),
        ("channel_normalization_plus_model", normalization_and_model),
    ):
        for _ in range(warmup):
            operation()
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        values = []
        for _ in range(repetitions):
            _synchronize(device)
            started = time.perf_counter_ns()
            operation()
            _synchronize(device)
            values.append((time.perf_counter_ns() - started) / 1_000_000.0)
        array = np.asarray(values, dtype=float)
        rows.append({
            "latency_mode": name,
            "iterations": repetitions,
            "warmup_iterations": warmup,
            "mean_ms": float(np.mean(array)),
            "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)),
            "p99_ms": float(np.percentile(array, 99)),
            "device": str(device),
            "dtype": "float32",
            "batch_size": 1,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
        })
    return rows


def _run_target(
    config: Mapping[str, Any], data_root: Path, output_dir: Path, target_id: str,
    *, reuse_existing: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric, task_type = _target_parts(target_id)
    benchmark_config = _benchmark_config(config, data_root, output_dir, target_id)
    fold = (
        _load_existing_fold(output_dir, target_id, benchmark_config)
        if reuse_existing else None
    )
    if fold is None:
        runner = BenchmarkRunner(benchmark_config)
        started = time.perf_counter()
        runner.run()
        wall_time = time.perf_counter() - started
        fold = runner.results[DATASET_NAME]["models"][target_id][
            MODEL_RUN_NAME
        ]["group_kfold_subject"]["folds"]["fold_01"]
    else:
        wall_time = float(fold["training_time"])
    artifacts = fold["artifacts"]
    checkpoint = Path(artifacts["model"])
    data = load_target_data(config, data_root, target_id)
    split = fold_one_split(config, data, target_id)
    params = dict(config["model"]["params"])
    params["sampling_rate"] = float(data.sampling_rate)
    params["channel_names"] = list(data.feature_names)
    adapter = build_model(
        config["model"]["type"],
        task_type,
        tuple(split.X_train.shape[1:]),
        1 if task_type == "regression" else 3,
        params,
    )
    adapter.load(checkpoint)
    first_window = np.asarray(split.X_test[0], dtype=np.float32)
    reloaded_prediction = np.asarray(adapter.predict(first_window[None])).reshape(-1)[0]
    predictions = pd.read_parquet(artifacts["predictions"])
    if not np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all():
        raise RuntimeError("Saved predictions contain NaN or Inf")
    expected = predictions.loc[
        predictions["sample_id"].astype(str).eq(str(split.sample_id_test[0])),
        "y_pred",
    ]
    if len(expected) != 1:
        raise RuntimeError("Saved predictions do not contain one row for the reload sample")
    reload_difference = abs(float(reloaded_prediction) - float(expected.iloc[0]))
    if not np.isclose(
        float(reloaded_prediction), float(expected.iloc[0]), rtol=1e-3, atol=5e-4
    ):
        raise RuntimeError(
            "Reloaded checkpoint prediction does not match saved output: "
            f"absolute_difference={reload_difference:.9g}"
        )
    if task_type == "classification":
        probability_columns = [f"proba_{index}" for index in range(3)]
        saved_probabilities = predictions[probability_columns].to_numpy(dtype=float)
        if (
            not set(predictions["y_true"].astype(int)).issubset({0, 1, 2})
            or not set(predictions["y_pred"].astype(int)).issubset({0, 1, 2})
            or not np.isfinite(saved_probabilities).all()
            or not np.allclose(saved_probabilities.sum(axis=1), 1.0, atol=1e-5)
        ):
            raise RuntimeError("Saved Q3 labels or probabilities are invalid")
        probabilities = adapter.predict_proba(first_window[None])
        if probabilities.shape != (1, 3) or not np.allclose(probabilities.sum(1), 1.0):
            raise RuntimeError("Reloaded Q3 probabilities are invalid")
        saved_transform = json.loads(Path(artifacts["target_transform"]).read_text())
        current_transform = split.metadata["target_transform"]
        if saved_transform["transform_hash"] != current_transform["transform_hash"]:
            raise RuntimeError("Reloaded fold-local target transform hash differs")

    latency = measure_single_window_latency(
        adapter,
        first_window,
        warmup=int(config["latency"]["warmup_iterations"]),
        repetitions=int(config["latency"]["measured_iterations"]),
    )
    for latency_row in latency:
        latency_row.update({"target_id": target_id, "task_type": task_type})
    metrics = fold["metrics"]
    training = fold["training"]
    validation_split = json.loads(Path(artifacts["validation_split"]).read_text())
    if (
        validation_split.get("inner_group_overlap") != 0
        or validation_split.get("outer_test_group_overlap") != 0
    ):
        raise RuntimeError("Inner validation split has record-group leakage")
    row = {
        "target_id": target_id,
        "source_pm": f"target_{metric}",
        "task_type": task_type,
        "status": "completed",
        "result_status": "preliminary",
        "model": "torch_shallow_convnet",
        "outer_fold": 1,
        "seed": 42,
        "preprocessing_hash": str(data.metadata["preprocessing_hashes"][0]),
        "preprocessing_specification": json.dumps(config["raw_preprocessing"]),
        "training_parameters": json.dumps(config["model"]["params"]),
        "input_shape": json.dumps(training["input_shape"]),
        "sample_rate_hz": 256,
        "window_seconds": 10,
        "train_samples": int(fold["n_train"]),
        "test_samples": int(fold["n_test"]),
        "train_subjects": int(len(np.unique(split.subject_train))),
        "test_subjects": int(len(np.unique(split.subject_test))),
        "train_subject_ids": json.dumps(sorted(map(str, np.unique(split.subject_train)))),
        "test_subject_ids": json.dumps(sorted(map(str, np.unique(split.subject_test)))),
        "subject_overlap": 0,
        "inner_validation_strategy": validation_split["validation_strategy"],
        "inner_validation_group_column": validation_split["validation_group_column"],
        "inner_group_overlap": validation_split["inner_group_overlap"],
        "outer_test_group_overlap": validation_split["outer_test_group_overlap"],
        "training_time_seconds": float(fold["training_time"]),
        "wall_time_seconds": float(wall_time),
        "epochs_trained": int(training["epochs_trained"]),
        "best_epoch": training["best_epoch"],
        "best_validation_loss": training["best_validation_loss"],
        "device": training["device"],
        "device_name": training["device_name"],
        "peak_training_gpu_memory_bytes": training["peak_gpu_memory_bytes"],
        "trainable_parameter_count": training["trainable_parameter_count"],
        "checkpoint_reload_absolute_difference": reload_difference,
        "checkpoint": _relative(checkpoint, output_dir),
        "predictions": _relative(artifacts["predictions"], output_dir),
        "training_log": _relative(artifacts["training_log"], output_dir),
        "validation_split": _relative(artifacts["validation_split"], output_dir),
        "normalization_stats": _relative(artifacts["normalization_stats"], output_dir),
        "error": "",
    }
    if task_type == "regression":
        row.update({
            "output_semantics": "continuous PM estimate",
            "mae": metrics.get("mae_macro", metrics.get("mae")),
            "rmse": metrics.get("rmse_macro", metrics.get("rmse")),
            "r2": metrics.get("r2_macro", metrics.get("r2")),
            "pearson": metrics.get("pearson_macro", metrics.get("pearson")),
            "spearman": metrics.get("spearman_macro", metrics.get("spearman")),
        })
    else:
        row.update({
            "output_semantics": "categorical logits and low/medium/high probabilities",
            "class_mapping": json.dumps({"0": "low", "1": "medium", "2": "high"}),
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "kappa": metrics.get("kappa"),
            "target_transform": _relative(artifacts["target_transform"], output_dir),
            "target_transform_hash": split.metadata["target_transform_hash"],
            "q3_boundaries": json.dumps(split.metadata["target_transform"]["boundaries"]),
        })
    return row, latency


def _load_existing_fold(
    output_dir: Path,
    target_id: str,
    expected_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    result_files = sorted(
        [
            *list((output_dir / "runs" / target_id).glob("benchmark_results_*.json")),
            *list((output_dir / "runs" / _target_slug(target_id)).glob(
                "benchmark_results_*.json"
            )),
        ],
        reverse=True,
    )
    for result_path in result_files:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        timestamp = result_path.stem.removeprefix("benchmark_results_")
        saved_config_path = result_path.parent / timestamp / "config.yaml"
        if not saved_config_path.is_file():
            continue
        saved_config = _read_yaml(saved_config_path)
        identity_keys = (
            "result_status", "raw_preprocessing", "datasets", "tasks",
            "task_config", "validation", "evaluation",
        )
        if any(saved_config.get(key) != expected_config.get(key) for key in identity_keys):
            continue
        saved_models = list(dict(saved_config.get("models", {})).values())
        expected_models = list(dict(expected_config.get("models", {})).values())
        if saved_models != expected_models:
            continue
        try:
            model_results = payload[DATASET_NAME]["models"][target_id]
            model_key = next(
                key for key in (MODEL_RUN_NAME, "torch_shallow_convnet")
                if key in model_results
            )
            fold = model_results[model_key]["group_kfold_subject"]["folds"][
                "fold_01"
            ]
        except (KeyError, StopIteration):
            continue
        checkpoint = Path(fold.get("artifacts", {}).get("model", ""))
        if checkpoint.is_file():
            return fold
    return None


def _environment() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if cuda else None,
        "device_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if cuda else 0
        ),
        "process_id": os.getpid(),
    }


def _write_readme(output_dir: Path, config: Mapping[str, Any]) -> None:
    parameters = config["model"]["params"]
    content = f"""# Preliminary streaming handoff: ShallowConvNet fold 1

Status: **preliminary**. These are single-outer-fold engineering results, not a final scientific benchmark.

## Runtime contract

- External logical window: `[14, 2560]` (`14` channels, `256 Hz`, `10 s`, `float32`).
- Adapter/model input: `[B, 1, 14, 2560]`; streaming uses `B=1`.
- Channel order: `{', '.join(CHANNEL_ORDER)}`.
- Offline preprocessing: resample to 256 Hz; band-pass disabled; notch disabled; rereference none; artifact rejection disabled.
- Runtime normalization: one mean and standard deviation per EEG channel, fitted on inner-train only and stored in both `model.pt` and `normalization_stats.json`.
- Regression output: `[B, 1]`, continuous PM estimate.
- Q3 output: `[B, 3]` categorical logits; `predict_proba` returns low/medium/high probabilities summing to one. Mapping: `0=low`, `1=medium`, `2=high`.
- Training: max epochs `{parameters['max_epochs']}`, batch size `{parameters['batch_size']}`, early stopping patience `{parameters['early_stopping_patience']}`; seed 42; outer fold 1 only.

## Artifacts

- `summary.csv`: run status, training details and quality metrics.
- `latency.csv`: model-only and channel-normalization-plus-model latency for batch size 1.
- `cohort_counts.csv`: post-QC target cohorts and fold-local Q3 definitions.
- `manifest.json`: environment, preprocessing identity and artifact index.
- `runs/<target_id>/.../fold_01/model.pt`: loadable adapter checkpoints.

## Loading a checkpoint

Build the same `torch_shallow_convnet` through `model_zoo.build_model`, using `input_shape=(1, 14, 2560)`, `num_outputs=1` for regression or `3` for Q3, then call `adapter.load(checkpoint_path)`. Pass one external window as `window[np.newaxis, np.newaxis, :, :]`. The checkpoint restores train-only channel normalization automatically.

```python
from pathlib import Path
import numpy as np
import yaml
from model_zoo import build_model

config = yaml.safe_load(Path("experiments/targets/preliminary_streaming_handoff_shallow_fold1.yaml").read_text())
adapter = build_model(
    "torch_shallow_convnet",
    task_type="regression",       # use "classification" for Q3
    input_shape=(1, 14, 2560),
    num_outputs=1,                 # use 3 for Q3
    params=config["model"]["params"],
)
adapter.load("path/to/model.pt")
window = np.asarray(window, dtype=np.float32)  # [14, 2560]
prediction = adapter.predict(window[None, None])
# For Q3: probabilities = adapter.predict_proba(window[None, None])
```

Raw signal filtering is offline cache preprocessing and is not included in streaming latency. The `channel_normalization_plus_model` latency includes fitted channel normalization, host-to-device transfer, model forward and Q3 decoding where applicable.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def run_preliminary_handoff(
    config_path: Path,
    *,
    data_root: Path,
    output_dir: Path | None = None,
    requested_target_ids: Iterable[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    if config.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        raise ValueError("Unsupported preliminary handoff schema")
    output = output_dir or _resolve(config_path.parents[2], config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 14-run preliminary matrix")

    audit, cohorts = audit_composite(config, data_root)
    cohorts.to_csv(output / "cohort_counts.csv", index=False)
    all_targets = _target_ids(config["targets"])
    selected_targets = (
        all_targets if requested_target_ids is None else list(requested_target_ids)
    )
    unknown = sorted(set(selected_targets) - set(all_targets))
    if unknown:
        raise ValueError(f"Unknown requested targets: {unknown}")

    summary_path = output / "summary.csv"
    latency_path = output / "latency.csv"
    existing_rows = _read_csv_records(summary_path) if resume else []
    existing_latency = _read_csv_records(latency_path) if resume else []
    rows_by_target = {str(row["target_id"]): row for row in existing_rows}
    latency_rows = list(existing_latency)

    for target_id in selected_targets:
        current = rows_by_target.get(target_id)
        if resume and current and current.get("status") == "completed":
            checkpoint = output / str(current["checkpoint"])
            if checkpoint.is_file():
                continue
        try:
            row, latency = _run_target(
                config, data_root, output, target_id, reuse_existing=resume
            )
            rows_by_target[target_id] = row
            latency_rows = [
                item for item in latency_rows if str(item.get("target_id")) != target_id
            ] + latency
        except Exception as exc:
            metric, task_type = _target_parts(target_id)
            rows_by_target[target_id] = {
                "target_id": target_id,
                "source_pm": f"target_{metric}",
                "task_type": task_type,
                "status": "failed",
                "result_status": "preliminary",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_outputs(
                output, config_path, config, audit, all_targets,
                rows_by_target, latency_rows,
            )
            if target_id == "pm_attention_regression":
                raise
            continue
        _write_outputs(
            output, config_path, config, audit, all_targets,
            rows_by_target, latency_rows,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _write_outputs(
        output, config_path, config, audit, all_targets,
        rows_by_target, latency_rows,
    )


def _write_outputs(
    output: Path,
    config_path: Path,
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
    all_targets: list[str],
    rows_by_target: Mapping[str, Mapping[str, Any]],
    latency_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    complete_rows = []
    status_counts = {"completed": 0, "failed": 0, "not_started": 0}
    for target_id in all_targets:
        if target_id in rows_by_target:
            row = dict(rows_by_target[target_id])
        else:
            metric, task_type = _target_parts(target_id)
            row = {
                "target_id": target_id,
                "source_pm": f"target_{metric}",
                "task_type": task_type,
                "status": "not_started",
                "result_status": "preliminary",
                "error": "",
            }
        status_counts[str(row["status"])] += 1
        complete_rows.append(row)
    pd.DataFrame(complete_rows).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(output / "latency.csv", index=False)
    _write_readme(output, config)
    manifest = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "preliminary",
        "config_path": str(config_path.relative_to(config_path.parents[2])),
        "config_sha256": _sha256(config_path),
        "environment": _environment(),
        "composite_audit": dict(audit),
        "preprocessing": dict(config["raw_preprocessing"]),
        "model": {
            "type": config["model"]["type"],
            "artifact_alias": MODEL_RUN_NAME,
            "params": dict(config["model"]["params"]),
        },
        "evaluation": dict(config["evaluation"]),
        "inner_validation": dict(config["validation"]),
        "input_contract": {
            "external_shape": [14, 2560],
            "adapter_shape": [1, 1, 14, 2560],
            "channel_order": list(CHANNEL_ORDER),
            "dtype": "float32",
            "sample_rate_hz": 256,
            "window_seconds": 10,
        },
        "status_counts": status_counts,
        "runs": [_run_manifest_row(row) for row in complete_rows],
        "artifacts": {
            "summary": "summary.csv",
            "latency": "latency.csv",
            "cohort_counts": "cohort_counts.csv",
            "readme": "README.md",
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def _run_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "target_id", "source_pm", "task_type", "status", "checkpoint",
        "predictions", "training_log", "validation_split",
        "normalization_stats", "target_transform", "target_transform_hash",
    ):
        value = row.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue
        result[key] = value
    return result
