"""Planning and aggregation for the preliminary one-fold model-zoo comparison."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import platform
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from bench.bench_runner import BenchmarkRunner
from bench.datasets.raw_eeg_window_dataset import RawEEGWindowArrayView
from bench.features.cogstate_feature_cache import load_feature_cache, load_feature_profile
from bench.features.cogstate_feature_cache import build_canonical_feature_index
from model_zoo import build_model
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
RUNNER_MODEL_ALIAS = "m"


def target_slug(target_id: str) -> str:
    if target_id.startswith("pm_") and target_id.endswith("_q3_fold_local"):
        return target_id.removeprefix("pm_").removesuffix("_fold_local")
    if target_id.startswith("pm_") and target_id.endswith("_regression"):
        return target_id.removeprefix("pm_").removesuffix("_regression") + "_reg"
    raise ValueError(f"Unknown preliminary target_id: {target_id!r}")


def comparison_protocol_hash(config: Mapping[str, Any]) -> str:
    """Stable scientific identity excluding runtime-only absolute data roots."""
    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "config": dict(config),
        "factory_models": factory_model_names(),
        "outer_fold": 1,
        "seed": 42,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
                    REGRESSION_MODEL_NAMES
                    | {"torch_mlp", "torch_eegnet", "torch_shallow_convnet"}
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
    if model_id in {"random_forest", "mlp", "xgboost", "logistic_regression"} or (
        model_id == "svm" and task_type == "classification"
    ):
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
        "output_dir": str(output_dir / "runs" / model_id / target_slug(target_id)),
        "result_status": "preliminary",
        "datasets": {dataset_name: dataset},
        "tasks": [target_id],
        "task_config": {"target_id": target_id, "random_state": 42},
        "models": {RUNNER_MODEL_ALIAS: {"type": model_id, "task_type": task_type, "params": params}},
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


def _measure_operation(
    operation: Any, *, warmup: int = 20, repetitions: int = 100
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    timings = []
    for _ in range(repetitions):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        operation()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return latency_percentiles(timings)


def measure_model_only_latency(
    model: Any, sample: np.ndarray, *, warmup: int = 20, repetitions: int = 100
) -> dict[str, float]:
    """Measure forward-only Torch latency or estimator predict latency."""
    batch = np.ascontiguousarray(sample[None], dtype=np.float32)
    module = getattr(model, "model", None)
    transform = getattr(model, "transform_features_for_audit", None)
    if module is None or not callable(transform):
        return _measure_operation(
            lambda: model.predict(batch), warmup=warmup, repetitions=repetitions
        )
    normalized = transform(batch)
    tensor = torch.from_numpy(np.ascontiguousarray(normalized)).to(model.device_)
    module.eval()

    def forward() -> None:
        with torch.no_grad():
            module(tensor)

    return _measure_operation(forward, warmup=warmup, repetitions=repetitions)


class ResourceSampler:
    """Small process-RSS sampler with CUDA peak counters."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_ram_bytes: int | None = None
        self.peak_ram_bytes: int | None = None
        self.started_cpu_seconds: float | None = None
        self.started_wall: float | None = None

    @staticmethod
    def _memory_bytes() -> int | None:
        if platform.system() != "Windows":
            return None
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        query = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        query.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        query.restype = wintypes.BOOL
        success = query(
            process, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if success else None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            rss = self._memory_bytes()
            if rss is not None:
                self.peak_ram_bytes = max(self.peak_ram_bytes or rss, rss)

    def __enter__(self) -> "ResourceSampler":
        current_memory = self._memory_bytes()
        if current_memory is not None:
            self.baseline_ram_bytes = current_memory
            self.peak_ram_bytes = self.baseline_ram_bytes
        self.started_cpu_seconds = time.process_time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started_wall = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))

    def result(self) -> dict[str, Any]:
        wall = time.perf_counter() - float(self.started_wall or time.perf_counter())
        current_memory = self._memory_bytes()
        if current_memory is not None:
            self.peak_ram_bytes = max(self.peak_ram_bytes or current_memory, current_memory)
        cpu_seconds = (
            None
            if self.started_cpu_seconds is None
            else float(time.process_time() - self.started_cpu_seconds)
        )
        return {
            "training_wall_time_s": wall,
            "process_cpu_time_s": cpu_seconds,
            "process_cpu_percent_one_core_equivalent": (
                None if cpu_seconds is None or wall <= 0 else 100.0 * cpu_seconds / wall
            ),
            "baseline_ram_mb": (
                None if self.baseline_ram_bytes is None else self.baseline_ram_bytes / 2**20
            ),
            "peak_ram_mb": (
                None if self.peak_ram_bytes is None else self.peak_ram_bytes / 2**20
            ),
            "peak_ram_delta_mb": (
                None
                if self.peak_ram_bytes is None or self.baseline_ram_bytes is None
                else (self.peak_ram_bytes - self.baseline_ram_bytes) / 2**20
            ),
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available() else 0.0
            ),
            "peak_reserved_vram_mb": (
                torch.cuda.max_memory_reserved() / 2**20
                if torch.cuda.is_available() else 0.0
            ),
            "gpu_utilization_percent": None,
        }


class PreliminaryComparisonExecutor:
    """Thin failure-isolating orchestration over ``BenchmarkRunner``."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        data_root: Path,
        resume: bool,
        retry_failed: bool = False,
    ) -> None:
        self.config = dict(config)
        self.data_root = Path(data_root)
        self.output = Path(str(config["output_dir"]))
        self.resume = bool(resume)
        self.retry_failed = bool(retry_failed)
        self.protocol_hash = comparison_protocol_hash(config)
        self.profile_payload, self.feature_pipeline = load_feature_profile(
            config["feature_profile"]
        )
        self.raw_index = build_canonical_feature_index(
            self.data_root / config["data"]["raw_manifest"],
            self.data_root / config["data"]["logical_recording_map"],
        )
        self.raw_by_sample = {
            value: position
            for position, value in enumerate(self.raw_index["sample_id"].tolist())
        }
        self.raw_view = RawEEGWindowArrayView(
            self.raw_index, cache_path_root=self.data_root
        )
        self._prepare_state()

    def _prepare_state(self) -> None:
        manifest_path = self.output / "manifest.json"
        if self.resume and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stored_hash = manifest.get("protocol_hash")
            if stored_hash is not None and stored_hash != self.protocol_hash:
                raise ValueError(
                    "Comparison protocol hash mismatch: "
                    f"stored={stored_hash}, requested={self.protocol_hash}"
                )
        elif not manifest_path.is_file():
            write_plan(output_dir=self.output, config=self.config)
        verify_feature_cache_for_comparison(
            self.output, self.config["feature_profile"]
        )
        status_path = self.output / "run_status.csv"
        if not status_path.is_file():
            build_run_status_matrix().to_csv(status_path, index=False)
        self.status = pd.read_csv(status_path)
        for column in ("status", "stage", "error_type", "error_message"):
            self.status[column] = self.status[column].fillna("").astype(str)
        self.rows = self._existing_comparison_rows()
        for row in self.rows:
            if (
                row.get("task_type") == "classification"
                and pd.notna(row.get("target_transform_hash"))
            ):
                self._canonical_q3_hash(
                    str(row["target"]), str(row["target_transform_hash"])
                )
        self.latency_rows = self._read_csv("latency.csv")
        self.resource_rows = self._read_csv("resource_usage.csv")
        for row in self.resource_rows:
            if row.get("model") in SKLEARN_MODEL_NAMES:
                row["device"] = "cpu"
                row["peak_vram_mb"] = 0.0
                row["peak_reserved_vram_mb"] = 0.0
        self.cohort_rows = self._read_csv("cohort_counts.csv")

    def _read_csv(self, name: str) -> list[dict[str, Any]]:
        path = self.output / name
        if not path.is_file() or path.stat().st_size == 0:
            return []
        return pd.read_csv(path).to_dict("records")

    def _existing_comparison_rows(self) -> list[dict[str, Any]]:
        rows = []
        for name in ("model_comparison_q3.csv", "model_comparison_regression.csv"):
            path = self.output / name
            if path.is_file() and path.stat().st_size:
                frame = pd.read_csv(path)
                rows.extend(frame.loc[frame.status.eq("completed")].to_dict("records"))
        reuse = self.config.get("reuse_shallowconvnet")
        reuse_by_target: dict[str, dict[str, Any]] = {}
        if reuse:
            reuse_path = Path(str(reuse["source_dir"])) / "summary.csv"
            if reuse_path.is_file():
                reuse_by_target = {
                    str(row["target_id"]): row
                    for row in pd.read_csv(reuse_path).to_dict("records")
                }
        for row in rows:
            if row.get("model") != "torch_shallow_convnet":
                continue
            saved = reuse_by_target.get(str(row.get("target")), {})
            row["evaluation_cohort"] = "single_window_full"
            row.setdefault("context_seconds", None)
            row["train_count"] = saved.get("train_samples", row.get("train_count"))
            row["test_count"] = saved.get("test_samples", row.get("test_count"))
            if pd.notna(saved.get("target_transform_hash")):
                row["target_transform_hash"] = saved["target_transform_hash"]
        return rows

    def _raw_window(self, sample_id: Any) -> np.ndarray:
        try:
            position = self.raw_by_sample[sample_id]
        except KeyError as exc:
            raise KeyError(f"Raw cache lacks latency sample_id={sample_id!r}") from exc
        return np.asarray(self.raw_view[position], dtype=np.float32)

    def _feature_operations(
        self, model: Any, model_sample: np.ndarray, sample_id: Any, family: str
    ) -> tuple[Any, Any]:
        raw = self._raw_window(sample_id)[0].T

        def extract() -> np.ndarray:
            return np.ascontiguousarray(
                self.feature_pipeline.transform_window(raw), dtype=np.float32
            )

        if family == "features":
            def end_to_end() -> None:
                model.predict(extract()[None])
        else:
            buffer = np.ascontiguousarray(model_sample.copy(), dtype=np.float32)

            def end_to_end() -> None:
                updated = buffer.copy()
                updated[-1] = extract()
                model.predict(updated[None])
        return extract, end_to_end

    @staticmethod
    def _result_fold(runner: BenchmarkRunner, model_id: str, target_id: str) -> dict[str, Any]:
        dataset_name = next(iter(runner.config["datasets"]))
        return runner.results[dataset_name]["models"][target_id][RUNNER_MODEL_ALIAS][
            "group_kfold_subject"
        ]["folds"]["fold_01"]

    def _verify_checkpoint(
        self,
        runner: BenchmarkRunner,
        run_config: Mapping[str, Any],
        model_id: str,
        task_type: str,
        artifacts: Mapping[str, Any],
    ) -> tuple[bool | None, float | None, int | None]:
        model = runner.last_fitted_model
        split = runner.last_evaluated_split
        if model is None or split is None:
            raise RuntimeError("BenchmarkRunner did not expose its completed fold")
        parameter_count = None
        module = getattr(model, "model", None)
        if module is not None:
            parameter_count = int(sum(parameter.numel() for parameter in module.parameters()))
        checkpoint_value = artifacts.get("model")
        if model_id not in TORCH_MODEL_NAMES or not checkpoint_value:
            return None, None, parameter_count
        checkpoint = Path(checkpoint_value)
        params = dict(run_config["models"][RUNNER_MODEL_ALIAS]["params"])
        if model_input_family(model_id) == "raw":
            params.setdefault("sampling_rate", 256.0)
            params.setdefault("channel_names", list(split.feature_names or []))
        reloaded = build_model(
            model_id, task_type, tuple(split.X_train.shape[1:]),
            3 if task_type == "classification" else 1, params,
        )
        reloaded.load(checkpoint)
        expected = np.asarray(model.predict(split.X_test[:1]))
        actual = np.asarray(reloaded.predict(split.X_test[:1]))
        return bool(np.allclose(expected, actual, rtol=1e-5, atol=1e-6)), checkpoint.stat().st_size / 2**20, parameter_count

    def _canonical_q3_hash(self, target_id: str, transform_hash: str) -> None:
        path = self.output / "q3_transform_registry.json"
        registry = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        stored = registry.get(target_id)
        if stored is not None and stored != transform_hash:
            raise RuntimeError(
                f"Q3 transform hash changed across models for {target_id}: {stored} != {transform_hash}"
            )
        registry[target_id] = transform_hash
        path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _save_state(self) -> None:
        self.status.to_csv(self.output / "run_status.csv", index=False)
        write_comparison_tables(self.output, pd.DataFrame(self.rows + [
            row for row in self.status.to_dict("records")
            if (row["model"], row["target"]) not in {
                (item["model"], item["target"]) for item in self.rows
            }
        ]))
        pd.DataFrame(self.latency_rows).to_csv(self.output / "latency.csv", index=False)
        pd.DataFrame(self.resource_rows).to_csv(self.output / "resource_usage.csv", index=False)
        pd.DataFrame(self.cohort_rows).to_csv(self.output / "cohort_counts.csv", index=False)
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["protocol_hash"] = self.protocol_hash
        manifest["status_counts"] = self.status["status"].value_counts().to_dict()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        q3_summary_path = self.output / "model_summary_q3.csv"
        q3_summary = (
            pd.read_csv(q3_summary_path)
            if q3_summary_path.is_file() and q3_summary_path.stat().st_size else pd.DataFrame()
        )
        completed = q3_summary.loc[
            pd.to_numeric(q3_summary.get("completed_targets"), errors="coerce").fillna(0).gt(0)
        ] if not q3_summary.empty else pd.DataFrame()
        best = None
        if not completed.empty and "mean_macro_f1" in completed:
            valid = completed.dropna(subset=["mean_macro_f1"])
            if not valid.empty:
                best = valid.sort_values("mean_macro_f1", ascending=False).iloc[0]
        (self.output / "README.md").write_text(
            "# PRELIMINARY model-zoo comparison — ONE OUTER FOLD ONLY\n\n"
            "Outer fold 1, seed 42. This is an engineering comparison, not a final five-fold scientific result.\n\n"
            f"Status counts: `{manifest['status_counts']}`.\n\n"
            + (
                "Current highest mean macro F1 among completed cells: "
                f"`{best['model']}` = {float(best['mean_macro_f1']):.4f}. "
                "Models with fewer completed PM targets are not directly comparable to seven-target summaries.\n\n"
                if best is not None else ""
            )
            + "Runtime contracts:\n\n"
            "- raw: `[14,2560] → layout/normalization → model`;\n"
            "- features: `[14,2560] → FeaturePipeline → [371] → model`;\n"
            "- sequence: one new raw window → FeaturePipeline → update `[10,371]` buffer → model; startup context is 100 s.\n\n"
            "Sequence results use `sequence_eligible`; raw/feature results use `single_window_full`. "
            "Do not interpret the cohort difference as strict apples-to-apples evidence.\n\n"
            "Classification resume command:\n\n"
            "```powershell\npython scripts\\run_preliminary_model_zoo_comparison.py --config experiments\\model_zoo\\preliminary_model_zoo_comparison_fold1.json --data-root F:\\EEG --execute --task-type classification --resume\n```\n\n"
            "Regression uses the same command with `--task-type regression`.\n",
            encoding="utf-8",
        )

    def run_one(self, model_id: str, target_id: str) -> dict[str, Any] | None:
        mask = self.status.model.eq(model_id) & self.status.target.eq(target_id)
        if int(mask.sum()) != 1:
            raise ValueError(f"Unknown model/target run: {model_id}/{target_id}")
        current = str(self.status.loc[mask, "status"].iloc[0])
        if current in {"completed", "unsupported"}:
            return None
        if current == "failed" and not self.retry_failed:
            return None
        task_type = str(self.status.loc[mask, "task_type"].iloc[0])
        family = model_input_family(model_id)
        run_config = benchmark_run_config(
            self.config, model_id=model_id, target_id=target_id,
            output_dir=self.output, data_root=self.data_root,
        )
        self.status.loc[mask, ["status", "stage"]] = ["blocked", "training"]
        self._save_state()
        try:
            runner = BenchmarkRunner(run_config)
            with ResourceSampler() as sampler:
                runner.run()
            resources = sampler.result()
            fold = self._result_fold(runner, model_id, target_id)
            split = runner.last_evaluated_split
            model = runner.last_fitted_model
            if split is None or model is None:
                raise RuntimeError("BenchmarkRunner did not retain the completed fold")
            actual_device = str(getattr(model, "device_", "cpu"))
            resources["device"] = actual_device
            if actual_device == "cpu":
                resources["peak_vram_mb"] = 0.0
                resources["peak_reserved_vram_mb"] = 0.0
            if split.metadata.get("subject_overlap"):
                raise RuntimeError("Outer subject overlap is non-zero")
            validation = getattr(model, "validation_split_", None)
            if validation is not None and int(validation.get("inner_group_overlap", -1)) != 0:
                raise RuntimeError("Inner record-group overlap is non-zero")
            transform_hash = split.metadata.get("target_transform_hash")
            if task_type == "classification":
                if not transform_hash:
                    raise RuntimeError("Q3 run lacks canonical target transform hash")
                self._canonical_q3_hash(target_id, str(transform_hash))
            sample = np.asarray(split.X_test[0], dtype=np.float32)
            sample_id = split.sample_id_test[0]
            model_latency = measure_model_only_latency(model, sample)
            feature_latency = None
            if family == "raw":
                end_to_end_latency = measure_prediction_latency(model, sample)
            else:
                extract, end_to_end = self._feature_operations(
                    model, sample, sample_id, family
                )
                feature_latency = _measure_operation(extract)
                end_to_end_latency = _measure_operation(end_to_end)
            artifacts = fold.get("artifacts", {})
            reload_verified, checkpoint_size, parameter_count = self._verify_checkpoint(
                runner, run_config, model_id, task_type, artifacts
            )
            metrics = fold["metrics"]
            sequence_stats = split.metadata.get("sequence_stats", {})
            train_sequence_stats = sequence_stats.get("train", {})
            test_sequence_stats = sequence_stats.get("test", {})
            row = {
                "model": model_id, "target": target_id, "task_type": task_type,
                "input_family": family,
                "evaluation_cohort": "sequence_eligible" if family == "sequence" else "single_window_full",
                "outer_fold": 1, "seed": 42,
                "train_count": int(fold["n_train"]), "test_count": int(fold["n_test"]),
                "full_target_train_count": train_sequence_stats.get("full_target_count"),
                "full_target_test_count": test_sequence_stats.get("full_target_count"),
                "sequence_train_endpoint_count": train_sequence_stats.get("sequence_endpoint_count"),
                "sequence_test_endpoint_count": test_sequence_stats.get("sequence_endpoint_count"),
                "dropped_no_history": (
                    int(train_sequence_stats.get("dropped_no_history", 0))
                    + int(test_sequence_stats.get("dropped_no_history", 0))
                    if family == "sequence" else None
                ),
                "dropped_gap": (
                    int(train_sequence_stats.get("dropped_gap", 0))
                    + int(test_sequence_stats.get("dropped_gap", 0))
                    if family == "sequence" else None
                ),
                "dropped_other": (
                    int(train_sequence_stats.get("dropped_other", 0))
                    + int(test_sequence_stats.get("dropped_other", 0))
                    if family == "sequence" else None
                ),
                "context_length_windows": 10 if family == "sequence" else None,
                "context_seconds": 100 if family == "sequence" else None,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"), "weighted_f1": metrics.get("weighted_f1"),
                "mae": metrics.get("mae_macro", metrics.get("mae")),
                "rmse": metrics.get("rmse_macro", metrics.get("rmse")),
                "r2": metrics.get("r2_macro", metrics.get("r2")),
                "pearson": metrics.get("pearson_macro", metrics.get("pearson")),
                "spearman": metrics.get("spearman_macro", metrics.get("spearman")),
                "train_time_s": float(fold["training_time"]),
                **{f"model_latency_{name}": value for name, value in model_latency.items()},
                "feature_extraction_p95_ms": (
                    None if feature_latency is None else feature_latency["p95_ms"]
                ),
                **{f"end_to_end_latency_{name}": value for name, value in end_to_end_latency.items()},
                "peak_ram_mb": resources["peak_ram_mb"],
                "peak_vram_mb": resources["peak_vram_mb"],
                "parameter_count": parameter_count,
                "checkpoint_size_mb": checkpoint_size,
                "checkpoint_reload_verified": reload_verified,
                "reused_existing_run": False, "status": "completed",
                "target_transform_hash": transform_hash,
                "notes": "PRELIMINARY; ONE OUTER FOLD ONLY",
            }
            self.rows = [
                item for item in self.rows
                if (item["model"], item["target"]) != (model_id, target_id)
            ] + [row]
            self.latency_rows = [
                item for item in self.latency_rows
                if (item.get("model"), item.get("target")) != (model_id, target_id)
            ] + [{
                "model": model_id, "target": target_id, "input_family": family,
                **{f"model_{key}": value for key, value in model_latency.items()},
                **({} if feature_latency is None else {f"feature_extraction_{key}": value for key, value in feature_latency.items()}),
                **{f"end_to_end_{key}": value for key, value in end_to_end_latency.items()},
                "warmup_iterations": 20, "measured_iterations": 100, "batch_size": 1,
            }]
            self.resource_rows = [
                item for item in self.resource_rows
                if (item.get("model"), item.get("target")) != (model_id, target_id)
            ] + [{"model": model_id, "target": target_id, **resources}]
            self.cohort_rows = [
                item for item in self.cohort_rows
                if (item.get("model"), item.get("target")) != (model_id, target_id)
            ] + [{
                "model": model_id, "target": target_id,
                "evaluation_cohort": row["evaluation_cohort"],
                "train_count": row["train_count"], "test_count": row["test_count"],
                "full_target_train_count": row["full_target_train_count"],
                "full_target_test_count": row["full_target_test_count"],
                "sequence_train_endpoint_count": row["sequence_train_endpoint_count"],
                "sequence_test_endpoint_count": row["sequence_test_endpoint_count"],
                "dropped_no_history": row["dropped_no_history"],
                "dropped_gap": row["dropped_gap"], "dropped_other": row["dropped_other"],
                "target_transform_hash": transform_hash,
            }]
            self.status.loc[mask, "status"] = "completed"
            self.status.loc[mask, "stage"] = "completed"
            self.status.loc[mask, ["error_type", "error_message"]] = ""
            self._save_state()
            return row
        except Exception as exc:
            self.status.loc[mask, "status"] = "failed"
            self.status.loc[mask, "stage"] = "execution"
            self.status.loc[mask, "error_type"] = type(exc).__name__
            self.status.loc[mask, "error_message"] = str(exc)
            self._save_state()
            return None

    def run(
        self,
        *,
        task_type: str = "all",
        models: Sequence[str] | None = None,
        targets: Sequence[str] | None = None,
    ) -> dict[str, int]:
        allowed_tasks = {"classification", "regression", "all"}
        if task_type not in allowed_tasks:
            raise ValueError(f"task_type must be one of {sorted(allowed_tasks)}")
        selected_models = set(factory_model_names() if models is None else models)
        unknown_models = selected_models - set(factory_model_names())
        if unknown_models:
            raise ValueError(f"Unknown models: {sorted(unknown_models)}")
        selected_targets = None if targets is None else set(targets)
        for status_row in self.status.to_dict("records"):
            if status_row["model"] not in selected_models:
                continue
            if task_type != "all" and status_row["task_type"] != task_type:
                continue
            if selected_targets is not None and status_row["target"] not in selected_targets:
                continue
            self.run_one(status_row["model"], status_row["target"])
        self._save_state()
        return {
            str(key): int(value)
            for key, value in self.status["status"].value_counts().items()
        }


def aggregate_model_summary(rows: pd.DataFrame, *, task_type: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    output: list[dict[str, Any]] = []
    for model, group in rows.groupby("model", sort=True):
        completed = group.loc[group["status"].eq("completed")]
        row: dict[str, Any] = {
            "model": model,
            "input_family": group["input_family"].iloc[0],
            "evaluation_cohort": group.get(
                "evaluation_cohort", pd.Series([None])
            ).iloc[0],
            "context_seconds": pd.to_numeric(
                group.get("context_seconds", pd.Series([np.nan])), errors="coerce"
            ).max(),
            "completed_targets": int(group["status"].eq("completed").sum()),
            "failed_targets": int(group["status"].eq("failed").sum()),
            "unsupported_targets": int(group["status"].eq("unsupported").sum()),
            "blocked_targets": int(group["status"].eq("blocked").sum()),
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
        "context_seconds",
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
            "input_family": "raw", "evaluation_cohort": "single_window_full",
            "context_seconds": np.nan,
            "train_count": saved.get("train_samples"),
            "test_count": saved.get("test_samples"),
            "accuracy": saved.get("accuracy"),
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
            "target_transform_hash": saved.get("target_transform_hash"),
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
        "no_target_features": not any(
            str(name).startswith("target_") or str(name).startswith("label_")
            for name in names
        ),
        "legacy_label_metadata_ignored": "label_q5" not in {
            "source", "subject_id", "record_id", "record_group_id", "sample_id",
            "t_start", "t_end", "outer_fold",
        },
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
        "protocol_hash": comparison_protocol_hash(config),
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
