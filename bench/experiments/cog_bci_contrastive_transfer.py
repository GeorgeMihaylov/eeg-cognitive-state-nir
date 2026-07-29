"""Limited COG-BCI contrastive pretraining and label_q5 transfer screening.

The module is experiment orchestration only.  It reuses the production EEGNet,
contrastive primitives, raw-window views, Torch adapter, split validation, and
metric implementations already present in the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor, nn

from bench.datasets.cog_bci_window_cache import _shard_stem
from bench.datasets.datasets_registry import get_dataset
from bench.datasets.raw_eeg_window_dataset import RawEEGWindowArrayView
from bench.validation.metrics import MetricsCalculator
from model_zoo.DL.adapter import seed_torch
from model_zoo.DL.contrastive import (
    ContrastiveFoldData,
    ContrastiveModule,
    ContrastiveObjective,
    EEGAugmentationPipeline,
    encoder_architecture_signature,
    export_encoder_checkpoint,
    load_encoder_checkpoint,
)
from model_zoo.factory import build_model


RESULT_STATUS = "diagnostic"
EXPECTED_COG_WINDOWS = 56_903
EXPECTED_COG_RECORDS = 1_044
EXPECTED_COG_SUBJECTS = 29
EXPECTED_COG_SESSIONS = 3
EXPECTED_TASK_FAMILIES = {
    "n_back",
    "matb",
    "pvt",
    "flanker",
    "resting_state",
}
EXPECTED_INPUT_SHAPE = (1, 14, 2560)
EXPECTED_CLASSES = 5


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str | Path, *, label: str) -> Path:
    text = str(value).replace("\\", "/")
    if Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise ValueError(f"{label} must be a repository-relative path")
    path = Path(text)
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{label} must not escape the repository root")
    return path


def _git_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _state_hash(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _encoder_state(model: nn.Module) -> dict[str, Tensor]:
    prefixes = tuple(model.output_head_parameter_prefixes())
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not any(name.startswith(prefix) for prefix in prefixes)
    }


def _encoder_parameter_state(model: nn.Module) -> dict[str, Tensor]:
    prefixes = tuple(model.output_head_parameter_prefixes())
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if not any(name.startswith(prefix) for prefix in prefixes)
    }


def _head_parameter_state(model: nn.Module) -> dict[str, Tensor]:
    prefixes = tuple(model.output_head_parameter_prefixes())
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    }


def _any_parameter_changed(
    before: Mapping[str, Tensor],
    after: Mapping[str, Tensor],
) -> bool:
    return any(
        name not in after or not torch.equal(value, after[name])
        for name, value in before.items()
    )


@dataclass(frozen=True)
class UnlabelledCOGWindows:
    """Validated label-free view over all accepted COG-BCI cache windows."""

    data: RawEEGWindowArrayView
    frame: pd.DataFrame
    manifest: dict[str, Any]
    channel_order: tuple[str, ...]


def validate_unlabelled_pretraining_columns(frame: pd.DataFrame) -> None:
    """Reject target-like columns before an index can enter pretraining."""
    target_like = {
        "target",
        "label",
        "label_q5",
        "kss",
        "rsme",
        "n_back_level",
    }
    present_targets = sorted(
        target_like & {str(column).lower() for column in frame}
    )
    if present_targets:
        raise ValueError(
            f"Unlabelled pretraining index contains target columns: {present_targets}"
        )


def load_unlabelled_cog_windows(
    cache_dir: Path,
    *,
    expected_sampling_rate_hz: float = 500.0,
    expected_windows: int = EXPECTED_COG_WINDOWS,
    expected_preprocessing_names: Sequence[str] = ("none",),
) -> UnlabelledCOGWindows:
    """Load cache provenance and accepted windows without any target table."""
    manifest_path = cache_dir / "dataset_manifest.json"
    index_path = cache_dir / "window_index.parquet"
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("COG-BCI cache manifest or window index is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("preprocessing_name") not in set(
        expected_preprocessing_names
    ):
        raise ValueError(
            "Contrastive screening cache preprocessing is incompatible: "
            f"{manifest.get('preprocessing_name')!r}"
        )
    if (
        int(manifest.get("channel_count", 0)) != EXPECTED_INPUT_SHAPE[1]
        or int(manifest.get("samples_per_window", 0))
        != EXPECTED_INPUT_SHAPE[2]
        or not math.isclose(
            float(manifest.get("sampling_rate_hz", 0.0)),
            float(expected_sampling_rate_hz),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or manifest.get("dtype") != "float32"
    ):
        raise ValueError("COG-BCI cache shape, rate, or dtype is incompatible")

    frame = pd.read_parquet(index_path)
    validate_unlabelled_pretraining_columns(frame)
    frame = frame.loc[frame["status"].eq("accepted")].copy()
    if (
        len(frame) != int(expected_windows)
        or frame["record_id"].nunique() != EXPECTED_COG_RECORDS
        or frame["subject_id"].nunique() != EXPECTED_COG_SUBJECTS
        or frame["session_id"].nunique() != EXPECTED_COG_SESSIONS
    ):
        raise ValueError("COG-BCI accepted cache inventory is unexpected")
    if set(frame["task_family"].astype(str)) != EXPECTED_TASK_FAMILIES:
        raise ValueError("Not every COG-BCI task family is represented")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("COG-BCI accepted sample_id values must be unique")

    cache_files: dict[str, str] = {}
    for record_id in frame["record_id"].astype(str).unique():
        stem = _shard_stem(record_id)
        array_path = cache_dir / "shards" / f"{stem}.npy"
        metadata_path = cache_dir / "shards" / f"{stem}.json"
        if not array_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete COG-BCI shard for {record_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("record_id") != record_id
            or metadata.get("config_hash") != manifest.get("config_hash")
            or tuple(metadata.get("array_shape", [])[1:]) != (14, 2560)
        ):
            raise ValueError(f"Incompatible COG-BCI shard for {record_id}")
        cache_files[record_id] = str(array_path)

    frame["cache_file"] = frame["record_id"].astype(str).map(cache_files)
    frame["n_channels"] = EXPECTED_INPUT_SHAPE[1]
    frame["n_samples_expected"] = EXPECTED_INPUT_SHAPE[2]
    frame["view_status"] = "ok"
    view = RawEEGWindowArrayView(
        frame.rename(columns={"status": "cache_status", "view_status": "status"})
    )
    channel_order = tuple(str(value) for value in manifest["channel_order"])
    return UnlabelledCOGWindows(
        data=view,
        frame=frame.reset_index(drop=True),
        manifest=manifest,
        channel_order=channel_order,
    )


def create_pretraining_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    validation_subjects: int,
) -> dict[str, Any]:
    """Create one deterministic subject-disjoint 24/5 pretraining split."""
    subjects = np.asarray(sorted(frame["subject_id"].astype(str).unique()))
    if validation_subjects <= 0 or validation_subjects >= len(subjects):
        raise ValueError("validation_subjects must leave non-empty train and validation")
    permutation = np.random.default_rng(int(seed)).permutation(subjects)
    validation = sorted(permutation[:validation_subjects].tolist())
    training = sorted(permutation[validation_subjects:].tolist())
    train_mask = frame["subject_id"].astype(str).isin(training).to_numpy()
    validation_mask = frame["subject_id"].astype(str).isin(validation).to_numpy()
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("Pretraining split contains overlapping indices")
    if set(training) & set(validation):
        raise RuntimeError("Pretraining split contains overlapping subjects")
    split_core = {
        "schema_version": 1,
        "seed": int(seed),
        "training_subject_ids": training,
        "validation_subject_ids": validation,
        "training_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "training_sample_ids_sha256": _canonical_hash(
            frame.iloc[train_indices]["sample_id"].astype(str).tolist()
        ),
        "validation_sample_ids_sha256": _canonical_hash(
            frame.iloc[validation_indices]["sample_id"].astype(str).tolist()
        ),
    }
    return {
        **split_core,
        "training_subject_count": len(training),
        "validation_subject_count": len(validation),
        "training_window_count": int(len(train_indices)),
        "validation_window_count": int(len(validation_indices)),
        "subject_overlap_count": 0,
        "split_hash": _canonical_hash(split_core),
    }


def embedding_diagnostics(
    embeddings: Tensor,
    *,
    positive_similarity: float,
    negative_similarity: float,
) -> dict[str, Any]:
    """Return finite collapse diagnostics for normalized projections."""
    values = embeddings.detach().float().cpu()
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Embedding diagnostics require [samples, features]")
    if not torch.isfinite(values).all():
        raise ValueError("Embeddings contain NaN or Inf")
    norms = values.norm(p=2, dim=1)
    feature_std = values.std(dim=0, unbiased=False)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(values) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if float(total) <= 0:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / total
        positive = probabilities[probabilities > 0]
        effective_rank = float(torch.exp(-(positive * positive.log()).sum()))
    gap = float(positive_similarity - negative_similarity)
    diagnostics = {
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std(unbiased=False)),
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
        "feature_std_mean": float(feature_std.mean()),
        "feature_std_min": float(feature_std.min()),
        "feature_std_max": float(feature_std.max()),
        "effective_rank": effective_rank,
        "positive_similarity": float(positive_similarity),
        "negative_similarity": float(negative_similarity),
        "positive_negative_gap": gap,
        "identical_embedding_fraction": float(
            torch.all(torch.isclose(values[1:], values[:1], atol=1e-8), dim=1)
            .float()
            .mean()
        ),
    }
    if not all(
        math.isfinite(float(value)) for value in diagnostics.values()
    ):
        raise ValueError("Embedding diagnostics produced NaN or Inf")
    return diagnostics


def assess_collapse(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered fatal-collapse checks to epoch diagnostics."""
    if not history:
        return {"fatal": True, "reasons": ["no_training_epochs"]}
    reasons: list[str] = []
    for row in history:
        numeric = [
            value
            for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not all(math.isfinite(float(value)) for value in numeric):
            reasons.append("nonfinite_epoch_diagnostic")
            break
        if float(row["validation_feature_std_mean"]) <= 1e-6:
            reasons.append("zero_embedding_variance")
        if float(row["validation_identical_embedding_fraction"]) >= 0.999:
            reasons.append("identical_embeddings")
        if not 1e-4 <= float(row["validation_embedding_norm_mean"]) <= 100:
            reasons.append("embedding_norm_explosion_or_zero")
    gaps = [abs(float(row["validation_positive_negative_gap"])) for row in history]
    if gaps and max(gaps) <= 1e-4:
        reasons.append("positive_negative_gap_near_zero_all_epochs")
    return {"fatal": bool(reasons), "reasons": sorted(set(reasons))}


def transfer_decision(
    metrics_by_mode: Mapping[str, Mapping[str, float]],
    *,
    collapse_fatal: bool,
    checkpoint_valid: bool,
    leakage_safe: bool,
    macro_f1_gain: float,
    balanced_accuracy_tolerance: float,
    strong_macro_f1_gain: float,
    strong_balanced_accuracy_gain: float,
) -> dict[str, Any]:
    """Apply the deterministic, non-statistical screening decision rule."""
    baseline = metrics_by_mode.get("random_init")
    if baseline is None:
        return {"decision": "inconclusive", "reason": "random_init_missing"}
    if collapse_fatal:
        return {"decision": "do_not_proceed", "reason": "representation_collapse"}
    if not checkpoint_valid or not leakage_safe:
        return {"decision": "inconclusive", "reason": "integrity_audit_failed"}
    comparisons: dict[str, Any] = {}
    qualifying: list[str] = []
    strong: list[str] = []
    for mode in ("head_only", "full_model"):
        current = metrics_by_mode.get(mode)
        if current is None:
            continue
        delta_macro = float(current["macro_f1"] - baseline["macro_f1"])
        delta_balanced = float(
            current["balanced_accuracy"] - baseline["balanced_accuracy"]
        )
        qualifies = (
            delta_macro >= macro_f1_gain
            and delta_balanced >= -balanced_accuracy_tolerance
        )
        is_strong = (
            delta_macro >= strong_macro_f1_gain
            and delta_balanced >= strong_balanced_accuracy_gain
        )
        comparisons[mode] = {
            "macro_f1_delta": delta_macro,
            "balanced_accuracy_delta": delta_balanced,
            "qualifies": qualifies,
            "strong": is_strong,
        }
        if qualifies:
            qualifying.append(mode)
        if is_strong:
            strong.append(mode)
    decision = (
        "strong_proceed" if strong else "proceed" if qualifying else "do_not_proceed"
    )
    return {
        "decision": decision,
        "reason": "deterministic_screening_rule",
        "qualifying_modes": qualifying,
        "strong_modes": strong,
        "comparisons": comparisons,
    }


def validate_encoder_manifest_for_downstream(
    manifest: Mapping[str, Any],
    *,
    input_shape: Sequence[int],
    channel_order: Sequence[str],
) -> None:
    if tuple(manifest.get("input_shape", ())) != tuple(input_shape):
        raise ValueError("Encoder checkpoint input shape mismatch")
    if tuple(manifest.get("channel_order", ())) != tuple(channel_order):
        raise ValueError("Encoder checkpoint channel-order mismatch")
    if int(manifest.get("latent_dim", 0)) <= 0:
        raise ValueError("Encoder checkpoint latent_dim mismatch")


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_classes: int = EXPECTED_CLASSES,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if truth.shape != prediction.shape or probability.shape != (
        len(truth),
        n_classes,
    ):
        raise ValueError("Classification metric arrays have incompatible shapes")
    if not np.isfinite(probability).all():
        raise ValueError("Classification probabilities must be finite")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Classification probabilities must sum to one")
    labels = np.arange(n_classes, dtype=np.int64)
    result = MetricsCalculator.calculate_all_metrics(
        truth,
        prediction,
        probability,
        labels=labels,
    )
    result.update({
        "macro_precision": float(
            precision_score(
                truth,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                truth,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": {
            str(label): float(value)
            for label, value in zip(
                labels,
                recall_score(
                    truth,
                    prediction,
                    labels=labels,
                    average=None,
                    zero_division=0,
                ),
            )
        },
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=labels
        ).tolist(),
    })
    return _jsonable(result)


class COGBCIContrastiveTransferRunner:
    """Coordinate one preregistered pretraining and downstream fold screen."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        repository_root: Path,
    ) -> None:
        self.config = deepcopy(dict(config))
        self.repository_root = repository_root
        self._validate_config()
        self.output_dir = repository_root / _relative_path(
            self.config["output_dir"], label="output_dir"
        )
        self.pretraining_dir = self.output_dir / "pretraining" / "eegnet_seed42"
        self.downstream_dir = (
            self.output_dir / "downstream" / "label_q5_fold1_seed42"
        )

    def _validate_config(self) -> None:
        required = {
            "result_status",
            "pretraining",
            "downstream",
            "decision_rule",
            "output_dir",
            "tracked_report",
        }
        missing = sorted(required - set(self.config))
        if missing:
            raise ValueError(f"Transfer config is missing fields: {missing}")
        if self.config["result_status"] != RESULT_STATUS:
            raise ValueError("Transfer screening result_status must be diagnostic")
        pretraining = self.config["pretraining"]
        downstream = self.config["downstream"]
        if int(pretraining["seed"]) != 42 or int(downstream["seed"]) != 42:
            raise ValueError("This screening is fixed to seed 42")
        if int(pretraining["split"]["validation_subjects"]) != 5:
            raise ValueError("Pretraining validation must contain five subjects")
        if int(downstream["fold"]) != 1:
            raise ValueError("This screening is fixed to downstream fold 1")
        if set(downstream["modes"]) != {
            "random_init",
            "head_only",
            "full_model",
        }:
            raise ValueError("Exactly three downstream modes are required")
        if pretraining["additional_preprocessing"] != "none":
            raise ValueError("Additional COG-BCI preprocessing is forbidden")
        for label, value in (
            ("pretraining.cache", pretraining["cache"]),
            ("downstream.data_path", downstream["dataset"]["data_path"]),
            (
                "downstream.logical_recording_map_path",
                downstream["dataset"]["logical_recording_map_path"],
            ),
            ("output_dir", self.config["output_dir"]),
            ("tracked_report", self.config["tracked_report"]),
        ):
            _relative_path(value, label=label)

    def _input_paths(self) -> dict[str, Path]:
        cache = self.repository_root / _relative_path(
            self.config["pretraining"]["cache"], label="pretraining.cache"
        )
        dataset = self.config["downstream"]["dataset"]
        paths = {
            "cog_dataset_manifest": cache / "dataset_manifest.json",
            "cog_window_index": cache / "window_index.parquet",
            "project_raw_manifest": self.repository_root
            / _relative_path(dataset["data_path"], label="downstream.data_path"),
            "project_logical_recording_map": self.repository_root
            / _relative_path(
                dataset["logical_recording_map_path"],
                label="downstream.logical_recording_map_path",
            ),
        }
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Required input {label} is missing: {path}")
        return paths

    def _input_hashes(self) -> dict[str, str]:
        return {
            name: _sha256_file(path)
            for name, path in self._input_paths().items()
        }

    def _build_eegnet_adapter(
        self,
        *,
        mode: str,
        channel_order: Sequence[str],
    ) -> Any:
        downstream = self.config["downstream"]
        model = downstream["model"]
        params = deepcopy(dict(model["params"]))
        params.update({
            "sampling_rate": float(
                model["kernel_reference_sampling_rate_hz"]
            ),
            "channel_names": list(channel_order),
            "batch_size": int(downstream["batch_size"]),
            "max_epochs": int(downstream["max_epochs"]),
            "learning_rate": float(downstream["optimizer"]["learning_rate"]),
            "weight_decay": float(downstream["optimizer"]["weight_decay"]),
            "validation_size": float(
                downstream["inner_validation"]["fraction"]
            ),
            "early_stopping_patience": int(
                downstream["early_stopping"]["patience"]
            ),
            "early_stopping_monitor": downstream["early_stopping"]["monitor"],
            "device": downstream["device"],
            "random_state": int(downstream["seed"]),
            "standardize": True,
            "num_workers": int(downstream.get("num_workers", 0)),
        })
        adapter = build_model(
            model_name="torch_eegnet",
            task_type="classification",
            input_shape=EXPECTED_INPUT_SHAPE,
            num_outputs=EXPECTED_CLASSES,
            params=params,
        )
        adapter.model_metadata.update({
            "transfer_screening_mode": mode,
            "result_status": RESULT_STATUS,
            "outer_fold": int(downstream["fold"]),
            "target": "label_q5",
        })
        return adapter

    def _build_pretraining_module(
        self,
        channel_order: Sequence[str],
    ) -> tuple[ContrastiveModule, Any]:
        pretraining = self.config["pretraining"]
        model = pretraining["model"]
        params = deepcopy(dict(model["params"]))
        params.update({
            "sampling_rate": float(
                model["kernel_reference_sampling_rate_hz"]
            ),
            "channel_names": list(channel_order),
            "batch_size": int(pretraining["batch_size"]),
            "max_epochs": 1,
            "device": pretraining["device"],
            "random_state": int(pretraining["seed"]),
            "standardize": False,
            "num_workers": int(pretraining.get("num_workers", 0)),
        })
        adapter = build_model(
            model_name="torch_eegnet",
            task_type="classification",
            input_shape=EXPECTED_INPUT_SHAPE,
            num_outputs=EXPECTED_CLASSES,
            params=params,
        )
        projection = pretraining["projection"]
        module = ContrastiveModule(
            adapter.model,
            projection_dim=int(projection["dimension"]),
            projection_hidden_dim=int(projection["hidden_dimension"]),
        )
        return module, adapter

    @staticmethod
    def _contrastive_epoch(
        module: ContrastiveModule,
        loader: Iterable[Any],
        *,
        augmentations: EEGAugmentationPipeline,
        objective: ContrastiveObjective,
        device: torch.device,
        generator_seed: int,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> tuple[dict[str, Any], bool, bool]:
        training = optimizer is not None
        module.train(training)
        generator = EEGAugmentationPipeline.make_generator(
            generator_seed, device=device
        )
        numerator = 0.0
        denominator = 0.0
        positive_weighted = 0.0
        negative_weighted = 0.0
        projections: list[Tensor] = []
        encoder_gradient = False
        projection_gradient = False
        for batch in loader:
            inputs = batch.inputs.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                outputs = module.augmented_forward(
                    inputs, augmentations, generator=generator
                )
                result = objective(outputs)
                loss = result.contrastive_loss
                if not torch.isfinite(loss):
                    raise ValueError("Contrastive loss became NaN or infinite")
                if optimizer is not None:
                    loss.backward()
                    encoder_gradient = encoder_gradient or any(
                        parameter.grad is not None
                        and torch.count_nonzero(parameter.grad).item() > 0
                        for parameter in module.encoder_model.parameters()
                    )
                    projection_gradient = projection_gradient or any(
                        parameter.grad is not None
                        and torch.count_nonzero(parameter.grad).item() > 0
                        for parameter in module.projection_head.parameters()
                    )
                    optimizer.step()
            weight = float(result.loss.denominator.detach().cpu())
            numerator += float(result.loss.numerator.detach().cpu())
            denominator += weight
            positive_weighted += (
                weight * float(result.positive_similarity.detach().cpu())
            )
            negative_weighted += (
                weight * float(result.negative_similarity.detach().cpu())
            )
            projections.extend([
                outputs.first_projection.detach().cpu(),
                outputs.second_projection.detach().cpu(),
            ])
        if denominator <= 0 or not projections:
            raise RuntimeError("Contrastive epoch produced no valid batches")
        positive = positive_weighted / denominator
        negative = negative_weighted / denominator
        diagnostics = embedding_diagnostics(
            torch.cat(projections, dim=0),
            positive_similarity=positive,
            negative_similarity=negative,
        )
        return (
            {
                "contrastive_loss": numerator / denominator,
                "loss_numerator": numerator,
                "loss_denominator": denominator,
                **diagnostics,
            },
            encoder_gradient,
            projection_gradient,
        )

    def _run_pretraining(
        self,
        cog: UnlabelledCOGWindows,
        split: Mapping[str, Any],
        *,
        source_commit: str,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        pretraining = self.config["pretraining"]
        self.pretraining_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.pretraining_dir / "pretraining_resolved_config.json",
            pretraining,
        )
        _write_json(self.pretraining_dir / "pretraining_split.json", split)
        checkpoint_path = self.pretraining_dir / "encoder_checkpoint.pt"
        checkpoint_manifest_path = (
            self.pretraining_dir / "encoder_checkpoint_manifest.json"
        )
        summary_path = self.pretraining_dir / "pretraining_summary.json"
        history_path = self.pretraining_dir / "pretraining_history.csv"
        diagnostics_path = self.pretraining_dir / "embedding_diagnostics.csv"
        if all(
            path.is_file()
            for path in (
                checkpoint_path,
                checkpoint_manifest_path,
                summary_path,
                history_path,
                diagnostics_path,
            )
        ):
            checkpoint_manifest = json.loads(
                checkpoint_manifest_path.read_text(encoding="utf-8")
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                checkpoint_manifest.get("pretraining_subject_split_hash")
                != split["split_hash"]
                or checkpoint_manifest.get("checkpoint_sha256")
                != _sha256_file(checkpoint_path)
            ):
                raise ValueError(
                    "Existing pretraining checkpoint provenance is incompatible"
                )
            summary["resumed"] = True
            return checkpoint_path, checkpoint_manifest, summary
        module, adapter = self._build_pretraining_module(cog.channel_order)
        device = _resolve_device(pretraining["device"])
        module.to(device)
        objective = ContrastiveObjective(
            temperature=float(pretraining["temperature"])
        )
        augmentations = EEGAugmentationPipeline.from_config(
            pretraining["augmentations"]
        ).to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(pretraining["optimizer"]["learning_rate"]),
            weight_decay=float(pretraining["optimizer"]["weight_decay"]),
        )
        sample_ids = cog.frame["sample_id"].astype(str).tolist()
        record_groups = cog.frame["record_group_id"].astype(str).tolist()
        subject_ids = cog.frame["subject_id"].astype(str).tolist()
        train_indices = split["training_indices"]
        validation_indices = split["validation_indices"]
        train_scope = ContrastiveFoldData.from_indexed_source(
            features=cog.data,
            sample_ids=sample_ids,
            record_group_ids=record_groups,
            subject_ids=subject_ids,
            training_indices=train_indices,
            inner_validation_indices=(),
            outer_test_indices=validation_indices,
            target_final_evaluation_indices=(),
            fold_id="cog-bci-pretraining-train",
        )
        validation_scope = ContrastiveFoldData.from_indexed_source(
            features=cog.data,
            sample_ids=sample_ids,
            record_group_ids=record_groups,
            subject_ids=subject_ids,
            training_indices=validation_indices,
            inner_validation_indices=(),
            outer_test_indices=train_indices,
            target_final_evaluation_indices=(),
            fold_id="cog-bci-pretraining-validation",
        )
        train_loader = train_scope.training_loader(
            batch_size=int(pretraining["batch_size"]),
            shuffle=True,
            random_state=int(pretraining["seed"]),
            drop_last=True,
        )
        validation_loader = validation_scope.training_loader(
            batch_size=int(pretraining["batch_size"]),
            shuffle=False,
            random_state=int(pretraining["seed"]),
            drop_last=True,
        )

        history: list[dict[str, Any]] = []
        best_state: Optional[dict[str, Tensor]] = None
        best_loss = float("inf")
        best_epoch: Optional[int] = None
        epochs_without_improvement = 0
        encoder_gradient_observed = False
        projection_gradient_observed = False
        started = time.perf_counter()
        for epoch in range(1, int(pretraining["max_epochs"]) + 1):
            epoch_started = time.perf_counter()
            train_metrics, encoder_gradient, projection_gradient = (
                self._contrastive_epoch(
                    module,
                    train_loader,
                    augmentations=augmentations,
                    objective=objective,
                    device=device,
                    generator_seed=int(pretraining["seed"]) + epoch,
                    optimizer=optimizer,
                )
            )
            validation_metrics, _, _ = self._contrastive_epoch(
                module,
                validation_loader,
                augmentations=augmentations,
                objective=objective,
                device=device,
                generator_seed=int(pretraining["seed"]) + 100_000,
                optimizer=None,
            )
            encoder_gradient_observed |= encoder_gradient
            projection_gradient_observed |= projection_gradient
            improved = (
                float(validation_metrics["contrastive_loss"]) < best_loss
            )
            if improved:
                best_loss = float(validation_metrics["contrastive_loss"])
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in module.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            row: dict[str, Any] = {
                "epoch": epoch,
                "temperature": float(pretraining["temperature"]),
                "batch_size": int(pretraining["batch_size"]),
                "projection_dimension": int(
                    pretraining["projection"]["dimension"]
                ),
                "normalized_embeddings": True,
                "is_best": improved,
                "epoch_time_seconds": time.perf_counter() - epoch_started,
            }
            for prefix, values in (
                ("train", train_metrics),
                ("validation", validation_metrics),
            ):
                row.update({
                    f"{prefix}_{name}": value
                    for name, value in values.items()
                })
            history.append(row)
            collapse = assess_collapse(history)
            if collapse["fatal"]:
                break
            if epochs_without_improvement >= int(
                pretraining["early_stopping_patience"]
            ):
                break
        elapsed = time.perf_counter() - started
        collapse = assess_collapse(history)
        if best_state is None or best_epoch is None:
            raise RuntimeError("Pretraining did not produce a valid checkpoint")
        module.load_state_dict(best_state, strict=True)
        module.to(device)
        module.eval()
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(
            self.pretraining_dir / "pretraining_history.csv", index=False
        )
        diagnostic_columns = [
            column
            for column in history_frame
            if column == "epoch"
            or "similarity" in column
            or "embedding_norm" in column
            or "feature_std" in column
            or "effective_rank" in column
            or "identical_embedding" in column
        ]
        history_frame[diagnostic_columns].to_csv(
            self.pretraining_dir / "embedding_diagnostics.csv", index=False
        )

        encoder_parameters = sum(
            parameter.numel()
            for name, parameter in module.encoder_model.named_parameters()
            if not name.startswith("classifier.")
        )
        projection_parameters = sum(
            parameter.numel()
            for parameter in module.projection_head.parameters()
        )
        metadata = {
            "architecture": "torch_eegnet",
            "input_shape": list(EXPECTED_INPUT_SHAPE),
            "channel_order": list(cog.channel_order),
            "window_samples": EXPECTED_INPUT_SHAPE[2],
            "source_sampling_rate_hz": float(
                cog.manifest["sampling_rate_hz"]
            ),
            "kernel_reference_sampling_rate_hz": float(
                pretraining["model"]["kernel_reference_sampling_rate_hz"]
            ),
            "pretraining_dataset": "cog_bci_emotiv_common_full",
            "pretraining_subject_split_hash": split["split_hash"],
            "augmentation_config": augmentations.configuration(),
            "temperature": float(pretraining["temperature"]),
            "projection": deepcopy(pretraining["projection"]),
            "seed": int(pretraining["seed"]),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "source_commit": source_commit,
            "additional_preprocessing": "none",
        }
        export_encoder_checkpoint(
            module.encoder_model, checkpoint_path, metadata=metadata
        )
        checkpoint_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        signature = encoder_architecture_signature(module.encoder_model)
        checkpoint_manifest = {
            "schema_version": checkpoint_payload["schema_version"],
            "architecture": "torch_eegnet",
            "architecture_signature": signature,
            "latent_dim": int(module.latent_dim),
            **metadata,
            "state_dict_keys": sorted(
                checkpoint_payload["encoder_state_dict"]
            ),
            "state_dict_shapes": {
                name: list(value.shape)
                for name, value in sorted(
                    checkpoint_payload["encoder_state_dict"].items()
                )
            },
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        }
        _write_json(
            self.pretraining_dir / "encoder_checkpoint_manifest.json",
            checkpoint_manifest,
        )
        best_row = next(row for row in history if row["epoch"] == best_epoch)
        summary = {
            "result_status": RESULT_STATUS,
            "status": "do_not_transfer" if collapse["fatal"] else "completed",
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or "CPU"
            ),
            "input_shape": list(EXPECTED_INPUT_SHAPE),
            "source_sampling_rate_hz": float(cog.manifest["sampling_rate_hz"]),
            "kernel_reference_sampling_rate_hz": float(
                pretraining["model"]["kernel_reference_sampling_rate_hz"]
            ),
            "windows": len(cog.frame),
            "records": int(cog.frame["record_id"].nunique()),
            "subjects": int(cog.frame["subject_id"].nunique()),
            "sessions": int(cog.frame["session_id"].nunique()),
            "task_families": sorted(
                cog.frame["task_family"].astype(str).unique()
            ),
            "labels_loaded": False,
            "training_windows": int(split["training_window_count"]),
            "validation_windows": int(split["validation_window_count"]),
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "training_time_seconds": elapsed,
            "latent_dim": int(module.latent_dim),
            "encoder_parameter_count": encoder_parameters,
            "projection_parameter_count": projection_parameters,
            "encoder_gradient_observed": encoder_gradient_observed,
            "projection_gradient_observed": projection_gradient_observed,
            "collapse": collapse,
            "best_epoch_diagnostics": best_row,
            "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
            "resumed": False,
        }
        _write_json(
            self.pretraining_dir / "pretraining_summary.json", summary
        )
        return checkpoint_path, checkpoint_manifest, summary

    def _load_downstream_data(self) -> Any:
        dataset = deepcopy(self.config["downstream"]["dataset"])
        dataset["data_path"] = self.repository_root / _relative_path(
            dataset["data_path"], label="downstream.data_path"
        )
        dataset["logical_recording_map_path"] = (
            self.repository_root
            / _relative_path(
                dataset["logical_recording_map_path"],
                label="downstream.logical_recording_map_path",
            )
        )
        return get_dataset("emotiv_raw_eeg", dataset).load()

    def _downstream_split(
        self,
        data: Any,
        channel_order: Sequence[str],
    ) -> dict[str, Any]:
        downstream = self.config["downstream"]
        fold = int(downstream["fold"])
        outer_fold = np.asarray(data.row_metadata["outer_fold"], dtype=int)
        outer_train = np.flatnonzero(outer_fold != fold)
        outer_test = np.flatnonzero(outer_fold == fold)
        resolver = self._build_eegnet_adapter(
            mode="split_resolver", channel_order=channel_order
        )
        outer_train_subjects = np.asarray(data.subject_ids)[outer_train]
        outer_train_records = np.asarray(data.record_ids)[outer_train]
        outer_test_subjects = np.asarray(data.subject_ids)[outer_test]
        resolver.set_validation_groups(
            outer_train_subjects,
            subject_ids=outer_train_subjects,
            record_ids=outer_train_records,
            outer_test_record_ids=np.asarray(data.record_ids)[outer_test],
            outer_test_group_ids=outer_test_subjects,
            strategy="group_holdout",
            group_column="subject_id",
            validation_size=float(
                downstream["inner_validation"]["fraction"]
            ),
            random_state=int(
                downstream["inner_validation"]["random_state"]
            ),
        )
        inner_train, inner_validation = resolver.resolve_validation_indices(
            np.asarray(data.labels)[outer_train]
        )
        sample_ids = np.asarray(data.sample_ids).astype(str)
        subjects = np.asarray(data.subject_ids).astype(str)
        records = np.asarray(data.record_ids).astype(str)
        inner_train_global = outer_train[inner_train]
        inner_validation_global = outer_train[inner_validation]
        core = {
            "outer_fold": fold,
            "outer_train_sample_ids_sha256": _canonical_hash(
                sample_ids[outer_train].tolist()
            ),
            "outer_test_sample_ids_sha256": _canonical_hash(
                sample_ids[outer_test].tolist()
            ),
            "inner_train_sample_ids_sha256": _canonical_hash(
                sample_ids[inner_train_global].tolist()
            ),
            "inner_validation_sample_ids_sha256": _canonical_hash(
                sample_ids[inner_validation_global].tolist()
            ),
        }
        return {
            "outer_train_indices": outer_train,
            "outer_test_indices": outer_test,
            "inner_train_local_indices": inner_train,
            "inner_validation_local_indices": inner_validation,
            "inner_train_global_indices": inner_train_global,
            "inner_validation_global_indices": inner_validation_global,
            "outer_train_subject_ids": sorted(set(subjects[outer_train])),
            "outer_test_subject_ids": sorted(set(subjects[outer_test])),
            "inner_train_subject_ids": sorted(set(subjects[inner_train_global])),
            "inner_validation_subject_ids": sorted(
                set(subjects[inner_validation_global])
            ),
            "inner_train_record_ids": sorted(set(records[inner_train_global])),
            "inner_validation_record_ids": sorted(
                set(records[inner_validation_global])
            ),
            "split_hash": _canonical_hash(core),
            **core,
        }

    @staticmethod
    def _overlap(left: Iterable[str], right: Iterable[str]) -> list[str]:
        return sorted(set(map(str, left)) & set(map(str, right)))

    def _leakage_audit(
        self,
        data: Any,
        split: Mapping[str, Any],
        pretraining_split: Mapping[str, Any],
    ) -> dict[str, Any]:
        sample_ids = np.asarray(data.sample_ids).astype(str)
        subjects = np.asarray(data.subject_ids).astype(str)
        records = np.asarray(data.record_ids).astype(str)
        outer_train = split["outer_train_indices"]
        outer_test = split["outer_test_indices"]
        inner_train = split["inner_train_global_indices"]
        inner_validation = split["inner_validation_global_indices"]
        overlaps = {
            "downstream_train_test_subject_overlap": self._overlap(
                subjects[outer_train], subjects[outer_test]
            ),
            "downstream_train_test_sample_overlap": self._overlap(
                sample_ids[outer_train], sample_ids[outer_test]
            ),
            "inner_train_validation_subject_overlap": self._overlap(
                subjects[inner_train], subjects[inner_validation]
            ),
            "inner_train_validation_record_overlap": self._overlap(
                records[inner_train], records[inner_validation]
            ),
            "inner_train_validation_sample_overlap": self._overlap(
                sample_ids[inner_train], sample_ids[inner_validation]
            ),
            "pretraining_train_validation_subject_overlap": self._overlap(
                pretraining_split["training_subject_ids"],
                pretraining_split["validation_subject_ids"],
            ),
        }
        return {
            "schema_version": 1,
            "cog_bci_project_subject_overlap": "not_applicable_distinct_datasets",
            **{key: len(value) for key, value in overlaps.items()},
            "overlap_details": overlaps,
            "outer_test_used_for_pretraining": False,
            "outer_test_used_for_augmentation_selection": False,
            "outer_test_used_for_epoch_selection": False,
            "outer_test_used_for_mode_selection": False,
            "leakage_safe": not any(len(value) for value in overlaps.values()),
        }

    def _baseline_compatibility(self) -> dict[str, Any]:
        baseline = self.config["downstream"]["baseline_reference"]
        config_path = self.repository_root / _relative_path(
            baseline["config"], label="baseline_reference.config"
        )
        artifact_path = self.repository_root / _relative_path(
            baseline["fold_metrics"], label="baseline_reference.fold_metrics"
        )
        return {
            "source": "controlled_random_init_run",
            "existing_config": baseline["config"],
            "existing_fold_metrics": baseline["fold_metrics"],
            "existing_artifacts_present": (
                config_path.is_file() and artifact_path.is_file()
            ),
            "existing_baseline_reused": False,
            "reason": (
                "Existing fold-1 EEGNet selected by validation loss and used "
                "record-group inner validation with subject overlap; this "
                "screening requires one shared subject-disjoint inner split "
                "and validation macro-F1 selection."
            ),
        }

    def _run_downstream(
        self,
        checkpoint_path: Path,
        checkpoint_manifest: Mapping[str, Any],
        pretraining_split: Mapping[str, Any],
        pretraining_summary: Mapping[str, Any],
        *,
        input_hashes_before: Mapping[str, str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        downstream = self.config["downstream"]
        data = self._load_downstream_data()
        if tuple(data.data.shape[1:]) != EXPECTED_INPUT_SHAPE:
            raise ValueError("Downstream raw EEG input shape is incompatible")
        channel_order = tuple(str(value) for value in data.feature_names)
        validate_encoder_manifest_for_downstream(
            checkpoint_manifest,
            input_shape=EXPECTED_INPUT_SHAPE,
            channel_order=channel_order,
        )
        split = self._downstream_split(data, channel_order)
        leakage = self._leakage_audit(data, split, pretraining_split)
        if not leakage["leakage_safe"]:
            raise RuntimeError("Transfer screening split failed leakage audit")
        self.downstream_dir.mkdir(parents=True, exist_ok=True)

        baseline_compatibility = self._baseline_compatibility()
        preregistration = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result_status": RESULT_STATUS,
            "pretraining_checkpoint_sha256": _sha256_file(checkpoint_path),
            "downstream_fold": int(downstream["fold"]),
            "seed": int(downstream["seed"]),
            "random_init_baseline_source": baseline_compatibility,
            "modes": {
                mode: {
                    "pretrained": mode != "random_init",
                    "trainable": (
                        "output_head_only" if mode == "head_only" else "all"
                    ),
                    "max_epochs": int(downstream["max_epochs"]),
                    "batch_size": int(downstream["batch_size"]),
                    "learning_rate": float(
                        downstream["optimizer"]["learning_rate"]
                    ),
                    "weight_decay": float(
                        downstream["optimizer"]["weight_decay"]
                    ),
                    "early_stopping_monitor": downstream[
                        "early_stopping"
                    ]["monitor"],
                }
                for mode in downstream["modes"]
            },
            "primary_metric": "macro_f1",
            "secondary_metric": "balanced_accuracy",
            "decision_thresholds": deepcopy(self.config["decision_rule"]),
            "outer_test_inference_started": False,
        }
        preregistration_path = (
            self.output_dir / "transfer_screening_preregistration.json"
        )
        _write_json(preregistration_path, preregistration)
        preregistration_hash = _sha256_file(preregistration_path)

        outer_train = split["outer_train_indices"]
        outer_test = split["outer_test_indices"]
        labels = np.asarray(data.labels, dtype=np.int64)
        predictions: list[pd.DataFrame] = []
        fold_metrics: list[dict[str, Any]] = []
        confusions: dict[str, Any] = {}
        checkpoint_checks: dict[str, Any] = {}
        metrics_by_mode: dict[str, dict[str, float]] = {}
        training_summaries: dict[str, Any] = {}
        for mode in downstream["modes"]:
            mode_dir = self.downstream_dir / str(mode)
            mode_dir.mkdir(parents=True, exist_ok=True)
            adapter = self._build_eegnet_adapter(
                mode=str(mode), channel_order=channel_order
            )
            random_encoder_hash = _state_hash(_encoder_state(adapter.model))
            checkpoint_loaded = False
            transferred_encoder_hash: Optional[str] = None
            head_hash_before_transfer = _state_hash(
                _head_parameter_state(adapter.model)
            )
            if mode != "random_init":
                load_encoder_checkpoint(adapter.model, checkpoint_path)
                checkpoint_loaded = True
                transferred_encoder_hash = _state_hash(
                    _encoder_state(adapter.model)
                )
                seed_torch(int(downstream["seed"]))
                adapter.replace_output_head(
                    EXPECTED_CLASSES, task_type="classification"
                )
            if mode == "head_only":
                adapter.freeze_encoder()
            else:
                adapter.unfreeze_encoder()
            encoder_before_fit = _encoder_parameter_state(adapter.model)
            head_before_fit = _head_parameter_state(adapter.model)
            adapter.set_validation_indices(
                split["inner_train_local_indices"],
                split["inner_validation_local_indices"],
                subject_ids=np.asarray(data.subject_ids)[outer_train],
                record_ids=np.asarray(data.record_ids)[outer_train],
                group_ids=np.asarray(data.subject_ids)[outer_train],
                outer_test_record_ids=np.asarray(data.record_ids)[outer_test],
                outer_test_group_ids=np.asarray(data.subject_ids)[outer_test],
                group_column="subject_id",
            )
            model_path = mode_dir / "model.pt"
            resumed = model_path.is_file()
            if resumed:
                adapter.load(model_path)
                training_seconds = float(
                    sum(
                        float(row.get("epoch_time_seconds", 0.0))
                        for row in adapter.training_log_
                    )
                )
            else:
                started = time.perf_counter()
                adapter.fit(data.data[outer_train], labels[outer_train])
                training_seconds = time.perf_counter() - started
                adapter.save(model_path)
            pd.DataFrame(adapter.training_log_).to_csv(
                mode_dir / "training_log.csv", index=False
            )
            _write_json(
                mode_dir / "validation_split.json",
                adapter.validation_split_ or {},
            )
            if getattr(adapter, "feature_mean_", None) is not None:
                _write_json(
                    mode_dir / "normalization_stats.json",
                    {
                        "scope": "inner_train_only",
                        "channel_names": list(channel_order),
                        "mean": np.asarray(adapter.feature_mean_).tolist(),
                        "scale": np.asarray(adapter.feature_scale_).tolist(),
                    },
                )
            encoder_after_fit = _encoder_parameter_state(adapter.model)
            head_after_fit = _head_parameter_state(adapter.model)
            encoder_changed = _any_parameter_changed(
                encoder_before_fit, encoder_after_fit
            )
            head_changed = _any_parameter_changed(
                head_before_fit, head_after_fit
            )
            expected_checkpoint_hash = checkpoint_manifest[
                "checkpoint_sha256"
            ]
            checkpoint_checks[str(mode)] = {
                "checkpoint_loaded": checkpoint_loaded,
                "random_encoder_hash": random_encoder_hash,
                "transferred_encoder_hash": transferred_encoder_hash,
                "pretrained_differs_from_random_before_load": (
                    None
                    if mode == "random_init"
                    else random_encoder_hash != transferred_encoder_hash
                ),
                "transferred_encoder_matches_checkpoint": (
                    mode == "random_init"
                    or transferred_encoder_hash
                    == _state_hash(
                        torch.load(
                            checkpoint_path,
                            map_location="cpu",
                            weights_only=False,
                        )["encoder_state_dict"]
                    )
                ),
                "downstream_head_independent": (
                    mode == "random_init"
                    or head_hash_before_transfer
                    != _state_hash(_head_parameter_state(adapter.model))
                ),
                "encoder_parameters_changed_during_fit": encoder_changed,
                "head_parameters_changed_during_fit": head_changed,
                "head_only_encoder_parameters_unchanged": (
                    None if mode != "head_only" else not encoder_changed
                ),
                "full_model_encoder_parameters_changed": (
                    None if mode != "full_model" else encoder_changed
                ),
                "forward_output_width": EXPECTED_CLASSES,
                "model_checkpoint_sha256": _sha256_file(model_path),
                "encoder_checkpoint_sha256": expected_checkpoint_hash,
            }

            # Preregistration exists before this first use of the outer test.
            probabilities = adapter.predict_proba(data.data[outer_test])
            prediction = probabilities.argmax(axis=1).astype(np.int64)
            metrics = classification_metrics(
                labels[outer_test], prediction, probabilities
            )
            metrics_by_mode[str(mode)] = {
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
            }
            confusions[str(mode)] = metrics["confusion_matrix"]
            subject_rows: list[dict[str, Any]] = []
            test_subjects = np.asarray(data.subject_ids)[outer_test].astype(str)
            for subject_id in sorted(set(test_subjects)):
                mask = test_subjects == subject_id
                subject_metrics = classification_metrics(
                    labels[outer_test][mask],
                    prediction[mask],
                    probabilities[mask],
                )
                subject_rows.append({
                    "mode": str(mode),
                    "subject_id": subject_id,
                    "n_samples": int(mask.sum()),
                    "accuracy": subject_metrics["accuracy"],
                    "balanced_accuracy": subject_metrics["balanced_accuracy"],
                    "macro_f1": subject_metrics["macro_f1"],
                })
            pd.DataFrame(subject_rows).to_csv(
                mode_dir / "subject_metrics.csv", index=False
            )
            subject_frame = pd.DataFrame(subject_rows)
            fold_metrics.extend([
                {
                    "mode": str(mode),
                    "level": "window",
                    **metrics_by_mode[str(mode)],
                },
                {
                    "mode": str(mode),
                    "level": "subject_macro",
                    "accuracy": float(subject_frame["accuracy"].mean()),
                    "balanced_accuracy": float(
                        subject_frame["balanced_accuracy"].mean()
                    ),
                    "macro_f1": float(subject_frame["macro_f1"].mean()),
                },
            ])
            frame = pd.DataFrame({
                "dataset": "emotiv_raw_eeg_deduplicated",
                "task": "label_q5",
                "model": "torch_eegnet",
                "mode": str(mode),
                "fold": int(downstream["fold"]),
                "sample_id": np.asarray(data.sample_ids)[outer_test].astype(str),
                "subject_id": test_subjects,
                "record_id": np.asarray(data.record_ids)[outer_test].astype(str),
                "y_true": labels[outer_test],
                "y_pred": prediction,
            })
            for class_index in range(EXPECTED_CLASSES):
                frame[f"proba_{class_index}"] = probabilities[:, class_index]
            predictions.append(frame)
            training_summary = {
                "mode": str(mode),
                "training_time_seconds": training_seconds,
                "resumed": resumed,
                "epochs_trained": adapter.n_epochs_trained_,
                "best_epoch": adapter.best_epoch_,
                "best_validation_macro_f1": adapter.best_monitor_value_,
                "best_validation_loss": adapter.best_validation_loss_,
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in adapter.model.parameters()
                ),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in adapter.model.parameters()
                    if parameter.requires_grad
                ),
                "device": str(adapter.device_),
                "metrics": metrics,
            }
            training_summaries[str(mode)] = training_summary
            _write_json(mode_dir / "metrics.json", training_summary)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        prediction_frame = pd.concat(predictions, ignore_index=True)
        if prediction_frame.duplicated(["mode", "sample_id"]).any():
            raise RuntimeError("Downstream predictions duplicate mode/sample_id")
        probability_columns = [
            f"proba_{index}" for index in range(EXPECTED_CLASSES)
        ]
        probability_values = prediction_frame[probability_columns].to_numpy()
        if not np.isfinite(probability_values).all() or not np.allclose(
            probability_values.sum(axis=1), 1.0, atol=1e-5
        ):
            raise RuntimeError("Downstream probability audit failed")
        prediction_frame.to_parquet(
            self.output_dir / "downstream_predictions.parquet", index=False
        )
        pd.DataFrame(fold_metrics).to_csv(
            self.output_dir / "downstream_fold_metrics.csv", index=False
        )
        _write_json(
            self.output_dir / "downstream_confusion_matrices.json", confusions
        )
        checkpoint_valid = all(
            (
                item["transferred_encoder_matches_checkpoint"]
                and item["head_parameters_changed_during_fit"]
                and (
                    mode != "head_only"
                    or item["head_only_encoder_parameters_unchanged"]
                )
                and (
                    mode != "full_model"
                    or item["full_model_encoder_parameters_changed"]
                )
            )
            for mode, item in checkpoint_checks.items()
        )
        checkpoint_document = {
            "schema_version": 1,
            "checkpoint_valid": checkpoint_valid,
            "modes": checkpoint_checks,
        }
        _write_json(
            self.output_dir / "checkpoint_verification.json",
            checkpoint_document,
        )
        input_hashes_after = self._input_hashes()
        leakage.update({
            "input_hashes_before": dict(input_hashes_before),
            "input_hashes_after": input_hashes_after,
            "inputs_unchanged": dict(input_hashes_before) == input_hashes_after,
            "downstream_split_hash": split["split_hash"],
            "pretraining_split_hash": pretraining_split["split_hash"],
        })
        leakage["leakage_safe"] = bool(
            leakage["leakage_safe"] and leakage["inputs_unchanged"]
        )
        _write_json(self.output_dir / "leakage_audit.json", leakage)
        decision = transfer_decision(
            metrics_by_mode,
            collapse_fatal=bool(pretraining_summary["collapse"]["fatal"]),
            checkpoint_valid=checkpoint_valid,
            leakage_safe=bool(leakage["leakage_safe"]),
            macro_f1_gain=float(
                self.config["decision_rule"]["macro_f1_minimum_gain"]
            ),
            balanced_accuracy_tolerance=float(
                self.config["decision_rule"][
                    "balanced_accuracy_maximum_degradation"
                ]
            ),
            strong_macro_f1_gain=float(
                self.config["decision_rule"]["strong_macro_f1_minimum_gain"]
            ),
            strong_balanced_accuracy_gain=float(
                self.config["decision_rule"][
                    "strong_balanced_accuracy_minimum_gain"
                ]
            ),
        )
        decision.update({
            "rule_is_statistical_significance_test": False,
            "preregistration_sha256": preregistration_hash,
        })
        _write_json(self.output_dir / "decision.json", decision)
        summary = {
            "result_status": RESULT_STATUS,
            "downstream_fold": int(downstream["fold"]),
            "seed": int(downstream["seed"]),
            "input_shape": list(EXPECTED_INPUT_SHAPE),
            "classes": EXPECTED_CLASSES,
            "outer_train_windows": int(len(outer_train)),
            "outer_test_windows": int(len(outer_test)),
            "outer_train_subjects": len(split["outer_train_subject_ids"]),
            "outer_test_subjects": len(split["outer_test_subject_ids"]),
            "inner_train_windows": int(
                len(split["inner_train_global_indices"])
            ),
            "inner_validation_windows": int(
                len(split["inner_validation_global_indices"])
            ),
            "split_hash": split["split_hash"],
            "baseline_compatibility": baseline_compatibility,
            "modes": training_summaries,
            "checkpoint_verification": checkpoint_document,
            "leakage_audit": leakage,
            "decision": decision,
            "preregistration_sha256": preregistration_hash,
        }
        return summary, split

    def _render_report(
        self,
        *,
        source_commit: str,
        cog: UnlabelledCOGWindows,
        pretraining_split: Mapping[str, Any],
        checkpoint_manifest: Mapping[str, Any],
        pretraining_summary: Mapping[str, Any],
        downstream_summary: Mapping[str, Any],
    ) -> str:
        modes = downstream_summary["modes"]
        decision = downstream_summary["decision"]
        lines = [
            "# COG-BCI contrastive EEGNet transfer screening",
            "",
            f"- Branch: `integration/benchmark-unification`.",
            f"- Source HEAD: `{source_commit}`.",
            f"- Result status: `{RESULT_STATUS}`.",
            f"- Decision: `{decision['decision']}`.",
            "",
            "## Input and pretraining contract",
            "",
            (
                f"The immutable `emotiv_common` raw cache contributed "
                f"{len(cog.frame):,} accepted windows from "
                f"{cog.frame['record_id'].nunique():,} records, "
                f"{cog.frame['subject_id'].nunique()} subjects and "
                f"{cog.frame['session_id'].nunique()} sessions."
            ),
            (
                "All task families were used without labels: "
                + ", ".join(
                    sorted(cog.frame["task_family"].astype(str).unique())
                )
                + "."
            ),
            (
                "Input is `[B, 1, 14, 2560]`, float32, from the existing raw "
                "cache. No band-pass, notch, demean, CAR, or resampling was added."
            ),
            (
                f"Subject-disjoint pretraining split: "
                f"{pretraining_split['training_subject_count']} train / "
                f"{pretraining_split['validation_subject_count']} validation "
                f"subjects; {pretraining_split['training_window_count']:,} / "
                f"{pretraining_split['validation_window_count']:,} windows; "
                f"split hash `{pretraining_split['split_hash']}`."
            ),
            "",
            "The fixed augmentation order was Gaussian noise, amplitude scaling, "
            "time masking, channel masking, and temporal shift. The existing "
            "ProjectionHead and normalized in-batch NT-Xent objective were used; "
            "checkpoint selection used only pretraining-validation contrastive loss.",
            "",
            "## Pretraining result",
            "",
            (
                f"EEGNet latent width: {pretraining_summary['latent_dim']}; "
                f"encoder parameters: {pretraining_summary['encoder_parameter_count']:,}; "
                f"projection parameters: "
                f"{pretraining_summary['projection_parameter_count']:,}."
            ),
            (
                f"Training completed {pretraining_summary['epochs_trained']} epochs "
                f"in {pretraining_summary['training_time_seconds']:.1f} s; best "
                f"epoch {pretraining_summary['best_epoch']}, validation NT-Xent "
                f"{pretraining_summary['best_validation_loss']:.6f}."
            ),
            (
                f"Collapse audit: fatal={pretraining_summary['collapse']['fatal']}, "
                f"reasons={pretraining_summary['collapse']['reasons']}."
            ),
            (
                f"Encoder checkpoint SHA-256: "
                f"`{checkpoint_manifest['checkpoint_sha256']}`."
            ),
            "",
            "## Downstream fold-1 comparison",
            "",
            (
                "The canonical deduplicated raw `label_q5` dataset and its "
                "precomputed subject GroupKFold outer fold 1 were retained. "
                "All three modes used the same subject-disjoint inner split, "
                "training budget, preprocessing, seed, batching, metrics, and "
                "inner-validation macro-F1 checkpoint selection."
            ),
            "",
            "| Mode | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Epochs | Best val macro F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mode in ("random_init", "head_only", "full_model"):
            item = modes[mode]
            metrics = item["metrics"]
            lines.append(
                f"| {mode} | {metrics['accuracy']:.4f} | "
                f"{metrics['balanced_accuracy']:.4f} | "
                f"{metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} | "
                f"{item['epochs_trained']} | "
                f"{item['best_validation_macro_f1']:.4f} |"
            )
        lines.extend([
            "",
            "Confusion matrices and per-class recall are preserved in the runtime "
            "JSON and mode metrics. Subject-level window metrics are also saved.",
            "",
            "## Leakage and checkpoint audit",
            "",
            (
                f"Leakage safe: "
                f"{downstream_summary['leakage_audit']['leakage_safe']}; "
                "outer train/test subject and sample overlap, inner "
                "train/validation subject, record and sample overlap are all zero. "
                "The outer test was not used for pretraining, augmentation, epoch, "
                "or mode selection."
            ),
            (
                f"Checkpoint valid: "
                f"{downstream_summary['checkpoint_verification']['checkpoint_valid']}. "
                "The projection head was not transferred; downstream heads were "
                "new five-output heads."
            ),
            "",
            "## Decision",
            "",
            (
                f"The preregistered deterministic screening rule returned "
                f"`{decision['decision']}`. This is not a statistical-significance "
                "claim."
            ),
            "",
            "## Limitations and next step",
            "",
            (
                "This is one outer fold and one seed. COG-BCI and the project "
                "share channel order and sample count but not sampling rate "
                "(500 versus 256 Hz); the EEGNet kernel shape is fixed to the "
                "downstream 256-Hz architecture for strict encoder transfer. "
                "No full five-fold or multi-seed experiment is justified unless "
                "the preregistered screening threshold is met."
            ),
            "",
        ])
        return "\n".join(lines)

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        source_commit = _git_commit(self.repository_root)
        input_hashes_before = self._input_hashes()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=["stage", "error_type", "message"]
        ).to_csv(self.output_dir / "errors.csv", index=False)
        cache_dir = self.repository_root / _relative_path(
            self.config["pretraining"]["cache"], label="pretraining.cache"
        )
        cog = load_unlabelled_cog_windows(cache_dir)
        split = create_pretraining_split(
            cog.frame,
            seed=int(self.config["pretraining"]["split"]["seed"]),
            validation_subjects=int(
                self.config["pretraining"]["split"]["validation_subjects"]
            ),
        )
        checkpoint, checkpoint_manifest, pretraining_summary = (
            self._run_pretraining(cog, split, source_commit=source_commit)
        )
        if pretraining_summary["collapse"]["fatal"]:
            decision = {
                "decision": "do_not_transfer",
                "reason": "representation_collapse",
            }
            _write_json(self.output_dir / "decision.json", decision)
            return {
                "result_status": RESULT_STATUS,
                "pretraining": pretraining_summary,
                "decision": decision,
            }
        downstream_summary, downstream_split = self._run_downstream(
            checkpoint,
            checkpoint_manifest,
            split,
            pretraining_summary,
            input_hashes_before=input_hashes_before,
        )
        report = self._render_report(
            source_commit=source_commit,
            cog=cog,
            pretraining_split=split,
            checkpoint_manifest=checkpoint_manifest,
            pretraining_summary=pretraining_summary,
            downstream_summary=downstream_summary,
        )
        (self.output_dir / "screening_report.md").write_text(
            report, encoding="utf-8"
        )
        tracked_report = self.repository_root / _relative_path(
            self.config["tracked_report"], label="tracked_report"
        )
        tracked_report.parent.mkdir(parents=True, exist_ok=True)
        tracked_report.write_text(report, encoding="utf-8")
        summary = {
            "schema_version": 1,
            "result_status": RESULT_STATUS,
            "source_commit": source_commit,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_time_seconds": time.perf_counter() - started,
            "input_hashes": input_hashes_before,
            "cog_cache_config_hash": cog.manifest["config_hash"],
            "cog_channel_mapping_hash": cog.manifest["channel_mapping_hash"],
            "pretraining": pretraining_summary,
            "pretraining_split_hash": split["split_hash"],
            "encoder_checkpoint": checkpoint_manifest,
            "downstream": downstream_summary,
            "downstream_split_hash": downstream_split["split_hash"],
            "decision": downstream_summary["decision"],
        }
        _write_json(self.output_dir / "screening_summary.json", summary)
        return summary


def run_cog_bci_contrastive_transfer(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    return COGBCIContrastiveTransferRunner(
        config, repository_root=Path(repository_root)
    ).run()


__all__ = [
    "COGBCIContrastiveTransferRunner",
    "EXPECTED_INPUT_SHAPE",
    "UnlabelledCOGWindows",
    "assess_collapse",
    "classification_metrics",
    "create_pretraining_split",
    "embedding_diagnostics",
    "load_unlabelled_cog_windows",
    "run_cog_bci_contrastive_transfer",
    "transfer_decision",
    "validate_unlabelled_pretraining_columns",
    "validate_encoder_manifest_for_downstream",
]
