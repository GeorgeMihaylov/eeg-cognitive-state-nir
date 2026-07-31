"""One-fold, one-seed diagnostic FOMAML experiment on canonical raw EEG."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_score, recall_score

from bench.analysis.subject_metrics import calculate_subject_metrics
from bench.datasets.datasets_registry import get_dataset
from bench.meta import FOMAMLConfig, FirstOrderMAML, model_state_hash
from bench.meta.production import audit_architectures
from bench.validation.metrics import MetricsCalculator
from model_zoo.DL.adapter import TorchClassificationAdapter, seed_torch
from model_zoo.factory import build_model


SCHEMA_VERSION = "fomaml-label-q5-diagnostic-v1"
MODES = (
    "zero_shot_supervised",
    "supervised_full_model",
    "selected_fomaml",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            _jsonable(value), indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8") + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _object_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=lambda candidate: str(candidate)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for member in item:
                update(member)
        else:
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def resolve_device(requested: str) -> str:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Diagnostic FOMAML supports CPU or CUDA only")
    return str(device)


def validate_diagnostic_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_enabled") is not True:
        raise ValueError("Diagnostic config must explicitly set execution_enabled=true")
    if int(config["seed"]) != 42 or int(config["protocol"]["outer_fold"]) != 1:
        raise ValueError("Task 8X is restricted to seed 42 and outer fold 1")
    if config["model"]["name"] != "torch_eegnet":
        raise ValueError("Task 8X is restricted to production EEGNet")
    if list(config["fomaml"]["buffer_policies"]) != [
        "frozen_global", "support_local"
    ]:
        raise ValueError("Both preregistered BatchNorm policies are required")
    if int(config["fomaml"]["inner_steps"]) <= 0:
        raise ValueError("inner_steps must be positive")
    if int(config["protocol"]["support_budget"]) != 32:
        raise ValueError("Task 8X support budget is fixed at 32")
    if int(config["protocol"]["query_budget"]) != 64:
        raise ValueError("Task 8X query budget is fixed at 64")


def select_buffer_policy(
    results: Mapping[str, Mapping[str, float]], *, tie_threshold: float
) -> dict[str, Any]:
    required = {"frozen_global", "support_local"}
    if set(results) != required:
        raise ValueError(f"Policy results must contain exactly {sorted(required)}")
    frozen = results["frozen_global"]
    local = results["support_local"]
    difference = float(local["macro_f1"] - frozen["macro_f1"])
    if abs(difference) < float(tie_threshold):
        selected = "frozen_global"
        reason = "absolute_macro_f1_difference_below_tie_threshold"
    elif difference > 0:
        selected = "support_local"
        reason = "higher_meta_validation_macro_f1"
    else:
        selected = "frozen_global"
        reason = "higher_meta_validation_macro_f1"
    return {
        "selected_policy": selected,
        "reason": reason,
        "tie_threshold": float(tie_threshold),
        "support_local_minus_frozen_global_macro_f1": difference,
        "results": _jsonable(results),
        "outer_test_used": False,
    }


def apply_decision_rule(
    comparison: Mapping[str, Any], rule: Mapping[str, Any]
) -> dict[str, Any]:
    macro = float(comparison["mean_delta_macro_f1"])
    balanced = float(comparison["mean_delta_balanced_accuracy"])
    wins = int(comparison["macro_f1_wins"])
    if (
        macro >= float(rule["strong_macro_f1_gain"])
        and balanced >= float(rule["strong_balanced_accuracy_gain"])
        and wins >= int(rule["strong_min_wins"])
    ):
        status = "strong_proceed"
    elif (
        macro >= float(rule["proceed_macro_f1_gain"])
        and balanced >= float(rule["proceed_balanced_accuracy_min_gain"])
        and wins >= int(rule["proceed_min_wins"])
    ):
        status = "proceed"
    elif (
        (macro < 0 and balanced < 0)
        or wins <= int(rule["do_not_proceed_max_wins"])
    ):
        status = "do_not_proceed"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "observed": _jsonable(comparison),
        "rule": _jsonable(rule),
        "statistical_significance_claimed": False,
    }


def paired_subject_comparison(
    subject_metrics: pd.DataFrame, candidate: str, reference: str
) -> dict[str, Any]:
    columns = ["subject_id", "macro_f1", "balanced_accuracy", "ordinal_mae"]
    candidate_frame = subject_metrics.loc[
        subject_metrics["mode"].eq(candidate), columns
    ].set_index("subject_id")
    reference_frame = subject_metrics.loc[
        subject_metrics["mode"].eq(reference), columns
    ].set_index("subject_id")
    if set(candidate_frame.index) != set(reference_frame.index):
        raise ValueError("Paired comparison requires identical subjects")
    paired = candidate_frame.join(reference_frame, lsuffix="_candidate", rsuffix="_reference")
    for metric in ("macro_f1", "balanced_accuracy", "ordinal_mae"):
        paired[f"delta_{metric}"] = (
            paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
        )
    macro_delta = paired["delta_macro_f1"].to_numpy(dtype=float)
    tolerance = 1e-12
    return {
        "candidate": candidate,
        "reference": reference,
        "n_subjects": int(len(paired)),
        "macro_f1_wins": int((macro_delta > tolerance).sum()),
        "macro_f1_losses": int((macro_delta < -tolerance).sum()),
        "macro_f1_ties": int((np.abs(macro_delta) <= tolerance).sum()),
        "mean_delta_macro_f1": float(paired["delta_macro_f1"].mean()),
        "median_delta_macro_f1": float(paired["delta_macro_f1"].median()),
        "mean_delta_balanced_accuracy": float(paired["delta_balanced_accuracy"].mean()),
        "median_delta_balanced_accuracy": float(paired["delta_balanced_accuracy"].median()),
        "mean_delta_ordinal_mae": float(paired["delta_ordinal_mae"].mean()),
        "median_delta_ordinal_mae": float(paired["delta_ordinal_mae"].median()),
        "subjects": paired.reset_index().to_dict("records"),
        "comparison_unit": "subject",
    }


def validate_episode_protocol(
    protocol: Mapping[str, Any], episode_index: pd.DataFrame,
    *, expected_hash: str, support_budget: int, query_budget: int,
) -> dict[str, Any]:
    if protocol.get("protocol_hash") != expected_hash:
        raise ValueError("Episode protocol hash does not match task 8F")
    required_scopes = {"meta_train": 23, "meta_validation": 9, "outer_test": 8}
    observed = {
        str(key): int(value)
        for key, value in episode_index["scope"].value_counts().items()
    }
    if observed != required_scopes:
        raise ValueError(f"Episode scope counts changed: {observed}")
    subject_sets = {
        scope: set(episode_index.loc[episode_index["scope"].eq(scope), "subject_id"].astype(str))
        for scope in required_scopes
    }
    if any(
        subject_sets[left] & subject_sets[right]
        for left, right in (
            ("meta_train", "meta_validation"),
            ("meta_train", "outer_test"),
            ("meta_validation", "outer_test"),
        )
    ):
        raise ValueError("Meta-learning subject partitions overlap")
    for row in episode_index.itertuples():
        if len(row.support_sample_ids) != support_budget or len(row.query_sample_ids) != query_budget:
            raise ValueError("Episode support/query budget changed")
        if set(map(str, row.support_sample_ids)) & set(map(str, row.query_sample_ids)):
            raise ValueError("Episode support/query samples overlap")
        if set(map(str, row.support_record_ids)) & set(map(str, row.query_record_ids)):
            raise ValueError("Episode support/query records overlap")
        if str(row.split_level) == "within_record":
            raise ValueError("Unsafe within-record fallback is forbidden")
    return {
        "valid": True,
        "protocol_hash": expected_hash,
        "scope_counts": observed,
        "subject_counts": {key: len(value) for key, value in subject_sets.items()},
        "subject_overlap": 0,
        "support_query_sample_overlap": 0,
        "support_query_record_overlap": 0,
        "unsafe_fallback_used": False,
    }


def audit_raw_episode_alignment(
    metadata: pd.DataFrame, episode_index: pd.DataFrame
) -> dict[str, Any]:
    """Prove that every preregistered ID exists in the raw-deduplicated universe."""
    required = {"sample_id", "subject_id", "label_q5"}
    missing_columns = sorted(required - set(metadata))
    if missing_columns:
        raise ValueError(f"Raw metadata is missing columns: {missing_columns}")
    lookup = metadata.assign(
        sample_id=metadata["sample_id"].astype(str),
        subject_id=metadata["subject_id"].astype(str),
    ).set_index("sample_id")
    available = set(lookup.index)
    rows: list[dict[str, Any]] = []
    semantic_mismatches: list[dict[str, Any]] = []
    present = 0
    total = 0
    for episode in episode_index.itertuples():
        for partition in ("support", "query"):
            sample_ids = tuple(
                map(str, getattr(episode, f"{partition}_sample_ids"))
            )
            targets = np.asarray(
                getattr(episode, f"{partition}_targets"), dtype=np.int64
            )
            absent = [sample_id for sample_id in sample_ids if sample_id not in available]
            total += len(sample_ids)
            present += len(sample_ids) - len(absent)
            for sample_id, target in zip(sample_ids, targets):
                if sample_id not in available:
                    continue
                raw = lookup.loc[sample_id]
                if (
                    str(raw["subject_id"]) != str(episode.subject_id)
                    or int(raw["label_q5"]) != int(target)
                ):
                    semantic_mismatches.append({
                        "scope": str(episode.scope),
                        "episode_id": str(episode.episode_id),
                        "partition": partition,
                        "sample_id": sample_id,
                        "episode_subject_id": str(episode.subject_id),
                        "raw_subject_id": str(raw["subject_id"]),
                        "episode_target": int(target),
                        "raw_target": int(raw["label_q5"]),
                    })
            rows.append({
                "scope": str(episode.scope),
                "subject_id": str(episode.subject_id),
                "episode_id": str(episode.episode_id),
                "partition": partition,
                "requested_samples": len(sample_ids),
                "missing_samples": len(absent),
                "missing_sample_ids": absent,
            })
    detail = pd.DataFrame(rows)
    scope_summary: dict[str, Any] = {}
    for scope, group in detail.groupby("scope", sort=True):
        per_episode = group.groupby("episode_id")["missing_samples"].sum()
        scope_summary[str(scope)] = {
            "requested_samples": int(group["requested_samples"].sum()),
            "missing_samples": int(group["missing_samples"].sum()),
            "episodes": int(group["episode_id"].nunique()),
            "affected_episodes": int(per_episode.gt(0).sum()),
            "fully_available_episodes": int(per_episode.eq(0).sum()),
        }
    missing_count = int(detail["missing_samples"].sum())
    return {
        "valid": missing_count == 0 and not semantic_mismatches,
        "raw_deduplicated_samples": int(len(metadata)),
        "requested_episode_samples": int(total),
        "present_episode_samples": int(present),
        "missing_episode_samples": missing_count,
        "semantic_mismatch_count_for_present_ids": len(semantic_mismatches),
        "semantic_mismatches": semantic_mismatches,
        "scope_summary": scope_summary,
        "affected_partitions": [
            row for row in rows if int(row["missing_samples"]) > 0
        ],
        "safe_remapping_applied": False,
        "episode_protocol_rebuilt": False,
    }


def prepare_preregistration(path: Path, payload: Mapping[str, Any]) -> str:
    """Write immutable preregistration bytes or verify an identical existing file."""
    content = _canonical_bytes(payload)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError("Existing preregistration differs from requested protocol")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return _sha256_file(path)


@dataclass(frozen=True)
class LoadedEpisode:
    episode: Any
    subject_id: str
    support_features: torch.Tensor
    support_targets: torch.Tensor
    query_features: torch.Tensor
    query_targets: torch.Tensor
    support_sample_ids: tuple[str, ...]
    query_sample_ids: tuple[str, ...]


class EpisodeTensorStore:
    """Materialize only requested manifest IDs from the existing lazy raw cache."""

    def __init__(
        self,
        data: Any,
        metadata: pd.DataFrame,
        mean: np.ndarray,
        scale: np.ndarray,
        *,
        maximum_cached_episodes: int = 6,
    ) -> None:
        self.data = data
        self.metadata = metadata.reset_index(drop=True)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.positions = {
            str(sample_id): index
            for index, sample_id in enumerate(self.metadata["sample_id"])
        }
        self.maximum_cached_episodes = int(maximum_cached_episodes)
        self.cache: OrderedDict[str, LoadedEpisode] = OrderedDict()

    def _partition(self, sample_ids: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = tuple(str(value) for value in sample_ids)
        try:
            positions = [self.positions[value] for value in ids]
        except KeyError as exc:
            raise ValueError(f"Episode sample is missing from raw cache: {exc}") from exc
        windows = np.stack(
            [np.asarray(self.data.data[index], dtype=np.float32) for index in positions]
        )
        windows = (
            windows - self.mean[None, None, :, None]
        ) / self.scale[None, None, :, None]
        targets = np.asarray(self.data.labels)[positions].astype(np.int64)
        if not np.isfinite(windows).all():
            raise ValueError("Normalized episode contains NaN or Inf")
        return torch.from_numpy(np.ascontiguousarray(windows)), torch.from_numpy(targets)

    def load(self, row: Any) -> LoadedEpisode:
        episode_id = str(row.episode_id)
        if episode_id in self.cache:
            cached = self.cache.pop(episode_id)
            self.cache[episode_id] = cached
            return cached
        support_features, support_targets = self._partition(row.support_sample_ids)
        query_features, query_targets = self._partition(row.query_sample_ids)
        expected_support = np.asarray(row.support_targets, dtype=np.int64)
        expected_query = np.asarray(row.query_targets, dtype=np.int64)
        if not np.array_equal(support_targets.numpy(), expected_support):
            raise ValueError("Raw support labels differ from episode manifest")
        if not np.array_equal(query_targets.numpy(), expected_query):
            raise ValueError("Raw query labels differ from episode manifest")
        loaded = LoadedEpisode(
            episode=SimpleNamespace(episode_id=episode_id),
            subject_id=str(row.subject_id),
            support_features=support_features,
            support_targets=support_targets,
            query_features=query_features,
            query_targets=query_targets,
            support_sample_ids=tuple(map(str, row.support_sample_ids)),
            query_sample_ids=tuple(map(str, row.query_sample_ids)),
        )
        self.cache[episode_id] = loaded
        while len(self.cache) > self.maximum_cached_episodes:
            self.cache.popitem(last=False)
        return loaded


def _episode_rows(frame: pd.DataFrame, scope: str) -> list[Any]:
    return list(frame.loc[frame["scope"].eq(scope)].sort_values("episode_id").itertuples())


def _classification_metrics(
    truth: np.ndarray, prediction: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    metrics = MetricsCalculator.calculate_all_metrics(
        truth, prediction, probabilities, labels=np.arange(5)
    )
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    )
    return {
        **metrics,
        "macro_precision": float(precision_score(
            truth, prediction, labels=np.arange(5), average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            truth, prediction, labels=np.arange(5), average="macro", zero_division=0
        )),
        "per_class_recall": recall_score(
            truth, prediction, labels=np.arange(5), average=None, zero_division=0
        ).astype(float).tolist(),
        "prediction_entropy": float(entropy.mean()),
    }


class FOMAMLLabelQ5Diagnostic:
    def __init__(self, config: Mapping[str, Any], *, repository_root: Path) -> None:
        validate_diagnostic_config(config)
        self.config = deepcopy(dict(config))
        self.root = repository_root
        self.output = self.root / str(config["output_dir"])
        self.device = resolve_device(str(config["device"]))
        self.seed = int(config["seed"])
        self.protocol_path = self.root / str(config["protocol"]["manifest"])
        self.episode_path = self.root / str(config["protocol"]["episode_index"])
        self.errors_path = self.root / str(config["protocol"]["errors"])
        self.source_hashes = {
            "protocol_manifest": _sha256_file(self.protocol_path),
            "episode_index": _sha256_file(self.episode_path),
            "errors": _sha256_file(self.errors_path),
        }

    def _architecture_audit(self) -> dict[str, Any]:
        production_config = json.loads(
            (self.root / "experiments/meta_learning/fomaml_production_contract.json")
            .read_text(encoding="utf-8")
        )
        rows, _, _ = audit_architectures(production_config, repository_root=self.root)
        row = next(item for item in rows if item["model_id"] == "torch_eegnet:canonical")
        expected = self.config["model"]
        checks = {
            "architecture_signature": row["architecture_signature"] == expected["architecture_signature"],
            "latent_dim": int(row["latent_dim"]) == int(expected["latent_dim"]),
            "parameter_count": int(row["parameter_count"]) == int(expected["parameter_count"]),
            "outputs": int(row["output_head_width"]) == int(expected["outputs"]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Production EEGNet architecture mismatch: {checks}")
        return {"checks": checks, "row": row}

    def _build_adapter(self, *, supervised: bool) -> TorchClassificationAdapter:
        dataset_config_path = self.root / str(self.config["dataset"]["config"])
        document = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8"))
        params = dict(document["models"]["torch_eegnet"]["params"])
        params.update({
            "device": self.device,
            "random_state": self.seed,
        })
        if supervised:
            spec = self.config["supervised"]
            params.update({
                "batch_size": int(spec["batch_size"]),
                "max_epochs": int(spec["maximum_epochs"]),
                "learning_rate": float(spec["learning_rate"]),
                "weight_decay": float(spec["weight_decay"]),
                "early_stopping_patience": int(spec["early_stopping_patience"]),
                "early_stopping_monitor": "validation_window_macro_f1",
                "standardize": True,
            })
        else:
            params.update({"standardize": False, "batch_size": 64})
        adapter = build_model(
            model_name="torch_eegnet",
            task_type="classification",
            input_shape=tuple(self.config["dataset"]["input_shape"]),
            num_outputs=5,
            params=params,
        )
        if not isinstance(adapter, TorchClassificationAdapter):
            raise TypeError("Factory did not return TorchClassificationAdapter")
        return adapter

    def _load_protocol(self) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
        protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        episodes = pd.read_parquet(self.episode_path)
        audit = validate_episode_protocol(
            protocol,
            episodes,
            expected_hash=str(self.config["protocol"]["expected_hash"]),
            support_budget=int(self.config["protocol"]["support_budget"]),
            query_budget=int(self.config["protocol"]["query_budget"]),
        )
        return protocol, episodes, audit

    def _preregister(
        self, protocol: Mapping[str, Any], episodes: pd.DataFrame,
        architecture: Mapping[str, Any],
    ) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "repository_commit": _git_head(self.root),
            "result_status": "diagnostic",
            "outer_fold": 1,
            "seed": self.seed,
            "model": "production_eegnet",
            "architecture_signature": architecture["row"]["architecture_signature"],
            "episode_protocol_hash": protocol["protocol_hash"],
            "episode_counts": protocol["episode_counts"],
            "support_budget": int(self.config["protocol"]["support_budget"]),
            "query_budget": int(self.config["protocol"]["query_budget"]),
            "inner_steps": int(self.config["fomaml"]["inner_steps"]),
            "inner_learning_rate": float(self.config["fomaml"]["inner_learning_rate"]),
            "meta_learning_rate": float(self.config["fomaml"]["meta_learning_rate"]),
            "meta_batch_size": int(self.config["fomaml"]["meta_batch_size"]),
            "maximum_meta_epochs": int(self.config["fomaml"]["maximum_meta_epochs"]),
            "supervised_maximum_epochs": int(self.config["supervised"]["maximum_epochs"]),
            "buffer_policies": list(self.config["fomaml"]["buffer_policies"]),
            "baseline_modes": list(self.config["baselines"]),
            "primary_metric": self.config["primary_metric"],
            "secondary_metrics": list(self.config["secondary_metrics"]),
            "checkpoint_criteria": {
                "supervised": self.config["supervised"]["checkpoint_criterion"],
                "fomaml": self.config["fomaml"]["checkpoint_criterion"],
            },
            "policy_selection": self.config["policy_selection"],
            "decision_rule": self.config["decision_rule"],
            "training_sample_contract": self.config["protocol"]["training_sample_contract"],
            "supervised_validation_contract": self.config["protocol"]["supervised_validation_contract"],
            "episode_ids": sorted(episodes["episode_id"].astype(str)),
            "source_hashes": self.source_hashes,
            "device": self.device,
            "execution_enabled": True,
        }
        root_path = self.output / "experiment_preregistration.json"
        preregistration_path = self.output / "preregistration/experiment_preregistration.json"
        digest = prepare_preregistration(root_path, payload)
        second_digest = prepare_preregistration(preregistration_path, payload)
        if digest != second_digest:
            raise RuntimeError("Preregistration copies differ")
        _write_json(self.output / "preregistration/preregistration_hash.json", {
            "sha256": digest,
            "parameters_frozen_before_training": True,
        })
        return digest

    def _load_data(self) -> tuple[Any, pd.DataFrame]:
        document = yaml.safe_load(
            (self.root / str(self.config["dataset"]["config"])).read_text(encoding="utf-8")
        )
        dataset_config = dict(document["datasets"]["emotiv_raw_eeg"])
        dataset_config["raw_preprocessing"] = document["raw_preprocessing"]
        data = get_dataset("emotiv_raw_eeg", dataset_config).load()
        if tuple(data.data.shape[1:]) != tuple(self.config["dataset"]["input_shape"]):
            raise RuntimeError("Raw EEG input shape changed")
        metadata = pd.DataFrame({
            "sample_id": np.asarray(data.sample_ids).astype(str),
            "subject_id": np.asarray(data.subject_ids).astype(str),
            "record_id": np.asarray(data.record_ids).astype(str),
            "label_q5": np.asarray(data.labels).astype(np.int64),
            "source": np.asarray(data.row_metadata.get("source", np.full(len(data.labels), "unknown"))).astype(str),
            "outer_fold": np.asarray(data.row_metadata["outer_fold"]).astype(int),
        })
        if metadata["sample_id"].duplicated().any():
            raise RuntimeError("Raw EEG cache contains duplicate sample_id")
        return data, metadata

    def _write_alignment_blocker(
        self,
        alignment: Mapping[str, Any],
        protocol_audit: Mapping[str, Any],
        preregistration_hash: str,
    ) -> None:
        _write_json(self.output / "protocol/raw_cache_alignment_audit.json", alignment)
        existing_errors = pd.read_csv(self.errors_path)
        mismatch_rows = [
            {
                "dataset_id": "emotiv_raw_eeg",
                "task_id": "label_q5",
                "fold_id": "1",
                "entity_id": row["subject_id"],
                "error_type": "ProtocolRawSampleMismatch",
                "message": (
                    f"{row['partition']} contains {row['missing_samples']} "
                    "preregistered sample IDs absent from raw deduplicated cache"
                ),
            }
            for row in alignment["affected_partitions"]
        ]
        pd.concat(
            [existing_errors, pd.DataFrame(mismatch_rows)], ignore_index=True
        ).to_csv(self.output / "errors.csv", index=False)
        leakage = {
            **dict(protocol_audit),
            "status": "blocked_before_training",
            "raw_episode_alignment_valid": False,
            "alignment": alignment,
            "outer_test_opened": False,
            "policy_selection_performed": False,
            "checkpoint_selection_performed": False,
            "source_manifests_unchanged": self.source_hashes == {
                "protocol_manifest": _sha256_file(self.protocol_path),
                "episode_index": _sha256_file(self.episode_path),
                "errors": _sha256_file(self.errors_path),
            },
        }
        _write_json(self.output / "leakage_audit.json", leakage)
        decision = {
            "status": "blocked_protocol_raw_sample_mismatch",
            "reason": (
                "The immutable task-8F episode manifest is not fully contained "
                "in the canonical raw-deduplicated sample universe."
            ),
            "missing_episode_samples": alignment["missing_episode_samples"],
            "semantic_mismatch_count_for_present_ids": alignment[
                "semantic_mismatch_count_for_present_ids"
            ],
            "training_performed": False,
            "outer_test_performed": False,
            "unsafe_fallback_used": False,
            "preregistration_hash": preregistration_hash,
        }
        _write_json(self.output / "decision.json", decision)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "result_status": "blocked",
            "preregistration_hash": preregistration_hash,
            "protocol_hash": self.config["protocol"]["expected_hash"],
            "architecture_signature": self.config["model"]["architecture_signature"],
            "alignment": alignment,
            "decision": decision,
            "real_eeg_training_performed": False,
            "outer_test_performed": False,
        }
        _write_json(self.output / "diagnostic_summary.json", summary)
        (self.output / "diagnostic_report.md").write_text(
            "# FOMAML label_q5 diagnostic — blocked pre-training\n\n"
            "The immutable task-8F episode sample universe is not fully present "
            "in the canonical raw-deduplicated EEG cache. Continuing would "
            "require rebuilding/remapping episodes, changing the protocol hash, "
            "dropping additional participants, or using non-deduplicated data. "
            "All are forbidden by task 8X.\n\n"
            f"- Missing preregistered sample IDs: "
            f"{alignment['missing_episode_samples']}.\n"
            f"- Present IDs with subject/target mismatch: "
            f"{alignment['semantic_mismatch_count_for_present_ids']}.\n"
            "- Supervised training: not started.\n"
            "- FOMAML training: not started.\n"
            "- Outer-test: not opened.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _ids(rows: Sequence[Any], field: str) -> list[str]:
        return sorted({str(value) for row in rows for value in getattr(row, field)})

    def _normalization(
        self, data: Any, metadata: pd.DataFrame, train_ids: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = metadata.index[metadata["sample_id"].isin(set(train_ids))].to_numpy(dtype=np.int64)
        if len(positions) != len(set(train_ids)):
            raise RuntimeError("Not all meta-train samples exist in raw cache")
        mean, scale = data.data[positions].compute_channel_statistics()
        if mean.shape != (14,) or scale.shape != (14,) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise RuntimeError("Invalid train-only channel normalization")
        _write_json(self.output / "protocol/normalization_stats.json", {
            "fit_partition": "materialized_meta_train_episode_samples_only",
            "fit_sample_count": len(positions),
            "mean": mean,
            "scale": scale,
        })
        return mean, scale

    def _train_supervised(
        self,
        data: Any,
        metadata: pd.DataFrame,
        train_ids: Sequence[str],
        validation_ids: Sequence[str],
        outer_rows: Sequence[Any],
        preregistration_hash: str,
    ) -> tuple[TorchClassificationAdapter, dict[str, Any]]:
        directory = self.output / "supervised"
        checkpoint = directory / "supervised_checkpoint.pt"
        manifest_path = directory / "supervised_checkpoint_manifest.json"
        adapter = self._build_adapter(supervised=True)
        initial_hash = _tensor_state_hash(adapter._initial_state)
        selected_ids = set(train_ids) | set(validation_ids)
        positions = metadata.index[metadata["sample_id"].isin(selected_ids)].to_numpy(dtype=np.int64)
        selected_metadata = metadata.iloc[positions].reset_index(drop=True)
        selected_view = data.data[positions]
        selected_labels = np.asarray(data.labels)[positions]
        train_indices = selected_metadata.index[selected_metadata["sample_id"].isin(set(train_ids))].to_numpy(dtype=np.int64)
        validation_indices = selected_metadata.index[selected_metadata["sample_id"].isin(set(validation_ids))].to_numpy(dtype=np.int64)
        outer_subjects = sorted({str(row.subject_id) for row in outer_rows})
        outer_records = sorted({str(value) for row in outer_rows for value in row.support_record_ids} | {str(value) for row in outer_rows for value in row.query_record_ids})
        adapter.set_validation_indices(
            train_indices,
            validation_indices,
            subject_ids=selected_metadata["subject_id"],
            record_ids=selected_metadata["record_id"],
            group_ids=selected_metadata["subject_id"],
            outer_test_record_ids=outer_records,
            outer_test_group_ids=outer_subjects,
            group_column="subject_id",
        )
        started = time.perf_counter()
        adapter.fit(selected_view, selected_labels)
        training_seconds = time.perf_counter() - started
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(adapter.training_log_).to_csv(
            directory / "supervised_training_history.csv", index=False
        )
        adapter.save(checkpoint)
        probabilities = adapter.predict_proba(selected_view[validation_indices])
        prediction = probabilities.argmax(axis=1)
        validation_metrics = _classification_metrics(
            selected_labels[validation_indices], prediction, probabilities
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_hash": _sha256_file(checkpoint),
            "model_hash": model_state_hash(adapter.model),
            "optimizer_state_hash": _object_hash(adapter.optimizer_state_),
            "initial_model_hash": initial_hash,
            "architecture_signature": self.config["model"]["architecture_signature"],
            "episode_protocol_hash": self.config["protocol"]["expected_hash"],
            "preregistration_hash": preregistration_hash,
            "seed": self.seed,
            "best_epoch": adapter.best_epoch_,
            "epochs_trained": adapter.n_epochs_trained_,
            "training_time_seconds": training_seconds,
            "checkpoint_criterion": self.config["supervised"]["checkpoint_criterion"],
            "meta_validation_metrics": validation_metrics,
            "train_samples": int(len(train_indices)),
            "validation_samples": int(len(validation_indices)),
            "train_subjects": sorted(selected_metadata.iloc[train_indices]["subject_id"].unique()),
            "validation_subjects": sorted(selected_metadata.iloc[validation_indices]["subject_id"].unique()),
            "outer_test_used": False,
        }
        _write_json(manifest_path, manifest)
        shutil.copy2(directory / "supervised_training_history.csv", self.output / "supervised_training_history.csv")
        shutil.copy2(checkpoint, self.output / "supervised_checkpoint.pt")
        shutil.copy2(manifest_path, self.output / "supervised_checkpoint_manifest.json")
        return adapter, manifest

    def _evaluate_adapted_episodes(
        self,
        learner: FirstOrderMAML,
        store: EpisodeTensorStore,
        rows: Sequence[Any],
        *,
        epoch: int,
        policy: str,
    ) -> tuple[dict[str, float], list[dict[str, Any]], pd.DataFrame]:
        metrics_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        base_hash = model_state_hash(learner.model)
        for row in rows:
            episode = store.load(row)
            adapted = learner.adapt(
                learner.model, (episode.support_features, episode.support_targets)
            )
            logits, buffer_audit = learner.predict_adapted(
                adapted, episode.query_features
            )
            probabilities = torch.softmax(logits, dim=1).numpy()
            prediction = probabilities.argmax(axis=1)
            truth = episode.query_targets.numpy()
            metrics = _classification_metrics(truth, prediction, probabilities)
            metrics_rows.append({
                "epoch": epoch,
                "policy": policy,
                "episode_id": str(row.episode_id),
                "subject_id": str(row.subject_id),
                "support_samples": len(episode.support_sample_ids),
                "query_samples": len(episode.query_sample_ids),
                "support_loss_before": adapted.support_losses[0],
                "support_loss_after": adapted.support_losses[-1],
                "query_buffers_changed": buffer_audit.buffers_changed,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            })
            frame = pd.DataFrame({
                "episode_id": str(row.episode_id),
                "subject_id": str(row.subject_id),
                "sample_id": episode.query_sample_ids,
                "y_true": truth,
                "y_pred": prediction,
            })
            for class_index in range(5):
                frame[f"proba_{class_index}"] = probabilities[:, class_index]
            prediction_frames.append(frame)
        if model_state_hash(learner.model) != base_hash:
            raise RuntimeError("Meta-validation changed the base model")
        metrics_frame = pd.DataFrame(metrics_rows)
        aggregate = {
            "macro_f1": float(metrics_frame["macro_f1"].mean()),
            "balanced_accuracy": float(metrics_frame["balanced_accuracy"].mean()),
            "query_loss": float(metrics_frame.get("query_loss", pd.Series(dtype=float)).mean()) if "query_loss" in metrics_frame else float("nan"),
        }
        return aggregate, metrics_rows, pd.concat(prediction_frames, ignore_index=True)

    def _train_fomaml_policy(
        self,
        policy: str,
        store: EpisodeTensorStore,
        train_rows: Sequence[Any],
        validation_rows: Sequence[Any],
        preregistration_hash: str,
        supervised_initial_hash: str,
        normalization: tuple[np.ndarray, np.ndarray],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        spec = self.config["fomaml"]
        adapter = self._build_adapter(supervised=False)
        initial_hash = _tensor_state_hash(adapter._initial_state)
        if initial_hash != supervised_initial_hash:
            raise RuntimeError("Supervised and FOMAML random initializations differ")
        steps_per_epoch = math.ceil(len(train_rows) / int(spec["meta_batch_size"]))
        learner = FirstOrderMAML(
            adapter.model,
            FOMAMLConfig(
                inner_steps=int(spec["inner_steps"]),
                inner_learning_rate=float(spec["inner_learning_rate"]),
                meta_learning_rate=float(spec["meta_learning_rate"]),
                episodes_per_meta_batch=int(spec["meta_batch_size"]),
                maximum_meta_steps=int(spec["maximum_meta_epochs"]) * steps_per_epoch,
                gradient_clip_norm=float(spec["gradient_clip_norm"]),
                buffer_policy=policy,
                device=self.device,
                seed=self.seed,
            ),
        )
        directory = self.output / "fomaml" / policy
        directory.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        episode_metrics: list[dict[str, Any]] = []
        gradient_rows: list[dict[str, Any]] = []
        best_metric = float("-inf")
        best_balanced = float("-inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        best_optimizer: dict[str, Any] | None = None
        epochs_without_improvement = 0
        started = time.perf_counter()
        for epoch in range(1, int(spec["maximum_meta_epochs"]) + 1):
            epoch_started = time.perf_counter()
            order = np.random.default_rng(self.seed + epoch).permutation(len(train_rows))
            step_results = []
            for batch_number, start in enumerate(
                range(0, len(order), int(spec["meta_batch_size"])), start=1
            ):
                selected = order[start:start + int(spec["meta_batch_size"])]
                loaded = [store.load(train_rows[int(index)]) for index in selected]
                step_result = learner.meta_train_step(loaded)
                step_results.append(step_result)
                gradient_rows.append({
                    "policy": policy,
                    "epoch": epoch,
                    "batch": batch_number,
                    "episodes": len(loaded),
                    "gradient_norm_before_clip": step_result.meta_gradient_norm_before_clip,
                    "gradient_norm_after_clip": step_result.meta_gradient_norm_after_clip,
                    "parameters_updated": step_result.parameters_updated,
                    "optimizer_state_finite": step_result.optimizer_state_finite,
                })
            validation, rows, _ = self._evaluate_adapted_episodes(
                learner, store, validation_rows, epoch=epoch, policy=policy
            )
            episode_metrics.extend(rows)
            improved = (
                validation["macro_f1"] > best_metric + 1e-12
                or (
                    abs(validation["macro_f1"] - best_metric) <= 1e-12
                    and validation["balanced_accuracy"] > best_balanced
                )
            )
            if improved:
                best_metric = validation["macro_f1"]
                best_balanced = validation["balanced_accuracy"]
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in learner.model.state_dict().items()
                }
                best_optimizer = deepcopy(learner.optimizer.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            history.append({
                "policy": policy,
                "epoch": epoch,
                "train_support_loss_before": float(np.mean([row.support_loss_before for row in step_results])),
                "train_support_loss_after": float(np.mean([row.support_loss_after for row in step_results])),
                "train_query_loss": float(np.mean([row.query_loss for row in step_results])),
                "meta_validation_subject_macro_f1": validation["macro_f1"],
                "meta_validation_subject_balanced_accuracy": validation["balanced_accuracy"],
                "is_best": improved,
                "epoch_time_seconds": time.perf_counter() - epoch_started,
            })
            if epochs_without_improvement >= int(spec["early_stopping_patience"]):
                break
        if best_state is None or best_optimizer is None:
            raise RuntimeError("FOMAML training produced no checkpoint")
        learner.model.load_state_dict(best_state, strict=True)
        checkpoint = directory / "fomaml_checkpoint.pt"
        payload = {
            "model_state_dict": best_state,
            "optimizer_state_dict": best_optimizer,
            "input_shape": tuple(self.config["dataset"]["input_shape"]),
            "num_classes": 5,
            "architecture_signature": self.config["model"]["architecture_signature"],
            "buffer_policy": policy,
            "episode_protocol_hash": self.config["protocol"]["expected_hash"],
            "preregistration_hash": preregistration_hash,
            "seed": self.seed,
            "best_epoch": best_epoch,
            "normalization_mean": normalization[0],
            "normalization_scale": normalization[1],
        }
        _atomic_torch_save(checkpoint, payload)
        pd.DataFrame(history).to_csv(directory / "fomaml_training_history.csv", index=False)
        pd.DataFrame(episode_metrics).to_parquet(directory / "fomaml_episode_metrics.parquet", index=False)
        pd.DataFrame(gradient_rows).to_csv(directory / "fomaml_gradient_audit.csv", index=False)
        manifest = {
            "checkpoint_hash": _sha256_file(checkpoint),
            "model_hash": model_state_hash(learner.model),
            "optimizer_state_hash": _object_hash(best_optimizer),
            "architecture_signature": self.config["model"]["architecture_signature"],
            "buffer_policy": policy,
            "episode_protocol_hash": self.config["protocol"]["expected_hash"],
            "preregistration_hash": preregistration_hash,
            "seed": self.seed,
            "best_epoch": best_epoch,
            "epochs_trained": len(history),
            "training_time_seconds": time.perf_counter() - started,
            "meta_validation_metrics": {
                "macro_f1": best_metric,
                "balanced_accuracy": best_balanced,
            },
            "outer_test_used": False,
            "query_used_for_meta_update": False,
        }
        _write_json(directory / "fomaml_checkpoint_manifest.json", manifest)
        return manifest, {"history": history, "episodes": episode_metrics, "gradients": gradient_rows}

    def _fresh_supervised(self, checkpoint: Path) -> TorchClassificationAdapter:
        adapter = self._build_adapter(supervised=True)
        return adapter.load(checkpoint)

    def _fresh_fomaml(self, checkpoint: Path) -> torch.nn.Module:
        adapter = self._build_adapter(supervised=False)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        adapter.model.load_state_dict(payload["model_state_dict"], strict=True)
        adapter.model.to(self.device)
        adapter.model.eval()
        return adapter.model

    def _predict_zero_shot(
        self, model: torch.nn.Module, features: torch.Tensor
    ) -> tuple[np.ndarray, bool]:
        before = model_state_hash(model)
        mode = model.training
        model.eval()
        with torch.no_grad():
            logits = model(features.to(self.device))
        model.train(mode)
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return probabilities, model_state_hash(model) == before

    def _outer_evaluation(
        self,
        store: EpisodeTensorStore,
        outer_rows: Sequence[Any],
        supervised_checkpoint: Path,
        fomaml_checkpoint: Path,
        policy: str,
        decision_manifest_path: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
        if not decision_manifest_path.exists():
            raise RuntimeError("Immutable pre-outer-test decision manifest is missing")
        supervised = self._fresh_supervised(supervised_checkpoint)
        supervised_hash = model_state_hash(supervised.model)
        fomaml_model = self._fresh_fomaml(fomaml_checkpoint)
        fomaml_hash = model_state_hash(fomaml_model)
        adaptation_config = FOMAMLConfig(
            inner_steps=int(self.config["fomaml"]["inner_steps"]),
            inner_learning_rate=float(self.config["fomaml"]["inner_learning_rate"]),
            meta_learning_rate=float(self.config["fomaml"]["meta_learning_rate"]),
            episodes_per_meta_batch=1,
            maximum_meta_steps=1,
            gradient_clip_norm=float(self.config["fomaml"]["gradient_clip_norm"]),
            buffer_policy=policy,
            device=self.device,
            seed=self.seed,
        )
        supervised_learner = FirstOrderMAML(supervised.model, adaptation_config)
        fomaml_learner = FirstOrderMAML(fomaml_model, adaptation_config)
        prediction_frames: list[pd.DataFrame] = []
        adaptation_rows: list[dict[str, Any]] = []
        for row in outer_rows:
            episode = store.load(row)
            zero_probabilities, zero_unchanged = self._predict_zero_shot(
                supervised.model, episode.query_features
            )
            supervised_adapted = supervised_learner.adapt(
                supervised.model, (episode.support_features, episode.support_targets)
            )
            supervised_logits, supervised_buffer = supervised_learner.predict_adapted(
                supervised_adapted, episode.query_features
            )
            supervised_probabilities = torch.softmax(supervised_logits, dim=1).numpy()
            fomaml_adapted = fomaml_learner.adapt(
                fomaml_model, (episode.support_features, episode.support_targets)
            )
            fomaml_logits, fomaml_buffer = fomaml_learner.predict_adapted(
                fomaml_adapted, episode.query_features
            )
            fomaml_probabilities = torch.softmax(fomaml_logits, dim=1).numpy()
            probability_sets = {
                "zero_shot_supervised": zero_probabilities,
                "supervised_full_model": supervised_probabilities,
                "selected_fomaml": fomaml_probabilities,
            }
            metadata = store.metadata.set_index("sample_id").loc[
                list(episode.query_sample_ids)
            ]
            for mode, probabilities in probability_sets.items():
                if not np.isfinite(probabilities).all() or not np.allclose(
                    probabilities.sum(axis=1), 1.0, atol=1e-5
                ):
                    raise RuntimeError(f"Invalid outer-test probabilities for {mode}")
                frame = pd.DataFrame({
                    "mode": mode,
                    "dataset": "emotiv_raw_eeg",
                    "task": "label_q5",
                    "model": "torch_eegnet",
                    "seed": self.seed,
                    "outer_fold": 1,
                    "episode_id": str(row.episode_id),
                    "subject_id": str(row.subject_id),
                    "sample_id": episode.query_sample_ids,
                    "record_id": metadata["record_id"].to_numpy(),
                    "source": metadata["source"].to_numpy(),
                    "y_true": episode.query_targets.numpy(),
                    "y_pred": probabilities.argmax(axis=1),
                })
                for class_index in range(5):
                    frame[f"proba_{class_index}"] = probabilities[:, class_index]
                prediction_frames.append(frame)
            adaptation_rows.append({
                "subject_id": str(row.subject_id),
                "episode_id": str(row.episode_id),
                "support_sample_ids": list(episode.support_sample_ids),
                "query_sample_ids": list(episode.query_sample_ids),
                "inner_steps": int(self.config["fomaml"]["inner_steps"]),
                "inner_learning_rate": float(self.config["fomaml"]["inner_learning_rate"]),
                "support_budget": len(episode.support_sample_ids),
                "query_budget": len(episode.query_sample_ids),
                "zero_shot_state_unchanged": zero_unchanged,
                "supervised_base_unchanged": model_state_hash(supervised.model) == supervised_hash,
                "fomaml_base_unchanged": model_state_hash(fomaml_model) == fomaml_hash,
                "supervised_query_buffers_changed": supervised_buffer.buffers_changed,
                "fomaml_query_buffers_changed": fomaml_buffer.buffers_changed,
                "episode_state_reused": False,
            })
        predictions = pd.concat(prediction_frames, ignore_index=True)
        key_columns = ["subject_id", "sample_id", "y_true"]
        reference = predictions.loc[predictions["mode"].eq(MODES[0]), key_columns].sort_values(key_columns).reset_index(drop=True)
        for mode in MODES[1:]:
            candidate = predictions.loc[predictions["mode"].eq(mode), key_columns].sort_values(key_columns).reset_index(drop=True)
            if not reference.equals(candidate):
                raise RuntimeError("Compared modes have different query IDs or y_true")
        subject_metrics = []
        for mode in MODES:
            frame = predictions.loc[predictions["mode"].eq(mode)]
            calculated = calculate_subject_metrics(
                frame, track="fomaml_diagnostic", model=mode, seed=self.seed
            ).rename(columns={"model": "mode"})
            probability_columns = [f"proba_{index}" for index in range(5)]
            entropy = frame.assign(
                entropy=-np.sum(
                    frame[probability_columns].to_numpy(dtype=float)
                    * np.log(np.clip(frame[probability_columns].to_numpy(dtype=float), 1e-12, 1.0)),
                    axis=1,
                )
            ).groupby("subject_id")["entropy"].mean()
            calculated["prediction_entropy"] = calculated["subject_id"].map(entropy)
            subject_metrics.append(calculated)
        subjects = pd.concat(subject_metrics, ignore_index=True)
        aggregate_rows = []
        confusion: dict[str, Any] = {}
        for mode in MODES:
            frame = predictions.loc[predictions["mode"].eq(mode)]
            window_metrics = _classification_metrics(
                frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(),
                frame[[f"proba_{index}" for index in range(5)]].to_numpy(),
            )
            confusion[mode] = window_metrics.pop("confusion_matrix")
            numeric = subjects.loc[subjects["mode"].eq(mode)].select_dtypes(include=[np.number])
            aggregate_rows.append({
                "mode": mode,
                "aggregation": "windows",
                **{key: value for key, value in window_metrics.items() if not isinstance(value, list)},
            })
            for aggregation, reducer in (("subject_mean", "mean"), ("subject_median", "median")):
                values = getattr(numeric, reducer)()
                aggregate_rows.append({"mode": mode, "aggregation": aggregation, **values.to_dict()})
        aggregates = pd.DataFrame(aggregate_rows)
        adaptation_audit = {
            "participants": adaptation_rows,
            "all_base_states_unchanged": all(
                row["zero_shot_state_unchanged"]
                and row["supervised_base_unchanged"]
                and row["fomaml_base_unchanged"]
                for row in adaptation_rows
            ),
            "query_buffers_changed": False,
            "participant_state_reuse": False,
        }
        return predictions, subjects, aggregates, confusion, adaptation_audit

    def run(self) -> dict[str, Any]:
        seed_torch(self.seed)
        architecture = self._architecture_audit()
        protocol, episodes, protocol_audit = self._load_protocol()
        preregistration_hash = self._preregister(protocol, episodes, architecture)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "protocol").mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.episode_path, self.output / "protocol/episode_index.parquet")
        shutil.copy2(self.episode_path, self.output / "episode_index.parquet")
        shutil.copy2(self.errors_path, self.output / "errors.csv")
        protocol_manifest = {
            "protocol": protocol,
            "audit": protocol_audit,
            "source_hashes": self.source_hashes,
            "preregistration_hash": preregistration_hash,
        }
        _write_json(self.output / "protocol/protocol_manifest.json", protocol_manifest)
        _write_json(self.output / "protocol_manifest.json", protocol_manifest)

        data, metadata = self._load_data()
        alignment = audit_raw_episode_alignment(metadata, episodes)
        if not alignment["valid"]:
            self._write_alignment_blocker(
                alignment, protocol_audit, preregistration_hash
            )
            raise RuntimeError(
                "Task 8X blocked before training: preregistered task-8F sample "
                "IDs are absent from the raw-deduplicated EEG cache"
            )
        train_rows = _episode_rows(episodes, "meta_train")
        validation_rows = _episode_rows(episodes, "meta_validation")
        outer_rows = _episode_rows(episodes, "outer_test")
        train_ids = self._ids(train_rows, "support_sample_ids")
        train_ids = sorted(set(train_ids) | set(self._ids(train_rows, "query_sample_ids")))
        validation_ids = self._ids(validation_rows, "query_sample_ids")
        mean, scale = self._normalization(data, metadata, train_ids)
        store = EpisodeTensorStore(data, metadata, mean, scale)

        supervised, supervised_manifest = self._train_supervised(
            data, metadata, train_ids, validation_ids, outer_rows, preregistration_hash
        )
        if not np.allclose(supervised.feature_mean_, mean, atol=1e-6) or not np.allclose(
            supervised.feature_scale_, scale, atol=1e-5
        ):
            raise RuntimeError("Supervised and FOMAML normalization states differ")

        policy_manifests: dict[str, Any] = {}
        policy_runtime: dict[str, Any] = {}
        for policy in self.config["fomaml"]["buffer_policies"]:
            manifest, runtime = self._train_fomaml_policy(
                str(policy), store, train_rows, validation_rows,
                preregistration_hash, supervised_manifest["initial_model_hash"],
                (mean, scale),
            )
            policy_manifests[str(policy)] = manifest
            policy_runtime[str(policy)] = runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        selection_inputs = {
            policy: {
                "macro_f1": manifest["meta_validation_metrics"]["macro_f1"],
                "balanced_accuracy": manifest["meta_validation_metrics"]["balanced_accuracy"],
            }
            for policy, manifest in policy_manifests.items()
        }
        selection = select_buffer_policy(
            selection_inputs,
            tie_threshold=float(self.config["policy_selection"]["frozen_global_tie_threshold"]),
        )
        selected_policy = selection["selected_policy"]
        selection["selected_checkpoint_hash"] = policy_manifests[selected_policy]["checkpoint_hash"]
        selection["preregistration_hash"] = preregistration_hash
        selection_directory = self.output / "policy_selection"
        _write_json(selection_directory / "policy_selection.json", selection)
        _write_json(self.output / "policy_selection.json", selection)

        combined_history = pd.concat(
            [pd.DataFrame(value["history"]) for value in policy_runtime.values()],
            ignore_index=True,
        )
        combined_episodes = pd.concat(
            [pd.DataFrame(value["episodes"]) for value in policy_runtime.values()],
            ignore_index=True,
        )
        combined_gradients = pd.concat(
            [pd.DataFrame(value["gradients"]) for value in policy_runtime.values()],
            ignore_index=True,
        )
        combined_history.to_csv(self.output / "fomaml_training_history.csv", index=False)
        combined_episodes.to_parquet(self.output / "fomaml_episode_metrics.parquet", index=False)
        combined_gradients.to_csv(self.output / "fomaml_gradient_audit.csv", index=False)
        selected_directory = self.output / "fomaml" / selected_policy
        shutil.copy2(selected_directory / "fomaml_checkpoint.pt", self.output / "fomaml_checkpoint.pt")
        shutil.copy2(selected_directory / "fomaml_checkpoint_manifest.json", self.output / "fomaml_checkpoint_manifest.json")

        pre_outer_hashes = {
            "preregistration": _sha256_file(self.output / "experiment_preregistration.json"),
            "protocol": _sha256_file(self.protocol_path),
            "episodes": _sha256_file(self.episode_path),
            "supervised_checkpoint": _sha256_file(self.output / "supervised_checkpoint.pt"),
            "fomaml_checkpoint": _sha256_file(self.output / "fomaml_checkpoint.pt"),
        }
        decision_manifest = {
            "selection_complete_before_outer_test": True,
            "selected_policy": selected_policy,
            "supervised_checkpoint_hash": pre_outer_hashes["supervised_checkpoint"],
            "fomaml_checkpoint_hash": pre_outer_hashes["fomaml_checkpoint"],
            "inner_steps": self.config["fomaml"]["inner_steps"],
            "inner_learning_rate": self.config["fomaml"]["inner_learning_rate"],
            "support_budget": self.config["protocol"]["support_budget"],
            "query_budget": self.config["protocol"]["query_budget"],
            "outer_test_used_for_selection": False,
            "hashes": pre_outer_hashes,
        }
        decision_manifest_path = self.output / "policy_selection/pre_outer_test_decision.json"
        _write_json(decision_manifest_path, decision_manifest)

        predictions, subject_metrics, aggregates, confusion, adaptation_audit = self._outer_evaluation(
            store, outer_rows,
            self.output / "supervised_checkpoint.pt",
            self.output / "fomaml_checkpoint.pt",
            selected_policy,
            decision_manifest_path,
        )
        outer_directory = self.output / "outer_test"
        outer_directory.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(self.output / "outer_test_predictions.parquet", index=False)
        subject_metrics.to_csv(self.output / "outer_test_subject_metrics.csv", index=False)
        aggregates.to_csv(self.output / "outer_test_aggregate_metrics.csv", index=False)
        for mode in MODES:
            mode_directory = outer_directory / {
                "zero_shot_supervised": "zero_shot",
                "supervised_full_model": "supervised_full_model",
                "selected_fomaml": "fomaml",
            }[mode]
            mode_directory.mkdir(parents=True, exist_ok=True)
            predictions.loc[predictions["mode"].eq(mode)].to_parquet(
                mode_directory / "predictions.parquet", index=False
            )
            subject_metrics.loc[subject_metrics["mode"].eq(mode)].to_csv(
                mode_directory / "subject_metrics.csv", index=False
            )
        _write_json(self.output / "confusion_matrices.json", confusion)
        _write_json(self.output / "adaptation_audit.json", adaptation_audit)
        _write_json(self.output / "buffer_audit.json", {
            "selected_policy": selected_policy,
            "query_buffers_changed": False,
            "original_models_unchanged": adaptation_audit["all_base_states_unchanged"],
            "support_local_query_frozen": True,
        })

        primary = paired_subject_comparison(
            subject_metrics, "selected_fomaml", "supervised_full_model"
        )
        secondary = paired_subject_comparison(
            subject_metrics, "selected_fomaml", "zero_shot_supervised"
        )
        comparisons = {"primary": primary, "secondary": secondary}
        _write_json(self.output / "paired_comparison.json", comparisons)
        final_decision = apply_decision_rule(primary, self.config["decision_rule"])
        final_decision.update({
            "selected_policy": selected_policy,
            "one_outer_fold": True,
            "one_seed": True,
            "result_status": "diagnostic",
        })
        _write_json(self.output / "decision.json", final_decision)

        post_hashes = {
            "preregistration": _sha256_file(self.output / "experiment_preregistration.json"),
            "protocol": _sha256_file(self.protocol_path),
            "episodes": _sha256_file(self.episode_path),
            "supervised_checkpoint": _sha256_file(self.output / "supervised_checkpoint.pt"),
            "fomaml_checkpoint": _sha256_file(self.output / "fomaml_checkpoint.pt"),
        }
        leakage_audit = {
            **protocol_audit,
            "source_hashes_before": self.source_hashes,
            "source_hashes_after": {
                "protocol_manifest": _sha256_file(self.protocol_path),
                "episode_index": _sha256_file(self.episode_path),
                "errors": _sha256_file(self.errors_path),
            },
            "pre_outer_hashes": pre_outer_hashes,
            "post_outer_hashes": post_hashes,
            "source_manifests_unchanged": self.source_hashes == {
                "protocol_manifest": _sha256_file(self.protocol_path),
                "episode_index": _sha256_file(self.episode_path),
                "errors": _sha256_file(self.errors_path),
            },
            "checkpoints_unchanged_during_outer_test": pre_outer_hashes == post_hashes,
            "query_ids_and_y_true_equal_across_modes": True,
            "outer_test_used_for_selection": False,
            "outer_test_used_for_training": False,
            "query_used_for_adaptation": False,
        }
        _write_json(self.output / "leakage_audit.json", leakage_audit)

        summary = {
            "schema_version": SCHEMA_VERSION,
            "result_status": "diagnostic",
            "device": self.device,
            "preregistration_hash": preregistration_hash,
            "protocol_hash": protocol["protocol_hash"],
            "architecture_signature": architecture["row"]["architecture_signature"],
            "subjects": {
                "meta_train": len(protocol["meta_train_subjects"]),
                "meta_validation": len(protocol["meta_validation_subjects"]),
                "outer_test": len(protocol["outer_test_subjects"]),
                "evaluated_outer_test": len(outer_rows),
            },
            "episodes": protocol["episode_counts"],
            "supervised": supervised_manifest,
            "fomaml": policy_manifests,
            "policy_selection": selection,
            "paired_comparison": comparisons,
            "decision": final_decision,
            "leakage_safe": all([
                leakage_audit["source_manifests_unchanged"],
                leakage_audit["checkpoints_unchanged_during_outer_test"],
                adaptation_audit["all_base_states_unchanged"],
            ]),
            "real_eeg_training_performed": True,
            "outer_folds_run": [1],
            "seeds_run": [42],
        }
        _write_json(self.output / "diagnostic_summary.json", summary)
        report = (
            "# FOMAML label_q5 diagnostic\n\n"
            f"- Status: `{final_decision['status']}` (diagnostic).\n"
            f"- Selected BatchNorm policy: `{selected_policy}`.\n"
            f"- Outer-test participants: {len(outer_rows)}.\n"
            f"- Mean subject macro-F1 delta versus supervised full-model: "
            f"{primary['mean_delta_macro_f1']:.6f}.\n"
            f"- Mean subject balanced-accuracy delta: "
            f"{primary['mean_delta_balanced_accuracy']:.6f}.\n"
            f"- Wins/losses/ties: {primary['macro_f1_wins']}/"
            f"{primary['macro_f1_losses']}/{primary['macro_f1_ties']}.\n"
            "- One fold and one seed only; no statistical-significance claim.\n"
        )
        (self.output / "diagnostic_report.md").write_text(report, encoding="utf-8")
        return summary


def run_fomaml_label_q5_diagnostic(
    config: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    return FOMAMLLabelQ5Diagnostic(config, repository_root=repository_root).run()
