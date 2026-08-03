"""One-fold CUDA diagnostic for Old_EEG to gpn_data DANN transfer.

The implementation deliberately consumes the immutable task-8Sh protocol
artifacts.  It does not rebuild the raw cache, outer fold, domain partitions,
or source-validation split.  Target-test tensors are guarded by an explicit
lock until both selected checkpoint hashes have been written.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from bench.analysis.subject_metrics import calculate_subject_metrics
from bench.datasets.datasets_registry import get_dataset
from bench.experiments.fomaml_label_q5_diagnostic import (
    _atomic_torch_save,
    _classification_metrics,
    _git_head,
    _jsonable,
    _sha256_file,
    _tensor_state_hash,
    _write_json,
    prepare_preregistration,
    resolve_device,
)
from bench.meta.production import audit_architectures
from model_zoo.DL.adapter import TorchClassificationAdapter, seed_torch
from model_zoo.DL.dann import DANNModule, DANNObjective
from model_zoo.DL.encoder import require_encoder_model
from model_zoo.factory import build_model


SCHEMA_VERSION = "dann-label-q5-raw-diagnostic-v1"
FORBIDDEN_TARGET_BATCH_KEYS = frozenset(
    {"label_q5", "target", "targets", "y", "task_label", "task_labels"}
)
MODES = ("source_only_matched", "dann")


def logistic_grl_alpha(progress: float) -> float:
    """Return the immutable task-8Sh logistic GRL coefficient."""
    value = float(progress)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("progress must be finite and in [0, 1]")
    return float(2.0 / (1.0 + math.exp(-10.0 * value)) - 1.0)


def enforce_target_batch_firewall(batch: Mapping[str, Any]) -> None:
    """Reject any target-training batch that can expose a task label."""
    forbidden = sorted(FORBIDDEN_TARGET_BATCH_KEYS & set(batch))
    required = {"eeg", "domain_label", "sample_id", "subject_id", "record_group_id"}
    missing = sorted(required - set(batch))
    if forbidden:
        raise RuntimeError(f"Target-label firewall rejected fields: {forbidden}")
    if missing:
        raise RuntimeError(f"Target batch is missing audit fields: {missing}")


class TargetTestLock:
    """Count and reject target-test tensor reads until an unlock hash exists."""

    def __init__(self) -> None:
        self.is_unlocked = False
        self.unlock_hash: str | None = None
        self.reads_before_unlock = 0
        self.reads_after_unlock = 0

    def require_access(self) -> None:
        if not self.is_unlocked:
            self.reads_before_unlock += 1
            raise RuntimeError("Target-test tensors are locked until checkpoints are fixed")
        self.reads_after_unlock += 1

    def unlock(self, manifest_hash: str) -> None:
        digest = str(manifest_hash)
        if len(digest) != 64:
            raise ValueError("Unlock manifest hash must be SHA-256")
        self.unlock_hash = digest
        self.is_unlocked = True


def deterministic_batch_plan(
    n_samples: int, batch_size: int, steps: int, seed: int
) -> list[list[int]]:
    """Create a deterministic shuffled loader plan with exact cycling."""
    if min(int(n_samples), int(batch_size), int(steps)) <= 0:
        raise ValueError("n_samples, batch_size and steps must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(int(n_samples), generator=generator).tolist()
    batches = [order[start : start + int(batch_size)] for start in range(0, len(order), int(batch_size))]
    return [list(batch) for batch in list(item for _, item in zip(range(int(steps)), cycle(batches)))]


def batch_plan_hash(plan: Sequence[Sequence[int]], sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for batch in plan:
        digest.update("\x1f".join(str(sample_ids[index]) for index in batch).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _global_norm(parameters: Iterable[nn.Parameter]) -> float:
    gradients = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return float(torch.sqrt(sum(torch.sum(value * value) for value in gradients)).item())


def _autograd_norm(loss: Tensor, parameters: Sequence[nn.Parameter]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [value.detach().float() for value in gradients if value is not None]
    if not values:
        return 0.0
    result = torch.sqrt(sum(torch.sum(value * value) for value in values))
    if not torch.isfinite(result):
        raise RuntimeError("Gradient decomposition produced a non-finite norm")
    return float(result.item())


def apply_dann_decision_rule(
    comparison: Mapping[str, Any], rule: Mapping[str, Any], *, unstable: bool = False,
    gradient_suppression: bool = False,
) -> dict[str, Any]:
    macro = float(comparison["mean_delta_macro_f1"])
    balanced = float(comparison["mean_delta_balanced_accuracy"])
    wins = int(comparison["macro_f1_wins"])
    if unstable or (gradient_suppression and macro < 0):
        status = "do_not_proceed"
    elif (
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
    elif (macro < 0 and balanced < 0) or wins <= int(rule["do_not_proceed_max_wins"]):
        status = "do_not_proceed"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "observed": _jsonable(comparison),
        "rule": _jsonable(rule),
        "numerical_instability": bool(unstable),
        "systematic_domain_gradient_suppression_with_quality_drop": bool(gradient_suppression and macro < 0),
        "statistical_significance_claimed": False,
    }


def paired_dann_comparison(subject_metrics: pd.DataFrame) -> dict[str, Any]:
    columns = ["subject_id", "macro_f1", "balanced_accuracy", "ordinal_mae"]
    candidate = subject_metrics.loc[subject_metrics["mode"].eq("dann"), columns].set_index("subject_id")
    reference = subject_metrics.loc[subject_metrics["mode"].eq("source_only_matched"), columns].set_index("subject_id")
    if len(candidate) != 8 or set(candidate.index) != set(reference.index):
        raise ValueError("Paired DANN comparison requires the same eight subjects")
    paired = candidate.join(reference, lsuffix="_dann", rsuffix="_source_only")
    for metric in ("macro_f1", "balanced_accuracy", "ordinal_mae"):
        paired[f"delta_{metric}"] = paired[f"{metric}_dann"] - paired[f"{metric}_source_only"]
    delta = paired["delta_macro_f1"].to_numpy(float)
    tolerance = 1e-12
    rng = np.random.default_rng(42)
    bootstrap = np.asarray([
        rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(10000)
    ])
    return {
        "candidate": "dann",
        "reference": "source_only_matched",
        "comparison_unit": "subject",
        "n_subjects": 8,
        "macro_f1_wins": int((delta > tolerance).sum()),
        "macro_f1_losses": int((delta < -tolerance).sum()),
        "macro_f1_ties": int((np.abs(delta) <= tolerance).sum()),
        "mean_delta_macro_f1": float(paired["delta_macro_f1"].mean()),
        "median_delta_macro_f1": float(paired["delta_macro_f1"].median()),
        "mean_delta_balanced_accuracy": float(paired["delta_balanced_accuracy"].mean()),
        "median_delta_balanced_accuracy": float(paired["delta_balanced_accuracy"].median()),
        "mean_delta_ordinal_mae": float(paired["delta_ordinal_mae"].mean()),
        "median_delta_ordinal_mae": float(paired["delta_ordinal_mae"].median()),
        "bootstrap_macro_f1_mean_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "bootstrap_seed": 42,
        "bootstrap_resamples": 10000,
        "low_power_warning": "Only eight target-test participants; descriptive diagnostic, not a significance test.",
        "subjects": paired.reset_index().to_dict("records"),
    }


class _RawPartitionDataset(Dataset):
    def __init__(
        self, raw: Any, positions: np.ndarray, metadata: pd.DataFrame, *,
        include_task_label: bool, domain_label: int, lock: TargetTestLock | None = None,
    ) -> None:
        self.raw = raw
        self.positions = np.asarray(positions, dtype=np.int64)
        self.metadata = metadata.iloc[self.positions].reset_index(drop=True)
        self.include_task_label = bool(include_task_label)
        self.domain_label = int(domain_label)
        self.lock = lock

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.lock is not None:
            self.lock.require_access()
        row = self.metadata.iloc[int(index)]
        result: dict[str, Any] = {
            "eeg": torch.from_numpy(np.ascontiguousarray(self.raw[int(self.positions[index])], dtype=np.float32)),
            "domain_label": self.domain_label,
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            "record_group_id": str(row["record_group_id"]),
        }
        if self.include_task_label:
            result["task_label"] = int(row["label_q5"])
        return result


@dataclass
class _ModeState:
    name: str
    model: nn.Module
    optimizer: torch.optim.Optimizer
    best_epoch: int = 0
    best_macro_f1: float = -math.inf
    best_balanced_accuracy: float = -math.inf
    stale_epochs: int = 0
    best_state: dict[str, Tensor] | None = None
    updates: int = 0


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute() or bool(len(value) > 2 and value[1:3] in {":/", ":\\"})
    return False


class DANNLabelQ5RawDiagnostic:
    """Strict one-direction, one-fold, one-seed diagnostic orchestrator."""

    def __init__(self, config: Mapping[str, Any], *, repository_root: Path, output_dir: Path | None = None) -> None:
        validate_dann_diagnostic_config(config)
        self.config = deepcopy(dict(config))
        self.root = repository_root
        self.output = output_dir or self.root / str(config["output_dir"])
        self.seed = int(config["seed"])
        self.device = resolve_device(str(config["device"]))
        if not self.device.startswith("cuda"):
            raise RuntimeError("Task 8Shch requires CUDA")
        self.lock = TargetTestLock()
        self.gradient_steps_started = 0
        self.paths = {
            key: self.root / str(value)
            for key, value in config["protocol"].items()
            if key.endswith(("manifest", "reference", "audit", "file", "preregistration"))
        }
        self.paths["raw_universe_manifest"] = self.root / str(config["dataset"]["raw_universe_manifest"])
        self.immutable_before = self._verify_source_files()

    def _verify_source_files(self) -> dict[str, str]:
        expected = {
            "protocol_manifest": self.config["protocol"]["protocol_manifest_sha256"],
            "protocol_hash_file": self.config["protocol"]["protocol_hash_file_sha256"],
            "source_validation_manifest": self.config["protocol"]["source_validation_manifest_sha256"],
            "target_unlabeled_manifest": self.config["protocol"]["target_unlabeled_manifest_sha256"],
            "target_test_reference": self.config["protocol"]["target_test_reference_sha256"],
            "architecture_audit": self.config["protocol"]["architecture_audit_sha256"],
            "disabled_preregistration": self.config["protocol"]["disabled_preregistration_sha256"],
            "raw_universe_manifest": self.config["dataset"]["raw_universe_manifest_sha256"],
        }
        observed = {name: _sha256_file(self.paths[name]) for name in expected}
        if observed != expected:
            raise RuntimeError(f"Immutable DANN protocol artifact changed: {observed}")
        protocol_hashes = json.loads(self.paths["protocol_hash_file"].read_text(encoding="utf-8"))
        if protocol_hashes["protocol_hash"] != self.config["protocol"]["expected_protocol_hash"]:
            raise RuntimeError("DANN protocol hash changed")
        if protocol_hashes["primary_candidate_hash"] != self.config["protocol"]["expected_candidate_hash"]:
            raise RuntimeError("DANN primary candidate hash changed")
        return observed

    def _load_protocol(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        protocol = json.loads(self.paths["protocol_manifest"].read_text(encoding="utf-8"))
        source = json.loads(self.paths["source_validation_manifest"].read_text(encoding="utf-8"))
        target = json.loads(self.paths["target_unlabeled_manifest"].read_text(encoding="utf-8"))
        test = json.loads(self.paths["target_test_reference"].read_text(encoding="utf-8"))
        primary = protocol["primary_direction"]
        checks = {
            "protocol_hash": protocol["protocol_hash"] == self.config["protocol"]["expected_protocol_hash"],
            "candidate_hash": primary["candidate_protocol_hash"] == self.config["protocol"]["expected_candidate_hash"],
            "outer_split_hash": primary["outer_split_hash"] == self.config["protocol"]["expected_outer_split_hash"],
            "direction": source["direction_id"] == self.config["protocol"]["direction"] == target["direction_id"] == test["direction_id"],
            "subject_policy": primary["subject_policy"] == self.config["protocol"]["subject_policy"],
            "counts": [source["source_task_train"]["samples"], source["source_validation"]["samples"], target["samples"], test["samples"]] == [3753, 1456, 18555, 4973],
        }
        if not all(checks.values()):
            raise RuntimeError(f"Immutable DANN protocol contract mismatch: {checks}")
        return source, target, test

    def _architecture_and_models(self) -> tuple[TorchClassificationAdapter, TorchClassificationAdapter, DANNModule, dict[str, Any]]:
        production = json.loads((self.root / "experiments/meta_learning/fomaml_production_contract.json").read_text(encoding="utf-8"))
        rows, _, _ = audit_architectures(production, repository_root=self.root)
        row = next(item for item in rows if item["model_id"] == "torch_eegnet:canonical")
        model_config = yaml.safe_load((self.root / str(self.config["dataset"]["config"])).read_text(encoding="utf-8"))
        params = dict(model_config["models"]["torch_eegnet"]["params"])
        params.update({"device": self.device, "random_state": self.seed, "standardize": False})
        seed_torch(self.seed)
        initial = build_model("torch_eegnet", "classification", tuple(self.config["dataset"]["input_shape"]), 5, params)
        if not isinstance(initial, TorchClassificationAdapter):
            raise TypeError("Factory did not return TorchClassificationAdapter")
        initial_state = {name: value.detach().cpu().clone() for name, value in initial.model.state_dict().items()}
        initial_hash = _tensor_state_hash(initial_state)
        _atomic_torch_save(self.output / "initial_model_state.pt", {"model_state_dict": initial_state})
        _atomic_torch_save(self.output / "initial_encoder_task_state.pt", {"model_state_dict": initial_state})

        def clone_adapter() -> TorchClassificationAdapter:
            adapter = build_model("torch_eegnet", "classification", tuple(self.config["dataset"]["input_shape"]), 5, params)
            if not isinstance(adapter, TorchClassificationAdapter):
                raise TypeError("Factory did not return TorchClassificationAdapter")
            adapter.model.load_state_dict(initial_state, strict=True)
            adapter.model.to(self.device)
            return adapter

        source_adapter = clone_adapter()
        dann_adapter = clone_adapter()
        seed_torch(self.seed + 1000)
        model_cfg = self.config["model"]
        dann = DANNModule(
            dann_adapter.model,
            n_domains=int(model_cfg["n_domains"]),
            gradient_reversal_alpha=1.0,
            domain_hidden_dims=model_cfg["domain_hidden_dims"],
            domain_dropout=float(model_cfg["domain_dropout"]),
        ).to(self.device)
        live = {
            "architecture_signature": row["architecture_signature"],
            "input_shape": list(self.config["dataset"]["input_shape"]),
            "num_outputs": 5,
            "latent_dim": int(require_encoder_model(source_adapter.model).latent_dim),
            "task_parameter_count": sum(p.numel() for p in source_adapter.model.parameters()),
            "domain_parameter_count": sum(p.numel() for p in dann.domain_discriminator.parameters()),
            "total_dann_parameter_count": sum(p.numel() for p in dann.parameters()),
            "initial_model_hash": initial_hash,
            "initial_states_identical": _tensor_state_hash(source_adapter.model.state_dict()) == _tensor_state_hash(dann.task_model.state_dict()) == initial_hash,
            "domain_head_separate": True,
            "production_audit_row": row,
        }
        expected = self.config["model"]
        required = {
            "architecture_signature": expected["architecture_signature"],
            "latent_dim": expected["latent_dim"],
            "task_parameter_count": expected["task_parameter_count"],
            "domain_parameter_count": expected["domain_parameter_count"],
            "total_dann_parameter_count": expected["total_dann_parameter_count"],
        }
        if any(live[key] != value for key, value in required.items()) or not live["initial_states_identical"]:
            raise RuntimeError(f"Production architecture or initialization mismatch: {live}")
        _write_json(self.output / "architecture_audit.json", live)
        _write_json(self.output / "initial_model_manifest.json", {
            "initial_model_hash": initial_hash,
            "state_file": "initial_model_state.pt",
            "source_only_and_dann_identical": True,
            "includes_encoder_task_head_and_batchnorm_buffers": True,
            "domain_head_initialized_separately_with_seed": self.seed + 1000,
        })
        return source_adapter, dann_adapter, dann, live

    def _preregister(self, source: Mapping[str, Any], target: Mapping[str, Any], test: Mapping[str, Any], architecture: Mapping[str, Any]) -> str:
        training = self.config["training"]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.config["experiment_id"],
            "execution_enabled": True,
            "repository_commit": _git_head(self.root),
            "result_status": "diagnostic",
            "protocol_hash": self.config["protocol"]["expected_protocol_hash"],
            "primary_candidate_hash": self.config["protocol"]["expected_candidate_hash"],
            "raw_universe_hash": self.config["dataset"]["raw_universe_hash"],
            "outer_split_hash": self.config["protocol"]["expected_outer_split_hash"],
            "outer_fold": 1,
            "direction": self.config["protocol"]["direction"],
            "subject_policy": self.config["protocol"]["subject_policy"],
            "partitions": {
                "source_task_train": {"samples": source["source_task_train"]["samples"], "subject_ids": source["source_task_train"]["subject_ids"], "sample_ids": source["source_task_train"]["sample_ids"]},
                "source_validation": {"samples": source["source_validation"]["samples"], "subject_ids": source["source_validation"]["subject_ids"], "sample_ids": source["source_validation"]["sample_ids"]},
                "target_train_unlabeled": {"samples": target["samples"], "subject_ids": target["subject_ids"], "sample_ids": target["sample_ids"]},
                "target_test": {"samples": test["samples"], "subject_ids": test["subject_ids"], "sample_ids": test["sample_ids"]},
            },
            "model": {key: architecture[key] for key in ("architecture_signature", "input_shape", "num_outputs", "latent_dim", "task_parameter_count", "domain_parameter_count", "total_dann_parameter_count", "initial_model_hash")},
            "seed": self.seed,
            "device": self.device,
            "device_name": torch.cuda.get_device_name(torch.device(self.device)),
            "batch_sizes": {"source": training["source_batch_size"], "target": training["target_batch_size"]},
            "steps_per_epoch": training["steps_per_epoch"],
            "maximum_epochs": training["maximum_epochs"],
            "early_stopping": {"patience": training["early_stopping_patience"], "policy": training["matched_early_stop"]},
            "optimizer": training["optimizer"],
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "learning_rate_schedule": training["learning_rate_schedule"],
            "gradient_reversal_alpha_schedule": self.config["schedule"]["gradient_reversal_formula"],
            "progress_schedule": self.config["schedule"]["progress_formula"],
            "domain_loss_lambda_schedule": {"name": "constant", "value": self.config["schedule"]["domain_loss_lambda"]},
            "gradient_clipping": training["gradient_clip_norm"],
            "checkpoint_criterion": {"primary": training["checkpoint_primary"], "secondary": training["checkpoint_secondary"], "partition": "source_validation"},
            "metrics": ["participant_macro_f1", "participant_balanced_accuracy", "accuracy", "weighted_f1", "macro_precision", "macro_recall", "ordinal_mae", "quadratic_weighted_kappa", "per_class_recall", "prediction_entropy"],
            "decision_rule": self.config["decision_rule"],
            "target_train_task_labels_accessible": False,
            "target_test_locked": True,
            "source_artifact_hashes": self.immutable_before,
        }
        if _contains_absolute_path(payload):
            raise RuntimeError("Executable preregistration contains an absolute path")
        path = self.output / "preregistration/experiment_preregistration.json"
        digest = prepare_preregistration(path, payload)
        _write_json(self.output / "preregistration/preregistration_hash.json", {"algorithm": "sha256", "sha256": digest, "parameters_frozen_before_gradient_step": True})
        if self.gradient_steps_started:
            raise RuntimeError("Preregistration was not created before gradient steps")
        return digest

    def _load_data(self) -> tuple[Any, pd.DataFrame]:
        document = yaml.safe_load((self.root / str(self.config["dataset"]["config"])).read_text(encoding="utf-8"))
        dataset_config = dict(document["datasets"]["emotiv_raw_eeg"])
        dataset_config["raw_preprocessing"] = document["raw_preprocessing"]
        data = get_dataset("emotiv_raw_eeg", dataset_config).load()
        row = data.row_metadata
        metadata = pd.DataFrame({
            "sample_id": np.asarray(data.sample_ids).astype(str),
            "subject_id": np.asarray(data.subject_ids).astype(str),
            "record_id": np.asarray(data.record_ids).astype(str),
            "record_group_id": np.asarray(row.get("record_group_id", data.record_ids)).astype(str),
            "label_q5": np.asarray(data.labels).astype(np.int64),
            "source": np.asarray(row["source"]).astype(str),
            "outer_fold": np.asarray(row["outer_fold"]).astype(int),
        })
        if len(metadata) != 30958 or tuple(data.data.shape[1:]) != (1, 14, 2560) or metadata["sample_id"].duplicated().any():
            raise RuntimeError("Canonical raw-deduplicated universe changed")
        return data, metadata

    @staticmethod
    def _positions(metadata: pd.DataFrame, ids: Sequence[str]) -> np.ndarray:
        lookup = pd.Series(np.arange(len(metadata), dtype=np.int64), index=metadata["sample_id"].astype(str))
        requested = pd.Index([str(value) for value in ids])
        missing = requested.difference(lookup.index)
        if len(missing):
            raise RuntimeError(f"Protocol sample IDs absent from raw universe: {missing[:5].tolist()}")
        return lookup.loc[requested].to_numpy(dtype=np.int64)

    def _partition_audit(self, metadata: pd.DataFrame, source: Mapping[str, Any], target: Mapping[str, Any], test: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        ids = {
            "source_train": source["source_task_train"]["sample_ids"],
            "source_validation": source["source_validation"]["sample_ids"],
            "target_train": target["sample_ids"],
            "target_test": test["sample_ids"],
        }
        positions = {key: self._positions(metadata, value) for key, value in ids.items()}
        frames = {key: metadata.iloc[value] for key, value in positions.items()}
        pairs = [(left, right) for i, left in enumerate(frames) for right in list(frames)[i + 1 :]]
        subject_overlap = {f"{a}__{b}": len(set(frames[a]["subject_id"]) & set(frames[b]["subject_id"])) for a, b in pairs}
        sample_overlap = {f"{a}__{b}": len(set(frames[a]["sample_id"]) & set(frames[b]["sample_id"])) for a, b in pairs}
        logical_overlap = {f"{a}__{b}": len(set(frames[a]["record_group_id"]) & set(frames[b]["record_group_id"])) for a, b in pairs}
        domains = {
            "source_train": sorted(frames["source_train"]["source"].unique().tolist()),
            "source_validation": sorted(frames["source_validation"]["source"].unique().tolist()),
            "target_train": sorted(frames["target_train"]["source"].unique().tolist()),
            "target_test": sorted(frames["target_test"]["source"].unique().tolist()),
        }
        audit = {
            "partition_counts": {key: len(value) for key, value in frames.items()},
            "subject_overlap": subject_overlap,
            "sample_overlap": sample_overlap,
            "logical_record_overlap": logical_overlap,
            "domains": domains,
            "all_overlaps_zero": not any(subject_overlap.values()) and not any(sample_overlap.values()) and not any(logical_overlap.values()),
            "target_labels_available_to_training_step": False,
            "target_test_reads_before_unlock": self.lock.reads_before_unlock,
            "raw_cache_rebuilt": False,
            "outer_split_rebuilt": False,
        }
        if not audit["all_overlaps_zero"] or domains != {"source_train": ["Old_EEG"], "source_validation": ["Old_EEG"], "target_train": ["gpn_data"], "target_test": ["gpn_data"]}:
            raise RuntimeError(f"DANN partition leakage or domain mismatch: {audit}")
        return positions, audit

    def _predict(self, model: nn.Module, dataset: Dataset, *, mode: str, partition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        model.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["eeg"].to(self.device, non_blocking=True))
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = probabilities.argmax(axis=1)
                labels = batch["task_label"].numpy().astype(int)
                for index in range(len(labels)):
                    row = {
                        "mode": mode, "partition": partition, "sample_id": str(batch["sample_id"][index]),
                        "subject_id": str(batch["subject_id"][index]), "record_group_id": str(batch["record_group_id"][index]),
                        "outer_fold": 1, "source": "gpn_data" if partition == "target_test" else "Old_EEG",
                        "y_true": int(labels[index]), "y_pred": int(predictions[index]),
                    }
                    row.update({f"proba_{class_id}": float(probabilities[index, class_id]) for class_id in range(5)})
                    rows.append(row)
        frame = pd.DataFrame(rows)
        probability = frame[[f"proba_{i}" for i in range(5)]].to_numpy(float)
        if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Prediction probabilities are invalid")
        metrics = _classification_metrics(frame["y_true"].to_numpy(int), frame["y_pred"].to_numpy(int), probability)
        return frame, metrics

    def _gradient_decomposition(self, model: DANNModule, source_batch: Mapping[str, Any], target_batch: Mapping[str, Any], *, alpha: float, lambda_domain: float, epoch: int) -> dict[str, Any]:
        before = _tensor_state_hash(model.state_dict())
        was_training = model.training
        model.eval()
        source_x = source_batch["eeg"].to(self.device)
        target_x = target_batch["eeg"].to(self.device)
        labels = source_batch["task_label"].to(self.device)
        domain_ids = torch.cat((source_batch["domain_label"], target_batch["domain_label"])).to(self.device)
        outputs = model(source_x, target_x, gradient_reversal_alpha=alpha)
        objective = DANNObjective(task_type="classification", lambda_domain=lambda_domain)(outputs, labels, domain_ids)
        encoder_parameters = list(require_encoder_model(model.task_model).encoder_parameters())  # type: ignore[attr-defined]
        task_norm = _autograd_norm(objective.task_loss, encoder_parameters)
        domain_norm = _autograd_norm(lambda_domain * objective.domain_loss, encoder_parameters)
        model.train(was_training)
        after = _tensor_state_hash(model.state_dict())
        if before != after:
            raise RuntimeError("Gradient decomposition changed model parameters or buffers")
        return {
            "epoch": epoch,
            "task_only_encoder_gradient_norm": task_norm,
            "weighted_domain_only_encoder_gradient_norm": domain_norm,
            "domain_to_task_encoder_gradient_ratio": float(domain_norm / max(task_norm, 1e-12)),
            "finite": bool(np.isfinite([task_norm, domain_norm]).all()),
            "optimizer_step_performed": False,
            "model_state_unchanged": True,
        }

    def _train(self, source_adapter: TorchClassificationAdapter, dann: DANNModule, raw: Any, metadata: pd.DataFrame, positions: Mapping[str, np.ndarray]) -> tuple[dict[str, _ModeState], dict[str, Any]]:
        training = self.config["training"]
        source_dataset = _RawPartitionDataset(raw, positions["source_train"], metadata, include_task_label=True, domain_label=1)
        target_dataset = _RawPartitionDataset(raw, positions["target_train"], metadata, include_task_label=False, domain_label=0)
        validation_dataset = _RawPartitionDataset(raw, positions["source_validation"], metadata, include_task_label=True, domain_label=1)
        source_model = source_adapter.model.to(self.device)
        states = {
            "source_only_matched": _ModeState("source_only_matched", source_model, torch.optim.AdamW(source_model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))),
            "dann": _ModeState("dann", dann, torch.optim.AdamW(dann.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))),
        }
        history: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
        schedule_rows: list[dict[str, Any]] = []
        domain_rows: list[dict[str, Any]] = []
        gradient_rows: list[dict[str, Any]] = []
        source_hash_rows: list[dict[str, Any]] = []
        total_steps = int(training["maximum_epochs"]) * int(training["steps_per_epoch"])
        start_time = time.perf_counter()
        for epoch in range(1, int(training["maximum_epochs"]) + 1):
            source_plan = deterministic_batch_plan(len(source_dataset), int(training["source_batch_size"]), int(training["steps_per_epoch"]), self.seed + epoch)
            target_plan = deterministic_batch_plan(len(target_dataset), int(training["target_batch_size"]), int(training["steps_per_epoch"]), self.seed + 10000 + epoch)
            source_hash = batch_plan_hash(source_plan, source_dataset.metadata["sample_id"].astype(str).tolist())
            source_hash_rows.append({"epoch": epoch, "source_only_batch_hash": source_hash, "dann_batch_hash": source_hash, "match": True})
            source_loaders = {
                mode: DataLoader(source_dataset, batch_sampler=source_plan, num_workers=0, pin_memory=bool(training["pin_memory"])) for mode in MODES
            }
            target_loader = DataLoader(target_dataset, batch_sampler=target_plan, num_workers=0, pin_memory=bool(training["pin_memory"]))
            target_batches = iter(target_loader)
            for mode in MODES:
                state = states[mode]
                state.model.train()
                task_numerator = 0.0
                task_denominator = 0
                domain_numerator = 0.0
                domain_denominator = 0
                domain_correct_source = domain_correct_target = 0
                domain_count_source = domain_count_target = 0
                step_start = time.perf_counter()
                target_batches = iter(target_loader) if mode == "dann" else None
                for local_step, source_batch in enumerate(source_loaders[mode]):
                    state.optimizer.zero_grad(set_to_none=True)
                    source_x = source_batch["eeg"].to(self.device, non_blocking=True)
                    source_y = source_batch["task_label"].to(self.device, non_blocking=True)
                    global_step = (epoch - 1) * int(training["steps_per_epoch"]) + local_step
                    progress = global_step / max(total_steps - 1, 1)
                    alpha = logistic_grl_alpha(progress)
                    if mode == "source_only_matched":
                        logits = state.model(source_x)
                        loss = nn.functional.cross_entropy(logits, source_y)
                        task_loss = loss
                    else:
                        assert target_batches is not None
                        target_batch = next(target_batches)
                        enforce_target_batch_firewall(target_batch)
                        if epoch == 1 and local_step == 0 and self.gradient_steps_started == 0:
                            enforce_target_batch_firewall(target_batch)
                        if local_step == 0:
                            gradient_rows.append(self._gradient_decomposition(dann, source_batch, target_batch, alpha=alpha, lambda_domain=float(self.config["schedule"]["domain_loss_lambda"]), epoch=epoch))
                        target_x = target_batch["eeg"].to(self.device, non_blocking=True)
                        domain_ids = torch.cat((source_batch["domain_label"], target_batch["domain_label"])).to(self.device)
                        outputs = dann(source_x, target_x, gradient_reversal_alpha=alpha)
                        result = DANNObjective(task_type="classification", lambda_domain=float(self.config["schedule"]["domain_loss_lambda"]))(outputs, source_y, domain_ids)
                        loss = result.total_loss
                        task_loss = result.task_loss
                        domain_loss = result.domain_loss
                        source_count = len(source_y)
                        domain_pred = outputs.domain_outputs.argmax(dim=1)
                        domain_correct_source += int((domain_pred[:source_count] == domain_ids[:source_count]).sum().item())
                        domain_correct_target += int((domain_pred[source_count:] == domain_ids[source_count:]).sum().item())
                        domain_count_source += source_count
                        domain_count_target += len(domain_ids) - source_count
                        domain_numerator += float(result.domain.numerator.detach().item())
                        domain_denominator += int(result.domain.denominator.detach().item())
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite {mode} objective")
                    loss.backward()
                    encoder = require_encoder_model(source_model if mode == "source_only_matched" else dann.task_model)
                    encoder_norm = _global_norm(encoder.encoder_parameters())  # type: ignore[attr-defined]
                    task_head_norm = _global_norm(encoder.get_output_head().parameters())
                    domain_head_norm = 0.0 if mode == "source_only_matched" else _global_norm(dann.domain_discriminator.parameters())
                    unclipped = float(torch.nn.utils.clip_grad_norm_(state.model.parameters(), float(training["gradient_clip_norm"])).item())
                    if not np.isfinite([encoder_norm, task_head_norm, domain_head_norm, unclipped]).all():
                        raise RuntimeError("Non-finite training gradient")
                    state.optimizer.step()
                    self.gradient_steps_started += 1
                    state.updates += 1
                    task_numerator += float(task_loss.detach().item()) * len(source_y)
                    task_denominator += len(source_y)
                    if mode == "dann":
                        schedule_rows.append({"epoch": epoch, "step": local_step + 1, "global_step": global_step, "progress": progress, "grl_alpha": alpha, "domain_lambda": float(self.config["schedule"]["domain_loss_lambda"]), "learning_rate": float(training["learning_rate"])})
                        domain_rows.append({"epoch": epoch, "step": local_step + 1, "task_loss": float(task_loss.detach().item()), "domain_loss": float(domain_loss.detach().item()), "weighted_domain_loss": float(domain_loss.detach().item()) * float(self.config["schedule"]["domain_loss_lambda"]), "encoder_total_gradient_norm": encoder_norm, "task_head_gradient_norm": task_head_norm, "domain_head_gradient_norm": domain_head_norm, "gradient_norm_before_clip": unclipped, "grl_alpha": alpha, "domain_lambda": float(self.config["schedule"]["domain_loss_lambda"]), "domain_accuracy_source": float((domain_pred[:source_count] == domain_ids[:source_count]).float().mean().item()), "domain_accuracy_target": float((domain_pred[source_count:] == domain_ids[source_count:]).float().mean().item()), "domain_accuracy_combined": float((domain_pred == domain_ids).float().mean().item())})
                task_model = source_model if mode == "source_only_matched" else dann.task_model
                validation_predictions, validation_metrics = self._predict(task_model, validation_dataset, mode=mode, partition="source_validation")
                macro = float(validation_metrics["macro_f1"])
                balanced = float(validation_metrics["balanced_accuracy"])
                improved = macro > state.best_macro_f1 + 1e-12 or (abs(macro - state.best_macro_f1) <= 1e-12 and balanced > state.best_balanced_accuracy + 1e-12)
                if improved:
                    state.best_epoch = epoch
                    state.best_macro_f1 = macro
                    state.best_balanced_accuracy = balanced
                    state.best_state = {name: value.detach().cpu().clone() for name, value in state.model.state_dict().items()}
                    state.stale_epochs = 0
                else:
                    state.stale_epochs += 1
                row = {
                    "epoch": epoch, "optimizer_updates": state.updates, "task_loss": task_numerator / task_denominator,
                    "source_validation_macro_f1": macro, "source_validation_balanced_accuracy": balanced,
                    "source_validation_accuracy": float(validation_metrics["accuracy"]), "best_epoch": state.best_epoch,
                    "is_best": improved, "stale_epochs": state.stale_epochs, "epoch_training_seconds": time.perf_counter() - step_start,
                }
                if mode == "dann":
                    row.update({"domain_loss": domain_numerator / domain_denominator, "domain_accuracy_source": domain_correct_source / domain_count_source, "domain_accuracy_target": domain_correct_target / domain_count_target})
                history[mode].append(row)
            if all(state.stale_epochs >= int(training["early_stopping_patience"]) for state in states.values()):
                break
        for mode, state in states.items():
            if state.best_state is None:
                raise RuntimeError(f"No checkpoint selected for {mode}")
            state.model.load_state_dict(state.best_state, strict=True)
            state.model.eval()
            directory = self.output / ("source_only" if mode == "source_only_matched" else "dann")
            pd.DataFrame(history[mode]).to_csv(directory / "training_history.csv", index=False)
            payload = (
                {"schema_version": 1, "model_state_dict": state.best_state, "metadata": {"mode": mode, "best_epoch": state.best_epoch}}
                if mode == "source_only_matched"
                else dann.checkpoint_payload(metadata={"mode": mode, "best_epoch": state.best_epoch})
            )
            _atomic_torch_save(directory / "checkpoint.pt", payload)
        if states["source_only_matched"].updates != states["dann"].updates:
            raise RuntimeError("Matched source optimizer update budget diverged")
        pd.DataFrame(domain_rows).to_csv(self.output / "dann/domain_metrics.csv", index=False)
        pd.DataFrame(gradient_rows).to_csv(self.output / "dann/gradient_audit.csv", index=False)
        pd.DataFrame(schedule_rows).to_csv(self.output / "dann/schedule_audit.csv", index=False)
        _write_json(self.output / "source_batch_audit.json", {"epochs": source_hash_rows, "all_hashes_match": True, "optimizer_updates_per_mode": states["dann"].updates})
        return states, {"epochs_trained": len(history["dann"]), "training_time_seconds": time.perf_counter() - start_time, "history": history}

    def _extended_subject_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for mode in MODES:
            selected = predictions.loc[predictions["mode"].eq(mode)].copy()
            base = calculate_subject_metrics(selected, track="dann_raw_diagnostic", model=mode, seed=self.seed)
            base["mode"] = mode
            for index, row in base.iterrows():
                subject = selected.loc[selected["subject_id"].eq(row["subject_id"])]
                truth = subject["y_true"].to_numpy(int)
                pred = subject["y_pred"].to_numpy(int)
                proba = subject[[f"proba_{i}" for i in range(5)]].to_numpy(float)
                base.loc[index, "macro_precision"] = precision_score(truth, pred, labels=np.arange(5), average="macro", zero_division=0)
                base.loc[index, "macro_recall"] = recall_score(truth, pred, labels=np.arange(5), average="macro", zero_division=0)
                base.loc[index, "prediction_entropy"] = float((-proba * np.log(np.clip(proba, 1e-12, 1))).sum(axis=1).mean())
                base.loc[index, "per_class_recall"] = json.dumps(recall_score(truth, pred, labels=np.arange(5), average=None, zero_division=0).tolist())
            frames.append(base)
        return pd.concat(frames, ignore_index=True)

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        for directory in (self.output / "source_only", self.output / "dann", self.output / "preregistration"):
            directory.mkdir(parents=True, exist_ok=True)
        errors = pd.DataFrame(columns=["stage", "code", "message"])
        errors.to_csv(self.output / "errors.csv", index=False)
        source, target, test = self._load_protocol()
        _write_json(self.output / "protocol_reference.json", {"protocol_hash": self.config["protocol"]["expected_protocol_hash"], "candidate_hash": self.config["protocol"]["expected_candidate_hash"], "disabled_preregistration_hash": self.config["protocol"]["disabled_preregistration_sha256"], "raw_universe_hash": self.config["dataset"]["raw_universe_hash"], "source_artifact_hashes": self.immutable_before})
        source_adapter, _, dann, architecture = self._architecture_and_models()
        preregistration_hash = self._preregister(source, target, test, architecture)
        data, metadata = self._load_data()
        positions, leakage = self._partition_audit(metadata, source, target, test)
        mean, scale = data.data[positions["source_train"]].compute_channel_statistics()
        raw = data.data.with_channel_normalization(mean, scale)
        _write_json(self.output / "normalization_stats.json", {"fit_partition": "source_task_train_only", "channel_mean": mean, "channel_scale": scale, "n_windows": len(positions["source_train"])})
        states, training_summary = self._train(source_adapter, dann, raw, metadata, positions)

        checkpoint_manifests: dict[str, Any] = {}
        validation_frames = []
        validation_dataset = _RawPartitionDataset(raw, positions["source_validation"], metadata, include_task_label=True, domain_label=1)
        for mode, state in states.items():
            directory_name = "source_only" if mode == "source_only_matched" else "dann"
            checkpoint_path = self.output / directory_name / "checkpoint.pt"
            checkpoint_hash = _sha256_file(checkpoint_path)
            task_model = state.model if mode == "source_only_matched" else dann.task_model
            predictions, metrics = self._predict(task_model, validation_dataset, mode=mode, partition="source_validation")
            predictions.to_parquet(self.output / directory_name / "source_validation_predictions.parquet", index=False)
            manifest = {"mode": mode, "checkpoint_sha256": checkpoint_hash, "best_epoch": state.best_epoch, "source_validation_metrics": metrics, "selection_primary": "macro_f1", "selection_secondary": "balanced_accuracy", "target_data_used_for_selection": False, "domain_accuracy_used_for_selection": False, "optimizer_updates": state.updates}
            _write_json(self.output / directory_name / "checkpoint_manifest.json", manifest)
            checkpoint_manifests[mode] = manifest
            validation_frames.append(predictions)

        unlock = {
            "protocol_hash": self.config["protocol"]["expected_protocol_hash"],
            "preregistration_hash": preregistration_hash,
            "source_only_checkpoint_hash": checkpoint_manifests["source_only_matched"]["checkpoint_sha256"],
            "dann_checkpoint_hash": checkpoint_manifests["dann"]["checkpoint_sha256"],
            "best_epochs": {mode: value.best_epoch for mode, value in states.items()},
            "architecture_signature": architecture["architecture_signature"],
            "seed": self.seed,
            "metrics": ["participant_macro_f1", "participant_balanced_accuracy"],
            "decision_rule": self.config["decision_rule"],
            "checkpoint_selection_partition": "source_validation",
            "target_test_opened": False,
        }
        unlock_path = self.output / "target_test_unlock_manifest.json"
        _write_json(unlock_path, unlock)
        unlock_hash = _sha256_file(unlock_path)
        _write_json(self.output / "target_test_unlock_hash.json", {"algorithm": "sha256", "sha256": unlock_hash})
        self.lock.unlock(unlock_hash)

        target_frames = []
        target_metrics = {}
        for mode, state in states.items():
            task_model = state.model if mode == "source_only_matched" else dann.task_model
            target_dataset = _RawPartitionDataset(raw, positions["target_test"], metadata, include_task_label=True, domain_label=0, lock=self.lock)
            predictions, metrics = self._predict(task_model, target_dataset, mode=mode, partition="target_test")
            target_frames.append(predictions)
            target_metrics[mode] = metrics
        predictions = pd.concat(target_frames, ignore_index=True)
        first = predictions.loc[predictions["mode"].eq(MODES[0])].reset_index(drop=True)
        second = predictions.loc[predictions["mode"].eq(MODES[1])].reset_index(drop=True)
        if not first[["sample_id", "subject_id", "record_group_id", "y_true"]].equals(second[["sample_id", "subject_id", "record_group_id", "y_true"]]):
            raise RuntimeError("Target-test prediction alignment differs between modes")
        predictions.to_parquet(self.output / "target_test_predictions.parquet", index=False)
        subject_metrics = self._extended_subject_metrics(predictions)
        subject_metrics.to_csv(self.output / "target_test_subject_metrics.csv", index=False)
        aggregate_rows = []
        for mode in MODES:
            selected = subject_metrics.loc[subject_metrics["mode"].eq(mode)]
            row = {"mode": mode, "subjects": len(selected), "windows": int((predictions["mode"] == mode).sum())}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa", "ordinal_mae", "macro_precision", "macro_recall", "prediction_entropy"):
                row[f"participant_mean_{metric}"] = float(selected[metric].mean())
                row[f"participant_median_{metric}"] = float(selected[metric].median())
            row.update({f"window_{key}": value for key, value in target_metrics[mode].items() if not isinstance(value, list)})
            aggregate_rows.append(row)
        aggregate = pd.DataFrame(aggregate_rows)
        aggregate.to_csv(self.output / "target_test_aggregate_metrics.csv", index=False)
        comparison = paired_dann_comparison(subject_metrics)
        _write_json(self.output / "paired_comparison.json", comparison)
        confusion = {
            mode: confusion_matrix(
                predictions.loc[predictions["mode"].eq(mode), "y_true"],
                predictions.loc[predictions["mode"].eq(mode), "y_pred"], labels=np.arange(5),
            ).tolist() for mode in MODES
        }
        _write_json(self.output / "confusion_matrices.json", confusion)
        gradient_frame = pd.read_csv(self.output / "dann/gradient_audit.csv")
        domain_frame = pd.read_csv(self.output / "dann/domain_metrics.csv")
        unstable = not gradient_frame["finite"].all()
        suppression = bool((gradient_frame["domain_to_task_encoder_gradient_ratio"] > 10.0).mean() > 0.5)
        decision = apply_dann_decision_rule(comparison, self.config["decision_rule"], unstable=unstable, gradient_suppression=suppression)
        _write_json(self.output / "decision.json", decision)
        leakage.update({
            "target_labels_available_to_training_step": False,
            "target_test_read_before_unlock": self.lock.reads_before_unlock > 0,
            "target_test_reads_before_unlock": self.lock.reads_before_unlock,
            "target_test_reads_after_unlock": self.lock.reads_after_unlock,
            "target_test_evaluations_per_checkpoint": 1,
            "source_artifacts_unchanged": self.immutable_before == self._verify_source_files(),
        })
        _write_json(self.output / "leakage_audit.json", leakage)
        checkpoint_hashes_after = {mode: _sha256_file(self.output / ("source_only" if mode == "source_only_matched" else "dann") / "checkpoint.pt") for mode in MODES}
        immutability = {
            "checkpoint_hashes_before_target_test": {mode: checkpoint_manifests[mode]["checkpoint_sha256"] for mode in MODES},
            "checkpoint_hashes_after_target_test": checkpoint_hashes_after,
            "unchanged": all(checkpoint_hashes_after[mode] == checkpoint_manifests[mode]["checkpoint_sha256"] for mode in MODES),
            "initial_state_unchanged": (
                _tensor_state_hash(torch.load(self.output / "initial_model_state.pt", map_location="cpu", weights_only=False)["model_state_dict"])
                == _tensor_state_hash(torch.load(self.output / "initial_encoder_task_state.pt", map_location="cpu", weights_only=False)["model_state_dict"])
                == architecture["initial_model_hash"]
            ),
            "training_after_target_test": False,
        }
        _write_json(self.output / "checkpoint_immutability_audit.json", immutability)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.config["experiment_id"],
            "result_status": "diagnostic",
            "direction": "Old_EEG_to_gpn_data",
            "outer_fold": 1,
            "seed": self.seed,
            "device": self.device,
            "device_name": torch.cuda.get_device_name(torch.device(self.device)),
            "git_commit": _git_head(self.root),
            "protocol_hash": self.config["protocol"]["expected_protocol_hash"],
            "candidate_hash": self.config["protocol"]["expected_candidate_hash"],
            "preregistration_hash": preregistration_hash,
            "initial_model_hash": architecture["initial_model_hash"],
            "architecture": architecture,
            "partition_counts": leakage["partition_counts"],
            "training": {"epochs": training_summary["epochs_trained"], "training_time_seconds": training_summary["training_time_seconds"], "optimizer_updates_per_mode": states["dann"].updates, "best_epochs": {mode: states[mode].best_epoch for mode in MODES}},
            "schedules": {
                "grl_formula": self.config["schedule"]["gradient_reversal_formula"],
                "grl_alpha_first": float(domain_frame["grl_alpha"].iloc[0]),
                "grl_alpha_last": float(domain_frame["grl_alpha"].iloc[-1]),
                "domain_lambda": float(self.config["schedule"]["domain_loss_lambda"]),
            },
            "dann_training_diagnostics": {
                "mean_task_loss": float(domain_frame["task_loss"].mean()),
                "final_task_loss": float(domain_frame["task_loss"].iloc[-1]),
                "mean_domain_loss": float(domain_frame["domain_loss"].mean()),
                "final_domain_loss": float(domain_frame["domain_loss"].iloc[-1]),
                "mean_domain_accuracy_source": float(domain_frame["domain_accuracy_source"].mean()),
                "mean_domain_accuracy_target": float(domain_frame["domain_accuracy_target"].mean()),
                "mean_domain_accuracy_combined": float(domain_frame["domain_accuracy_combined"].mean()),
                "mean_encoder_total_gradient_norm": float(domain_frame["encoder_total_gradient_norm"].mean()),
                "mean_task_head_gradient_norm": float(domain_frame["task_head_gradient_norm"].mean()),
                "mean_domain_head_gradient_norm": float(domain_frame["domain_head_gradient_norm"].mean()),
            },
            "gradient_decomposition": {
                "epochs_audited": int(len(gradient_frame)),
                "mean_task_only_encoder_gradient_norm": float(gradient_frame["task_only_encoder_gradient_norm"].mean()),
                "mean_weighted_domain_only_encoder_gradient_norm": float(gradient_frame["weighted_domain_only_encoder_gradient_norm"].mean()),
                "mean_domain_to_task_encoder_gradient_ratio": float(gradient_frame["domain_to_task_encoder_gradient_ratio"].mean()),
                "maximum_domain_to_task_encoder_gradient_ratio": float(gradient_frame["domain_to_task_encoder_gradient_ratio"].max()),
                "all_finite": bool(gradient_frame["finite"].all()),
                "all_model_states_unchanged": bool(gradient_frame["model_state_unchanged"].all()),
            },
            "checkpoints": checkpoint_manifests,
            "unlock_hash": unlock_hash,
            "target_test_window_metrics": target_metrics,
            "target_test_participant_aggregate": aggregate.to_dict("records"),
            "paired_comparison": comparison,
            "confusion_matrices": confusion,
            "decision": decision,
            "leakage_safe": leakage["all_overlaps_zero"] and not leakage["target_test_read_before_unlock"],
            "checkpoint_immutable": immutability["unchanged"],
            "limitations": ["one outer fold", "one seed", "three source-validation participants", "eight target-test participants", "diagnostic result only"],
        }
        _write_json(self.output / "diagnostic_summary.json", summary)
        report = render_diagnostic_report(summary, subject_metrics)
        (self.output / "diagnostic_report.md").write_text(report, encoding="utf-8")
        tracked = self.root / "reports/integration/dann_label_q5_raw_diagnostic.md"
        tracked.parent.mkdir(parents=True, exist_ok=True)
        tracked.write_text(report, encoding="utf-8")
        return summary


def validate_dann_diagnostic_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_enabled") is not True:
        raise ValueError("Diagnostic DANN config must set execution_enabled=true")
    if int(config.get("seed", -1)) != 42 or int(config["protocol"]["outer_fold"]) != 1:
        raise ValueError("Task 8Shch is restricted to seed 42 and outer fold 1")
    if config["protocol"]["direction"] != "Old_EEG_to_gpn_data":
        raise ValueError("Task 8Shch is restricted to Old_EEG_to_gpn_data")
    if config["protocol"]["subject_policy"] != "strict_cross_domain_subject_disjoint":
        raise ValueError("Strict cross-domain subject disjoint policy is required")
    if config["model"]["name"] != "torch_eegnet" or int(config["training"]["steps_per_epoch"]) != 580:
        raise ValueError("Production EEGNet and 580 matched steps are fixed")
    if config["device"] != "cuda":
        raise ValueError("Task 8Shch requires device=cuda")
    if _contains_absolute_path(config):
        raise ValueError("Tracked diagnostic config cannot contain absolute paths")


def render_diagnostic_report(summary: Mapping[str, Any], subject_metrics: pd.DataFrame) -> str:
    comparison = summary["paired_comparison"]
    aggregate = {row["mode"]: row for row in summary["target_test_participant_aggregate"]}
    checkpoint = summary["checkpoints"]
    lines = [
        "# Diagnostic DANN: Old_EEG → gpn_data",
        "",
        f"- Branch/HEAD at execution: `integration/benchmark-unification` / `{summary['git_commit']}`.",
        "- Scientific hypothesis: unlabeled gpn_data domain training improves direct label_q5 transfer over a matched source-only EEGNet.",
        f"- Status: **{summary['decision']['status']}** (`diagnostic`, not a final scientific result).",
        f"- Protocol / candidate / executable preregistration: `{summary['protocol_hash']}` / `{summary['candidate_hash']}` / `{summary['preregistration_hash']}`.",
        f"- Direction: Old_EEG labeled train → gpn_data unseen participants; strict subject-disjoint fold 1, seed 42.",
        f"- Partitions: {summary['partition_counts']}.",
        f"- Production EEGNet: 8,501 task parameters, latent 1,280; fixed discriminator 172,354; total DANN 180,855.",
        f"- Matched budget: {summary['training']['optimizer_updates_per_mode']} source optimizer updates per mode over {summary['training']['epochs']} epochs; 580 steps/epoch.",
        f"- Best source-validation epochs: source-only {summary['training']['best_epochs']['source_only_matched']}, DANN {summary['training']['best_epochs']['dann']}.",
        f"- Source-validation macro F1: source-only {checkpoint['source_only_matched']['source_validation_metrics']['macro_f1']:.6f}, DANN {checkpoint['dann']['source_validation_metrics']['macro_f1']:.6f}.",
        "- GRL: `2/(1+exp(-10*p))-1`; domain lambda: constant 1.0. Domain accuracy is diagnostic and never selected a checkpoint.",
        f"- GRL alpha range: {summary['schedules']['grl_alpha_first']:.6f} → {summary['schedules']['grl_alpha_last']:.6f}; mean/final domain loss: {summary['dann_training_diagnostics']['mean_domain_loss']:.6f}/{summary['dann_training_diagnostics']['final_domain_loss']:.6f}.",
        f"- Mean source/target/combined domain accuracy: {summary['dann_training_diagnostics']['mean_domain_accuracy_source']:.6f}/{summary['dann_training_diagnostics']['mean_domain_accuracy_target']:.6f}/{summary['dann_training_diagnostics']['mean_domain_accuracy_combined']:.6f}.",
        f"- Mean encoder/task-head/domain-head gradient norms: {summary['dann_training_diagnostics']['mean_encoder_total_gradient_norm']:.6f}/{summary['dann_training_diagnostics']['mean_task_head_gradient_norm']:.6f}/{summary['dann_training_diagnostics']['mean_domain_head_gradient_norm']:.6f}.",
        f"- Gradient decomposition mean domain/task encoder ratio: {summary['gradient_decomposition']['mean_domain_to_task_encoder_gradient_ratio']:.6f} (maximum {summary['gradient_decomposition']['maximum_domain_to_task_encoder_gradient_ratio']:.6f}); all finite and state-preserving: {summary['gradient_decomposition']['all_finite'] and summary['gradient_decomposition']['all_model_states_unchanged']}.",
        f"- Checkpoints: source-only `{checkpoint['source_only_matched']['checkpoint_sha256']}`, DANN `{checkpoint['dann']['checkpoint_sha256']}`.",
        f"- Target-test unlock hash: `{summary['unlock_hash']}`; checkpoints remained immutable: {summary['checkpoint_immutable']}.",
        "",
        "## Target-test participant-level aggregate",
        "",
        "| mode | mean macro F1 | mean balanced accuracy | median macro F1 | mean ordinal MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = aggregate[mode]
        lines.append(f"| {mode} | {row['participant_mean_macro_f1']:.6f} | {row['participant_mean_balanced_accuracy']:.6f} | {row['participant_median_macro_f1']:.6f} | {row['participant_mean_ordinal_mae']:.6f} |")
    lines.extend([
        "",
        "## Paired result",
        "",
        f"DANN − source-only mean participant Δmacro F1 = {comparison['mean_delta_macro_f1']:+.6f}; Δbalanced accuracy = {comparison['mean_delta_balanced_accuracy']:+.6f}; Δordinal MAE = {comparison['mean_delta_ordinal_mae']:+.6f}.",
        f"Macro-F1 wins/losses/ties: {comparison['macro_f1_wins']}/{comparison['macro_f1_losses']}/{comparison['macro_f1_ties']} across eight participants.",
        "The participant bootstrap is descriptive only; eight target participants and three source-validation participants provide low statistical power.",
        "",
        "## Confusion matrices (rows=true, columns=predicted)",
        "",
        "```json",
        json.dumps(summary["confusion_matrices"], indent=2),
        "```",
        "",
        "## Participant metrics",
        "",
        "| mode | subject | n | macro F1 | balanced accuracy | ordinal MAE |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in subject_metrics.sort_values(["mode", "subject_id"]).itertuples():
        lines.append(f"| {row.mode} | {row.subject_id} | {row.n_samples} | {row.macro_f1:.6f} | {row.balanced_accuracy:.6f} | {row.ordinal_mae:.6f} |")
    lines.extend([
        "",
        "## Integrity and interpretation",
        "",
        "All subject, sample, and logical-record overlaps are zero. Target-train batches contain no task-label field. Target-test tensors were opened only after both checkpoint hashes, best epochs, schedules, metrics, and the decision rule were fixed. Gradient decomposition was diagnostic and performed no optimizer step; gradient clipping was applied during normal updates.",
        "",
        "This is one fold and one seed, so it cannot establish robustness or statistical significance. Reverse direction, additional folds/seeds, other models, target-supervised bounds, and hyperparameter search were not run. Any follow-up requires a separate approved question; no automatic next experiment is selected.",
        "",
    ])
    return "\n".join(lines)


def run_dann_label_q5_raw_diagnostic(config_path: str | Path, *, repository_root: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path)
    root = Path(repository_root) if repository_root is not None else Path.cwd()
    config = json.loads((root / path).read_text(encoding="utf-8") if not path.is_absolute() else path.read_text(encoding="utf-8"))
    return DANNLabelQ5RawDiagnostic(config, repository_root=root).run()


__all__ = [
    "DANNLabelQ5RawDiagnostic", "TargetTestLock", "apply_dann_decision_rule",
    "batch_plan_hash", "deterministic_batch_plan", "enforce_target_batch_firewall",
    "logistic_grl_alpha", "paired_dann_comparison", "run_dann_label_q5_raw_diagnostic",
    "validate_dann_diagnostic_config",
]
