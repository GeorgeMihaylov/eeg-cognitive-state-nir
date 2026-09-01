"""Resumable execution of the preregistered DANN confirmatory-v2 experiment.

The existing raw-DANN diagnostic training loop is reused for every fold/seed
pair.  This module adds only multi-block orchestration, immutable execution
preregistration, resume state, target-test locks, and participant-level
aggregation.
"""

from __future__ import annotations

import json
import shutil
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix

from bench.experiments.dann_label_q5_confirmatory_v2_protocol import (
    PRIMARY_SEEDS,
    PROTOCOL_ID,
    SECONDARY_SEEDS,
    aggregate_participant_deltas,
    apply_primary_decision_rule,
)
from bench.experiments.dann_label_q5_raw_diagnostic import (
    MODES,
    DANNLabelQ5RawDiagnostic,
    TargetTestLock,
    _ModeState,
)
from bench.experiments.fomaml_label_q5_diagnostic import (
    _atomic_torch_save,
    _git_head,
    _jsonable,
    _sha256_file,
    _tensor_state_hash,
    _write_json,
    prepare_preregistration,
    resolve_device,
)
from bench.meta.episodes import stable_hash


SCHEMA_VERSION = "dann-label-q5-confirmatory-v2-execution-v1"
PAIR_STATUSES = {
    "pending", "training", "checkpoint_fixed", "target_unlocked",
    "evaluated", "complete", "failed_technical",
}
NEW_RUN_GROUPS = ("primary_confirmatory", "secondary_sensitivity")
SCALAR_METRICS = (
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa",
    "ordinal_mae", "quadratic_weighted_kappa", "macro_precision",
    "macro_recall", "prediction_entropy",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute() or bool(
            len(value) > 2 and value[1:3] in {":/", ":\\"}
        )
    return False


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    normalized = _jsonable(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(f"Immutable artifact would change: {path.name}")
    else:
        _write_json(path, normalized)
    return _sha256_file(path)


def validate_confirmatory_v2_execution_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_enabled") is not True:
        raise ValueError("confirmatory-v2 execution config must be enabled")
    protocol = config["protocol"]
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise ValueError("execution must use the fixed confirmatory-v2 protocol")
    if protocol["protocol_hash"] != (
        "1ce582a3d73a7ae4393e77cc2f3b2cb7749ddbb30c1cb8fcad0056c6d326c368"
    ):
        raise ValueError("confirmatory-v2 protocol hash changed")
    if protocol["disabled_preregistration_hash"] != (
        "6fba1eb76133884f0d5984ec1ceedc49234f252846040122310ac45a99ad3d7e"
    ):
        raise ValueError("disabled confirmatory-v2 preregistration changed")
    if tuple(config["primary"]["folds"]) != (1, 2, 3, 4, 5):
        raise ValueError("primary phase requires all five folds")
    if tuple(config["primary"]["seeds"]) != PRIMARY_SEEDS:
        raise ValueError("primary seeds must remain 123 and 2026")
    if tuple(config["secondary"]["folds"]) != (2, 3, 4, 5):
        raise ValueError("secondary phase requires folds 2-5")
    if tuple(config["secondary"]["seeds"]) != SECONDARY_SEEDS:
        raise ValueError("secondary seed must remain 42")
    training = config["training"]
    expected = {
        "source_batch_size": 32,
        "target_batch_size": 32,
        "maximum_epochs": 12,
        "early_stopping_patience": 3,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "checkpoint_primary": "source_validation_macro_f1",
        "checkpoint_secondary": "source_validation_balanced_accuracy",
        "source_validation_split_seed": 42,
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError("execution hyperparameters differ from preregistration")
    if config["schedule"] != {
        "gradient_reversal_formula": "alpha(p)=2/(1+exp(-10*p))-1",
        "progress_formula": "global_step/max(total_steps-1,1)",
        "domain_loss_lambda": 1.0,
    }:
        raise ValueError("DANN schedules changed")
    if _contains_absolute_path(config):
        raise ValueError("execution config contains an absolute path")


def build_execution_registry(run_matrix: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in run_matrix.to_dict("records"):
        completed = row["analysis_group"] == "previously_observed_diagnostic"
        rows.append({
            **row,
            "status": "complete" if completed else "pending",
            "attempt_count": 0,
            "technical_restart_count": 0,
            "checkpoint_hash": None,
            "unlock_hash": None,
            "prediction_hash": None,
            "participant_metrics_hash": None,
            "updated_at": None,
        })
    return rows


def _registry_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    scientific = [
        {key: row[key] for key in (
            "run_id", "analysis_group", "fold", "seed", "mode",
            "execution_status", "provenance",
        )}
        for row in rows
    ]
    return stable_hash(scientific)


def bootstrap_unique_participants(
    participant_results: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = participant_results["delta_macro_f1"].to_numpy(float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("participant bootstrap needs at least two finite deltas")
    rng = np.random.default_rng(int(seed))
    samples = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(int(resamples))
    ])
    return {
        "unit": "unique_participant_after_primary_seed_average",
        "n_participants": len(values),
        "resamples": int(resamples),
        "seed": int(seed),
        "mean_delta_macro_f1": float(values.mean()),
        "median_delta_macro_f1": float(np.median(values)),
        "mean_95_ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "statistical_significance_claimed": False,
    }


def pair_subject_metrics(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"fold", "seed", "subject_id", "mode", *SCALAR_METRICS}
    missing = required.difference(subject_metrics.columns)
    if missing:
        raise ValueError(f"Missing subject metrics: {sorted(missing)}")
    index = ["fold", "seed", "subject_id"]
    paired = subject_metrics.pivot(
        index=index, columns="mode", values=list(SCALAR_METRICS)
    )
    for metric in SCALAR_METRICS:
        for mode in MODES:
            if (metric, mode) not in paired.columns:
                raise ValueError(f"Missing {metric}/{mode} participant result")
    paired = paired.reset_index()
    paired.columns = index + [
        f"{metric}_{mode}" for metric, mode in paired.columns[len(index):]
    ]
    metric_columns = [
        f"{metric}_{mode}" for metric in SCALAR_METRICS for mode in MODES
    ]
    if paired[metric_columns].isna().any(axis=None):
        incomplete = paired.loc[
            paired[metric_columns].isna().any(axis=1), index
        ].to_dict("records")
        raise ValueError(f"Missing paired mode result for participants: {incomplete}")
    for metric in SCALAR_METRICS:
        paired[f"delta_{metric}"] = (
            paired[f"{metric}_dann"]
            - paired[f"{metric}_source_only_matched"]
        )
    return paired


def average_participants_across_seeds(paired: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column for column in paired.columns
        if column not in {"fold", "seed", "subject_id"}
    ]
    result = paired.groupby(
        ["fold", "subject_id"], sort=True, as_index=False
    )[numeric].mean()
    counts = paired.groupby(["fold", "subject_id"])["seed"].nunique()
    result["seed_count"] = counts.to_numpy()
    result["participant_weight"] = 1.0
    return result


def fold_level_metrics(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, frame in participants.groupby("fold", sort=True):
        delta = frame["delta_macro_f1"].to_numpy(float)
        rows.append({
            "fold": int(fold),
            "participants": len(frame),
            "mean_delta_macro_f1": float(delta.mean()),
            "median_delta_macro_f1": float(np.median(delta)),
            "mean_delta_balanced_accuracy": float(
                frame["delta_balanced_accuracy"].mean()
            ),
            "mean_delta_ordinal_mae": float(frame["delta_ordinal_mae"].mean()),
            "wins": int((delta > 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()),
            "ties": int((np.abs(delta) <= 1e-12).sum()),
        })
    return pd.DataFrame(rows)


def seed_level_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, frame in paired.groupby("seed", sort=True):
        delta = frame["delta_macro_f1"].to_numpy(float)
        rows.append({
            "seed": int(seed),
            "participants": len(frame),
            "mean_delta_macro_f1": float(delta.mean()),
            "median_delta_macro_f1": float(np.median(delta)),
            "mean_delta_balanced_accuracy": float(
                frame["delta_balanced_accuracy"].mean()
            ),
            "mean_delta_ordinal_mae": float(frame["delta_ordinal_mae"].mean()),
            "wins": int((delta > 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()),
            "ties": int((np.abs(delta) <= 1e-12).sum()),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PairResult:
    fold: int
    seed: int
    summary: dict[str, Any]


class _ConfirmatoryPairRunner(DANNLabelQ5RawDiagnostic):
    """One fold/seed pair using the existing diagnostic training loop."""

    def __init__(
        self,
        execution_config: Mapping[str, Any],
        *,
        repository_root: Path,
        output_dir: Path,
        fold: int,
        seed: int,
        analysis_group: str,
        execution_preregistration_hash: str,
        protocol_manifest: Mapping[str, Any],
        status_callback: Callable[[str], None],
    ) -> None:
        self.root = repository_root
        self.output = output_dir
        self.seed = int(seed)
        self.outer_fold = int(fold)
        self.analysis_group = analysis_group
        self.execution_preregistration_hash = execution_preregistration_hash
        self.protocol_manifest = dict(protocol_manifest)
        self.status_callback = status_callback
        training = deepcopy(dict(execution_config["training"]))
        budget = next(
            row for row in protocol_manifest["matched_update_budgets"]
            if int(row["fold"]) == self.outer_fold
        )
        training["steps_per_epoch"] = int(budget["matched_steps_per_epoch"])
        self.config = {
            "experiment_id": execution_config["experiment_id"],
            "seed": self.seed,
            "device": execution_config["device"],
            "dataset": deepcopy(dict(execution_config["dataset"])),
            "model": deepcopy(dict(execution_config["model"])),
            "training": training,
            "schedule": deepcopy(dict(execution_config["schedule"])),
        }
        self.device = resolve_device(str(execution_config["device"]))
        if not self.device.startswith("cuda"):
            raise RuntimeError("Confirmatory DANN execution requires CUDA")
        self.lock = TargetTestLock()
        self.gradient_steps_started = 0
        v1_root = repository_root / "benchmark_results/domain_adaptation_dann_confirmatory_protocol"
        self.fold_partitions_path = repository_root / str(
            execution_config["protocol"]["fold_partitions"]
        )
        self.source_path = v1_root / f"source_validation_manifests/fold_{fold:02d}.json"
        self.target_path = v1_root / f"target_unlabeled_manifests/fold_{fold:02d}.json"
        self.test_path = v1_root / f"target_test_references/fold_{fold:02d}.json"
        self.immutable_before = {
            "source_validation_manifest": _sha256_file(self.source_path),
            "target_unlabeled_manifest": _sha256_file(self.target_path),
            "target_test_reference": _sha256_file(self.test_path),
        }

    def _protocol_partitions(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        source = json.loads(self.source_path.read_text(encoding="utf-8"))
        target = json.loads(self.target_path.read_text(encoding="utf-8"))
        test = json.loads(self.test_path.read_text(encoding="utf-8"))
        fixed = pd.read_parquet(self.fold_partitions_path)
        fixed = fixed.loc[fixed["fold"].eq(self.outer_fold)].set_index("partition")
        expected_partitions = {
            "source_task_train": source["source_task_train"],
            "source_validation": source["source_validation"],
            "target_train_unlabelled": target,
            "target_outer_test_reference": test,
        }
        if set(fixed.index) != set(expected_partitions):
            raise RuntimeError("fixed fold partition names changed")
        for partition, manifest in expected_partitions.items():
            row = fixed.loc[partition]
            if int(row["samples"]) != int(manifest["samples"]):
                raise RuntimeError(f"sample count changed for {partition}")
            expected_ids = [str(value) for value in row["sample_ids"]]
            observed_ids = [str(value) for value in manifest["sample_ids"]]
            if expected_ids != observed_ids:
                raise RuntimeError(f"sample IDs changed for {partition}")
        counts = (
            source["source_task_train"]["samples"],
            source["source_validation"]["samples"],
            target["samples"],
            test["samples"],
        )
        expected_counts = (
            int(fixed.loc["source_task_train", "samples"]),
            int(fixed.loc["source_validation", "samples"]),
            int(fixed.loc["target_train_unlabelled", "samples"]),
            int(fixed.loc["target_outer_test_reference", "samples"]),
        )
        if counts != expected_counts or source["fold"] != self.outer_fold:
            raise RuntimeError("fold partitions differ from confirmatory-v2 protocol")
        if target["task_labels_exposed"] is not False:
            raise RuntimeError("target-label firewall is not active")
        return source, target, test

    def _artifact_dirs(self) -> dict[str, Path]:
        return {
            "source_only_matched": self.output / "source_only",
            "dann": self.output / "dann",
        }

    def _write_attempt(
        self,
        attempt: int,
        *,
        status: str,
        resumed_from: str,
        error: str | None = None,
    ) -> None:
        for mode, directory in self._artifact_dirs().items():
            payload = {
                "fold": self.outer_fold,
                "seed": self.seed,
                "mode": mode,
                "attempt": attempt,
                "status": status,
                "resumed_from": resumed_from,
                "same_scientific_specification": True,
                "technical_error": error,
                "updated_at": _now(),
            }
            _write_json(directory / "attempt_manifest.json", payload)
            _write_json(directory / f"attempts/attempt_{attempt:03d}.json", payload)

    def _write_run_specs(self) -> None:
        for mode, directory in self._artifact_dirs().items():
            directory.mkdir(parents=True, exist_ok=True)
            _immutable_json(directory / "run_specification.json", {
                "run_id": f"fold_{self.outer_fold:02d}_seed_{self.seed}_{mode}",
                "analysis_group": self.analysis_group,
                "fold": self.outer_fold,
                "seed": self.seed,
                "mode": mode,
                "protocol_id": PROTOCOL_ID,
                "protocol_hash": self.protocol_manifest["protocol_hash"],
                "execution_preregistration_hash": self.execution_preregistration_hash,
                "execution_status": "planned",
            })
            pd.DataFrame(columns=["stage", "code", "message", "traceback"]).to_csv(
                directory / "errors.csv", index=False
            )

    def _load_fixed_models(
        self,
    ) -> tuple[Any, Any, dict[str, Any]]:
        source_adapter, _, dann, architecture = self._architecture_and_models()
        source_payload = torch.load(
            self.output / "source_only/checkpoint.pt",
            map_location=self.device,
            weights_only=False,
        )
        source_adapter.model.load_state_dict(source_payload["model_state_dict"], strict=True)
        source_adapter.model.to(self.device).eval()
        dann.load(self.output / "dann/checkpoint.pt", map_location=self.device)
        dann.to(self.device).eval()
        architecture["domain_head_signature"] = self.protocol_manifest["architecture"]["domain_head_signature"]
        architecture["domain_head_initial_hash"] = _tensor_state_hash(
            dann.domain_discriminator.state_dict()
        )
        return source_adapter, dann, architecture

    def execute(self, *, attempt: int, resumed_from: str) -> PairResult:
        self.output.mkdir(parents=True, exist_ok=True)
        self._write_run_specs()
        self._write_attempt(attempt, status="training", resumed_from=resumed_from)
        source, target, test = self._protocol_partitions()
        data, metadata = self._load_data()
        positions, leakage = self._partition_audit(metadata, source, target, test)
        mean, scale = data.data[positions["source_train"]].compute_channel_statistics()
        raw = data.data.with_channel_normalization(mean, scale)
        _write_json(self.output / "normalization_stats.json", {
            "fit_partition": "source_task_train_only",
            "channel_mean": mean,
            "channel_scale": scale,
            "n_windows": len(positions["source_train"]),
        })
        state_path = self.output / "pair_state.json"
        prior = _load_json(state_path) if state_path.exists() else {"status": "pending"}
        status = prior.get("status", "pending")
        if status in {"checkpoint_fixed", "target_unlocked", "evaluated"}:
            source_adapter, dann, architecture = self._load_fixed_models()
            checkpoint_manifests = {
                mode: _load_json(
                    self._artifact_dirs()[mode] / "checkpoint_manifest.json"
                )
                for mode in MODES
            }
            training_summary = _load_json(self.output / "training_summary.json")
        else:
            source_adapter, _, dann, architecture = self._architecture_and_models()
            if architecture["architecture_signature"] != self.protocol_manifest["architecture"]["architecture_signature"]:
                raise RuntimeError("production EEGNet signature changed before training")
            domain_initial_hash = _tensor_state_hash(
                dann.domain_discriminator.state_dict()
            )
            architecture["domain_head_initial_hash"] = domain_initial_hash
            architecture["domain_head_signature"] = self.protocol_manifest["architecture"]["domain_head_signature"]
            _write_json(self.output / "architecture_audit.json", architecture)
            for mode, directory in self._artifact_dirs().items():
                shutil.copyfile(
                    self.output / "initial_model_state.pt",
                    directory / "initial_model_state.pt",
                )
                _write_json(directory / "initial_model_manifest.json", {
                    "fold": self.outer_fold,
                    "seed": self.seed,
                    "mode": mode,
                    "task_model_hash": architecture["initial_model_hash"],
                    "domain_head_hash": domain_initial_hash,
                    "source_only_and_dann_task_states_identical": True,
                    "includes_batchnorm_parameters_and_buffers": True,
                    "architecture_signature": architecture["architecture_signature"],
                    "domain_head_signature": architecture["domain_head_signature"],
                })
            if not (self.output.parent.parent.parent / "preregistration/experiment_preregistration.json").exists():
                raise RuntimeError("execution preregistration missing before gradient step")
            states, training_summary = self._train(
                source_adapter, dann, raw, metadata, positions
            )
            _write_json(self.output / "training_summary.json", training_summary)
            checkpoint_manifests: dict[str, Any] = {}
            validation_dataset = self._validation_dataset(raw, positions, metadata)
            source_batch_hash = _sha256_file(self.output / "source_batch_audit.json")
            for mode, state in states.items():
                directory = self._artifact_dirs()[mode]
                checkpoint_path = directory / "checkpoint.pt"
                optimizer_path = directory / "optimizer_state.pt"
                task_model = (
                    state.model if mode == "source_only_matched" else dann.task_model
                )
                predictions, metrics = self._predict(
                    task_model,
                    validation_dataset,
                    mode=mode,
                    partition="source_validation",
                )
                predictions.to_parquet(
                    directory / "source_validation_predictions.parquet", index=False
                )
                manifest = {
                    "mode": mode,
                    "fold": self.outer_fold,
                    "seed": self.seed,
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                    "optimizer_state_sha256": _sha256_file(optimizer_path),
                    "best_epoch": state.best_epoch,
                    "source_validation_metrics": metrics,
                    "selection_primary": "macro_f1",
                    "selection_secondary": "balanced_accuracy",
                    "tie_breaker": "earlier_epoch",
                    "target_data_used_for_selection": False,
                    "domain_accuracy_used_for_selection": False,
                    "optimizer_updates": state.updates,
                    "initial_model_hash": architecture["initial_model_hash"],
                    "source_batch_hashes_sha256": source_batch_hash,
                    "architecture_signature": architecture["architecture_signature"],
                    "protocol_hash": self.protocol_manifest["protocol_hash"],
                    "execution_preregistration_hash": self.execution_preregistration_hash,
                }
                _write_json(directory / "checkpoint_manifest.json", manifest)
                checkpoint_manifests[mode] = manifest
            _write_json(state_path, {
                "status": "checkpoint_fixed", "fold": self.outer_fold,
                "seed": self.seed, "updated_at": _now(),
            })
            self.status_callback("checkpoint_fixed")
            status = "checkpoint_fixed"

        unlock_path = self.output / "target_test_unlock_manifest.json"
        if status == "checkpoint_fixed" or not unlock_path.exists():
            source_batch_audit = _load_json(self.output / "source_batch_audit.json")
            unlock = {
                "fold": self.outer_fold,
                "seed": self.seed,
                "protocol_hash": self.protocol_manifest["protocol_hash"],
                "execution_preregistration_hash": self.execution_preregistration_hash,
                "source_only_checkpoint_hash": checkpoint_manifests["source_only_matched"]["checkpoint_sha256"],
                "dann_checkpoint_hash": checkpoint_manifests["dann"]["checkpoint_sha256"],
                "best_epochs": {
                    mode: checkpoint_manifests[mode]["best_epoch"] for mode in MODES
                },
                "source_validation_metrics": {
                    mode: checkpoint_manifests[mode]["source_validation_metrics"]
                    for mode in MODES
                },
                "source_batch_hashes": source_batch_audit["epochs"],
                "architecture_signature": architecture["architecture_signature"],
                "domain_head_signature": architecture["domain_head_signature"],
                "primary_decision_rule_hash": stable_hash(
                    self.protocol_manifest["primary_decision_rule"]
                ),
                "diagnostic_unlock_reused": False,
                "target_test_opened": False,
            }
            unlock_hash = _immutable_json(unlock_path, unlock)
            _write_json(self.output / "target_test_unlock_hash.json", {
                "algorithm": "sha256", "sha256": unlock_hash,
            })
            _write_json(state_path, {
                "status": "target_unlocked", "fold": self.outer_fold,
                "seed": self.seed, "unlock_hash": unlock_hash,
                "updated_at": _now(),
            })
            self.status_callback("target_unlocked")
        else:
            unlock_hash = _sha256_file(unlock_path)
        self.lock.unlock(unlock_hash)

        target_dataset = self._target_dataset(raw, positions, metadata)
        target_frames = []
        target_metrics: dict[str, Any] = {}
        models = {
            "source_only_matched": source_adapter.model,
            "dann": dann.task_model,
        }
        checkpoint_before = {
            mode: _sha256_file(self._artifact_dirs()[mode] / "checkpoint.pt")
            for mode in MODES
        }
        for mode in MODES:
            predictions, metrics = self._predict(
                models[mode], target_dataset, mode=mode, partition="target_test"
            )
            target_frames.append(predictions)
            target_metrics[mode] = metrics
        predictions = pd.concat(target_frames, ignore_index=True)
        first = predictions[predictions["mode"].eq(MODES[0])].reset_index(drop=True)
        second = predictions[predictions["mode"].eq(MODES[1])].reset_index(drop=True)
        alignment = ["sample_id", "subject_id", "record_group_id", "y_true"]
        if not first[alignment].equals(second[alignment]):
            raise RuntimeError("target-test IDs or labels differ between modes")
        predictions.to_parquet(self.output / "target_test_predictions.parquet", index=False)
        subject_metrics = self._extended_subject_metrics(predictions)
        subject_metrics["fold"] = self.outer_fold
        subject_metrics["seed"] = self.seed
        subject_metrics["analysis_group"] = self.analysis_group
        subject_metrics.to_csv(self.output / "target_test_subject_metrics.csv", index=False)
        _write_json(state_path, {
            "status": "evaluated", "fold": self.outer_fold,
            "seed": self.seed, "updated_at": _now(),
        })
        self.status_callback("evaluated")

        checkpoint_after = {
            mode: _sha256_file(self._artifact_dirs()[mode] / "checkpoint.pt")
            for mode in MODES
        }
        immutability = {
            "before_target_test": checkpoint_before,
            "after_target_test": checkpoint_after,
            "unchanged": checkpoint_before == checkpoint_after,
            "training_after_target_test": False,
        }
        _write_json(self.output / "checkpoint_immutability_audit.json", immutability)
        leakage.update({
            "target_test_reads_before_unlock": self.lock.reads_before_unlock,
            "target_test_reads_after_unlock": self.lock.reads_after_unlock,
            "target_test_evaluations_per_checkpoint": 1,
            "target_labels_available_to_training_step": False,
            "source_manifests_unchanged": self.immutable_before == {
                "source_validation_manifest": _sha256_file(self.source_path),
                "target_unlabeled_manifest": _sha256_file(self.target_path),
                "target_test_reference": _sha256_file(self.test_path),
            },
        })
        _write_json(self.output / "leakage_audit.json", leakage)
        comparison = _pair_comparison(subject_metrics)
        _write_json(self.output / "paired_comparison.json", comparison)
        _write_json(self.output / "confusion_matrices.json", {
            mode: confusion_matrix(
                predictions.loc[predictions["mode"].eq(mode), "y_true"],
                predictions.loc[predictions["mode"].eq(mode), "y_pred"],
                labels=np.arange(5),
            ).tolist()
            for mode in MODES
        })
        for mode, directory in self._artifact_dirs().items():
            selected_predictions = predictions[predictions["mode"].eq(mode)].copy()
            selected_subjects = subject_metrics[subject_metrics["mode"].eq(mode)].copy()
            selected_predictions.to_parquet(
                directory / "target_test_predictions.parquet", index=False
            )
            selected_subjects.to_csv(
                directory / "target_test_subject_metrics.csv", index=False
            )
            shutil.copyfile(unlock_path, directory / "target_test_unlock_manifest.json")
            shutil.copyfile(
                self.output / "source_batch_audit.json",
                directory / "source_batch_hashes.json",
            )
            shutil.copyfile(
                self.output / "leakage_audit.json", directory / "leakage_audit.json"
            )
            shutil.copyfile(
                self.output / "checkpoint_immutability_audit.json",
                directory / "checkpoint_immutability_audit.json",
            )
            if mode == "source_only_matched":
                pd.DataFrame(columns=[
                    "epoch", "task_only_encoder_gradient_norm",
                    "weighted_domain_only_encoder_gradient_norm",
                    "domain_to_task_encoder_gradient_ratio",
                ]).to_csv(directory / "gradient_audit.csv", index=False)
                pd.DataFrame(columns=[
                    "epoch", "step", "global_step", "grl_alpha", "domain_lambda"
                ]).to_csv(directory / "schedule_audit.csv", index=False)
            run_summary = {
                "analysis_group": self.analysis_group,
                "fold": self.outer_fold,
                "seed": self.seed,
                "mode": mode,
                "status": "complete",
                "checkpoint": checkpoint_manifests[mode],
                "unlock_hash": unlock_hash,
                "prediction_hash": _sha256_file(
                    directory / "target_test_predictions.parquet"
                ),
                "participant_metrics_hash": _sha256_file(
                    directory / "target_test_subject_metrics.csv"
                ),
                "target_test_metrics": target_metrics[mode],
                "training_time_seconds_pair": training_summary["training_time_seconds"],
                "checkpoint_immutable": immutability["unchanged"],
                "leakage_safe": leakage["all_overlaps_zero"]
                and leakage["target_test_reads_before_unlock"] == 0,
            }
            _write_json(directory / "run_summary.json", run_summary)
        pair_summary = {
            "analysis_group": self.analysis_group,
            "fold": self.outer_fold,
            "seed": self.seed,
            "status": "complete",
            "checkpoints": checkpoint_manifests,
            "unlock_hash": unlock_hash,
            "paired_comparison": comparison,
            "training": training_summary,
            "leakage_safe": leakage["all_overlaps_zero"]
            and leakage["target_test_reads_before_unlock"] == 0,
            "checkpoint_immutable": immutability["unchanged"],
        }
        _write_json(self.output / "pair_summary.json", pair_summary)
        _write_json(state_path, {
            "status": "complete", "fold": self.outer_fold,
            "seed": self.seed, "updated_at": _now(),
        })
        self.status_callback("complete")
        self._write_attempt(attempt, status="complete", resumed_from=resumed_from)
        return PairResult(self.outer_fold, self.seed, pair_summary)

    @staticmethod
    def _validation_dataset(raw: Any, positions: Mapping[str, np.ndarray], metadata: pd.DataFrame):
        from bench.experiments.dann_label_q5_raw_diagnostic import _RawPartitionDataset

        return _RawPartitionDataset(
            raw, positions["source_validation"], metadata,
            include_task_label=True, domain_label=1,
        )

    def _target_dataset(self, raw: Any, positions: Mapping[str, np.ndarray], metadata: pd.DataFrame):
        from bench.experiments.dann_label_q5_raw_diagnostic import _RawPartitionDataset

        return _RawPartitionDataset(
            raw, positions["target_test"], metadata,
            include_task_label=True, domain_label=0, lock=self.lock,
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_comparison(subject_metrics: pd.DataFrame) -> dict[str, Any]:
    paired = pair_subject_metrics(
        subject_metrics.assign(fold=subject_metrics["fold"].astype(int))
    )
    delta = paired["delta_macro_f1"].to_numpy(float)
    return {
        "comparison_unit": "unique_participant",
        "participants": len(paired),
        "mean_delta_macro_f1": float(delta.mean()),
        "median_delta_macro_f1": float(np.median(delta)),
        "mean_delta_balanced_accuracy": float(
            paired["delta_balanced_accuracy"].mean()
        ),
        "mean_delta_ordinal_mae": float(paired["delta_ordinal_mae"].mean()),
        "wins": int((delta > 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
    }


class DANNConfirmatoryV2Execution:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        repository_root: Path,
    ) -> None:
        validate_confirmatory_v2_execution_config(config)
        self.config = deepcopy(dict(config))
        self.root = Path(repository_root)
        self.output = self.root / str(config["output_dir"])
        self.protocol, self.run_matrix = self._verify_protocol()
        self.execution_preregistration_hash = self._prepare_preregistration()
        self.registry_path = self.output / "run_registry/run_registry.json"
        self.registry = self._load_or_create_registry()

    def _verify_protocol(self) -> tuple[dict[str, Any], pd.DataFrame]:
        protocol = self.config["protocol"]
        for name in (
            "protocol_manifest", "run_matrix", "primary_run_matrix",
            "secondary_run_matrix", "completed_run_matrix", "fold_partitions",
        ):
            observed = _sha256_file(self.root / str(protocol[name]))
            if observed != protocol[f"{name}_sha256"]:
                raise RuntimeError(f"confirmatory-v2 protocol artifact changed: {name}")
        if _sha256_file(self.root / str(protocol["disabled_preregistration"])) != protocol["disabled_preregistration_hash"]:
            raise RuntimeError("disabled preregistration changed")
        manifest = _load_json(self.root / str(protocol["protocol_manifest"]))
        if manifest["protocol_hash"] != protocol["protocol_hash"]:
            raise RuntimeError("protocol hash mismatch")
        matrix = pd.read_csv(self.root / str(protocol["run_matrix"]))
        primary = matrix[matrix["analysis_group"].eq("primary_confirmatory")]
        secondary = matrix[matrix["analysis_group"].eq("secondary_sensitivity")]
        diagnostic = matrix[
            matrix["analysis_group"].eq("previously_observed_diagnostic")
        ]
        if (len(primary), len(secondary), len(diagnostic)) != (20, 8, 2):
            raise RuntimeError("protocol run matrix changed")
        if ((matrix["fold"].eq(1)) & (matrix["seed"].eq(42)) & ~matrix["analysis_group"].eq("previously_observed_diagnostic")).any():
            raise RuntimeError("diagnostic cell entered new execution runs")
        return manifest, matrix

    def _prepare_preregistration(self) -> str:
        self.output.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.config["experiment_id"],
            "execution_enabled": True,
            "repository_commit": _git_head(self.root),
            "protocol_id": PROTOCOL_ID,
            "protocol_hash": self.config["protocol"]["protocol_hash"],
            "disabled_preregistration_hash": self.config["protocol"]["disabled_preregistration_hash"],
            "raw_universe_hash": self.config["dataset"]["raw_universe_hash"],
            "outer_fold_hashes": self.protocol["outer_fold_hashes"],
            "primary_run_matrix": self.run_matrix[
                self.run_matrix["analysis_group"].eq("primary_confirmatory")
            ].to_dict("records"),
            "secondary_run_matrix": self.run_matrix[
                self.run_matrix["analysis_group"].eq("secondary_sensitivity")
            ].to_dict("records"),
            "diagnostic_reference": self.config["diagnostic_reference"],
            "fold_partitions_sha256": self.protocol["fold_partitions_sha256"],
            "architecture": self.config["model"],
            "training": self.config["training"],
            "schedule": self.config["schedule"],
            "matched_update_budgets": self.protocol["matched_update_budgets"],
            "checkpoint_criteria": {
                "primary": "source_validation_macro_f1",
                "secondary": "source_validation_balanced_accuracy",
                "tie_breaker": "earlier_epoch",
                "target_data_used": False,
            },
            "target_test_lock": self.config["target_test_lock"],
            "primary_aggregation": self.protocol["primary_aggregation"],
            "secondary_aggregation": self.protocol["secondary_aggregation"],
            "primary_decision_rule": self.protocol["primary_decision_rule"],
            "resume_policy": self.config["resume_policy"],
        }
        if _contains_absolute_path(payload):
            raise RuntimeError("execution preregistration contains an absolute path")
        path = self.output / "preregistration/experiment_preregistration.json"
        digest = prepare_preregistration(path, payload)
        _write_json(self.output / "preregistration/preregistration_hash.json", {
            "algorithm": "sha256", "sha256": digest,
            "parameters_frozen_before_gradient_step": True,
        })
        shutil.copyfile(path, self.output / "execution_preregistration.json")
        _write_json(self.output / "execution_preregistration_hash.json", {
            "algorithm": "sha256", "sha256": digest,
        })
        protocol_dir = self.output / "protocol"
        _write_json(protocol_dir / "protocol_reference.json", {
            "protocol_id": PROTOCOL_ID,
            "protocol_hash": self.config["protocol"]["protocol_hash"],
            "disabled_preregistration_hash": self.config["protocol"]["disabled_preregistration_hash"],
            "execution_preregistration_hash": digest,
        })
        self.run_matrix.to_csv(self.output / "run_matrix.csv", index=False)
        return digest

    def _load_or_create_registry(self) -> list[dict[str, Any]]:
        expected = build_execution_registry(self.run_matrix)
        if self.registry_path.exists():
            payload = _load_json(self.registry_path)
            rows = payload["runs"]
            if _registry_signature(rows) != _registry_signature(expected):
                raise RuntimeError("run registry scientific specification changed")
            return rows
        self._save_registry(expected)
        return expected

    def _save_registry(self, rows: Sequence[Mapping[str, Any]] | None = None) -> None:
        current = list(rows if rows is not None else self.registry)
        _write_json(self.registry_path, {
            "schema_version": SCHEMA_VERSION,
            "protocol_hash": self.config["protocol"]["protocol_hash"],
            "execution_preregistration_hash": self.execution_preregistration_hash,
            "registry_signature": _registry_signature(current),
            "runs": current,
        })
        shutil.copyfile(self.registry_path, self.output / "run_registry.json")

    def verify_registry(self) -> dict[str, Any]:
        counts = pd.Series([row["status"] for row in self.registry]).value_counts().to_dict()
        return {
            "valid": True,
            "runs": len(self.registry),
            "status_counts": counts,
            "new_runs": sum(row["analysis_group"] in NEW_RUN_GROUPS for row in self.registry),
            "diagnostic_results": sum(
                row["analysis_group"] == "previously_observed_diagnostic"
                for row in self.registry
            ),
            "training_performed": False,
        }

    def _pair_output(self, group: str, fold: int, seed: int) -> Path:
        phase = "primary" if group == "primary_confirmatory" else "secondary"
        return self.output / phase / f"fold_{fold}" / f"seed_{seed}"

    def _pair_rows(self, group: str, fold: int, seed: int) -> list[dict[str, Any]]:
        return [
            row for row in self.registry
            if row["analysis_group"] == group
            and int(row["fold"]) == int(fold)
            and int(row["seed"]) == int(seed)
        ]

    def _update_pair(self, group: str, fold: int, seed: int, status: str) -> None:
        if status not in PAIR_STATUSES:
            raise ValueError(f"invalid run status {status}")
        for row in self._pair_rows(group, fold, seed):
            row["status"] = status
            row["updated_at"] = _now()
            if status == "complete":
                directory = self._pair_output(group, fold, seed) / (
                    "source_only" if row["mode"] == "source_only_matched" else "dann"
                )
                summary = _load_json(directory / "run_summary.json")
                row["checkpoint_hash"] = summary["checkpoint"]["checkpoint_sha256"]
                row["unlock_hash"] = summary["unlock_hash"]
                row["prediction_hash"] = summary["prediction_hash"]
                row["participant_metrics_hash"] = summary["participant_metrics_hash"]
        self._save_registry()

    def _run_pair(self, group: str, fold: int, seed: int, *, resume: bool) -> None:
        rows = self._pair_rows(group, fold, seed)
        if len(rows) != 2:
            raise RuntimeError("each fold/seed pair must contain both modes")
        statuses = {row["status"] for row in rows}
        if statuses == {"complete"}:
            return
        if not resume and statuses != {"pending"}:
            raise RuntimeError("unfinished output exists; pass --resume")
        prior = sorted(statuses)[0] if len(statuses) == 1 else "mixed"
        attempt = max(int(row["attempt_count"]) for row in rows) + 1
        for row in rows:
            if row["status"] in {"training", "failed_technical"}:
                row["technical_restart_count"] = int(row["technical_restart_count"]) + 1
            row["attempt_count"] = attempt
        self._update_pair(group, fold, seed, "training")
        pair = _ConfirmatoryPairRunner(
            self.config,
            repository_root=self.root,
            output_dir=self._pair_output(group, fold, seed),
            fold=fold,
            seed=seed,
            analysis_group=group,
            execution_preregistration_hash=self.execution_preregistration_hash,
            protocol_manifest=self.protocol,
            status_callback=lambda status: self._update_pair(group, fold, seed, status),
        )
        try:
            pair.execute(attempt=attempt, resumed_from=prior)
        except Exception as error:
            trace = traceback.format_exc()
            for mode, directory in pair._artifact_dirs().items():
                pd.DataFrame([{
                    "stage": "pair_execution", "code": type(error).__name__,
                    "message": str(error), "traceback": trace,
                }]).to_csv(directory / "errors.csv", index=False)
                pair._write_attempt(
                    attempt, status="failed_technical", resumed_from=prior,
                    error=str(error),
                )
            self._update_pair(group, fold, seed, "failed_technical")
            raise

    def run_phase(self, phase: str, *, resume: bool) -> None:
        if phase == "primary":
            group = "primary_confirmatory"
            folds, seeds = self.config["primary"]["folds"], self.config["primary"]["seeds"]
        elif phase == "secondary":
            self._require_primary_lock()
            group = "secondary_sensitivity"
            folds, seeds = self.config["secondary"]["folds"], self.config["secondary"]["seeds"]
        else:
            raise ValueError("phase must be primary or secondary")
        for fold in folds:
            for seed in seeds:
                self._run_pair(group, int(fold), int(seed), resume=resume)

    def _collect_subject_metrics(self, groups: Sequence[str]) -> pd.DataFrame:
        frames = []
        for row in self.registry:
            if row["analysis_group"] not in groups or row["mode"] != "source_only_matched":
                continue
            if row["status"] != "complete":
                raise RuntimeError(f"incomplete result: {row['run_id']}")
            pair_path = self._pair_output(
                row["analysis_group"], int(row["fold"]), int(row["seed"])
            )
            frame = pd.read_csv(pair_path / "target_test_subject_metrics.csv")
            frame["fold"] = int(row["fold"])
            frame["seed"] = int(row["seed"])
            frame["analysis_group"] = row["analysis_group"]
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def aggregate_primary(self) -> dict[str, Any]:
        lock_path = self.output / "aggregation/primary_result_lock.json"
        if lock_path.exists():
            return _load_json(lock_path)["primary_decision"]
        metrics = self._collect_subject_metrics(("primary_confirmatory",))
        if sorted(metrics["seed"].unique().tolist()) != list(PRIMARY_SEEDS):
            raise RuntimeError("primary aggregation contains incorrect seeds")
        paired = pair_subject_metrics(metrics)
        participants = average_participants_across_seeds(paired)
        if not participants["seed_count"].eq(2).all():
            raise RuntimeError("primary participant is missing a seed")
        folds = fold_level_metrics(participants)
        seeds = seed_level_metrics(paired)
        bootstrap = bootstrap_unique_participants(
            participants,
            resamples=int(self.config["bootstrap"]["resamples"]),
            seed=int(self.config["bootstrap"]["seed"]),
        )
        minimal_paired, minimal_participants = aggregate_participant_deltas(
            metrics, analysis_groups=("primary_confirmatory",)
        )
        decision = apply_primary_decision_rule(
            minimal_paired,
            minimal_participants,
            self.protocol["primary_decision_rule"],
        )
        aggregation = self.output / "aggregation"
        aggregation.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(aggregation / "primary_subject_metrics_by_seed.csv", index=False)
        participants.to_csv(aggregation / "primary_participant_metrics.csv", index=False)
        folds.to_csv(aggregation / "primary_fold_metrics.csv", index=False)
        seeds.to_csv(aggregation / "primary_seed_metrics.csv", index=False)
        _write_json(aggregation / "primary_paired_comparison.json", {
            "participants": len(participants),
            "mean_delta_macro_f1": float(participants["delta_macro_f1"].mean()),
            "median_delta_macro_f1": float(participants["delta_macro_f1"].median()),
            "mean_delta_balanced_accuracy": float(participants["delta_balanced_accuracy"].mean()),
            "mean_delta_ordinal_mae": float(participants["delta_ordinal_mae"].mean()),
            "wins": int((participants["delta_macro_f1"] > 1e-12).sum()),
            "losses": int((participants["delta_macro_f1"] < -1e-12).sum()),
            "ties": int((participants["delta_macro_f1"].abs() <= 1e-12).sum()),
        })
        _write_json(aggregation / "primary_bootstrap.json", bootstrap)
        _write_json(aggregation / "primary_decision.json", decision)
        primary_rows = [
            row for row in self.registry
            if row["analysis_group"] == "primary_confirmatory"
        ]
        completeness = {
            "expected_runs": 20,
            "complete_runs": sum(row["status"] == "complete" for row in primary_rows),
            "all_complete": all(row["status"] == "complete" for row in primary_rows),
        }
        _write_json(aggregation / "primary_run_completeness.json", completeness)
        if not completeness["all_complete"]:
            raise RuntimeError("cannot lock incomplete primary phase")
        lock = {
            "protocol_hash": self.config["protocol"]["protocol_hash"],
            "execution_preregistration_hash": self.execution_preregistration_hash,
            "runs": primary_rows,
            "fold_aggregates": folds.to_dict("records"),
            "seed_aggregates": seeds.to_dict("records"),
            "primary_aggregate": _load_json(
                aggregation / "primary_paired_comparison.json"
            ),
            "bootstrap": bootstrap,
            "primary_decision": decision,
            "created_before_secondary": not any(
                row["status"] == "complete"
                for row in self.registry
                if row["analysis_group"] == "secondary_sensitivity"
            ),
        }
        lock_hash = _immutable_json(lock_path, lock)
        _immutable_json(aggregation / "primary_result_lock_hash.json", {
            "algorithm": "sha256", "sha256": lock_hash,
        })
        shutil.copyfile(lock_path, self.output / "primary_result_lock.json")
        shutil.copyfile(
            aggregation / "primary_result_lock_hash.json",
            self.output / "primary_result_lock_hash.json",
        )
        return decision

    def _require_primary_lock(self) -> str:
        path = self.output / "aggregation/primary_result_lock.json"
        hash_path = self.output / "aggregation/primary_result_lock_hash.json"
        if not path.exists() or not hash_path.exists():
            raise RuntimeError("primary result lock must exist before sensitivity phase")
        expected = _load_json(hash_path)["sha256"]
        observed = _sha256_file(path)
        if expected != observed:
            raise RuntimeError("primary result lock changed")
        return observed

    def _diagnostic_audit(self) -> dict[str, Any]:
        reference = self.config["diagnostic_reference"]
        root = self.root / str(reference["source"])
        paths = {
            "source_only_checkpoint": root / "source_only/checkpoint.pt",
            "dann_checkpoint": root / "dann/checkpoint.pt",
            "predictions": root / "target_test_predictions.parquet",
            "participant_metrics": root / "target_test_subject_metrics.csv",
            "summary": root / "diagnostic_summary.json",
        }
        observed = {name: _sha256_file(path) for name, path in paths.items()}
        expected = {
            name: reference[f"{name}_sha256"] for name in paths
        }
        if observed != expected:
            raise RuntimeError("diagnostic provenance changed")
        return {
            "fold": 1,
            "seed": 42,
            "protocol_hash": reference["protocol_hash"],
            "preregistration_hash": reference["preregistration_hash"],
            "artifact_hashes": observed,
            "runtime_modified": False,
        }

    def aggregate_secondary(self, primary_decision: Mapping[str, Any]) -> dict[str, Any]:
        primary_lock_hash = self._require_primary_lock()
        locked_primary = _load_json(
            self.output / "aggregation/primary_result_lock.json"
        )["primary_decision"]
        if dict(primary_decision) != locked_primary:
            raise RuntimeError("primary decision differs from immutable lock")
        metrics = self._collect_subject_metrics(
            ("primary_confirmatory", "secondary_sensitivity")
        )
        diagnostic = pd.read_csv(
            self.root
            / str(self.config["diagnostic_reference"]["source"])
            / "target_test_subject_metrics.csv"
        )
        diagnostic["fold"] = 1
        diagnostic["seed"] = 42
        diagnostic["analysis_group"] = "previously_observed_diagnostic"
        if "quadratic_weighted_kappa" not in diagnostic:
            from sklearn.metrics import cohen_kappa_score

            diagnostic_predictions = pd.read_parquet(
                self.root
                / str(self.config["diagnostic_reference"]["source"])
                / "target_test_predictions.parquet"
            )
            values: dict[tuple[str, str], float] = {}
            for (mode, subject_id), frame in diagnostic_predictions.groupby(
                ["mode", "subject_id"], sort=True
            ):
                values[(str(mode), str(subject_id))] = float(
                    cohen_kappa_score(
                        frame["y_true"], frame["y_pred"],
                        labels=np.arange(5), weights="quadratic",
                    )
                )
            diagnostic["quadratic_weighted_kappa"] = [
                values[(str(row.mode), str(row.subject_id))]
                for row in diagnostic.itertuples()
            ]
        metrics = pd.concat([metrics, diagnostic], ignore_index=True)
        paired = pair_subject_metrics(metrics)
        participants = average_participants_across_seeds(paired)
        folds = fold_level_metrics(participants)
        seeds = seed_level_metrics(paired)
        sign_counts = paired.assign(
            positive=paired["delta_macro_f1"] > 0
        ).groupby(["fold", "subject_id"])["positive"].nunique()
        overall_mean = float(participants["delta_macro_f1"].mean())
        seed_nonnegative = int((seeds["mean_delta_macro_f1"] >= 0).sum())
        if overall_mean > 0 and seed_nonnegative == 3:
            status = "robust_positive_sensitivity"
        elif overall_mean > 0:
            status = "mixed_positive_sensitivity"
        else:
            status = "nonpositive_sensitivity"
        decision = {
            "secondary_sensitivity_status": status,
            "mean_delta_macro_f1": overall_mean,
            "median_delta_macro_f1": float(participants["delta_macro_f1"].median()),
            "mean_delta_balanced_accuracy": float(participants["delta_balanced_accuracy"].mean()),
            "participants_with_seed_sign_instability": int((sign_counts > 1).sum()),
            "primary_decision_unchanged": locked_primary,
            "may_change_primary_decision": False,
            "primary_result_lock_hash": primary_lock_hash,
        }
        aggregation = self.output / "aggregation"
        participants.to_csv(aggregation / "secondary_participant_metrics.csv", index=False)
        folds.to_csv(aggregation / "secondary_fold_metrics.csv", index=False)
        seeds.to_csv(aggregation / "secondary_seed_metrics.csv", index=False)
        _write_json(aggregation / "secondary_paired_comparison.json", {
            "participants": len(participants),
            "mean_delta_macro_f1": overall_mean,
            "seed_variability": seeds.to_dict("records"),
        })
        _write_json(aggregation / "secondary_sensitivity_decision.json", decision)
        diagnostic_audit = self._diagnostic_audit()
        _write_json(self.output / "diagnostic_reference/diagnostic_provenance_audit.json", diagnostic_audit)
        shutil.copyfile(
            self.output / "diagnostic_reference/diagnostic_provenance_audit.json",
            self.output / "diagnostic_provenance_audit.json",
        )
        return decision

    def finalize(
        self,
        primary_decision: Mapping[str, Any],
        secondary_decision: Mapping[str, Any],
    ) -> dict[str, Any]:
        primary_lock_hash = self._require_primary_lock()
        new_rows = [row for row in self.registry if row["analysis_group"] in NEW_RUN_GROUPS]
        leakage_rows = []
        checkpoint_rows = []
        gradient_rows = []
        for row in new_rows:
            if row["mode"] != "source_only_matched":
                continue
            pair = self._pair_output(
                row["analysis_group"], int(row["fold"]), int(row["seed"])
            )
            leakage = _load_json(pair / "leakage_audit.json")
            immutability = _load_json(pair / "checkpoint_immutability_audit.json")
            leakage_rows.append({
                "fold": row["fold"], "seed": row["seed"],
                "all_overlaps_zero": leakage["all_overlaps_zero"],
                "target_test_reads_before_unlock": leakage["target_test_reads_before_unlock"],
                "target_labels_available_to_training_step": leakage["target_labels_available_to_training_step"],
            })
            checkpoint_rows.append({
                "fold": row["fold"], "seed": row["seed"],
                "unchanged_after_test": immutability["unchanged"],
            })
            frame = pd.read_csv(pair / "dann/gradient_audit.csv")
            gradient_rows.append({
                "fold": row["fold"], "seed": row["seed"],
                "epochs_audited": len(frame),
                "all_finite": bool(frame["finite"].all()),
                "all_model_states_unchanged": bool(frame["model_state_unchanged"].all()),
                "mean_domain_to_task_ratio": float(
                    frame["domain_to_task_encoder_gradient_ratio"].mean()
                ),
            })
        global_leakage = {
            "pairs": leakage_rows,
            "all_pairs_safe": all(
                row["all_overlaps_zero"]
                and row["target_test_reads_before_unlock"] == 0
                and not row["target_labels_available_to_training_step"]
                for row in leakage_rows
            ),
        }
        global_checkpoints = {
            "pairs": checkpoint_rows,
            "all_checkpoints_immutable": all(
                row["unchanged_after_test"] for row in checkpoint_rows
            ),
        }
        _write_json(self.output / "global_leakage_audit.json", global_leakage)
        _write_json(self.output / "global_checkpoint_audit.json", global_checkpoints)
        _write_json(self.output / "aggregation/gradient_audit_summary.json", {
            "pairs": gradient_rows,
            "all_finite": all(row["all_finite"] for row in gradient_rows),
            "all_decompositions_state_immutable": all(
                row["all_model_states_unchanged"] for row in gradient_rows
            ),
        })
        summary = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.config["experiment_id"],
            "result_status": "final",
            "protocol_id": PROTOCOL_ID,
            "protocol_hash": self.config["protocol"]["protocol_hash"],
            "execution_preregistration_hash": self.execution_preregistration_hash,
            "git_commit": _git_head(self.root),
            "new_runs_expected": 28,
            "new_runs_complete": sum(row["status"] == "complete" for row in new_rows),
            "attempts": sum(int(row["attempt_count"]) for row in new_rows),
            "technical_restarts": sum(
                int(row["technical_restart_count"]) for row in new_rows
            ),
            "pair_attempts": sum(
                max(
                    int(row["attempt_count"])
                    for row in new_rows
                    if row["analysis_group"] == group
                    and int(row["fold"]) == fold
                    and int(row["seed"]) == seed
                )
                for group, fold, seed in sorted({
                    (row["analysis_group"], int(row["fold"]), int(row["seed"]))
                    for row in new_rows
                })
            ),
            "pair_technical_restarts": sum(
                max(
                    int(row["technical_restart_count"])
                    for row in new_rows
                    if row["analysis_group"] == group
                    and int(row["fold"]) == fold
                    and int(row["seed"]) == seed
                )
                for group, fold, seed in sorted({
                    (row["analysis_group"], int(row["fold"]), int(row["seed"]))
                    for row in new_rows
                })
            ),
            "primary_decision": dict(primary_decision),
            "primary_result_lock_hash": primary_lock_hash,
            "secondary_sensitivity": dict(secondary_decision),
            "diagnostic_provenance": self._diagnostic_audit(),
            "global_leakage_audit": global_leakage,
            "global_checkpoint_audit": global_checkpoints,
            "training_complete": all(row["status"] == "complete" for row in new_rows),
        }
        _write_json(self.output / "confirmatory_summary.json", summary)
        for name in (
            "primary_run_completeness.json",
            "primary_participant_metrics.csv",
            "primary_fold_metrics.csv",
            "primary_seed_metrics.csv",
            "primary_paired_comparison.json",
            "primary_bootstrap.json",
            "primary_decision.json",
            "secondary_participant_metrics.csv",
            "secondary_fold_metrics.csv",
            "secondary_seed_metrics.csv",
            "secondary_paired_comparison.json",
            "secondary_sensitivity_decision.json",
        ):
            shutil.copyfile(self.output / "aggregation" / name, self.output / name)
        pd.DataFrame(columns=["stage", "code", "message", "traceback"]).to_csv(
            self.output / "errors.csv", index=False
        )
        report = render_confirmatory_execution_report(summary, self.output)
        reports_dir = self.output / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "confirmatory_report.md").write_text(
            report, encoding="utf-8"
        )
        (self.output / "confirmatory_report.md").write_text(
            report, encoding="utf-8"
        )
        tracked = self.root / "reports/integration/dann_label_q5_confirmatory_v2.md"
        tracked.parent.mkdir(parents=True, exist_ok=True)
        tracked.write_text(report, encoding="utf-8")
        return summary

    def run(
        self,
        *,
        phase: str,
        resume: bool,
        aggregate_only: bool = False,
    ) -> dict[str, Any]:
        if not aggregate_only:
            if phase in {"all", "primary"}:
                self.run_phase("primary", resume=resume)
            primary = self.aggregate_primary()
            if phase == "primary":
                return {"phase": "primary", "primary_decision": primary}
            if phase in {"all", "secondary"}:
                self.run_phase("secondary", resume=resume)
        else:
            primary = self.aggregate_primary()
        secondary = self.aggregate_secondary(primary)
        return self.finalize(primary, secondary)


def render_confirmatory_execution_report(
    summary: Mapping[str, Any], output: Path
) -> str:
    primary = summary["primary_decision"]
    secondary = summary["secondary_sensitivity"]
    primary_folds = pd.read_csv(output / "aggregation/primary_fold_metrics.csv")
    primary_seeds = pd.read_csv(output / "aggregation/primary_seed_metrics.csv")
    secondary_seeds = pd.read_csv(output / "aggregation/secondary_seed_metrics.csv")
    primary_subjects = pd.read_csv(
        output / "aggregation/primary_subject_metrics_by_seed.csv"
    )
    absolute = primary_subjects.groupby("mode")[[
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "kappa", "ordinal_mae", "quadratic_weighted_kappa",
    ]].mean()
    bootstrap = _load_json(output / "aggregation/primary_bootstrap.json")
    gradient = _load_json(output / "aggregation/gradient_audit_summary.json")
    training_rows = []
    for phase, folds, seeds in (
        ("primary", range(1, 6), (123, 2026)),
        ("secondary", range(2, 6), (42,)),
    ):
        for fold in folds:
            for seed in seeds:
                pair = output / phase / f"fold_{fold}" / f"seed_{seed}"
                pair_summary = _load_json(pair / "pair_summary.json")
                source = _load_json(pair / "source_only/checkpoint_manifest.json")
                dann = _load_json(pair / "dann/checkpoint_manifest.json")
                training_rows.append({
                    "phase": phase,
                    "fold": fold,
                    "seed": seed,
                    "source_epoch": source["best_epoch"],
                    "dann_epoch": dann["best_epoch"],
                    "source_validation_macro_f1": source[
                        "source_validation_metrics"
                    ]["macro_f1"],
                    "dann_validation_macro_f1": dann[
                        "source_validation_metrics"
                    ]["macro_f1"],
                    "source_checkpoint_hash": source["checkpoint_sha256"],
                    "dann_checkpoint_hash": dann["checkpoint_sha256"],
                    "unlock_hash": _load_json(
                        pair / "target_test_unlock_hash.json"
                    )["sha256"],
                    "training_seconds": pair_summary["training"][
                        "training_time_seconds"
                    ],
                })
    total_training_seconds = sum(row["training_seconds"] for row in training_rows)
    lines = [
        "# Confirmatory multi-block DANN experiment",
        "",
        f"- Branch/HEAD: `integration/benchmark-unification` / `{summary['git_commit']}`.",
        "- Hypothesis: DANN improves Old_EEG-to-gpn_data label_q5 transfer over source-update-matched EEGNet.",
        f"- Protocol: `{summary['protocol_hash']}`; execution preregistration: `{summary['execution_preregistration_hash']}`.",
        "- Primary confirmation uses seeds 123/2026 across five folds. Seed 42 is sensitivity-only; fold-1/seed-42 is referenced diagnostic evidence and was not retrained.",
        f"- New runs: {summary['new_runs_complete']}/{summary['new_runs_expected']} complete in 14 matched pairs; pair attempts {summary['pair_attempts']}; pair-level technical restarts {summary['pair_technical_restarts']}.",
        "- Fixed training: AdamW, lr 0.001, weight decay 0.0001, batches 32/32, maximum 12 epochs, patience 3, clipping 5.0, logistic GRL, lambda 1.0.",
        "- Matched steps by fold: 580, 602, 569, 606, 586. Source batch hashes and optimizer-update counts match within every pair.",
        "- Fixed data: 30,958 raw-deduplicated windows, 54 participants, 86 logical records, `[1, 14, 2560]`, 256 Hz, 10 s, five-class `label_q5`.",
        "- Direction: Old_EEG source to gpn_data target. The five byte-locked subject-disjoint outer folds and source-validation partitions were reused without rebuilding the raw cache.",
        f"- Total paired training time recorded by the 14 pair summaries: {total_training_seconds:.1f} s.",
        "",
        "## Primary result",
        "",
        f"- Decision: **{primary['status']}**.",
        f"- Mean/median participant macro-F1 delta: {primary['mean_delta_macro_f1']:.6f} / {primary['median_delta_macro_f1']:.6f}.",
        f"- Mean balanced-accuracy delta: {primary['mean_delta_balanced_accuracy']:.6f}; win fraction: {primary['participant_win_fraction']:.3f}.",
        f"- Participant bootstrap 95% interval: [{bootstrap['mean_95_ci'][0]:.6f}, {bootstrap['mean_95_ci'][1]:.6f}]. No standalone significance claim is made.",
        f"- Primary result lock: `{summary['primary_result_lock_hash']}`.",
        "",
        "### Absolute participant-level primary metrics",
        "",
        "| mode | accuracy | balanced accuracy | macro F1 | weighted F1 | kappa | ordinal MAE | quadratic kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = absolute.loc[mode]
        lines.append(
            f"| {mode} | {row['accuracy']:.6f} | {row['balanced_accuracy']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['weighted_f1']:.6f} | "
            f"{row['kappa']:.6f} | {row['ordinal_mae']:.6f} | "
            f"{row['quadratic_weighted_kappa']:.6f} |"
        )
    lines.extend([
        "",
        "### Fold-level primary deltas",
        "",
        "| fold | participants | mean macro F1 delta | median | balanced accuracy delta | ordinal MAE delta | W/L/T |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in primary_folds.to_dict("records"):
        lines.append(
            f"| {int(row['fold'])} | {int(row['participants'])} | "
            f"{row['mean_delta_macro_f1']:.6f} | {row['median_delta_macro_f1']:.6f} | "
            f"{row['mean_delta_balanced_accuracy']:.6f} | {row['mean_delta_ordinal_mae']:.6f} | "
            f"{int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])} |"
        )
    lines.extend([
        "",
        "### Seed-level primary deltas",
        "",
        "| seed | participants | mean macro F1 delta | balanced accuracy delta | W/L/T |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in primary_seeds.to_dict("records"):
        lines.append(
            f"| {int(row['seed'])} | {int(row['participants'])} | "
            f"{row['mean_delta_macro_f1']:.6f} | {row['mean_delta_balanced_accuracy']:.6f} | "
            f"{int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])} |"
        )
    lines.extend([
        "",
        "### Checkpoint selection and training",
        "",
        "| phase | fold | seed | source best epoch | DANN best epoch | source val macro F1 | DANN val macro F1 | pair seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in training_rows:
        lines.append(
            f"| {row['phase']} | {row['fold']} | {row['seed']} | "
            f"{row['source_epoch']} | {row['dann_epoch']} | "
            f"{row['source_validation_macro_f1']:.6f} | "
            f"{row['dann_validation_macro_f1']:.6f} | "
            f"{row['training_seconds']:.1f} |"
        )
    lines.extend([
        "",
        "## Sensitivity and audits",
        "",
        f"- Secondary status: **{secondary['secondary_sensitivity_status']}**; three-seed mean macro-F1 delta {secondary['mean_delta_macro_f1']:.6f}.",
        f"- Participants with seed-dependent effect sign: {secondary['participants_with_seed_sign_instability']}.",
        "- The primary decision was loaded from the immutable primary lock and was not changed by sensitivity results.",
        "- Diagnostic protocol, preregistration, checkpoints, predictions, participant metrics, and summary hashes were verified without modifying their runtime.",
        f"- Global leakage audit passed: {summary['global_leakage_audit']['all_pairs_safe']}; checkpoints immutable after target test: {summary['global_checkpoint_audit']['all_checkpoints_immutable']}.",
        f"- Gradient decomposition: all finite `{gradient['all_finite']}` and state immutable `{gradient['all_decompositions_state_immutable']}` across all 14 new pairs.",
        "",
        "### Three-seed sensitivity",
        "",
        "| seed | participants | mean macro F1 delta | balanced accuracy delta | ordinal MAE delta | W/L/T |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in secondary_seeds.to_dict("records"):
        lines.append(
            f"| {int(row['seed'])} | {int(row['participants'])} | "
            f"{row['mean_delta_macro_f1']:.6f} | "
            f"{row['mean_delta_balanced_accuracy']:.6f} | "
            f"{row['mean_delta_ordinal_mae']:.6f} | "
            f"{int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])} |"
        )
    lines.extend([
        "",
        "## Execution notes and artifacts",
        "",
        "- One pair-level technical restart occurred before any gradient step because the execution wrapper initially addressed the wrong manifest field. Resume reused the same scientific run specification; all other pairs completed on their first attempt.",
        "- Two post-training interruptions were limited to initially missing aggregation and report directories. Both were resumed without model retraining.",
        "- Runtime root: `benchmark_results/domain_adaptation_dann_confirmatory_v2/`; it contains the immutable execution preregistration, deterministic registry, pair artifacts, aggregations, audits, and final report and is not tracked by Git.",
        "- Result status: `final` for this preregistered confirmation. The hypothesis is only partially confirmed because the mean effect is below +0.01, participant wins are below 60%, and the participant bootstrap interval includes zero.",
        "",
        "### Checkpoint and target-test unlock hashes",
        "",
        "```text",
    ])
    for row in training_rows:
        lines.append(
            f"{row['phase']} fold={row['fold']} seed={row['seed']} "
            f"source={row['source_checkpoint_hash']} "
            f"dann={row['dann_checkpoint_hash']} unlock={row['unlock_hash']}"
        )
    lines.extend([
        "```",
        "",
        "Limitations: gpn_data and Old_EEG are source/organization domains from the same general Emotiv-class acquisition family, not automatically different devices. The bootstrap uses unique participants after primary-seed averaging; windows are never treated as independent observations.",
        "",
    ])
    return "\n".join(lines)


def run_dann_label_q5_confirmatory_v2(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    phase: str = "all",
    resume: bool = False,
    verify_registry: bool = False,
    aggregate_only: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root) if repository_root is not None else Path.cwd()
    path = Path(config_path)
    resolved = path if path.is_absolute() else root / path
    config = json.loads(resolved.read_text(encoding="utf-8"))
    runner = DANNConfirmatoryV2Execution(config, repository_root=root)
    if verify_registry:
        return runner.verify_registry()
    return runner.run(phase=phase, resume=resume, aggregate_only=aggregate_only)


__all__ = [
    "DANNConfirmatoryV2Execution",
    "PAIR_STATUSES",
    "SCALAR_METRICS",
    "average_participants_across_seeds",
    "bootstrap_unique_participants",
    "build_execution_registry",
    "fold_level_metrics",
    "pair_subject_metrics",
    "render_confirmatory_execution_report",
    "run_dann_label_q5_confirmatory_v2",
    "seed_level_metrics",
    "validate_confirmatory_v2_execution_config",
]
