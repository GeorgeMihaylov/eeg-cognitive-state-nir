"""Authorized one-fold FOMAML diagnostic on the raw-deduplicated protocol."""

from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from bench.datasets.raw_eeg_window_dataset import CANONICAL_EEG_CHANNELS
from bench.meta import FOMAMLConfig, FirstOrderMAML
from bench.experiments.fomaml_label_q5_diagnostic import (
    FOMAMLLabelQ5Diagnostic,
    SCHEMA_VERSION,
    _git_head,
    _jsonable,
    _sha256_file,
    _write_json,
    audit_raw_episode_alignment,
    prepare_preregistration,
    resolve_device,
)
from bench.experiments.fomaml_label_q5_raw_protocol import compute_protocol_hash


RAW_DIAGNOSTIC_SCHEMA_VERSION = "fomaml-label-q5-raw-diagnostic-v1"
EXPECTED_PROTOCOL_ID = "fomaml_label_q5_raw_deduplicated_v2"
EXPECTED_PROTOCOL_HASH = (
    "e73703a443aea3b34f62606efa76bd592ff70099a30cdca80d292f1d76a1fd60"
)
EXPECTED_RAW_UNIVERSE_HASH = (
    "308fdd96523565417d0fc2f3b3bfdea74639224ce1ee44f5e87af6cc0b7e94cf"
)
EXPECTED_SCOPE_COUNTS = {"meta_train": 11, "meta_validation": 5, "outer_test": 5}


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, str) and bool(re.search(r"[A-Za-z]:[\\/]", value))


def validate_raw_diagnostic_config(config: Mapping[str, Any]) -> None:
    """Fail closed if any preregistered scientific choice has drifted."""
    if config.get("execution_enabled") is not True:
        raise ValueError("Raw diagnostic must explicitly set execution_enabled=true")
    if config.get("experiment_id") != "fomaml_label_q5_raw_deduplicated_diagnostic_v1":
        raise ValueError("Unexpected raw diagnostic experiment_id")
    protocol = config["protocol"]
    if protocol.get("id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("Only the task-8C raw-deduplicated protocol is executable")
    if protocol.get("expected_hash") != EXPECTED_PROTOCOL_HASH:
        raise ValueError("Raw protocol hash changed")
    if protocol.get("raw_universe_hash") != EXPECTED_RAW_UNIVERSE_HASH:
        raise ValueError("Raw universe hash changed")
    if int(protocol["outer_fold"]) != 1 or int(config["seed"]) != 42:
        raise ValueError("Task 8Ch is restricted to outer fold 1 and seed 42")
    if config["model"]["name"] != "torch_eegnet":
        raise ValueError("Task 8Ch is restricted to production EEGNet")
    if list(config["model"]["input_shape"] if "input_shape" in config["model"] else config["dataset"]["input_shape"]) != [1, 14, 2560]:
        raise ValueError("Production EEGNet input shape must be [1,14,2560]")
    if list(config["fomaml"]["buffer_policies"]) != [
        "frozen_global", "support_local"
    ]:
        raise ValueError("Both preregistered BatchNorm policies are required")
    if int(config["fomaml"]["inner_steps"]) != 1:
        raise ValueError("inner_steps is fixed at one")
    if float(config["fomaml"]["inner_learning_rate"]) != 0.01:
        raise ValueError("inner_learning_rate is fixed at 0.01")
    if dict(protocol["episode_counts"]) != EXPECTED_SCOPE_COUNTS:
        raise ValueError("Episode counts changed")
    if _contains_absolute_path(config):
        raise ValueError("Tracked diagnostic config must not contain absolute paths")


def validate_raw_episode_protocol(
    protocol: Mapping[str, Any],
    episodes: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact task-8C protocol without rebuilding a split."""
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("Raw episode protocol ID changed")
    if protocol.get("protocol_hash") != config["protocol"]["expected_hash"]:
        raise ValueError("Raw episode protocol hash changed")
    recomputed = compute_protocol_hash({
        key: value for key, value in protocol.items() if key != "protocol_hash"
    })
    if recomputed != protocol["protocol_hash"]:
        raise ValueError("Raw episode protocol semantic hash is not reproducible")
    dataset_signature = protocol["dataset_cache_signature"]
    if dataset_signature["raw_universe_hash"] != config["protocol"]["raw_universe_hash"]:
        raise ValueError("Raw universe hash differs from the protocol")
    outer = protocol["outer_fold_manifest"]
    if outer["source_fold_assignments_sha256"] != config["protocol"]["outer_fold_artifact_sha256"]:
        raise ValueError("Outer fold artifact changed")
    if outer["outer_split_hash"] != config["protocol"]["outer_split_hash"]:
        raise ValueError("Outer split semantic hash changed")
    if protocol["meta_split"]["meta_split_hash"] != config["protocol"]["meta_split_hash"]:
        raise ValueError("Nested meta split changed")
    if protocol["episode_spec"]["episode_spec_hash"] != config["protocol"]["episode_spec_hash"]:
        raise ValueError("Episode specification changed")

    observed = {
        str(key): int(value)
        for key, value in episodes["scope"].value_counts().items()
    }
    if observed != EXPECTED_SCOPE_COUNTS:
        raise ValueError(f"Raw episode scope counts changed: {observed}")
    if episodes["episode_id"].astype(str).duplicated().any():
        raise ValueError("Raw episode IDs are not unique")
    manifest_ids = {str(row["episode_id"]) for row in protocol["episodes"]}
    parquet_ids = set(episodes["episode_id"].astype(str))
    if manifest_ids != parquet_ids:
        raise ValueError("Parquet episode IDs differ from protocol manifest")

    subjects = {
        scope: set(episodes.loc[episodes["scope"].eq(scope), "subject_id"].astype(str))
        for scope in EXPECTED_SCOPE_COUNTS
    }
    expected_subjects = {
        "meta_train": set(protocol["meta_split"]["meta_train_subjects"]),
        "meta_validation": set(protocol["meta_split"]["meta_validation_subjects"]),
        "outer_test": set(protocol["eligible_participants"]["outer_test"]),
    }
    if subjects != expected_subjects:
        raise ValueError("Episode subjects differ from the fixed raw protocol")
    if any(
        subjects[left] & subjects[right]
        for left, right in (
            ("meta_train", "meta_validation"),
            ("meta_train", "outer_test"),
            ("meta_validation", "outer_test"),
        )
    ):
        raise ValueError("Raw meta-learning subject partitions overlap")

    seen_samples: set[str] = set()
    support_sizes: list[int] = []
    query_sizes: list[int] = []
    for row in episodes.itertuples():
        support = tuple(map(str, row.support_sample_ids))
        query = tuple(map(str, row.query_sample_ids))
        if len(set(support)) != len(support) or len(set(query)) != len(query):
            raise ValueError("Episode contains duplicate sample IDs")
        if set(support) & set(query):
            raise ValueError("Episode support/query sample IDs overlap")
        if set(map(str, row.support_record_ids)) & set(map(str, row.query_record_ids)):
            raise ValueError("Episode support/query logical records overlap")
        if str(row.split_level) != "record":
            raise ValueError("Window-level fallback is forbidden")
        if set(np.asarray(row.support_targets, dtype=np.int64)) != set(range(5)):
            raise ValueError("Support partition does not contain all five classes")
        if set(np.asarray(row.query_targets, dtype=np.int64)) != set(range(5)):
            raise ValueError("Query partition does not contain all five classes")
        if seen_samples & (set(support) | set(query)):
            raise ValueError("A raw sample is referenced by multiple episodes")
        seen_samples.update(support)
        seen_samples.update(query)
        support_sizes.append(len(support))
        query_sizes.append(len(query))
    leakage = protocol["leakage_audit"]
    if not leakage["valid"] or int(leakage["missing_raw_ids"]) != 0:
        raise ValueError("Task-8C leakage audit is not valid")
    return {
        "valid": True,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_hash": protocol["protocol_hash"],
        "raw_universe_hash": dataset_signature["raw_universe_hash"],
        "scope_counts": observed,
        "subject_counts": {key: len(value) for key, value in subjects.items()},
        "subject_overlap": 0,
        "support_query_sample_overlap": 0,
        "support_query_record_overlap": 0,
        "duplicate_episode_sample_references": 0,
        "missing_raw_ids": 0,
        "unsafe_fallback_used": False,
        "support_windows": {"minimum": min(support_sizes), "maximum": max(support_sizes)},
        "query_windows": {"minimum": min(query_sizes), "maximum": max(query_sizes)},
    }


def paired_subject_bootstrap(
    comparison: Mapping[str, Any], *, resamples: int, seed: int
) -> dict[str, Any]:
    """Diagnostic paired bootstrap over participants, never over windows."""
    rows = list(comparison["subjects"])
    if not rows:
        raise ValueError("Paired bootstrap requires participant rows")
    rng = np.random.default_rng(int(seed))
    metrics = ("macro_f1", "balanced_accuracy", "ordinal_mae")
    result: dict[str, Any] = {
        "unit": "subject",
        "n_subjects": len(rows),
        "resamples": int(resamples),
        "seed": int(seed),
        "diagnostic_only": True,
        "statistical_significance_claimed": False,
    }
    for metric in metrics:
        values = np.asarray([row[f"delta_{metric}"] for row in rows], dtype=float)
        sampled = values[rng.integers(0, len(values), size=(int(resamples), len(values)))]
        means = sampled.mean(axis=1)
        result[metric] = {
            "observed_mean": float(values.mean()),
            "percentile_95_low": float(np.quantile(means, 0.025)),
            "percentile_95_high": float(np.quantile(means, 0.975)),
        }
    return result


def build_support_budget_analysis(
    episodes: pd.DataFrame, subject_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Describe variable complete-record support and its paired FOMAML gain."""
    fomaml = subject_metrics.loc[
        subject_metrics["mode"].eq("selected_fomaml"),
        ["subject_id", "macro_f1", "balanced_accuracy"],
    ].set_index("subject_id")
    supervised = subject_metrics.loc[
        subject_metrics["mode"].eq("supervised_full_model"),
        ["subject_id", "macro_f1", "balanced_accuracy"],
    ].set_index("subject_id")
    rows: list[dict[str, Any]] = []
    for episode in episodes.loc[episodes["scope"].eq("outer_test")].itertuples():
        subject = str(episode.subject_id)
        support_targets = np.asarray(episode.support_targets, dtype=np.int64)
        query_targets = np.asarray(episode.query_targets, dtype=np.int64)
        row: dict[str, Any] = {
            "subject_id": subject,
            "episode_id": str(episode.episode_id),
            "support_windows": len(episode.support_sample_ids),
            "query_windows": len(episode.query_sample_ids),
            "support_records": len(set(map(str, episode.support_record_ids))),
            "query_records": len(set(map(str, episode.query_record_ids))),
            "delta_macro_f1": float(fomaml.loc[subject, "macro_f1"] - supervised.loc[subject, "macro_f1"]),
            "delta_balanced_accuracy": float(fomaml.loc[subject, "balanced_accuracy"] - supervised.loc[subject, "balanced_accuracy"]),
        }
        for class_index in range(5):
            row[f"support_class_{class_index}"] = int((support_targets == class_index).sum())
            row[f"query_class_{class_index}"] = int((query_targets == class_index).sum())
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    summary = {
        "n_subjects": int(len(frame)),
        "analysis": "descriptive_noncausal",
        "support_windows_minimum": int(frame["support_windows"].min()),
        "support_windows_maximum": int(frame["support_windows"].max()),
        "pearson_support_size_vs_macro_f1_gain": float(
            frame["support_windows"].corr(frame["delta_macro_f1"], method="pearson")
        ),
        "spearman_support_size_vs_macro_f1_gain": float(
            frame["support_windows"].corr(frame["delta_macro_f1"], method="spearman")
        ),
        "causal_interpretation": False,
    }
    return frame, summary


class FOMAMLLabelQ5RawDiagnostic(FOMAMLLabelQ5Diagnostic):
    """Thin task-8C binding around the existing task-8X training orchestration."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        repository_root: Path,
        output_dir: Path | None = None,
    ) -> None:
        validate_raw_diagnostic_config(config)
        self.config = deepcopy(dict(config))
        self.root = repository_root
        self.output = output_dir or self.root / str(config["output_dir"])
        self.device = resolve_device(str(config["device"]))
        self.seed = int(config["seed"])
        protocol = config["protocol"]
        self.protocol_path = self.root / str(protocol["manifest"])
        self.episode_path = self.root / str(protocol["episode_index"])
        self.errors_path = self.root / str(protocol["errors"])
        self.raw_universe_path = self.root / str(protocol["raw_universe_manifest"])
        self.disabled_preregistration_path = self.root / str(
            protocol["disabled_preregistration"]
        )
        self.source_hashes = {
            "protocol_manifest": _sha256_file(self.protocol_path),
            "episode_index": _sha256_file(self.episode_path),
            "errors": _sha256_file(self.errors_path),
        }
        self.immutable_hashes_before = {
            **self.source_hashes,
            "raw_universe_manifest": _sha256_file(self.raw_universe_path),
            "disabled_preregistration": _sha256_file(
                self.disabled_preregistration_path
            ),
        }
        expected_files = {
            "protocol_manifest": protocol["protocol_file_sha256"],
            "episode_index": protocol["episode_file_sha256"],
            "errors": protocol["errors_file_sha256"],
            "raw_universe_manifest": protocol["raw_universe_file_sha256"],
            "disabled_preregistration": protocol["disabled_preregistration_sha256"],
        }
        if self.immutable_hashes_before != expected_files:
            raise RuntimeError(
                "Raw protocol or disabled preregistration file hash changed"
            )
        self.raw_episodes: pd.DataFrame | None = None
        self.outer_unlock_hash: str | None = None

    def _architecture_audit(self) -> dict[str, Any]:
        audit = super()._architecture_audit()
        self.output.mkdir(parents=True, exist_ok=True)
        _write_json(self.output / "architecture_audit.json", audit)
        return audit

    def _load_protocol(self) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
        protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        episodes = pd.read_parquet(self.episode_path)
        audit = validate_raw_episode_protocol(protocol, episodes, self.config)
        self.raw_episodes = episodes.copy()
        compatible = deepcopy(protocol)
        compatible.update({
            "meta_train_subjects": list(protocol["meta_split"]["meta_train_subjects"]),
            "meta_validation_subjects": list(protocol["meta_split"]["meta_validation_subjects"]),
            "outer_test_subjects": list(protocol["eligible_participants"]["outer_test"]),
            "episode_counts": dict(EXPECTED_SCOPE_COUNTS),
        })
        return compatible, episodes, audit

    def _preregister(
        self,
        protocol: Mapping[str, Any],
        episodes: pd.DataFrame,
        architecture: Mapping[str, Any],
    ) -> str:
        episode_ids = {
            scope: sorted(
                episodes.loc[episodes["scope"].eq(scope), "episode_id"].astype(str)
            )
            for scope in EXPECTED_SCOPE_COUNTS
        }
        payload = {
            "schema_version": RAW_DIAGNOSTIC_SCHEMA_VERSION,
            "experiment_id": self.config["experiment_id"],
            "repository_commit": _git_head(self.root),
            "result_status": "diagnostic",
            "execution_enabled": True,
            "protocol_id": protocol["protocol_id"],
            "protocol_hash": protocol["protocol_hash"],
            "raw_universe_hash": protocol["dataset_cache_signature"]["raw_universe_hash"],
            "outer_fold": 1,
            "outer_fold_artifact_sha256": self.config["protocol"]["outer_fold_artifact_sha256"],
            "outer_split_hash": self.config["protocol"]["outer_split_hash"],
            "meta_split_hash": self.config["protocol"]["meta_split_hash"],
            "episode_spec_hash": self.config["protocol"]["episode_spec_hash"],
            "episode_counts": dict(EXPECTED_SCOPE_COUNTS),
            "episode_ids": episode_ids,
            "model": "production_eegnet",
            "architecture_signature": architecture["row"]["architecture_signature"],
            "latent_dim": int(architecture["row"]["latent_dim"]),
            "parameter_count": int(architecture["row"]["parameter_count"]),
            "outputs": int(architecture["row"]["output_head_width"]),
            "seed": self.seed,
            "device": self.device,
            "support_query_contract": protocol["episode_spec"],
            "support_budget": protocol["support_budget"],
            "supervised": self.config["supervised"],
            "fomaml": self.config["fomaml"],
            "baseline_modes": list(self.config["baselines"]),
            "primary_metric": self.config["primary_metric"],
            "secondary_metrics": list(self.config["secondary_metrics"]),
            "policy_selection": self.config["policy_selection"],
            "decision_rule": self.config["decision_rule"],
            "source_hashes": self.immutable_hashes_before,
            "output_directory": self.config["output_dir"],
            "outer_test_locked": True,
        }
        if _contains_absolute_path(payload):
            raise RuntimeError("Execution preregistration contains an absolute path")
        root_path = self.output / "experiment_preregistration.json"
        nested_path = self.output / "preregistration/experiment_preregistration.json"
        digest = prepare_preregistration(root_path, payload)
        if prepare_preregistration(nested_path, payload) != digest:
            raise RuntimeError("Execution preregistration copies differ")
        _write_json(self.output / "preregistration/preregistration_hash.json", {
            "sha256": digest,
            "parameters_frozen_before_training": True,
            "outer_test_locked": True,
        })
        return digest

    def _load_data(self) -> tuple[Any, pd.DataFrame]:
        data, metadata = super()._load_data()
        if len(metadata) != int(self.config["dataset"]["expected_samples"]):
            raise RuntimeError("Raw-deduplicated sample count changed")
        if list(data.feature_names) != list(CANONICAL_EEG_CHANNELS):
            raise RuntimeError("Canonical EEG channel order changed")
        if np.dtype(data.data.dtype) != np.dtype(np.float32):
            raise RuntimeError("Raw mmap tensor dtype must be float32")
        if float(data.sampling_rate) != float(self.config["dataset"]["sampling_rate"]):
            raise RuntimeError("Raw EEG sampling rate changed")
        for position in (0, len(metadata) - 1):
            window = np.asarray(data.data[position])
            if window.shape != (1, 14, 2560) or not np.isfinite(window).all():
                raise RuntimeError("Raw mmap tensor shape or values are invalid")
        return data, metadata

    def _train_supervised(
        self,
        data: Any,
        metadata: pd.DataFrame,
        train_ids: Sequence[str],
        validation_ids: Sequence[str],
        outer_rows: Sequence[Any],
        preregistration_hash: str,
    ) -> tuple[Any, dict[str, Any]]:
        adapter, manifest = super()._train_supervised(
            data, metadata, train_ids, validation_ids, outer_rows,
            preregistration_hash,
        )
        directory = self.output / "supervised"
        shutil.copy2(directory / "supervised_training_history.csv", directory / "training_history.csv")
        shutil.copy2(directory / "supervised_checkpoint.pt", directory / "checkpoint.pt")
        exact_manifest = {
            **manifest,
            "optimizer_partition": "meta_train_only",
            "meta_validation_used_for_optimizer_steps": False,
            "outer_test_opened": False,
        }
        _write_json(directory / "checkpoint_manifest.json", exact_manifest)
        positions = metadata.index[
            metadata["sample_id"].isin(set(map(str, validation_ids)))
        ].to_numpy(dtype=np.int64)
        probabilities = adapter.predict_proba(data.data[positions])
        frame = pd.DataFrame({
            "dataset": "emotiv_raw_eeg",
            "task": "label_q5",
            "model": "torch_eegnet",
            "mode": "supervised",
            "seed": self.seed,
            "outer_fold": 1,
            "sample_id": metadata.iloc[positions]["sample_id"].to_numpy(),
            "subject_id": metadata.iloc[positions]["subject_id"].to_numpy(),
            "record_id": metadata.iloc[positions]["record_id"].to_numpy(),
            "y_true": np.asarray(data.labels)[positions],
            "y_pred": probabilities.argmax(axis=1),
        })
        for class_index in range(5):
            frame[f"proba_{class_index}"] = probabilities[:, class_index]
        frame.to_parquet(directory / "meta_validation_predictions.parquet", index=False)
        return adapter, exact_manifest

    def _train_fomaml_policy(
        self,
        policy: str,
        store: Any,
        train_rows: Sequence[Any],
        validation_rows: Sequence[Any],
        preregistration_hash: str,
        supervised_initial_hash: str,
        normalization: tuple[np.ndarray, np.ndarray],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest, runtime = super()._train_fomaml_policy(
            policy, store, train_rows, validation_rows, preregistration_hash,
            supervised_initial_hash, normalization,
        )
        directory = self.output / "fomaml" / policy
        shutil.copy2(directory / "fomaml_training_history.csv", directory / "training_history.csv")
        shutil.copy2(directory / "fomaml_checkpoint.pt", directory / "checkpoint.pt")
        model = self._fresh_fomaml(directory / "fomaml_checkpoint.pt")
        spec = self.config["fomaml"]
        learner = FirstOrderMAML(
            model,
            FOMAMLConfig(
                inner_steps=int(spec["inner_steps"]),
                inner_learning_rate=float(spec["inner_learning_rate"]),
                meta_learning_rate=float(spec["meta_learning_rate"]),
                episodes_per_meta_batch=1,
                maximum_meta_steps=1,
                gradient_clip_norm=float(spec["gradient_clip_norm"]),
                buffer_policy=policy,
                device=self.device,
                seed=self.seed,
            ),
        )
        validation_metrics, _, predictions = self._evaluate_adapted_episodes(
            learner,
            store,
            validation_rows,
            epoch=int(manifest["best_epoch"]),
            policy=policy,
        )
        predictions.insert(0, "mode", f"fomaml_{policy}")
        predictions.to_parquet(
            directory / "meta_validation_predictions.parquet", index=False
        )
        exact_manifest = {
            **manifest,
            "meta_validation_metrics": validation_metrics,
            "optimizer_partition": "meta_train_only",
            "meta_validation_used_for_optimizer_steps": False,
            "query_used_for_meta_gradient": True,
            "query_used_for_inner_adaptation": False,
            "query_used_for_meta_update": True,
            "outer_test_opened": False,
        }
        _write_json(directory / "checkpoint_manifest.json", exact_manifest)
        _write_json(directory / "fomaml_checkpoint_manifest.json", exact_manifest)
        return exact_manifest, runtime

    def _outer_evaluation(
        self,
        store: Any,
        outer_rows: Sequence[Any],
        supervised_checkpoint: Path,
        fomaml_checkpoint: Path,
        policy: str,
        decision_manifest_path: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
        if not decision_manifest_path.exists():
            raise RuntimeError("Pre-outer-test decision manifest is missing")
        selection_path = self.output / "policy_selection.json"
        preregistration_path = self.output / "experiment_preregistration.json"
        required = (
            selection_path,
            preregistration_path,
            supervised_checkpoint,
            fomaml_checkpoint,
        )
        if any(not path.exists() for path in required):
            raise RuntimeError("Outer-test cannot unlock before all selected objects exist")
        unlock = {
            "schema_version": RAW_DIAGNOSTIC_SCHEMA_VERSION,
            "selection_complete_before_outer_test": True,
            "outer_test_used_for_selection": False,
            "selected_policy": policy,
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "protocol_hash": self.config["protocol"]["expected_hash"],
            "raw_universe_hash": self.config["protocol"]["raw_universe_hash"],
            "episode_ids": sorted(str(row.episode_id) for row in outer_rows),
            "inner_steps": int(self.config["fomaml"]["inner_steps"]),
            "inner_learning_rate": float(self.config["fomaml"]["inner_learning_rate"]),
            "gradient_clip_norm": float(self.config["fomaml"]["gradient_clip_norm"]),
            "hashes": {
                "preregistration": _sha256_file(preregistration_path),
                "policy_selection": _sha256_file(selection_path),
                "pre_outer_test_decision": _sha256_file(decision_manifest_path),
                "supervised_checkpoint": _sha256_file(supervised_checkpoint),
                "fomaml_checkpoint": _sha256_file(fomaml_checkpoint),
                "protocol_manifest": _sha256_file(self.protocol_path),
                "episode_index": _sha256_file(self.episode_path),
            },
        }
        unlock_path = self.output / "outer_test_unlock_manifest.json"
        self.outer_unlock_hash = prepare_preregistration(unlock_path, unlock)
        _write_json(self.output / "outer_test_unlock_hash.json", {
            "sha256": self.outer_unlock_hash,
            "outer_test_opened_after_unlock": True,
        })
        return super()._outer_evaluation(
            store,
            outer_rows,
            supervised_checkpoint,
            fomaml_checkpoint,
            policy,
            decision_manifest_path,
        )

    def run(self) -> dict[str, Any]:
        summary = super().run()
        if self.raw_episodes is None or self.outer_unlock_hash is None:
            raise RuntimeError("Raw diagnostic finished without protocol or unlock state")
        immutable_after = {
            "protocol_manifest": _sha256_file(self.protocol_path),
            "episode_index": _sha256_file(self.episode_path),
            "errors": _sha256_file(self.errors_path),
            "raw_universe_manifest": _sha256_file(self.raw_universe_path),
            "disabled_preregistration": _sha256_file(
                self.disabled_preregistration_path
            ),
        }
        if immutable_after != self.immutable_hashes_before:
            raise RuntimeError("Task-8C protocol or disabled preregistration changed")

        _write_json(self.output / "protocol_reference.json", {
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "raw_universe_hash": EXPECTED_RAW_UNIVERSE_HASH,
            "outer_fold_artifact_sha256": self.config["protocol"]["outer_fold_artifact_sha256"],
            "source_hashes": immutable_after,
            "cache_rebuilt": False,
            "outer_split_rebuilt": False,
        })
        comparison_path = self.output / "paired_comparison.json"
        comparisons = json.loads(comparison_path.read_text(encoding="utf-8"))
        for offset, key in enumerate(("primary", "secondary")):
            comparisons[key]["paired_bootstrap"] = paired_subject_bootstrap(
                comparisons[key],
                resamples=int(self.config["bootstrap_resamples"]),
                seed=self.seed + offset,
            )
        _write_json(comparison_path, comparisons)

        subject_metrics = pd.read_csv(self.output / "outer_test_subject_metrics.csv")
        support_frame, support_summary = build_support_budget_analysis(
            self.raw_episodes, subject_metrics
        )
        support_frame.to_csv(self.output / "support_budget_analysis.csv", index=False)

        leakage_path = self.output / "leakage_audit.json"
        leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
        leakage.update({
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "raw_universe_hash": EXPECTED_RAW_UNIVERSE_HASH,
            "missing_raw_ids": 0,
            "meta_validation_used_for_optimizer_steps": False,
            "outer_test_opened_after_unlock": True,
            "outer_test_unlock_hash": self.outer_unlock_hash,
            "outer_test_used_for_checkpoint_selection": False,
            "outer_test_used_for_epoch_selection": False,
            "query_used_for_inner_adaptation": False,
            "query_used_for_meta_gradient": True,
            "query_updated_batchnorm_buffers": False,
            "disabled_preregistration_unchanged": True,
            "raw_protocol_unchanged": True,
        })
        _write_json(leakage_path, leakage)

        buffer_path = self.output / "buffer_audit.json"
        buffer_audit = json.loads(buffer_path.read_text(encoding="utf-8"))
        buffer_audit.update({
            "policy_selected_on": "meta_validation_only",
            "query_used_to_update_buffers": False,
            "episode_states_isolated": True,
        })
        _write_json(buffer_path, buffer_audit)

        summary.update({
            "schema_version": RAW_DIAGNOSTIC_SCHEMA_VERSION,
            "experiment_id": self.config["experiment_id"],
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "raw_universe_hash": EXPECTED_RAW_UNIVERSE_HASH,
            "outer_test_unlock_hash": self.outer_unlock_hash,
            "paired_comparison": comparisons,
            "support_budget_analysis": support_summary,
            "immutable_source_hashes": immutable_after,
            "cache_rebuilt": False,
            "outer_split_rebuilt": False,
        })
        _write_json(self.output / "diagnostic_summary.json", summary)
        decision = json.loads((self.output / "decision.json").read_text(encoding="utf-8"))
        decision.update({
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "outer_test_unlock_hash": self.outer_unlock_hash,
            "comparison_unit": "subject",
            "n_subjects": 5,
        })
        _write_json(self.output / "decision.json", decision)
        primary = comparisons["primary"]
        (self.output / "diagnostic_report.md").write_text(
            "# Raw-deduplicated FOMAML label_q5 diagnostic\n\n"
            f"- Status: `{decision['status']}` (one-fold diagnostic).\n"
            f"- Protocol: `{EXPECTED_PROTOCOL_ID}` / `{EXPECTED_PROTOCOL_HASH}`.\n"
            f"- Device: `{self.device}`; seed: 42; outer fold: 1.\n"
            f"- Selected BatchNorm policy: `{summary['policy_selection']['selected_policy']}`.\n"
            f"- Outer-test unlock SHA-256: `{self.outer_unlock_hash}`.\n"
            f"- Mean subject macro-F1 delta versus supervised full-model: "
            f"{primary['mean_delta_macro_f1']:.6f}.\n"
            f"- Mean subject balanced-accuracy delta: "
            f"{primary['mean_delta_balanced_accuracy']:.6f}.\n"
            f"- Wins/losses/ties: {primary['macro_f1_wins']}/"
            f"{primary['macro_f1_losses']}/{primary['macro_f1_ties']}.\n"
            "- Five participants, one fold and one seed: bootstrap intervals are "
            "diagnostic and do not establish statistical significance.\n",
            encoding="utf-8",
        )
        return summary


def run_fomaml_label_q5_raw_diagnostic(
    config: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    return FOMAMLLabelQ5RawDiagnostic(
        config, repository_root=repository_root
    ).run()
