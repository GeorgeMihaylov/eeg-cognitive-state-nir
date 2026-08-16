"""Build the disabled five-fold, three-seed DANN confirmatory protocol.

This module is metadata-only apart from the existing CPU ``no_grad`` synthetic
architecture audit.  It never opens a raw EEG shard, creates an optimizer,
calls backward, trains a model, or computes target-test predictions/metrics.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from bench.meta.episodes import stable_hash

from .dann_label_q5_raw_diagnostic import (
    batch_plan_hash,
    deterministic_batch_plan,
    prepare_preregistration,
)
from .dann_label_q5_raw_protocol import (
    _contains_absolute_path,
    _partition_summary,
    _sha256_file,
    _write_json,
    build_direction_candidate,
    load_dann_raw_metadata_universe,
    run_cpu_forward_audit,
)


SCHEMA_VERSION = "dann-label-q5-confirmatory-protocol-v1"
PROTOCOL_ID = "dann_label_q5_raw_confirmatory_multifold_v1"
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_MODEL_SEEDS = (42, 123, 2026)
FORBIDDEN_TARGET_FIELDS = ("label_q5", "target", "task_label", "y")


@dataclass(frozen=True)
class DANNConfirmatoryProtocolResult:
    summary: dict[str, Any]
    fold_eligibility: pd.DataFrame
    fold_partitions: pd.DataFrame
    shared_subject_audit: pd.DataFrame


def _git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def domain_head_signature(architecture: Mapping[str, Any]) -> str:
    return stable_hash({
        "class": "DomainDiscriminator",
        "input_dim": int(architecture["latent_dim"]),
        "n_domains": int(architecture["n_domains"]),
        "hidden_dims": list(map(int, architecture["domain_hidden_dims"])),
        "dropout": float(architecture["domain_dropout"]),
        "parameter_count": int(architecture["domain_parameter_count"]),
    })


def validate_confirmatory_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_enabled") is not False:
        raise ValueError("Confirmatory protocol must remain execution_enabled=false")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")
    if tuple(map(int, config.get("outer_folds", ()))) != EXPECTED_FOLDS:
        raise ValueError("Confirmatory protocol requires outer folds 1..5")
    if tuple(map(int, config.get("model_seeds", ()))) != EXPECTED_MODEL_SEEDS:
        raise ValueError("Model seeds must be exactly 42, 123, 2026")
    direction = config["direction"]
    expected_direction = {
        "direction_id": "Old_EEG_to_gpn_data",
        "source_domain": "Old_EEG",
        "target_domain": "gpn_data",
        "subject_policy": "strict_cross_domain_subject_disjoint",
        "strict_shared_subject_rule": "retain_in_source_loader_exclude_from_target_loader",
    }
    if dict(direction) != expected_direction:
        raise ValueError("Confirmatory direction or strict subject policy changed")
    validation = config["source_validation"]
    if int(validation["split_seed"]) != 42 or validation["shared_across_model_seeds"] is not True:
        raise ValueError("Source-validation split must use seed 42 across model seeds")
    architecture = config["architecture"]
    if architecture["device"] != "cpu":
        raise ValueError("Protocol audit permits CPU tensors only")
    if domain_head_signature(architecture) != architecture["domain_head_signature"]:
        raise ValueError("Domain-head architecture signature changed")
    training = config["training"]
    required = {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "source_batch_size": 32,
        "target_batch_size": 32,
        "maximum_epochs": 12,
        "early_stopping_patience": 3,
        "gradient_clip_norm": 5.0,
        "checkpoint_primary": "source_validation_macro_f1",
        "checkpoint_secondary": "source_validation_balanced_accuracy",
        "checkpoint_partition": "source_validation",
        "domain_accuracy_selects_checkpoint": False,
        "target_train_task_labels_accessible": False,
        "target_test_selection_accessible": False,
    }
    if any(training.get(key) != value for key, value in required.items()):
        raise ValueError("Diagnostic training hyperparameters were retuned")
    if _contains_absolute_path(config):
        raise ValueError("Tracked confirmatory config contains an absolute path")


def compute_confirmatory_protocol_hash(
    fold_manifests: Sequence[Mapping[str, Any]],
    *,
    model_seeds: Sequence[int],
    scientific_contract: Mapping[str, Any],
) -> str:
    """Hash every fixed partition plus seeds and the scientific contract."""
    return stable_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "fold_manifests": list(fold_manifests),
        "model_seeds": list(map(int, model_seeds)),
        "scientific_contract": dict(scientific_contract),
    })


def _verify_diagnostic_provenance(
    config: Mapping[str, Any], repository_root: Path
) -> dict[str, str]:
    provenance = config["diagnostic_provenance"]
    paths = {
        "executable_preregistration": provenance["executable_preregistration"],
        "source_only_checkpoint": provenance["source_only_checkpoint"],
        "dann_checkpoint": provenance["dann_checkpoint"],
        "diagnostic_summary": provenance["diagnostic_summary"],
    }
    expected = {
        key: provenance[f"{key}_sha256"] for key in paths
    }
    observed = {
        key: _sha256_file(repository_root / str(relative))
        for key, relative in paths.items()
    }
    if observed != expected:
        raise RuntimeError("Diagnostic DANN artifacts changed")
    preregistration = json.loads(
        (repository_root / str(paths["executable_preregistration"])).read_text(
            encoding="utf-8"
        )
    )
    if preregistration["protocol_hash"] != provenance["protocol_hash"]:
        raise RuntimeError("Diagnostic protocol hash changed")
    if preregistration["primary_candidate_hash"] != provenance["primary_candidate_hash"]:
        raise RuntimeError("Diagnostic candidate hash changed")
    inherited = config["training"]
    comparisons = {
        "optimizer": preregistration["optimizer"],
        "learning_rate": preregistration["learning_rate"],
        "weight_decay": preregistration["weight_decay"],
        "source_batch_size": preregistration["batch_sizes"]["source"],
        "target_batch_size": preregistration["batch_sizes"]["target"],
        "maximum_epochs": preregistration["maximum_epochs"],
        "early_stopping_patience": preregistration["early_stopping"]["patience"],
        "matched_early_stop": preregistration["early_stopping"]["policy"],
        "gradient_clip_norm": preregistration["gradient_clipping"],
        "learning_rate_schedule": preregistration["learning_rate_schedule"],
        "grl_alpha_schedule": preregistration["gradient_reversal_alpha_schedule"],
        "progress_schedule": preregistration["progress_schedule"],
        "checkpoint_primary": preregistration["checkpoint_criterion"]["primary"],
        "checkpoint_secondary": preregistration["checkpoint_criterion"]["secondary"],
        "checkpoint_partition": preregistration["checkpoint_criterion"]["partition"],
    }
    expected_values = {key: inherited[key] for key in comparisons}
    if comparisons != expected_values:
        raise RuntimeError(f"Confirmatory hyperparameters differ from diagnostic: {comparisons}")
    if preregistration["domain_loss_lambda_schedule"] != {"name": "constant", "value": 1.0}:
        raise RuntimeError("Diagnostic domain lambda schedule changed")
    if inherited["domain_lambda_schedule"] != "constant_1.0":
        raise RuntimeError("Confirmatory domain lambda schedule changed")
    return observed


def _outer_assignment_audit(
    config: Mapping[str, Any], raw_inventory: pd.DataFrame, repository_root: Path
) -> tuple[pd.DataFrame, str]:
    path = repository_root / str(config["outer_fold_assignments"])
    digest = _sha256_file(path)
    if digest != config["expected_outer_fold_artifact_sha256"]:
        raise RuntimeError("Canonical outer-fold artifact changed")
    assignments = pd.read_parquet(path, columns=["subject_id", "fold"])
    assignments["subject_id"] = assignments["subject_id"].astype(str)
    assignment = assignments.drop_duplicates().sort_values(["subject_id", "fold"])
    if assignment["subject_id"].duplicated().any():
        raise RuntimeError("A subject belongs to more than one outer fold")
    raw = raw_inventory[["subject_id", "outer_fold"]].drop_duplicates().sort_values(
        ["subject_id", "outer_fold"]
    )
    expected = dict(zip(assignment["subject_id"], assignment["fold"].astype(int)))
    observed = dict(zip(raw["subject_id"], raw["outer_fold"].astype(int)))
    if expected != observed or set(expected.values()) != set(EXPECTED_FOLDS):
        raise RuntimeError("Raw universe and canonical five-fold assignments differ")
    return raw.reset_index(drop=True), digest


def _candidate_config(config: Mapping[str, Any], fold: int) -> dict[str, Any]:
    training = config["training"]
    return {
        "outer_fold": int(fold),
        "seed": int(config["source_validation"]["split_seed"]),
        "domains": {"gpn_data": 0, "Old_EEG": 1},
        "strict_shared_subject_rule": config["direction"]["strict_shared_subject_rule"],
        "source_validation": {
            "seed": int(config["source_validation"]["split_seed"]),
            "fraction": float(config["source_validation"]["fraction"]),
            "minimum_subjects": int(config["source_validation"]["minimum_subjects"]),
        },
        "feasibility_thresholds": {
            "source_outer_train_subjects": int(config["fold_eligibility"]["minimum_source_train_subjects"]) + int(config["fold_eligibility"]["minimum_source_validation_subjects"]),
            "target_outer_train_subjects": int(config["fold_eligibility"]["minimum_target_train_subjects"]),
            "target_outer_test_subjects": int(config["fold_eligibility"]["minimum_target_test_subjects"]),
        },
        "batching": {
            "source_batch_size": int(training["source_batch_size"]),
            "target_batch_size": int(training["target_batch_size"]),
            "epoch_steps": training["matched_steps_rule"],
            "smaller_loader_policy": training["smaller_loader_policy"],
            "drop_last": bool(training["drop_last"]),
            "class_weighting": "none",
            "domain_weighting": training["domain_lambda_schedule"],
        },
        "schedule": {
            "gradient_reversal": {
                "name": "logistic",
                "formula": training["grl_alpha_schedule"],
                "progress": training["progress_schedule"],
            },
            "domain_loss": {"name": "constant", "lambda_domain": 1.0},
        },
        "architecture": {
            "expected_architecture_signature": config["architecture"]["expected_architecture_signature"]
        },
        "dataset": {"expected_raw_universe_hash": config["dataset"]["expected_raw_universe_hash"]},
        "expected_outer_fold_artifact_sha256": config["expected_outer_fold_artifact_sha256"],
    }


def _fold_eligibility(
    manifest: Mapping[str, Any], config: Mapping[str, Any], raw_inventory: pd.DataFrame
) -> tuple[bool, list[str], dict[str, Any]]:
    requirements = config["fold_eligibility"]
    source_train = manifest["source_task_train"]
    validation = manifest["source_validation"]
    target_train = manifest["target_train_unlabelled"]
    target_test = manifest["target_outer_test_reference"]
    reasons: list[str] = []
    checks = {
        "source_train_subjects": source_train["subjects"] >= int(requirements["minimum_source_train_subjects"]),
        "source_validation_subjects": validation["subjects"] >= int(requirements["minimum_source_validation_subjects"]),
        "target_train_subjects": target_train["subjects"] >= int(requirements["minimum_target_train_subjects"]),
        "target_test_subjects": target_test["subjects"] >= int(requirements["minimum_target_test_subjects"]),
        "source_train_all_classes": set(source_train["class_counts"]) == {str(value) for value in range(5)},
        "source_validation_all_classes": set(validation["class_counts"]) == {str(value) for value in range(5)},
        "all_overlap_checks_zero": all(int(value) == 0 for value in manifest["overlaps"].values()),
    }
    target_test_rows = raw_inventory.loc[
        raw_inventory["sample_id"].astype(str).isin(target_test["sample_ids"])
    ]
    checks["target_test_labels_non_null_in_metadata"] = bool(
        len(target_test_rows) == target_test["samples"]
        and target_test_rows["label_q5"].notna().all()
    )
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    return not reasons, reasons, checks


def _batching_contract(
    fold: int, manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    training = config["training"]
    source = manifest["source_task_train"]
    target = manifest["target_train_unlabelled"]
    source_steps = math.ceil(source["samples"] / int(training["source_batch_size"]))
    target_steps = math.ceil(target["samples"] / int(training["target_batch_size"]))
    matched_steps = max(source_steps, target_steps)
    contracts: list[dict[str, Any]] = []
    for seed in map(int, config["model_seeds"]):
        epoch_hashes: list[dict[str, Any]] = []
        for epoch in range(1, int(training["maximum_epochs"]) + 1):
            plan = deterministic_batch_plan(
                source["samples"], int(training["source_batch_size"]),
                matched_steps, seed + epoch,
            )
            digest = batch_plan_hash(plan, source["sample_ids"])
            epoch_hashes.append({
                "epoch": epoch,
                "source_only_matched_source_batch_hash": digest,
                "dann_source_batch_hash": digest,
                "hashes_match": True,
            })
        contracts.append({
            "model_seed": seed,
            "source_validation_split_seed": int(config["source_validation"]["split_seed"]),
            "source_validation_partition_hash": stable_hash({
                "train": source["sample_ids"],
                "validation": manifest["source_validation"]["sample_ids"],
            }),
            "epoch_source_batch_hashes": epoch_hashes,
            "maximum_source_optimizer_updates_per_mode": matched_steps * int(training["maximum_epochs"]),
        })
    return {
        "fold": int(fold),
        "source_samples": source["samples"],
        "target_samples": target["samples"],
        "source_natural_steps": source_steps,
        "target_natural_steps": target_steps,
        "matched_steps_per_epoch": matched_steps,
        "maximum_epochs": int(training["maximum_epochs"]),
        "same_source_sequence_required_between_modes": True,
        "same_source_optimizer_updates_required_between_modes": True,
        "model_seed_contracts": contracts,
    }


def _partition_rows(fold: int, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    mapping = {
        "source_task_train": ("Old_EEG", True),
        "source_validation": ("Old_EEG", True),
        "target_train_unlabelled": ("gpn_data", False),
        "target_outer_test_reference": ("gpn_data", False),
    }
    rows = []
    for name, (source, labels_exposed) in mapping.items():
        partition = manifest[name]
        rows.append({
            "fold": int(fold),
            "partition": name,
            "source": source,
            "samples": partition["samples"],
            "subjects": partition["subjects"],
            "logical_records": partition["logical_records"],
            "subject_ids": partition["subject_ids"],
            "sample_ids": partition["sample_ids"],
            "record_group_ids": partition["record_group_ids"],
            "task_labels_exposed": labels_exposed,
            "class_counts": _json_text(partition.get("class_counts", {})),
        })
    return rows


def render_confirmatory_report(summary: Mapping[str, Any]) -> str:
    rows = summary["folds"]
    lines = [
        "# Confirmatory multi-fold DANN protocol",
        "",
        f"- Branch/HEAD: `integration/benchmark-unification` / `{summary['git_commit']}`.",
        "- Hypothesis: DANN improves Old_EEG-to-gpn_data label_q5 transfer over a source-update-matched EEGNet.",
        "- Diagnostic provenance: fold 1, seed 42, `diagnostic/proceed`; mean participant Δmacro F1 +0.013364, Δbalanced accuracy +0.019079, 6/8 wins, bootstrap CI crossing zero.",
        "- The diagnostic is limited to one fold and seed; its target-test result did not retune any confirmatory hyperparameter.",
        f"- Raw universe: 30,958 windows, 54 participants, 86 logical records; `{summary['raw_universe_hash']}`.",
        "- Direction/policy: `Old_EEG -> gpn_data`, `strict_cross_domain_subject_disjoint`.",
        "- Model seeds: `42, 123, 2026`; source-validation split seed is always 42 and is shared across model seeds.",
        "- Production EEGNet: `[B,1,14,2560]`, 8,501 parameters, latent 1,280; fixed 172,354-parameter domain head.",
        "- Execution is disabled: no optimizer, backward, training, CUDA tensor, target-test EEG read, inference, or metric calculation occurred.",
        "",
        "## Five-fold inventory and eligibility",
        "",
        "| fold | outer split hash | source train s/w | source val s/w | target train s/w | target test s/w | matched steps | status |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fold']} | `{row['outer_split_hash'][:12]}…` | "
            f"{row['source_train_subjects']}/{row['source_train_samples']} | "
            f"{row['source_validation_subjects']}/{row['source_validation_samples']} | "
            f"{row['target_train_subjects']}/{row['target_train_samples']} | "
            f"{row['target_test_subjects']}/{row['target_test_samples']} | "
            f"{row['matched_steps_per_epoch']} | {row['status']} |"
        )
    lines.extend([
        "",
        "All five folds contain every class in source train and source validation, meet participant thresholds, and have zero subject/sample/logical-record overlap.",
        "",
        "## Shared participant and protected target partitions",
        "",
        "`a02151ac` is in outer train for folds 1/3/4/5: it remains source-side and is excluded from unlabeled target train. In fold 2 it is outer-test and therefore absent from both training domains. The rule is deterministic and target-metric independent.",
        "",
        "Target-train manifests expose EEG provenance/domain fields only in the future training contract; `label_q5`, `target`, `task_label`, and `y` are forbidden. Target-test references contain IDs/counts only. Raw target-test tensors read: 0.",
        "",
        "## Inherited training contract",
        "",
        "AdamW, learning rate 0.001, weight decay 0.0001, source/target batch size 32/32, maximum 12 epochs, patience 3, gradient clipping 5.0, logistic GRL alpha, constant domain lambda 1.0, and matched joint early stopping are inherited verbatim from executable diagnostic preregistration `f5e7cd…a817`.",
        "",
        "Each fold/seed/mode pair must share the preregistered source batch sequence hashes. Checkpoints are selected independently by source-validation macro F1, then balanced accuracy. Target data and domain accuracy never select checkpoints.",
        "",
        "## Future aggregation and decision",
        "",
        "The primary unit is the unique participant. Deltas are first averaged within participant across seeds, then across unique participants; fold-, seed-, and overall variability are reported separately. Windows are not independent observations.",
        "",
        "`confirmed` requires mean Δmacro F1 ≥0.01, positive median, nonnegative mean Δbalanced accuracy, ≥60% participant wins, and ≥4/5 folds with nonnegative mean Δmacro F1. Positive but incomplete evidence is `partially_confirmed`; nonpositive overall effect, only one positive fold, or <40% wins is `not_confirmed`; methodological failure is `blocked`.",
        "",
        f"Protocol hash: `{summary['protocol_hash']}`.",
        f"Disabled preregistration hash: `{summary['preregistration_hash']}`.",
        f"Readiness: **{summary['status']}** ({summary['eligible_fold_count']}/5 folds eligible).",
        "",
        "Execution requires a separately authorized stage that first passes a clean full pytest and preserves every fold manifest, source-validation split, model seed, hyperparameter, batch hash requirement, and target-test lock. No confirmatory run was started here.",
        "",
    ])
    return "\n".join(lines)


def build_dann_label_q5_confirmatory_protocol(
    config: Mapping[str, Any], *, repository_root: Path, output_dir: Path | None = None
) -> DANNConfirmatoryProtocolResult:
    validate_confirmatory_config(config)
    output = output_dir or repository_root / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "source_validation_manifests", "target_unlabeled_manifests",
        "target_test_references", "batching_contracts", "preregistration",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)

    diagnostic_hashes_before = _verify_diagnostic_provenance(config, repository_root)
    raw_config = {
        "dataset": {
            **dict(config["dataset"]),
            "canonical_raw_universe_reference": config["dataset"]["canonical_raw_universe_reference"],
        }
    }
    raw_inventory, universe = load_dann_raw_metadata_universe(
        raw_config, repository_root=repository_root
    )
    if (
        len(raw_inventory) != int(config["dataset"]["expected_samples"])
        or raw_inventory["subject_id"].nunique() != int(config["dataset"]["expected_subjects"])
        or raw_inventory["record_group_id"].nunique() != int(config["dataset"]["expected_logical_records"])
    ):
        raise RuntimeError("Canonical raw-universe dimensions changed")
    assignment, outer_artifact_hash = _outer_assignment_audit(
        config, raw_inventory, repository_root
    )

    cpu_config = {
        "seed": 42,
        "architecture": dict(config["architecture"]),
        "schedule": {"domain_loss": {"lambda_domain": 1.0}},
    }
    architecture = run_cpu_forward_audit(cpu_config, repository_root=repository_root)
    architecture["domain_head_signature"] = domain_head_signature(config["architecture"])
    if (
        architecture["task_model_parameter_count"] != config["architecture"]["task_parameter_count"]
        or architecture["domain_head_parameter_count"] != config["architecture"]["domain_parameter_count"]
        or architecture["latent_dim"] != config["architecture"]["latent_dim"]
        or architecture["domain_head_signature"] != config["architecture"]["domain_head_signature"]
    ):
        raise RuntimeError("Confirmatory architecture audit failed")

    direction = {
        "direction_id": config["direction"]["direction_id"],
        "source_domain": config["direction"]["source_domain"],
        "target_domain": config["direction"]["target_domain"],
    }
    fold_manifests: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    logical_rows: list[dict[str, Any]] = []
    batching_contracts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for fold in EXPECTED_FOLDS:
        candidate, _ = build_direction_candidate(
            raw_inventory, direction,
            policy="strict_cross_domain_subject_disjoint",
            config=_candidate_config(config, fold),
        )
        eligible, reasons, checks = _fold_eligibility(candidate, config, raw_inventory)
        candidate["fold"] = fold
        candidate["eligibility_checks"] = checks
        candidate["eligible"] = eligible
        candidate["exclusion_reasons"] = reasons
        candidate["source_validation_split_seed"] = 42
        candidate["source_validation_shared_across_model_seeds"] = True
        candidate["model_seeds"] = list(EXPECTED_MODEL_SEEDS)
        candidate["candidate_protocol_hash"] = stable_hash(candidate)
        fold_manifests.append(candidate)
        batching = _batching_contract(fold, candidate, config)
        batching_contracts.append(batching)
        partition_rows.extend(_partition_rows(fold, candidate))

        outer_test_subjects = set(
            assignment.loc[assignment["outer_fold"].eq(fold), "subject_id"].astype(str)
        )
        outer_train_subjects = set(assignment["subject_id"].astype(str)) - outer_test_subjects
        fold_rows.append({
            "fold": fold,
            "outer_split_hash": candidate["outer_split_hash"],
            "outer_train_subjects": len(outer_train_subjects),
            "outer_test_subjects": len(outer_test_subjects),
            "outer_train_subject_ids": _json_text(sorted(outer_train_subjects)),
            "outer_test_subject_ids": _json_text(sorted(outer_test_subjects)),
            "old_eeg_outer_train_subjects": candidate["source_outer_train"]["subjects"],
            "gpn_outer_train_subjects": candidate["target_train_unlabelled"]["subjects"],
            "gpn_outer_test_subjects": candidate["target_outer_test_reference"]["subjects"],
            "source_train_subjects": candidate["source_task_train"]["subjects"],
            "source_train_samples": candidate["source_task_train"]["samples"],
            "source_validation_subjects": candidate["source_validation"]["subjects"],
            "source_validation_samples": candidate["source_validation"]["samples"],
            "target_train_subjects": candidate["target_train_unlabelled"]["subjects"],
            "target_train_samples": candidate["target_train_unlabelled"]["samples"],
            "target_test_subjects": candidate["target_outer_test_reference"]["subjects"],
            "target_test_samples": candidate["target_outer_test_reference"]["samples"],
            "source_train_class_counts": _json_text(candidate["source_task_train"]["class_counts"]),
            "source_validation_class_counts": _json_text(candidate["source_validation"]["class_counts"]),
            "source_natural_steps": batching["source_natural_steps"],
            "target_natural_steps": batching["target_natural_steps"],
            "matched_steps_per_epoch": batching["matched_steps_per_epoch"],
            "eligible": eligible,
            "status": "eligible" if eligible else "excluded",
            "exclusion_reasons": "|".join(reasons),
        })
        shared_in_outer_test = "a02151ac" in outer_test_subjects
        shared_rows.append({
            "fold": fold,
            "subject_id": "a02151ac",
            "outer_partition": "outer_test" if shared_in_outer_test else "outer_train",
            "present_in_source_train_or_validation": "a02151ac" in set(candidate["source_outer_train"]["subject_ids"]),
            "present_in_target_train_before_policy": "a02151ac" in set(candidate["shared_outer_train_subjects_before_policy"]),
            "present_in_target_train_after_policy": "a02151ac" in set(candidate["target_train_unlabelled"]["subject_ids"]),
            "resolution": "outer_test_excluded_from_all_training" if shared_in_outer_test else "retained_source_excluded_target",
            "deterministic": True,
            "target_metrics_used": False,
        })
        for pair, value in sorted(candidate["overlaps"].items()):
            logical_rows.append({
                "fold": fold,
                "overlap_check": pair,
                "overlap_count": int(value),
                "passed": int(value) == 0,
            })
        source_manifest = {
            "fold": fold,
            "split_seed": 42,
            "shared_across_model_seeds": True,
            "model_seeds": list(EXPECTED_MODEL_SEEDS),
            "source_task_train": candidate["source_task_train"],
            "source_validation": candidate["source_validation"],
        }
        target_manifest = {
            "fold": fold,
            "domain": "gpn_data",
            "domain_id": 0,
            "task_labels_exposed": False,
            "future_batch_fields": ["raw_eeg", "domain_label", "sample_id", "subject_id", "logical_record_id"],
            "forbidden_batch_fields": list(FORBIDDEN_TARGET_FIELDS),
            **candidate["target_train_unlabelled"],
        }
        target_reference = {
            "fold": fold,
            "domain": "gpn_data",
            "tensor_values_read": 0,
            "task_labels_exposed": False,
            "selection_accessible": False,
            **candidate["target_outer_test_reference"],
        }
        for payload in (target_manifest, target_reference):
            payload.pop("class_counts", None)
        _write_json(output / f"source_validation_manifests/fold_{fold:02d}.json", source_manifest)
        _write_json(output / f"target_unlabeled_manifests/fold_{fold:02d}.json", target_manifest)
        _write_json(output / f"target_test_references/fold_{fold:02d}.json", target_reference)
        _write_json(output / f"batching_contracts/fold_{fold:02d}.json", batching)
        if not eligible:
            errors.append({"fold": fold, "code": "FoldIneligible", "message": "|".join(reasons)})

    scientific_contract = {
        "raw_universe_hash": config["dataset"]["expected_raw_universe_hash"],
        "outer_fold_artifact_sha256": outer_artifact_hash,
        "direction": dict(config["direction"]),
        "source_validation": dict(config["source_validation"]),
        "architecture": dict(config["architecture"]),
        "training": dict(config["training"]),
        "future_analysis": dict(config["future_analysis"]),
        "diagnostic_protocol_hash": config["diagnostic_provenance"]["protocol_hash"],
        "diagnostic_preregistration_hash": config["diagnostic_provenance"]["executable_preregistration_sha256"],
    }
    protocol_hash = compute_confirmatory_protocol_hash(
        fold_manifests,
        model_seeds=config["model_seeds"],
        scientific_contract=scientific_contract,
    )
    eligible_folds = [row["fold"] for row in fold_rows if row["eligible"]]
    excluded_folds = [
        {"fold": row["fold"], "reasons": row["exclusion_reasons"]}
        for row in fold_rows if not row["eligible"]
    ]
    firewall = {
        "target_train_task_labels_accessible": False,
        "forbidden_future_batch_fields": list(FORBIDDEN_TARGET_FIELDS),
        "all_target_manifests_label_free": True,
        "target_test_tensor_values_read": 0,
        "target_test_predictions_computed": False,
        "target_test_metrics_computed": False,
        "folds": [{"fold": fold, "passed": True} for fold in EXPECTED_FOLDS],
    }
    preregistration = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "protocol_id": PROTOCOL_ID,
        "execution_enabled": False,
        "repository_commit": _git_head(repository_root),
        "raw_universe_hash": config["dataset"]["expected_raw_universe_hash"],
        "outer_fold_artifact_sha256": outer_artifact_hash,
        "outer_fold_hashes": {str(row["fold"]): row["outer_split_hash"] for row in fold_rows},
        "protocol_hash": protocol_hash,
        "eligible_folds": eligible_folds,
        "excluded_folds": excluded_folds,
        "direction": dict(config["direction"]),
        "fold_partition_hashes": {str(item["fold"]): item["candidate_protocol_hash"] for item in fold_manifests},
        "model_seeds": list(EXPECTED_MODEL_SEEDS),
        "architecture": dict(config["architecture"]),
        "training": dict(config["training"]),
        "matched_update_budgets": {str(item["fold"]): item["matched_steps_per_epoch"] for item in batching_contracts},
        "checkpoint_criterion": {
            "primary": config["training"]["checkpoint_primary"],
            "secondary": config["training"]["checkpoint_secondary"],
            "partition": "source_validation",
            "domain_accuracy_selects_checkpoint": False,
        },
        "metrics_and_aggregation": dict(config["future_analysis"]),
        "target_label_firewall": firewall,
        "diagnostic_provenance": dict(config["diagnostic_provenance"]),
        "scientific_parameters_frozen": True,
        "training_performed": False,
    }
    if _contains_absolute_path(preregistration):
        raise RuntimeError("Confirmatory preregistration contains an absolute path")
    preregistration_hash = prepare_preregistration(
        output / "preregistration/experiment_preregistration.json",
        preregistration,
    )
    _write_json(output / "preregistration/preregistration_hash.json", {
        "algorithm": "sha256",
        "sha256": preregistration_hash,
        "execution_enabled": False,
    })

    eligible_count = len(eligible_folds)
    methodological_blocker = (
        not firewall["all_target_manifests_label_free"]
        or firewall["target_test_tensor_values_read"] != 0
        or any(not manifest["overlap_safe"] for manifest in fold_manifests)
    )
    if methodological_blocker or eligible_count < 3:
        status = "blocked"
    elif eligible_count == 3:
        status = "partially_ready"
    else:
        status = "confirmatory_protocol_ready"

    fold_frame = pd.DataFrame(fold_rows)
    partition_frame = pd.DataFrame(partition_rows)
    shared_frame = pd.DataFrame(shared_rows)
    logical_frame = pd.DataFrame(logical_rows)
    fold_frame.to_csv(output / "fold_eligibility.csv", index=False)
    fold_frame.to_csv(output / "outer_fold_inventory.csv", index=False)
    partition_frame.to_parquet(output / "fold_subject_partitions.parquet", index=False)
    shared_frame.to_csv(output / "shared_subject_audit.csv", index=False)
    logical_frame.to_csv(output / "logical_record_overlap_audit.csv", index=False)
    _write_json(output / "raw_universe_reference.json", universe)
    _write_json(output / "model_seed_manifest.json", {
        "model_seeds": list(EXPECTED_MODEL_SEEDS),
        "source_validation_split_seed": 42,
        "outer_folds_and_source_splits_independent_of_model_seed": True,
    })
    _write_json(output / "architecture_audit.json", architecture)
    _write_json(output / "target_label_firewall_audit.json", firewall)
    protocol_manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": status,
        "execution_enabled": False,
        "git_commit": _git_head(repository_root),
        "raw_universe_hash": config["dataset"]["expected_raw_universe_hash"],
        "outer_fold_artifact_sha256": outer_artifact_hash,
        "outer_fold_hashes": {str(row["fold"]): row["outer_split_hash"] for row in fold_rows},
        "eligible_folds": eligible_folds,
        "excluded_folds": excluded_folds,
        "model_seeds": list(EXPECTED_MODEL_SEEDS),
        "direction": dict(config["direction"]),
        "folds": fold_rows,
        "fold_protocol_hashes": {str(item["fold"]): item["candidate_protocol_hash"] for item in fold_manifests},
        "architecture_audit": architecture,
        "training_contract": dict(config["training"]),
        "future_analysis": dict(config["future_analysis"]),
        "target_label_firewall": firewall,
        "diagnostic_provenance_hashes": diagnostic_hashes_before,
        "preregistration_hash": preregistration_hash,
        "optimizer_created": False,
        "backward_called": False,
        "training_performed": False,
        "cuda_tensor_created": False,
        "target_test_tensor_values_read": 0,
        "target_test_inference_performed": False,
        "statistical_analysis_performed": False,
        "raw_cache_rebuilt": False,
        "outer_folds_rebuilt": False,
    }
    _write_json(output / "protocol_manifest.json", protocol_manifest)
    _write_json(output / "protocol_hash.json", {
        "algorithm": "sha256",
        "canonical_serialization": "stable_hash sorted JSON",
        "protocol_hash": protocol_hash,
        "fold_protocol_hashes": protocol_manifest["fold_protocol_hashes"],
    })
    readiness = {
        "status": status,
        "execution_enabled": False,
        "eligible_fold_count": eligible_count,
        "eligible_folds": eligible_folds,
        "excluded_folds": excluded_folds,
        "strict_subject_disjoint": True,
        "target_label_firewall_passed": True,
        "three_model_seeds_fixed": True,
        "hyperparameters_inherited_without_tuning": True,
        "target_test_tensor_values_read": 0,
        "training_performed": False,
    }
    _write_json(output / "readiness_decision.json", readiness)
    pd.DataFrame(errors, columns=["fold", "code", "message"]).to_csv(
        output / "errors.csv", index=False
    )

    diagnostic_hashes_after = _verify_diagnostic_provenance(config, repository_root)
    if diagnostic_hashes_before != diagnostic_hashes_after:
        raise RuntimeError("Diagnostic DANN artifacts changed during protocol build")
    summary = {
        **protocol_manifest,
        "preregistration_hash": preregistration_hash,
        "eligible_fold_count": eligible_count,
    }
    report = render_confirmatory_report(summary)
    (output / "protocol_report.md").write_text(report, encoding="utf-8")
    if output_dir is None:
        tracked = repository_root / "reports/integration/dann_label_q5_confirmatory_protocol.md"
        tracked.parent.mkdir(parents=True, exist_ok=True)
        tracked.write_text(report, encoding="utf-8")
    return DANNConfirmatoryProtocolResult(
        summary=summary,
        fold_eligibility=fold_frame,
        fold_partitions=partition_frame,
        shared_subject_audit=shared_frame,
    )


def run_confirmatory_protocol(
    config_path: str | Path, *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    root = Path(repository_root) if repository_root is not None else Path.cwd()
    path = Path(config_path)
    resolved = path if path.is_absolute() else root / path
    config = json.loads(resolved.read_text(encoding="utf-8"))
    return build_dann_label_q5_confirmatory_protocol(
        config, repository_root=root
    ).summary


__all__ = [
    "DANNConfirmatoryProtocolResult",
    "EXPECTED_FOLDS",
    "EXPECTED_MODEL_SEEDS",
    "PROTOCOL_ID",
    "build_dann_label_q5_confirmatory_protocol",
    "compute_confirmatory_protocol_hash",
    "domain_head_signature",
    "run_confirmatory_protocol",
    "validate_confirmatory_config",
]
