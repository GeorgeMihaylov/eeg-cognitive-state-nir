"""Prepare the disabled DANN confirmatory-v2 analysis protocol.

Version 2 is a metadata-only layer over the immutable task-8E partitions.  It
separates new primary seeds from the already observed fold-1/seed-42 cell.  No
raw EEG arrays, optimizer, backward pass, inference, or training are used.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from bench.meta.episodes import stable_hash

from .dann_label_q5_confirmatory_protocol import EXPECTED_FOLDS
from .dann_label_q5_raw_diagnostic import prepare_preregistration
from .dann_label_q5_raw_protocol import (
    _contains_absolute_path,
    _sha256_file,
    _write_json,
)


SCHEMA_VERSION = "dann-label-q5-confirmatory-v2-protocol-v1"
PROTOCOL_ID = "dann_label_q5_old_eeg_to_gpn_confirmatory_v2"
PRIMARY_SEEDS = (123, 2026)
SECONDARY_SEEDS = (42,)
MODES = ("source_only_matched", "dann")
OBSERVED_CELL = (1, 42)
ANALYSIS_GROUPS = (
    "primary_confirmatory",
    "secondary_sensitivity",
    "previously_observed_diagnostic",
)


@dataclass(frozen=True)
class DANNConfirmatoryV2ProtocolResult:
    summary: dict[str, Any]
    run_matrix: pd.DataFrame
    fold_partitions: pd.DataFrame


def _git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repository_root,
        text=True,
    ).strip()


def validate_confirmatory_v2_config(config: Mapping[str, Any]) -> None:
    if config.get("execution_enabled") is not False:
        raise ValueError("confirmatory-v2 execution must remain disabled")
    if config.get("rerun_observed_diagnostic") is not False:
        raise ValueError("fold-1 seed-42 diagnostic rerun must remain disabled")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")
    if tuple(map(int, config.get("outer_folds", ()))) != EXPECTED_FOLDS:
        raise ValueError("all five fixed outer folds are required")
    if tuple(map(int, config.get("primary_seeds", ()))) != PRIMARY_SEEDS:
        raise ValueError("primary seeds must be exactly 123 and 2026")
    if tuple(map(int, config.get("secondary_seeds", ()))) != SECONDARY_SEEDS:
        raise ValueError("secondary seed must be exactly 42")
    if tuple(config.get("modes", ())) != MODES:
        raise ValueError("both matched source-only and DANN modes are required")
    observed = config.get("observed_diagnostic_cell", {})
    if (int(observed.get("fold", -1)), int(observed.get("seed", -1))) != OBSERVED_CELL:
        raise ValueError("the observed diagnostic cell must be fold 1 / seed 42")
    if observed.get("analysis_group") != "previously_observed_diagnostic":
        raise ValueError("the observed cell must remain diagnostic evidence")
    if config.get("direction") != "Old_EEG_to_gpn_data":
        raise ValueError("the diagnostic direction is frozen")
    training = config["training"]
    required_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "source_batch_size": 32,
        "target_batch_size": 32,
        "maximum_epochs": 12,
        "early_stopping_patience": 3,
        "gradient_clip_norm": 5.0,
        "grl_alpha_schedule": "alpha(p)=2/(1+exp(-10*p))-1",
        "domain_lambda_schedule": "constant_1.0",
        "checkpoint_primary": "source_validation_macro_f1",
        "checkpoint_secondary": "source_validation_balanced_accuracy",
        "checkpoint_partition": "source_validation",
        "domain_accuracy_selects_checkpoint": False,
        "source_validation_split_seed": 42,
        "source_validation_shared_across_model_seeds": True,
        "target_train_task_labels_accessible": False,
    }
    if any(training.get(key) != value for key, value in required_training.items()):
        raise ValueError("confirmatory-v2 hyperparameters differ from task 8E")
    if config["secondary_aggregation"]["may_change_primary_decision"] is not False:
        raise ValueError("sensitivity results cannot alter the primary decision")
    if _contains_absolute_path(config):
        raise ValueError("tracked v2 config contains an absolute path")


def refuse_observed_diagnostic_rerun(
    fold: int,
    seed: int,
    *,
    allow_technical_reproduction: bool = False,
) -> None:
    """Guard future orchestration from silently rerunning discovery evidence."""
    if (int(fold), int(seed)) == OBSERVED_CELL and not allow_technical_reproduction:
        raise RuntimeError(
            "fold 1 / seed 42 is previously observed diagnostic evidence; "
            "an explicit technical-reproduction authorization is required"
        )


def build_run_matrix(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        group: str,
        fold: int,
        seed: int,
        mode: str,
        status: str,
        provenance: str,
    ) -> None:
        rows.append({
            "run_id": f"fold_{fold:02d}_seed_{seed}_{mode}",
            "analysis_group": group,
            "fold": int(fold),
            "seed": int(seed),
            "mode": mode,
            "execution_status": status,
            "provenance": provenance,
        })

    for fold in EXPECTED_FOLDS:
        for seed in PRIMARY_SEEDS:
            for mode in MODES:
                add(
                    "primary_confirmatory", fold, seed, mode,
                    "planned_disabled", PROTOCOL_ID,
                )
    for fold in EXPECTED_FOLDS[1:]:
        for mode in MODES:
            add(
                "secondary_sensitivity", fold, 42, mode,
                "planned_disabled", PROTOCOL_ID,
            )
    diagnostic_source = str(config["diagnostic_reference"]["source"])
    for mode in MODES:
        add(
            "previously_observed_diagnostic", 1, 42, mode,
            "already_completed", diagnostic_source,
        )
    frame = pd.DataFrame(rows).sort_values(
        ["analysis_group", "fold", "seed", "mode"], kind="stable"
    ).reset_index(drop=True)
    if frame["run_id"].duplicated().any():
        raise RuntimeError("run matrix contains duplicate run_id")
    return frame


def compute_confirmatory_v2_protocol_hash(
    *,
    run_matrix: Sequence[Mapping[str, Any]],
    scientific_contract: Mapping[str, Any],
) -> str:
    return stable_hash({
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_matrix": list(run_matrix),
        "scientific_contract": dict(scientific_contract),
    })


def aggregate_participant_deltas(
    results: pd.DataFrame,
    *,
    analysis_groups: Sequence[str] = ("primary_confirmatory",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair modes and average seeds per participant without window weights."""
    required = {
        "analysis_group", "fold", "seed", "subject_id", "mode",
        "macro_f1", "balanced_accuracy", "ordinal_mae",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing participant-result columns: {sorted(missing)}")
    selected = results.loc[results["analysis_group"].isin(analysis_groups)].copy()
    if selected.empty:
        raise ValueError("No rows for the requested analysis groups")
    index = ["fold", "seed", "subject_id"]
    paired = selected.pivot(
        index=index,
        columns="mode",
        values=["macro_f1", "balanced_accuracy", "ordinal_mae"],
    )
    for metric in ("macro_f1", "balanced_accuracy", "ordinal_mae"):
        for mode in MODES:
            if (metric, mode) not in paired.columns:
                raise ValueError(f"Missing mode {mode!r} for metric {metric!r}")
    paired = paired.reset_index()
    flattened = index + [
        f"{metric}_{mode}" for metric, mode in paired.columns[len(index):]
    ]
    paired.columns = flattened
    for metric in ("macro_f1", "balanced_accuracy", "ordinal_mae"):
        paired[f"delta_{metric}"] = (
            paired[f"{metric}_dann"] - paired[f"{metric}_source_only_matched"]
        )
    numeric = [
        column for column in paired.columns
        if column not in {"fold", "seed", "subject_id"}
    ]
    participants = (
        paired.groupby(["fold", "subject_id"], sort=True, as_index=False)[numeric]
        .mean()
    )
    seed_counts = paired.groupby(["fold", "subject_id"])["seed"].nunique()
    participants["seed_count"] = seed_counts.to_numpy()
    participants["participant_weight"] = 1.0
    return paired, participants


def apply_primary_decision_rule(
    paired_seed_results: pd.DataFrame,
    participant_results: pd.DataFrame,
    rule: Mapping[str, Any],
    *,
    protocol_valid: bool = True,
) -> dict[str, Any]:
    if not protocol_valid:
        return {"status": "blocked", "protocol_valid": False}
    mean_delta = float(participant_results["delta_macro_f1"].mean())
    median_delta = float(participant_results["delta_macro_f1"].median())
    mean_balanced = float(participant_results["delta_balanced_accuracy"].mean())
    win_fraction = float((participant_results["delta_macro_f1"] > 0).mean())
    fold_means = participant_results.groupby("fold")["delta_macro_f1"].mean()
    seed_means = paired_seed_results.groupby("seed")["delta_macro_f1"].mean()
    positive_folds = int((fold_means > 0).sum())
    nonnegative_folds = int((fold_means >= 0).sum())
    nonnegative_seeds = int((seed_means >= 0).sum())
    confirmed = (
        mean_delta >= float(rule["confirmed_mean_delta_macro_f1_min"])
        and median_delta > 0
        and mean_balanced >= float(rule["confirmed_mean_delta_balanced_accuracy_min"])
        and win_fraction >= float(rule["confirmed_participant_win_fraction_min"])
        and nonnegative_folds >= int(rule["confirmed_nonnegative_fold_count_min"])
        and nonnegative_seeds >= int(rule["confirmed_nonnegative_primary_seed_count"])
    )
    not_confirmed = (
        mean_delta <= float(rule["not_confirmed_overall_mean_max"])
        or positive_folds <= int(rule["not_confirmed_positive_fold_count_max"])
        or win_fraction < float(rule["not_confirmed_participant_win_fraction_below"])
    )
    if confirmed:
        status = "confirmed"
    elif not_confirmed:
        status = "not_confirmed"
    else:
        status = "partially_confirmed"
    return {
        "status": status,
        "mean_delta_macro_f1": mean_delta,
        "median_delta_macro_f1": median_delta,
        "mean_delta_balanced_accuracy": mean_balanced,
        "participant_win_fraction": win_fraction,
        "positive_fold_count": positive_folds,
        "nonnegative_fold_count": nonnegative_folds,
        "nonnegative_primary_seed_count": nonnegative_seeds,
        "protocol_valid": True,
    }


def _reference_hashes(
    config: Mapping[str, Any], repository_root: Path
) -> dict[str, dict[str, str]]:
    v1 = config["confirmatory_v1_reference"]
    diagnostic = config["diagnostic_reference"]
    v1_mapping = {
        "protocol_manifest": "protocol_manifest_sha256",
        "preregistration": "preregistration_hash",
        "fold_partitions": "fold_partitions_sha256",
        "model_seed_manifest": "model_seed_manifest_sha256",
        "target_label_firewall": "target_label_firewall_sha256",
    }
    diagnostic_mapping = {
        "source_only_checkpoint": "source_only_checkpoint_sha256",
        "dann_checkpoint": "dann_checkpoint_sha256",
        "target_test_predictions": "target_test_predictions_sha256",
        "participant_metrics": "participant_metrics_sha256",
        "diagnostic_summary": "diagnostic_summary_sha256",
    }

    def verify(reference: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, str]:
        observed = {
            name: _sha256_file(repository_root / str(reference[name]))
            for name in mapping
        }
        expected = {name: str(reference[hash_key]) for name, hash_key in mapping.items()}
        if observed != expected:
            raise RuntimeError("referenced diagnostic or task-8E artifact changed")
        return observed

    return {
        "confirmatory_v1": verify(v1, v1_mapping),
        "diagnostic": verify(diagnostic, diagnostic_mapping),
    }


def _audit_v1_contract(
    config: Mapping[str, Any], repository_root: Path
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    reference = config["confirmatory_v1_reference"]
    manifest = json.loads(
        (repository_root / str(reference["protocol_manifest"])).read_text(encoding="utf-8")
    )
    if manifest["protocol_hash"] != reference["protocol_hash"]:
        raise RuntimeError("task-8E protocol hash changed")
    if manifest["preregistration_hash"] != reference["preregistration_hash"]:
        raise RuntimeError("task-8E preregistration hash changed")
    if manifest["raw_universe_hash"] != config["raw_universe_hash"]:
        raise RuntimeError("raw universe changed since task 8E")
    if manifest["eligible_folds"] != list(EXPECTED_FOLDS):
        raise RuntimeError("task-8E five-fold eligibility changed")
    v1_training = manifest["training_contract"]
    if any(
        v1_training.get(key) != value
        for key, value in config["training"].items()
        if key in v1_training
    ):
        raise RuntimeError("task-8E training contract changed")
    architecture = manifest["architecture_audit"]
    expected_architecture = config["architecture"]
    architecture_checks = {
        "architecture_signature": expected_architecture["architecture_signature"],
        "domain_head_signature": expected_architecture["domain_head_signature"],
        "task_model_parameter_count": expected_architecture["task_parameter_count"],
        "domain_head_parameter_count": expected_architecture["domain_parameter_count"],
        "latent_dim": expected_architecture["latent_dim"],
    }
    if any(architecture.get(key) != value for key, value in architecture_checks.items()):
        raise RuntimeError("task-8E architecture contract changed")
    firewall = json.loads(
        (repository_root / str(reference["target_label_firewall"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        firewall["all_target_manifests_label_free"] is not True
        or firewall["target_test_tensor_values_read"] != 0
        or firewall["target_train_task_labels_accessible"] is not False
    ):
        raise RuntimeError("task-8E target-label firewall is not intact")
    seed_manifest = json.loads(
        (repository_root / str(reference["model_seed_manifest"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        seed_manifest["source_validation_split_seed"] != 42
        or seed_manifest["outer_folds_and_source_splits_independent_of_model_seed"]
        is not True
    ):
        raise RuntimeError("task-8E source-validation seed contract changed")
    partitions = pd.read_parquet(repository_root / str(reference["fold_partitions"]))
    if sorted(partitions["fold"].unique().tolist()) != list(EXPECTED_FOLDS):
        raise RuntimeError("task-8E fold partitions are incomplete")
    root = (repository_root / str(reference["protocol_manifest"])).parent
    budgets: list[dict[str, Any]] = []
    source_splits: dict[str, str] = {}
    for fold in EXPECTED_FOLDS:
        batch_path = root / f"batching_contracts/fold_{fold:02d}.json"
        source_path = root / f"source_validation_manifests/fold_{fold:02d}.json"
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8"))
        budgets.append({
            "fold": fold,
            "source_natural_steps": batch["source_natural_steps"],
            "target_natural_steps": batch["target_natural_steps"],
            "matched_steps_per_epoch": batch["matched_steps_per_epoch"],
            "maximum_epochs": batch["maximum_epochs"],
            "artifact_sha256": _sha256_file(batch_path),
        })
        source_splits[str(fold)] = stable_hash({
            "source_task_train": source["source_task_train"]["sample_ids"],
            "source_validation": source["source_validation"]["sample_ids"],
            "split_seed": 42,
        })
    return manifest, partitions, budgets, {
        "source_validation_partition_hashes": source_splits,
        "target_label_firewall": firewall,
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    budget_rows = summary["matched_update_budgets"]
    lines = [
        "# DANN confirmatory-v2 analysis protocol",
        "",
        f"- Branch/HEAD: `integration/benchmark-unification` / `{summary['git_commit']}`.",
        "- Diagnostic discovery: Old_EEG to gpn_data, fold 1, seed 42, status `proceed`; mean participant macro-F1 delta +0.013364, but its bootstrap interval crosses zero.",
        "- Fold 1 / seed 42 was already inspected and is retained as discovery evidence, not independent confirmation.",
        "- Primary confirmation uses only new seeds 123 and 2026 across all five folds: 20 planned model runs.",
        "- Secondary sensitivity uses seed 42 on folds 2-5: 8 new runs. The two fold-1/seed-42 mode results are referenced, not retrained.",
        "- Total result cells: 30; new training runs: 28. Training has not started.",
        "",
        "## Run groups",
        "",
        "| analysis group | folds | seeds | modes | results | status |",
        "|---|---|---|---|---:|---|",
        "| primary_confirmatory | 1-5 | 123, 2026 | source_only_matched, dann | 20 | planned_disabled |",
        "| secondary_sensitivity | 2-5 | 42 | source_only_matched, dann | 8 | planned_disabled |",
        "| previously_observed_diagnostic | 1 | 42 | source_only_matched, dann | 2 | already_completed |",
        "",
        "All five task-8E outer-fold hashes, fold partitions, strict shared-participant policy, source-validation splits, target-label firewall, architecture, hyperparameters, and matched budgets are unchanged.",
        "",
        "## Matched update budgets",
        "",
        "| fold | source natural | target natural | matched per epoch | max epochs |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in budget_rows:
        lines.append(
            f"| {row['fold']} | {row['source_natural_steps']} | "
            f"{row['target_natural_steps']} | {row['matched_steps_per_epoch']} | "
            f"{row['maximum_epochs']} |"
        )
    lines.extend([
        "",
        "## Analysis and target-test locks",
        "",
        "Primary mode differences are paired within fold/seed/participant, averaged within each participant across seeds 123 and 2026, then aggregated with equal participant weight. Fold and seed summaries are reported separately; no best seed is selected.",
        "",
        "The primary decision uses only `primary_confirmatory`. The combined three-seed sensitivity analysis is reported separately and cannot change the primary status.",
        "",
        "Each of 14 new fold/seed pairs has its own locked unlock contract. Target test remains unavailable until both checkpoint hashes, best epochs, source-validation metrics, batch-sequence hashes, and v2 protocol/preregistration hashes are fixed. The diagnostic unlock is not reusable.",
        "",
        "The primary `confirmed` rule additionally requires both primary seeds and at least four folds to have nonnegative mean participant macro-F1 deltas, alongside the preregistered effect, median, balanced-accuracy, and win-fraction thresholds.",
        "",
        f"Task-8E protocol/preregistration: `{summary['confirmatory_v1_protocol_hash']}` / `{summary['confirmatory_v1_preregistration_hash']}`.",
        f"V2 protocol/preregistration: `{summary['protocol_hash']}` / `{summary['preregistration_hash']}`.",
        f"Readiness: **{summary['status']}**; execution enabled: `{str(summary['execution_enabled']).lower()}`.",
        "",
        "Limitations: this is a disabled protocol, not a scientific result. No new DANN/source-only model, inference, target-test access, or statistical analysis was run.",
        "",
    ])
    return "\n".join(lines)


def build_dann_label_q5_confirmatory_v2_protocol(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    output_dir: Path | None = None,
) -> DANNConfirmatoryV2ProtocolResult:
    validate_confirmatory_v2_config(config)
    root = Path(repository_root)
    output = output_dir or root / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    reference_hashes_before = _reference_hashes(config, root)
    v1, partitions, budgets, audits = _audit_v1_contract(config, root)
    run_matrix = build_run_matrix(config)
    primary = run_matrix.loc[run_matrix["analysis_group"].eq("primary_confirmatory")]
    secondary = run_matrix.loc[run_matrix["analysis_group"].eq("secondary_sensitivity")]
    completed = run_matrix.loc[
        run_matrix["analysis_group"].eq("previously_observed_diagnostic")
    ]
    if len(primary) != 20 or len(secondary) != 8 or len(completed) != 2:
        raise RuntimeError("confirmatory-v2 run matrices have unexpected sizes")
    if ((primary["fold"].eq(1)) & (primary["seed"].eq(42))).any():
        raise RuntimeError("observed diagnostic cell leaked into primary analysis")
    if ((secondary["fold"].eq(1)) & (secondary["seed"].eq(42))).any():
        raise RuntimeError("observed diagnostic cell would be rerun as sensitivity")
    for row in completed.itertuples():
        try:
            refuse_observed_diagnostic_rerun(row.fold, row.seed)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("observed diagnostic rerun guard did not reject the cell")

    scientific_contract = {
        "raw_universe_hash": config["raw_universe_hash"],
        "outer_fold_hashes": v1["outer_fold_hashes"],
        "primary_seeds": list(PRIMARY_SEEDS),
        "secondary_seeds": list(SECONDARY_SEEDS),
        "observed_diagnostic_cell": dict(config["observed_diagnostic_cell"]),
        "diagnostic_reference": dict(config["diagnostic_reference"]),
        "fold_partitions_sha256": config["confirmatory_v1_reference"]["fold_partitions_sha256"],
        "source_validation_partition_hashes": audits["source_validation_partition_hashes"],
        "architecture": dict(config["architecture"]),
        "training": dict(config["training"]),
        "matched_update_budgets": budgets,
        "primary_aggregation": dict(config["primary_aggregation"]),
        "secondary_aggregation": dict(config["secondary_aggregation"]),
        "primary_decision_rule": dict(config["primary_decision_rule"]),
        "secondary_sensitivity_rule": dict(config["secondary_sensitivity_rule"]),
        "target_test_lock": dict(config["target_test_lock"]),
    }
    records = run_matrix.to_dict("records")
    protocol_hash = compute_confirmatory_v2_protocol_hash(
        run_matrix=records, scientific_contract=scientific_contract
    )
    preregistration = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "execution_enabled": False,
        "rerun_observed_diagnostic": False,
        "repository_commit": _git_head(root),
        "confirmatory_v1_reference": dict(config["confirmatory_v1_reference"]),
        "diagnostic_reference": dict(config["diagnostic_reference"]),
        "primary_seeds": list(PRIMARY_SEEDS),
        "secondary_seeds": list(SECONDARY_SEEDS),
        "observed_diagnostic_cell": dict(config["observed_diagnostic_cell"]),
        "run_matrix": records,
        "scientific_contract": scientific_contract,
        "target_test_locked": True,
        "training_performed": False,
    }
    if "allow_technical_reproduction" in preregistration:
        raise RuntimeError("technical reproduction flag entered preregistration")
    if _contains_absolute_path(preregistration):
        raise RuntimeError("v2 preregistration contains an absolute path")

    preregistration_path = output / "preregistration/experiment_preregistration.json"
    preregistration_hash = prepare_preregistration(preregistration_path, preregistration)
    _write_json(output / "preregistration/preregistration_hash.json", {
        "algorithm": "sha256", "sha256": preregistration_hash,
        "execution_enabled": False,
    })
    lock_dir = output / "target_test_unlock_contracts"
    for fold, seed in sorted(
        set(zip(
            pd.concat([primary, secondary])["fold"].astype(int),
            pd.concat([primary, secondary])["seed"].astype(int),
        ))
    ):
        _write_json(lock_dir / f"fold_{fold:02d}_seed_{seed}.json", {
            "fold": fold,
            "seed": seed,
            "status": "locked",
            "target_test_opened": False,
            "diagnostic_unlock_reused": False,
            "required_before_unlock": list(
                config["target_test_lock"]["required_before_unlock"]
            ),
            "protocol_hash": protocol_hash,
            "preregistration_hash": preregistration_hash,
            "source_only_checkpoint_hash": None,
            "dann_checkpoint_hash": None,
            "source_only_best_epoch": None,
            "dann_best_epoch": None,
            "source_validation_metrics": None,
            "batch_sequence_hashes": None,
        })

    run_matrix.to_csv(output / "run_matrix.csv", index=False)
    primary.to_csv(output / "primary_run_matrix.csv", index=False)
    secondary.to_csv(output / "secondary_run_matrix.csv", index=False)
    completed.to_csv(output / "completed_run_matrix.csv", index=False)
    source_partitions = root / str(config["confirmatory_v1_reference"]["fold_partitions"])
    shutil.copyfile(source_partitions, output / "fold_partitions.parquet")
    _write_json(output / "protocol_reference.json", {
        "confirmatory_v1": dict(config["confirmatory_v1_reference"]),
        "reference_hashes": reference_hashes_before["confirmatory_v1"],
        "partitions_rebuilt": False,
    })
    _write_json(output / "diagnostic_cell_reference.json", {
        **dict(config["diagnostic_reference"]),
        "reference_hashes": reference_hashes_before["diagnostic"],
        "analysis_group": "previously_observed_diagnostic",
        "fold": 1,
        "seed": 42,
        "rerun": False,
    })
    _write_json(output / "model_seed_manifest.json", {
        "primary_seeds": list(PRIMARY_SEEDS),
        "secondary_seeds": list(SECONDARY_SEEDS),
        "source_validation_split_seed": 42,
        "source_validation_shared_across_model_seeds": True,
        "primary_seed_selection_after_results": False,
    })
    _write_json(output / "aggregation_contract.json", {
        "primary": dict(config["primary_aggregation"]),
        "secondary": dict(config["secondary_aggregation"]),
    })
    _write_json(output / "primary_decision_rule.json", config["primary_decision_rule"])
    _write_json(
        output / "secondary_sensitivity_rule.json",
        config["secondary_sensitivity_rule"],
    )
    protocol_manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash,
        "status": "confirmatory_v2_protocol_ready",
        "execution_enabled": False,
        "git_commit": _git_head(root),
        "raw_universe_hash": config["raw_universe_hash"],
        "confirmatory_v1_protocol_hash": config["confirmatory_v1_reference"]["protocol_hash"],
        "confirmatory_v1_preregistration_hash": config["confirmatory_v1_reference"]["preregistration_hash"],
        "diagnostic_protocol_hash": config["diagnostic_reference"]["protocol_hash"],
        "diagnostic_preregistration_hash": config["diagnostic_reference"]["preregistration_hash"],
        "primary_seeds": list(PRIMARY_SEEDS),
        "secondary_seeds": list(SECONDARY_SEEDS),
        "primary_runs": len(primary),
        "new_secondary_runs": len(secondary),
        "completed_diagnostic_results": len(completed),
        "new_training_runs": len(primary) + len(secondary),
        "total_model_results": len(run_matrix),
        "outer_fold_hashes": v1["outer_fold_hashes"],
        "fold_partitions_sha256": config["confirmatory_v1_reference"]["fold_partitions_sha256"],
        "source_validation_partition_hashes": audits["source_validation_partition_hashes"],
        "matched_update_budgets": budgets,
        "architecture": dict(config["architecture"]),
        "training": dict(config["training"]),
        "target_label_firewall": audits["target_label_firewall"],
        "primary_aggregation": dict(config["primary_aggregation"]),
        "secondary_aggregation": dict(config["secondary_aggregation"]),
        "primary_decision_rule": dict(config["primary_decision_rule"]),
        "target_test_unlock_contract_count": 14,
        "preregistration_hash": preregistration_hash,
        "optimizer_created": False,
        "backward_called": False,
        "training_performed": False,
        "cuda_tensor_created": False,
        "target_test_tensor_values_read": 0,
        "target_test_inference_performed": False,
        "statistical_analysis_performed": False,
        "partitions_rebuilt": False,
    }
    _write_json(output / "protocol_manifest.json", protocol_manifest)
    _write_json(output / "protocol_hash.json", {
        "algorithm": "sha256",
        "canonical_serialization": "stable_hash sorted JSON",
        "protocol_hash": protocol_hash,
    })
    readiness = {
        "status": "confirmatory_v2_protocol_ready",
        "execution_enabled": False,
        "primary_and_secondary_separated": True,
        "diagnostic_cell_excluded_from_primary": True,
        "all_five_primary_folds_present": True,
        "diagnostic_provenance_verified": True,
        "partitions_and_hyperparameters_unchanged": True,
        "target_test_locked": True,
        "training_performed": False,
    }
    _write_json(output / "readiness_decision.json", readiness)
    pd.DataFrame(columns=["code", "message"]).to_csv(output / "errors.csv", index=False)

    reference_hashes_after = _reference_hashes(config, root)
    if reference_hashes_before != reference_hashes_after:
        raise RuntimeError("diagnostic or task-8E runtime changed during v2 build")
    if _sha256_file(output / "fold_partitions.parquet") != str(
        config["confirmatory_v1_reference"]["fold_partitions_sha256"]
    ):
        raise RuntimeError("copied fold partitions differ from task 8E")
    summary = dict(protocol_manifest)
    report = _render_report(summary)
    (output / "protocol_report.md").write_text(report, encoding="utf-8")
    if output_dir is None:
        tracked = root / "reports/integration/dann_label_q5_confirmatory_v2_protocol.md"
        tracked.parent.mkdir(parents=True, exist_ok=True)
        tracked.write_text(report, encoding="utf-8")
    return DANNConfirmatoryV2ProtocolResult(
        summary=summary,
        run_matrix=run_matrix,
        fold_partitions=partitions,
    )


def run_confirmatory_v2_protocol(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root) if repository_root is not None else Path.cwd()
    path = Path(config_path)
    resolved = path if path.is_absolute() else root / path
    config = json.loads(resolved.read_text(encoding="utf-8"))
    return build_dann_label_q5_confirmatory_v2_protocol(
        config, repository_root=root
    ).summary


__all__ = [
    "ANALYSIS_GROUPS",
    "DANNConfirmatoryV2ProtocolResult",
    "MODES",
    "OBSERVED_CELL",
    "PRIMARY_SEEDS",
    "PROTOCOL_ID",
    "SECONDARY_SEEDS",
    "aggregate_participant_deltas",
    "apply_primary_decision_rule",
    "build_dann_label_q5_confirmatory_v2_protocol",
    "build_run_matrix",
    "compute_confirmatory_v2_protocol_hash",
    "refuse_observed_diagnostic_rerun",
    "run_confirmatory_v2_protocol",
    "validate_confirmatory_v2_config",
]
