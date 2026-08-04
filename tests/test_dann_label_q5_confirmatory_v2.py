from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.dann_label_q5_confirmatory_v2 import (
    DANNConfirmatoryV2Execution,
    _contains_absolute_path,
    _registry_signature,
    average_participants_across_seeds,
    bootstrap_unique_participants,
    build_execution_registry,
    fold_level_metrics,
    pair_subject_metrics,
    seed_level_metrics,
    validate_confirmatory_v2_execution_config,
)
from bench.experiments.dann_label_q5_confirmatory_v2_protocol import (
    apply_primary_decision_rule,
)
from bench.experiments.dann_label_q5_raw_diagnostic import (
    MODES,
    TargetTestLock,
    enforce_target_batch_firewall,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments/domain_adaptation/dann_label_q5_confirmatory_v2_execution.json"
)
PROTOCOL_ROOT = (
    ROOT / "benchmark_results/domain_adaptation_dann_confirmatory_v2_protocol"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _matrix() -> pd.DataFrame:
    return pd.read_csv(PROTOCOL_ROOT / "run_matrix.csv")


def _subject_metrics() -> pd.DataFrame:
    rows = []
    for fold in (1, 2):
        for seed in (123, 2026):
            for subject_index in range(3):
                subject_id = f"subject-{fold}-{subject_index}"
                reference = 0.20 + 0.01 * subject_index
                for mode in MODES:
                    gain = 0.02 if mode == "dann" else 0.0
                    rows.append({
                        "fold": fold,
                        "seed": seed,
                        "subject_id": subject_id,
                        "mode": mode,
                        "accuracy": reference + gain,
                        "balanced_accuracy": reference + gain,
                        "macro_f1": reference + gain,
                        "weighted_f1": reference + gain,
                        "kappa": reference + gain,
                        "ordinal_mae": 1.0 - gain,
                        "quadratic_weighted_kappa": reference + gain,
                        "macro_precision": reference + gain,
                        "macro_recall": reference + gain,
                        "prediction_entropy": 1.0 - gain,
                    })
    return pd.DataFrame(rows)


def test_execution_config_is_enabled_fixed_and_portable() -> None:
    config = _config()
    validate_confirmatory_v2_execution_config(config)
    assert config["execution_enabled"] is True
    assert config["protocol"]["protocol_hash"] == (
        "1ce582a3d73a7ae4393e77cc2f3b2cb7749ddbb30c1cb8fcad0056c6d326c368"
    )
    assert config["protocol"]["disabled_preregistration_hash"] == (
        "6fba1eb76133884f0d5984ec1ceedc49234f252846040122310ac45a99ad3d7e"
    )
    assert not _contains_absolute_path(config)
    assert not re.search(r"[A-Za-z]:[\\/]", CONFIG_PATH.read_text(encoding="utf-8"))


def test_fixed_hyperparameters_cannot_be_changed() -> None:
    config = _config()
    changed = copy.deepcopy(config)
    changed["training"]["learning_rate"] = 0.002
    with pytest.raises(ValueError, match="hyperparameters"):
        validate_confirmatory_v2_execution_config(changed)


def test_run_matrix_has_exact_primary_secondary_and_diagnostic_cells() -> None:
    matrix = _matrix()
    counts = matrix.groupby("analysis_group").size().to_dict()
    assert counts == {
        "previously_observed_diagnostic": 2,
        "primary_confirmatory": 20,
        "secondary_sensitivity": 8,
    }
    primary = matrix[matrix["analysis_group"].eq("primary_confirmatory")]
    secondary = matrix[matrix["analysis_group"].eq("secondary_sensitivity")]
    assert sorted(primary["fold"].unique()) == [1, 2, 3, 4, 5]
    assert sorted(primary["seed"].unique()) == [123, 2026]
    assert sorted(secondary["fold"].unique()) == [2, 3, 4, 5]
    assert secondary["seed"].unique().tolist() == [42]


def test_diagnostic_cell_is_never_a_new_training_run() -> None:
    matrix = _matrix()
    diagnostic_cell = matrix[matrix["fold"].eq(1) & matrix["seed"].eq(42)]
    assert len(diagnostic_cell) == 2
    assert diagnostic_cell["analysis_group"].eq(
        "previously_observed_diagnostic"
    ).all()
    assert diagnostic_cell["execution_status"].eq("already_completed").all()


def test_registry_is_deterministic_and_diagnostic_is_precompleted() -> None:
    first = build_execution_registry(_matrix())
    second = build_execution_registry(_matrix())
    assert _registry_signature(first) == _registry_signature(second)
    diagnostic = [
        row for row in first
        if row["analysis_group"] == "previously_observed_diagnostic"
    ]
    new = [row for row in first if row not in diagnostic]
    assert len(new) == 28
    assert all(row["status"] == "pending" for row in new)
    assert all(row["status"] == "complete" for row in diagnostic)
    assert all(row["attempt_count"] == 0 for row in first)


def test_complete_pair_is_skipped_without_creating_a_runner(monkeypatch) -> None:
    execution = object.__new__(DANNConfirmatoryV2Execution)
    execution.registry = [
        {
            "analysis_group": "primary_confirmatory", "fold": 1,
            "seed": 123, "mode": mode, "status": "complete",
            "attempt_count": 1, "technical_restart_count": 0,
        }
        for mode in MODES
    ]
    monkeypatch.setattr(
        "bench.experiments.dann_label_q5_confirmatory_v2._ConfirmatoryPairRunner",
        lambda *args, **kwargs: pytest.fail("complete pair was silently retrained"),
    )
    execution._run_pair("primary_confirmatory", 1, 123, resume=True)


def test_primary_lock_is_required_before_secondary(tmp_path: Path) -> None:
    execution = object.__new__(DANNConfirmatoryV2Execution)
    execution.output = tmp_path
    with pytest.raises(RuntimeError, match="primary result lock"):
        execution._require_primary_lock()


def test_target_test_lock_is_pair_specific_and_rejects_early_reads() -> None:
    first = TargetTestLock()
    second = TargetTestLock()
    with pytest.raises(RuntimeError, match="locked"):
        first.require_access()
    first.unlock("a" * 64)
    first.require_access()
    assert first.reads_before_unlock == 1
    assert first.reads_after_unlock == 1
    assert second.is_unlocked is False


def test_target_training_firewall_rejects_task_labels() -> None:
    safe = {
        "eeg": np.zeros((1, 14, 2560), dtype=np.float32),
        "domain_label": 0,
        "sample_id": "s", "subject_id": "p", "record_group_id": "r",
    }
    enforce_target_batch_firewall(safe)
    with pytest.raises(RuntimeError, match="task_label"):
        enforce_target_batch_firewall({**safe, "task_label": 3})


def test_pairing_requires_both_modes_and_computes_dann_minus_source() -> None:
    metrics = _subject_metrics()
    paired = pair_subject_metrics(metrics)
    assert len(paired) == 12
    assert np.allclose(paired["delta_macro_f1"], 0.02)
    assert np.allclose(paired["delta_balanced_accuracy"], 0.02)
    assert np.allclose(paired["delta_ordinal_mae"], -0.02)
    incomplete = metrics[~(
        metrics["mode"].eq("dann") & metrics["subject_id"].eq("subject-1-0")
    )]
    with pytest.raises(ValueError, match="Missing"):
        pair_subject_metrics(incomplete)


def test_primary_participants_average_seeds_with_equal_weight() -> None:
    paired = pair_subject_metrics(_subject_metrics())
    participants = average_participants_across_seeds(paired)
    assert len(participants) == 6
    assert participants["seed_count"].eq(2).all()
    assert participants["participant_weight"].eq(1.0).all()
    assert np.allclose(participants["delta_macro_f1"], 0.02)


def test_fold_and_seed_aggregations_keep_all_blocks() -> None:
    paired = pair_subject_metrics(_subject_metrics())
    participants = average_participants_across_seeds(paired)
    folds = fold_level_metrics(participants)
    seeds = seed_level_metrics(paired)
    assert folds["fold"].tolist() == [1, 2]
    assert folds["participants"].tolist() == [3, 3]
    assert folds["wins"].tolist() == [3, 3]
    assert seeds["seed"].tolist() == [123, 2026]
    assert seeds["participants"].tolist() == [6, 6]


def test_participant_bootstrap_is_deterministic_and_not_window_based() -> None:
    participants = average_participants_across_seeds(
        pair_subject_metrics(_subject_metrics())
    )
    first = bootstrap_unique_participants(participants, resamples=250, seed=42)
    second = bootstrap_unique_participants(participants, resamples=250, seed=42)
    assert first == second
    assert first["unit"] == "unique_participant_after_primary_seed_average"
    assert first["n_participants"] == 6
    assert first["statistical_significance_claimed"] is False


def test_primary_decision_is_deterministic() -> None:
    paired = pair_subject_metrics(_subject_metrics())
    participants = average_participants_across_seeds(paired)
    rule = {
        "confirmed_mean_delta_macro_f1_min": 0.01,
        "confirmed_mean_delta_balanced_accuracy_min": 0.0,
        "confirmed_participant_win_fraction_min": 0.6,
        "confirmed_nonnegative_fold_count_min": 2,
        "confirmed_nonnegative_primary_seed_count": 2,
        "not_confirmed_overall_mean_max": 0.0,
        "not_confirmed_positive_fold_count_max": 1,
        "not_confirmed_participant_win_fraction_below": 0.4,
    }
    first = apply_primary_decision_rule(paired, participants, rule)
    second = apply_primary_decision_rule(paired, participants, rule)
    assert first == second
    assert first["status"] == "confirmed"


def test_execution_preregistration_is_written_before_registry(tmp_path: Path) -> None:
    execution = object.__new__(DANNConfirmatoryV2Execution)
    execution.config = _config()
    execution.root = ROOT
    execution.output = tmp_path
    execution.protocol = json.loads(
        (PROTOCOL_ROOT / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    execution.run_matrix = _matrix()
    digest = execution._prepare_preregistration()
    preregistration = tmp_path / "preregistration/experiment_preregistration.json"
    assert preregistration.exists()
    assert len(digest) == 64
    assert json.loads(preregistration.read_text(encoding="utf-8"))[
        "execution_enabled"
    ] is True
    assert not (tmp_path / "run_registry/run_registry.json").exists()

