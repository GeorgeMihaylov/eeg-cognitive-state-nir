from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.fomaml_label_q5_diagnostic import (
    FOMAMLLabelQ5Diagnostic,
    apply_decision_rule,
    select_buffer_policy,
)
from bench.experiments.fomaml_label_q5_raw_diagnostic import (
    EXPECTED_PROTOCOL_HASH,
    EXPECTED_PROTOCOL_ID,
    EXPECTED_RAW_UNIVERSE_HASH,
    FOMAMLLabelQ5RawDiagnostic,
    build_support_budget_analysis,
    paired_subject_bootstrap,
    validate_raw_diagnostic_config,
    validate_raw_episode_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/meta_learning/fomaml_label_q5_raw_diagnostic.json"
PROTOCOL_PATH = ROOT / "benchmark_results/meta_learning_fomaml_label_q5_raw_protocol/protocol_manifest.json"
EPISODE_PATH = ROOT / "benchmark_results/meta_learning_fomaml_label_q5_raw_protocol/episode_index.parquet"
DISABLED_PREREGISTRATION = ROOT / "benchmark_results/meta_learning_fomaml_label_q5_raw_protocol/preregistration/experiment_preregistration.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _protocol() -> tuple[dict, pd.DataFrame]:
    return (
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")),
        pd.read_parquet(EPISODE_PATH),
    )


def test_config_is_exactly_one_authorized_raw_run_without_absolute_paths() -> None:
    config = _config()
    validate_raw_diagnostic_config(config)
    assert config["execution_enabled"] is True
    assert config["protocol"]["id"] == EXPECTED_PROTOCOL_ID
    assert config["protocol"]["expected_hash"] == EXPECTED_PROTOCOL_HASH
    assert config["protocol"]["raw_universe_hash"] == EXPECTED_RAW_UNIVERSE_HASH
    assert config["protocol"]["outer_fold"] == 1
    assert config["seed"] == 42
    assert config["model"]["name"] == "torch_eegnet"
    assert not re.search(r"[A-Za-z]:[\\/]", CONFIG_PATH.read_text(encoding="utf-8"))


def test_raw_protocol_semantic_hash_subjects_and_episode_ids_match() -> None:
    protocol, episodes = _protocol()
    audit = validate_raw_episode_protocol(protocol, episodes, _config())
    assert audit["valid"]
    assert audit["protocol_hash"] == EXPECTED_PROTOCOL_HASH
    assert audit["raw_universe_hash"] == EXPECTED_RAW_UNIVERSE_HASH
    assert audit["scope_counts"] == {
        "meta_train": 11, "meta_validation": 5, "outer_test": 5
    }
    assert audit["subject_counts"] == {
        "meta_train": 11, "meta_validation": 5, "outer_test": 5
    }
    assert audit["missing_raw_ids"] == 0


def test_raw_episodes_are_complete_record_disjoint_and_class_complete() -> None:
    _, episodes = _protocol()
    all_samples: set[str] = set()
    for episode in episodes.itertuples():
        support = set(map(str, episode.support_sample_ids))
        query = set(map(str, episode.query_sample_ids))
        assert support.isdisjoint(query)
        assert set(map(str, episode.support_record_ids)).isdisjoint(
            map(str, episode.query_record_ids)
        )
        assert not all_samples & (support | query)
        all_samples.update(support | query)
        assert set(episode.support_targets) == {0, 1, 2, 3, 4}
        assert set(episode.query_targets) == {0, 1, 2, 3, 4}
        assert episode.split_level == "record"


def test_meta_and_outer_splits_are_fixed_and_disjoint() -> None:
    protocol, episodes = _protocol()
    split = protocol["meta_split"]
    assert split["meta_split_hash"] == _config()["protocol"]["meta_split_hash"]
    meta_train = set(split["meta_train_subjects"])
    meta_validation = set(split["meta_validation_subjects"])
    outer_test = set(protocol["eligible_participants"]["outer_test"])
    assert not meta_train & meta_validation
    assert not meta_train & outer_test
    assert not meta_validation & outer_test
    assert set(episodes.loc[episodes["scope"].eq("meta_train"), "subject_id"]) == meta_train
    assert set(episodes.loc[episodes["scope"].eq("meta_validation"), "subject_id"]) == meta_validation
    assert set(episodes.loc[episodes["scope"].eq("outer_test"), "subject_id"]) == outer_test


def test_disabled_preregistration_is_immutable_when_execution_prereg_is_created(
    tmp_path: Path,
) -> None:
    before = hashlib.sha256(DISABLED_PREREGISTRATION.read_bytes()).hexdigest()
    runner = FOMAMLLabelQ5RawDiagnostic(
        _config(), repository_root=ROOT, output_dir=tmp_path
    )
    protocol, episodes, _ = runner._load_protocol()
    architecture = {"row": {
        "architecture_signature": _config()["model"]["architecture_signature"],
        "latent_dim": 1280,
        "parameter_count": 8501,
        "output_head_width": 5,
    }}
    digest = runner._preregister(protocol, episodes, architecture)
    execution_path = tmp_path / "preregistration/experiment_preregistration.json"
    preregistration = json.loads(execution_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(execution_path.read_bytes()).hexdigest()
    assert preregistration["execution_enabled"] is True
    assert preregistration["outer_test_locked"] is True
    assert preregistration["episode_counts"] == {
        "meta_train": 11, "meta_validation": 5, "outer_test": 5
    }
    assert hashlib.sha256(DISABLED_PREREGISTRATION.read_bytes()).hexdigest() == before


def test_outer_test_gate_writes_unlock_before_delegating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FOMAMLLabelQ5RawDiagnostic(
        _config(), repository_root=ROOT, output_dir=tmp_path
    )
    for relative in (
        "policy_selection.json",
        "experiment_preregistration.json",
        "supervised/checkpoint.pt",
        "fomaml/frozen_global/checkpoint.pt",
        "policy_selection/pre_outer_test_decision.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixed-object")
    observed = {}

    def fake_outer(self, store, rows, supervised, fomaml, policy, decision):
        unlock = tmp_path / "outer_test_unlock_manifest.json"
        observed["unlock_exists"] = unlock.exists()
        observed["unlock"] = json.loads(unlock.read_text(encoding="utf-8"))
        return "predictions", "subjects", "aggregates", "confusion", "audit"

    monkeypatch.setattr(FOMAMLLabelQ5Diagnostic, "_outer_evaluation", fake_outer)
    result = runner._outer_evaluation(
        object(),
        [type("Episode", (), {"episode_id": "outer-episode"})()],
        tmp_path / "supervised/checkpoint.pt",
        tmp_path / "fomaml/frozen_global/checkpoint.pt",
        "frozen_global",
        tmp_path / "policy_selection/pre_outer_test_decision.json",
    )
    assert result[0] == "predictions"
    assert observed["unlock_exists"]
    assert observed["unlock"]["outer_test_used_for_selection"] is False
    assert observed["unlock"]["selected_policy"] == "frozen_global"


def test_outer_test_gate_fails_before_access_without_fixed_decision(
    tmp_path: Path,
) -> None:
    runner = FOMAMLLabelQ5RawDiagnostic(
        _config(), repository_root=ROOT, output_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="decision manifest"):
        runner._outer_evaluation(
            object(), [], tmp_path / "supervised.pt", tmp_path / "fomaml.pt",
            "frozen_global", tmp_path / "missing-decision.json",
        )


def test_policy_selection_is_meta_validation_only_and_uses_frozen_tie_break() -> None:
    selection = select_buffer_policy(
        {
            "frozen_global": {"macro_f1": 0.30, "balanced_accuracy": 0.30},
            "support_local": {"macro_f1": 0.304, "balanced_accuracy": 0.40},
        },
        tie_threshold=0.005,
    )
    assert selection["selected_policy"] == "frozen_global"
    assert selection["outer_test_used"] is False


def _synthetic_subject_metrics() -> pd.DataFrame:
    rows = []
    for index in range(5):
        subject = f"s{index}"
        rows.extend([
            {
                "subject_id": subject,
                "mode": "supervised_full_model",
                "macro_f1": 0.20 + index * 0.01,
                "balanced_accuracy": 0.21 + index * 0.01,
            },
            {
                "subject_id": subject,
                "mode": "selected_fomaml",
                "macro_f1": 0.22 + index * 0.015,
                "balanced_accuracy": 0.225 + index * 0.014,
            },
        ])
    return pd.DataFrame(rows)


def test_support_budget_preserves_variable_sizes_and_class_counts() -> None:
    episodes = pd.DataFrame([
        {
            "scope": "outer_test",
            "subject_id": f"s{index}",
            "episode_id": f"e{index}",
            "support_sample_ids": list(range(128 + index, 256 + 2 * index)),
            "query_sample_ids": list(range(300, 397 + index)),
            "support_record_ids": [f"sr{index}"],
            "query_record_ids": [f"qr{index}", f"qr2-{index}"],
            "support_targets": [0, 1, 2, 3, 4] * 26,
            "query_targets": [0, 1, 2, 3, 4] * 20,
        }
        for index in range(5)
    ])
    frame, summary = build_support_budget_analysis(
        episodes, _synthetic_subject_metrics()
    )
    assert len(frame) == 5
    assert frame["support_records"].eq(1).all()
    assert frame["query_records"].eq(2).all()
    assert all(f"support_class_{index}" in frame for index in range(5))
    assert summary["n_subjects"] == 5
    assert summary["analysis"] == "descriptive_noncausal"
    assert summary["causal_interpretation"] is False


def test_paired_bootstrap_is_deterministic_and_subject_level() -> None:
    comparison = {
        "subjects": [
            {
                "delta_macro_f1": value,
                "delta_balanced_accuracy": value / 2,
                "delta_ordinal_mae": -value,
            }
            for value in (0.01, -0.02, 0.03, 0.00, 0.04)
        ]
    }
    first = paired_subject_bootstrap(comparison, resamples=1000, seed=42)
    second = paired_subject_bootstrap(comparison, resamples=1000, seed=42)
    assert first == second
    assert first["unit"] == "subject"
    assert first["n_subjects"] == 5
    assert first["statistical_significance_claimed"] is False


@pytest.mark.parametrize(
    ("macro", "balanced", "wins", "expected"),
    [
        (0.021, 0.011, 4, "strong_proceed"),
        (0.011, -0.004, 3, "proceed"),
        (0.005, 0.001, 2, "inconclusive"),
        (-0.001, -0.001, 4, "do_not_proceed"),
        (0.02, 0.01, 1, "do_not_proceed"),
    ],
)
def test_task8ch_decision_rule_is_deterministic(
    macro: float, balanced: float, wins: int, expected: str
) -> None:
    comparison = {
        "mean_delta_macro_f1": macro,
        "mean_delta_balanced_accuracy": balanced,
        "macro_f1_wins": wins,
    }
    result = apply_decision_rule(comparison, _config()["decision_rule"])
    assert result["status"] == expected


def test_runner_loads_existing_raw_cache_without_rebuilding() -> None:
    runner = FOMAMLLabelQ5RawDiagnostic(_config(), repository_root=ROOT)
    data, metadata = runner._load_data()
    assert len(metadata) == 30_958
    assert data.data.shape == (30_958, 1, 14, 2560)
    assert data.data.dtype == np.dtype(np.float32)
    assert data.metadata["dataset_mode"] == "raw_deduplicated_logical_records"
    assert data.metadata["accepted_windows_after_deduplication"] == 30_958
