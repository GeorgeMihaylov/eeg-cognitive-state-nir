from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from bench.experiments.ordinal_transformer import build_ordinal_transformer_experiment
from bench.experiments.ordinal_transformer_multiseed import (
    CategoricalCandidateAudit,
    OrdinalTransformerMultiseedExperiment,
)


SPEC = Path("experiments/ordinal_transformer_multiseed.yaml")


def _context() -> dict:
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    canonical = pd.DataFrame({
        "sequence_id": [f"s-{index}" for index in range(10)],
        "fold": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "subject_id": [f"u-{index}" for index in range(10)],
        "record_id": [f"r-{index}" for index in range(10)],
        "record_group_id": [f"g-{index}" for index in range(10)],
        "source": ["synthetic"] * 10,
        "target_sample_id": list(range(10)),
        "target_time": list(range(10)),
        "y_true": [index % 5 for index in range(10)],
    })
    return {
        "supervised_rows": document["dataset"]["expected_supervised_rows"],
        "canonical": canonical,
        "sequence_count": document["dataset"]["expected_sequences"],
        "subject_count": document["dataset"]["expected_subjects"],
        "sequence_index_sha256": document["dataset"]["sequence_index_sha256"],
        "source_parquet_sha256": document["dataset"]["parquet_sha256"],
        "sequence_build_stats": {},
        "features": {
            group: {
                "count": definition["feature_count"],
                "sha256": definition["feature_list_sha256"],
                "names": [f"{group}-{index}" for index in range(definition["feature_count"])],
            }
            for group, definition in document["feature_definitions"].items()
        },
        "fold_summaries": {
            f"fold_{fold:02d}": {
                "outer_subject_overlap": [],
                "inner_group_overlap": [],
                "inner_validation_group_ids": [f"group-{fold}"],
            }
            for fold in range(1, 6)
        },
    }


def _experiment(tmp_path: Path) -> OrdinalTransformerMultiseedExperiment:
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    document["experiment"]["output_dir"] = str(tmp_path / "output")
    spec = tmp_path / "multiseed.yaml"
    spec.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    experiment = OrdinalTransformerMultiseedExperiment(
        spec,
        context_builder=_context,
        completed_run_finder=lambda *args, **kwargs: None,
    )
    candidates = tuple(
        CategoricalCandidateAudit(
            group, 42, tmp_path / f"categorical-{group}", True, ()
        )
        for group in ("eeg_only", "eeg_pow")
    )
    experiment._discover_categorical_candidates = lambda: (
        list(candidates),
        {(candidate.feature_group, 42): candidate.run_directory for candidate in candidates},
    )
    return experiment


def test_multiseed_plan_has_eight_ordinal_and_four_missing_baselines(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    plan = experiment.plan()
    ordinal = [trial for trial in plan.trials if trial.head_type != "categorical"]
    categorical = [trial for trial in plan.trials if trial.head_type == "categorical"]
    assert len(ordinal) == 8
    assert len(categorical) == 4
    assert {trial.seed for trial in ordinal} == {7, 123}
    assert all(trial.folds == (1, 2, 3, 4, 5) for trial in plan.trials)
    assert plan.fold_runs == 60
    assert not (tmp_path / "output").exists()


def test_split_seeds_do_not_follow_model_seed(tmp_path: Path) -> None:
    plan = _experiment(tmp_path).plan()
    for trial in plan.trials:
        params = trial.resolved_config["models"]["torch_transformer"]["params"]
        assert params["random_state"] == trial.seed
        assert trial.resolved_config["validation"]["random_state"] == 42
        assert trial.resolved_config["evaluation"]["random_state"] == 42
        assert trial.resolved_config["task_config"]["random_state"] == 42


def test_builder_selects_multiseed_experiment() -> None:
    assert isinstance(
        build_ordinal_transformer_experiment(SPEC),
        OrdinalTransformerMultiseedExperiment,
    )


def test_mismatched_categorical_candidate_is_rejected(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    run = tmp_path / "candidate"
    run.mkdir()
    audit = experiment._audit_categorical_candidate(run, "eeg_pow", 7)
    assert not audit.eligible
    assert any("unreadable candidate" in reason for reason in audit.reasons)
