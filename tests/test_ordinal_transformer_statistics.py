from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

import cli
from bench.analysis.alignment import AlignmentError
from bench.analysis.ordinal_transformer_statistics import (
    FEATURE_GROUPS,
    METHODS,
    OrdinalTransformerStatistics,
    build_hypothesis_tables,
    build_subject_effect_types,
    calculate_subject_metrics,
    categorical_expected_rank,
    class_error_analysis,
    discover_canonical_runs,
    error_distance_analysis,
    metric_improvement,
    paired_metric_comparison,
    require_six_way_alignment,
    select_decision,
)
from cogstate.model_zoo.DL.sequence_utils import sequence_index_sha256


def _predictions(subjects: int = 53) -> dict[str, pd.DataFrame]:
    base_rows: list[dict[str, object]] = []
    for subject in range(subjects):
        for class_id in range(5):
            base_rows.append({
                "sequence_id": f"sequence-{subject:02d}-{class_id}",
                "fold": subject % 5 + 1,
                "subject_id": f"subject-{subject:02d}",
                "record_id": f"record-{subject:02d}",
                "source": "Old_EEG" if subject % 2 else "gpn_data",
                "target_sample_id": subject * 5 + class_id,
                "target_time": float(class_id * 10),
                "y_true": class_id,
            })
    base = pd.DataFrame(base_rows)
    output: dict[str, pd.DataFrame] = {}
    for feature_group in FEATURE_GROUPS:
        for method in METHODS:
            frame = base.copy()
            if method == "categorical":
                y_pred = np.where(frame.index % 7 == 0, (frame.y_true + 2) % 5, frame.y_true)
            elif method == "coral":
                y_pred = np.where(frame.index % 9 == 0, (frame.y_true + 1) % 5, frame.y_true)
            else:
                y_pred = np.where(frame.index % 11 == 0, (frame.y_true + 1) % 5, frame.y_true)
            frame["y_pred"] = y_pred.astype(int)
            probabilities = np.full((len(frame), 5), 0.025, dtype=float)
            probabilities[np.arange(len(frame)), frame["y_pred"].to_numpy()] = 0.9
            for index in range(5):
                frame[f"proba_{index}"] = probabilities[:, index]
            if method != "categorical":
                for index in range(5):
                    frame[f"class_probability_{index}"] = probabilities[:, index]
                frame["expected_rank"] = probabilities @ np.arange(5)
                frame["ordinal_argmax"] = np.argmax(probabilities, axis=1)
            output[f"{method}_{feature_group}"] = frame
    return output


def _write_fake_run(
    root: Path,
    *,
    method: str,
    feature_group: str,
    frame: pd.DataFrame,
    seed: int = 42,
    smoke: bool = False,
) -> None:
    trial = f"{method}_{feature_group}"
    run = root / trial / "20260101_000000"
    protocol = run / "dataset" / "task" / "model" / "group_kfold_subject"
    protocol.mkdir(parents=True)
    frame.to_parquet(protocol / "predictions.parquet", index=False)
    if method == "categorical":
        experiment = {
            "name": "feature_group_transformer_ablation",
            "trial_id": f"transformer_classification_{feature_group}",
            "task": "classification",
            "feature_group": feature_group,
            "seed": seed,
        }
    else:
        experiment = {
            "type": "ordinal_transformer_smoke" if smoke else "ordinal_transformer_full",
            "trial_id": trial,
            "head_type": method,
            "feature_group": feature_group,
            "seed": seed,
            "full_sequence_index_sha256": sequence_index_sha256(frame),
        }
    config = {
        "experiment": experiment,
        "sequence": {"length": 8},
        "evaluation": {"n_splits": 5, "folds": [1, 2, 3, 4, 5]},
        "models": {"torch_transformer": {"params": {"random_state": seed}}},
    }
    (run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run / "run_manifest.json").write_text(json.dumps({
        "status": "completed", "timestamp": "20260101_000000", "config_hash": trial
    }), encoding="utf-8")


def test_run_discovery_loads_exactly_six_and_excludes_smoke_and_wrong_seed(
    tmp_path: Path,
) -> None:
    predictions = _predictions(subjects=53)
    root = tmp_path / "runs"
    for feature_group in FEATURE_GROUPS:
        for method in METHODS:
            _write_fake_run(
                root, method=method, feature_group=feature_group,
                frame=predictions[f"{method}_{feature_group}"],
            )
    _write_fake_run(
        tmp_path / "smoke", method="coral", feature_group="eeg_only",
        frame=predictions["coral_eeg_only"], smoke=True,
    )
    _write_fake_run(
        tmp_path / "wrong-seed", method="corn", feature_group="eeg_pow",
        frame=predictions["corn_eeg_pow"], seed=7,
    )
    frame = predictions["categorical_eeg_only"]
    document = {
        "analysis": {"run_roots": [root, tmp_path / "smoke", tmp_path / "wrong-seed"]},
        "expected": {
            "seed": 42, "sequence_length": 8, "sequences": len(frame),
            "subjects": 53, "folds": 5,
            "sequence_index_sha256": sequence_index_sha256(frame),
        },
    }
    resolved = discover_canonical_runs(document)
    assert set(resolved) == {
        f"{method}_{group}" for group in FEATURE_GROUPS for method in METHODS
    }
    assert all(run.seed == 42 for run in resolved.values())


def test_sequence_hash_and_exact_six_way_alignment_detect_mismatch() -> None:
    predictions = _predictions()
    audit = require_six_way_alignment(predictions)
    assert audit["exact_match"] is True
    assert audit["rows"] == 265 and audit["subjects"] == 53
    changed = {key: value.copy() for key, value in predictions.items()}
    changed["corn_eeg_pow"].loc[0, "record_id"] = "mismatch"
    with pytest.raises(AlignmentError, match="record_id"):
        require_six_way_alignment(changed)


def test_categorical_expected_rank_uses_class_probabilities() -> None:
    frame = _predictions()["categorical_eeg_only"]
    expected = categorical_expected_rank(frame)
    manual = sum(frame[f"proba_{index}"].to_numpy() * index for index in range(5))
    np.testing.assert_allclose(expected, manual)


def test_subject_metrics_are_deterministic_and_keep_undefined_reasons() -> None:
    predictions = _predictions()
    first = calculate_subject_metrics(predictions)
    second = calculate_subject_metrics(predictions)
    pdt.assert_frame_equal(first, second)
    assert len(first) == 318
    assert first["subject_id"].nunique() == 53
    assert set(first["method"]) == set(METHODS)
    assert "undefined_metric_reason" in first


def test_metric_directions_and_severe_error_definition() -> None:
    raw, improvement = metric_improvement([0.30, 0.40], [0.20, 0.50], "balanced_accuracy")
    np.testing.assert_allclose(raw, [0.10, -0.10])
    np.testing.assert_allclose(improvement, raw)
    raw, improvement = metric_improvement([0.8, 1.2], [1.0, 1.0], "ordinal_mae")
    np.testing.assert_allclose(raw, [-0.2, 0.2])
    np.testing.assert_allclose(improvement, [0.2, -0.2])
    frame = pd.DataFrame({
        "y_true": [0, 0, 4], "y_pred": [1, 2, 1],
        **{f"proba_{index}": [0.2, 0.2, 0.2] for index in range(5)},
    })
    distances = np.abs(frame.y_pred - frame.y_true)
    assert (distances >= 2).tolist() == [False, True, True]


def test_paired_bootstrap_is_subject_level_deterministic_and_ties_are_explicit() -> None:
    metrics = calculate_subject_metrics(_predictions())
    first = paired_metric_comparison(
        metrics, candidate_key="corn_eeg_only", reference_key="categorical_eeg_only",
        metric="ordinal_mae", family="test", hypothesis_tier="primary",
        n_resamples=250, random_state=42,
    )
    second = paired_metric_comparison(
        metrics, candidate_key="corn_eeg_only", reference_key="categorical_eeg_only",
        metric="ordinal_mae", family="test", hypothesis_tier="primary",
        n_resamples=250, random_state=42,
    )
    assert first == second
    assert first["n_valid_pairs"] == 53
    assert first["bootstrap_seed"] == 42
    tied = metrics.copy()
    source = tied.loc[tied.run_key == "categorical_eeg_only", "ordinal_mae"].to_numpy()
    tied.loc[tied.run_key == "corn_eeg_only", "ordinal_mae"] = source
    result = paired_metric_comparison(
        tied, candidate_key="corn_eeg_only", reference_key="categorical_eeg_only",
        metric="ordinal_mae", family="test", hypothesis_tier="primary",
        n_resamples=25,
    )
    assert result["ties"] == 53
    assert result["wilcoxon_status"] == "undefined_all_differences_zero"
    assert result["sign_test_status"] == "undefined_all_differences_zero"


def test_holm_families_are_separate_and_fold_values_are_not_inputs() -> None:
    subject_metrics = calculate_subject_metrics(_predictions())
    primary, secondary, feature = build_hypothesis_tables(
        subject_metrics, n_resamples=50, random_state=42
    )
    assert len(primary) == 8 and len(secondary) == 36 and len(feature) == 15
    assert set(row["family"] for row in primary) == {
        "primary_eeg_only", "primary_eeg_pow"
    }
    assert set(row["family"] for row in secondary) == {
        "secondary_eeg_only", "secondary_eeg_pow"
    }
    assert {row["family"] for row in feature} == {"feature_group_effect"}
    assert all(row["n_valid_pairs"] == 53 for row in primary + secondary + feature)
    assert all("holm_adjusted_p_value" in row for row in primary + secondary + feature)


def test_subject_quartiles_use_categorical_reference_and_effect_types_are_complete() -> None:
    metrics = calculate_subject_metrics(_predictions())
    effects = build_subject_effect_types(metrics)
    assert len(effects) == 4 * 53
    assert set(effects["difficulty_group"]) == {
        "best_quartile", "middle_half", "worst_quartile"
    }
    assert (effects.groupby(["feature_group", "candidate"]).size() == 53).all()
    assert effects["baseline_ordinal_mae"].notna().all()


def test_error_distance_transitions_and_class_analysis_are_consistent() -> None:
    predictions = _predictions()
    distributions, transitions = error_distance_analysis(predictions)
    assert len(transitions) == 4 * 25
    for keys, group in distributions.loc[
        distributions.row_type == "distance_distribution"
    ].groupby(["feature_group", "method"]):
        assert group["count"].sum() == 265
        assert group["fraction"].sum() == pytest.approx(1.0)
    classes = class_error_analysis(predictions)
    assert len(classes) == 6 * 5
    assert set(classes.true_class) == set(range(5))


def _comparison_row(
    method: str, group: str, metric: str, *, improvement: float, confirmed: bool
) -> dict[str, object]:
    return {
        "candidate": f"{method}_{group}",
        "reference": f"categorical_{group}",
        "feature_group": group,
        "metric": metric,
        "reference_mean": 1.0,
        "candidate_mean": 1.0 - improvement,
        "mean_improvement": improvement,
        "fraction_degraded": 0.40,
        "bootstrap_ci_low": 0.01 if confirmed else -0.01,
        "bootstrap_ci_high": 0.03 if improvement >= 0 else -0.01,
        "holm_adjusted_p_value": 0.01 if confirmed else 0.20,
    }


def test_decision_rule_is_deterministic_and_selects_exactly_one_option() -> None:
    primary = []
    secondary = []
    hard = []
    for method in ("coral", "corn"):
        for group in FEATURE_GROUPS:
            for metric in ("ordinal_mae", "severe_error_rate"):
                primary.append(_comparison_row(
                    method, group, metric,
                    improvement=0.02 if method == "corn" else -0.01,
                    confirmed=method == "corn",
                ))
            for metric in ("balanced_accuracy", "macro_f1"):
                secondary.append(_comparison_row(
                    method, group, metric, improvement=-0.005, confirmed=False
                ))
            hard.append({
                "candidate": method, "feature_group": group,
                "difficulty_group": "worst_quartile",
                "mean_ordinal_mae_improvement": 0.01 if method == "corn" else -0.01,
            })
    first = select_decision(primary, secondary, hard)
    second = select_decision(primary, secondary, hard)
    assert first == second
    assert first["selected_decision_id"] == 1
    assert first["selected_ordinal_method"] == "corn"


def test_plan_only_does_not_create_output_or_reports(tmp_path: Path) -> None:
    analysis = OrdinalTransformerStatistics(
        "experiments/ordinal_transformer_statistics.yaml",
        output_dir=tmp_path / "generated",
    )
    plan = analysis.plan()
    rendered = analysis.render_plan(plan)
    assert len(plan["resolved_runs"]) == 6
    assert plan["seed"] == 42 and plan["subjects"] == 53
    assert "Plan-only" in rendered
    assert not (tmp_path / "generated").exists()


def test_cli_plan_only_uses_analysis_without_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeAnalysis:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("init")

        def plan(self) -> dict[str, bool]:
            calls.append("plan")
            return {"valid": True}

        @staticmethod
        def render_plan(plan: object) -> str:
            return "valid plan"

        def execute(self) -> None:
            calls.append("execute")

    monkeypatch.setattr(
        "bench.analysis.ordinal_transformer_statistics.OrdinalTransformerStatistics",
        FakeAnalysis,
    )
    cli.main([
        "--ordinal-transformer-analysis", "synthetic.yaml", "--plan-only"
    ])
    assert calls == ["init", "plan"]


def test_analysis_source_contains_no_training_or_model_construction() -> None:
    source = inspect.getsource(OrdinalTransformerStatistics.execute)
    assert ".fit(" not in source
    assert "BenchmarkRunner" not in source
    assert "build_model" not in source
