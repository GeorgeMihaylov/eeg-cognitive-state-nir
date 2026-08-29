from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.pm_eeg_lag_confirmatory import build_previous_window_pairing
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    FIXED_LAG_SECONDS,
    PM_NAMES,
    aggregate_results,
    apply_extreme_labels,
    build_pm_temporal_cohorts,
    build_threshold_audit,
    execute_run,
    fit_extreme_thresholds,
    load_config,
    participant_binary_metrics,
    prepare_protocol,
    write_dry_run,
)
from bench.features.cogstate_feature_cache import sample_id_universe_hash


CONFIG_PATH = Path(
    "experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json"
)


def _index() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_id = 0
    for fold in range(1, 6):
        for subject_number in range(2):
            subject = f"s{fold}-{subject_number}"
            for window in range(5):
                rows.append({
                    "sample_id": sample_id,
                    "source": "synthetic",
                    "subject_id": subject,
                    "record_id": f"record-{subject}",
                    "record_group_id": f"group-{subject}",
                    "t_start": 5.0 + 10.0 * window,
                    "outer_fold": fold,
                })
                sample_id += 1
    return pd.DataFrame(rows)


def _targets(index: pd.DataFrame) -> pd.DataFrame:
    frame = index[["sample_id", "subject_id", "record_id"]].copy()
    values = (frame["sample_id"].to_numpy(dtype=float) % 10.0) / 9.0
    for offset, pm in enumerate(PM_NAMES):
        frame[f"target_{pm}"] = values + offset * 0.001
    frame.loc[frame["sample_id"].eq(1), "target_attention"] = np.nan
    return frame


def _fake_cache(tmp_path: Path, config: dict) -> tuple[Path, pd.DataFrame]:
    cache = tmp_path / "cache"
    cache.mkdir()
    index = _index()
    names = [f"feature_{column}" for column in range(371)]
    matrix = np.arange(len(index) * len(names), dtype=np.float32).reshape(
        len(index), len(names)
    )
    np.save(cache / "features.npy", matrix)
    index.to_parquet(cache / "feature_index.parquet", index=False)
    (cache / "feature_names.json").write_text(
        json.dumps({"feature_names": names}), encoding="utf-8"
    )
    identity = {
        "cache_schema_version": "cogstate-feature-cache-v1",
        "cache_identity_hash": "synthetic-cache",
        "feature_hash": "synthetic-features",
        "sample_id_universe_hash": sample_id_universe_hash(index["sample_id"]),
        "raw_preprocessing_hash": "synthetic-raw",
        "rows": len(index),
        "n_features": 371,
        "dtype": "float32",
        "target_columns_present": False,
    }
    (cache / "feature_materialization_manifest.json").write_text(
        json.dumps({
            "schema_version": "cogstate-feature-cache-v1",
            "status": "complete",
            "identity": identity,
        }),
        encoding="utf-8",
    )
    config["feature_cache_identity"] = identity
    return cache, index


def _synthetic_context(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    cache, index = _fake_cache(tmp_path, config)
    target_path = tmp_path / config["data"]["processed_targets"]
    target_path.parent.mkdir(parents=True)
    _targets(index).set_index("sample_id").to_parquet(target_path)
    return prepare_protocol(
        config,
        root=tmp_path,
        feature_cache_dir=cache,
        output_dir=tmp_path / "output",
    )


def test_train_only_q33_q67_and_test_reuses_thresholds() -> None:
    train = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    test = np.array([-1000.0, 1.8, 3.2, 1000.0])
    thresholds = fit_extreme_thresholds(train)
    expected = np.quantile(train, [1.0 / 3.0, 2.0 / 3.0])
    assert thresholds.q_low == pytest.approx(expected[0])
    assert thresholds.q_high == pytest.approx(expected[1])
    before = (thresholds.q_low, thresholds.q_high)
    labels = thresholds.transform(test)
    assert (thresholds.q_low, thresholds.q_high) == before
    assert labels[0] == 0
    assert np.isnan(labels[1]) and np.isnan(labels[2])
    assert labels[3] == 1


def test_middle_tertile_excluded_and_boundary_labels_are_inclusive() -> None:
    labels = apply_extreme_labels(
        [0.0, 1.0, 1.5, 2.0, 3.0], q_low=1.0, q_high=2.0
    )
    assert labels[0] == 0
    assert labels[1] == 0
    assert np.isnan(labels[2])
    assert labels[3] == 1
    assert labels[4] == 1


def test_config_forbids_q2_q5_lag0_and_focus_overrides(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    serialized = json.dumps(config).lower()
    assert "median" not in serialized
    assert "label_q5" not in serialized
    assert "lag_0" not in serialized
    assert config["alignment"]["lag_seconds"] == FIXED_LAG_SECONDS
    bad = deepcopy(config)
    bad["target_transform"]["name"] = "median"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="q33/q67"):
        load_config(path)


def test_exact_minus_10_pairing_rejects_gaps_and_cross_record_candidates() -> None:
    frame = pd.DataFrame([
        {"sample_id": 1, "record_id": "a", "subject_id": "s1", "outer_fold": 1, "t_start": 5.0},
        {"sample_id": 2, "record_id": "a", "subject_id": "s1", "outer_fold": 1, "t_start": 15.0},
        {"sample_id": 3, "record_id": "a", "subject_id": "s1", "outer_fold": 1, "t_start": 35.0},
        {"sample_id": 4, "record_id": "b", "subject_id": "s2", "outer_fold": 2, "t_start": 25.0},
        {"sample_id": 5, "record_id": "b", "subject_id": "s2", "outer_fold": 2, "t_start": 35.0},
    ])
    pairing, summary = build_previous_window_pairing(frame)
    assert pairing["target_sample_id"].tolist() == [2, 5]
    assert pairing["lag_minus_10s_feature_sample_id"].tolist() == [1, 4]
    assert np.allclose(
        pairing["target_time"] - pairing["feature_time_lag_minus_10s"], 10.0
    )
    assert summary["additional_gap_losses"] == 1
    assert summary["cross_record_pairs"] == 0
    assert summary["cross_subject_pairs"] == 0
    assert summary["cross_fold_pairs"] == 0


def test_threshold_audit_is_train_only_and_subject_disjoint() -> None:
    index = _index()
    pairing, _ = build_previous_window_pairing(index)
    full = index.merge(_targets(index), on=["sample_id", "subject_id", "record_id"])
    cohorts, _ = build_pm_temporal_cohorts(full, pairing)
    transforms, audit = build_threshold_audit(full, cohorts, [1, 2, 3, 4, 5])
    assert len(transforms) == 35
    assert len(audit) == 35
    assert audit["subject_overlap_count"].eq(0).all()
    assert (
        audit["n_train_low"]
        + audit["n_train_high"]
        + audit["n_train_excluded_middle"]
    ).equals(audit["n_train_before_exclusion"])
    assert (
        audit["n_test_low"]
        + audit["n_test_high"]
        + audit["n_test_excluded_middle"]
    ).equals(audit["n_test_before_exclusion"])
    row = audit.loc[(audit["outer_fold"] == 1) & (audit["pm"] == "engagement")].iloc[0]
    train = full.loc[full["outer_fold"].ne(1), "target_engagement"].to_numpy()
    expected = np.quantile(train[np.isfinite(train)], [1.0 / 3.0, 2.0 / 3.0])
    assert row["q_low"] == pytest.approx(expected[0])
    assert row["q_high"] == pytest.approx(expected[1])


def test_dry_run_has_seven_pm_five_folds_35_fits_and_never_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _synthetic_context(tmp_path)
    monkeypatch.setattr(
        "bench.experiments.pm_low_high_q3_extremes_confirmatory.build_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training called")),
    )
    summary = write_dry_run(context)
    assert summary["training_executed"] is False
    assert summary["planned_fits"] == 35
    assert context.matrix.shape == (50, 371)
    assert len(context.run_matrix) == 35
    assert context.run_matrix.groupby("pm").size().eq(5).all()
    assert context.run_matrix.groupby("outer_fold").size().eq(7).all()
    assert set(context.run_matrix["lag_seconds"]) == {-10}
    assert set(context.run_matrix["target_id"]) == {f"target_{pm}" for pm in PM_NAMES}
    assert set(context.run_matrix.loc[context.run_matrix["pm"].eq("focus"), "condition"]) == {"lag_minus_10s"}
    assert not any(name.startswith("target_") or "label" in name for name in context.feature_names)
    for name in (
        "protocol.json",
        "dry_run_summary.json",
        "cohort_summary.csv",
        "thresholds_by_fold.csv",
        "run_matrix.csv",
        "README.md",
    ):
        assert (context.output_dir / name).is_file()
    assert not (context.output_dir / "results_by_fold.csv").exists()


def test_participant_macro_auc_uses_probabilities_and_excludes_one_class() -> None:
    y_true = np.array([0, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0])
    probability_high = np.array([0.8, 0.2, 0.1, 0.2])
    subjects = np.array(["both", "both", "low-only", "low-only"])
    frame, macro = participant_binary_metrics(
        y_true, y_pred, probability_high, subjects
    )
    both = frame.set_index("subject_id").loc["both"]
    low_only = frame.set_index("subject_id").loc["low-only"]
    assert both["roc_auc"] == pytest.approx(0.0)
    assert np.isnan(low_only["roc_auc"])
    assert np.isnan(low_only["pr_auc"])
    assert macro["participant_macro_roc_auc"] == pytest.approx(0.0)
    assert macro["participant_valid_roc_auc"] == 1
    assert macro["participant_valid_pr_auc"] == 1
    assert macro["n_test_participants"] == 2


def test_execute_run_uses_canonical_binary_factory_and_probabilities(tmp_path: Path) -> None:
    context = _synthetic_context(tmp_path)

    class DummyClassifier:
        classes_ = np.array([0, 1])

        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            assert set(np.unique(y)) == {0, 1}

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.arange(len(X), dtype=int) % 2

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            high = np.linspace(0.1, 0.9, len(X))
            return np.column_stack([1.0 - high, high])

    calls: list[tuple] = []

    def builder(*args):
        calls.append(args)
        return DummyClassifier()

    spec = context.run_matrix.iloc[0].to_dict()
    result = execute_run(context, spec, model_builder=builder)
    assert calls[0][:4] == ("xgboost", "classification", (371,), 2)
    assert calls[0][4] == {"n_estimators": 200, "n_jobs": 4, "random_state": 42}
    assert result["status"] == "complete"
    assert result["lag_seconds"] == -10
    predictions = pd.read_parquet(
        context.output_dir / "runs" / spec["run_id"] / "predictions.parquet"
    )
    assert "probability_high" in predictions
    assert predictions["lag_seconds"].eq(-10).all()


def test_aggregation_requires_exactly_35_fixed_runs(tmp_path: Path) -> None:
    context = _synthetic_context(tmp_path)
    summaries = []
    for spec in context.run_matrix.to_dict("records"):
        summaries.append({
            **spec,
            "status": "complete",
            "result_status": "confirmatory",
            "protocol_hash": context.protocol["protocol_hash"],
            "training_time_seconds": 0.0,
            "n_test_participants": 2,
            **{f"participant_macro_{metric}": 0.75 for metric in (
                "balanced_accuracy", "f1", "roc_auc", "pr_auc",
                "low_recall", "high_recall", "precision", "accuracy",
            )},
            **{f"participant_valid_{metric}": 2 for metric in (
                "balanced_accuracy", "f1", "roc_auc", "pr_auc",
                "low_recall", "high_recall", "precision", "accuracy",
            )},
        })
    aggregate_results(context, summaries)
    assert len(pd.read_csv(context.output_dir / "results_by_fold.csv")) == 35
    assert len(pd.read_csv(context.output_dir / "summary_by_pm.csv")) == 7
    assert pd.read_csv(context.output_dir / "pooled_summary.csv").loc[
        0, "n_fold_pm_runs"
    ] == 35
