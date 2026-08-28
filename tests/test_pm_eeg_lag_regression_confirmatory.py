from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.pm_eeg_lag_confirmatory import build_previous_window_pairing
from bench.experiments.pm_eeg_lag_regression_confirmatory import (
    CONDITIONS,
    PM_NAMES,
    aggregate_results,
    build_pm_fold_audit,
    build_pm_matched_cohorts,
    execute_run,
    load_config,
    load_resumable_summary,
    participant_regression_metrics,
    prepare_protocol,
    write_dry_run,
)
from bench.features.cogstate_feature_cache import sample_id_universe_hash


CONFIG_PATH = Path(
    "experiments/pm_diagnostics/pm_eeg_lag_regression_confirmatory_v1.json"
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
    values = np.linspace(0.0, 1.0, len(frame), dtype=float)
    for offset, pm in enumerate(PM_NAMES):
        frame[f"target_{pm}"] = values + offset * 0.01
    # One target-complete temporal pair is removed only for Attention.
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


def test_exact_pairing_rejects_gap_and_never_crosses_record() -> None:
    frame = pd.DataFrame([
        {"sample_id": 1, "record_id": "a", "subject_id": "s1", "outer_fold": 1, "t_start": 5.0},
        {"sample_id": 2, "record_id": "a", "subject_id": "s1", "outer_fold": 1, "t_start": 15.0},
        {"sample_id": 3, "record_id": "a", "subject_id": "s1", "outer_fold": 1, "t_start": 35.0},
        {"sample_id": 4, "record_id": "b", "subject_id": "s2", "outer_fold": 2, "t_start": 5.0},
        {"sample_id": 5, "record_id": "b", "subject_id": "s2", "outer_fold": 2, "t_start": 15.0},
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


def test_config_is_seven_continuous_pm_only_with_fixed_lags() -> None:
    config = load_config(CONFIG_PATH)
    assert tuple(config["pm_names"]) == PM_NAMES
    assert config["target_ids"] == [f"target_{pm}" for pm in PM_NAMES]
    assert config["task"] == "regression"
    assert config["target_transform"] == {"name": "none", "reason": "continuous_pm"}
    assert [row["lag_seconds"] for row in config["conditions"]] == [0, -10]
    assert "q3" not in json.dumps(config).lower()
    assert "label_" not in json.dumps(config).lower()
    assert config["model"]["params"] == {
        "n_estimators": 200,
        "n_jobs": 4,
        "random_state": 42,
    }


def test_pm_complete_case_cohorts_are_identical_between_conditions() -> None:
    index = _index()
    pairing, _ = build_previous_window_pairing(index)
    full = index.merge(_targets(index), on=["sample_id", "subject_id", "record_id"])
    cohorts, summary = build_pm_matched_cohorts(full, pairing)
    assert len(cohorts["attention"]) == len(pairing) - 1
    assert len(cohorts["engagement"]) == len(pairing)
    for pm, cohort in cohorts.items():
        assert cohort["target_sample_id"].is_unique
        assert summary[pm]["identical_target_ids_between_conditions"] is True
        assert summary[pm]["identical_subject_ids_between_conditions"] is True
        assert summary[pm]["identical_fold_membership_between_conditions"] is True


def test_fold_audit_is_subject_disjoint_and_counts_are_condition_invariant() -> None:
    index = _index()
    pairing, _ = build_previous_window_pairing(index)
    full = index.merge(_targets(index), on=["sample_id", "subject_id", "record_id"])
    cohorts, _ = build_pm_matched_cohorts(full, pairing)
    audit = build_pm_fold_audit(full, cohorts, [1, 2, 3, 4, 5])
    assert len(audit) == 35
    assert audit["subject_overlap_count"].eq(0).all()
    assert audit["conditions_n_train_identical"].all()
    assert audit["conditions_n_test_identical"].all()


def test_dry_run_builds_70_runs_and_never_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _synthetic_context(tmp_path)
    monkeypatch.setattr(
        "bench.experiments.pm_eeg_lag_regression_confirmatory.build_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training called")),
    )
    summary = write_dry_run(context)
    assert summary["training_executed"] is False
    assert summary["planned_fits"] == 70
    assert context.matrix.shape == (50, 371)
    assert context.run_matrix.groupby(["outer_fold", "pm"]).size().eq(2).all()
    assert set(context.run_matrix["lag_seconds"]) == {0, -10}
    assert set(context.run_matrix["target_id"]) == {f"target_{pm}" for pm in PM_NAMES}
    assert (context.output_dir / "protocol.json").is_file()
    assert (context.output_dir / "matched_cohort_by_fold.csv").is_file()
    assert not (context.output_dir / "results_by_fold.csv").exists()


def test_participant_macro_regression_uses_equal_subject_weight() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 2.0, 0.0, 1.0, 2.0])
    subjects = np.array(["short", "short", "long", "long", "long"])
    frame, macro = participant_regression_metrics(y_true, y_pred, subjects)
    assert len(frame) == 2
    # Subject MAEs are 0.5 and 0.0; window-weighted MAE would be 0.2.
    assert macro["participant_macro_mae"] == pytest.approx(0.25)
    assert macro["participant_macro_r2"] == pytest.approx(0.0)
    assert macro["participant_macro_pearson"] == pytest.approx(1.0)


def test_synthetic_run_uses_shared_xgb_regression_factory_contract(tmp_path: Path) -> None:
    context = _synthetic_context(tmp_path)

    class DummyRegressor:
        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            self.value = float(np.mean(y))

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(len(X), self.value, dtype=float)

    calls: list[tuple] = []

    def builder(*args):
        calls.append(args)
        return DummyRegressor()

    spec = context.run_matrix.iloc[0].to_dict()
    result = execute_run(context, spec, model_builder=builder)
    assert calls[0][:4] == ("xgboost", "regression", (371,), 1)
    assert result["status"] == "complete"
    assert result["target_id"].startswith("target_")
    assert result["n_test_participants"] == 2
    assert (
        context.output_dir / "runs" / spec["run_id"] / "predictions.parquet"
    ).is_file()


def test_resume_requires_current_protocol_and_specification_hash(tmp_path: Path) -> None:
    context = _synthetic_context(tmp_path)
    spec = context.run_matrix.iloc[0].to_dict()
    run_dir = context.output_dir / "runs" / spec["run_id"]
    run_dir.mkdir(parents=True)
    pd.DataFrame({"y_true": [0.0], "y_pred": [0.0]}).to_parquet(
        run_dir / "predictions.parquet", index=False
    )
    payload = {
        "status": "complete",
        "protocol_hash": "stale",
        "specification_hash": spec["specification_hash"],
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_resumable_summary(context, spec) is None
    payload["protocol_hash"] = context.protocol["protocol_hash"]
    (run_dir / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_resumable_summary(context, spec) == payload


def test_aggregation_preserves_delta_mae_sign_convention(tmp_path: Path) -> None:
    context = _synthetic_context(tmp_path)
    summaries = []
    for spec in context.run_matrix.to_dict("records"):
        lagged = spec["condition"] == "lag_minus_10s"
        summaries.append({
            **spec,
            "status": "complete",
            "result_status": "confirmatory",
            "protocol_hash": context.protocol["protocol_hash"],
            "training_time_seconds": 0.0,
            "participant_macro_mae": 0.8 if lagged else 1.0,
            "participant_macro_r2": 0.3 if lagged else 0.2,
            "participant_macro_pearson": 0.5 if lagged else 0.4,
            "participant_macro_rmse": 0.9 if lagged else 1.1,
            "participant_macro_spearman": 0.45 if lagged else 0.35,
        })
    aggregate_results(context, summaries)
    paired = pd.read_csv(context.output_dir / "paired_delta_by_fold.csv")
    pooled = pd.read_csv(context.output_dir / "pooled_summary.csv")
    assert len(paired) == 35
    assert np.allclose(paired["delta_mae"], -0.2)
    assert np.allclose(paired["delta_r2"], 0.1)
    assert np.allclose(paired["delta_pearson"], 0.1)
    assert pooled.loc[0, "favorable_fold_pm_mae"] == 35
    assert pooled.loc[0, "favorable_pm_mean_mae"] == 7
    assert json.loads((context.output_dir / "protocol.json").read_text())[
        "result_status"
    ] == "confirmatory_complete"
