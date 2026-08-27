from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.experiments.pm_eeg_lag_confirmatory import (
    PM_NAMES,
    aggregate_results,
    build_fold_audit,
    build_fold_transforms,
    build_previous_window_pairing,
    condition_target_ids,
    execute_run,
    load_config,
    prepare_protocol,
    validate_cache_contract,
    write_dry_run,
)
from bench.features.cogstate_feature_cache import sample_id_universe_hash


CONFIG_PATH = Path(
    "experiments/pm_diagnostics/pm_eeg_lag_confirmatory_v1.json"
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
        frame[f"target_{pm}"] = values + offset * 1e-3
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


def test_pairing_never_crosses_record_and_requires_exact_previous_window() -> None:
    frame = pd.DataFrame([
        {"sample_id": 1, "record_id": "a", "subject_id": "s1", "outer_fold": 1, "t_start": 5.0},
        {"sample_id": 2, "record_id": "a", "subject_id": "s1", "outer_fold": 1, "t_start": 15.0},
        # Missing 25 s: the 35 s row must not use 15 s as its previous window.
        {"sample_id": 3, "record_id": "a", "subject_id": "s1", "outer_fold": 1, "t_start": 35.0},
        {"sample_id": 4, "record_id": "b", "subject_id": "s1", "outer_fold": 1, "t_start": 5.0},
        {"sample_id": 5, "record_id": "b", "subject_id": "s1", "outer_fold": 1, "t_start": 15.0},
    ])
    pairing, summary = build_previous_window_pairing(frame)
    assert pairing["target_sample_id"].tolist() == [2, 5]
    assert pairing["lag_minus_10s_feature_sample_id"].tolist() == [1, 4]
    assert np.allclose(
        pairing["target_time"] - pairing["feature_time_lag_minus_10s"], 10.0
    )
    assert summary["cross_record_pairs"] == 0
    assert summary["additional_gap_losses"] == 1


def test_conditions_have_identical_target_subject_and_fold_identity() -> None:
    index = _index()
    pairing, _ = build_previous_window_pairing(index)
    conditions = condition_target_ids(pairing)
    assert np.array_equal(conditions["lag_0"], conditions["lag_minus_10s"])
    assert pairing["target_sample_id"].is_unique
    assert pairing.groupby("target_sample_id")["subject_id"].nunique().eq(1).all()
    assert pairing.groupby("target_sample_id")["outer_fold"].nunique().eq(1).all()


def test_fixed_folds_have_no_subject_leakage() -> None:
    index = _index()
    pairing, _ = build_previous_window_pairing(index)
    audit = build_fold_audit(index, pairing, [1, 2, 3, 4, 5])
    assert audit["subject_overlap_count"].eq(0).all()
    assert audit["matched_test_rows"].gt(0).all()


def test_q3_fit_receives_outer_train_only() -> None:
    index = _index()
    full = index.merge(_targets(index), on=["sample_id", "subject_id", "record_id"])
    # Values from fold 1 are made unmistakable and must not enter fold-1 fits.
    for pm in PM_NAMES:
        full.loc[full.outer_fold.eq(1), f"target_{pm}"] = 999.0
    seen: list[np.ndarray] = []

    class RecordingTransform:
        def __init__(self) -> None:
            from bench.tasks.target_transforms import FoldLocalQuantileTargetTransform

            self.inner = FoldLocalQuantileTargetTransform(3, duplicates="drop")

        def fit(self, values: np.ndarray):
            seen.append(np.asarray(values).copy())
            self.inner.fit(values)
            return self

        @property
        def actual_class_count(self) -> int:
            return self.inner.actual_class_count

        def manifest(self):
            return self.inner.manifest()

    transforms, manifests = build_fold_transforms(
        full,
        [1, 2, 3, 4, 5],
        transform_factory=RecordingTransform,
    )
    assert len(transforms) == len(manifests) == 35
    assert all(999.0 not in values for values in seen[:7])
    assert all(payload["fit_scope"] == "outer_train_only" for payload in manifests.values())


def test_config_contains_all_pm_and_no_focus_specific_lag() -> None:
    config = load_config(CONFIG_PATH)
    assert tuple(config["pm_names"]) == PM_NAMES
    assert [row["lag_seconds"] for row in config["conditions"]] == [0, -10]
    assert "-20" not in json.dumps(config)
    assert config["model"]["params"] == {
        "n_estimators": 200,
        "n_jobs": 4,
        "random_state": 42,
    }


def test_cache_contract_is_371_finite_and_target_free(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    cache, index = _fake_cache(tmp_path, config)
    matrix = np.load(cache / "features.npy", mmap_mode="r")
    names = json.loads((cache / "feature_names.json").read_text())["feature_names"]
    manifest = json.loads(
        (cache / "feature_materialization_manifest.json").read_text()
    )
    identity = validate_cache_contract(
        matrix, index, names, manifest, config["feature_cache_identity"]
    )
    assert identity["n_features"] == 371
    with pytest.raises(ValueError, match="Target/label columns"):
        validate_cache_contract(
            matrix,
            index.assign(target_focus=0.5),
            names,
            manifest,
            config["feature_cache_identity"],
        )


def test_dry_run_builds_70_runs_and_does_not_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _synthetic_context(tmp_path)
    monkeypatch.setattr(
        "bench.experiments.pm_eeg_lag_confirmatory.build_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training called")),
    )
    summary = write_dry_run(context)
    assert summary["training_executed"] is False
    assert summary["planned_fits"] == 70
    assert context.run_matrix.groupby(["outer_fold", "pm"]).size().eq(2).all()
    assert set(context.run_matrix["lag_seconds"]) == {0, -10}
    assert (context.output_dir / "protocol.json").is_file()
    assert (context.output_dir / "dry_run_summary.json").is_file()
    assert not (context.output_dir / "results_by_fold.csv").exists()


def test_one_synthetic_run_uses_shared_model_factory_contract(tmp_path: Path) -> None:
    context = _synthetic_context(tmp_path)

    class DummyClassifier:
        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            self.value = int(np.bincount(y).argmax())

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(len(X), self.value, dtype=int)

    calls: list[tuple] = []

    def builder(*args):
        calls.append(args)
        return DummyClassifier()

    spec = context.run_matrix.iloc[0].to_dict()
    result = execute_run(context, spec, model_builder=builder)
    assert calls[0][:4] == ("xgboost", "classification", (371,), 3)
    assert result["status"] == "complete"
    assert result["n_test_participants"] == 2
    assert (
        context.output_dir / "runs" / spec["run_id"] / "predictions.parquet"
    ).is_file()


def test_full_aggregation_writes_paired_pm_and_pooled_outputs(tmp_path: Path) -> None:
    context = _synthetic_context(tmp_path)
    summaries = []
    for spec in context.run_matrix.to_dict("records"):
        lagged = spec["condition"] == "lag_minus_10s"
        summaries.append({
            **spec,
            "status": "complete",
            "result_status": "confirmatory",
            "protocol_hash": context.protocol["protocol_hash"],
            "n_train": 32,
            "n_test": 8,
            "n_test_participants": 2,
            "training_time_seconds": 0.0,
            "participant_macro_f1": 0.4 + (0.05 if lagged else 0.0),
            "participant_macro_balanced_accuracy": 0.42 + (0.04 if lagged else 0.0),
            "participant_macro_accuracy": 0.44 + (0.03 if lagged else 0.0),
        })
    aggregate_results(context, summaries)
    paired = pd.read_csv(context.output_dir / "paired_delta_by_fold.csv")
    per_pm = pd.read_csv(context.output_dir / "summary_by_pm.csv")
    pooled = pd.read_csv(context.output_dir / "pooled_summary.csv")
    assert len(paired) == 35
    assert len(per_pm) == 7
    assert len(pooled) == 1
    assert np.allclose(paired["delta_participant_macro_f1"], 0.05)
    assert np.allclose(
        paired["delta_participant_macro_balanced_accuracy"], 0.04
    )
    assert json.loads((context.output_dir / "protocol.json").read_text())[
        "result_status"
    ] == "confirmatory_complete"
