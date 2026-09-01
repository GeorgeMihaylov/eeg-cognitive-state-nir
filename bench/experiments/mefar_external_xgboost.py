"""MEFAR adapter for the shared external XGBoost execution contract."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from bench.experiments import mefar_multimodal as mefar
from bench.validation.metrics import MetricsCalculator
from cogstate.model_zoo.factory import build_model


SCHEMA_VERSION = "external-mefar-xgboost-v1"
EXPECTED_ARCHIVE_SHA256 = "c591ac136150032f58365248adbe52c68d063bc80a8846d22a32f29ad202048a"
EXPECTED_RF_PROTOCOL_HASH = "5a3339cab659e53f67b21da4b083191de9ce1e6c6a3eb7bb8bd593e852400ff3"
EXPECTED_FOLD_MANIFEST_HASH = "c8b9e80fbe9978eb1252e9fea60d172eee86ed0c91db4689e75e9f7116310232"
EXPECTED_SAMPLE_IDS_HASH = "c700c71a533686f949e544bc5a759821d5a22050ee99db8ab0f2f98fb13a9adf"
MODES = mefar.MODES
SEMANTIC_MODES = {
    "eeg_only": "eeg_only",
    "wearable_only": "peripheral_only",
    "eeg_wearable": "eeg_peripheral",
}
EXPECTED_XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if config.get("experiment_id") != "mefar_multimodal_xgboost_v1":
        raise ValueError("Unexpected MEFAR XGBoost experiment_id")
    if config.get("dataset", {}).get("name") != "mefar":
        raise ValueError("MEFAR XGBoost adapter requires dataset.name='mefar'")
    if config["dataset"].get("expected_archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("MEFAR archive identity changed")
    source = config.get("source_rf", {})
    expected_source_ids = {
        "protocol_hash": EXPECTED_RF_PROTOCOL_HASH,
        "fold_manifest_hash": EXPECTED_FOLD_MANIFEST_HASH,
        "sample_ids_hash": EXPECTED_SAMPLE_IDS_HASH,
    }
    if any(source.get(key) != value for key, value in expected_source_ids.items()):
        raise ValueError("MEFAR RF data/split contract identifiers changed")
    target = config.get("target", {})
    if target.get("target_id") != "mefar_cfs_fatigue_binary":
        raise ValueError("MEFAR target_id must remain mefar_cfs_fatigue_binary")
    if target.get("threshold") != {"operator": ">=", "value": 12}:
        raise ValueError("MEFAR CFS threshold must remain >= 12")
    if tuple(config.get("features", {}).get("modes", ())) != MODES:
        raise ValueError(f"MEFAR mode IDs must remain {MODES}")
    evaluation = config.get("evaluation", {})
    if evaluation.get("fold_source") != "existing_rf_fold_manifest":
        raise ValueError("MEFAR XGBoost must reuse the existing RF fold manifest")
    if evaluation.get("imputation") != "outer_train_median":
        raise ValueError("MEFAR XGBoost requires outer-train median imputation")
    if evaluation.get("scaling") != "none" or evaluation.get("oversampling") is not False:
        raise ValueError("MEFAR XGBoost forbids scaling and oversampling")
    model = config.get("model", {})
    if model.get("name") != "xgboost" or model.get("hyperparameter_search") is not False:
        raise ValueError("MEFAR model must be fixed XGBoost without search")
    if model.get("params") != EXPECTED_XGBOOST_PARAMS:
        raise ValueError("MEFAR must reuse the fixed external XGBoost parameters")
    return config


def _source_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(config["source_rf"]["output_dir"])
    source_config_path = Path(config["source_rf"]["config"])
    source_config = mefar.load_config(source_config_path)
    protocol = json.loads((source / "protocol_manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((source / "fold_manifest.json").read_text(encoding="utf-8"))
    sessions = pd.read_csv(source / "session_inventory.csv")
    if protocol.get("protocol_hash") != EXPECTED_RF_PROTOCOL_HASH:
        raise ValueError("Existing MEFAR RF protocol hash changed")
    if mefar.stable_hash(folds) != EXPECTED_FOLD_MANIFEST_HASH:
        raise ValueError("Existing MEFAR fold manifest semantic hash changed")
    usable = sessions.loc[sessions["usable_multimodal"].astype(bool)].copy()
    sample_ids = sorted(usable["record_id"].astype(str))
    if mefar.stable_hash(sample_ids) != EXPECTED_SAMPLE_IDS_HASH:
        raise ValueError("Existing MEFAR evaluation sample IDs changed")
    if len(usable) != 46 or usable["participant_id"].nunique() != 23:
        raise ValueError("Existing MEFAR cohort is no longer 23 participants / 46 sessions")
    if usable["target"].value_counts().sort_index().to_dict() != {0: 22, 1: 24}:
        raise ValueError("Existing MEFAR CFS class distribution changed")
    for fold in folds["folds"]:
        if set(fold["train_participants"]) & set(fold["test_participants"]):
            raise ValueError(f"Participant leakage in existing MEFAR fold {fold['fold']}")
    protected = [
        source / "summary.csv",
        source / "protocol_manifest.json",
        source / "fold_manifest.json",
        *sorted(source.glob("*/metrics.json")),
    ]
    return {
        "source": source,
        "source_config": source_config,
        "protocol": protocol,
        "folds": folds,
        "sessions": usable,
        "sample_ids": sample_ids,
        "protected_hashes": {
            path.relative_to(source).as_posix(): file_sha256(path) for path in protected
        },
    }


def _run_matrix(config: Mapping[str, Any], folds: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds["folds"]:
        for mode in MODES:
            payload = {
                "experiment_id": config["experiment_id"],
                "fold": int(fold["fold"]),
                "model": "xgboost",
                "mode": mode,
                "semantic_mode": SEMANTIC_MODES[mode],
                "target_id": "mefar_cfs_fatigue_binary",
                "feature_names": mefar.feature_names(mode),
                "test_sample_ids": fold["test_sample_ids"],
                "params": config["model"]["params"],
            }
            rows.append({
                "run_id": f"xgboost__{mode}__fold{int(fold['fold']):02d}__{stable_hash(payload)[:10]}",
                "fold": int(fold["fold"]),
                "model": "xgboost",
                "mode": mode,
                "semantic_mode": SEMANTIC_MODES[mode],
                "feature_count": len(mefar.feature_names(mode)),
                "n_train_samples": len(fold["train_sample_ids"]),
                "n_test_samples": len(fold["test_sample_ids"]),
                "evaluation_sample_ids_hash": mefar.stable_hash(sorted(fold["test_sample_ids"])),
                "specification_hash": stable_hash(payload),
            })
    return pd.DataFrame(rows)


def fit_outer_train_median(
    train_values: Any,
    test_values: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit imputation on outer-train and apply it unchanged to outer-test."""
    imputer = SimpleImputer(strategy="median")
    train = imputer.fit_transform(train_values)
    test = imputer.transform(test_values)
    return train, test, np.asarray(imputer.statistics_, dtype=float)


def build_plan(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    source = _source_contract(config)
    matrix = _run_matrix(config, source["folds"])
    modes = {
        mode: {
            "semantic_mode": SEMANTIC_MODES[mode],
            "feature_count": len(mefar.feature_names(mode)),
            "sample_ids_hash": EXPECTED_SAMPLE_IDS_HASH,
        }
        for mode in MODES
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": config["result_status"],
        "dataset": "mefar",
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "source_rf_experiment_id": source["protocol"]["experiment_id"],
        "source_rf_protocol_hash": EXPECTED_RF_PROTOCOL_HASH,
        "target": config["target"],
        "participants": 23,
        "sessions": 46,
        "class_distribution": {"0": 22, "1": 24},
        "synchronization_level": "participant_session_summary",
        "modes": modes,
        "model": config["model"],
        "model_contract_difference": {
            "reason": "binary rather than three-class target",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "all_other_xgboost_parameters_match_clare_cl_drive": True,
        },
        "fold_source": "immutable existing RF fold manifest",
        "fold_manifest_hash": EXPECTED_FOLD_MANIFEST_HASH,
        "sample_ids_hash": EXPECTED_SAMPLE_IDS_HASH,
        "run_matrix_hash": stable_hash(matrix.to_dict(orient="records")),
        "run_count": int(len(matrix)),
        "shallow_supported": False,
        "shallow_unsupported_reason": "MEFAR contains derived NeuroSky band powers, not suitable raw multichannel EEG",
        "leakage_guards": {
            "participant_disjoint_outer_folds": True,
            "same_evaluation_sample_ids": True,
            "train_only_median_imputation": True,
            "global_scaler": False,
            "oversampling": False,
            "target_columns_excluded": True,
            "attention_meditation_derived_excluded": True,
        },
        "output_dir": config["output_dir"],
    }
    protocol["protocol_hash"] = stable_hash(protocol)
    summary = {
        **protocol,
        "folds": [{
            "fold": fold["fold"],
            "train_participants": len(fold["train_participants"]),
            "test_participants": len(fold["test_participants"]),
            "train_class_counts": fold["train_class_counts"],
            "test_class_counts": fold["test_class_counts"],
            "participant_overlap": fold["participant_overlap"],
        } for fold in source["folds"]["folds"]],
        "training_units": 15,
        "models_trained": 0,
        "writes_performed": False,
        "expected_artifacts": [
            "protocol_manifest.json", "fold_manifest.json", "run_matrix.csv",
            "plan_summary.json", "summary_xgboost.csv", "runs/<run_id>/metrics.json",
            "runs/<run_id>/predictions.parquet", "runs/<run_id>/normalization_stats.json",
        ],
    }
    return {"config": config, "source": source, "matrix": matrix, "protocol": protocol, "summary": summary}


def write_plan_artifacts(config_path: str | Path) -> dict[str, Any]:
    plan = build_plan(config_path)
    output = Path(plan["config"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan["source"]["source"] / "fold_manifest.json", output / "fold_manifest.json")
    plan["matrix"].to_csv(output / "run_matrix.csv", index=False, lineterminator="\n")
    _write_json(output / "protocol_manifest.json", plan["protocol"])
    summary = dict(plan["summary"])
    summary["writes_performed"] = True
    _write_json(output / "plan_summary.json", summary)
    _write_json(output / "source_rf_integrity.json", {
        "source_namespace": plan["source"]["source"].as_posix(),
        "protected_files": plan["source"]["protected_hashes"],
    })
    return summary


def plan_experiment(config_path: str | Path) -> dict[str, Any]:
    plan = build_plan(config_path)
    output = Path(plan["config"]["output_dir"])
    required = ["protocol_manifest.json", "fold_manifest.json", "run_matrix.csv", "plan_summary.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run --inventory before --plan-only; missing {missing}")
    if (output / "fold_manifest.json").read_bytes() != (
        plan["source"]["source"] / "fold_manifest.json"
    ).read_bytes():
        raise ValueError("Copied MEFAR fold manifest differs from the RF source")
    stored = json.loads((output / "protocol_manifest.json").read_text(encoding="utf-8"))
    if stored != plan["protocol"]:
        raise ValueError("Stored MEFAR XGBoost protocol does not match the resolved plan")
    return plan["summary"]


def run_xgboost(config_path: str | Path) -> dict[str, Any]:
    """Execute only after explicit user approval; plan-only never calls this."""
    plan = build_plan(config_path)
    config = plan["config"]
    source = plan["source"]
    inventory = mefar.build_inventory(source["source_config"])
    features = mefar.materialize_session_features(source["source_config"], inventory["sessions"])
    output = Path(config["output_dir"])
    summary_rows: list[dict[str, Any]] = []
    for run in plan["matrix"].to_dict(orient="records"):
        fold = source["folds"]["folds"][int(run["fold"]) - 1]
        frame = features[str(run["mode"])]
        train = frame["sample_id"].isin(fold["train_sample_ids"])
        test = frame["sample_id"].isin(fold["test_sample_ids"])
        columns = mefar.feature_names(str(run["mode"]))
        x_train, x_test, imputation_statistics = fit_outer_train_median(
            frame.loc[train, columns], frame.loc[test, columns]
        )
        y_train = frame.loc[train, "target"].to_numpy(dtype=int)
        y_test = frame.loc[test, "target"].to_numpy(dtype=int)
        model = build_model("xgboost", "classification", (len(columns),), 2, config["model"]["params"])
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_test), dtype=int)
        probability = np.asarray(model.predict_proba(x_test), dtype=float)
        metrics = MetricsCalculator.calculate_all_metrics(
            y_test, prediction, y_proba=probability, labels=np.asarray([0, 1])
        )
        metrics["class_metrics"] = MetricsCalculator.calculate_class_metrics(
            y_test, prediction, labels=np.asarray([0, 1])
        )
        run_dir = output / "runs" / str(run["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions = frame.loc[test, ["sample_id", "participant_id", "session_id", "target"]].copy()
        predictions = predictions.rename(columns={"target": "y_true"})
        predictions["y_pred"] = prediction
        predictions["proba_0"] = probability[:, 0]
        predictions["proba_1"] = probability[:, 1]
        predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "normalization_stats.json", {
            "imputation": "outer_train_median",
            "scaling": "none",
            "statistics_fit_sample_ids": sorted(frame.loc[train, "sample_id"].astype(str)),
            "median": imputation_statistics.tolist(),
        })
        summary_rows.append({
            **run,
            **{key: value for key, value in metrics.items() if np.isscalar(value)},
        })
    pd.DataFrame(summary_rows).to_csv(output / "summary_xgboost.csv", index=False, lineterminator="\n")
    return {
        "experiment_id": config["experiment_id"],
        "models_trained": len(summary_rows),
        "summary_path": (output / "summary_xgboost.csv").as_posix(),
    }


__all__ = [
    "EXPECTED_FOLD_MANIFEST_HASH", "EXPECTED_SAMPLE_IDS_HASH", "MODES",
    "SCHEMA_VERSION", "build_plan", "file_sha256", "fit_outer_train_median", "load_config",
    "plan_experiment", "run_xgboost", "stable_hash", "write_plan_artifacts",
]
