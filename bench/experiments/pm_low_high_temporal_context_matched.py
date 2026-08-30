"""Matched causal temporal-context comparison for PM LOW/HIGH extremes.

All four models are evaluated on the exact same target-sample cohort: LOW/HIGH
targets whose EEG(t-10s) feature endpoint has ten contiguous 10-second feature
windows available inside one logical recording.

Inputs differ only by preregistered information set:
- LightGBM / XGBoost: 371 features at EEG(t-10s);
- LSTM: 10 windows EEG(t-100s)..EEG(t-10s);
- Transformer: last 8 windows EEG(t-80s)..EEG(t-10s).

The 35 LSTM fits are reused from neural robustness only after exact train/test
sample hashes, endpoint hashes, threshold hashes, configuration and metrics
artifacts are verified. 105 new fits are required.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.experiments.pm_low_high_neural_robustness import (
    exact_history_endpoint_ids,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    FIXED_LAG_SECONDS,
    PM_NAMES,
    ProtocolContext,
    load_config as load_reference_config,
    participant_binary_metrics,
    prepare_protocol as prepare_reference_protocol,
    stable_hash,
)
from cogstate.model_zoo import build_model
from cogstate.model_zoo.DL.sequence_utils import build_sequences


SCHEMA_VERSION = "pm-low-high-temporal-context-matched-v1"
MODEL_ORDER = ("lightgbm", "xgboost", "torch_lstm", "torch_transformer")
NEW_MODEL_ORDER = ("lightgbm", "xgboost", "torch_transformer")
TEMPORAL_TABULAR_PAIRS = (
    ("torch_lstm", "lightgbm"),
    ("torch_lstm", "xgboost"),
    ("torch_transformer", "lightgbm"),
    ("torch_transformer", "xgboost"),
)
METRICS = (
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "low_recall",
    "high_recall",
    "precision",
    "accuracy",
)
BOOTSTRAP_METRICS = ("balanced_accuracy", "f1", "roc_auc")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
PRACTICAL_EQUIVALENCE_MARGIN = 0.01

EXPECTED_MODELS = {
    "lightgbm": {
        "input_family": "single_window_features",
        "context_windows": 1,
        "input_rule": "371 features at EEG(t-10s)",
        "params": {"n_estimators": 200, "n_jobs": 4, "random_state": 42},
    },
    "xgboost": {
        "input_family": "single_window_features",
        "context_windows": 1,
        "input_rule": "371 features at EEG(t-10s)",
        "params": {"n_estimators": 200, "n_jobs": 4, "random_state": 42},
    },
    "torch_lstm": {
        "input_family": "sequence_features",
        "context_windows": 10,
        "input_rule": "371-feature windows EEG(t-100s)..EEG(t-10s)",
        "execution": "reuse_exact_neural_robustness_reference",
        "sequence": {
            "length": 10,
            "stride": 1,
            "target_position": "last",
            "expected_step_seconds": 10.0,
            "max_gap_seconds": 10.01,
        },
        "params": {
            "hidden_size": 128,
            "num_layers": 1,
            "bidirectional": False,
            "dropout": 0.2,
            "classifier_hidden": 64,
            "batch_size": 256,
            "max_epochs": 5,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "validation_size": 0.15,
            "early_stopping_patience": 2,
            "device": "auto",
            "random_state": 42,
            "standardize": True,
            "num_workers": 0,
        },
    },
    "torch_transformer": {
        "input_family": "sequence_features",
        "context_windows": 8,
        "input_rule": "last 8 of common history: EEG(t-80s)..EEG(t-10s)",
        "execution": "new_training_on_common_10_window_eligible_cohort",
        "sequence": {
            "length": 8,
            "stride": 1,
            "target_position": "last",
            "expected_step_seconds": 10.0,
            "max_gap_seconds": 10.01,
        },
        "params": {
            "sequence_length": 8,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "activation": "gelu",
            "pooling": "last",
            "positional_encoding": "learned",
            "batch_size": 128,
            "max_epochs": 15,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "validation_size": 0.15,
            "early_stopping_patience": 4,
            "device": "auto",
            "random_state": 42,
            "standardize": True,
            "num_workers": 0,
        },
    },
}


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _sample_hash(values: Sequence[Any]) -> str:
    return stable_hash([str(value) for value in values])


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    refs = config.get("references", {})
    if refs.get("low_high", {}).get("protocol_hash") != (
        "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
    ):
        raise ValueError("LOW/HIGH reference protocol changed")
    if refs.get("neural_robustness", {}).get("protocol_hash") != (
        "e902f4dbe8f317be4ac6ed5061104cf5a3399eea5125414a675add88f4105a8d"
    ):
        raise ValueError("Neural robustness reference protocol changed")
    contract = config.get("scientific_contract", {})
    if tuple(contract.get("pm_names", ())) != PM_NAMES:
        raise ValueError("All seven PM in canonical order are required")
    if int(contract.get("lag_seconds", 999)) != FIXED_LAG_SECONDS:
        raise ValueError("Matched comparison is frozen at lag -10 s")
    if contract.get("common_cohort") != (
        "10_contiguous_feature_windows_ending_at_t_minus_10s"
    ):
        raise ValueError("Common matched cohort changed")
    if contract.get("target_transform") != "outer_train_q33_q67_extremes":
        raise ValueError("LOW/HIGH target transform changed")
    if contract.get("middle_policy") != "exclude":
        raise ValueError("Middle tertile must remain excluded")
    if contract.get("folds") != [1, 2, 3, 4, 5]:
        raise ValueError("Five fixed folds are required")
    if int(contract.get("seed", -1)) != 42:
        raise ValueError("Seed must remain 42")
    if config.get("models") != EXPECTED_MODELS:
        raise ValueError("Matched model set or historical hyperparameters changed")
    if config.get("validation") != {
        "strategy": "group_record",
        "group_column": "record_group_id",
        "validation_size": 0.15,
        "random_state": 42,
    }:
        raise ValueError("Inner validation contract changed")
    evaluation = config.get("evaluation", {})
    pairs = tuple(tuple(pair) for pair in evaluation.get("paired_temporal_vs_tabular", ()))
    if pairs != TEMPORAL_TABULAR_PAIRS:
        raise ValueError("Paired temporal-vs-tabular comparisons changed")
    bootstrap = evaluation.get("participant_cluster_bootstrap", {})
    if bootstrap != {
        "replicates": 10000,
        "seed": 42,
        "cluster": "subject_id",
        "metrics": ["balanced_accuracy", "f1", "roc_auc"],
    }:
        raise ValueError("Clustered bootstrap contract changed")
    selection = config.get("model_selection_for_personalization", {})
    if (
        selection.get("ranking_metric") != "participant_macro_balanced_accuracy"
        or float(selection.get("practical_equivalence_margin", -1))
        != PRACTICAL_EQUIVALENCE_MARGIN
        or int(selection.get("maximum_models_advanced", -1)) != 2
        or selection.get("fixed_model_order") != list(MODEL_ORDER)
    ):
        raise ValueError("Personalization transition rule changed")
    if config.get("matrix_cells") != 140:
        raise ValueError("Exactly 140 matched model/fold/PM cells are required")
    if config.get("planned_new_fits") != 105:
        raise ValueError("Exactly 105 new fits are required")
    if config.get("planned_reused_reference_fits") != 35:
        raise ValueError("Exactly 35 LSTM reference fits must be reused")
    forbidden = config.get("forbidden", {})
    if not forbidden or not all(value is True for value in forbidden.values()):
        raise ValueError("All forbidden switches must remain true")
    return config


@dataclass
class MatchedContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    reference: ProtocolContext
    common_history_endpoints: set[Any]
    sample_to_position: dict[Any, int]
    run_matrix: pd.DataFrame
    cohort_audit: pd.DataFrame
    lstm_reuse_audit: pd.DataFrame
    neural_run_matrix: pd.DataFrame
    protocol: dict[str, Any]


def _retained_common_rows(
    reference: ProtocolContext,
    *,
    fold: int,
    pm: str,
    common_endpoints: set[Any],
) -> pd.DataFrame:
    cohort = reference.cohorts[pm].copy()
    transform = reference.transforms[(fold, pm)]
    labels = transform.transform(cohort["continuous_target"].to_numpy())
    keep = np.isfinite(labels)
    retained = cohort.loc[keep].copy().reset_index(drop=True)
    retained["label"] = labels[keep].astype(np.int64)
    retained = retained.loc[
        retained["lag_minus_10s_feature_sample_id"].isin(common_endpoints)
    ].copy().reset_index(drop=True)
    if retained["target_sample_id"].duplicated().any():
        raise RuntimeError(f"fold {fold} {pm}: duplicate target sample IDs")
    if retained["lag_minus_10s_feature_sample_id"].duplicated().any():
        raise RuntimeError(f"fold {fold} {pm}: duplicate feature endpoints")
    return retained


def _lstm_reuse_audit(
    *,
    root: Path,
    config: Mapping[str, Any],
    run_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    neural_root = root / config["references"]["neural_robustness"]["output_dir"]
    protocol_path = neural_root / "protocol.json"
    matrix_path = neural_root / "run_matrix.csv"
    if not protocol_path.is_file() or not matrix_path.is_file():
        raise FileNotFoundError("Completed neural robustness artifacts are missing")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_hash") != config["references"]["neural_robustness"]["protocol_hash"]:
        raise RuntimeError("Stored neural robustness protocol hash changed")
    if protocol.get("result_status") != "confirmatory_screening_complete":
        raise RuntimeError("Neural robustness reference is not complete")
    matrix = pd.read_csv(matrix_path)
    lstm = matrix.loc[matrix["model"].eq("torch_lstm")].copy()
    if len(lstm) != 35:
        raise RuntimeError("Neural robustness reference must contain 35 LSTM rows")
    lookup = lstm.set_index(["outer_fold", "pm"])
    audit_rows = []
    for row in run_rows:
        if row["model"] != "torch_lstm":
            continue
        key = (int(row["outer_fold"]), str(row["pm"]))
        if key not in lookup.index:
            raise RuntimeError(f"Missing reference LSTM row {key}")
        prior = lookup.loc[key]
        checks = {
            "n_train": int(prior["n_train"]) == int(row["n_train"]),
            "n_test": int(prior["n_test"]) == int(row["n_test"]),
            "threshold_hash": str(prior["threshold_hash"]) == str(row["threshold_hash"]),
            "train_target_sample_hash": (
                str(prior["train_target_sample_hash"])
                == str(row["train_target_sample_hash"])
            ),
            "test_target_sample_hash": (
                str(prior["test_target_sample_hash"])
                == str(row["test_target_sample_hash"])
            ),
            "train_input_endpoint_hash": (
                str(prior["train_input_endpoint_hash"])
                == str(row["train_input_endpoint_hash"])
            ),
            "test_input_endpoint_hash": (
                str(prior["test_input_endpoint_hash"])
                == str(row["test_input_endpoint_hash"])
            ),
            "context_windows": int(prior["context_windows"]) == 10,
        }
        prior_run_id = str(prior["run_id"])
        prior_dir = neural_root / "runs" / prior_run_id
        artifacts_ok = all(
            (prior_dir / name).is_file()
            for name in (
                "run_summary.json",
                "predictions.parquet",
                "participant_metrics.csv",
            )
        )
        checks["artifacts"] = artifacts_ok
        audit_rows.append({
            "outer_fold": key[0],
            "pm": key[1],
            "reference_run_id": prior_run_id,
            **{f"match_{name}": bool(value) for name, value in checks.items()},
            "all_match": bool(all(checks.values())),
        })
    audit = pd.DataFrame(audit_rows)
    if len(audit) != 35 or not audit["all_match"].all():
        bad = audit.loc[~audit["all_match"]].to_dict("records")
        raise RuntimeError(f"LSTM reference reuse audit failed: {bad[:3]}")
    return audit, matrix


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> MatchedContext:
    root_path = Path(root).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root_path / output

    low = config["references"]["low_high"]
    ref_config = load_reference_config(root_path / low["config"])
    reference = prepare_reference_protocol(
        ref_config,
        root=root_path,
        feature_cache_dir=feature_cache_dir,
        output_dir=root_path / low["output_dir"],
    )
    if reference.protocol["protocol_hash"] != low["protocol_hash"]:
        raise RuntimeError("Recomputed LOW/HIGH reference protocol hash changed")
    stored_low = json.loads(
        (root_path / low["output_dir"] / "protocol.json").read_text(encoding="utf-8")
    )
    if stored_low.get("protocol_hash") != low["protocol_hash"]:
        raise RuntimeError("Stored LOW/HIGH reference protocol hash changed")
    if stored_low.get("result_status") != "confirmatory_complete":
        raise RuntimeError("LOW/HIGH reference is not complete")

    common_endpoints = exact_history_endpoint_ids(
        reference.feature_index,
        length=10,
        step_seconds=10.0,
    )
    if not common_endpoints:
        raise RuntimeError("Common 10-window history cohort is empty")
    sample_to_position = {
        sample_id: position
        for position, sample_id in enumerate(reference.feature_index["sample_id"].tolist())
    }
    if len(sample_to_position) != len(reference.feature_index):
        raise RuntimeError("Feature index sample IDs are not unique")

    threshold_audit = reference.threshold_audit.set_index(["outer_fold", "pm"])
    run_rows = []
    cohort_rows = []
    for fold in config["scientific_contract"]["folds"]:
        for pm in PM_NAMES:
            retained = _retained_common_rows(
                reference,
                fold=int(fold),
                pm=pm,
                common_endpoints=common_endpoints,
            )
            train = retained.loc[retained["outer_fold"].astype(int).ne(int(fold))]
            test = retained.loc[retained["outer_fold"].astype(int).eq(int(fold))]
            if not len(train) or not len(test):
                raise RuntimeError(f"fold {fold} {pm}: empty common train/test cohort")
            if sorted(train["label"].unique().tolist()) != [0, 1]:
                raise RuntimeError(f"fold {fold} {pm}: common train is not class-complete")
            if set(train["subject_id"].astype(str)) & set(test["subject_id"].astype(str)):
                raise RuntimeError(f"fold {fold} {pm}: subject leakage")
            threshold = threshold_audit.loc[(int(fold), pm)]
            cohort_payload = {
                "outer_fold": int(fold),
                "pm": pm,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_train_subjects": int(train["subject_id"].nunique()),
                "n_test_subjects": int(test["subject_id"].nunique()),
                "threshold_hash": str(threshold["threshold_hash"]),
                "q_low": float(threshold["q_low"]),
                "q_high": float(threshold["q_high"]),
                "train_target_sample_hash": _sample_hash(train["target_sample_id"].tolist()),
                "test_target_sample_hash": _sample_hash(test["target_sample_id"].tolist()),
                "train_input_endpoint_hash": _sample_hash(
                    train["lag_minus_10s_feature_sample_id"].tolist()
                ),
                "test_input_endpoint_hash": _sample_hash(
                    test["lag_minus_10s_feature_sample_id"].tolist()
                ),
            }
            cohort_rows.append({
                **cohort_payload,
                "reference_n_train": int(threshold["n_train_retained"]),
                "reference_n_test": int(threshold["n_test_retained"]),
                "excluded_no_10_window_history_train": int(
                    threshold["n_train_retained"] - len(train)
                ),
                "excluded_no_10_window_history_test": int(
                    threshold["n_test_retained"] - len(test)
                ),
                "subject_overlap": 0,
            })
            for model_name in MODEL_ORDER:
                model = config["models"][model_name]
                execution_source = (
                    "reference_neural_robustness"
                    if model_name == "torch_lstm"
                    else "new_training"
                )
                row = {
                    "model": model_name,
                    "input_family": model["input_family"],
                    "context_windows": int(model["context_windows"]),
                    "input_rule": model["input_rule"],
                    "execution_source": execution_source,
                    **cohort_payload,
                }
                run_rows.append(row)

    scientific_payload = {
        "schema_version": SCHEMA_VERSION,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "models": config["models"],
        "validation": config["validation"],
        "evaluation": config["evaluation"],
        "model_selection_for_personalization": config[
            "model_selection_for_personalization"
        ],
        "forbidden": config["forbidden"],
        "feature_cache_identity": reference.cache_identity,
        "fixed_fold_hash": reference.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": reference.protocol["temporal_pairing_hash"],
        "threshold_hashes": reference.protocol["threshold_hashes"],
        "common_history_endpoint_hash": _sample_hash(
            sorted(common_endpoints, key=str)
        ),
    }
    protocol_hash = stable_hash(scientific_payload)

    final_rows = []
    for row in run_rows:
        row = dict(row)
        spec_hash = stable_hash({
            "protocol_hash": protocol_hash,
            "model": row["model"],
            "outer_fold": row["outer_fold"],
            "pm": row["pm"],
            "input_family": row["input_family"],
            "context_windows": row["context_windows"],
            "train_target_sample_hash": row["train_target_sample_hash"],
            "test_target_sample_hash": row["test_target_sample_hash"],
            "train_input_endpoint_hash": row["train_input_endpoint_hash"],
            "test_input_endpoint_hash": row["test_input_endpoint_hash"],
            "threshold_hash": row["threshold_hash"],
        })
        row["specification_hash"] = spec_hash
        row["run_id"] = (
            f"{row['model']}__fold_{row['outer_fold']:02d}"
            f"__{row['pm']}__{spec_hash[:12]}"
        )
        final_rows.append(row)
    run_matrix = pd.DataFrame(final_rows)
    if len(run_matrix) != 140 or run_matrix["run_id"].duplicated().any():
        raise RuntimeError("Exactly 140 unique matched cells are required")

    reuse_audit, neural_matrix = _lstm_reuse_audit(
        root=root_path,
        config=config,
        run_rows=final_rows,
    )

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "preregistered_candidate",
        "training_executed": False,
        "git_commit": _git_head(root_path),
        "low_high_reference_protocol_hash": low["protocol_hash"],
        "neural_robustness_reference_protocol_hash": (
            config["references"]["neural_robustness"]["protocol_hash"]
        ),
        "feature_cache_identity": reference.cache_identity,
        "fixed_fold_hash": reference.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": reference.protocol["temporal_pairing_hash"],
        "threshold_hashes": reference.protocol["threshold_hashes"],
        "common_history_endpoint_count": int(len(common_endpoints)),
        "common_history_endpoint_hash": _sample_hash(
            sorted(common_endpoints, key=str)
        ),
        "matrix_cells": 140,
        "planned_new_fits": 105,
        "planned_reused_reference_fits": 35,
        "lstm_reuse_audit_rows": int(len(reuse_audit)),
        "lstm_reuse_mismatches": int((~reuse_audit["all_match"]).sum()),
        "models": config["models"],
        "validation": config["validation"],
        "evaluation": config["evaluation"],
        "model_selection_for_personalization": config[
            "model_selection_for_personalization"
        ],
        "protocol_hash": protocol_hash,
    }

    return MatchedContext(
        root=root_path,
        output_dir=output,
        config=dict(config),
        reference=reference,
        common_history_endpoints=common_endpoints,
        sample_to_position=sample_to_position,
        run_matrix=run_matrix,
        cohort_audit=pd.DataFrame(cohort_rows),
        lstm_reuse_audit=reuse_audit,
        neural_run_matrix=neural_matrix,
        protocol=protocol,
    )


def _factory_audit(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    shapes = {
        "lightgbm": (371,),
        "xgboost": (371,),
        "torch_lstm": (10, 371),
        "torch_transformer": (8, 371),
    }
    rows = []
    for model_name in MODEL_ORDER:
        model = build_model(
            model_name,
            "classification",
            shapes[model_name],
            2,
            config["models"][model_name]["params"],
        )
        rows.append({
            "model": model_name,
            "estimator_or_adapter": type(model).__name__,
            "input_shape": list(shapes[model_name]),
            "execution_source": (
                "reference_neural_robustness"
                if model_name == "torch_lstm"
                else "new_training"
            ),
            "device": str(getattr(model, "device_", "cpu_or_estimator_native")),
        })
    return rows


def write_dry_run(context: MatchedContext) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(context.output_dir / "run_matrix.csv", context.run_matrix)
    _write_csv(context.output_dir / "cohort_audit.csv", context.cohort_audit)
    _write_csv(
        context.output_dir / "lstm_reference_reuse_audit.csv",
        context.lstm_reuse_audit,
    )
    summary = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "low_high_reference_protocol_hash": (
            context.protocol["low_high_reference_protocol_hash"]
        ),
        "neural_robustness_reference_protocol_hash": (
            context.protocol["neural_robustness_reference_protocol_hash"]
        ),
        "matrix_cells": 140,
        "planned_new_fits": 105,
        "planned_reused_reference_fits": 35,
        "common_history_endpoint_count": (
            context.protocol["common_history_endpoint_count"]
        ),
        "lstm_reuse_audit_rows": 35,
        "lstm_reuse_mismatches": 0,
        "training_executed": False,
        "factory_audit": _factory_audit(context.config),
        "cohort": {
            "fold_pm_rows": int(len(context.cohort_audit)),
            "test_rows_sum": int(context.cohort_audit["n_test"].sum()),
            "train_rows_sum": int(context.cohort_audit["n_train"].sum()),
            "min_test_rows": int(context.cohort_audit["n_test"].min()),
            "max_test_rows": int(context.cohort_audit["n_test"].max()),
        },
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    readme = f"""# PM LOW/HIGH matched temporal-context comparison v1

All models use the exact same 10-window-eligible LOW/HIGH target cohort.

Information sets:
- LightGBM: 371 features at EEG(t-10s)
- XGBoost: 371 features at EEG(t-10s)
- LSTM: EEG feature history t-100s..t-10s
- Transformer: last 8 windows t-80s..t-10s

Execution:
- matrix cells: 140
- new fits: 105
- reused LSTM fits: 35
- LSTM reuse allowed only after exact hash/config/artifact audit
- protocol hash: `{context.protocol['protocol_hash']}`

Primary metric: participant-macro balanced accuracy.
Paired temporal-vs-tabular comparisons use identical participant/PM cohorts.
Clustered bootstrap resamples subject_id with 10,000 replicates, seed 42.

Personalization transition rule: rank all four models by mean participant-macro
balanced accuracy over 35 fold×PM rows; advance at most two, with the second
advancing only when within 0.01 absolute BA of the best.
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def _run_dir(context: MatchedContext, spec: Mapping[str, Any]) -> Path:
    return context.output_dir / "runs" / str(spec["run_id"])


def _common_rows_for_spec(
    context: MatchedContext,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    retained = _retained_common_rows(
        context.reference,
        fold=int(spec["outer_fold"]),
        pm=str(spec["pm"]),
        common_endpoints=context.common_history_endpoints,
    )
    fold = int(spec["outer_fold"])
    train = retained.loc[retained["outer_fold"].astype(int).ne(fold)]
    test = retained.loc[retained["outer_fold"].astype(int).eq(fold)]
    checks = {
        "n_train": len(train) == int(spec["n_train"]),
        "n_test": len(test) == int(spec["n_test"]),
        "train_target": _sample_hash(train["target_sample_id"].tolist())
        == spec["train_target_sample_hash"],
        "test_target": _sample_hash(test["target_sample_id"].tolist())
        == spec["test_target_sample_hash"],
        "train_endpoint": _sample_hash(
            train["lag_minus_10s_feature_sample_id"].tolist()
        ) == spec["train_input_endpoint_hash"],
        "test_endpoint": _sample_hash(
            test["lag_minus_10s_feature_sample_id"].tolist()
        ) == spec["test_input_endpoint_hash"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Runtime common cohort differs from frozen spec: {checks}")
    return retained


def _single_window_data(
    context: MatchedContext,
    retained: pd.DataFrame,
    fold: int,
):
    train_mask = retained["outer_fold"].astype(int).ne(fold).to_numpy()
    test_mask = retained["outer_fold"].astype(int).eq(fold).to_numpy()
    endpoint_ids = retained["lag_minus_10s_feature_sample_id"].tolist()
    positions = np.asarray(
        [context.sample_to_position[sample] for sample in endpoint_ids],
        dtype=np.int64,
    )
    X = np.asarray(context.reference.matrix[positions], dtype=np.float32)
    y = retained["label"].to_numpy(dtype=np.int64)
    return (
        X[train_mask],
        y[train_mask],
        retained.loc[train_mask].reset_index(drop=True),
        X[test_mask],
        y[test_mask],
        retained.loc[test_mask].reset_index(drop=True),
    )


def _transformer_data(
    context: MatchedContext,
    retained: pd.DataFrame,
    fold: int,
):
    sequence_cfg = context.config["models"]["torch_transformer"]["sequence"]
    endpoint_targets = {
        row.lag_minus_10s_feature_sample_id: int(row.label)
        for row in retained.itertuples(index=False)
    }
    dummy = np.zeros(len(context.reference.matrix), dtype=np.int64)
    built = build_sequences(
        context.reference.matrix,
        dummy,
        context.reference.feature_index,
        sequence_length=int(sequence_cfg["length"]),
        stride=int(sequence_cfg["stride"]),
        target_position=str(sequence_cfg["target_position"]),
        expected_step_seconds=float(sequence_cfg["expected_step_seconds"]),
        max_gap_seconds=float(sequence_cfg["max_gap_seconds"]),
        endpoint_targets=endpoint_targets,
    )
    expected = set(retained["lag_minus_10s_feature_sample_id"].tolist())
    actual = set(built.metadata["target_sample_id"].tolist())
    if actual != expected:
        raise RuntimeError(
            "Transformer sequence endpoints differ from frozen common cohort: "
            f"missing={len(expected-actual)}, extra={len(actual-expected)}"
        )
    endpoint_meta = retained.set_index("lag_minus_10s_feature_sample_id")
    meta = built.metadata.copy()
    meta["pm_target_sample_id"] = [
        endpoint_meta.loc[value, "target_sample_id"]
        for value in meta["target_sample_id"]
    ]
    meta["outer_fold"] = [
        int(endpoint_meta.loc[value, "outer_fold"])
        for value in meta["target_sample_id"]
    ]
    meta["pm_target_time"] = [
        float(endpoint_meta.loc[value, "target_time"])
        for value in meta["target_sample_id"]
    ]
    delta = (
        meta["pm_target_time"].to_numpy(dtype=float)
        - meta["sequence_end_time"].to_numpy(dtype=float)
    )
    if not np.allclose(delta, 10.0, atol=1e-6, rtol=0.0):
        raise RuntimeError("Transformer history does not end at EEG(t-10s)")
    train_mask = meta["outer_fold"].astype(int).ne(fold).to_numpy()
    test_mask = meta["outer_fold"].astype(int).eq(fold).to_numpy()
    y = built.y.astype(np.int64)
    return (
        built.X[train_mask],
        y[train_mask],
        meta.loc[train_mask].reset_index(drop=True),
        built.X[test_mask],
        y[test_mask],
        meta.loc[test_mask].reset_index(drop=True),
    )


def execute_new_run(
    context: MatchedContext,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    model_name = str(spec["model"])
    if model_name not in NEW_MODEL_ORDER:
        raise ValueError(f"{model_name} is not a new-training model")
    fold = int(spec["outer_fold"])
    pm = str(spec["pm"])
    retained = _common_rows_for_spec(context, spec)

    if model_name in {"lightgbm", "xgboost"}:
        x_train, y_train, train_meta, x_test, y_test, test_meta = (
            _single_window_data(context, retained, fold)
        )
        input_shape = (371,)
        prediction_target_ids = test_meta["target_sample_id"].to_numpy()
        prediction_endpoint_ids = test_meta[
            "lag_minus_10s_feature_sample_id"
        ].to_numpy()
    else:
        x_train, y_train, train_meta, x_test, y_test, test_meta = (
            _transformer_data(context, retained, fold)
        )
        input_shape = (8, 371)
        prediction_target_ids = test_meta["pm_target_sample_id"].to_numpy()
        prediction_endpoint_ids = test_meta["target_sample_id"].to_numpy()

    if sorted(np.unique(y_train).tolist()) != [0, 1]:
        raise RuntimeError("Outer train is not class-complete")

    model = build_model(
        model_name,
        "classification",
        input_shape,
        2,
        context.config["models"][model_name]["params"],
    )
    if model_name == "torch_transformer":
        model.set_validation_groups(
            train_meta["record_group_id"].astype(str).to_numpy(),
            subject_ids=train_meta["subject_id"].astype(str).to_numpy(),
            record_ids=train_meta["record_id"].astype(str).to_numpy(),
            outer_test_record_ids=test_meta["record_id"].astype(str).to_numpy(),
            outer_test_group_ids=test_meta["record_group_id"].astype(str).to_numpy(),
            strategy="group_record",
            group_column="record_group_id",
            validation_size=0.15,
            random_state=42,
        )

    started = time.perf_counter()
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_test), dtype=np.int64).reshape(-1)
    proba = np.asarray(model.predict_proba(x_test), dtype=float)
    elapsed = time.perf_counter() - started
    classes = np.asarray(getattr(model, "classes_", [0, 1]), dtype=int)
    high = np.flatnonzero(classes == 1)
    if len(high) != 1:
        raise RuntimeError("Classifier lacks unique HIGH probability column")
    probability_high = proba[:, int(high[0])]

    participants, macro = participant_binary_metrics(
        y_test,
        pred,
        probability_high,
        test_meta["subject_id"].astype(str).to_numpy(),
    )

    run_dir = _run_dir(context, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "target_sample_id": prediction_target_ids,
        "feature_endpoint_sample_id": prediction_endpoint_ids,
        "subject_id": test_meta["subject_id"].astype(str).to_numpy(),
        "record_id": test_meta["record_id"].astype(str).to_numpy(),
        "record_group_id": test_meta["record_group_id"].astype(str).to_numpy(),
        "outer_fold": fold,
        "pm": pm,
        "model": model_name,
        "y_true": y_test,
        "y_pred": pred,
        "probability_high": probability_high,
    })
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    participants.insert(0, "model", model_name)
    participants.insert(0, "pm", pm)
    participants.insert(0, "outer_fold", fold)
    _write_csv(run_dir / "participant_metrics.csv", participants)

    validation = dict(getattr(model, "validation_split_", {}) or {})
    if model_name == "torch_transformer":
        if (
            validation.get("inner_group_overlap", 0)
            or validation.get("outer_test_group_overlap", 0)
        ):
            raise RuntimeError("Transformer inner-validation leakage detected")

    summary = {
        "status": "complete",
        "result_status": "matched_confirmatory",
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hash": spec["specification_hash"],
        "run_id": spec["run_id"],
        "execution_source": "new_training",
        "model": model_name,
        "outer_fold": fold,
        "pm": pm,
        "lag_seconds": -10,
        "context_windows": int(spec["context_windows"]),
        "threshold_hash": spec["threshold_hash"],
        "q_low": float(spec["q_low"]),
        "q_high": float(spec["q_high"]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_test_participants": int(len(participants)),
        "training_time_seconds": float(elapsed),
        "validation_strategy": (
            validation.get("strategy")
            if model_name == "torch_transformer"
            else "estimator_native_no_validation"
        ),
        "inner_group_overlap": int(validation.get("inner_group_overlap", 0)),
        "outer_test_group_overlap": int(
            validation.get("outer_test_group_overlap", 0)
        ),
        "n_epochs_trained": int(getattr(model, "n_epochs_trained_", 0)),
        "best_epoch": getattr(model, "best_epoch_", None),
        **macro,
    }
    _atomic_json(run_dir / "run_summary.json", summary)
    return summary


def load_resumable_summary(
    context: MatchedContext,
    spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    if spec["execution_source"] != "new_training":
        return None
    run_dir = _run_dir(context, spec)
    required = (
        run_dir / "run_summary.json",
        run_dir / "predictions.parquet",
        run_dir / "participant_metrics.csv",
    )
    if not all(path.is_file() for path in required):
        return None
    payload = json.loads(required[0].read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    if payload.get("protocol_hash") != context.protocol["protocol_hash"]:
        return None
    if payload.get("specification_hash") != spec["specification_hash"]:
        return None
    return payload


def _prior_lstm_paths(
    context: MatchedContext,
    spec: Mapping[str, Any],
) -> tuple[Path, str]:
    audit = context.lstm_reuse_audit
    match = audit.loc[
        audit["outer_fold"].astype(int).eq(int(spec["outer_fold"]))
        & audit["pm"].astype(str).eq(str(spec["pm"]))
    ]
    if len(match) != 1 or not bool(match.iloc[0]["all_match"]):
        raise RuntimeError("No unique audited reusable LSTM reference run")
    prior_run_id = str(match.iloc[0]["reference_run_id"])
    root = (
        context.root
        / context.config["references"]["neural_robustness"]["output_dir"]
        / "runs"
        / prior_run_id
    )
    return root, prior_run_id


def _reused_lstm_summary(
    context: MatchedContext,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    prior_dir, prior_run_id = _prior_lstm_paths(context, spec)
    prior = json.loads((prior_dir / "run_summary.json").read_text(encoding="utf-8"))
    summary = dict(prior)
    summary.update({
        "result_status": "matched_confirmatory_reused_reference",
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hash": spec["specification_hash"],
        "run_id": spec["run_id"],
        "execution_source": "reference_neural_robustness",
        "reference_run_id": prior_run_id,
        "model": "torch_lstm",
        "outer_fold": int(spec["outer_fold"]),
        "pm": str(spec["pm"]),
        "n_train": int(spec["n_train"]),
        "n_test": int(spec["n_test"]),
        "threshold_hash": str(spec["threshold_hash"]),
    })
    return summary


def _participant_metrics_for_spec(
    context: MatchedContext,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    if spec["execution_source"] == "reference_neural_robustness":
        prior_dir, _ = _prior_lstm_paths(context, spec)
        path = prior_dir / "participant_metrics.csv"
    else:
        path = _run_dir(context, spec) / "participant_metrics.csv"
    frame = pd.read_csv(path)
    metric_columns = {
        "balanced_accuracy": "balanced_accuracy",
        "f1": "macro_f1",
        "roc_auc": "roc_auc",
        "pr_auc": "pr_auc",
        "low_recall": "low_recall",
        "high_recall": "high_recall",
        "precision": "precision",
        "accuracy": "accuracy",
    }
    required = {"subject_id", *metric_columns.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Participant metrics missing columns: {missing}")
    return frame


def _summary_by_model(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_pm = []
    for model_name in MODEL_ORDER:
        for pm in PM_NAMES:
            group = results.loc[
                results["model"].eq(model_name) & results["pm"].eq(pm)
            ]
            if len(group) != 5:
                raise RuntimeError(f"{model_name}/{pm}: expected five folds")
            row = {"model": model_name, "pm": pm, "n_folds": 5}
            for metric in METRICS:
                col = f"participant_macro_{metric}"
                values = group[col].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                row[f"{col}_mean"] = (
                    float(np.mean(finite)) if len(finite) else float("nan")
                )
                row[f"{col}_std"] = (
                    float(np.std(finite, ddof=1))
                    if len(finite) > 1 else float("nan")
                )
            rows_pm.append(row)
    pm_frame = pd.DataFrame(rows_pm)

    rows_model = []
    for model_name in MODEL_ORDER:
        group = results.loc[results["model"].eq(model_name)]
        row = {
            "model": model_name,
            "n_fold_pm_rows": int(len(group)),
            "context_windows": int(group["context_windows"].iloc[0]),
        }
        for metric in METRICS:
            col = f"participant_macro_{metric}"
            values = group[col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{col}_mean"] = (
                float(np.mean(finite)) if len(finite) else float("nan")
            )
            row[f"{col}_std"] = (
                float(np.std(finite, ddof=1))
                if len(finite) > 1 else float("nan")
            )
            row[f"{col}_median"] = (
                float(np.median(finite)) if len(finite) else float("nan")
            )
        rows_model.append(row)
    return pd.DataFrame(rows_model), pm_frame


def _fold_pm_paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    lookup = results.set_index(["model", "outer_fold", "pm"])
    rows = []
    for temporal, tabular in TEMPORAL_TABULAR_PAIRS:
        for fold in (1, 2, 3, 4, 5):
            for pm in PM_NAMES:
                a = lookup.loc[(temporal, fold, pm)]
                b = lookup.loc[(tabular, fold, pm)]
                row = {
                    "temporal_model": temporal,
                    "tabular_model": tabular,
                    "outer_fold": fold,
                    "pm": pm,
                }
                for metric in METRICS:
                    col = f"participant_macro_{metric}"
                    av = float(a[col])
                    bv = float(b[col])
                    row[f"delta_{metric}_temporal_minus_tabular"] = (
                        av - bv
                        if np.isfinite(av) and np.isfinite(bv)
                        else float("nan")
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _participant_paired_deltas(
    context: MatchedContext,
) -> pd.DataFrame:
    frames = {}
    for spec in context.run_matrix.to_dict("records"):
        key = (str(spec["model"]), int(spec["outer_fold"]), str(spec["pm"]))
        frame = _participant_metrics_for_spec(context, spec).copy()
        frame["subject_id"] = frame["subject_id"].astype(str)
        frames[key] = frame

    rows = []
    for temporal, tabular in TEMPORAL_TABULAR_PAIRS:
        for fold in (1, 2, 3, 4, 5):
            for pm in PM_NAMES:
                left = frames[(temporal, fold, pm)]
                right = frames[(tabular, fold, pm)]
                merged = left.merge(
                    right,
                    on="subject_id",
                    how="outer",
                    validate="one_to_one",
                    suffixes=("_temporal", "_tabular"),
                    indicator=True,
                )
                if not merged["_merge"].eq("both").all():
                    raise RuntimeError(
                        f"Participant cohort mismatch {temporal} vs {tabular}, "
                        f"fold={fold}, pm={pm}"
                    )
                for row in merged.itertuples(index=False):
                    out = {
                        "temporal_model": temporal,
                        "tabular_model": tabular,
                        "outer_fold": fold,
                        "pm": pm,
                        "subject_id": str(row.subject_id),
                    }
                    for metric in METRICS:
                        participant_column = "macro_f1" if metric == "f1" else metric
                        a = float(getattr(row, f"{participant_column}_temporal"))
                        b = float(getattr(row, f"{participant_column}_tabular"))
                        out[f"delta_{metric}_temporal_minus_tabular"] = (
                            a - b
                            if np.isfinite(a) and np.isfinite(b)
                            else float("nan")
                        )
                    rows.append(out)
    return pd.DataFrame(rows)


def _cluster_bootstrap(participant_deltas: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for temporal, tabular in TEMPORAL_TABULAR_PAIRS:
        pair = participant_deltas.loc[
            participant_deltas["temporal_model"].eq(temporal)
            & participant_deltas["tabular_model"].eq(tabular)
        ].copy()
        subjects = np.asarray(sorted(pair["subject_id"].astype(str).unique()))
        n_subjects = len(subjects)
        if not n_subjects:
            raise RuntimeError("Cluster bootstrap has no subjects")

        for metric in BOOTSTRAP_METRICS:
            column = f"delta_{metric}_temporal_minus_tabular"
            values = pair[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            observed = (
                float(np.mean(finite)) if len(finite) else float("nan")
            )

            sums = np.zeros(n_subjects, dtype=np.float64)
            counts = np.zeros(n_subjects, dtype=np.int64)
            for index, subject in enumerate(subjects):
                subject_values = pair.loc[
                    pair["subject_id"].astype(str).eq(subject), column
                ].to_numpy(dtype=float)
                subject_finite = subject_values[np.isfinite(subject_values)]
                if len(subject_finite):
                    sums[index] = float(subject_finite.sum())
                    counts[index] = int(len(subject_finite))

            samples = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=np.float64)
            for replicate in range(BOOTSTRAP_REPLICATES):
                drawn = rng.choice(n_subjects, size=n_subjects, replace=True)
                denominator = int(counts[drawn].sum())
                if denominator:
                    samples[replicate] = float(sums[drawn].sum() / denominator)
            finite_samples = samples[np.isfinite(samples)]
            if not len(finite_samples):
                raise RuntimeError("No valid clustered bootstrap replicates")
            rows.append({
                "temporal_model": temporal,
                "tabular_model": tabular,
                "metric": metric,
                "observed_mean_delta": observed,
                "bootstrap_ci_low": float(np.quantile(finite_samples, 0.025)),
                "bootstrap_ci_high": float(np.quantile(finite_samples, 0.975)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "resampling_unit": "subject_id_cluster",
                "n_unique_subjects": n_subjects,
            })
    return pd.DataFrame(rows)


def _select_for_personalization(summary: pd.DataFrame) -> dict[str, Any]:
    fixed_order = {name: index for index, name in enumerate(MODEL_ORDER)}
    records = summary.to_dict("records")
    records.sort(
        key=lambda row: (
            -float(row["participant_macro_balanced_accuracy_mean"]),
            -float(row["participant_macro_roc_auc_mean"]),
            float(row["participant_macro_balanced_accuracy_std"]),
            fixed_order[str(row["model"])],
        )
    )
    best = records[0]
    second = records[1]
    best_ba = float(best["participant_macro_balanced_accuracy_mean"])
    second_ba = float(second["participant_macro_balanced_accuracy_mean"])
    difference = best_ba - second_ba
    advanced = [str(best["model"])]
    if difference <= PRACTICAL_EQUIVALENCE_MARGIN:
        advanced.append(str(second["model"]))
    return {
        "best_model": str(best["model"]),
        "second_model": str(second["model"]),
        "best_balanced_accuracy": best_ba,
        "second_balanced_accuracy": second_ba,
        "best_minus_second_balanced_accuracy": difference,
        "practical_equivalence_margin": PRACTICAL_EQUIVALENCE_MARGIN,
        "advanced_models": advanced,
    }


def aggregate_results(
    context: MatchedContext,
    new_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    combined = list(new_summaries)
    for spec in context.run_matrix.loc[
        context.run_matrix["model"].eq("torch_lstm")
    ].to_dict("records"):
        combined.append(_reused_lstm_summary(context, spec))

    results = pd.DataFrame(combined).sort_values(
        ["model", "outer_fold", "pm"], kind="stable"
    )
    if len(results) != 140:
        raise RuntimeError("Matched aggregation requires exactly 140 cells")
    if results.groupby(["model", "outer_fold", "pm"]).size().ne(1).any():
        raise RuntimeError("Duplicate or missing matched model/fold/PM cells")
    _write_csv(context.output_dir / "results_by_fold.csv", results)

    summary_model, summary_pm = _summary_by_model(results)
    _write_csv(context.output_dir / "summary_by_model.csv", summary_model)
    _write_csv(context.output_dir / "summary_by_model_pm.csv", summary_pm)

    fold_deltas = _fold_pm_paired_deltas(results)
    _write_csv(
        context.output_dir / "paired_delta_temporal_vs_tabular.csv",
        fold_deltas,
    )
    participant_deltas = _participant_paired_deltas(context)
    _write_csv(
        context.output_dir / "participant_paired_delta.csv",
        participant_deltas,
    )
    bootstrap = _cluster_bootstrap(participant_deltas)
    _write_csv(
        context.output_dir / "paired_delta_bootstrap_ci.csv",
        bootstrap,
    )

    selection = _select_for_personalization(summary_model)
    pooled = {
        "n_models": 4,
        "n_pm": 7,
        "n_folds": 5,
        "n_matched_cells": 140,
        "new_fits": 105,
        "reused_lstm_reference_fits": 35,
        **selection,
    }
    _write_csv(context.output_dir / "pooled_summary.csv", pd.DataFrame([pooled]))

    protocol = dict(context.protocol)
    protocol.update({
        "training_executed": True,
        "result_status": "confirmatory_complete",
        "completed_new_fits": 105,
        "completed_reused_reference_fits": 35,
        "model_selection_for_personalization_result": selection,
    })
    _atomic_json(context.output_dir / "protocol.json", protocol)
    return selection


def run_experiment(
    context: MatchedContext,
    *,
    resume: bool,
) -> dict[str, Any]:
    new_specs = context.run_matrix.loc[
        context.run_matrix["execution_source"].eq("new_training")
    ]
    if len(new_specs) != 105:
        raise RuntimeError("Expected exactly 105 new-training cells")
    summaries = []
    trained = 0
    reused_new = 0
    for spec in new_specs.to_dict("records"):
        existing = load_resumable_summary(context, spec) if resume else None
        if existing is not None:
            summaries.append(existing)
            reused_new += 1
            continue
        run_dir = _run_dir(context, spec)
        if run_dir.exists() and not resume:
            raise FileExistsError(
                f"Run directory exists; use --resume after audit: {run_dir}"
            )
        summaries.append(execute_new_run(context, spec))
        trained += 1
    selection = aggregate_results(context, summaries)
    return {
        "combined_complete": 140,
        "new_complete": 105,
        "new_trained": trained,
        "new_reused": reused_new,
        "reference_lstm_reused": 35,
        "advanced_models": selection["advanced_models"],
    }


__all__ = [
    "MODEL_ORDER",
    "MatchedContext",
    "load_config",
    "prepare_protocol",
    "run_experiment",
    "write_dry_run",
]
