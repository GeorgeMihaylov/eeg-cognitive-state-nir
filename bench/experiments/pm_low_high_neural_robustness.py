"""Neural robustness screening for the frozen PM LOW/HIGH extreme-state task.

This experiment reuses the completed LOW/HIGH target contract verbatim and
changes only the input representation / neural architecture:
- ShallowConvNet: raw EEG window at t-10 s;
- LSTM: 10 canonical feature windows ending at t-10 s;
- Transformer: 8 canonical feature windows ending at t-10 s.

The experiment is a screening study, not a direct architecture ranking, because
the three models consume different representations/context lengths.
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

from bench.datasets.raw_eeg_window_dataset import RawEEGWindowArrayView
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


SCHEMA_VERSION = "pm-low-high-neural-robustness-v1"
MODEL_ORDER = (
    "torch_shallow_convnet",
    "torch_lstm",
    "torch_transformer",
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

EXPECTED_MODELS = {
    "torch_shallow_convnet": {
        "input_family": "raw",
        "context_windows": 1,
        "params": {
            "n_filters": 40,
            "temporal_kernel_samples": 25,
            "pool_size": 75,
            "pool_stride": 15,
            "dropout": 0.5,
            "batch_size": 128,
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
    "torch_lstm": {
        "input_family": "sequence_features",
        "context_windows": 10,
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def _sample_hash(values: Sequence[Any]) -> str:
    return stable_hash([str(value) for value in values])


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if config.get("planned_fits") != 105:
        raise ValueError("Exactly 105 neural screening fits are preregistered")
    ref = config.get("reference", {})
    if ref.get("protocol_hash") != (
        "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
    ):
        raise ValueError("Unexpected LOW/HIGH reference protocol hash")
    contract = config.get("scientific_contract", {})
    if tuple(contract.get("pm_names", ())) != PM_NAMES:
        raise ValueError("All seven PM in canonical order are required")
    if int(contract.get("lag_seconds", 999)) != FIXED_LAG_SECONDS:
        raise ValueError("Neural screening is frozen at lag -10 s")
    if contract.get("target_transform") != "outer_train_q33_q67_extremes":
        raise ValueError("LOW/HIGH target transform changed")
    if contract.get("middle_policy") != "exclude":
        raise ValueError("Middle tertile must remain excluded")
    if contract.get("folds") != [1, 2, 3, 4, 5]:
        raise ValueError("Five fixed folds are required")
    if int(contract.get("seed", -1)) != 42:
        raise ValueError("Seed must remain 42")
    if config.get("models") != EXPECTED_MODELS:
        raise ValueError("Neural models or historical hyperparameters changed")
    raw = config.get("raw_input", {})
    if raw.get("shape") != [1, 14, 2560]:
        raise ValueError("ShallowConvNet raw input must be [1,14,2560]")
    if raw.get("expected_preprocessing_hash") != (
        "2251ca950a467267dcccc1c5b83157f26e02768f46c6073d33f5dc16225bda84"
    ):
        raise ValueError("Raw preprocessing hash changed")
    validation = config.get("validation", {})
    if validation != {
        "strategy": "group_record",
        "group_column": "record_group_id",
        "validation_size": 0.15,
        "random_state": 42,
    }:
        raise ValueError("Inner validation contract changed")
    forbidden = config.get("forbidden", {})
    if not forbidden or not all(value is True for value in forbidden.values()):
        raise ValueError("All forbidden switches must remain true")
    return config


def exact_history_endpoint_ids(
    metadata: pd.DataFrame,
    *,
    length: int,
    step_seconds: float = 10.0,
) -> set[Any]:
    """Return endpoints with exactly `length` contiguous same-record windows."""
    required = {
        "sample_id", "source", "subject_id", "record_id",
        "record_group_id", "t_start",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Feature metadata missing columns: {missing}")
    endpoints: set[Any] = set()
    grouped = metadata.groupby(
        ["source", "subject_id", "record_group_id"],
        sort=True,
        dropna=False,
    )
    for _, group in grouped:
        if group["record_id"].astype(str).nunique() != 1:
            raise RuntimeError("Sequence group spans multiple record_id values")
        ordered = group.sort_values(["t_start", "sample_id"], kind="stable")
        times = ordered["t_start"].to_numpy(dtype=float)
        ids = ordered["sample_id"].to_numpy()
        for end in range(length - 1, len(ordered)):
            start = end - length + 1
            deltas = np.diff(times[start : end + 1])
            if len(deltas) == length - 1 and np.allclose(
                deltas, step_seconds, rtol=0.0, atol=1e-6
            ):
                endpoints.add(ids[end])
    return endpoints


def _raw_manifest_audit(
    manifest: pd.DataFrame,
    required_sample_ids: Sequence[Any],
    *,
    expected_hash: str,
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "sample_id" not in manifest:
        raise ValueError("Raw manifest lacks sample_id")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Raw manifest contains duplicate sample_id")
    lookup = manifest.set_index("sample_id", drop=False)
    required = list(dict.fromkeys(required_sample_ids))
    missing = [sample for sample in required if sample not in lookup.index]
    if missing:
        raise RuntimeError(f"Raw manifest misses {len(missing)} lagged feature windows")
    selected = lookup.loc[required].copy()
    bad_status = selected.loc[selected["status"].astype(str).ne("ok")]
    if len(bad_status):
        raise RuntimeError(f"{len(bad_status)} required raw windows are not status=ok")
    if set(selected["preprocessing_hash"].dropna().astype(str)) != {expected_hash}:
        raise RuntimeError("Required raw windows have unexpected preprocessing hash")
    if set(selected["n_channels"].astype(int)) != {14}:
        raise RuntimeError("Required raw windows are not 14-channel")
    if set(selected["n_samples_expected"].astype(int)) != {2560}:
        raise RuntimeError("Required raw windows are not 2560-sample")
    cache_files = sorted(set(selected["cache_file"].astype(str)))
    missing_cache_files = []
    for item in cache_files:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            missing_cache_files.append(str(path))
    if missing_cache_files:
        raise FileNotFoundError(
            f"Missing {len(missing_cache_files)} raw cache shards; "
            f"first={missing_cache_files[:3]}"
        )
    return selected.reset_index(drop=True), {
        "required_unique_sample_ids": len(required),
        "available_unique_sample_ids": len(selected),
        "missing_sample_ids": 0,
        "non_ok_rows": 0,
        "preprocessing_hash": expected_hash,
        "n_channels": 14,
        "n_samples_expected": 2560,
        "cache_files": len(cache_files),
        "missing_cache_files": 0,
        "raw_sample_hash": _sample_hash(sorted(required, key=str)),
    }


@dataclass
class NeuralContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    reference: ProtocolContext
    raw_manifest: pd.DataFrame
    raw_lookup: pd.DataFrame
    history_endpoints: dict[str, set[Any]]
    run_matrix: pd.DataFrame
    cohort_audit: pd.DataFrame
    protocol: dict[str, Any]


def _retained_rows(
    reference: ProtocolContext,
    *,
    fold: int,
    pm: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    cohort = reference.cohorts[pm].copy()
    thresholds = reference.transforms[(fold, pm)]
    labels = thresholds.transform(cohort["continuous_target"].to_numpy())
    keep = np.isfinite(labels)
    retained = cohort.loc[keep].copy().reset_index(drop=True)
    retained["label"] = labels[keep].astype(np.int64)
    if retained["lag_minus_10s_feature_sample_id"].duplicated().any():
        raise RuntimeError(f"fold {fold} {pm}: duplicate lag feature endpoints")
    return retained, retained["label"].to_numpy(dtype=np.int64)


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> NeuralContext:
    root_path = Path(root).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root_path / output

    ref_config = load_reference_config(root_path / config["reference"]["config"])
    reference = prepare_reference_protocol(
        ref_config,
        root=root_path,
        feature_cache_dir=feature_cache_dir,
        output_dir=root_path / config["reference"]["output_dir"],
    )
    if reference.protocol["protocol_hash"] != config["reference"]["protocol_hash"]:
        raise RuntimeError("Recomputed LOW/HIGH reference protocol hash changed")
    completed_protocol = json.loads(
        (root_path / config["reference"]["output_dir"] / "protocol.json")
        .read_text(encoding="utf-8")
    )
    if completed_protocol.get("protocol_hash") != config["reference"]["protocol_hash"]:
        raise RuntimeError("Stored LOW/HIGH reference protocol hash changed")
    if completed_protocol.get("result_status") != "confirmatory_complete":
        raise RuntimeError("LOW/HIGH reference is not confirmatory_complete")

    feature_index = reference.feature_index.copy()
    history_endpoints = {
        "torch_lstm": exact_history_endpoint_ids(feature_index, length=10),
        "torch_transformer": exact_history_endpoint_ids(feature_index, length=8),
    }

    all_lag_ids = reference.temporal_pairing[
        "lag_minus_10s_feature_sample_id"
    ].tolist()
    raw_manifest = pd.read_parquet(root_path / config["data"]["raw_manifest"])
    raw_lookup, raw_summary = _raw_manifest_audit(
        raw_manifest,
        all_lag_ids,
        expected_hash=config["raw_input"]["expected_preprocessing_hash"],
        root=root_path,
    )
    raw_lookup = raw_lookup.set_index("sample_id", drop=False)

    rows = []
    audit_rows = []
    audit_by_key = reference.threshold_audit.set_index(["outer_fold", "pm"])
    for model_name in MODEL_ORDER:
        family = config["models"][model_name]["input_family"]
        for fold in config["scientific_contract"]["folds"]:
            for pm in PM_NAMES:
                retained, _ = _retained_rows(reference, fold=fold, pm=pm)
                if family == "sequence_features":
                    allowed = history_endpoints[model_name]
                    model_rows = retained.loc[
                        retained["lag_minus_10s_feature_sample_id"].isin(allowed)
                    ].copy()
                else:
                    model_rows = retained.copy()
                train = model_rows.loc[model_rows["outer_fold"].astype(int).ne(fold)]
                test = model_rows.loc[model_rows["outer_fold"].astype(int).eq(fold)]
                if not len(train) or not len(test):
                    raise RuntimeError(f"{model_name} fold {fold} {pm}: empty train/test")
                if sorted(train["label"].unique().tolist()) != [0, 1]:
                    raise RuntimeError(
                        f"{model_name} fold {fold} {pm}: train not class-complete"
                    )
                train_subjects = set(train["subject_id"].astype(str))
                test_subjects = set(test["subject_id"].astype(str))
                if train_subjects & test_subjects:
                    raise RuntimeError(
                        f"{model_name} fold {fold} {pm}: subject leakage"
                    )
                if family == "sequence_features":
                    earliest_offset = -10 * int(config["models"][model_name]["context_windows"])
                    context_desc = f"PM(t) <- EEG features t{earliest_offset}..t-10"
                else:
                    earliest_offset = -10
                    context_desc = "PM(t) <- raw EEG(t-10)"
                ref_audit = audit_by_key.loc[(fold, pm)]
                spec = {
                    "model": model_name,
                    "input_family": family,
                    "outer_fold": int(fold),
                    "pm": pm,
                    "target_id": f"target_{pm}",
                    "lag_seconds": -10,
                    "context_windows": int(config["models"][model_name]["context_windows"]),
                    "earliest_context_offset_seconds": int(earliest_offset),
                    "context_description": context_desc,
                    "threshold_hash": str(ref_audit["threshold_hash"]),
                    "q_low": float(ref_audit["q_low"]),
                    "q_high": float(ref_audit["q_high"]),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_train_subjects": int(train["subject_id"].nunique()),
                    "n_test_subjects": int(test["subject_id"].nunique()),
                    "train_target_sample_hash": _sample_hash(
                        train["target_sample_id"].tolist()
                    ),
                    "test_target_sample_hash": _sample_hash(
                        test["target_sample_id"].tolist()
                    ),
                    "train_input_endpoint_hash": _sample_hash(
                        train["lag_minus_10s_feature_sample_id"].tolist()
                    ),
                    "test_input_endpoint_hash": _sample_hash(
                        test["lag_minus_10s_feature_sample_id"].tolist()
                    ),
                }
                rows.append(spec)
                audit_rows.append({
                    **spec,
                    "reference_n_train": int(ref_audit["n_train_retained"]),
                    "reference_n_test": int(ref_audit["n_test_retained"]),
                    "excluded_for_history_train": int(
                        ref_audit["n_train_retained"] - len(train)
                    ),
                    "excluded_for_history_test": int(
                        ref_audit["n_test_retained"] - len(test)
                    ),
                    "raw_availability_mismatch": 0,
                    "subject_overlap": 0,
                })

    scientific_payload = {
        "schema_version": SCHEMA_VERSION,
        "reference_protocol_hash": config["reference"]["protocol_hash"],
        "scientific_contract": config["scientific_contract"],
        "raw_input": config["raw_input"],
        "models": config["models"],
        "validation": config["validation"],
        "evaluation": config["evaluation"],
        "forbidden": config["forbidden"],
        "feature_cache_identity": reference.cache_identity,
        "fixed_fold_hash": reference.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": reference.protocol["temporal_pairing_hash"],
        "threshold_hashes": reference.protocol["threshold_hashes"],
        "raw_audit": raw_summary,
        "history_endpoint_hashes": {
            name: _sample_hash(sorted(ids, key=str))
            for name, ids in history_endpoints.items()
        },
    }
    protocol_hash = stable_hash(scientific_payload)
    final_rows = []
    for spec in rows:
        spec = dict(spec)
        spec_hash = stable_hash({
            "protocol_hash": protocol_hash,
            "run_spec": spec,
        })
        spec["specification_hash"] = spec_hash
        spec["run_id"] = (
            f"{spec['model']}__fold_{spec['outer_fold']:02d}"
            f"__{spec['pm']}__{spec_hash[:12]}"
        )
        final_rows.append(spec)
    run_matrix = pd.DataFrame(final_rows)
    if len(run_matrix) != 105 or run_matrix["run_id"].duplicated().any():
        raise RuntimeError("Exactly 105 unique neural runs are required")

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "preregistered_candidate",
        "training_executed": False,
        "git_commit": _git_head(root_path),
        "reference_protocol_hash": config["reference"]["protocol_hash"],
        "feature_cache_identity": reference.cache_identity,
        "fixed_fold_hash": reference.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": reference.protocol["temporal_pairing_hash"],
        "threshold_hashes": reference.protocol["threshold_hashes"],
        "raw_audit": raw_summary,
        "history_endpoint_counts": {
            name: len(ids) for name, ids in history_endpoints.items()
        },
        "models": config["models"],
        "validation": config["validation"],
        "evaluation": config["evaluation"],
        "planned_fits": 105,
        "protocol_hash": protocol_hash,
    }
    return NeuralContext(
        root=root_path,
        output_dir=output,
        config=dict(config),
        reference=reference,
        raw_manifest=raw_manifest,
        raw_lookup=raw_lookup,
        history_endpoints=history_endpoints,
        run_matrix=run_matrix,
        cohort_audit=pd.DataFrame(audit_rows),
        protocol=protocol,
    )


def _factory_audit(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    shapes = {
        "torch_shallow_convnet": (1, 14, 2560),
        "torch_lstm": (10, 371),
        "torch_transformer": (8, 371),
    }
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
            "adapter": type(model).__name__,
            "input_shape": list(shapes[model_name]),
            "device": str(getattr(model, "device_", "unknown")),
        })
    return rows


def write_dry_run(context: NeuralContext) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(context.output_dir / "run_matrix.csv", context.run_matrix)
    _write_csv(context.output_dir / "cohort_audit.csv", context.cohort_audit)

    per_model = {}
    for model_name in MODEL_ORDER:
        group = context.run_matrix.loc[context.run_matrix["model"].eq(model_name)]
        per_model[model_name] = {
            "fits": int(len(group)),
            "train_rows_sum": int(group["n_train"].sum()),
            "test_rows_sum": int(group["n_test"].sum()),
            "min_test_rows": int(group["n_test"].min()),
            "max_test_rows": int(group["n_test"].max()),
            "context_windows": int(group["context_windows"].iloc[0]),
            "earliest_context_offset_seconds": int(
                group["earliest_context_offset_seconds"].iloc[0]
            ),
        }
    summary = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "reference_protocol_hash": context.config["reference"]["protocol_hash"],
        "planned_fits": 105,
        "training_executed": False,
        "feature_count": 371,
        "fixed_lag_seconds": -10,
        "raw_availability_mismatches": int(
            context.cohort_audit["raw_availability_mismatch"].sum()
        ),
        "subject_overlap_rows": int(context.cohort_audit["subject_overlap"].sum()),
        "factory_audit": _factory_audit(context.config),
        "per_model": per_model,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    readme = f"""# PM LOW/HIGH neural robustness v1

Screening experiment for the frozen extreme-state target contract.

- reference LOW/HIGH protocol: `{context.config['reference']['protocol_hash']}`
- new protocol hash: `{context.protocol['protocol_hash']}`
- models: ShallowConvNet, LSTM, Transformer
- PM / folds / fits: 7 / 5 / 105
- ShallowConvNet: raw EEG(t-10s), historical fixed configuration
- LSTM: 10 feature windows ending at t-10s, historical fixed configuration
- Transformer: 8 feature windows ending at t-10s, historical fixed configuration
- thresholds: exact outer-train Q33/Q67 from the completed LOW/HIGH contract
- inner validation: record-group disjoint
- training executed by dry-run: false

This is screening, not direct cross-architecture ranking. Sequence models are
evaluated on history-eligible cohorts; a matched-cohort tabular follow-up is
required before claiming an architecture advantage.
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def _run_dir(context: NeuralContext, spec: Mapping[str, Any]) -> Path:
    return context.output_dir / "runs" / str(spec["run_id"])


def _prepare_retained_for_spec(
    context: NeuralContext,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    fold = int(spec["outer_fold"])
    pm = str(spec["pm"])
    retained, _ = _retained_rows(context.reference, fold=fold, pm=pm)
    if spec["input_family"] == "sequence_features":
        retained = retained.loc[
            retained["lag_minus_10s_feature_sample_id"].isin(
                context.history_endpoints[str(spec["model"])]
            )
        ].copy()
    if int(len(retained.loc[retained["outer_fold"].astype(int).ne(fold)])) != int(spec["n_train"]):
        raise RuntimeError("Runtime neural train cohort differs from frozen spec")
    if int(len(retained.loc[retained["outer_fold"].astype(int).eq(fold)])) != int(spec["n_test"]):
        raise RuntimeError("Runtime neural test cohort differs from frozen spec")
    return retained.reset_index(drop=True)


def _raw_view(
    context: NeuralContext,
    feature_sample_ids: Sequence[Any],
) -> RawEEGWindowArrayView:
    rows = context.raw_lookup.loc[list(feature_sample_ids)].copy()
    return RawEEGWindowArrayView(
        rows.reset_index(drop=True),
        cache_path_root=context.root,
    )


def _sequence_data(
    context: NeuralContext,
    spec: Mapping[str, Any],
    retained: pd.DataFrame,
):
    model_name = str(spec["model"])
    sequence_cfg = context.config["models"][model_name]["sequence"]
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
    expected_ids = set(retained["lag_minus_10s_feature_sample_id"].tolist())
    actual_ids = set(built.metadata["target_sample_id"].tolist())
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Runtime sequence endpoints differ from frozen exact-history set: "
            f"missing={len(expected_ids-actual_ids)}, extra={len(actual_ids-expected_ids)}"
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
    deltas = (
        meta["pm_target_time"].to_numpy(dtype=float)
        - meta["sequence_end_time"].to_numpy(dtype=float)
    )
    if not np.allclose(deltas, 10.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("Sequence endpoint is not exactly EEG(t-10) for PM(t)")
    return built.X, built.y.astype(np.int64), meta


def execute_run(
    context: NeuralContext,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    fold = int(spec["outer_fold"])
    pm = str(spec["pm"])
    model_name = str(spec["model"])
    retained = _prepare_retained_for_spec(context, spec)
    train_mask_rows = retained["outer_fold"].astype(int).ne(fold).to_numpy()
    test_mask_rows = retained["outer_fold"].astype(int).eq(fold).to_numpy()

    if spec["input_family"] == "raw":
        y_all = retained["label"].to_numpy(dtype=np.int64)
        x_train = _raw_view(
            context,
            retained.loc[
                train_mask_rows, "lag_minus_10s_feature_sample_id"
            ].tolist(),
        )
        x_test = _raw_view(
            context,
            retained.loc[
                test_mask_rows, "lag_minus_10s_feature_sample_id"
            ].tolist(),
        )
        y_train = y_all[train_mask_rows]
        y_test = y_all[test_mask_rows]
        train_meta = retained.loc[train_mask_rows].reset_index(drop=True)
        test_meta = retained.loc[test_mask_rows].reset_index(drop=True)
        input_shape = (1, 14, 2560)
        prediction_target_ids = test_meta["target_sample_id"].to_numpy()
        prediction_feature_ids = test_meta[
            "lag_minus_10s_feature_sample_id"
        ].to_numpy()
    else:
        X, y, meta = _sequence_data(context, spec, retained)
        train_mask = meta["outer_fold"].astype(int).ne(fold).to_numpy()
        test_mask = meta["outer_fold"].astype(int).eq(fold).to_numpy()
        x_train = X[train_mask]
        x_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        train_meta = meta.loc[train_mask].reset_index(drop=True)
        test_meta = meta.loc[test_mask].reset_index(drop=True)
        input_shape = (
            int(context.config["models"][model_name]["sequence"]["length"]),
            371,
        )
        prediction_target_ids = test_meta["pm_target_sample_id"].to_numpy()
        prediction_feature_ids = test_meta["target_sample_id"].to_numpy()

    if sorted(np.unique(y_train).tolist()) != [0, 1]:
        raise RuntimeError("Outer train is not class-complete")
    model = build_model(
        model_name,
        "classification",
        input_shape,
        2,
        context.config["models"][model_name]["params"],
    )
    if not hasattr(model, "set_validation_groups"):
        raise RuntimeError("Torch adapter lacks group-aware validation")
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
        raise RuntimeError("Classifier lacks a unique HIGH probability column")
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
        "feature_endpoint_sample_id": prediction_feature_ids,
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

    validation_split = dict(getattr(model, "validation_split_", {}) or {})
    if (
        validation_split.get("inner_group_overlap", 0)
        or validation_split.get("outer_test_group_overlap", 0)
    ):
        raise RuntimeError("Inner validation leakage detected")
    summary = {
        "status": "complete",
        "result_status": "confirmatory_screening",
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hash": spec["specification_hash"],
        "run_id": spec["run_id"],
        "model": model_name,
        "input_family": spec["input_family"],
        "outer_fold": fold,
        "pm": pm,
        "target_id": f"target_{pm}",
        "lag_seconds": -10,
        "context_windows": int(spec["context_windows"]),
        "earliest_context_offset_seconds": int(
            spec["earliest_context_offset_seconds"]
        ),
        "threshold_hash": spec["threshold_hash"],
        "q_low": float(spec["q_low"]),
        "q_high": float(spec["q_high"]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_test_participants": int(len(participants)),
        "training_time_seconds": float(elapsed),
        "validation_strategy": validation_split.get("strategy"),
        "inner_group_overlap": int(
            validation_split.get("inner_group_overlap", 0)
        ),
        "outer_test_group_overlap": int(
            validation_split.get("outer_test_group_overlap", 0)
        ),
        "n_epochs_trained": int(getattr(model, "n_epochs_trained_", 0)),
        "best_epoch": getattr(model, "best_epoch_", None),
        **macro,
    }
    _atomic_json(run_dir / "run_summary.json", summary)
    return summary


def load_resumable_summary(
    context: NeuralContext,
    spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    run_dir = _run_dir(context, spec)
    paths = (
        run_dir / "run_summary.json",
        run_dir / "predictions.parquet",
        run_dir / "participant_metrics.csv",
    )
    if not all(path.is_file() for path in paths):
        return None
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    if payload.get("protocol_hash") != context.protocol["protocol_hash"]:
        return None
    if payload.get("specification_hash") != spec["specification_hash"]:
        return None
    return payload


def aggregate_results(
    context: NeuralContext,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    results = pd.DataFrame(summaries).sort_values(
        ["model", "outer_fold", "pm"], kind="stable"
    )
    if len(results) != 105 or results["run_id"].duplicated().any():
        raise RuntimeError("Aggregation requires 105 unique completed runs")
    _write_csv(context.output_dir / "results_by_fold.csv", results)

    by_model_pm = []
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
            by_model_pm.append(row)
    summary_pm = pd.DataFrame(by_model_pm)
    _write_csv(context.output_dir / "summary_by_model_pm.csv", summary_pm)

    by_model = []
    for model_name in MODEL_ORDER:
        group = results.loc[results["model"].eq(model_name)]
        row = {
            "model": model_name,
            "n_fold_pm_rows": int(len(group)),
            "context_windows": int(group["context_windows"].iloc[0]),
            "earliest_context_offset_seconds": int(
                group["earliest_context_offset_seconds"].iloc[0]
            ),
            "comparison_note": (
                "screening_only_different_representation_or_context"
            ),
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
        by_model.append(row)
    _write_csv(context.output_dir / "summary_by_model.csv", pd.DataFrame(by_model))

    protocol = dict(context.protocol)
    protocol["training_executed"] = True
    protocol["result_status"] = "confirmatory_screening_complete"
    protocol["completed_fits"] = 105
    _atomic_json(context.output_dir / "protocol.json", protocol)


def run_experiment(
    context: NeuralContext,
    *,
    resume: bool,
) -> dict[str, int]:
    summaries = []
    trained = 0
    reused = 0
    for spec in context.run_matrix.to_dict("records"):
        existing = load_resumable_summary(context, spec) if resume else None
        if existing is not None:
            summaries.append(existing)
            reused += 1
            continue
        run_dir = _run_dir(context, spec)
        if run_dir.exists() and not resume:
            raise FileExistsError(
                f"Run directory exists; use --resume after audit: {run_dir}"
            )
        summaries.append(execute_run(context, spec))
        trained += 1
    aggregate_results(context, summaries)
    return {"complete": len(summaries), "trained": trained, "reused": reused}


__all__ = [
    "MODEL_ORDER",
    "NeuralContext",
    "exact_history_endpoint_ids",
    "load_config",
    "prepare_protocol",
    "run_experiment",
    "write_dry_run",
]
